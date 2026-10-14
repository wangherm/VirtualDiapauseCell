"""Software continuation tests, no paid requests or scientific results."""
from unittest.mock import patch
import pytest
from vdc.io import read_json, write_json
from vdc.research_checkpoints import checkpoint, due, bind_controller
from vdc.research_codex import controller, advise
from vdc.research_bridge import rpc, process_one, snapshot
from tests.test_research_r2 import make_engine


def test_checkpoint_counts_new_unique_actions_and_keeps_negative_results(tmp_path):
    bad = {'type':'experiment', 'status':'completed', 'result':{
        'metrics':{'macro_unit_mse':2, 'units':1},
        'comparison':{'relative_improvement':-0.5, 'independent_confirmation':False}}}
    write_json(tmp_path/'turns/000001/response.json', {'results':{'negative':bad}})
    s = {'next_turn':2, 'status':'running', 'worker_progress':{'experiments':1}}
    first = checkpoint(tmp_path, s, 'fixture')
    assert first['new_completed_experiments'] == 1
    write_json(tmp_path/'turns/000002/response.json', {'results':{
        'negative':bad, 'failed':{'type':'experiment','status':'failed','error_type':'KeyError'}}})
    s['next_turn']=3
    second = checkpoint(tmp_path, s, 'fixture')
    assert second['new_completed_experiments'] == 0
    assert [r['action'] for r in second['new_actions']] == ['failed']
    assert not second['scientific_approval']
    assert first['new_actions'][0]['comparison']['relative_improvement'] < 0


def test_due_by_experiments_or_turns_not_by_file_reads(tmp_path):
    s={'next_turn':25,'worker_progress':{'experiments':13}}
    checkpoint(tmp_path,s,'continuation_baseline')
    assert not due(tmp_path,{**s,'next_turn':26,'worker_progress':{'experiments':13,'completed_preparations':100}})
    assert due(tmp_path,{**s,'worker_progress':{'experiments':18}})
    assert due(tmp_path,{**s,'next_turn':49})


def test_migration_rejects_pending_changed_run_and_misalignment(tmp_path):
    old={'transport':{'run':'same'},'model':None,'protocol':'p','controller_source_sha256':'old'}
    new={**old,'controller_source_sha256':'new'}
    write_json(tmp_path/'controller_binding.json', old)
    state={'next_turn':25}
    snap={'next_decision':25,'snapshot_sha256':'s'}
    with pytest.raises(ValueError,match='explicitly'):
        bind_controller(tmp_path,new,state,snap)
    with pytest.raises(ValueError,match='Pending'):
        bind_controller(tmp_path,new,{**state,'pending':{}},snap,True)
    with pytest.raises(ValueError,match='another run'):
        bind_controller(tmp_path,{**new,'transport':{'run':'other'}},state,snap,True)
    with pytest.raises(ValueError,match='do not match'):
        bind_controller(tmp_path,new,state,{**snap,'next_decision':26},True)
    write_json(tmp_path/'turns/000025/codex_receipt.json', {'status':'submitted_outcome_uncertain'})
    with pytest.raises(ValueError,match='Pending'):
        bind_controller(tmp_path,new,state,snap,True)
    (tmp_path/'turns/000025/codex_receipt.json').unlink()
    write_json(tmp_path/'codex_session.json', {'thread_id':'same-session'})
    bind_controller(tmp_path,new,state,snap,True)
    assert read_json(tmp_path/'controller_binding.json') == new
    assert len(list((tmp_path/'controller_versions').glob('*.json'))) == 1
    assert read_json(tmp_path/'codex_session.json')['thread_id'] == 'same-session'


def test_advisor_budget_is_cumulative_across_turn_directories(tmp_path, monkeypatch):
    monkeypatch.setenv('VDC_RESEARCH_API_KEY','fixture-not-real')
    write_json(tmp_path/'advisor/000001/api/advice_001.json', {'budget_charge':100,'usage':{'total_tokens':10}})
    with patch('vdc.research_codex.Client') as client:
        r=advise(tmp_path,2,'question',{}, {'max_api_calls':1,'max_token_budget':None})
        assert r['status']=='not_run_advisor_budget_exhausted'
        client.assert_not_called()
        r=advise(tmp_path,3,'question',{}, {'max_api_calls':10,'max_token_budget':100})
        assert r['status']=='not_run_advisor_budget_exhausted'
        client.assert_not_called()


def test_continuous_crosses_batch_and_stops_without_repeating_decisions(tmp_path):
    e=make_engine(tmp_path)
    e.config.update(no_progress_warning_decisions=20,no_progress_stop_decisions=24)
    snapshot(e)
    workspace=tmp_path/'local'
    config={'transport':'files','run':str(e.run)}
    def remote(c,packet):
        result=rpc(e.run,packet)
        if packet['operation']=='submit':
            process_one(e)
        if packet['operation']=='poll' and packet['index']==3:
            (workspace/'STOP').write_text('test operator pause')
        return result
    def primary(w,index,model):
        return {'actions':[{'type':'inventory','id':'read_'+str(index)}]}
    with patch('vdc.research_codex.transport',side_effect=remote), patch('vdc.research_codex.run_codex',side_effect=primary) as runner:
        result=controller(config,workspace,turns=1,continuous=True,checkpoint_turns=2)
    assert runner.call_count==3
    assert result['next_turn']==4
    assert result['status']=='paused_by_operator'
    assert e.state['next_decision']==4
    assert len(list((workspace/'checkpoints').glob('*.json'))) == 3


def test_worker_finish_requires_human_review(tmp_path):
    e=make_engine(tmp_path); e.state['status']='completed_declared_scope'; snapshot(e)
    with patch('vdc.research_codex.run_codex') as primary:
        r=controller({'transport':'files','run':str(e.run)},tmp_path/'local',continuous=True)
        assert r['status']=='awaiting_stage_review'
        primary.assert_not_called()


def test_missing_advisor_credentials_do_not_block_primary(tmp_path, monkeypatch):
    monkeypatch.delenv('VDC_RESEARCH_API_KEY', raising=False)
    e=make_engine(tmp_path); snapshot(e)
    workspace=tmp_path/'local'
    def remote(c, packet):
        r=rpc(e.run,packet)
        if packet['operation']=='submit': process_one(e)
        if packet['operation']=='poll': (workspace/'STOP').write_text('fixture pause')
        return r
    proposal={'actions':[{'id':'one','type':'inventory'}], 'advisor_question':'review'}
    with patch('vdc.research_codex.transport',side_effect=remote), patch('vdc.research_codex.run_codex',return_value=proposal) as primary:
        r=controller({'transport':'files','run':str(e.run)},workspace,continuous=True,
                     advisor_config={'max_api_calls':1000,'max_token_budget':None})
        assert r['status']=='paused_by_operator'
        assert r['next_turn']==2
        assert r['advisor_health']['status']=='not_run_missing_deepseek_key'
        primary.assert_called_once()

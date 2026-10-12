import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from vdc.io import write_json,read_json,save_npz
from vdc.research_r2_tools import inside,write_code,inventory,asset_context,resolve_assets
from vdc.research_r2_runtime import seal_task,verify_task,compare,audit_groups,numeric_npz
from vdc.research_r2 import Engine,decision


def fixture(tmp_path):
    p=tmp_path/'prepared'
    x=np.arange(24,dtype=float).reshape(8,3);y=x[:,:2]*.25
    save_npz(p/'train.npz',x=x[:5],y=y[:5]);save_npz(p/'query.npz',x=x[5:],hidden_truth=y[5:])
    save_npz(p/'truth.npz',y=y[5:])
    for name,idx in [('train',range(5)),('query',range(5,8))]:
        write_json(p/(name+'_rows.json'),[{'study_family':'test','biological_unit':str(i),
                   'link_ids':[],'origin':'unit_test_fixture','historical_role':'reserved'} for i in idx])
    write_json(p/'schema.json',{'features':['a','b','c'],'targets':['x','y']})
    c={k:'fixture' for k in ('question','target_definition','target_scale','allowed_inputs',
                            'unit_definition','preprocessing','split_rationale')}
    c.update(mission='state',source_ids=['s'])
    return p,c


def test_task_seals_new_target_and_strips_hidden_query_keys(tmp_path):
    p,c=fixture(tmp_path);task=tmp_path/'task'
    r=seal_task(p,task,c,['s'])
    with np.load(task/'query.npz') as z:assert z.files==['x']
    assert r['fit_units']==5 and r['query_units']==3
    assert set(read_json(task/'baselines.json'))=={'train_mean','same_input_ridge'}
    verify_task(task)
    save_npz(task/'truth.npz',y=np.zeros((3,2)))
    with pytest.raises(ValueError,match='changed'):verify_task(task)


def test_r1_parent_mistake_cannot_be_marked_supported():
    current={'macro_unit_mse':.02302267022866416}
    references={'parent':{'macro_unit_mse':.018877265395898672},
                'same_input_ridge':{'macro_unit_mse':.04543079002886234}}
    result=compare(current,references,{'reference':'parent'})
    assert result['status']=='not_supported_on_task'
    assert result['relative_improvement']<-.21
    assert compare(current,{}, {'reference':'parent'})['status']=='not_evaluable'


def test_shared_control_transitive_link_cannot_cross_split():
    a={'study_family':'s','biological_unit':'a','link_ids':['control','culture']}
    b={'study_family':'s','biological_unit':'b','link_ids':['culture']}
    with pytest.raises(ValueError,match='Linked'):audit_groups([a],[b])
    b['link_ids']=[]
    assert len(audit_groups([a],[b]))==2


@pytest.mark.parametrize('path',['../escape.py','/root/key','a/../../x','C:/secret','a\\b'])
def test_candidate_paths_are_relative(tmp_path,path):
    with pytest.raises(ValueError):write_code(tmp_path,{path:'bad'})


def test_inventory_opens_old_roles_and_real_input_folder(tmp_path):
    repo=tmp_path/'repo'; private=tmp_path/'private'
    write_json(repo/'knowledge/research_public_data.json',{'studies':[{'accession':'GSE1','role':'test'}]})
    write_json(repo/'configs/research_public_assets.json',{'files':[]})
    write_json(private/'sample_roles.json',{'samples':[{'role':'reserved','sample_key':'a'}]})
    save_npz(private/'input/actual.npz',x=np.ones((2,2)))
    (private/'input/api_key.txt').write_text('secret')
    before=(private/'sample_roles.json').read_bytes()
    inv=inventory(repo,private)
    assert inv['sources']['GSE1']['role']=='available_for_adaptive_research'
    assert inv['sources']['GSE1']['historical_role']=='test'
    assert len(inv['assets'])==2
    assert not any('secret' in str(v) for v in asset_context(inv)['assets'].values())
    assert (private/'sample_roles.json').read_bytes()==before


def test_source_changes_invalidate_reuse(tmp_path):
    p=tmp_path/'a.csv';p.write_text('a,b\n1,2\n')
    inv={'assets':{'a':{'path':str(p),'source':'s','sha256':hashlib.sha256(p.read_bytes()).hexdigest()}}}
    resolve_assets(inv,['a']);p.write_text('a,b\n2,3\n')
    with pytest.raises(ValueError,match='changed'):resolve_assets(inv,['a'])


def test_nonfinite_and_pickle_targets_rejected(tmp_path):
    p=tmp_path/'x.npz';save_npz(p,x=np.array([[np.nan]]))
    with pytest.raises(ValueError):numeric_npz(p,('x',))
    save_npz(p,x=np.array([['abc']],dtype=object))
    with pytest.raises(ValueError):numeric_npz(p,('x',))


def test_duplicate_decision_ids_rejected():
    with pytest.raises(ValueError):decision({'actions':[{'id':'x','type':'inventory'}]*2})


def make_engine(tmp_path):
    repo=tmp_path/'repo';private=tmp_path/'private'
    write_json(repo/'knowledge/research_public_data.json',{'studies':[]})
    write_json(repo/'configs/research_public_assets.json',{'files':[]})
    config={'mission':'universal dormancy','runtime':{'image':'fake'},'max_api_calls':30,
            'max_output_tokens':16000,'max_request_budget':60000,'max_token_budget':1000000}
    with patch('vdc.research_r2.resolve_runtime',return_value='test-runtime'):
        return Engine(repo,private,tmp_path/'run',config)


def test_durable_results_survive_context_compaction_and_reuse(tmp_path):
    from vdc.io import object_hash
    engine=make_engine(tmp_path)
    action={'id':'old','type':'inspect','asset_ids':['example']}
    engine.state['actions']['old']={'type':'inspect','status':'completed',
        'signature':object_hash(action),'result':{'columns':['gene','counts']}}
    for i in range(300):
        engine.state['actions'][f'later{i}']={'type':'inventory','status':'completed','result':{'text':'x'*2000}}
    with patch('vdc.research_r2.Client') as client:
        client.return_value.call.side_effect=[{'actions':[action]},RuntimeError('stop fixture')]
        with pytest.raises(RuntimeError):engine.run_loop()
    assert 'old' in engine.context()['recent_actions']
    result=engine.action({'id':'read','type':'read_action','action':'old'})
    assert 'counts' in result['text']
    assert engine.context()['progress']['experiments']==0
    assert engine.action({'id':'list','type':'list_actions','tool_type':'inspect'})['total']==1


def test_operator_stop_and_unlimited_budget_do_not_make_api_calls(tmp_path):
    engine=make_engine(tmp_path);engine.config['max_token_budget']=None
    write_json(engine.run/'api/history.json',{'budget_charge':99_000_000})
    (engine.run/'STOP').touch()
    with patch('vdc.research_r2.Client') as client:
        assert engine.run_loop()['status']=='paused_by_operator'
        client.return_value.call.assert_not_called()
    assert engine.budget()['token_limit'] is None
    assert engine.budget()['unknown_usage_requests']==1


def test_preparation_inspection_never_returns_private_values(tmp_path):
    engine=make_engine(tmp_path)
    engine.state['actions']['prep']={'type':'prepare','status':'completed'}
    write_json(engine.run/'actions/prep/provenance.json',{})
    write_json(engine.run/'actions/prep/output/summary.json',{'private_measurement':172635.124})
    result=engine.action({'id':'check','type':'inspect_preparation','preparation':'prep','path':'summary.json'})
    assert '172635' not in json.dumps(result)
    assert result['output']['fields']['private_measurement']=='float'
    with pytest.raises(ValueError):
        engine.action({'id':'escape','type':'inspect_preparation','preparation':'prep','path':'../../state.json'})


def test_preparation_fork_preserves_history_without_fabricating_experiments(tmp_path):
    from vdc.research_r2_continue import continue_preparation_run
    from vdc.research_r2_tools import tree_hash
    engine=make_engine(tmp_path)
    parent=engine.private/'runs/old';parent.parent.mkdir(parents=True)
    engine.state['status']='budget_exhausted_partial';engine.state['next_decision']=100
    engine.save()
    import shutil
    shutil.copytree(engine.run,parent)
    write_json(parent/'actions/prep/output/real_preparation.json',{'shape':[2,3]})
    write_json(parent/'api/decision_0099_001.json',{'budget_charge':60000,'usage':{'total_tokens':200}})
    before=tree_hash(parent)
    config={**engine.config,'max_api_calls':1000,'max_token_budget':None}
    target=engine.private/'runs/new'
    with patch('vdc.research_r2.resolve_runtime',return_value='test-runtime'):
        child=continue_preparation_run(engine.repo,engine.private,parent,target,config)
    assert tree_hash(parent)==before
    assert child.state['next_decision']==100 and not child.state['experiments']
    assert child.budget()['requests']==1 and child.budget()['token_limit'] is None
    assert read_json(target/'status.json')['new_decisions']==0
    assert read_json(target/'migration_status.json')['status']=='verified'
    assert (target/'actions/prep/output/real_preparation.json').read_bytes()==(parent/'actions/prep/output/real_preparation.json').read_bytes()
    with pytest.raises(ValueError,match='must not exist'):
        continue_preparation_run(engine.repo,engine.private,parent,target,config)


def test_finish_does_not_accept_empty_inventory_as_completed(tmp_path):
    engine=make_engine(tmp_path)
    r=engine.action({'type':'finish','id':'done','summary':'Everything is complete'})
    assert r['status']=='scope_incomplete_continue'
    assert engine.state['status']!='completed_declared_scope'


def test_engine_code_or_config_change_cannot_resume(tmp_path):
    engine=make_engine(tmp_path); c=dict(engine.config);c['mission']='different'
    with patch('vdc.research_r2.resolve_runtime',return_value='test-runtime'):
        with pytest.raises(ValueError,match='changed'):
            Engine(engine.repo,engine.private,engine.run,c)


def test_interrupted_action_not_automatically_executed_twice(tmp_path):
    engine=make_engine(tmp_path)
    action={'id':'previous','type':'inventory'}
    from vdc.io import object_hash
    engine.state['actions']['previous']={'status':'running','type':'inventory','signature':object_hash(action)}
    engine.save()
    with patch('vdc.research_r2.Client') as client:
        client.return_value.call.side_effect=[{'actions':[action]},RuntimeError('stop fixture')]
        with pytest.raises(RuntimeError):engine.run_loop()
    assert engine.state['actions']['previous']['status']=='interrupted'


def test_failed_action_does_not_cancel_independent_tool(tmp_path):
    engine=make_engine(tmp_path)
    with patch('vdc.research_r2.Client') as client:
        client.return_value.call.side_effect=[{'actions':[{'id':'bad','type':'unknown'},
            {'id':'good','type':'inventory'}]},RuntimeError('stop fixture')]
        with pytest.raises(RuntimeError):engine.run_loop()
    assert engine.state['actions']['bad']['status']=='failed'
    assert engine.state['actions']['good']['status']=='completed'


def test_longer_iteration_does_not_allow_early_finish(tmp_path):
    engine=make_engine(tmp_path)
    engine.config['min_decisions_before_finish']=24
    assert engine.missing()['decisions_before_finish']==23
    engine.state['next_decision']=24
    assert engine.missing()['decisions_before_finish']==0


def test_insufficient_remaining_budget_is_partial_not_api_failure(tmp_path):
    engine=make_engine(tmp_path)
    with patch('vdc.research_r2.Client') as client:
        client.return_value.call.side_effect=RuntimeError('API call/token upper-bound budget exhausted; no further requests submitted')
        result=engine.run_loop()
    assert result['status']=='budget_exhausted_partial'


def test_runtime_materializes_external_symlink(tmp_path):
    from vdc.research_r2_linux import copy_runtime
    original=tmp_path/'original';original.write_text('nameserver 192.0.2.1\n')
    link=tmp_path/'resolv.conf'
    try:link.symlink_to(original)
    except OSError:pytest.skip('OS does not allow test symlinks')
    root=tmp_path/'root'
    # Runtime copying is Linux-specific; Windows absolute path semantics differ.
    import sys
    if sys.platform!='linux':return
    copy_runtime(link,root)
    target=root/str(link).lstrip('/')
    assert not target.is_symlink() and target.read_text()==original.read_text()


def test_private_structure_inspection_excludes_values(tmp_path):
    from vdc.research_r2_tools import inspect_assets
    path=tmp_path/'matrix.npz';save_npz(path,counts=np.full((3,4),987654321.0))
    inv={'assets':{'a':{'path':str(path),'name':path.name,'bytes':path.stat().st_size,'source':'s','public':False}}}
    r=inspect_assets(inv,['a'])['a']
    assert r['arrays'][0]['shape']==(3,4)
    assert '987654321' not in json.dumps(r)


def test_completed_invalid_answers_get_bounded_corrective_decisions(tmp_path):
    engine=make_engine(tmp_path)
    def bad_answer(key,*args,**kwargs):
        write_json(engine.run/'api'/f'{key}_001.json',{'status':'failed','error_type':'ValueError',
            'finish_reason':'length','content':'{','budget_charge':100})
        raise RuntimeError('provider answer invalid')
    with patch('vdc.research_r2.Client') as client:
        client.return_value.call.side_effect=bad_answer
        result=engine.run_loop()
    assert result['status']=='format_failures_partial'
    assert result['next_decision']==4 and not result['experiments']

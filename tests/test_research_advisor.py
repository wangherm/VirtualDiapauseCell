"""No paid calls: strict advisory JSON and nonblocking failure regressions."""
import json
from unittest.mock import patch
import pytest
from vdc.io import read_json, write_json, sha256
from vdc.research_api import parse_object
from vdc.research_codex import advise, validate_advice, advisor_cooldown, controller
from vdc.research_bridge import rpc, snapshot, process_one
from tests.test_research_r2 import make_engine


def test_flat_advice_contract_rejects_partial_arrays_and_tool_fields():
    good={'concerns':'c', 'recommended_next_experiment':'e', 'evidence_gaps':'g'}
    validate_advice(good)
    for bad in ({**good,'actions':[]}, {**good,'concerns':['c']},
                {**good,'evidence_gaps':''}, {**good,'concerns':'c'*4001}):
        with pytest.raises(ValueError): validate_advice(bad)
    # Both observed failures mixed a string recommendation with an array closer.
    malformed='{"concerns":"x","recommended_next_experiment":"y"\n],"evidence_gaps":"z"}'
    with pytest.raises(json.JSONDecodeError): parse_object(malformed)


def test_invalid_advice_is_preserved_never_salvaged_or_retried(tmp_path, monkeypatch):
    monkeypatch.setenv('VDC_RESEARCH_API_KEY','test-only-value')
    ledger=tmp_path/'advisor/000049/api/advice_001.json'
    raw={'status':'failed','error_type':'JSONDecodeError','finish_reason':'stop',
         'content':'{"concerns":"x","recommended_next_experiment":"y" ],"evidence_gaps":"z"}',
         'budget_charge':20}
    def call(*args, **kwargs):
        write_json(ledger,raw)
        raise RuntimeError('fixture invalid JSON')
    with patch('vdc.research_codex.Client') as client:
        client.return_value.call.side_effect=call
        result=advise(tmp_path,49,'question',{}, {})
        digest=sha256(ledger)
        assert result['status']=='failed_invalid_format'
        assert not result['main_workflow_blocked']
        assert 'response' not in result
        assert result['api_attempted']
        assert advise(tmp_path,49,'question',{}, {}) == result
        assert client.return_value.call.call_count == 1
        assert sha256(ledger)==digest


def test_cooldown_skips_without_paid_call_and_expires(tmp_path, monkeypatch):
    monkeypatch.setenv('VDC_RESEARCH_API_KEY','test-only')
    for turn in (49,50):
        write_json(tmp_path/f'advisor/{turn:06d}/status.json', {'schema_version':2,
            'status':'failed_invalid_format','api_attempted':True,'local_turn':turn})
    with patch('vdc.research_codex.Client') as client:
        for turn in (51,55):
            assert advise(tmp_path,turn,'new question',{}, {})['status']=='not_run_advisor_cooldown'
        client.assert_not_called()
    assert advisor_cooldown(tmp_path,56) is None
    assert advisor_cooldown(tmp_path,55) == 55


def test_failed_review_does_not_replay_turn48_or_block_turn50(tmp_path, monkeypatch):
    monkeypatch.setenv('VDC_RESEARCH_API_KEY','test-only')
    e=make_engine(tmp_path)
    e.state['next_decision']=49
    e.state['actions']['old48']={'type':'inventory','status':'completed','result':{'old':True}}
    snapshot(e)
    workspace=tmp_path/'local'
    write_json(workspace/'controller_state.json',{'next_turn':49,'status':'advisor_unavailable_requires_inspection'})
    def remote(c,packet):
        r=rpc(e.run,packet)
        if packet['operation']=='submit': process_one(e)
        if packet['operation']=='poll' and packet['index']==50:
            (workspace/'STOP').write_text('fixture stop')
        return r
    def primary(w,index,model):
        return {'actions':[{'id':'new'+str(index),'type':'inventory'}], 'advisor_question':'review'}
    with patch('vdc.research_codex.transport',side_effect=remote), patch('vdc.research_codex.run_codex',side_effect=primary) as model, patch('vdc.research_codex.Client') as client:
        client.return_value.call.side_effect=RuntimeError('fixture failure')
        state=controller({'transport':'files','run':str(e.run)},workspace,continuous=True,
                         advisor_config={'max_api_calls':10})
    assert [c.args[1] for c in model.call_args_list]==[49,50]
    assert state['next_turn']==51
    assert state['status']=='paused_by_operator'
    assert e.state['actions']['old48']['result']=={'old':True}
    assert set(e.state['actions'])=={'old48','new49','new50'}


def test_successful_call_has_json_example_and_validator(tmp_path, monkeypatch):
    monkeypatch.setenv('VDC_RESEARCH_API_KEY','test-only')
    good={'concerns':'c','recommended_next_experiment':'e','evidence_gaps':'g'}
    with patch('vdc.research_codex.Client') as client:
        client.return_value.call.return_value=good
        r=advise(tmp_path,49,'q',{}, {'json_response_format':False})
        assert client.call_args.args[1]['json_response_format'] is True
        args=client.return_value.call.call_args.args
        assert 'EXAMPLE JSON OUTPUT' in args[1]
        assert args[3] is validate_advice
        assert r['status']=='completed'

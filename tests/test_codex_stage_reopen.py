import importlib.util
from pathlib import Path
import copy
import pytest
from vdc.io import read_json,write_json
from vdc.research_bridge import snapshot
from tests.test_research_r2 import make_engine

spec=importlib.util.spec_from_file_location('reopen_stage',Path(__file__).resolve().parents[1]/'scripts/reopen_codex_stage.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

def setup(tmp_path):
    e=make_engine(tmp_path)
    e.state.update(status='completed_declared_scope',next_decision=151,final_summary='old stage')
    e.save();snapshot(e)
    brief=tmp_path/'brief.md';brief.write_text('Authorised new scientific question')
    return e,brief

def test_stage_transition_preserves_history_and_is_idempotent(tmp_path):
    e,b=setup(tmp_path);before=copy.deepcopy(e.state)
    result=mod.reopen(e,151,b,snapshot,apply=True)
    assert result['status']=='completed' and not result['scientific_approval']
    assert e.state['next_decision']==151
    for k in before:
        if k!='status':assert e.state[k]==before[k]
    assert read_json(e.run/'stage_reviews/before_000151/before_state.json')==before
    assert mod.reopen(e,151,b,snapshot,apply=True)['operation']=='already_reopened_no_replay'
    assert len(e.state['operator_stage_reviews'])==1
    b.write_text('Different objective')
    with pytest.raises(ValueError,match='inspection'):mod.reopen(e,151,b,snapshot,True)

def test_dry_run_does_not_reopen(tmp_path):
    e,b=setup(tmp_path)
    assert mod.reopen(e,151,b,snapshot)['status']=='validated_not_applied'
    assert e.state['status']=='completed_declared_scope'
    assert not (e.run/'stage_reviews').exists()

@pytest.mark.parametrize('kind',['request','running','response'])
def test_pending_or_uncertain_mailbox_rejected(tmp_path,kind):
    e,b=setup(tmp_path)
    write_json(e.run/'mailbox'/f'000151.{kind}.json',{})
    with pytest.raises(ValueError):mod.reopen(e,151,b,snapshot,True)
    assert e.state['status']=='completed_declared_scope'

def test_misaligned_or_unreviewed_stage_rejected(tmp_path):
    e,b=setup(tmp_path)
    with pytest.raises(ValueError,match='rewind'):mod.reopen(e,150,b,snapshot,True)
    e.state['status']='stalled_no_progress';e.save();snapshot(e)
    with pytest.raises(ValueError,match='reviewed'):mod.reopen(e,151,b,snapshot,True)

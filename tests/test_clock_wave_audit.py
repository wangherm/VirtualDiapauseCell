import copy,json,sys,zipfile
from pathlib import Path
import pytest
from vdc.io import write_json,read_json,sha256
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_clock_wave_runs import inspect_run,scan,PROTOCOL
from pack_clock_wave_report import pack
from run_clock_wave import task_signature


def fixture(root,name='old',revision=False):
    run=root/'runs'/name
    plan=([{'id':'identity_original','kind':'identity'}]+
          [{'id':kind+'_'+view,'kind':kind} for kind in ('residual','support') for view in ('bulk','core','coarse')]+
          [{'id':kind+'_'+mode,'kind':kind} for kind in ('qwen','chain') for mode in ('base','domain')]) if revision else [{'id':'old_numeric','kind':'numeric'}]
    c={'protocol':{'protocol':PROTOCOL if revision else 'VDC_CW1_development_clock_identity_wave','identity_protocol':'short_slots_v2' if revision else 'legacy'}}
    write_json(run/'config.json',c);write_json(run/'plan.json',plan)
    tasks={}
    for j in plan:
        out=run/'tasks'/j['id'];result={'status':'evaluated'}
        if j['kind']=='residual':result.update(hidden_input_invariance='passed',clock_refitted=False)
        if j['kind']=='support':result.update(support_by_condition_identity=[],range_changed=False)
        if j['kind']=='identity':
            out.mkdir(parents=True,exist_ok=True);(out/'cards.jsonl').write_text('{}\n',encoding='utf-8')
        if j['kind']=='qwen':
            result.update(contract='short_slots_v2',generation_count=3)
            write_json(out/'identity_original.json',{'complete':True,'responses':[dict(mapping={},prompt_tokens=10,response_tokens=4,raw_answer='{}',stop_observation='eos',identity_status='schema_failure',json_valid=True,schema_valid=False,evidence_valid=None)]})
        if j['kind']=='chain':write_json(out/'stage_status.json',[])
        write_json(out/'result.json',result)
        tasks[j['id']]={'status':'completed','files':{p.relative_to(run).as_posix():sha256(p) for p in out.iterdir()},
                       'execution_kind':'reused_verified' if j['kind']=='identity' else 'added'}
    write_json(run/'queue_status.json',{'status':'completed_current_scope','tasks':tasks})
    return run


def test_scan_chooses_verified_revision_even_if_latest_stale(tmp_path):
    old=fixture(tmp_path);new=fixture(tmp_path,'not_named_clock_wave',True)
    (tmp_path/'LATEST_CLOCK_WAVE.txt').write_text(str(old))
    r=scan(tmp_path)
    assert r['selected_run']==str(new) and r['next_action']=='export_completed_revision'
    assert not r['latest_matches_selected'] and not r['training_started_by_audit']
    assert inspect_run(new)['fresh_tasks_verified']==10
    q=read_json(new/'queue_status.json');q['tasks']['identity_original']['task_signature']='original-parent-signature';write_json(new/'queue_status.json',q)
    assert inspect_run(new)['targeted_complete']  # Explicit parent reuse keeps the parent's signature.


def test_revision_name_or_completed_state_does_not_prove_new_outputs(tmp_path):
    old=fixture(tmp_path,'clock_wave_revision_misnamed')
    assert not inspect_run(old)['revision_config_and_plan']
    new=fixture(tmp_path,'new',True)
    path=new/'tasks/qwen_base/result.json';write_json(path,{'status':'evaluated','old_long_json':True})
    q=read_json(new/'queue_status.json');q['tasks']['qwen_base']['files']['tasks/qwen_base/result.json']=sha256(path);write_json(new/'queue_status.json',q)
    r=inspect_run(new);assert not r['targeted_complete']
    assert any(x['task']=='qwen_base' for x in r['categories']['unfinished'])


def test_partial_run_is_exportable_but_not_restarted(tmp_path):
    run=fixture(tmp_path,'revision',True);q=read_json(run/'queue_status.json')
    q['tasks']['qwen_domain']={'status':'running'};write_json(run/'queue_status.json',q)
    r=scan(tmp_path);assert r['next_action']=='export_partial_revision_do_not_restart'
    assert r['selected_run']==str(run)


def test_pack_is_explicit_repack_with_frozen_payload_and_rejects_old_as_revision(tmp_path):
    run=fixture(tmp_path)
    with pytest.raises(ValueError,match='not a revision'):pack(run,require_revision=True)
    a=pack(run);b=pack(run)
    with zipfile.ZipFile(a) as za,zipfile.ZipFile(b) as zb:
        aa=json.loads(za.read('PACKAGING_PROVENANCE.json'));bb=json.loads(zb.read('PACKAGING_PROVENANCE.json'))
        assert bb['operation']=='repack_only' and bb['identical_previous_packages']==[str(a)]
        assert aa['payload_fingerprint']==bb['payload_fingerprint']
        assert bb['explicit_run']==str(run) and not bb['packaging_executed_training']
        assert 'config.json' not in zb.namelist()


def test_changed_prompt_protocol_or_features_invalidate_completion_signature():
    c={'code':'a','protocol':{'identity_protocol':'short_slots_v2'}};j={'id':'qwen_base','kind':'qwen'}
    before=task_signature(c,j);other=copy.deepcopy(c);other['code']='b'
    assert task_signature(other,j)!=before
    other=copy.deepcopy(c);other['protocol']['identity_protocol']='new'
    assert task_signature(other,j)!=before


def test_uninitialized_directory_prevents_duplicate_submission(tmp_path):
    fixture(tmp_path);(tmp_path/'runs/clock_wave_revision_pending').mkdir()
    r=scan(tmp_path)
    assert r['next_action']=='inspect_uninitialized_or_unreadable_run'

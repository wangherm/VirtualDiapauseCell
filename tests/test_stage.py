import json,sys
from pathlib import Path
import numpy as np
import pytest
from vdc import identity_protocol as protocol
from vdc.io import read_json,write_json,sha256,object_hash
from vdc.stage import build_view,finalize,predict_route
from vdc.clock_wave_revision import residual_task
from tests.test_clock_wave_revision import setup_data,card
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))


def test_fixed_evidence_fields_do_not_repair_old_answers_or_relax_gate():
    _,mapping=protocol.short_card(card(),'original')
    raw='{"identity":"I01","e1":"m00","e2":null,"e3":null,"status":"supported"}'
    assert protocol.parse_fixed(raw,mapping)['prediction']=='beta'
    assert protocol.parse_fixed(raw.replace('I01','I00'),mapping)['identity_status']=='numeric_conflict'
    for wrong in ('{"identity":"I01","evidence_slots":["m00"],"status":"supported"}',
                  raw.replace('"e1":"m00"','"e1":["m00","m01"]'),raw[:-1],
                  raw.replace('"e2":null','"e2":"m00"')):
        assert not protocol.parse_fixed(wrong,mapping)['format_valid']
    assert protocol.parse_fixed(raw.replace('m00','m99'),mapping)['identity_status']=='evidence_invalid'
    unknown='{"identity":"unknown","e1":null,"e2":null,"e3":null,"status":"unknown"}'
    assert protocol.parse_fixed(unknown,mapping)['identity_status']=='biological_unknown'


def test_stage_plan_excludes_exploration_and_never_reuses_changed_qwen():
    from run_clock_wave import stage_plan
    jobs=stage_plan(9)
    assert len(jobs)==34 and sum(bool(j.get('reuse_parent')) for j in jobs)==24
    assert not any(j['id'].startswith(('pool_','family_','semantic_','stress_')) for j in jobs)
    assert all(not j.get('reuse_parent') for j in jobs if j['kind']=='qwen')
    assert {j['alpha'] for j in jobs if j['id'].startswith('shrink_')}=={100.}
    assert all(j.get('reuse_parent') for j in jobs if j['kind']=='support')


def test_stage_default_replay_and_partial_snapshot_never_imply_gpu_completion(tmp_path):
    x,g,d,r,c=setup_data(tmp_path)
    residual_task(tmp_path,'core',tmp_path/'tasks/residual_core',c)
    cfg={**c,**read_json(ROOT/'configs/stage.json')};cfg['stage_routes']={'core':'clock_identity'}
    folder=tmp_path/'tasks/stage_core';build_view(tmp_path,'core',folder,cfg)
    result=read_json(folder/'result.json');assert result['main_route']=='clock_identity'
    assert result['actual_default_invocation']=='passed'
    files={p.relative_to(tmp_path).as_posix():sha256(p) for p in folder.rglob('*') if p.is_file()}
    states={'stage_core':{'status':'completed','files':files},'qwen_base':{'status':'blocked','reason':'no GPU in software test'}}
    summary=finalize(tmp_path,{'protocol':cfg,'code':'software_fixture'},states)
    assert summary['status']=='partial_not_closed' and summary['new_input_web_inference'] is False
    assert summary['knowledge']['base']['status']=='not_completed'
    snap=read_json(tmp_path/'stage_snapshot.json')
    assert snap['snapshot_id']==object_hash({k:v for k,v in snap.items() if k!='snapshot_id'})
    with pytest.raises(ValueError,match='no fallback'):predict_route(tmp_path,'core',x,['all']*len(x),'automatic_best')


def test_stage_routes_are_prespecified_not_selected_from_new_sensitivity():
    c=read_json(ROOT/'configs/stage.json')
    assert c['stage_routes']=={'bulk':'clock_identity_residual','core':'clock_identity','coarse':'clock_identity'}
    assert c['residual_sensitivity_alpha']==100.


def test_stage_audit_requires_all_three_identity_assays(tmp_path):
    from tests.test_clock_wave_audit import fixture
    from audit_clock_wave_runs import inspect_run
    run=fixture(tmp_path,'stage',True)
    c=read_json(run/'config.json');c['protocol'].update(read_json(ROOT/'configs/stage.json'));write_json(run/'config.json',c)
    jobs=read_json(run/'plan.json');q=read_json(run/'queue_status.json')
    for view in ('bulk','core','coarse'):
        for kind in ('shrink','stage'):
            name=kind+'_'+view;jobs.append({'id':name,'kind':'residual' if kind=='shrink' else 'stage_view'})
            write_json(run/'tasks'/name/'result.json',{'hidden_input_invariance':'passed','clock_refitted':False,'actual_default_invocation':'passed','reload':'passed'})
            q['tasks'][name]={'status':'completed','files':{'tasks/'+name+'/result.json':sha256(run/'tasks'/name/'result.json')}}
    for mode in ('base','domain'):
        out=run/'tasks'/('qwen_'+mode)
        write_json(out/'result.json',{'contract':'fixed_evidence_v3','generation_count':3})
        for condition in ('no_numeric','challenge'):write_json(out/('identity_original_'+condition+'.json'),read_json(out/'identity_original.json'))
        q['tasks']['qwen_'+mode]['files']={p.relative_to(run).as_posix():sha256(p) for p in out.iterdir()}
    write_json(run/'plan.json',jobs);write_json(run/'queue_status.json',q)
    assert inspect_run(run)['targeted_complete']
    path=run/'tasks/qwen_base/identity_original_challenge.json';write_json(path,{'complete':False,'responses':[]})
    q['tasks']['qwen_base']['files'][path.relative_to(run).as_posix()]=sha256(path);write_json(run/'queue_status.json',q)
    assert not inspect_run(run)['targeted_complete']

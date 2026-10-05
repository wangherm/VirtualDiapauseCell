import copy,json
from pathlib import Path
import numpy as np
import pytest
from vdc import identity_protocol as protocol
from vdc.clock_wave import ClockWave
from vdc.clock_wave_revision import residual_features, residual_task, support_task
from vdc.io import write_json, read_json, save_npz, sha256
from tests.test_clock_wave import fixture


def card():
    return {'query_id':'q1','observed_markers':[{'gene':'geneA','symbol':'A','relative_log_expression':2.},
              {'gene':'geneB','symbol':'B','relative_log_expression':0.}],
            'reference_markers':{'alpha':[{'gene':'referenceOnly'}],'beta':[{'gene':'geneA'}]},
            'numeric_candidates':[{'identity':'beta','distance':0.1},{'identity':'alpha','distance':2.}],
            'numeric_gate':'beta'}


def test_protocol_nohint_hides_ranking_and_never_uses_zero_or_reference_only_evidence():
    c,m=protocol.short_card(card(),'no_numeric')
    assert 'numeric_gate' not in c and 'numeric_candidates' not in c
    assert list(m['slot_map'])==['m00'] and m['identity_map']=={'I00':'alpha','I01':'beta'}
    assert (c,m)==protocol.short_card(card(),'no_numeric')
    a=protocol.parse('{"identity":"I01","evidence_slots":["referenceOnly"],"status":"supported"}',m)
    assert a['json_valid'] and a['schema_valid'] and a['identity_status']=='evidence_invalid'
    a=protocol.parse('{"identity":"I01","evidence_slots":["m00"],"status":"supported"}',m)
    assert a['prediction']=='beta' and a['evidence_genes']==['geneA']


def test_failure_layers_and_independent_nohint_scoring():
    _,m=protocol.short_card(card(),'original')
    assert protocol.parse('{"identity":',m)['identity_status']=='parse_failure'
    assert protocol.parse('{"identity":"I01"}',m)['identity_status']=='schema_failure'
    unknown='{"identity":"unknown","evidence_slots":[],"status":"unknown"}'
    assert protocol.parse(unknown,m)['identity_status']=='biological_unknown'
    text='{"identity":"I00","evidence_slots":["m00"],"status":"supported"}'
    assert protocol.parse(text,m)['identity_status']=='numeric_conflict'
    _,independent=protocol.short_card(card(),'no_numeric')
    assert protocol.parse(text,independent)['prediction']=='alpha'
    assert protocol.stop_record([10,20],20,256)['stop_observation']=='eos'
    assert protocol.stop_record([10]*256,20,256)['stop_observation']=='token_budget_reached'
    assert protocol.stop_record([10],20,256)['stop_observation']=='other_stop_unknown'


def test_challenges_are_deterministic_and_do_not_disclose_removed_evidence():
    kinds=set()
    for i in range(20):
        c=card();c['query_id']=str(i);public,m=protocol.short_card(c,'challenge');kinds.add(m['challenge']['kind'])
        assert 'numeric_gate' not in public and 'challenge' not in public
        if m['challenge']['kind']=='remove_numeric_top_candidate':
            assert 'beta' not in m['identity_map'].values()
        else:assert 'm00' not in m['slot_map']
    assert len(kinds)==2


def setup_data(tmp):
    x,g,d,r,c=fixture();save_npz(tmp/'data/core/counts.npz',counts=x)
    write_json(tmp/'data/core/data.json',{'genes':g,'definitions':d,'rows':r,'counts_sha256':sha256(tmp/'data/core/counts.npz')})
    for rep in ('C0','C1','C2'):ClockWave.fit(x,g,d,r,rep,c).save(tmp/'tasks'/('numeric_core_'+rep)/'model')
    return x,g,d,r,{**c,'anchor_bootstrap_draws':5}


def test_residual_readout_reload_targets_and_outer_invariance(tmp_path):
    x,g,d,r,c=setup_data(tmp_path);out=tmp_path/'tasks/residual_core'
    residual_task(tmp_path,'core',out,c);first=read_json(out/'result.json')
    assert first['hidden_input_invariance']=='passed' and first['clock_refitted'] is False
    params=(out/'readout.npz').read_bytes()
    with np.load(out/'readout.npz') as z:before={k:z[k] for k in z.files}
    changed=x.copy();changed[8:]=changed[8:]*77+11
    save_npz(tmp_path/'data/core/counts.npz',counts=changed)
    meta=read_json(tmp_path/'data/core/data.json');meta['counts_sha256']=sha256(tmp_path/'data/core/counts.npz');write_json(tmp_path/'data/core/data.json',meta)
    residual_task(tmp_path,'core',tmp_path/'second',c)
    with np.load(tmp_path/'second/readout.npz') as z:
        for k,v in before.items():np.testing.assert_array_equal(v,z[k])


def test_support_preserves_model_and_marks_unreplicated_anchors(tmp_path):
    x,g,d,r,c=setup_data(tmp_path)
    model=tmp_path/'tasks/numeric_core_C2/model/model.json';prior=sha256(model)
    support_task(tmp_path,'core',tmp_path/'support',c)
    result=read_json(tmp_path/'support/result.json')
    assert result['range_changed'] is False and sha256(model)==prior
    assert result['C1_translation_max_abs_delta']==0
    assert result['anchor_sensitivity']['all']['insufficient_replication_for_uncertainty']


def test_revision_queue_does_not_retrain_clock_or_reuse_qwen():
    from run_clock_wave import revision_plan
    plan=revision_plan(9)
    assert len(plan)==61
    assert sum(j.get('reuse_parent',False) for j in plan)==51
    assert all(j.get('reuse_parent') for j in plan if j['kind']=='numeric')
    assert all(not j.get('reuse_parent') for j in plan if j['kind']=='qwen')
    assert len([j for j in plan if j['kind']=='residual'])==3


def test_parent_reuse_checks_data_and_roles_without_mutating_parent(tmp_path):
    from run_clock_wave import reuse_parent
    parent=tmp_path/'parent';run=tmp_path/'run';private=tmp_path/'private'
    write_json(private/'sample_roles.json',{'approval':'software fixture'})
    write_json(parent/'config.json',{'role_manifest_sha256':sha256(private/'sample_roles.json')})
    write_json(parent/'tasks/prepare/result.json',{'status':'prepared'})
    write_json(parent/'data/data.json',{'test':'cache'})
    files={p.relative_to(parent).as_posix():sha256(p) for folder in ('tasks','data') for p in (parent/folder).rglob('*') if p.is_file()}
    write_json(parent/'queue_status.json',{'tasks':{'prepare':{'status':'completed','files':files}}})
    before=sha256(parent/'queue_status.json')
    state=reuse_parent(parent,run,{'id':'prepare','kind':'prepare'},private)
    assert state['execution_kind']=='reused_verified' and sha256(parent/'queue_status.json')==before
    write_json(parent/'data/data.json',{'test':'modified'})
    with pytest.raises(ValueError):reuse_parent(parent,run,{'id':'prepare','kind':'prepare'},private)


def test_identity_chain_common_support_keeps_rejections_in_coverage(tmp_path):
    from run_clock_wave import compare_identity_chains
    names=['numeric_coarse_C2','chain_numeric','chain_base','chain_domain'];states={}
    rows=[{'study_family':'fixture','biological_unit':'u'+str(i)} for i in range(2)]
    for name in names:
        folder=tmp_path/'tasks'/name/'validation';pred=np.ones((2,2))
        status=['located','located']
        if name=='chain_domain':pred[1]=np.nan;status[1]='unsupported_identity'
        save_npz(folder/'predictions.npz',hidden_prediction=pred,hidden_target=np.ones((2,2)))
        write_json(folder/'rows.json',rows);write_json(folder/'query_status.json',{'status':status});states[name]={'status':'completed'}
    r=compare_identity_chains(tmp_path,states)
    assert r['common_in_reference_coverage']==.5
    assert r['models']['chain_numeric']['own_support']['coverage']==1
    assert r['models']['chain_numeric']['common_support']['coverage']==.5
    states['chain_base']['status']='failed'
    assert compare_identity_chains(tmp_path,states)['status']=='unavailable'

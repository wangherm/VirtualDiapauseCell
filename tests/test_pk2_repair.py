import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from vdc.io import read_json,read_jsonl,write_json,object_hash,sha256
from vdc.pk2_semantics import semantic_mode,validate_semantics,load_semantics,matrix_hash
from vdc.pk2_repair import repair_invalidations,import_verified,AUDITED_SOURCE_CODE
from vdc.pk2_numeric import train_job,dynamic_ratio
from vdc.state import fit_state,StateConfig
from vdc.knowledge_train import evidence_cases
from .fixtures import observation_fixture
from .test_pk2_training import ROOT,build_plan
from run_all_modules import inventory,verify_inventory


def test_literal_catalogue_semantic_conditions_and_targeted_retrains():
    c=read_json(ROOT/'configs/pk2_train.json');c['repair_from']='audited-source'
    plan=build_plan(c);jobs=[t['params']['job'] for t in plan.values() if t['kind']=='numeric']
    modes=[semantic_mode(j) for j in jobs if j['family']=='semantic_increment']
    assert modes.count('base')==12 and modes.count('domain')==12 and modes.count('shuffle')==12
    invalid=repair_invalidations(plan)
    numerical=[plan[n]['params']['job'] for n in invalid if plan[n]['kind']=='numeric']
    assert len(numerical)==42
    assert sum(j['family']=='pool_generalization' for j in numerical)==18
    assert not any(plan[n]['kind']=='qwen' for n in invalid)
    for value in ('new_domian',None,'zero'):
        with pytest.raises(ValueError):semantic_mode({'family':'semantic_increment','semantics':value})


def test_cached_adapter_order_and_shuffle_identity(tmp_path):
    features=['b','a','c'];v=np.arange(1,13,dtype=np.float32).reshape(3,4)
    adapter=tmp_path/'qwen/trained/adapter';adapter.mkdir(parents=True);(adapter/'weights').write_text('software fixture')
    write_json(tmp_path/'select_adapter/selection.json',{'selected':{'job':'qwen'}})
    cache=tmp_path/'embeddings_domain';cache.mkdir()
    np.savez(cache/'embeddings.npz',vectors=v,feature_ids=np.array(['a','b','c']))
    meta={'arrays_sha256':sha256(cache/'embeddings.npz'),'weight_kind':'domain_adapter',
          'adapter_hash':object_hash({p.name:sha256(p) for p in adapter.iterdir()})}
    write_json(cache/'embeddings.json',meta)
    c={'semantic_dim':4,'semantic_shuffle_seed':20261004}
    j={'family':'semantic_increment','semantics':'new_domain'}
    m,p=load_semantics(tmp_path,j,features,c);np.testing.assert_equal(m,v[[1,0,2]])
    j['semantics']='shuffled_new_domain';s,q=load_semantics(tmp_path,j,features,c)
    np.testing.assert_equal(s,m[q['permutation']]);assert matrix_hash(s)!=matrix_hash(m)
    for bad,prov in [(np.zeros_like(s),q),(s,{**q,'feature_ids':['x']}),(s,{**q,'adapter_hash':None}),
                     (s,{**q,'permutation':[0,0,1]})]:
        with pytest.raises(ValueError):validate_semantics(j,bad,prov,features,4)
    (adapter/'weights').write_text('changed')
    with pytest.raises(ValueError,match='selected adapter'):load_semantics(tmp_path,j,features,c)


def test_nonzero_semantics_reaches_actual_checkpoint_and_forward(tmp_path):
    b=observation_fixture();b.save(tmp_path/'input')
    features=b.feature_ids;m=np.random.default_rng(7).normal(size=(len(features),4)).astype('float32')
    p={'weight_kind':'domain_adapter','adapter_hash':'explicit_synthetic_software_fixture',
       'arrays_sha256':'fixture','feature_ids':features}
    j={'family':'semantic_increment','semantics':'new_domain','max_steps':2,'seed':42}
    c={'state_config':{'hidden_dim':8,'latent_dim':4,'layers':1,'heads':2,'validation_every':1},'semantic_dim':4,'mask_repetitions':1}
    train_job(tmp_path/'input',tmp_path/'run',j,c,semantics=m,semantic_provenance=p)
    a=read_json(tmp_path/'run/semantic_input.json')
    assert a['pretraining_forward_check']['max_abs_delta']>0
    assert a['forward_check']['max_abs_delta']['programme']>0
    ck=torch.load(tmp_path/'run/model/best.pt',weights_only=True)
    np.testing.assert_array_equal(ck['model']['semantics'].numpy(),m)
    with pytest.raises(ValueError,match='provenance|zero'):
        train_job(tmp_path/'input',tmp_path/'bad',j,c)


def test_outer_values_and_clock_cannot_select_fixed_final_weights(tmp_path):
    b=observation_fixture();other=copy.deepcopy(b)
    other.values[other.indices('validation')]+=1000
    other.clock[other.indices('validation')]-=500
    cfg=StateConfig(hidden_dim=8,latent_dim=4,layers=1,heads=2,validation_every=1)
    for name,bundle in [('first',b),('changed',other)]:
        fit_state(bundle,tmp_path/name,3,cfg,checkpoint_selection='fixed_final',checkpoint_steps=[1,2,3])
        logs=read_jsonl(tmp_path/name/'steps.jsonl')
        assert all(r['selection_score'] is None and r['validation_hidden_mse_scaled'] is None for r in logs)
    a=torch.load(tmp_path/'first/best.pt',weights_only=True);d=torch.load(tmp_path/'changed/best.pt',weights_only=True)
    assert a['step']==d['step']==3
    for key in a['model']:torch.testing.assert_close(a['model'][key],d['model'][key],rtol=0,atol=0)
    with pytest.raises(ValueError,match='selection protocol'):
        fit_state(b,tmp_path/'first',4,cfg,resume=True)
    assert dynamic_ratio(np.ones((1,3)),np.ones((1,3)),np.ones((1,3),bool))['dynamic_std_ratio_median'] is None


def test_verified_import_keeps_original_bytes_and_fails_on_drift(tmp_path):
    old=tmp_path/'old';new=tmp_path/'new';new.mkdir();c={'a':1}
    plan={'existing':{'kind':'numeric','params':{'job':{'family':'killifish_adaptation'}}}}
    write_json(old/'config.json',c);write_json(old/'plan.json',plan)
    write_json(old/'run_manifest.json',{'code':AUDITED_SOURCE_CODE,'signature':'fixture','integration_only':False})
    write_json(old/'tasks/existing/result.json',{'status':'fixture_not_research'})
    files=inventory(old/'tasks/existing')
    write_json(old/'queue_status.json',{'tasks':{'existing':{'status':'evaluated_new','files':files}}})
    before=inventory(old)
    got=import_verified(old,new,plan,c,verify_inventory,inventory)
    assert got['existing']['status']=='reused_verified' and got['existing']['new_execution'] is False
    assert inventory(old)==before and inventory(new/'tasks/existing')==files
    write_json(old/'tasks/existing/result.json',{'changed':True});second=tmp_path/'second';second.mkdir()
    with pytest.raises(ValueError,match='inventory changed'):import_verified(old,second,plan,c,verify_inventory,inventory)


def test_paired_evidence_has_answerable_cases_and_no_completion_in_index():
    rows=read_jsonl(ROOT/'knowledge/pk2/literature.jsonl')
    cases=evidence_cases(rows,balanced=True)
    support=[r for r,m in cases if m=='addressed_supported']
    empty=[r for r,m in cases if m=='addressed_withheld']
    assert len(support)==len(empty)>0
    assert any(json.loads(r['completion'])['answer']!='unknown' for r in support)
    assert all(json.loads(r['completion'])['answer']=='unknown' for r in empty)
    assert all(r['evaluation_inputs']['documents']==[] and 'Evidence type:' not in r['prompt'][-1]['content'] for r in empty)
    changed=copy.deepcopy(rows)
    for r in changed:r['completion']=json.dumps({'answer':'SECRET_GOLD','source':'x','context':'x','uncertain':False})
    new=evidence_cases(changed,balanced=True)
    assert [r['prompt'] for r,m in cases]==[r['prompt'] for r,m in new]
    assert all('SECRET_GOLD' not in json.dumps(r['evaluation_inputs']) for r,m in new)

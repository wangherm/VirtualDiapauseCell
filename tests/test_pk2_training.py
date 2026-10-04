import copy,json,sys
from pathlib import Path
import numpy as np
import pytest
import torch
from vdc.io import read_json,write_json,object_hash,read_jsonl
from vdc.state import fit_state,StatePredictor,StateConfig,ProgrammeStateModel
from vdc.knowledge import audit_knowledge
from .fixtures import observation_fixture
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_pk2 import build_plan

def test_full_queue_dependencies_are_real_and_acyclic():
    plan=build_plan(read_json(ROOT/'configs/pk2_train.json'))
    logical={k for k,v in plan.items() if v['kind'] in {'numeric','qwen'}}
    assert len(logical)==171
    assert sum(v['kind']=='qwen' for v in plan.values())==6
    done=set()
    while len(done)<len(plan):
        ready={k for k,v in plan.items() if k not in done and set(v['depends'])<=done}
        assert ready,'cyclic/unresolved dependencies'
        done|=ready
    assert 'embeddings_domain' not in plan['adapt_R3_bulk_42']['depends']
    assert 'adapt_R0_bulk_42' in plan['learning_1_R0_43']['depends']
    assert 'data_GSE303716' not in plan['source_P-R']['depends']
    assert sum(plan[k]['params']['job']['family']=='feature_resolution' for k in logical)==18

def test_expanded_literature_source_family_split_and_no_numeric_outcomes():
    records=read_jsonl(ROOT/'knowledge/pk2/literature.jsonl');audit=audit_knowledge(records)
    assert audit['records']==90 and len({r['paper_id'] for r in records})==30
    assert all('curated_weak_reference' in r['label_source'] for r in records)
    assert not any('GSE288723' in json.dumps(r) or 'GSE291659' in json.dumps(r) for r in records)
    assert {r['split'] for r in records}=={'train','validation'}

def test_multicontext_scaling_ignores_validation_and_transfers_only_encoder(tmp_path):
    b=observation_fixture();b=b.subset(np.r_[b.indices('train'),b.indices('validation')])
    for i,r in enumerate(b.rows):r['study_family']='A' if i%2 else 'B'
    b.values[::2]+=100
    cfg=StateConfig(hidden_dim=8,latent_dim=4,layers=1,heads=2,validation_every=1)
    fit_state(b,tmp_path/'parent',2,cfg,multi_context=True,checkpoint_steps=[1,2])
    parent=StatePredictor.load(tmp_path/'parent');assert parent.mean.shape==(2,len(b.feature_ids))
    changed=copy.deepcopy(b);changed.values[changed.indices('validation')]+=10000
    fit_state(changed,tmp_path/'changed',1,cfg,multi_context=True)
    np.testing.assert_array_equal(parent.mean,StatePredictor.load(tmp_path/'changed').mean)
    assert (tmp_path/'parent/step_1.pt').exists()
    fit_state(b,tmp_path/'encoder',1,cfg,pretrained=tmp_path/'parent',transfer_mode='encoder_only')
    keys=read_json(tmp_path/'encoder/run.json')['transfer']['copied_parameters']
    assert keys and all(k.startswith(('numeric.','feature_id.','encoder.','to_latent.','gene_encoder.')) for k in keys)
    fit_state(b,tmp_path/'shared',1,cfg,pretrained=tmp_path/'parent')
    shared=read_json(tmp_path/'shared/run.json')['transfer']['copied_parameters']
    assert len(shared)>len(keys) and not any(k.startswith('context_readout.') for k in shared)

def test_gene_encoder_uses_gene_input_gradients():
    cfg=StateConfig(hidden_dim=8,latent_dim=4,layers=1,heads=2,encoder_kind='gene_mlp')
    m=ProgrammeStateModel(50,cfg);x=torch.randn(4,50)
    got=m(x,torch.ones_like(x,dtype=torch.bool),torch.ones_like(x))['programme']
    got.square().mean().backward()
    assert next(m.gene_encoder.parameters()).grad.abs().sum()>0

def test_development_fold_never_admits_reserved_sample(tmp_path,monkeypatch):
    from vdc.admission import approve_roles,validate_internal_row
    rows=[{'sample_key':'a','biological_unit':'p1','link_ids':['a','p1'],'cohort_id':'c1','role':'development','split':'train','allowed_tasks':['state']},
        {'sample_key':'r','biological_unit':'p2','link_ids':['r','p2'],'cohort_id':'c2','role':'reserved_evaluation','split':'locked_test','allowed_tasks':[]}]
    p=tmp_path/'roles.json';write_json(p,{'protocol':'VDC_PK1','status':'draft','blocking_questions':[],'samples':rows});h=approve_roles(p,'software test')
    monkeypatch.setenv('VDC_ROLE_MANIFEST',str(p))
    f={'protocol':'PK2_development_resampling','parent_approval_hash':h,'assignments':{'a':'validation','r':'train'}};f['hash']=object_hash(f)
    fp=tmp_path/'fold.json';write_json(fp,f);monkeypatch.setenv('VDC_DEVELOPMENT_FOLD',str(fp))
    row={'source_sample_key':'a','biological_unit':'p1','link_ids':['a','p1'],'admission_protocol':'VDC_PK1','admission_hash':h,'split':'validation','development_fold_hash':f['hash']}
    with pytest.raises(ValueError,match='development|reserved|Reserved'):validate_internal_row(row)

def test_training_packer_excludes_expression_and_weights(tmp_path):
    from pack_pk2_training_report import pack
    import zipfile
    run=tmp_path/'run';write_json(run/'queue_status.json',{'tasks':{}})
    write_json(run/'tasks/a/result.json',{'status':'failed'})
    p=run/'tasks/a/best.pt';p.write_bytes(b'weights')
    p=run/'tasks/a/bundle';p.mkdir();(p/'observations.npz').write_bytes(b'expression')
    z=pack(run,tmp_path/'report.zip')
    with zipfile.ZipFile(z) as archive:
        assert 'tasks/a/result.json' in archive.namelist()
        assert not any(n.endswith(('.pt','.npz')) for n in archive.namelist())

def test_ridge_baseline_does_not_learn_missing_targets_as_zero():
    from vdc.pk2_numeric import dual_ridge
    x=np.arange(8,dtype=float).reshape(4,2);y=np.arange(12,dtype=float).reshape(4,3)
    mask=np.ones_like(y,bool);mask[0,0]=False;mask[:,2]=False
    rows=[{'study_family':'s','biological_unit':str(i)} for i in range(4)]
    p=dual_ridge(x,y,rows,target_mask=mask)(x);y[0,0]=1e10;y[:,2]=-1e20
    q=dual_ridge(x,y,rows,target_mask=mask)(x)
    np.testing.assert_equal(p,q);assert np.isnan(p[:,2]).all()

def test_public_units_follow_conservative_replicate_blocks():
    from vdc.pk2_data import public_rows
    records=[{'accession':str(i),'fields':{'Sample_title':[title]}} for i,title in enumerate(['WT, rep1','KO, rep1','WT, rep3'])]
    rows=public_rows(records,['0','1','2'],'GSE202844')
    assert rows[0]['biological_unit']==rows[1]['biological_unit']
    assert rows[0]['split']=='train' and rows[2]['split']=='validation'

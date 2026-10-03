"""Synthetic fixtures exercise wiring, never claim real Qwen or biological results."""
import copy
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
from vdc.io import read_json,write_json,save_npz,sha256
from vdc.pk1_numeric import KillifishExitReference,save_initial_weights,semantic_cache,state_route
from vdc.state import StateConfig,fit_state
from .fixtures import observation_fixture

ROOT=Path(__file__).resolve().parents[1]

def test_clock_anchor_fit_ignores_validation_and_maintenance(tmp_path):
    genes=[f'synthetic_gene_{i}' for i in range(50)]
    x=np.random.default_rng(1).normal(size=(5,50))
    rows=[{'observation_id':str(i),'origin':'synthetic','split':'train' if i<3 else 'validation',
           'condition':['Early Diapause','Developing Day 7','Late Diapause','Early Diapause','Developing'][i]} for i in range(5)]
    a=KillifishExitReference().fit(x,genes,rows)
    changed=x.copy();changed[2:]+=1000
    b=KillifishExitReference().fit(changed,genes,rows)
    np.testing.assert_array_equal(a.direction,b.direction)
    np.testing.assert_array_equal(a.indices,b.indices)
    assert set(a.metadata['axis_genes']).isdisjoint(a.metadata['readout_genes'])
    assert a.predict((x[0]+2*(x[1]-x[0]))[None],genes)[0]==pytest.approx(2)
    a.save(tmp_path/'axis');loaded=KillifishExitReference.load(tmp_path/'axis')
    np.testing.assert_array_equal(a.predict(x,genes),loaded.predict(x,genes))

def test_shared_initialisation_keeps_actual_semantics_and_provenance(tmp_path):
    b=observation_fixture();cfg=StateConfig(hidden_dim=16,latent_dim=8,layers=1,heads=2,validation_every=1)
    save_initial_weights(b.feature_ids,4,cfg.__dict__,tmp_path/'init.pt')
    sem=np.ones((len(b.feature_ids),4),np.float32)
    fit_state(b,tmp_path/'fit',1,cfg,semantics=sem,semantic_provenance={'fixture':True},initial_weights=tmp_path/'init.pt')
    ck=torch.load(tmp_path/'fit/last.pt',weights_only=True)
    torch.testing.assert_close(ck['model']['semantics'],torch.tensor(sem))
    fit_state(b,tmp_path/'fit',2,cfg,resume=True,semantics=sem,semantic_provenance={'fixture':True})
    assert read_json(tmp_path/'fit/run.json')['initial_weights_sha256']==sha256(tmp_path/'init.pt')
    bad=torch.load(tmp_path/'init.pt',weights_only=True);bad['model'].pop('clock_head.bias');torch.save(bad,tmp_path/'bad.pt')
    with pytest.raises(ValueError,match='architecture'):
        fit_state(b,tmp_path/'bad',1,cfg,semantics=sem,semantic_provenance={'fixture':True},initial_weights=tmp_path/'bad.pt')

def test_all_semantic_routes_use_cache_and_same_evaluation_masks(tmp_path):
    b=observation_fixture();b=b.subset([i for i,r in enumerate(b.rows) if r['split'] in {'train','validation'}]);b.save(tmp_path/'bundle')
    c=read_json(ROOT/'configs/pk1.json');c['state_steps']=1;c['state_config']['validation_every']=1
    c['evaluation_mask_seeds']=[71];c['noise_levels']=[0,.2]
    save_initial_weights(b.feature_ids,4,c['state_config'],tmp_path/'init.pt')
    for kind,folder in [('domain_adapter','domain'),('frozen_base','base')]:
        save_npz(tmp_path/folder/'embeddings.npz',feature_ids=np.array(b.feature_ids),vectors=np.random.default_rng(9).normal(size=(12,4)).astype('float32'))
        write_json(tmp_path/folder/'embeddings.json',{'weight_kind':kind,'arrays_sha256':sha256(tmp_path/folder/'embeddings.npz'),'fixture_only':True})
    with pytest.raises(ValueError,match='identity'):
        semantic_cache(tmp_path/'base',b.feature_ids,'domain_adapter')
    for route in ('K0','K1','K2','K3','K4','K5'):
        state_route(tmp_path/'bundle',tmp_path/route,c,tmp_path/'init.pt',route,4,
            domain=tmp_path/'domain',base=tmp_path/'base',pretrained=tmp_path/'K0' if route in {'K1','K3','K4','K5'} else None)
    for route in ('K1','K2','K3','K4','K5'):
        with np.load(tmp_path/'K0/ridge_71.npz') as a,np.load(tmp_path/route/'ridge_71.npz') as z:
            np.testing.assert_array_equal(a['mask'],z['mask']);np.testing.assert_array_equal(a['prediction'],z['prediction'])
    for route in ('K2','K3','K5'):
        assert read_json(tmp_path/route/'run.json')['semantic_provenance']['weight_kind']=='domain_adapter'
    assert read_json(tmp_path/'K4/run.json')['semantic_provenance']['weight_kind']=='frozen_base'

def test_dag_controls_do_not_depend_on_gpu_and_pretraining_is_explicit():
    sys.path.insert(0,str(ROOT/'scripts'))
    from run_pk1 import tasks
    c=read_json(ROOT/'configs/pk1.json');plan=tasks(c)
    assert len(plan)==79
    assert 'semantics_domain' not in plan['state_core_celltypes_K0_42'][0]
    assert 'semantics_base' not in plan['state_core_celltypes_K1_42'][0]
    assert 'pretrained_42' in plan['state_core_celltypes_K3_42'][0]
    assert 'semantics_domain' in plan['state_core_celltypes_K3_42'][0]
    assert 'semantics_base' in plan['state_core_celltypes_K4_42'][0]

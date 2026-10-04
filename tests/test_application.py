import copy,json,zipfile,sys
from pathlib import Path
import numpy as np
import pytest
from vdc.io import read_json,read_jsonl,write_json,save_npz,sha256,object_hash
from vdc.application_data import align_counts,non_target_input,projection
from vdc.application import predict_ridge,verify_freeze,inventory,zero_target_bundle
from vdc.application_knowledge import clean_cases,source_independence
from vdc.application_stress import fit_reference,apply_stress
from vdc.pk2_numeric import dual_ridge,query
from vdc.state import fit_state,StateConfig,StatePredictor
from .fixtures import observation_fixture
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))


def test_hidden_genes_cannot_change_any_rank_or_input_denominator():
    genes=['a','b','c','d','hidden'];x=np.array([[1,3,7,8,2],[8,2,3,1,9]])
    defs=[{'id':'overlap','members':dict.fromkeys(genes,1),'source_ref':'software fixture'}]
    y,s=non_target_input(x,genes,defs,{'hidden'});x[:,-1]=10000000
    yy,ss=non_target_input(x,genes,defs,{'hidden'})
    np.testing.assert_array_equal(y,yy)
    for k in ('values','mask','coverage'):np.testing.assert_array_equal(s[k],ss[k])
    assert s['mask'].all() and np.all(s['coverage']==.8)


def test_count_alignment_rejects_wrong_scale_or_gene_universe():
    np.testing.assert_equal(align_counts([[1,2]],['b','a'],['a','b']),[[2,1]])
    for x,g,expect in [([[1.5,2]],['a','b'],['a','b']),([[1,2]],['a','a'],['a','b']),([[1,2]],['a','b'],['a','c'])]:
        with pytest.raises(ValueError):align_counts(x,g,expect)


def test_saved_partial_ridge_preserves_missing_targets(tmp_path):
    x=np.arange(8).reshape(4,2);y=np.arange(12).reshape(4,3);m=np.ones_like(y,bool);m[0,0]=False;m[:,2]=False
    rows=[{'study_family':'synthetic','biological_unit':str(i)} for i in range(4)]
    fn=dual_ridge(x,y,rows,target_mask=m);save_npz(tmp_path/'ridge.npz',**fn.parameters)
    np.testing.assert_equal(predict_ridge(tmp_path/'ridge.npz',x,3),fn(x))


def test_clean_withholding_removes_species_source_and_duplicate_prompts():
    records=read_jsonl(ROOT/'knowledge/pk2/literature.jsonl')
    cases=clean_cases(records+records)
    prompts=[]
    for r,mode in cases:
        if mode=='addressed_withheld':
            user=r['prompt'][-1]['content'];assert user.startswith('Source: not_supplied\nContext: not_supplied\nEvidence: None supplied.')
            assert r['source_ref'] not in user
            expected=json.loads(r['completion']);assert expected=={'answer':'unknown','context':'not_supplied','source':'not_supplied','uncertain':True}
        prompts.append(object_hash([mode,r['prompt']]))
    assert len(prompts)==len(set(prompts))
    assert len(cases)<3*len([r for r in records+records if r['split']=='validation'])


def test_new_paper_families_are_evaluation_only():
    new=read_jsonl(ROOT/'knowledge/application/evaluation_only.jsonl');old=read_jsonl(ROOT/'knowledge/pk2/literature.jsonl')
    assert source_independence(old,new)['new_source_count']==3
    assert len(new)==9 and {r['split'] for r in new}=={'validation'}
    with pytest.raises(ValueError):source_independence(new,new)


def test_frozen_inventory_detects_changes_before_query(tmp_path):
    with pytest.raises(FileNotFoundError):verify_freeze(tmp_path)
    write_json(tmp_path/'snapshot/model.json',{'value':1})
    m={'files':inventory(tmp_path/'snapshot')};m['freeze_id']=object_hash(m);write_json(tmp_path/'freeze.json',m)
    verify_freeze(tmp_path)
    write_json(tmp_path/'snapshot/model.json',{'value':2})
    with pytest.raises(ValueError):verify_freeze(tmp_path)


def test_projection_keeps_role_and_scope_audit_with_real_state_forward(tmp_path):
    b=observation_fixture(p=3);b=b.subset(np.r_[b.indices('train'),b.indices('validation')]);b.save(tmp_path/'reference/bundle')
    fit_state(b,tmp_path/'model',2,StateConfig(hidden_dim=8,latent_dim=4,layers=1,heads=2,validation_every=1))
    genes=['a','b','c','d'];defs=[{'id':p,'members':dict.fromkeys(genes,1),'source_ref':'synthetic software fixture'} for p in b.feature_ids]
    write_json(tmp_path/'prepared/expression_contract.json',{'gene_ids':genes});write_json(tmp_path/'prepared/programmes.json',defs)
    row={'sample_key':'fixture:query','biological_unit':'fixture:q','link_ids':['fixture:q'],'source':'synthetic_material',
         'condition':'fixture','role':'reserved_evaluation','split':'locked_test','allowed_tasks':['locate']}
    c=read_json(ROOT/'configs/application_v1.json');policy={'approval':{'content_hash':'software_fixture'}}
    q,y,x,g,d,scopes=projection([[1,2,3,4]],genes,[row],policy,tmp_path/'prepared',tmp_path/'reference',c)
    assert q.rows[0]['application_role']=='reserved_evaluation' and scopes[0]['calibration_claim'] is False
    p=StatePredictor.load(tmp_path/'model');answer=query(p,q)
    assert answer['programme'].shape==(1,3)
    q.context['assay']='changed'
    with pytest.raises(ValueError):query(p,q)
    row['role']='excluded'
    with pytest.raises(ValueError):projection([[1,2,3,4]],genes,[row],policy,tmp_path/'prepared',tmp_path/'reference',c)


def test_normalised_stress_uses_only_train_reference_and_no_clock(tmp_path):
    train={'sample_key':'train','source':'core','sample_id':'train','condition':'Early Diapause','split':'train','role':'development'}
    held={'sample_key':'stress','source':'sc_reserved','sample_id':'stress','condition':'Stress Early Diapause','split':'locked_test','role':'reserved_evaluation'}
    policy={'samples':[train,held]};config=read_json(ROOT/'configs/application_v1.json');sup=tmp_path/'supplement';run=tmp_path/'run'
    save_npz(sup/'stress_train.npz',values=np.array([[.1,.2]]),mask=np.array([[True,False]]),feature_ids=np.array(['a','b']))
    save_npz(sup/'stress_queries.npz',values=np.array([[.4,.5]]),mask=np.ones((1,2),bool),coverage=np.ones((1,2)),feature_ids=np.array(['a','b']))
    s={'rows':{'stress_train':[{**train,'cell_count':20}],'stress_queries':[{**held,'cell_count':20}]},
       'files':inventory(sup),'scale':'archived_normalised_expression_not_counts','aggregation':'mean normalised raw.X then rank','claim':'descriptive only'}
    write_json(sup/'manifest.json',{'stress_substitute':s});write_json(run/'snapshot/sample_roles.json',policy)
    fit_reference(sup,run/'snapshot',policy,config);before=sha256(run/'snapshot/stress/reference.npz')
    save_npz(sup/'stress_queries.npz',values=np.array([[.9,.5]]),mask=np.ones((1,2),bool),coverage=np.ones((1,2)),feature_ids=np.array(['a','b']))
    s['files']['stress_queries.npz']=sha256(sup/'stress_queries.npz');write_json(sup/'manifest.json',{'stress_substitute':s})
    fit_reference(sup,run/'snapshot',policy,config);assert before==sha256(run/'snapshot/stress/reference.npz')
    freeze={'files':inventory(run/'snapshot'),'config':config,'supplement_identity':{'manifest_sha256':sha256(sup/'manifest.json')}}
    freeze['freeze_id']=object_hash(freeze);write_json(run/'freeze.json',freeze)
    rows=apply_stress(run,sup);assert rows[0]['clock'] is None and rows[0]['depth'] is None and not rows[0]['counts_model_used']
    with np.load(run/'applications/stress'/rows[0]['result_folder']/'descriptive.npz') as a:
        assert a['difference'][0]==pytest.approx(.8) and np.isnan(a['difference'][1])
    s['rows']['stress_train'][0]['split']='validation';write_json(sup/'manifest.json',{'stress_substitute':s})
    with pytest.raises(ValueError,match='train rows'):fit_reference(sup,run/'snapshot',policy,config)


def test_pack_excludes_source_counts_and_weights(tmp_path):
    from pack_application_report import pack
    run=tmp_path/'run';write_json(run/'applications/bulk/result.json',{'rows':[]})
    save_npz(run/'snapshot/bulk/prepared/gene_expression.npz',counts=np.ones((1,2)))
    save_npz(run/'snapshot/bulk/hidden/readout.npz',coefficients=np.ones((1,2)))
    (run/'best.pt').write_bytes(b'weights')
    z=pack(run,tmp_path/'private.zip')
    with zipfile.ZipFile(z) as f:
        assert 'applications/bulk/result.json' in f.namelist()
        assert 'snapshot/bulk/hidden/readout.npz' in f.namelist()
        assert not any(n.endswith('gene_expression.npz') or n.endswith('.pt') for n in f.namelist())


def test_saved_result_browser_private_read_only(tmp_path):
    from serve_application import create_app
    from fastapi.testclient import TestClient
    group=tmp_path/'applications/bulk';identity='a'*20;folder=group/identity
    row={'sample_key':'fixture','status':'queried','result_folder':identity,'freeze_id':'fixture-freeze'}
    write_json(folder/'result.json',row);write_json(group/'result.json',{'rows':[row]});save_npz(folder/'observations.npz',observed=np.array([[.2]]))
    write_json(tmp_path/'application_snapshot.json',{'files':{p.relative_to(tmp_path).as_posix():sha256(p) for p in group.rglob('*') if p.is_file()},'freeze_id':'fixture-freeze'})
    client=TestClient(create_app(tmp_path));assert client.get('/health').json()['status']=='ready'
    assert client.get('/sample/bulk/'+identity).json()['arrays']['observations']['observed']==[[.2]]
    assert client.post('/sample/bulk/'+identity).status_code==405
    assert client.get('/sample/invalid/'+identity).status_code==404
    save_npz(folder/'observations.npz',observed=np.array([[.3]]))
    assert client.get('/sample/bulk/'+identity).status_code==409

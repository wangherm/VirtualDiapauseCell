"""Software-only fixtures; no mocked generation counts as full GPU acceptance."""
import copy,json,shutil
from pathlib import Path
import numpy as np
import pytest
from vdc.io import read_json,write_json,sha256,object_hash
from vdc.integration_bundle import export_model,SPECIES,CONTEXT
from vdc.integration import VirtualDiapauseCell,read_request
from vdc.integration_knowledge import parse_explanation
from tests.test_clock_wave_revision import setup_data
from vdc.clock_wave_revision import residual_task,support_task
from vdc.clock_wave_tasks import gene_waves_task
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def bundle(tmp_path):
    stage=tmp_path/'stage';x,g,d,r,c=setup_data(stage)
    residual_task(stage,'core',stage/'tasks/residual_core',c);support_task(stage,'core',stage/'tasks/support_core',c)
    gene_waves_task(stage,'core',stage/'tasks/gene_waves_core',c,ROOT)
    for view in ('bulk','coarse'):
        for prefix,suffix in [('numeric','_C2'),('residual',''),('support',''),('gene_waves','')]:
            shutil.copytree(stage/'tasks'/(prefix+'_core'+suffix),stage/'tasks'/(prefix+'_'+view+suffix))
    write_json(stage/'tasks/identity_original/reference.json',{'reference':None,'calibration':{},'genes':g})
    snap={'status':'software_fixture','files':{p.relative_to(stage).as_posix():sha256(p) for p in (stage/'tasks').rglob('*') if p.is_file()}}
    snap['snapshot_id']=object_hash(snap);write_json(stage/'stage_snapshot.json',snap)
    out=tmp_path/'bundle';export_model(stage,out)
    request={'request_id':'never-in-training','view':'core','species':SPECIES,'context':CONTEXT,'scale':'counts','genes':g,'counts':x[9:11].tolist(),
        'samples':[{'sample_id':'new-'+str(i),'biological_unit':'new-unit-'+str(i),'identity':'all','identity_source':'provided_material'} for i in range(2)]}
    return out,request

def test_portable_fresh_inference_no_fitting_or_historical_prediction(bundle,tmp_path,monkeypatch):
    model,r=bundle;moved=tmp_path/'elsewhere/model';shutil.copytree(model,moved)
    from vdc.clock_wave import ClockWave
    monkeypatch.setattr(ClockWave,'fit',lambda *a,**k:pytest.fail('Inference attempted fit'))
    app=VirtualDiapauseCell(moved);a=app.analyse(r,'numeric_only','cw_stage_baseline',tmp_path/'result')
    assert a['status']=='completed' and not a['provenance']['fit_during_request']
    assert (tmp_path/'result/report.html').is_file() and (tmp_path/'result/tables/waves.csv').is_file()
    for i in range(2):
        q=copy.deepcopy(r);q['samples']=[q['samples'][i]];q['counts']=[q['counts'][i]]
        b=app.analyse(q,'numeric_only','cw_stage_baseline')
        np.testing.assert_allclose(b['numeric']['gene_prediction'][0],a['numeric']['gene_prediction'][i],atol=1e-10)
    q=copy.deepcopy(r);q['question']='a different question'
    assert app.analyse(q,'numeric_only','cw_stage_baseline')['numeric']==a['numeric']
    q['counts'][0][5]*=50
    assert app.analyse(q,'numeric_only','cw_stage_baseline')['numeric']!=a['numeric']
    from vdc.integration_service import create_app
    from fastapi.testclient import TestClient
    with TestClient(create_app(moved,tmp_path/'http')) as client:
        resp=client.post('/analyse',json={'request':r,'mode':'numeric_only','variant':'cw_stage_baseline'})
        assert resp.status_code==200;data=resp.json();assert data['result']['numeric']==a['numeric']
        assert client.get(data['report_url']).status_code==200 and client.get(data['download_url']).status_code==200
        assert client.post('/analyse',json={'request':{'input':{'path':'/secret'}}}).status_code==422

def test_target_isolation_identity_scope_and_failure_status(bundle):
    model,r=bundle;app=VirtualDiapauseCell(model);a=app.analyse(r,'numeric_only','cw_stage_baseline')
    from vdc.clock_wave import hidden_partition
    q=copy.deepcopy(r)
    for i,g in enumerate(q['genes']):
        if g in hidden_partition(q['genes']):q['counts'][0][i]=123456
    b=app.analyse(q,'numeric_only','cw_stage_baseline')
    assert a['numeric']['clock']==b['numeric']['clock'] and a['numeric']['gene_prediction']==b['numeric']['gene_prediction']
    assert a['provenance']['field_registry']==b['provenance']['field_registry']
    q=copy.deepcopy(r);q['samples'][0]['identity']='unknown';b=app.analyse(q,'numeric_only','cw_stage_baseline')
    assert b['numeric']['status'][0]=='unsupported_identity' and b['numeric']['clock'][0] is None
    q['species']='Homo sapiens';assert app.analyse(q,'numeric_only','cw_stage_baseline')['components']['numeric']['status']=='unsupported'
    assert app.analyse(r)['status']=='partial'
    for wrong in ('logCPM','scaled','FPKM'):
        q=copy.deepcopy(r);q['scale']=wrong
        with pytest.raises(ValueError):app.analyse(q,'numeric_only','cw_stage_baseline')
    q=copy.deepcopy(r);q['counts'][0][0]=1.001
    with pytest.raises(ValueError):app.analyse(q,'numeric_only','cw_stage_baseline')

def test_csv_h5ad_explicit_layer_and_id_order(bundle,tmp_path):
    import anndata,csv
    model,r=bundle;ids=[s['sample_id'] for s in r['samples']]
    with (tmp_path/'samples.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(r['samples'][0]));w.writeheader();w.writerows(r['samples'][::-1])
    with (tmp_path/'counts.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['sample_id']+r['genes']);w.writerows([[s]+v for s,v in zip(ids,r['counts'])])
    q={k:v for k,v in r.items() if k not in {'genes','counts','samples'}};q['input']={'format':'csv','matrix':'counts.csv','samples':'samples.csv'}
    write_json(tmp_path/'r.json',q);loaded=read_request(tmp_path/'r.json');assert loaded['samples']==r['samples']
    a=anndata.AnnData(np.zeros_like(r['counts'],dtype=float));a.obs_names=ids;a.var_names=r['genes'];a.layers['raw_counts']=np.array(r['counts']);a.write_h5ad(tmp_path/'a.h5ad')
    q['input']={'format':'h5ad','path':'a.h5ad','samples':'samples.csv'};write_json(tmp_path/'r.json',q)
    with pytest.raises(ValueError,match='counts_layer'):read_request(tmp_path/'r.json')
    q['input']['counts_layer']='raw_counts';write_json(tmp_path/'r.json',q);assert read_request(tmp_path/'r.json')['counts']==r['counts']

def test_explanation_references_and_numbers_are_strict():
    fields={'s0.clock':{'value':.2}};ev=[{'record_id':'source1'}]
    c={'kind':'reference','field_ids':['s0.clock'],'evidence_ids':[],'text':'Position on the supplied reference.'};raw=lambda c:json.dumps({'claims':[c]})
    assert parse_explanation(raw(c),fields,ev)['status']=='generated_reference_checks_passed'
    assert parse_explanation(raw({**c,'evidence_ids':['fabricated']}),fields,ev)['status']=='evidence_failure'
    assert parse_explanation(raw({**c,'text':'The clock is 0.2'}),fields,ev)['status']=='numeric_citation_failure'
    assert parse_explanation(raw({**c,'kind':'hypothesis'}),fields,ev)['status']=='evidence_failure'
    assert parse_explanation('{"claims":',fields,ev)['status']=='parse_failure'

def test_semantic_graph_is_consumed_and_target_selection_is_unchanged(tmp_path):
    from tests.test_clock_wave import fixture
    from vdc.clock_wave import ClockWave
    from vdc.integration_build import graph_from
    x,g,d,r,c=fixture();vectors=np.arange(len(d)*8,dtype=float).reshape(len(d),8)+1
    graph,meta=graph_from(vectors,d,g,'domain',c['semantic_shuffle_seed']);base=ClockWave.fit(x,g,d,r,'C2',c);model=ClockWave.fit(x,g,d,r,'C2',c,graph)
    assert np.max(abs(base.arrays['coef']-model.arrays['coef']))>0 and base.meta['target_genes']==model.meta['target_genes']
    shuffled,sm=graph_from(vectors,d,g,'shuffle',c['semantic_shuffle_seed']);assert sm['permutation']!=list(range(len(d)))
    changed=x.copy();changed[8:]*=20;other=ClockWave.fit(changed,g,d,r,'C2',c,graph)
    for k in model.arrays:np.testing.assert_array_equal(model.arrays[k],other.arrays[k])


def test_scoped_public_export_and_new_response_computations(bundle,tmp_path):
    from tests.fixtures import response_fixture
    from vdc.response import ResponseRegressor
    from vdc.alpha_numeric import ridge_fit
    from vdc.pk1_baselines import bounded_functional
    from vdc.integration_build import export_public
    from vdc.integration_bundle import seal
    from vdc.io import save_npz
    pk1=tmp_path/'pk1';modules={};datasets={}
    for name in ('endpoint','transition','functional'):
        out=pk1/name
        if name=='functional':bounded_functional(ROOT/'data/curated/functional_fig1b.json',out)
        else:
            d=response_fixture(name);datasets[name]=d;d.save(out/'dataset');ResponseRegressor().fit(d).save(out)
            if name=='transition':
                tr=np.arange(18);coef,mu,scale=ridge_fit(np.column_stack([d.elapsed[tr],d.action[tr]]),d.target[tr])
                save_npz(out/'time_baseline.npz',coefficients=coef,mean=mu,scale=scale,prediction=np.zeros((1,4)))
        modules[name]={'execution_status':'completed','artifact':name,'files':{p.relative_to(out).as_posix():sha256(p) for p in out.rglob('*') if p.is_file()}}
    write_json(pk1/'module_status.json',{'modules':modules})
    model,_=bundle;meta=read_json(model/'bundle.json');meta['public_models']=export_public(pk1,model,ROOT)
    for k in ('bundle_id','implementation','files','format'):meta.pop(k)
    seal(model,meta);app=VirtualDiapauseCell(model)
    for name,d in datasets.items():
        req={'request_id':'new-response','responses':[{'module':name,'species':'Caenorhabditis elegans','scope':d.scope,'current':d.current[:1].tolist(),'action':d.action[:1].tolist()}]}
        if name=='transition':req['responses'][0]['elapsed']=d.elapsed[:1].tolist()
        got=app.analyse(req,'numeric_only')['responses'][0]
        expected=ResponseRegressor.load(pk1/name).predict(d.current[:1],d.action[:1],d.scope,None if d.elapsed is None else d.elapsed[:1])
        np.testing.assert_allclose(got['prediction'],expected);assert got['baseline']['status']=='computed'
        req['responses'][0]['action']=[]
        with pytest.raises(ValueError,match='finite response'):app.analyse(req,'numeric_only')
    q={'request_id':'new-functional','responses':[{'module':'functional','species':'Caenorhabditis elegans','scope':meta['public_models']['functional']['scope'],'hours_after_release':12,'history_days':4}]}
    with np.load(model/'public/functional/model.npz') as a:q['responses'][0]['hours_after_release']=float(a['hour_levels'][0])
    got=app.analyse(q,'numeric_only')['responses'][0];assert 0<=got['prediction']<=1 and 'baseline' in got
    q['responses'][0]['species']=SPECIES;assert app.analyse(q,'numeric_only')['responses'][0]['status']=='unsupported'


def test_regulon_missing_coverage_is_null_and_upload_executes(bundle,tmp_path):
    from vdc.integration_bundle import seal
    from vdc.integration_service import create_app
    from fastapi.testclient import TestClient
    import csv,io
    model,r=bundle;meta=read_json(model/'bundle.json')
    write_json(model/'regulons/public/members.json',[{'id':'example','members':{'absent1':1,'absent2':1,'absent3':1},'source_ref':'software-fixture'}])
    meta['regulons']={'public':{'path':'regulons/public/members.json','scope':'software-fixture'}}
    for k in ('bundle_id','implementation','files','format'):meta.pop(k)
    seal(model,meta);app=VirtualDiapauseCell(model)
    got=app.analyse({'request_id':'new-regulon','responses':[{'module':'regulon','species':'Caenorhabditis elegans','scale':'estimated_counts','genes':['g1','g2','g3'],'expression':[[1.5,2,3]]}]},'numeric_only')
    assert got['responses'][0]['prediction']==[[None]] and got['responses'][0]['mask']==[[False]]
    matrix=io.StringIO();w=csv.writer(matrix);w.writerow(['sample_id']+r['genes']);w.writerows([[s['sample_id']]+row for s,row in zip(r['samples'],r['counts'])])
    samples=io.StringIO();w=csv.DictWriter(samples,fieldnames=list(r['samples'][0]));w.writeheader();w.writerows(r['samples'])
    req={k:v for k,v in r.items() if k not in {'genes','counts','samples'}}
    with TestClient(create_app(model,tmp_path/'http')) as client:
        res=client.post('/upload',data={'format':'csv','metadata':json.dumps({'request':req,'mode':'numeric_only','variant':'cw_stage_baseline'})},files={'matrix':('x.csv',matrix.getvalue()),'samples':('s.csv',samples.getvalue())})
        assert res.status_code==200 and res.json()['result']['components']['numeric']['new_input_computed']

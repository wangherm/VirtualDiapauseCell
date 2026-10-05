"""Leakage, independence and persistence tests; fixtures are software-only."""
import copy,sys
from pathlib import Path
import numpy as np
import pytest
from vdc.clock_wave import inputs,hidden_partition,ClockWave,NotIdentifiable,evaluate
from vdc.io import read_json,write_json,sha256
from vdc.clock_wave_data import coarsen
from vdc import identity
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))


def fixture():
    rng=np.random.default_rng(52);genes=[f'gene{i}' for i in range(140)]
    defs=[{'id':f'p{k}','members':dict.fromkeys(genes[k*16:(k+1)*16],1.),'source_ref':'synthetic software fixture'} for k in range(8)]
    rows=[];values=[]
    for i in range(16):
        t=(i%8)/7
        values.append(rng.poisson(np.exp(3+np.linspace(-2,2,len(genes))*t)))
        rows.append({'observation_id':f'fixture{i}','biological_unit':f'u{i}','source_sample_key':f's{i}',
                     'study_family':'software','split':'train' if i<8 else 'validation','origin':'synthetic',
                     'link_ids':[f's{i}'],'condition':'Early Diapause' if i%8==0 else 'Developing' if i%8==7 else 'Exit',
                     'reference_eligible':True})
    return np.array(values),genes,defs,rows,read_json(ROOT/'configs/clock_wave.json')


def test_all_input_paths_exclude_hidden_targets():
    x,g,d,r,c=fixture();a=inputs(x,g,d);changed=x.copy()
    changed[:,[i for i,s in enumerate(g) if s in hidden_partition(g)]]=99999999
    b=inputs(changed,g,d)
    np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[2],b[2])
    for key in ('values','mask','coverage'):np.testing.assert_array_equal(a[1][key],b[1][key])
    model=ClockWave.fit(x,g,d,r,'C2',c)
    np.testing.assert_allclose(model.predict(x,['all']*len(x))['clock'],model.predict(changed,['all']*len(x))['clock'],equal_nan=True)


def test_outer_targets_cannot_change_model_or_selected_panel(tmp_path):
    x,g,d,r,c=fixture();first=ClockWave.fit(x,g,d,r,'C2',c)
    changed=x.copy();changed[8:]=changed[8:]*99+7
    second=ClockWave.fit(changed,g,d,r,'C2',c)
    assert first.meta['target_indices']==second.meta['target_indices']
    for key in first.arrays:np.testing.assert_array_equal(first.arrays[key],second.arrays[key])
    for key in first.readout:np.testing.assert_array_equal(first.readout[key],second.readout[key])
    first.save(tmp_path/'m');loaded=ClockWave.load(tmp_path/'m')
    np.testing.assert_array_equal(first.hidden_predictions(x,['all']*len(x))['hidden_prediction'],loaded.hidden_predictions(x,['all']*len(x))['hidden_prediction'])
    (tmp_path/'m/arrays.npz').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='changed'):ClockWave.load(tmp_path/'m')


def test_centred_rank_is_affine_control_not_new_information():
    x,g,d,r,c=fixture();a=ClockWave.fit(x,g,d,r,'C0',c);b=ClockWave.fit(x,g,d,r,'C1',c)
    # Ridge centring makes the direct numerical control invariant to the offset.
    aa=a.hidden_predictions(x,['all']*len(x));bb=b.hidden_predictions(x,['all']*len(x))
    np.testing.assert_allclose(aa['hidden_direct_ridge'],bb['hidden_direct_ridge'],atol=1e-8)


def test_missing_anchor_and_query_support_are_explicit():
    x,g,d,r,c=fixture();bad=copy.deepcopy(r)
    for row in bad:
        if row['split']=='train' and row['condition']=='Developing':row['condition']='Exit'
    with pytest.raises(NotIdentifiable):ClockWave.fit(x,g,d,bad,'C0',c)
    model=ClockWave.fit(x,g,d,r,'C2',c)
    assert set(model.predict(x,['unknown']*len(x))['status'])=={'unsupported_identity'}
    q=model.predict(x,['all']*len(x),np.zeros((len(x),len(d)),bool))
    assert np.isnan(q['clock']).all() and not q['residual_mask'].any()


def test_family_targets_removed_from_every_feature():
    x,g,d,r,c=fixture();c={**c,'extra_hidden':list(d[0]['members']),'target_panel':list(d[0]['members'])}
    model=ClockWave.fit(x,g,d,r,'C2',c);changed=x.copy();changed[:,:16]=77777
    np.testing.assert_array_equal(model._features(x)[1],model._features(changed)[1])
    assert not model._features(x)[2][:,0].any()


def test_coarse_profiles_sum_counts_not_scores():
    rows=[{'observation_id':t,'source_sample_key':'one','split':'train','cell_type':t,'cell_count':20} for t in ['a','b']]
    x,r,omitted=coarsen(np.array([[1,10],[10,1]]),rows,{'groups':{'coarse':['a','b']}})
    np.testing.assert_array_equal(x,[[11,11]]);assert r[0]['cell_count']==40 and not omitted


def test_cards_hide_query_label_time_and_clock_and_require_valid_evidence():
    x,g,d,rows,c=fixture();y,_,_=inputs(x,g,d)
    rr=[{**r,'split':'train','coarse_identity':'first' if i%2 else 'second','cell_type':'SECRET_QUERY_LABEL','clock':777,'elapsed_hours':888} for i,r in enumerate(rows)]
    labels=[r['coarse_identity'] for r in rr];ref=identity.fit_reference(y,g,labels,rr,min_units=2)
    cal=identity.calibrate(y,g,labels,rr,c);pred=identity.predict(ref,cal,y)
    cards=identity.evidence_cards(ref,pred,y,g,rr,{})
    import json
    assert 'SECRET_QUERY_LABEL' not in json.dumps(cards) and 'elapsed_hours' not in json.dumps(cards)
    for card in cards:assert not (hidden_partition(g)&{r['gene'] for r in card['observed_markers']})
    assert identity.parse_answer('{"identity":"invented","evidence_genes":["madeup"]}',cards[0])['prediction']=='unknown'
    assert not identity.parse_answer('not json',cards[0])['format_valid']
    assert 'unknown' in [r['prediction'] for r in identity.predict(None,{},y)]


def test_queue_separates_gpu_and_no_optional_gradient_tasks():
    from run_clock_wave import build_plan
    jobs=build_plan(9);assert len([j for j in jobs if j['kind']=='numeric' and j.get('fold') is not None])==18
    assert {j['mode'] for j in jobs if j['resource']=='gpu'}=={'base','domain'}
    assert all('train' not in j['kind'] for j in jobs)


def test_semantic_graph_really_changes_model_and_retains_input_isolation():
    x,g,d,r,c=fixture();base=ClockWave.fit(x,g,d,r,'C2',c)
    graph=np.ones((len(d),len(d)))/len(d);np.fill_diagonal(graph,0)
    sem=ClockWave.fit(x,g,d,r,'C2',c,graph)
    assert np.max(abs(base.arrays['coef']-sem.arrays['coef']))>1e-6
    assert base.meta['target_indices']==sem.meta['target_indices']
    np.testing.assert_array_equal(base._features(x)[1],sem._features(x)[1])


def test_fold_closes_cohort_and_never_reassigns_reserved(tmp_path,monkeypatch):
    from vdc.clock_wave_data import split_fold
    from vdc.admission import approve_roles
    from vdc.io import object_hash
    samples=[];rows=[]
    for i in range(4):
        samples.append({'sample_key':f's{i}','biological_unit':f'u{i}','link_ids':[f's{i}'],
            'cohort_id':'shared' if i<2 else None,'role':'development' if i<3 else 'reserved_evaluation',
            'split':'train' if i<3 else 'locked_test','allowed_tasks':['state','clock','waves']})
    path=tmp_path/'roles.json';write_json(path,{'protocol':'VDC_PK1','status':'draft','blocking_questions':[],'samples':samples})
    h=approve_roles(path,'software fixture reviewer');policy=read_json(path);before=sha256(path)
    monkeypatch.setenv('VDC_ROLE_MANIFEST',str(path))
    for s in samples[:3]:rows.append({'observation_id':s['sample_key'],'source_sample_key':s['sample_key'],
        'biological_unit':s['biological_unit'],'link_ids':s['link_ids'],'study_family':'fixture','origin':'internal',
        'split':'train','admission_protocol':'VDC_PK1','admission_hash':h})
    result=split_fold(rows,policy,'u0',tmp_path/'fold.json')
    assert [r['split'] for r in result]==['validation','validation','train']
    assert 's3' not in read_json(tmp_path/'fold.json')['assignments'] and sha256(path)==before
    monkeypatch.delenv('VDC_DEVELOPMENT_FOLD',raising=False)


def test_pack_omits_counts_and_service_checks_identity(tmp_path):
    from pack_clock_wave_report import pack
    from serve_clock_wave import create_app
    from fastapi.testclient import TestClient
    from vdc.io import object_hash,save_npz
    import zipfile
    run=tmp_path/'run';write_json(run/'tasks/numeric_a/result.json',{'status':'evaluated'})
    write_json(run/'queue_status.json',{'tasks':{'numeric_a':{'status':'completed'}}})
    save_npz(run/'data/counts.npz',counts=np.ones((2,2)))
    snap={'files':{'tasks/numeric_a/result.json':sha256(run/'tasks/numeric_a/result.json')}}
    snap['snapshot_id']=object_hash(snap);write_json(run/'results_snapshot.json',snap)
    client=TestClient(create_app(run));assert client.get('/health').json()['live_inference'] is False
    assert client.get('/task/numeric_a').status_code==200
    assert client.get('/task/missing').status_code==404
    package=pack(run)
    with zipfile.ZipFile(package) as z:assert 'data/counts.npz' not in z.namelist()
    write_json(run/'tasks/numeric_a/result.json',{'changed':True})
    with pytest.raises(ValueError):create_app(run)

"""Actual development requests, independent CLI/HTTP and restart qualification; no new fit."""
from pathlib import Path
import argparse,copy,json,os,socket,subprocess,sys,time,urllib.request
import numpy as np
from vdc.io import read_json,write_json
from vdc.integration import VirtualDiapauseCell
from vdc.integration_bundle import SPECIES,CONTEXT
from vdc.clock_wave_data import load_data
from vdc.clock_wave import identities,hidden_partition


def verify(model,data,out,*,full=True,pk1=None,numeric_variant='cw_stage_baseline'):
    model,data,out=Path(model).resolve(),Path(data).resolve(),Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    app=VirtualDiapauseCell(model);variant='domain_clock_wave' if full else numeric_variant
    mode='full' if full else 'numeric_only';cases=[];requests=[]
    for view in ('bulk','core','coarse'):
        x,g,defs,rows=load_data(data/view);va=[i for i,r in enumerate(rows) if r['split']=='validation']
        supported=app.models[view,variant].hidden_predictions(x[va],identities([rows[i] for i in va]))
        finite=[va[i] for i,v in enumerate(supported['clock']) if np.isfinite(v)]
        if not finite:raise ValueError('No supported development request for sensitivity qualification: '+view)
        # One computable request plus the first other development row; unsupported identities stay unsupported.
        ix=[finite[0],next(i for i in va if i!=finite[0])]
        req={'request_id':'INT1-'+view+'-new-request','view':view,'species':SPECIES,'context':CONTEXT,'scale':'counts','genes':g,'counts':x[ix].tolist(),
             'question':'Describe the measured programme deviations and evidence limitations',
             'samples':[{'sample_id':f'INT1-{view}-{j}','biological_unit':rows[i]['biological_unit'],'identity':identities([rows[i]])[0],
                         'identity_source':'inherited_development_annotation'} for j,i in enumerate(ix)]}
        req['regulons']=True
        requests.append(req);write_json(out/'inputs'/(view+'.json'),req)
        report=app.analyse(req,mode,variant,out/(view+'_request'))
        cases.append({'case':view,'status':report['status'],'online_qwen':report['components']['qwen'],'source_role':'already_viewed_development'})
        numeric=app.analyse(req,'numeric_only',variant)
        assert report['numeric']==numeric['numeric']
        for j in range(len(ix)):
            one=copy.deepcopy(req);one['counts']=[one['counts'][j]];one['samples']=[one['samples'][j]]
            pred=app.analyse(one,'numeric_only',variant)['numeric']
            np.testing.assert_allclose(np.array(pred['gene_prediction'],float)[0],np.array(numeric['numeric']['gene_prediction'],float)[j],atol=1e-10,equal_nan=True)
        other=copy.deepcopy(req);other['question']='What remains unknown about this reference?'
        assert app.analyse(other,'numeric_only',variant)['numeric']==numeric['numeric']
        if full and view=='core':
            changed_question=app.analyse(other,'full',variant,out/'changed_question')
            assert changed_question['numeric']==report['numeric']
            cases.append({'case':'new_question_same_measurements','status':changed_question['status'],'online_qwen':changed_question['components']['qwen']})
        changed=copy.deepcopy(req)
        hidden=set(hidden_partition(g))
        for k,gene in enumerate(g):
            if gene in hidden:changed['counts'][0][k]+=100000
        blind=app.analyse(changed,'numeric_only',variant)
        assert blind['numeric']['clock']==numeric['numeric']['clock'] and blind['numeric']['gene_prediction']==numeric['numeric']['gene_prediction']
        assert blind['provenance']['field_registry']==numeric['provenance']['field_registry']
        changed=copy.deepcopy(req);visible=[k for k,gene in enumerate(g) if gene not in hidden]
        for k in visible[::3]:changed['counts'][0][k]+=100000
        perturbed=app.analyse(changed,'numeric_only',variant)
        assert perturbed['numeric']['clock']!=numeric['numeric']['clock'] or perturbed['numeric']['gene_prediction']!=numeric['numeric']['gene_prediction']
        unknown=copy.deepcopy(req);unknown['samples'][0]['identity']='unknown'
        assert app.analyse(unknown,'numeric_only',variant)['numeric']['status'][0]=='unsupported_identity'
        command=[sys.executable,'-m','vdc','analyse','--model',str(model),'--request',str(out/'inputs'/(view+'.json')),'--mode','numeric_only','--variant',variant,'--output',str(out/(view+'_cli'))]
        subprocess.run(command,cwd=out,check=True)
        assert read_json(out/(view+'_cli')/'result.json')['numeric']==numeric['numeric']
    if pk1:
        from vdc.response import ResponseDataset
        from vdc.pk1_assets import alpha_module
        for name in ('endpoint','transition','functional'):
            e=app.manifest['public_models'][name];q={'module':name,'scope':e['scope'],'species':'Caenorhabditis elegans'}
            if name=='functional':
                with np.load(model/e['path']/'model.npz') as a:hour=float(a['hour_levels'][0])
                q.update(hours_after_release=hour,history_days=4.)
            else:
                p,_=alpha_module(pk1,name);d=ResponseDataset.load(p/'dataset');i=next(i for i,r in enumerate(d.rows) if r['split']=='validation')
                q.update(current=d.current[i:i+1].tolist(),action=d.action[i:i+1].tolist())
                if d.elapsed is not None:q['elapsed']=d.elapsed[i:i+1].tolist()
            r={'request_id':'INT1-new-public-'+name,'context':'Caenorhabditis elegans public assay','responses':[q]}
            p=app.analyse(r,mode,variant,out/('public_'+name));assert p['responses'][0]['status']=='computed'
            cases.append({'case':name,'status':p['status'],'online_qwen':p['components']['qwen']})
            wrong=copy.deepcopy(r);wrong['responses'][0]['species']=SPECIES
            assert app.analyse(wrong,'numeric_only',variant)['responses'][0]['status']=='unsupported'
        public_data=Path(pk1)/'public/data/dauer'
        if full or public_data.exists():
            from vdc.alpha_numeric import gene_data
            x,g=gene_data(public_data)
            req={'request_id':'INT1-new-public-regulon','responses':[{'module':'regulon','species':'Caenorhabditis elegans',
                 'scale':'log1p_library_10000','genes':g,'expression':x[:1].tolist()}]}
            got=app.analyse(req,mode,variant,out/'public_regulon');assert got['responses'][0]['status']=='computed'
            cases.append({'case':'public_regulon','status':got['status'],'online_qwen':got['components']['qwen']})
    # Release GPU before an independent HTTP process loads the same model.
    del app
    import gc;gc.collect()
    try:
        import torch
        if torch.cuda.is_available():torch.cuda.empty_cache()
    except ImportError:pass
    http=[]
    for attempt in range(2):
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        with (out/f'http_{attempt}.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,'-m','vdc','serve-app','--model',str(model),'--output',str(out/f'http_{attempt}'),'--port',str(port)],cwd=out,stdout=log,stderr=log)
            try:
                end=time.monotonic()+90
                while True:
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as f:health=json.load(f)
                        break
                    except OSError:
                        if proc.poll() is not None or time.monotonic()>end:raise RuntimeError('Independent app failed; see HTTP log')
                        time.sleep(.5)
                for j,r in enumerate(requests):
                    payload={'request':r,'mode':mode if attempt==0 and j==0 else 'numeric_only','variant':variant}
                    req=urllib.request.Request(f'http://127.0.0.1:{port}/analyse',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                    with urllib.request.urlopen(req,timeout=600) as f:got=json.load(f)
                    assert got['result']['numeric']==read_json(out/(r['view']+'_cli')/'result.json')['numeric']
                    http.append({'restart':attempt,'view':r['view'],'status':got['result']['status'],'mode':payload['mode'],'bundle_id':health['bundle_id']})
            finally:
                proc.terminate()
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
    passed=all(c['status']=='completed' for c in cases+http)
    actual_qwen=any(c.get('online_qwen',{}).get('actual_online_generation',False) for c in cases)
    result={'status':'full_verified' if full and passed else 'numeric_verified_full_not_run' if not full else 'partial',
            'full_requested':full,'full_gpu_executed':actual_qwen,'cases':cases,'http':http,'new_request_ids':True,'no_prediction_lookup':True,
            'single_batch_reload_cli_http':'passed','hidden_target_isolation':'passed','visible_input_sensitivity':'passed',
            'question_numeric_invariance':'passed','scientific_validation':'development_integration_only'}
    write_json(out/'acceptance.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--numeric-only',action='store_true');p.add_argument('--pk1');a=p.parse_args()
    result=verify(a.model,a.data,a.out,full=not a.numeric_only,pk1=a.pk1);print(json.dumps({'status':result['status']}))
    if result['status']=='partial':raise SystemExit(2)

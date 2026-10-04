"""Frozen PK1 development models served during PK2 acquisition and development."""
from pathlib import Path
from functools import lru_cache
import os,re,shutil
import importlib.metadata
import numpy as np
from .io import read_json,write_json,sha256,object_hash
from .pk1_assets import checked_member,verify_files
from .experimental import jsonable

def runtime_identity():
    return {name:importlib.metadata.version(name) for name in ['vdc-research','torch','numpy','scipy','fastapi']}

def implementation_identity():
    names=['pk2_service.py','state.py','contracts.py','admission.py','waves.py','response.py','io.py','experimental.py','pk1_assets.py']
    return {name:sha256(Path(__file__).with_name(name)) for name in names}

def freeze_service(run,roles,out,*,allow_synthetic_fixture=False):
    run=Path(run).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError('Choose a new immutable service snapshot')
    statuses=read_json(run/'module_status.json')['modules'];registry={};allfiles={}
    def copy_module(name,names,destination=None):
        m=statuses.get(name,{})
        if m.get('execution_status')!='completed':raise ValueError('Source task not completed: '+name)
        folder=checked_member(run,m.get('artifact',name));target=out/(destination or name)
        for rel in names:
            p=checked_member(folder,rel)
            if rel not in m['files'] or not p.is_file() or sha256(p)!=m['files'][rel]:raise ValueError('Source artifact missing or changed: '+name+'/'+rel)
            q=checked_member(target,rel);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
            allfiles[q.relative_to(out).as_posix()]=sha256(q)
        return target
    selected=[n for n,s in statuses.items() if re.fullmatch(r'state_(bulk|core|core_celltypes)_K[0-5]_\d+',n) and s['execution_status']=='completed']
    if not selected:raise ValueError('No completed PK1 state models')
    for name in selected:
        meta=read_json(checked_member(run,statuses[name].get('artifact',name))/'run.json')
        if meta['is_synthetic'] and not allow_synthetic_fixture:raise ValueError('Synthetic models cannot be deployed as research models')
    if not allow_synthetic_fixture:
        from .admission import role_manifest
        policy,_=role_manifest(roles)
        audit_folder=checked_member(run,statuses['audit'].get('artifact','audit'))
        verify_files(audit_folder,{'audit.json':statuses['audit']['files']['audit.json']})
        audit=read_json(audit_folder/'audit.json')
        if policy['approval']['content_hash']!=audit['role_hash']:raise ValueError('Source run and approved roles differ')
        os.environ['VDC_ROLE_MANIFEST']=str(Path(roles).resolve())
    out.mkdir(parents=True)
    if roles:
        shutil.copy2(roles,out/'sample_roles.json');allfiles['sample_roles.json']=sha256(out/'sample_roles.json')
    from .contracts import ObservationBundle
    views=sorted({n.removeprefix('state_').rsplit('_',2)[0] for n in selected})
    for view in views:
        folder=copy_module('clock_'+view,['bundle/manifest.json','bundle/observations.npz'])
        b=ObservationBundle.load(folder/'bundle')
        if any(r['split'] not in {'train','validation'} for r in b.rows):raise ValueError('Service accepts approved development only')
    for name in selected:
        folder=copy_module(name,['run.json','best.pt','result.json','metrics.json'])
        meta=read_json(folder/'run.json');view,route,seed=name.removeprefix('state_').rsplit('_',2)
        registry[name]={'view':view,'route':route,'seed':int(seed),'checkpoint_sha256':sha256(folder/'best.pt'),
            'training_steps':read_json(folder/'metrics.json')['steps_completed'],'scope':meta['scope'],
            'science_status':'unvalidated','wave':None}
        wave=f'waves_{view}_{route}'
        if statuses.get(wave,{}).get('execution_status')=='completed':
            source=checked_member(run,statuses[wave].get('artifact',wave));w=read_json(source/'result.json')
            if w['encoder_sha256']==registry[name]['checkpoint_sha256']:
                if not (out/wave).exists():copy_module(wave,['programme/reference.json','programme/reference.npz','result.json'])
                registry[name]['wave']=wave
    public={}
    for name in ['endpoint','transition']:
        if statuses.get(name,{}).get('execution_status')=='completed':
            copy_module(name,['response_model.json','response_model.npz','dataset/response.json','dataset/response.npz','result.json'])
            public[name]={'kind':'saved_response_model','scope':read_json(out/name/'response_model.json')['scope']}
    if statuses.get('functional',{}).get('execution_status')=='completed':
        copy_module('functional',['model.npz','result.json'])
        public['functional']={'kind':'group_level_binomial_readout','expression_depth':False}
    if statuses.get('public_regulon',{}).get('execution_status')=='completed':copy_module('public_regulon',['result.json'])
    profile='synthetic_software_test' if allow_synthetic_fixture else 'experimental'
    manifest={'profile':profile,'models':registry,'views':views,'public_models':public,'files':allfiles,
        'source_run_signature':read_json(run/'run_manifest.json')['signature'],
        'implementation':implementation_identity(),'runtime':runtime_identity(),
        'query_policy':'existing development observations only; no heldout or arbitrary file paths',
        'qwen_live_generation':'not_enabled; actual semantic caches are inside frozen numeric checkpoints',
        'new_PK2_training_executed':False,'scientific_approval':False}
    manifest['snapshot_id']=object_hash(manifest);write_json(out/'service_manifest.json',manifest)
    return manifest


def create_app(snapshot):
    from fastapi import FastAPI,HTTPException
    from fastapi.responses import HTMLResponse
    from .contracts import ObservationBundle
    from .state import StatePredictor
    from .waves import WaveReference
    from .response import ResponseDataset,ResponseRegressor
    from scipy.special import expit
    root=Path(snapshot).resolve();m=read_json(root/'service_manifest.json')
    if m['snapshot_id']!=object_hash({k:v for k,v in m.items() if k!='snapshot_id'}):raise ValueError('Service manifest changed')
    if implementation_identity()!=m['implementation'] or runtime_identity()!=m['runtime']:
        raise ValueError('Inference implementation changed; create a new reviewed service snapshot')
    verify_files(root,m['files'])
    if (root/'sample_roles.json').exists():os.environ['VDC_ROLE_MANIFEST']=str(root/'sample_roles.json')
    bundles={v:ObservationBundle.load(root/f'clock_{v}/bundle') for v in m['views']}
    app=FastAPI(title='VDC development service',docs_url=None,redoc_url=None)
    @lru_cache(maxsize=8)
    def model(name):return StatePredictor.load(root/name)
    def envelope(result):return jsonable({'mode':m['profile'],'snapshot_id':m['snapshot_id'],'science_status':'unvalidated','result':result})
    @app.get('/health')
    def health():return {'status':'ready','snapshot_id':m['snapshot_id'],'mode':m['profile']}
    @app.get('/analyses')
    def analyses():return envelope(m.get('analyses',{}))
    @app.get('/analysis')
    def analysis(module:str):
        entry=m.get('analyses',{}).get(module)
        if not entry:raise HTTPException(422,'Unknown analysis')
        report=read_json(root/entry['report']) if entry.get('report') else None
        return envelope({'identity':entry,'saved_evaluation':report,'new_prediction_executed':False})
    @app.get('/capabilities')
    def capabilities():return {k:m[k] for k in ['profile','snapshot_id','models','views','public_models','query_policy','qwen_live_generation']}
    @app.get('/samples')
    def samples(view:str):
        if view not in bundles:raise HTTPException(422,'Unknown view')
        return [{'observation_id':r['observation_id'],'split':r['split'],'condition':r.get('condition'),
                 'cell_type':r.get('cell_type')} for r in bundles[view].rows]
    @app.post('/analyse')
    def analyse(request:dict):
        try:
            name=request['model'];entry=m['models'][name];b=bundles[entry['view']]
            ids=request['observation_ids']
            if not isinstance(ids,list) or not 1<=len(ids)<=64 or len(ids)!=len(set(ids)):raise ValueError('Select 1–64 distinct admitted observation IDs')
            index={r['observation_id']:i for i,r in enumerate(b.rows)}
            if any(i not in index for i in ids):raise ValueError('Unknown or reserved observation is not queryable')
            q=b.subset([index[i] for i in ids]);q.clock[:]=0;q.clock_mask[:]=False
            p=model(name);pred=p.predict(q)
            result={'model':name,'checkpoint_sha256':entry['checkpoint_sha256'],'observation_ids':ids,
                'source_splits':[r['split'] for r in q.rows],'feature_ids':b.feature_ids,'observed_programme':q.values,
                'observed_mask':q.mask,'coverage':q.coverage,
                'reconstructed_programme':pred['programme'],'latent':pred['latent'],'clock':pred['clock'],
                'clock_definition':'reference-derived molecular coordinate; not true time or depth',
                'training_mean_baseline':p.mean,'evaluation':read_json(root/name/'result.json'),
                'depth':{'status':'unavailable','reason':'No matched expression/function cohort'},
                'programme_wave':{'status':'unavailable_for_this_checkpoint'}}
            if entry['wave'] and pred['clock'] is not None:
                wave=WaveReference.load(root/entry['wave']/'programme')
                result['programme_wave']=wave.residual(pred['clock'],q.values,q.mask,wave.coordinate_id,wave.context_id)
            return envelope(result)
        except (KeyError,ValueError,TypeError) as exc:raise HTTPException(422,str(exc))
    @app.get('/public_cases')
    def public_cases(module:str):
        if module not in {'endpoint','transition'} or module not in m['public_models']:raise HTTPException(422,'Unavailable response module')
        d=ResponseDataset.load(root/module/'dataset')
        return [{'observation_id':r['observation_id'],'split':r['split']} for r in d.rows if r['split'] in {'train','validation'}]
    @app.post('/public_response')
    def public_response(request:dict):
        try:
            name=request['module']
            if name not in {'endpoint','transition'} or name not in m['public_models']:raise ValueError('Unavailable response module')
            d=ResponseDataset.load(root/name/'dataset');indices=[i for i,r in enumerate(d.rows) if r['observation_id']==request['observation_id'] and r['split'] in {'train','validation'}]
            if len(indices)!=1:raise ValueError('Unknown development case')
            pred=ResponseRegressor.load(root/name).predict(d.current[indices],d.action[indices],d.scope,None if d.elapsed is None else d.elapsed[indices])
            return envelope({'module':name,'predicted':pred,'observed':d.target[indices],
                'target_mask':d.target_mask[indices],'scope':d.scope,'evaluation':read_json(root/name/'result.json')})
        except (KeyError,ValueError,TypeError) as exc:raise HTTPException(422,str(exc))
    @app.post('/functional')
    def functional(request:dict):
        try:
            if 'functional' not in m['public_models']:raise ValueError('Functional model unavailable')
            hour=float(request['hours_after_release']);history=float(request['history_days'])
            if not np.isfinite([hour,history]).all() or not 1<=history<=30:raise ValueError('History outside fitted 1–30 day scope')
            with np.load(root/'functional/model.npz',allow_pickle=False) as a:
                levels=a['hour_levels']
                if hour not in levels:raise ValueError('Hour outside fitted categorical levels')
                design=np.array([1]+[float(hour==h) for h in levels[1:]]+[(np.log1p(history)-float(a['history_mean']))/float(a['history_scale'])])
                predicted=float(expit(design@a['coefficients']))
            return envelope({'predicted_young_adult_fraction':predicted,'scope_and_evaluation':read_json(root/'functional/result.json'),
                'interpretation':'group-level assay readout; not expression-derived depth'})
        except (KeyError,ValueError,TypeError) as exc:raise HTTPException(422,str(exc))
    @app.get('/',response_class=HTMLResponse)
    def home():return PAGE
    return app


PAGE='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>VDC 开发实验</title>
<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:16px;background:#f6f8fb;color:#17283d}select,button{padding:9px;margin:6px;max-width:95%}pre{white-space:pre-wrap;background:white;padding:20px;border:1px solid #ccd6df}small{display:block;color:#566}</style>
<h1>Virtual Diapause Cell 开发实验</h1><p>冻结 PK1 模型 · experimental · 效果未验证</p>
<p>选择已准入的开发样本，查看实际模型输出与基线。Clock 为参考派生坐标；功能 depth 不可用。</p>
<label>模型 <select id="model"></select></label><label>样本 <select id="sample"></select></label><button id="run">分析</button>
<small id="identity"></small><pre id="result">加载中…</pre>
<script>const el=x=>document.getElementById(x);let caps;
async function change(){const info=caps.models[el('model').value];const rows=await fetch('samples?view='+encodeURIComponent(info.view)).then(r=>r.json());el('sample').replaceChildren();for(const r of rows){const o=document.createElement('option');o.value=r.observation_id;o.textContent=r.observation_id+' ['+r.split+']';el('sample').append(o)}}
async function init(){caps=await fetch('capabilities').then(r=>r.json());el('identity').textContent='快照 '+caps.snapshot_id;for(const n of Object.keys(caps.models)){const o=document.createElement('option');o.value=o.textContent=n;el('model').append(o)}await change();el('result').textContent='选择样本后点击分析';}
el('model').onchange=()=>change().catch(e=>el('result').textContent=String(e));el('run').onclick=async()=>{el('result').textContent='计算中…';try{const r=await fetch('analyse',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:el('model').value,observation_ids:[el('sample').value]})});el('result').textContent=JSON.stringify(await r.json(),null,2)}catch(e){el('result').textContent=String(e)}};init().catch(e=>el('result').textContent=String(e));</script></html>'''

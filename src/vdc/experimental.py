"""Internal experimental profile; never bypasses or rewrites research release approval."""
from pathlib import Path
import json
import numpy as np
from .io import read_json, write_json, sha256, object_hash


def jsonable(obj):
    if isinstance(obj,np.ndarray): return jsonable(obj.tolist())
    if isinstance(obj,(np.integer,np.floating,np.bool_)): return jsonable(obj.item())
    if isinstance(obj,float) and not np.isfinite(obj): return None
    if isinstance(obj,dict): return {k:jsonable(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)): return [jsonable(v) for v in obj]
    return obj


def build_registry(run):
    run=Path(run);statuses=read_json(run/'module_status.json')['modules']; capabilities={}
    for name,status in statuses.items():
        if name=='experimental_interface':continue
        available=status['execution_status']=='completed' and name not in {'knowledge','qwen_smoke','knowledge_corpus','semantics','data_dauer','data_ard'}
        files=status.get('files',{});folder=run/status.get('artifact',name)
        if status['execution_status']=='completed':
            if not files or not all((folder/n).is_file() and sha256(folder/n)==h for n,h in files.items()): raise ValueError('Artifact changed '+name)
        capabilities[name]={'execution_status':status['execution_status'],'deployment_status':'experimental_available' if available else 'unavailable',
            'science_status':'unvalidated','artifact':status.get('artifact'),'files':files,
            'reason':status.get('reason'),'model_identity':object_hash(files) if available else None}
    write_json(run/'experimental_release/registry.json',{'profile':'experimental','scientific_approval':False,
        'service_code_sha256':sha256(__file__),
        'capabilities':capabilities,'scope_policy':'context-bound; no new cross-species mapping',
        'limitations':['No calibrated depth or uncertainty','Poor models remain visible with their baselines',
            'Knowledge answers evaluated separately; numerical explanation uses only fixed values and allowed train evidence']})


def create_experimental_app(run):
    from fastapi import FastAPI,HTTPException
    from fastapi.responses import HTMLResponse
    from .contracts import ObservationBundle
    from .state import StatePredictor
    from .response import ResponseRegressor
    from .waves import WaveReference
    from .clock_reference import ExitReference
    from .observation import score_programmes,normalise_expression
    from .knowledge import retrieve
    from .io import read_jsonl
    run=Path(run).resolve();registry=read_json(run/'experimental_release/registry.json')
    if registry.get('profile')!='experimental' or registry.get('scientific_approval') is not False: raise ValueError('Invalid experimental registry')
    for name,c in registry['capabilities'].items():
        if c['execution_status']=='completed':
            if not all((run/c['artifact']/f).is_file() and sha256(run/c['artifact']/f)==h for f,h in c['files'].items()): raise ValueError('Registry artifact changed '+name)
    app=FastAPI(title='Virtual Diapause Cell — experimental')
    import threading
    knowledge_lock=threading.Lock()
    def require(name):
        c=registry['capabilities'].get(name,{})
        if c.get('deployment_status')!='experimental_available': raise HTTPException(409,detail={'module':name,'status':'unavailable','reason':c.get('reason','No completed model')})
        return c
    def envelope(name,result):
        c=require(name)
        return jsonable({'mode':'experimental','module':name,'model_identity':c['model_identity'],'science_status':'unvalidated','result':result})
    @app.get('/capabilities')
    def capabilities():return registry
    @app.get('/reports')
    def reports():
        result={}
        for n,c in registry['capabilities'].items():
            if c['deployment_status']!='experimental_available':continue
            for leaf in ('result.json','diagnostics.json','metrics.json','evaluation.json'):
                p=run/c['artifact']/leaf
                if p.exists():result[n+'/'+leaf]=read_json(p)
        return result
    def analyse_payload(request):
        context=request.get('context');name=request.get('state_model','state_base' if context=='dauer' else 'state_ard')
        if context not in {'dauer','ard'}:raise ValueError('Select admitted context dauer or ard')
        allowed={'state_base','state_semantic','state_matched_zero'} if context=='dauer' else {'state_ard'}
        if name not in allowed:raise ValueError('Model and context differ')
        require(name)
        if request.get('origin')!='public':raise ValueError('Alpha accepts declared public development inputs; internal killifish remains locked')
        folder=run/'data'/context;contract=read_json(folder/'expression_contract.json')
        if request.get('expression_scale')!=contract['input_scale']:raise ValueError('Expression scale differs from admitted model')
        genes=request['gene_ids'];expected=contract['gene_ids']
        if len(set(genes))!=len(genes) or set(genes)!=set(expected):raise ValueError('Exact admitted gene universe required; no guessed identifier mapping')
        x=np.asarray(request['expression'],float)
        if x.shape!=(len(genes),) or not np.isfinite(x).all():raise ValueError('One finite expression sample required')
        index={g:i for i,g in enumerate(genes)};x=x[[index[g] for g in expected]][None]
        y=normalise_expression(x,np.ones_like(x,bool),contract['input_scale'])
        definitions=read_json(folder/'programmes.json');scored=score_programmes(y,np.ones_like(y,bool),expected,definitions)
        p=StatePredictor.load(run/name);scope=p.manifest['scope']
        rows=[{'observation_id':'public_query','biological_unit':'public_query','study_family':'public_query',
            'split':'validation','origin':'public','link_ids':['public_query']}]
        b=ObservationBundle(scored['values'],scored['mask'],scored['coverage'],scored['feature_ids'],rows,contract['context'],
            'unit_holdout',np.zeros(1),np.zeros(1,bool),scope['clock_reference_id']).validate()
        pred=p.predict(b)
        result={'context':context,'scope':b.scope,'feature_ids':b.feature_ids,'observed':b.values,
            'reconstructed':pred['programme'],'latent':pred['latent'],'clock':pred['clock'],
            'training_mean_baseline':p.mean[None],
            'functional_depth':{'status':'unavailable','reason':'No expression-matched functional labels'},'uncertainty':'not_calibrated'}
        result['condition_readout']={'status':'unavailable','reason':'No matching trained readout for this encoder'}
        if name=='state_base' and registry['capabilities'].get('state_readout',{}).get('deployment_status')=='experimental_available':
            from .alpha_numeric import ridge_predict
            rm=read_json(run/'state_readout/result.json')
            if rm['encoder_sha256']!=sha256(run/name/'best.pt'):raise ValueError('State readout encoder changed')
            with np.load(run/'state_readout/latent.npz',allow_pickle=False) as a:
                scores=ridge_predict(pred['latent'],a['coefficients'],a['mean'],a['scale'])
            result['condition_readout']={'status':'computed','classes':rm['classes'],'scores':scores,
                'predicted_class':[rm['classes'][i] for i in scores.argmax(1)],'score_is_probability':False,
                'label_source':rm['label_source']}
        if context=='dauer':
            ref=ExitReference.load(run/'clock_reference');result['observed_reference_coordinate']=ref.predict(y,expected)
            result['clock_definition']='reference-derived local molecular coordinate; not independent clock truth or elapsed hours'
            if registry['capabilities'].get('waves',{}).get('deployment_status')=='experimental_available':
                w=WaveReference.load(run/'waves/programme')
                result['programme_wave']=w.residual(pred['clock'],b.values,b.mask,w.coordinate_id,w.context_id)
        return name,result
    @app.post('/analyse')
    def analyse(request:dict):
        try:
            name,result=analyse_payload(request);return envelope(name,result)
        except (ValueError,KeyError,TypeError) as e:raise HTTPException(422,str(e))
    @app.post('/response')
    def response(request:dict):
        name=request.get('module')
        if not isinstance(name,str):raise HTTPException(422,'A module name is required')
        require(name)
        if not (name.startswith('endpoint') or name.startswith('transition') or name=='functional'):raise HTTPException(422,'Not a response module')
        try:
            model=ResponseRegressor.load(run/name)
            current=np.asarray(request['current'],float);action=np.asarray(request['action'],float)
            elapsed=np.asarray(request['elapsed'],float) if 'elapsed' in request else None
            pred=model.predict(current,action,request['scope'],elapsed)
            baseline=np.zeros_like(pred) if model.scope['mode']=='endpoint' else current if model.scope['mode']=='transition' else None
            return envelope(name,{'prediction':pred,'zero_or_last_state_baseline':baseline,'scope':model.scope,
                'functional_scope':'Group protocol readout only, never expression-derived depth' if name=='functional' else None})
        except (ValueError,KeyError,TypeError) as e:raise HTTPException(422,str(e))
    @app.post('/pipeline')
    def pipeline(request:dict):
        try:
            name,analysis=analyse_payload(request['sample']);mode=request['mode']
            if mode not in {'endpoint','transition'}:raise ValueError('Choose endpoint or transition')
            module=mode+'_'+name
            require(module);response_model=ResponseRegressor.load(run/module)
            if response_model.scope['representation_id']!=sha256(run/name/'best.pt'):raise ValueError('Upstream representation changed')
            elapsed=np.asarray([request['elapsed_hours']],float) if mode=='transition' else None
            predicted=response_model.predict(analysis['reconstructed'],np.asarray([request['action']],float),response_model.scope,elapsed)
            return envelope(module,{'upstream':envelope(name,analysis),'prediction':predicted,'response_scope':response_model.scope,
                'future_clock_used':False,'knowledge_explanation':'Numerical predictions are experimental; no causal, calibrated depth or cross-context claim.'})
        except (ValueError,KeyError,TypeError) as e:raise HTTPException(422,str(e))
    @app.post('/waves')
    def waves(request:dict):
        require('waves')
        try:
            kind=request['kind']
            if kind not in {'gene','TF','programme'}:raise ValueError('Regulon is unavailable; choose gene, TF or programme')
            w=WaveReference.load(run/'waves'/kind)
            if request.get('context_id')!=w.context_id or request.get('coordinate_id')!=w.coordinate_id:raise ValueError('Wave context/reference differs')
            features=request['feature_ids']
            if not 1<=len(features)<=64 or len(set(features))!=len(features):raise ValueError('Choose 1..64 distinct reference features')
            ix=[w.feature_ids.index(g) for g in features]
            out=w.predict(np.array(request['coordinate'],float),w.coordinate_id,w.context_id)
            return envelope('waves',{'feature_ids':features,'kind':kind,'reference_expected':out['expected'][:,ix],
                'within_observed_range':out['within_observed_range'][:,ix], 'coordinate_id':w.coordinate_id,'context_id':w.context_id,
                'units':'log1p_library_10000_gene_expression' if kind in {'gene','TF'} else 'centred_rank_proxy'})
        except (ValueError,KeyError,TypeError) as e:raise HTTPException(422,str(e))
    @app.post('/knowledge/search')
    def search(request:dict):
        # Explicit lexical evidence route; it never claims to generate with the Qwen adapter.
        root=Path(__file__).resolve().parents[2]
        if not (run/'knowledge_corpus/audit.json').exists():raise HTTPException(409,'Audited knowledge corpus unavailable')
        records=read_jsonl(root/'knowledge/alpha/corpus.jsonl')
        if object_hash(records)!=read_json(run/'knowledge_corpus/audit.json')['corpus_hash']:
            raise HTTPException(409,'Knowledge corpus differs from the audited run')
        excluded=set(read_json(run/'config.json')['excluded_families'])
        return {'mode':'experimental','method':'train_only_lexical_evidence_retrieval',
            'records':retrieve(records,str(request.get('query','')),excluded,limit=5),
            'qwen_generated':False}
    @app.post('/knowledge/answer')
    def answer(request:dict):
        require('knowledge_adapter_eval')
        # Single GPU, one request at a time; explicit adapter load, never a hidden base fallback.
        from .knowledge_train import require_gpu,_base,_tokenizer
        import torch,gc
        from peft import PeftModel
        evidence=search(request)['records']
        c=read_json(run/'config.json')
        try:
            with knowledge_lock:
                require_gpu()
                tok=_tokenizer(c['resolved_model'],c['model_revision'])
                model=PeftModel.from_pretrained(_base(c['resolved_model'],c['model_revision']),str(run/'knowledge/adapter'),is_trainable=False).eval()
                prompt=[{'role':'system','content':'Answer only from the supplied evidence. State source and experimental context. Say uncertain when unsupported. Do not invent numerical depth, expression, or a clock. Generated prose cannot change numerical outputs.'},
                    {'role':'user','content':json.dumps({'question':str(request.get('query',''))[:1000],
                        'evidence':[{'source':r['source_ref'],'text':r['text']} for r in evidence]},ensure_ascii=False)}]
                text=tok.apply_chat_template(prompt,tokenize=False,add_generation_prompt=True)
                inputs=tok(text,return_tensors='pt',add_special_tokens=False).to('cuda')
                if inputs['input_ids'].shape[1]>2048:raise ValueError('Evidence/query length exceeds reviewed budget')
                with torch.inference_mode():ids=model.generate(**inputs,max_new_tokens=256,do_sample=False,pad_token_id=tok.pad_token_id)
                generated=tok.decode(ids[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
                del model,inputs,ids;gc.collect();torch.cuda.empty_cache()
                return envelope('knowledge_adapter_eval',{'answer':generated,'evidence':evidence,'weight_kind':'domain_adapter','numerical_outputs_modified':False})
        except (RuntimeError,ValueError,ImportError) as e:raise HTTPException(409,{'status':'unavailable','reason':str(e)})
    @app.get('/',response_class=HTMLResponse)
    def home():
        return '''<!doctype html><meta charset="utf-8"><title>Virtual Diapause Cell Alpha</title>
        <style>body{max-width:1100px;margin:32px auto;font:16px system-ui;background:#f5f7f8;color:#193a40}button,select,input{padding:10px;margin:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:18px;border-radius:8px}h1{font-size:28px}</style>
        <h1>Virtual Diapause Cell · Experimental</h1><p>内部实验版，尚未科学验证。低于基线的结果仍展示；缺少标签的 depth 不输出数值。</p>
        <button onclick="load('/capabilities')">模块状态</button><button onclick="load('/reports')">指标与基线</button>
        <p>上传 JSON 请求（运行目录 experimental_release/examples 提供真实开发样本）。</p>
        <select id="route"><option>/analyse</option><option>/pipeline</option><option>/response</option><option>/waves</option><option>/knowledge/search</option><option>/knowledge/answer</option></select>
        <input type="file" id="file" accept=".json"><button onclick="send()">运行</button><pre id="out">请选择模块状态或上传请求。</pre>
        <script>async function load(p){let r=await fetch(p);document.getElementById('out').textContent=JSON.stringify(await r.json(),null,2)}
        async function send(){try{let f=document.getElementById('file').files[0];let r=await fetch(document.getElementById('route').value,{method:'POST',headers:{'Content-Type':'application/json'},body:await f.text()});document.getElementById('out').textContent=JSON.stringify(await r.json(),null,2)}catch(e){document.getElementById('out').textContent=String(e)}}</script>'''
    return app


def exercise_app(run):
    """Real ASGI requests against freshly loaded saved models, not mock predictors."""
    from fastapi.testclient import TestClient
    from .contracts import ObservationBundle
    from .response import ResponseDataset, ResponseRegressor
    run=Path(run);app=create_experimental_app(run);client=TestClient(app)
    status=read_json(run/'module_status.json')['modules'];results={}
    # Use admitted development input, invert only the exact saved log transform for a real-request fixture.
    from .alpha_numeric import gene_data
    for context,modelname in [('dauer','state_base'),('ard','state_ard'),('dauer','state_semantic')]:
        if status.get(modelname,{}).get('execution_status')!='completed':continue
        b=ObservationBundle.load(run/'data'/context);x,genes=gene_data(run/'data'/context);i=int(b.indices('validation')[0])
        contract=read_json(run/'data'/context/'expression_contract.json')
        request={'context':context,'state_model':modelname,'origin':'public','gene_ids':genes,'expression':np.expm1(x[i]).tolist(),
            'expression_scale':contract['input_scale'],'fixture_note':'Exact inverse-log normalised source expression; ranks and library normalisation equivalent within float precision'}
        calls=[('/analyse',request,modelname)]
        mode='transition' if context=='dauer' else 'endpoint'
        if status.get(mode+'_'+modelname,{}).get('execution_status')=='completed':
            calls.append(('/pipeline',{'sample':request,'mode':mode,
                'action':[np.log1p(b.rows[i]['history_days'])] if context=='dauer' else [1,0,0], 'elapsed_hours':6},modelname+'_pipeline'))
        for route,payload,label in calls:
            response=client.post(route,json=payload)
            if response.status_code!=200:raise RuntimeError(f'{route}: {response.text}')
            write_json(run/'experimental_release/examples'/(label+'.json'),payload)
            write_json(run/'experimental_release/responses'/(label+'.json'),response.json());results[label]=response.status_code
    for name,s in status.items():
        if s['execution_status']!='completed' or not ((run/name/'response_model.json').exists()):continue
        d=ResponseDataset.load(run/name/'dataset');i=next(i for i,r in enumerate(d.rows) if r['split']=='validation')
        payload={'module':name,'current':d.current[[i]].tolist(),'action':d.action[[i]].tolist(),'scope':d.scope}
        if d.elapsed is not None:payload['elapsed']=d.elapsed[[i]].tolist()
        response=client.post('/response',json=payload)
        if response.status_code!=200:raise RuntimeError(response.text)
        write_json(run/'experimental_release/examples'/(name+'.json'),payload)
        write_json(run/'experimental_release/responses'/(name+'.json'),response.json());results[name]=response.status_code
    if status.get('waves',{}).get('execution_status')=='completed':
        from .waves import WaveReference
        for kind in ('gene','TF','programme'):
            w=WaveReference.load(run/'waves'/kind)
            payload={'kind':kind,'context_id':w.context_id,'coordinate_id':w.coordinate_id,'coordinate':[.5],'feature_ids':w.feature_ids[:5]}
            r=client.post('/waves',json=payload)
            if r.status_code!=200:raise RuntimeError(r.text)
            write_json(run/'experimental_release/examples'/('waves_'+kind+'.json'),payload)
            write_json(run/'experimental_release/responses'/('waves_'+kind+'.json'),r.json());results['waves_'+kind]=r.status_code
    if status.get('knowledge_corpus',{}).get('execution_status')=='completed':
        r=client.post('/knowledge/search',json={'query':'FOXO1 lipid'})
        write_json(run/'experimental_release/responses/knowledge_evidence.json',r.json())
    if status.get('knowledge_adapter_eval',{}).get('execution_status')=='completed':
        r=client.post('/knowledge/answer',json={'query':'What does the evidence support about FOXO1 in dormant mouse embryos?'})
        if r.status_code!=200:raise RuntimeError(r.text)
        write_json(run/'experimental_release/responses/knowledge_adapter.json',r.json());results['knowledge_adapter']=r.status_code
    if not results:raise RuntimeError('No actual model invocation available')
    write_json(run/'experimental_release/invocation.json',{'actual_saved_models':True,'transport':'local_in_process_ASGI','public_server_started':False,'requests':results})

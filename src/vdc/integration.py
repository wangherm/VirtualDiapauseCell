"""One new-input Python/CLI/HTTP inference path. No fitting, run lookup, or saved predictions."""
from pathlib import Path
import copy,os,threading
import numpy as np
from .io import read_json,read_jsonl,object_hash
from .integration_bundle import load_manifest,SPECIES,CONTEXT
from .clock_wave import ClockWave,inputs
from .clock_wave_revision import residual_features
from .application import predict_ridge
from .application_data import align_counts
from .observation import normalise_expression,score_programmes
from .experimental import jsonable


def read_request(path):
    """CLI file boundary. HTTP takes inline arrays only and cannot read arbitrary paths."""
    import csv
    path=Path(path);r=read_json(path);source=r.pop('input',None)
    if source is None:return r
    def file(name):return (path.parent/name).resolve()
    if source.get('format')=='h5ad':
        import anndata
        if not source.get('counts_layer'):raise ValueError('Explicit counts_layer required; .X is never assumed counts')
        a=anndata.read_h5ad(file(source['path']))
        if source['counts_layer'] not in a.layers:raise ValueError('Named counts layer missing')
        x=a.layers[source['counts_layer']];x=x.toarray() if hasattr(x,'toarray') else x
        r['genes']=a.var_names.tolist();r['counts']=np.asarray(x).tolist();ids=a.obs_names.tolist()
    elif source.get('format')=='csv':
        with file(source['matrix']).open(encoding='utf-8-sig',newline='') as f:
            table=list(csv.reader(f))
        if not table or table[0][0]!='sample_id':raise ValueError('CSV first column must be sample_id; remaining columns are gene IDs')
        r['genes']=table[0][1:];ids=[v[0] for v in table[1:]];r['counts']=[[float(v) for v in row[1:]] for row in table[1:]]
    else:raise ValueError('Input format must be csv or h5ad')
    with file(source['samples']).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    if len({v['sample_id'] for v in rows})!=len(rows) or set(ids)!={v['sample_id'] for v in rows} or len(set(ids))!=len(ids):raise ValueError('Matrix/sample IDs must match exactly and uniquely')
    index={v['sample_id']:v for v in rows};r['samples']=[index[i] for i in ids];return r


class VirtualDiapauseCell:
    def __init__(self,model,base_path=None):
        self.root=Path(model).resolve();self.manifest=load_manifest(self.root)
        self.base_path=base_path or os.environ.get('VDC_QWEN_BASE');self.engine=None;self.engine_lock=threading.Lock()
        self.models={}
        for view,entry in self.manifest['views'].items():
            for variant,key in [('cw_stage_baseline','baseline'),('domain_clock_wave','domain')]:
                if entry.get(key):self.models[view,variant]=ClockWave.load(self.root/entry[key]/'model')
        self.records=read_jsonl(self.root/'knowledge/records.jsonl') if self.manifest.get('knowledge') else []

    def _engine(self):
        with self.engine_lock:
            if self.engine is None:
                if not self.manifest.get('knowledge') or not self.base_path:raise RuntimeError('Domain Qwen/base snapshot unavailable; full execution incomplete')
                from .integration_knowledge import DomainQwen
                self.engine=DomainQwen(self.root,self.manifest,self.base_path)
        return self.engine

    def _numeric(self,x,types,view,variant):
        m=self.models[view,variant];key='domain' if variant=='domain_clock_wave' else 'baseline';folder=self.root/self.manifest['views'][view][key]
        q=m.hidden_predictions(x,types);pred=q['hidden_prediction'];finite=np.isfinite(q['clock'])
        if self.manifest['views'][view]['route']=='clock_identity_residual':
            design,_=residual_features(m,x,types);pred=np.full_like(pred,np.nan)
            if finite.any():pred[finite]=predict_ridge(folder/'readout.npz',design[finite])
        _,features,mask,_=m._features(x)
        matched=predict_ridge(folder/'matched_direct.npz',m.direct_features(features,mask,types))
        observed=normalise_expression(x,np.ones_like(x,bool),'counts');shape=read_json(folder/'shapes.json')
        selected=sorted({s['gene'] for s in shape});gene_index={g:i for i,g in enumerate(m.meta['genes'])};indices=[gene_index[g] for g in selected]
        expected=np.full((len(x),len(m.meta['genes'])),np.nan)
        if finite.any():expected[finite]=predict_ridge(folder/'wave_parameters.npz',m.clock_design(q['clock'][finite],np.array(types)[finite]))
        inside=np.array([v=='located' for v in q['status']])
        return {'clock':q['clock'],'status':q['status'],'robust_distance':q['robust_distance'],
                'reference_coordinate':m.meta['coordinate'],'uncertainty':m.meta['uncertainty'],
                'programme_ids':[d['id'] for d in m.meta['definitions']],
                'feature_channels':['rank_minus_early','relative_log_amplitude_minus_early'],
                'observed':q['observed'],'expected':q['reference_expected'],'residual':q['residual'],
                'linear_extrapolation':q['expected_linear_extrapolation'],'mask':q['input_mask'],'coverage':q['coverage'],
                'hidden_gene_ids':m.meta['target_genes'],'gene_prediction':pred,'direct_ridge':matched,
                'original_direct_ridge':q['hidden_direct_ridge'],'identity_mean':q['hidden_identity_mean'],
                'hidden_gene_measurement':observed[:,m.meta['target_indices']],
                'waves':{'genes':selected,'TF_RNA_genes':sorted({s['gene'] for s in shape if s['TF_RNA']}),
                         'observed':observed[:,indices],'expected':np.where(inside[:,None],expected[:,indices],np.nan),
                         'linear_extrapolation':expected[:,indices],
                         'residual':np.where(inside[:,None],observed[:,indices]-expected[:,indices],np.nan)},
                'support_range':{t:[float(m.arrays['lower'][i]),float(m.arrays['upper'][i])] for i,t in enumerate(m.meta['types'])},
                'target_isolation_scope':m.meta['target_isolation_scope'],'route':self.manifest['views'][view]['route']}

    def _cards(self,x,genes,model,samples):
        from . import identity
        saved=read_json(self.root/'identity/reference.json');ref=copy.deepcopy(saved['reference'])
        y,_,_=inputs(x,genes,model.meta['definitions'],model.meta['config'].get('extra_hidden',[]))
        if ref and all(saved['genes'][j] in genes for j in ref['indices']):
            ref['indices']=[genes.index(saved['genes'][j]) for j in ref['indices']]
            prediction=identity.predict(ref,saved['calibration'],y)
        else:ref=None;prediction=[{'prediction':'unknown','candidates':[],'reason':'unsupported_marker_universe'} for _ in samples]
        rows=[{'observation_id':s['sample_id']} for s in samples]
        return identity.evidence_cards(ref,prediction,y,genes,rows,{})

    def _response(self,r):
        if r['module']=='regulon':
            entry=self.manifest.get('regulons',{}).get('public')
            if not entry or r.get('species')!='Caenorhabditis elegans' or r.get('scale') not in {'counts','estimated_counts','log1p_library_10000'}:return {'module':'regulon','status':'unsupported','reason':'Public target sets require declared worm counts, estimated_counts or prepared log1p_library_10000; no cross-species remapping'}
            x=np.asarray(r['expression'],float);genes=r['genes']
            if x.ndim!=2 or x.shape[1]!=len(genes) or len(set(genes))!=len(genes) or not np.isfinite(x).all() or (x<0).any() or (x.sum(1)==0).any():raise ValueError('Valid finite nonnegative libraries with unique gene IDs required')
            y=normalise_expression(x,np.ones_like(x,bool),r['scale']) if r['scale'] in {'counts','estimated_counts'} else x
            defs=read_json(self.root/entry['path'])
            scored=score_programmes(y,np.ones_like(y,bool),genes,defs)
            return {'module':'regulon','status':'computed','prediction':np.where(scored['mask'],scored['values'],np.nan),'mask':scored['mask'],
                    'coverage':scored['coverage'],'feature_ids':scored['feature_ids'],'scope':entry['scope'],
                    'interpretation':'Unsigned target expression rank proxy, not causal TF activity'}
        name=r['module'];entry=self.manifest['public_models'].get(name)
        if not entry:return {'module':name,'status':'unsupported','reason':'No packaged compatible model'}
        scope=entry['scope']
        if r.get('species')!=entry.get('species',scope.get('species')):return {'module':name,'status':'unsupported','reason':'Species mismatch'}
        if r.get('scope')!=scope:return {'module':name,'status':'unsupported','reason':'Exact species/context/action/representation scope required','required_scope':scope}
        folder=self.root/entry['path']
        if name=='functional':
            from scipy.special import expit
            hour=float(r['hours_after_release']);history=float(r['history_days'])
            with np.load(folder/'model.npz',allow_pickle=False) as a:
                levels=a['hour_levels']
                if not np.isfinite([hour,history]).all() or hour not in levels or not 1<=history<=30:return {'module':name,'status':'unsupported','reason':'Outside fitted categorical hours or maintenance history'}
                design=np.array([1]+[float(hour==h) for h in levels[1:]]+[(np.log1p(history)-float(a['history_mean']))/float(a['history_scale'])])
                pred=float(expit(design@a['coefficients']))
            baseline=read_json(folder/'baselines.json')[str(hour)]
            return {'module':name,'status':'computed','prediction':pred,'baseline':{'kind':'train_hour_binomial_mean','prediction':baseline},'scope':scope,'interpretation':'group young-adult fraction under this assay; not expression depth'}
        from .response import ResponseRegressor
        current=np.asarray(r['current'],float);action=np.asarray(r['action'],float)
        if current.shape!=(1,len(scope['input_names'])) or action.shape!=(1,len(scope['action_names'])) or not np.isfinite(current).all() or not np.isfinite(action).all():raise ValueError('One finite response with the exact fitted feature order is required')
        if r.get('new_acute_ko') and r.get('same_genotype_already_in_initial'):raise ValueError('Constitutive genotype cannot be applied as acute KO')
        if not any(np.allclose(action[0],v,rtol=0,atol=0) for v in entry['admitted_actions']) or len(action)!=1:return {'module':name,'status':'unsupported','reason':'Action outside fitted contrasts; submit one response at a time'}
        elapsed=r.get('elapsed')
        if name=='transition' and (elapsed is None or len(elapsed)!=1 or not entry['elapsed_range'][0]<=float(elapsed[0])<=entry['elapsed_range'][1]):return {'module':name,'status':'unsupported','reason':'Elapsed time outside fitted scope'}
        pred=ResponseRegressor.load(folder).predict(current,action,scope,elapsed)
        time_baseline=None
        if name=='transition':
            from .alpha_numeric import ridge_predict
            with np.load(folder/'time_baseline.npz',allow_pickle=False) as a:
                time_baseline=ridge_predict(np.column_stack([elapsed,action]),a['coefficients'],a['mean'],a['scale'])
        key=action[0].tolist()+([] if elapsed is None else [float(elapsed[0])]);baseline={'status':'unsupported','reason':'No exact train condition/time mean'}
        for item in read_json(folder/'baselines.json'):
            if item['key']==key:baseline={'status':'computed','prediction':np.where(item['available'],item['mean'],np.nan),'training_units':item['training_units'],'kind':'train_condition_time_mean'}
        return {'module':name,'status':'computed','prediction':pred,'baseline':baseline,
                'time_history_baseline':time_baseline,'zero_effect_or_persistence':current if name=='transition' else np.zeros_like(pred),'scope':scope,'input':r,'science_status':'unvalidated'}

    def analyse(self,request,mode='full',variant='domain_clock_wave',output=None):
        if mode not in {'full','numeric_only'} or variant not in {'domain_clock_wave','cw_stage_baseline'}:raise ValueError('Unknown execution mode or numerical variant')
        r=copy.deepcopy(request)
        if not isinstance(r.get('request_id'),str) or not r['request_id']:raise ValueError('New request_id required')
        result={'request_id':r['request_id'],'bundle_id':self.manifest['bundle_id'],'execution_mode':mode,'model_variant':variant,
                'status':'completed','components':{},'numeric':None,'baseline':None,'identity':[], 'knowledge':[],
                'responses':[],'regulons':{'status':'not_requested'},'depth':{'status':'unsupported','reason':'No matched expression/function cohort'},
                'science_status':'experimental; new request computation is not independent biological validation'}
        fields={};cards=[];context=r.get('context','unspecified')
        if 'counts' in r:
            view=r.get('view');entry=self.manifest['views'].get(view)
            if not entry:raise ValueError('Unknown view')
            samples=r.get('samples',[])
            if not 1<=len(samples)<=64 or len({s['sample_id'] for s in samples})!=len(samples):raise ValueError('Supply 1–64 distinct sample IDs')
            if any(not s.get('biological_unit') or not s.get('identity') or not s.get('identity_source') for s in samples):raise ValueError('Every sample requires biological_unit, identity, identity_source; explicitly use unknown when unavailable')
            if r.get('scale')!='counts':raise ValueError('Explicit counts scale required; no log/scaled expression conversion')
            x=align_counts(r['counts'],r['genes'],entry['gene_ids'])
            if x.shape[0]!=len(samples) or not np.equal(x,np.rint(x)).all():raise ValueError('Exact integer counts and matching metadata required')
            if (x.sum(1)==0).any():raise ValueError('Empty count library')
            if r.get('species')!=SPECIES or context!=CONTEXT:
                result['components']['numeric']={'status':'unsupported','reason':'Species or material context differs from frozen reference'}
            elif (view,variant) not in self.models:
                result['status']='partial';result['components']['numeric']={'status':'failed','reason':'Requested fitted model variant missing'}
            else:
                types=[s['identity'] for s in samples]
                result['numeric']=self._numeric(x,types,view,variant);result['baseline']=self._numeric(x,types,view,'cw_stage_baseline')
                result['components']['numeric']={'status':'computed','view':view,'new_input_computed':True,'fit_during_request':False}
                result['samples']=samples;result['view']=view
                m=self.models[view,variant];cards=self._cards(x,entry['gene_ids'],m,samples)
                for i,s in enumerate(samples):
                    result['identity'].append({'sample_id':s['sample_id'],'actual_identity':s['identity'],'source':s['identity_source'],'qwen':{'status':'not_run'}})
                    fields[f's{i}.clock']={'value':jsonable(result['numeric']['clock'][i]),'kind':'reference_location','support':result['numeric']['status'][i]}
                    p=len(m.meta['definitions']);rr=result['numeric']['residual'][i,:p]
                    for j in sorted(np.flatnonzero(np.isfinite(rr)),key=lambda j:-abs(rr[j]))[:3]:
                        fields[f's{i}.{m.meta["definitions"][j]["id"]}.rank_residual']={'value':float(rr[j]),'kind':'measured_visible_rank_minus_reference'}
                if r.get('regulons'):
                    e=self.manifest.get('regulons',{}).get('local')
                    result['regulons']={'status':'unsupported','reason':'Local network unavailable'}
                    if e:
                        net=read_json(self.root/e['path']);safe,_,_=inputs(x,entry['gene_ids'],m.meta['definitions'])
                        hidden=set(m.meta['hidden_removed']);index={g:j for j,g in enumerate(entry['gene_ids'])};values=[]
                        for n in net:
                            ids=[index[g] for g in n['targets'] if g in index and g not in hidden]
                            values.append({'tf':n['tf'],'coverage':len(ids)/len(n['targets']),'visible_target_mean':safe[:,ids].mean(1) if len(ids)>=3 else None})
                        result['regulons']={'status':'computed','kind':'unsigned visible target RNA mean; not original standardized activity or TF activity','networks':values,'scope':e['scope']}
        for response in r.get('responses',[]):
            q=self._response(response);result['responses'].append(q)
            if q['status']=='computed':fields['response'+str(len(result['responses']))]={'value':jsonable(q['prediction']),'kind':'model_prediction','scope':q['scope']}
        if not result['components'] and not result['responses']:raise ValueError('Supply counts and metadata or compatible response requests')
        if mode=='full':
            if variant!='domain_clock_wave' or not self.manifest.get('full_ready'):
                result['status']='partial';result['components']['semantic_model']={'status':'failed','reason':'Full requires exported domain-semantic fit and linked knowledge artifacts'}
            try:
                from .integration_knowledge import explain,review_identity
                engine=self._engine()
                for i,card in enumerate(cards):
                    item=review_identity(engine,card);result['identity'][i]['qwen']=item
                    if not item['result']['format_valid']:result['status']='partial'
                item=explain(engine,self.records,r.get('question','Describe observed deviations and limitations'),jsonable(fields),context)
                result['knowledge']=[item];result['components']['qwen']={'status':item['explanation']['status'],'actual_online_generation':True,'provenance':engine.provenance}
                if item['explanation']['status']!='generated_reference_checks_passed':result['status']='partial'
            except Exception as exc:
                result['status']='partial';result['components']['qwen']={'status':'failed','reason':str(exc),'error_type':type(exc).__name__}
        else:result['components']['qwen']={'status':'not_run','reason':'explicit_numeric_only'}
        result['provenance']={'bundle_id':self.manifest['bundle_id'],'input_hash':object_hash(r),'execution_mode':mode,'model_variant':variant,
                              'artifact_hashes':self.manifest['files'],'fit_during_request':False,'answer_cache':False,
                              'software':self.manifest['implementation'],'field_registry':jsonable(fields)}
        result=jsonable(result)
        if output:
            from .integration_report import export_result
            export_result(result,output)
        return result

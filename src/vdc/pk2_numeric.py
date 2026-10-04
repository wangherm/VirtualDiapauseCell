"""PK2 numerical training and fixed development diagnostics, canonical state model."""
from pathlib import Path
import copy,shutil,time
import numpy as np
import torch
from .io import read_json,write_json,save_npz,sha256,object_hash
from .contracts import ObservationBundle
from .state import StateConfig,StatePredictor,fit_state,corrupt
from .metrics import regression_metrics,study_unit_weights


def dynamic_ratio(pred, target, mask):
    ratios=[]
    for j in range(target.shape[1]):
        idx=np.flatnonzero(mask[:,j])
        if len(idx)<2 or np.std(target[idx,j])<=1e-8:continue
        ratios.append(float(np.std(pred[idx,j])/np.std(target[idx,j])))
    return {'dynamic_std_ratio_median':float(np.median(ratios)) if ratios else None,
            'dynamic_std_ratio_status':'available' if ratios else 'unavailable_insufficient_observations_or_variation',
            'dynamic_std_ratio_features':len(ratios)}

def reuse_equivalent(bundle,candidate_bundle,candidate,out,job,prior,config):
    """Reuse only an identical numerical experiment inside this immutable run."""
    b=ObservationBundle.load(bundle);old=ObservationBundle.load(candidate_bundle)
    def key(j):
        return {'seed':j.get('seed',j.get('algorithm_seed',42)),
            'steps':config.get('integration_steps') or j['max_steps'],
            'checkpoints':j.get('checkpoint_steps',[]),'route':j.get('route_config'),
            'parent':j.get('parent_task'),'semantics':j.get('semantics'),
            'features':j.get('feature_view')}
    if b.fingerprint!=old.fingerprint or key(job)!=key(prior):return False
    candidate=Path(candidate);out=Path(out)
    from .pk1_assets import verify_files
    queue=read_json(candidate.parents[1]/'queue_status.json')['tasks'][candidate.name]
    if queue['status'] not in {'evaluated_new','reused_verified'}:return False
    verify_files(candidate,queue['files']);result=read_json(candidate/'result.json')
    if result['input_bundle_fingerprint']!=b.fingerprint:raise ValueError('Reuse input fingerprint mismatch')
    shutil.copytree(candidate,out)
    write_json(out/'reuse.json',{'source_job':candidate.name,'source_inventory_hash':object_hash(queue['files']),
        'equivalence':key(job),'input_bundle_fingerprint':b.fingerprint,'new_training':False})
    result.update(job=job,new_training=False,status='reused_verified');write_json(out/'result.json',result);return True

def query(p,b):
    q=copy.deepcopy(b);q.clock[:]=0;q.clock_mask[:]=False
    return p.predict(q)

def scaling(p,b):
    contexts=p.manifest.get('contexts',[])
    if contexts:
        indices=[contexts.index(r['study_family']) for r in b.rows];return p.mean[indices],p.scale[indices]
    return np.broadcast_to(p.mean,b.values.shape),np.broadcast_to(p.scale,b.values.shape)

def metrics(pred,target,mask,rows):
    units={}
    for group in sorted({(r['study_family'],r['biological_unit']) for r in rows}):
        idx=[i for i,r in enumerate(rows) if (r['study_family'],r['biological_unit'])==group]
        units['|'.join(group)]=regression_metrics(pred[idx],target[idx],mask[idx])
    values=[v['mse'] for v in units.values() if v['mse'] is not None]
    return {'pooled':regression_metrics(pred,target,mask),'by_unit':units,
        'macro_unit_mse':float(np.mean(values)) if values else None,'coverage':float(mask.mean())}

def dual_ridge(x,y,train_rows,alpha=1.,target_mask=None):
    # Dual solve scales with independent observations, including the 2000-gene diagnostic.
    if target_mask is not None and not np.asarray(target_mask).all():
        groups={};params={};models=[]
        for j in range(y.shape[1]):groups.setdefault(np.asarray(target_mask[:,j],bool).tobytes(),[]).append(j)
        for number,(key,cols) in enumerate(groups.items()):
            valid=np.frombuffer(key,dtype=bool);idx=np.flatnonzero(valid)
            if not len(idx):continue
            fn=dual_ridge(x[idx],y[np.ix_(idx,cols)],[train_rows[i] for i in idx],alpha)
            models.append((cols,fn));params[f'g{number}_columns']=np.array(cols)
            params.update({f'g{number}_{k}':v for k,v in fn.parameters.items()})
        def masked_predict(q):
            got=np.full((len(q),y.shape[1]),np.nan)
            for cols,fn in models:got[:,cols]=fn(q)
            return got
        masked_predict.parameters=params;return masked_predict
    w=study_unit_weights(train_rows);w=w/w.mean();mean=np.average(x,axis=0,weights=w);sd=np.maximum(np.sqrt(np.average((x-mean)**2,axis=0,weights=w)),.001)
    ym=np.average(y,axis=0,weights=w);z=(x-mean)/sd;zw=z*np.sqrt(w[:,None])
    coef=zw.T@np.linalg.solve(zw@zw.T+alpha*np.eye(len(x)),(y-ym)*np.sqrt(w[:,None]))
    def predict(q):return ((q-mean)/sd)@coef+ym
    predict.parameters={'coefficients':coef,'mean':mean,'scale':sd,'target_mean':ym}
    return predict

def evaluate(run,b,out,mask_repetitions=10):
    run=Path(run);out=Path(out);p=StatePredictor.load(run);tr=b.indices('train');va=b.indices('validation');v=b.subset(va)
    original=query(p,v);second=query(StatePredictor.load(run),v)
    if not np.array_equal(original['programme'],second['programme']):raise RuntimeError('Reload changed numerical output')
    write_json(out/'reload.json',{'status':'passed','max_abs_delta':0.,'new_process':False,'checkpoint_sha256':sha256(run/'best.pt')})
    mu,sd=scaling(p,b);x=(b.values-mu)/sd;pred=original['programme']
    saved={'observed':v.values,'prediction':pred,'mask':v.mask,'mean_baseline':mu[va],'latent':original['latent']}
    if original['clock'] is not None:saved['clock']=original['clock'];saved['reference_clock']=v.clock
    save_npz(out/'predictions.npz',**saved)
    write_json(out/'rows.json',v.rows)
    ridge={}
    for study in sorted({r['study_family'] for r in b.rows}):
        idx=np.array([i for i in tr if b.rows[i]['study_family']==study]);xx,mm,_,_=corrupt(torch.tensor(x[idx]),torch.tensor(b.mask[idx]),torch.tensor(b.coverage[idx]),.2,.05,91001)
        ridge[study]=dual_ridge(np.column_stack([xx.numpy(),mm.numpy()]),b.values[idx],[b.rows[i] for i in idx],target_mask=b.mask[idx])
        save_npz(out/'ridge_parameters'/f'{object_hash(study)[:16]}.npz',**ridge[study].parameters)
    masks=[];all_pred=[];all_ridge=[];all_mask=[]
    for fraction in [.1,.3,.5]:
        for repeat in range(mask_repetitions):
            seed=210000+int(fraction*100)*100+repeat
            vx,vm,vc,hidden=corrupt(torch.tensor(x[va]),torch.tensor(v.mask),torch.tensor(v.coverage),fraction,.05,seed)
            q=copy.deepcopy(v);q.values=vx.numpy()*sd[va]+mu[va];q.mask=vm.numpy();q.coverage=vc.numpy()
            got=query(p,q)['programme'];rp=np.zeros_like(got)
            for study,fn in ridge.items():
                idx=[i for i,r in enumerate(v.rows) if r['study_family']==study]
                if idx:rp[idx]=fn(np.column_stack([vx.numpy()[idx],vm.numpy()[idx]]))
            hm=hidden.numpy();masks.append({'fraction':fraction,'seed':seed,'model':metrics(got,v.values,hm,v.rows),
                'ridge':metrics(rp,v.values,hm & np.isfinite(rp),v.rows),'train_mean':metrics(mu[va],v.values,hm,v.rows),
                'visible_preservation':metrics(got,v.values,vm.numpy(),v.rows)})
            all_pred.append(got);all_ridge.append(rp);all_mask.append(hm)
    save_npz(out/'mask_predictions.npz',model=np.stack(all_pred),ridge=np.stack(all_ridge),mask=np.stack(all_mask))
    write_json(out/'mask_metrics.json',masks)
    per_feature=[]
    for j,fid in enumerate(b.feature_ids):
        keep=v.mask[:,j];yy=v.values[keep,j];pp=pred[keep,j]
        correlation=float(np.corrcoef(yy,pp)[0,1]) if len(yy)>2 and yy.std()>1e-8 and pp.std()>1e-8 else None
        per_feature.append({'feature_id':fid,**regression_metrics(pred[:,j],v.values[:,j],v.mask[:,j]),'correlation':correlation,
            'observed_std':float(yy.std()) if len(yy) else None,'predicted_std':float(pp.std()) if len(pp) else None})
    contrasts=[]
    for study in sorted({r['study_family'] for r in v.rows}):
        conditions=sorted({r.get('condition','unspecified') for r in v.rows if r['study_family']==study})
        if len(conditions)<2:continue
        first=conditions[0];a=[i for i,r in enumerate(v.rows) if r['study_family']==study and r.get('condition','unspecified')==first]
        for condition in conditions[1:]:
            z=[i for i,r in enumerate(v.rows) if r['study_family']==study and r.get('condition','unspecified')==condition]
            mask=v.mask[a].all(0)&v.mask[z].all(0);actual=v.values[z].mean(0)-v.values[a].mean(0);got=pred[z].mean(0)-pred[a].mean(0)
            contrasts.append({'study':study,'reference':first,'condition':condition,'actual':actual.tolist(),'predicted':got.tolist(),
                'metric':regression_metrics(got,actual,mask),'direction_agreement':float(np.mean(np.sign(actual[mask])==np.sign(got[mask]))) if mask.any() else None})
    write_json(out/'feature_metrics.json',per_feature);write_json(out/'contrasts.json',contrasts)
    # Known deviations are simulation diagnostics, never additional biological observations.
    rng=np.random.default_rng(830041);deviation=np.zeros_like(v.values);deviation[:,::5]=sd[va][:,::5]*.5
    q=copy.deepcopy(v);q.values=q.values+deviation+rng.normal(0,.05,q.values.shape)*sd[va]
    noisy=query(p,q)['programme'];measured=v.mask & (deviation!=0)
    save_npz(out/'simulated_deviation.npz',injected=deviation,predicted_change=noisy-pred,mask=measured)
    report={'full_input':metrics(pred,v.values,v.mask,v.rows),'train_mean':metrics(mu[va],v.values,v.mask,v.rows),
        'reference_clock':regression_metrics(original['clock'],v.clock,v.clock_mask) if original['clock'] is not None else {'status':'unavailable'},
        **dynamic_ratio(pred,v.values,v.mask),
        'simulated_deviation_retention':regression_metrics(noisy-pred,deviation,measured),
        'training_steps':read_json(run/'metrics.json')['steps_completed'],'mask_runs':len(masks),'reserved_used':False,
        'scope':'development; reconstruction and reference-derived clock are not independent biological ground truth',
        'science_status':'unvalidated','checkpoint_sha256':sha256(run/'best.pt')}
    write_json(out/'result.json',report);return report

def train_job(bundle_path,out,job,config,pretrained=None,semantics=None,semantic_provenance=None,device='cpu',steps_override=None):
    b=ObservationBundle.load(bundle_path);out=Path(out);cfg=StateConfig(**config['state_config']);cfg.seed=job.get('seed',job.get('algorithm_seed',42))
    if job.get('feature_view')=='local_gene_panel2000':cfg.encoder_kind='gene_mlp'
    dimension=1 if job['family']=='feature_resolution' else config['semantic_dim']
    sem=np.zeros((len(b.feature_ids),dimension),np.float32) if semantics is None else semantics
    provenance=semantic_provenance or {'ablation':'matched_zero','dimension':dimension}
    from .pk2_semantics import validate_semantics,semantic_mode,check_forward,preflight
    contract=validate_semantics(job,sem,provenance,b.feature_ids,dimension)
    if semantic_mode(job)!='zero':contract['pretraining_forward_check']=preflight(b,cfg,sem)
    write_json(out/'semantic_input.json',contract)
    steps=steps_override or job['max_steps'];start=time.monotonic()
    mode=job.get('route_config',{}).get('transfer','none');encoder_only=mode=='encoder_only'
    write_json(out/'phase.json',{'phase':'training','requested_steps':steps})
    fit_state(b,out/'model',steps,cfg,device=device,semantics=sem,semantic_provenance=provenance,
        pretrained=pretrained,transfer_mode='encoder_only' if encoder_only else 'shared_model',
        checkpoint_steps=[n for n in job.get('checkpoint_steps',[]) if n<=steps],multi_context=job['family']=='public_pretraining',
        checkpoint_selection='fixed_final' if job['family']=='pool_generalization' else 'validation')
    if semantic_mode(job)!='zero':
        contract['forward_check']=check_forward(StatePredictor.load(out/'model'),b.subset(b.indices('validation')),sem)
        write_json(out/'semantic_input.json',contract)
    write_json(out/'phase.json',{'phase':'trained_evaluation_pending','completed_steps':steps})
    report=evaluate(out/'model',b,out/'evaluation',config.get('mask_repetitions',10))
    for path in sorted((out/'model').glob('step_*.pt')):
        if job['family']=='pool_generalization':continue  # Outer pool is evaluated only at the fixed final checkpoint.
        target=out/'checkpoints'/path.stem;target.mkdir(parents=True,exist_ok=True)
        shutil.copy2(out/'model/run.json',target/'run.json');shutil.copy2(path,target/'best.pt')
        p=StatePredictor.load(target);v=b.subset(b.indices('validation'));prediction=query(p,v)
        save_npz(target/'predictions.npz',prediction=prediction['programme'],target=v.values,mask=v.mask)
        write_json(target/'evaluation.json',metrics(prediction['programme'],v.values,v.mask,v.rows))
    write_json(out/'result.json',{'job':job,'evaluation':report,'wall_seconds':time.monotonic()-start,'device':device,
        'actual_steps':steps,'budget_overridden_for_integration_check':steps_override is not None,
        'input_bundle_fingerprint':b.fingerprint,'new_training':True,'status':'evaluated_new'})
    write_json(out/'phase.json',{'phase':'evaluated','completed_steps':steps})

"""CPU fits and diagnostics; no learned causal claims from proxy targets."""
from pathlib import Path
import copy
import numpy as np
from .io import read_json,write_json,save_npz,object_hash,sha256
from .contracts import ObservationBundle
from .alpha_numeric import gene_data,response_dataset,metric
from .pk2_numeric import dual_ridge,metrics
from .pk2_data import DataUnavailable,annotation_members

def type_waves(bundle,out):
    b=ObservationBundle.load(bundle);tr=np.array([i for i in b.indices('train') if b.rows[i]['reference_eligible']]);va=b.indices('validation')
    if len(tr)<3:raise DataUnavailable('Too few eligible training profiles')
    types=sorted({b.rows[i].get('cell_type','unspecified') for i in tr})
    onehot=np.array([[float(r.get('cell_type','unspecified')==t) for t in types] for r in b.rows]);known=onehot.sum(1)>0
    designs={'global_linear':b.clock[:,None],'type_intercept_shared_clock':np.column_stack([onehot,b.clock]),
        'type_intercept_ridge_slopes':np.column_stack([onehot,b.clock,onehot*b.clock[:,None]])}
    report={}
    for name,x in designs.items():
        fn=dual_ridge(x[tr],b.values[tr],[b.rows[i] for i in tr]);pred=fn(x[va]);mask=b.mask[va]&b.clock_mask[va,None]
        if name!='global_linear':mask &= known[va,None]
        folder=Path(out)/name;save_npz(folder/'model.npz',**fn.parameters);save_npz(folder/'predictions.npz',prediction=pred,target=b.values[va],mask=mask)
        report[name]=metrics(pred,b.values[va],mask,[b.rows[i] for i in va])
    write_json(Path(out)/'result.json',{'models':report,'fit_ids':[b.rows[i]['observation_id'] for i in tr],
        'types':types,'coordinate':b.clock_reference_id,'unknown_types':int(sum(~known[va])),'scope':'reference-derived clock; type intercepts are not independent cell replicates'})

def transition_residual(public,out):
    d=response_dataset(public,'transition');tr=np.array([i for i,r in enumerate(d.rows) if r['split']=='train']);va=np.array([i for i,r in enumerate(d.rows) if r['split']=='validation'])
    # Group by all shared control links, not individual derived endpoints.
    groups=[tuple(r['control_or_initial_ids']) for r in d.rows];base_x=np.column_stack([d.elapsed,d.action]);oof=np.zeros_like(d.target[tr])
    for g in sorted(set(groups[i] for i in tr)):
        keep=np.array([i for i in tr if groups[i]!=g]);hold=np.array([i for i in tr if groups[i]==g])
        if len(keep)<2:raise DataUnavailable('Cannot cross-fit independent initial/control groups')
        fn=dual_ridge(base_x[keep],d.target[keep],[d.rows[i] for i in keep]);oof[[list(tr).index(i) for i in hold]]=fn(base_x[hold])
    time_model=dual_ridge(base_x[tr],d.target[tr],[d.rows[i] for i in tr]);residual=d.target[tr]-oof
    init=dual_ridge(d.current[tr],residual,[d.rows[i] for i in tr]);time_pred=time_model(base_x[va]);pred=time_pred+init(d.current[va])
    perm=np.random.default_rng(69011).permutation(len(tr));shuffled=dual_ridge(d.current[tr][perm],residual,[d.rows[i] for i in tr]);sp=time_pred+shuffled(d.current[va])
    for name,fn in [('time_history',time_model),('initial_residual',init),('shuffled_initial',shuffled)]:save_npz(Path(out)/f'{name}.npz',**fn.parameters)
    save_npz(Path(out)/'validation.npz',model=pred,time_history=time_pred,shuffled=sp,target=d.target[va],mask=d.target_mask[va],crossfit_training_residual=residual)
    write_json(Path(out)/'result.json',{'model':metric(pred,d.target[va],d.target_mask[va]),'time_history':metric(time_pred,d.target[va],d.target_mask[va]),
        'shuffled_initial':metric(sp,d.target[va],d.target_mask[va]),'training_reference_crossfit':'leave shared initial/control group out',
        'future_inputs':False,'initial_scope':'observed initial programme; no target expression or target clock','science_status':'unvalidated'})

def local_regulon(prepared,annotation,out):
    b=ObservationBundle.load(prepared);x,genes=gene_data(prepared);members=annotation_members(annotation,'stable');tf=set(members.get('GO:0003700',set()))|set(members.get('GO:0000981',set()))
    tr=b.indices('train');va=b.indices('validation')
    if len(tr)<4:raise DataUnavailable('Too few independent training pools for exploratory coexpression network')
    # Only training variance selects genes/TFs. This is an exploratory network, not binding truth.
    var=x[tr].var(0);targets=np.argsort(-var)[:min(2000,int((var>1e-8).sum()))];tfidx=[i for i,g in enumerate(genes) if g in tf and var[i]>1e-8][:128]
    if not tfidx:raise DataUnavailable('No variable annotated TFs')
    z=(x-x[tr].mean(0))/np.maximum(x[tr].std(0),.001);corr=z[tr][:,tfidx].T@z[tr][:,targets]/len(tr)
    net=[];observed=[];predicted=[]
    for i,idx in enumerate(tfidx):
        selected=[j for j in np.argsort(-np.abs(corr[i])) if targets[j]!=idx][:20]
        if len(selected)<3:continue
        ids=targets[selected];sign=np.sign(corr[i,selected]);y=(z[:,ids]*sign).mean(1);fn=dual_ridge(z[tr,idx,None],y[tr,None],[b.rows[k] for k in tr])
        net.append({'tf':genes[idx],'targets':[genes[j] for j in ids],'correlations':corr[i,selected].tolist()});observed.append(y[va]);predicted.append(fn(z[va,idx,None])[:,0])
    save_npz(Path(out)/'validation.npz',prediction=np.array(predicted).T,target=np.array(observed).T)
    write_json(Path(out)/'members.json',net);write_json(Path(out)/'result.json',{'networks':len(net),'prediction':metric(np.array(predicted).T,np.array(observed).T),
        'fit_units':[b.rows[i]['biological_unit'] for i in tr],'network_kind':'training-only TF-target coexpression candidates',
        'causal_KO_labels':False,'binding_evidence':False,'limitations':['Very few training pools; unstable correlations expected','Not independent validation of causal TF activity']})

def public_generalisation(public,out):
    from .response import ResponseRegressor
    d=response_dataset(public,'endpoint');train=[i for i,r in enumerate(d.rows) if r['split']=='train' and not (d.action[i,0] and d.action[i,1])]
    val=[i for i,r in enumerate(d.rows) if r['split']=='validation' and d.action[i,0] and d.action[i,1]]
    if not val:raise DataUnavailable('No permitted double-mutant validation contrasts')
    # Fit arrays directly: controls remain in train only; original response contract remains unchanged.
    design=np.column_stack([d.current,d.action]);fn=dual_ridge(design[train],d.target[train],[d.rows[i] for i in train]);pred=fn(design[val]);additive=[]
    for i in val:
        effects=[]
        for target in (0,1):
            donors=[j for j in train if d.action[j,target]==1 and d.action[j,1-target]==0 and d.action[j,2]==d.action[i,2]]
            if not donors:raise DataUnavailable('Single-mutant additive baseline lacks matching condition')
            effects.append(d.target[donors].mean(0))
        additive.append(sum(effects))
    additive=np.array(additive);save_npz(Path(out)/'double_mutant.npz',prediction=pred,additive=additive,target=d.target[val],mask=d.target_mask[val],**fn.parameters)
    write_json(Path(out)/'result.json',{'double_mutant':{'model':metric(pred,d.target[val]),'single_effect_sum':metric(additive,d.target[val]),'zero_effect':metric(np.zeros_like(pred),d.target[val]),
        'train_ids':[d.rows[i]['observation_id'] for i in train],'validation_ids':[d.rows[i]['observation_id'] for i in val],
        'scope':'dedicated observed-input fit; no public pretrained model used; validation block controls never fitted'},
        'future_time_and_history':'separate task required; no assertion from double-mutant evaluation'})

def history_future(public,out):
    # Crucially load observations without an axis fitted using the future target.
    d=response_dataset(public,'transition',observed_bundle=Path(public)/'data/dauer')
    train=np.array([i for i,r in enumerate(d.rows) if r['split']=='train'])
    val=np.array([i for i,r in enumerate(d.rows) if r['split']=='validation'])
    history=np.array([r['history_days'] for r in d.rows]);held_history=max(history)
    reports={}
    for name,tr,va in [('history',train[history[train]!=held_history],val[history[val]==held_history]),
        ('future_time',train[d.elapsed[train]<max(d.elapsed)],val[d.elapsed[val]==max(d.elapsed)])]:
        if len(tr)<2 or not len(va):raise DataUnavailable('Insufficient permitted observations for '+name)
        base=np.column_stack([d.elapsed,d.action]);design=np.column_stack([base,d.current])
        model=dual_ridge(design[tr],d.target[tr],[d.rows[i] for i in tr]);time_model=dual_ridge(base[tr],d.target[tr],[d.rows[i] for i in tr])
        permutation=np.random.default_rng(82113).permutation(tr)
        shuffled=dual_ridge(np.column_stack([base[tr],d.current[permutation]]),d.target[tr],[d.rows[i] for i in tr])
        predictions={'model':model(design[va]),'time_history':time_model(base[va]),'initial_persistence':d.current[va],
            'shuffled_initial':shuffled(design[va])}
        for label,fn in [('model',model),('time_history',time_model),('shuffled_initial',shuffled)]:save_npz(Path(out)/name/(label+'.npz'),**fn.parameters)
        save_npz(Path(out)/name/'predictions.npz',**predictions,target=d.target[va],mask=d.target_mask[va])
        reports[name]={'metrics':{k:metrics(v,d.target[va],d.target_mask[va],[d.rows[i] for i in va]) for k,v in predictions.items()},
            'train_ids':[d.rows[i]['observation_id'] for i in tr],'validation_ids':[d.rows[i]['observation_id'] for i in va],
            'fitted_clock_used':False,'pretrained_state_used':False,'target_in_initial_inputs':False,
            'fit_elapsed_hours':sorted(set(d.elapsed[tr].tolist())),'held_elapsed_hours':sorted(set(d.elapsed[va].tolist()))}
    write_json(Path(out)/'result.json',reports)

def array_series(bundle,out):
    b=ObservationBundle.load(bundle);tr=b.indices('train');va=b.indices('validation')
    time=np.array([r['elapsed_hours_since_release'] for r in b.rows]);design=np.column_stack([time,time**2])
    pred=np.zeros_like(b.values[va]);mean=np.zeros_like(pred);mask=b.mask[va].copy();parameters=[]
    for j in range(len(b.feature_ids)):
        fit=tr[b.mask[tr,j]]
        if len(fit)<3:mask[:,j]=False;parameters.append(None);continue
        fn=dual_ridge(design[fit],b.values[fit,j,None],[b.rows[i] for i in fit]);pred[:,j]=fn(design[va])[:,0];mean[:,j]=b.values[fit,j].mean()
        parameters.append({k:v.tolist() for k,v in fn.parameters.items()})
    save_npz(Path(out)/'predictions.npz',model=pred,mean=mean,target=b.values[va],mask=mask,time=time[va])
    write_json(Path(out)/'parameters.json',parameters)
    write_json(Path(out)/'result.json',{'model':metrics(pred,b.values[va],mask,[b.rows[i] for i in va]),
        'train_mean':metrics(mean,b.values[va],mask,[b.rows[i] for i in va]),'fit_series':sorted({b.rows[i]['biological_unit'] for i in tr}),
        'held_series':sorted({b.rows[i]['biological_unit'] for i in va}),
        'coordinate':'observed hours, not a learned molecular clock','scope':'Dauer MTC series holdout; quadratic ridge wave baseline'})

def source_gaps(acquired,out):
    from .io import read_jsonl
    root=Path(acquired);meta=read_jsonl(root/'GSE303716/samples.jsonl')
    write_json(Path(out)/'result.json',{'SMAD2':{'status':'blocked_data','reason':'Only differential summary tables; no sample-level matrix in verified acquisition',
        'available_samples':len(meta),'raw_archive_relations':{r['accession']:r['fields'].get('Sample_relation',[]) for r in meta},
        'required':'Pinned reference, downloaded raw reads and audited count reprocessing; not fabricated from DEG statistics'},
        'GSE104616':{'status':'separate_descriptive_comparison','reason':'Joint RMA/ComBat matrix and technical replicates; see nhdf_comparison, not a train-only preprocessing claim',
            'sample_count':len(read_jsonl(root/'GSE104616/samples.jsonl'))},
        'expression_depth':{'status':'blocked_data','reason':'No matched expression-function cohort; group-level functional readout is separate'}})

def nhdf_comparison(acquired,out):
    import re
    from .io import read_jsonl
    from .pk2_data import verify_acquisition
    verify_acquisition(acquired,['GSE104616']);source=Path(acquired)/'GSE104616'
    meta=read_json(source/'log2_RMA/expression.json');samples={r['accession']:r['fields'] for r in read_jsonl(source/'samples.jsonl')}
    with np.load(source/'log2_RMA/expression.npz',allow_pickle=False) as a:x=a['values'].copy();mask=a['mask'].copy()
    groups={}
    for i,gsm in enumerate(meta['sample_accessions']):
        title=samples[gsm]['Sample_title'][0];match=re.search(r', (\d+) h_',title)
        if not match:continue
        if not any('technical replica' in t for t in samples[gsm].get('Sample_description',[])):raise DataUnavailable('NHDF replicate type requires review')
        groups.setdefault(int(match[1]),[]).append(i)
    hours=np.array(sorted(groups));values=[];masks=[]
    for hour in hours:
        idx=groups[int(hour)];mm=mask[idx];values.append((x[idx]*mm).sum(0)/np.maximum(mm.sum(0),1));masks.append(mm.any(0))
    y=np.array(values);mm=np.array(masks);tr=np.flatnonzero(~np.isin(hours,[12,30,42]));va=np.flatnonzero(np.isin(hours,[12,30,42]))
    design=np.column_stack([hours,hours**2]);rows=[{'study_family':'GSE104616','biological_unit':'single_series_not_independent_timepoints'} for _ in hours]
    fn=dual_ridge(design[tr],y[tr],[rows[i] for i in tr],target_mask=mm[tr]);pred=fn(design[va]);valid=mm[va]&np.isfinite(pred)
    save_npz(Path(out)/'parameters.npz',**fn.parameters);save_npz(Path(out)/'predictions.npz',prediction=pred,target=y[va],mask=valid,hours=hours[va])
    write_json(Path(out)/'probes.json',meta['feature_ids'])
    write_json(Path(out)/'result.json',{'scope':'descriptive probe-level time interpolation on published jointly RMA/ComBat-normalised data',
        'independent_biological_generalisation':False,'molecular_clock':False,'technical_arrays_aggregated':sum(len(v) for v in groups.values()),
        'timepoints':len(hours),'fit_hours':hours[tr].tolist(),'held_hours':hours[va].tolist(),'unsynchronised_arrays_excluded':True,
        'model':metrics(pred,y[va],valid,[rows[i] for i in va]),'training_mean':metrics(np.broadcast_to(y[tr].mean(0),pred.shape),y[va],valid,[rows[i] for i in va]),
        'limitations':['Common upstream normalisation includes held timepoints','No assertion of independent donor/culture replicates','Probe profiles are not mapped gene or TF activity']})

def composed_response(run,mode,out):
    from .state import StatePredictor
    from .pk2_numeric import query
    base=Path(run)/'tasks';bundle=ObservationBundle.load(base/'source_P-R/bundle')
    state=StatePredictor.load(base/'pretrain_P-R_42/model');prediction=query(state,bundle)['programme']
    positions={r['observation_id']:i for i,r in enumerate(bundle.rows)}
    d=response_dataset(base/'public_reuse',mode,observed_bundle=base/'public_reuse/data/dauer' if mode=='transition' else None)
    columns=[d.input_names.index(fid) for fid in bundle.feature_ids]
    indices=[positions[r['control_or_initial_ids'][0]] for r in d.rows]
    target=d.target[:,columns];mask=d.target_mask[:,columns];observed=d.current[:,columns];composed=prediction[indices]
    tr=np.array([i for i,r in enumerate(d.rows) if r['split']=='train']);va=np.array([i for i,r in enumerate(d.rows) if r['split']=='validation'])
    action=d.action if mode=='endpoint' else np.column_stack([d.action,d.elapsed])
    outputs={};reports={}
    for name,initial in [('observed',observed),('composed',composed),('no_initial',np.zeros_like(observed))]:
        x=np.column_stack([initial,action]);fn=dual_ridge(x[tr],target[tr],[d.rows[i] for i in tr],target_mask=mask[tr]);pred=fn(x[va])
        outputs[name]=pred;reports[name]=metrics(pred,target[va],mask[va]&np.isfinite(pred),[d.rows[i] for i in va])
        save_npz(Path(out)/(name+'.npz'),**fn.parameters)
    save_npz(Path(out)/'predictions.npz',**outputs,target=target[va],mask=mask[va],observed_initial=observed[va],composed_initial=composed[va])
    write_json(Path(out)/'result.json',{'mode':mode,'metrics':reports,'encoder_sha256':sha256(base/'pretrain_P-R_42/model/best.pt'),
        'source_bundle':bundle.fingerprint,'features':bundle.feature_ids,'target_expression_in_current':False,
        'fit_scope':'separately fitted matched Ridge heads on observed versus pretrained reconstructed initial state',
        'limitation':'development only; pretraining representation has no molecular clock; no causal simulation claim'})

def cross_platform_waves(run,out):
    base=Path(run)/'tasks';a=ObservationBundle.load(base/'data_GSE3169/bundle');r=ObservationBundle.load(base/'public_reuse/data/dauer')
    cols=[r.feature_ids.index(fid) for fid in a.feature_ids];ri=r.indices('train');ai=a.indices('train');av=a.indices('validation')
    rt=np.array([row['elapsed_hours_since_release'] for row in r.rows]);at=np.array([row['elapsed_hours_since_release'] for row in a.rows])
    prediction=np.full(a.values[av].shape,np.nan);local=np.full_like(prediction,np.nan);params={}
    for j,col in enumerate(cols):
        rfit=ri[r.mask[ri,col]];afit=ai[a.mask[ai,j]]
        if len(rfit)<3 or len(afit)<3:continue
        rm=float(r.values[rfit,col].mean());rs=max(float(r.values[rfit,col].std()),.001);am=float(a.values[afit,j].mean());ast=max(float(a.values[afit,j].std()),.001)
        x=lambda t:np.column_stack([t,t**2])
        fn=dual_ridge(x(rt[rfit]),((r.values[rfit,col]-rm)/rs)[:,None],[r.rows[i] for i in rfit]);prediction[:,j]=fn(x(at[av]))[:,0]*ast+am
        af=dual_ridge(x(at[afit]),a.values[afit,j,None],[a.rows[i] for i in afit]);local[:,j]=af(x(at[av]))[:,0]
        params[str(j)]={'rna':{k:v.tolist() for k,v in fn.parameters.items()},'array':{k:v.tolist() for k,v in af.parameters.items()},'array_mean':am,'array_scale':ast,'rna_mean':rm,'rna_scale':rs}
    mask=a.mask[av]&np.isfinite(prediction)&np.isfinite(local)&(at[av,None]<=max(rt[ri]))&(at[av,None]>=min(rt[ri]))
    save_npz(Path(out)/'predictions.npz',rna_transferred=prediction,array_local=local,target=a.values[av],mask=mask)
    write_json(Path(out)/'parameters.json',params)
    write_json(Path(out)/'result.json',{'rna_transfer':metrics(prediction,a.values[av],mask,[a.rows[i] for i in av]),
        'array_local':metrics(local,a.values[av],mask,[a.rows[i] for i in av]),'coordinate':'observed hours within common range, not shared biotime',
        'array_scaling':'array training series only','scope':'cross-study/platform programme wave diagnostic; different histories and protocols remain confounded',
        'source_train_only':True,'reserved_used':False})

def run_analysis(name,run,out,c):
    base=Path(run)/'tasks';private=Path(c['private_root']);public=base/'public_reuse'
    if name=='type_waves':type_waves(base/'input_core_celltypes/bundle',out)
    elif name=='endpoint':
        from .pk1_baselines import response_with_controls
        response_with_controls(public,'endpoint',out)
    elif name=='transition':transition_residual(public,out)
    elif name=='functional':
        from .pk1_baselines import bounded_functional
        bounded_functional(Path(__file__).resolve().parents[2]/'data/curated/functional_fig1b.json',out)
    elif name=='local_regulon':local_regulon(private/'prepared/core',Path(__file__).resolve().parents[2]/'knowledge/snapshots/ensembl_2026-10-03/nfurzeri_go.tsv',out)
    elif name=='public_generalisation':public_generalisation(public,out)
    elif name=='history_future':history_future(public,out)
    elif name=='array_series':array_series(base/'data_GSE3169/bundle',out)
    elif name=='source_gaps':source_gaps(c['acquire_run'],out)
    elif name=='nhdf_comparison':nhdf_comparison(c['acquire_run'],out)
    elif name.startswith('composed_'):composed_response(run,name.removeprefix('composed_'),out)
    elif name=='cross_platform_waves':cross_platform_waves(run,out)
    elif name=='public_regulon':
        from .pk1_assets import checked_member,verify_files
        import shutil
        root=Path(c['pk1_run']);m=read_json(root/'module_status.json')['modules']['public_regulon'];source=checked_member(root,m['artifact']);verify_files(source,m['files']);shutil.copytree(source,out)
        write_json(Path(out)/'reuse.json',{'kind':'verified_PK1_external_regulon_reuse','new_training':False,'source_inventory_hash':object_hash(m['files'])})
    elif name=='observation_stress':
        # Counts and cell-level checks are separate from log-scale programme masks.
        from .pk2_stress import observation_stress
        observation_stress(run,out,c)
    else:raise ValueError('Unknown analysis '+name)

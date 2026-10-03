"""PK1 adapters around the canonical state, reference and evaluation implementations."""
from pathlib import Path
import copy
import hashlib
import csv
import numpy as np
import torch
from .io import read_json,write_json,save_npz,sha256,object_hash
from .contracts import ObservationBundle
from .clock_reference import ExitReference
from .state import StateConfig,ProgrammeStateModel,StatePredictor,fit_state,corrupt
from .alpha_numeric import gene_data,metric,ridge_fit,ridge_predict
from .evaluate import evaluate_state
from .waves import WaveReference,describe_wave


class KillifishExitReference(ExitReference):
    def fit(self,expression,genes,rows,max_genes=512):
        from .admission import audit_internal_task
        audit_internal_task(rows,'clock')
        x=np.asarray(expression,float)
        if x.shape!=(len(rows),len(genes)) or not np.isfinite(x).all():raise ValueError('Invalid expression')
        early=[i for i,r in enumerate(rows) if r['split']=='train' and r['condition']=='Early Diapause']
        developing=[i for i,r in enumerate(rows) if r['split']=='train' and r['condition'] in {'Developing','Developing Day 7'}]
        if not early or not developing:raise ValueError('Missing train Early/D7 anchors')
        idx=early+developing
        readout=np.array([int(hashlib.sha256(g.encode()).hexdigest()[:8],16)%5==0 for g in genes])
        variance=x[idx].var(0);eligible=np.flatnonzero(~readout & (variance>1e-8))
        self.indices=eligible[np.argsort(-variance[eligible],kind='stable')[:max_genes]]
        if len(self.indices)<8:raise ValueError('Not identifiable: fewer than eight axis genes')
        self.mean=x[idx][:,self.indices].mean(0);self.scale=np.maximum(x[idx][:,self.indices].std(0),.001)
        z=(x[:,self.indices]-self.mean)/self.scale;self.origin=z[early].mean(0)
        delta=z[developing].mean(0)-self.origin
        if delta@delta<1e-8:raise ValueError('Not identifiable: coincident anchors')
        self.direction=delta/(delta@delta)
        self.reference_id='PK1_Early_to_D7_'+object_hash({'anchors':[rows[i]['observation_id'] for i in idx],'genes':[genes[i] for i in self.indices]})[:16]
        self.metadata={'reference_id':self.reference_id,'genes':list(genes),'axis_genes':[genes[i] for i in self.indices],
            'readout_genes':[g for g,keep in zip(genes,readout) if keep],
            'fit_ids':[rows[i]['observation_id'] for i in idx],'anchor_units':{'Early':len(early),'D7':len(developing)},
            'label_source':'reference_derived_not_legacy_M4','clipped':False,
            'limitations':['Very few independent anchor pools/embryos','Not independent clock truth, time or depth',
                'Late maintenance is excluded from clock supervision and Exit wave fitting',
                'Axis/readout genes disjoint but library normalisation and GO membership overlap remain']}
        return self


def clock_task(prepared,out,parent_reference=None):
    prepared=Path(prepared);out=Path(out);b=ObservationBundle.load(prepared);x,genes=gene_data(prepared)
    if any(r['split'] not in {'train','validation'} for r in b.rows):raise ValueError('Development data only')
    ref=KillifishExitReference.load(parent_reference) if parent_reference else KillifishExitReference().fit(x,genes,b.rows)
    ref.save(out/'reference');score=ref.predict(x,genes)
    mask=np.array(['late' not in r['condition'].lower() for r in b.rows])
    b.clock,b.clock_mask,b.clock_reference_id=score,mask,ref.metadata['reference_id'];b.save(out/'bundle')
    save_npz(out/'projection.npz',score=score,supervision_mask=mask)
    write_json(out/'result.json',{'execution_status':'completed','reference':ref.metadata,
        'reference_reused_from_whole_pool':parent_reference is not None,'training_clock_rows':int(mask[b.indices('train')].sum()),
        'reserved_used':False,'science_status':'unvalidated'})


def save_initial_weights(feature_ids,semantic_dim,config,out):
    cfg=StateConfig(**config);torch.manual_seed(cfg.seed)
    model=ProgrammeStateModel(len(feature_ids),cfg,torch.zeros(len(feature_ids),semantic_dim))
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'feature_ids':feature_ids,'model':model.state_dict()},out)


def shared_public_bundle(public,feature_ids,out):
    b=ObservationBundle.load(public)
    if any(r['origin']!='public' or r['split'] not in {'train','validation'} for r in b.rows):raise ValueError('Public development only')
    idx=[b.feature_ids.index(fid) for fid in feature_ids]
    b.values,b.mask,b.coverage=b.values[:,idx],b.mask[:,idx],b.coverage[:,idx]
    b.feature_ids=list(feature_ids);b.context={**b.context,'programme_definition_id':object_hash({'parent':b.context['programme_definition_id'],'features':feature_ids})}
    b.clock[:]=0;b.clock_mask[:]=False;b.clock_reference_id=None;b.save(out)


def semantic_cache(path,feature_ids,expected_kind):
    path=Path(path);m=read_json(path/'embeddings.json')
    if m['weight_kind']!=expected_kind or sha256(path/'embeddings.npz')!=m['arrays_sha256']:raise ValueError('Semantic identity mismatch')
    with np.load(path/'embeddings.npz',allow_pickle=False) as a:
        index={fid:i for i,fid in enumerate(a['feature_ids'].tolist())}
        vectors=a['vectors'][[index[fid] for fid in feature_ids]].copy()
    return vectors,m


def state_route(bundle_path,out,config,initial,route,semantic_dim,domain=None,base=None,pretrained=None):
    b=ObservationBundle.load(bundle_path);out=Path(out);cfg=StateConfig(**config['state_config'])
    sem=np.zeros((len(b.feature_ids),semantic_dim),dtype='float32');provenance={'ablation':'matched_capacity_zero','dimension':semantic_dim}
    if route in {'K2','K3','K5'}:
        sem,provenance=semantic_cache(domain,b.feature_ids,'domain_adapter')
        if route=='K5':
            order=np.random.default_rng(config['semantic_shuffle_seed']).permutation(len(sem))
            sem=sem[order];provenance={**provenance,'ablation':'fixed_shuffled_domain','permutation':order.tolist()}
    elif route=='K4':sem,provenance=semantic_cache(base,b.feature_ids,'frozen_base')
    elif route not in {'K0','K1','public'}:raise ValueError('Unknown route')
    if sem.shape[1]!=semantic_dim:raise ValueError('Semantic dimension differs from matched control')
    fit_state(b,out,steps=config['state_steps'],config=cfg,semantics=sem,semantic_provenance=provenance,
        pretrained=pretrained,initial_weights=initial,device='cpu')
    p=StatePredictor.load(out);q=copy.deepcopy(b);q.clock[:]=0;q.clock_mask[:]=False;pred=p.predict(q)
    arrays={'programme':pred['programme'],'latent':pred['latent'],'observed':b.values}
    if pred['clock'] is not None:arrays['clock']=pred['clock']
    save_npz(out/'clean_predictions.npz',**arrays)
    # Explicitly exercise serialization, without pretending this is independent biology.
    pred2=StatePredictor.load(out).predict(q)
    delta=float(np.max(np.abs(pred['programme']-pred2['programme'])))
    if delta>1e-6:raise RuntimeError('Numeric save/reload mismatch')
    write_json(out/'reload.json',{'status':'passed','max_abs_delta':delta,'new_process':False})
    tr,va=b.indices('train'),b.indices('validation');scaled=torch.tensor((b.values-p.mean)/p.scale)
    xx,mm,_,_=corrupt(scaled[tr],torch.tensor(b.mask[tr]),torch.tensor(b.coverage[tr]),cfg.corrupt_fraction,cfg.additive_noise,config['ridge_training_mask_seed'])
    coef,mu,sd=ridge_fit(np.column_stack([xx.numpy(),mm.numpy()]),b.values[tr])
    comparisons=[]
    for seed in config['evaluation_mask_seeds']:
        report=evaluate_state(out,b,out/f'evaluation_{seed}',split='validation',seed=seed)
        vx,vm,_,hidden=corrupt(scaled[va],torch.tensor(b.mask[va]),torch.tensor(b.coverage[va]),cfg.corrupt_fraction,cfg.additive_noise,seed)
        rp=ridge_predict(np.column_stack([vx.numpy(),vm.numpy()]),coef,mu,sd)
        with np.load(out/f'evaluation_{seed}/predictions.npz',allow_pickle=False) as a:
            model_metric=metric(a['programme'],b.values[va],hidden.numpy())
        comparisons.append({'mask_seed':seed,'model':model_metric,'ridge':metric(rp,b.values[va],hidden.numpy())})
        save_npz(out/f'ridge_{seed}.npz',prediction=rp,target=b.values[va],mask=hidden.numpy())
    noise=[]
    for missing in config['noise_levels']:
        vx,vm,vc,_=corrupt(scaled[va],torch.tensor(b.mask[va]),torch.tensor(b.coverage[va]),max(missing,1e-8),missing/4,config['noise_seed'])
        if missing==0:vx,vm,vc=scaled[va],torch.tensor(b.mask[va]),torch.tensor(b.coverage[va])
        query=b.subset(va);query.values=(vx.numpy()*p.scale+p.mean).astype('float32');query.mask=vm.numpy();query.coverage=vc.numpy();query.clock[:]=0;query.clock_mask[:]=False
        observed=p.predict(query)['programme']
        noise.append({'missing_fraction':missing,'model':metric(observed,b.values[va],b.mask[va]),
            'input_train_mean':metric(np.where(query.mask,query.values,p.mean),b.values[va],b.mask[va])})
    write_json(out/'noise_curve.json',noise)
    write_json(out/'result.json',{'route':route,'seed':cfg.seed,'evaluations':comparisons,
        'reference_clock':metric(pred['clock'][va],b.clock[va],b.clock_mask[va]) if pred['clock'] is not None else None,
        'dynamic_std_ratio_median':float(np.median(pred['programme'][va].std(0)/np.maximum(b.values[va].std(0),1e-8))),
        'independent_units':len(set(r['biological_unit'] for r in b.rows)),
        'extra_public_pretraining_compute':pretrained is not None,'reserved_used':False,'science_status':'unvalidated'})


def waves_task(prepared,clock_dir,state_dir,out,annotation):
    prepared=Path(prepared);clock_dir=Path(clock_dir);out=Path(out)
    b=ObservationBundle.load(clock_dir/'bundle');x,genes=gene_data(prepared)
    ref=KillifishExitReference.load(clock_dir/'reference');readout=set(ref.metadata['readout_genes'])
    tf=set()
    with Path(annotation).open(encoding='utf-8') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            if r['GO term accession'] in {'GO:0003700','GO:0000981'}:tf.add(r['Gene stable ID'])
    tr=np.array([i for i in b.indices('train') if b.rows[i]['reference_eligible']]);va=np.array([i for i in b.indices('validation') if b.rows[i]['reference_eligible']])
    if not len(va):raise ValueError('No eligible validation waves')
    q=b.subset(va);q.clock[:]=0;q.clock_mask[:]=False
    model_clock=StatePredictor.load(state_dir).predict(q)['clock'];context=object_hash(b.context)
    reports={}
    for kind,idx in [('gene',list(range(len(genes)))),('TF',[i for i,g in enumerate(genes) if g in tf]),('programme',None)]:
        ids=b.feature_ids if idx is None else [genes[i] for i in idx]
        if not ids:reports[kind]={'status':'unavailable_no_mapped_members'};continue
        values=b.values if idx is None else x[:,idx];mask=b.mask if idx is None else np.ones_like(values,bool)
        w=WaveReference.fit(b.clock,values,mask,ids,b.rows,b.clock_reference_id,context,feature_kind=kind,degree=1);w.save(out/kind)
        r=w.residual(b.clock[va],values[va],mask[va],b.clock_reference_id,context)
        chained=w.residual(model_clock,values[va],mask[va],b.clock_reference_id,context)
        common=r['residual_mask'] & chained['residual_mask'];baseline=np.broadcast_to(values[tr].mean(0),values[va].shape)
        save_npz(out/kind/'validation.npz',observed=values[va],reference_expected=r['expected'],reference_residual=r['residual'],
            chained_expected=chained['expected'],chained_residual=chained['residual'],reference_mask=r['residual_mask'],chained_mask=chained['residual_mask'],common_mask=common,baseline=baseline)
        reports[kind]={'reference':metric(r['expected'],values[va],r['residual_mask']),
            'chained_same_support':metric(chained['expected'],values[va],common),'reference_same_support':metric(r['expected'],values[va],common),
            'baseline_same_support':metric(baseline,values[va],common),'reference_coverage':float(r['residual_mask'].mean()),
            'chained_coverage':float(chained['residual_mask'].mean()),'fine_peak_timing':'not_identifiable_with_linear_reference'}
        if kind=='gene':
            read_mask=r['residual_mask'] & np.array([g in readout for g in ids])[None]
            reports[kind]['preassigned_non_axis_readout']=metric(r['expected'],values[va],read_mask)
        grid=np.linspace(float(b.clock[tr].min()),float(b.clock[tr].max()),50)
        curves=w.predict(grid,b.clock_reference_id,context,allow_extrapolation=True)['expected']
        save_npz(out/kind/'curves.npz',coordinate=grid,expected=curves)
        write_json(out/kind/'shapes.json',[{'feature_id':fid,**describe_wave(grid,curves[:,j])} for j,fid in enumerate(ids)])
    write_json(out/'result.json',{'features':reports,'encoder_sha256':sha256(Path(state_dir)/'best.pt'),
        'degree':1,'Late_maintenance_in_wave_fit':False,'reserved_used':False,'science_status':'unvalidated'})

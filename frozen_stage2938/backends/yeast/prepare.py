"""Remote-only guarded LOBO fit. No held-endpoint predictions or scoring here."""
from pathlib import Path
from collections import defaultdict, Counter
import json, hashlib, gzip, traceback
import numpy as np

O=Path('/out');O.mkdir(exist_ok=True)
G=3598
def read(p):return json.loads(Path(p).read_text())
def pin(p):
 b=Path(p).read_bytes();return dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def put(n,v,compact=False):
 (O/n).write_text(json.dumps(v,indent=None if compact else 2,allow_nan=False)+'\n')

completed=[]
def main():
 P=read('/code/protocol.json')
 assert pin('/code/prepare.py')['sha256']==P['code_sha256']['prepare.py']
 A=read('/input/assets.json');assert set(A)=={P['metadata_asset']}
 mp=Path(A[P['metadata_asset']]['container_path']);assert pin(mp)==P['metadata_pin']
 M=read('/input/artifacts.json');assert len(M)==len(P['references'])==7
 paths={}
 for k,r in P['references'].items():
  hits=[v for v in M.values() if v['ref']==r['ref']];assert len(hits)==1
  paths[k]=Path(hits[0]['container_path']);assert pin(paths[k])==r['pin']
 before={k:pin(v) for k,v in paths.items()}
 original={};sid=None
 with gzip.open(mp,'rt',encoding='utf-8-sig') as inp:
  for line in inp:
   line=line.rstrip('\r\n')
   if line.startswith('^'):
    sid=None
    if line.startswith('^SAMPLE = '):sid=line.split(' = ',1)[1];original[sid]=defaultdict(list)
   elif sid and line.startswith('!Sample_') and ' = ' in line:
    key,val=line.split(' = ',1);original[sid][key].append(val)
 assert len(original)==97
 with np.load(paths['train'],allow_pickle=False) as z:
  assert set(z.files)=={'x','y'};tx=z['x'].copy();ty=z['y'].copy()
 with np.load(paths['query'],allow_pickle=False) as z:
  assert set(z.files)=={'x'};qx=z['x'].copy()
 with np.load(paths['truth'],allow_pickle=False) as z:
  assert set(z.files)=={'y'};qy=z['y'].copy()
 assert tx.shape==(9,7197) and qx.shape==(4,7197) and ty.shape==(9,G) and qy.shape==(4,G)
 rows=read(paths['train_rows'])+read(paths['rows']);assert len(rows)==13
 x=np.concatenate([tx,qx]);y=np.concatenate([ty,qy])
 assert all(np.isfinite(v).all() and ((v>=0)&(v<=1)).all() for v in [x,y])
 schema=read(paths['schema']);genes=[s[len('future_rank__'):] for s in schema['targets']]
 assert len(genes)==len(set(genes))==G
 assert schema['targets']==['future_rank__'+g for g in genes]
 assert schema['features']==['current24_rank__'+g for g in genes]+['initial0_rank__'+g for g in genes]+['requested_endpoint_is336h']
 assert np.isin(x[:,-1],[0,1]).all()
 with np.load(paths['original39'],allow_pickle=False) as z:
  means=z['population_mean_rank'].copy();assays=z['original_assay_rank'].copy()
  gsms=z['original_future_GSM'].tolist();ai=z['assay_target_index'].copy();tids=z['target_ids'].tolist()
 assert means.shape==(13,G) and assays.shape==(39,G) and len(tids)==len(set(tids))==13 and len(gsms)==len(set(gsms))==39
 assert all(np.isfinite(v).all() and ((v>=0)&(v<=1)).all() for v in [means,assays])
 assert len({r['case_id'] for r in rows})==len({r['target_id'] for r in rows})==13
 assert set(tids)=={r['target_id'] for r in rows}
 backgrounds=sorted({r['background'] for r in rows})
 assert backgrounds==['ARP42','ASP1','HHT2','IEC1','IES2','PHT1','SMT0']
 assert Counter(r['future_hours'] for r in rows)=={168.:7,336.:6}
 groups={k:set() for k in ['initial','current','future']};linksets=[];members=[]
 for j,r in enumerate(rows):
  bg=r['background'];hr=r['future_hours']
  assert r['current_hours']==24 and r['elapsed_hours']==hr-24 and int(x[j,-1])==int(hr==336)
  assert r['verified_individual_pairs']==0 and r['independent_cultures'] is None
  assert not r['endpoint_expression_used_as_current_input'] and not r['serial_culture_or_individual_pairing_claim']
  assert r['initial0_expression_available_at_current24_request'] and r['earlier_observed_context_hours']==0
  g0,g24,gf=r['aggregate_initial0_GSMs'],r['aggregate_current_GSMs'],r['aggregate_future_GSMs']
  assert len(g0)==len(g24)==len(gf)==3 and len(set(g0+g24+gf))==9
  target=tids.index(r['target_id']);ix=np.flatnonzero(ai==target)
  assert len(ix)==3 and [gsms[int(k)] for k in ix]==gf
  assert np.array_equal(assays[ix].mean(axis=0),means[target]) and np.array_equal(means[target],y[j])
  links=set(r['link_ids']);details={}
  for group,title in [(g0,'T0'),(g24,'T1D'),(gf,'T1W' if hr==168 else 'T2W')]:
   for g in group:
    f=dict(original[g]);assert r['original_member_metadata'][g]==f
    assert f['!Sample_organism_ch1']==['Schizosaccharomyces pombe'] and f['!Sample_library_strategy']==['RNA-Seq']
    assert f['!Sample_title'][0].strip().startswith(bg+'_'+title+'_')
    links.add(g);links.update(f.get('!Sample_relation',[]))
    details[g]=dict(title=f['!Sample_title'],relations=f.get('!Sample_relation',[]))
  for key,gs in [('initial',g0),('current',g24),('future',gf)]:groups[key].update(gs)
  linksets.append(links)
  members.append(dict(case_id=r['case_id'],target_id=r['target_id'],background=bg,endpoint_hours=hr,initial0_GSMs=g0,current24_GSMs=g24,future_GSMs=gf,original_members=details))
 assert [len(groups[k]) for k in ['initial','current','future']]==[21,21,39]
 assert not(groups['initial']&groups['current'] or groups['initial']&groups['future'] or groups['current']&groups['future'])
 for bg in backgrounds:
  ix=[i for i,r in enumerate(rows) if r['background']==bg]
  assert len(ix)==(1 if bg=='IES2' else 2)
  for j in ix:
   assert np.array_equal(x[j,:2*G],x[ix[0],:2*G])
   assert rows[j]['aggregate_initial0_GSMs']==rows[ix[0]]['aggregate_initial0_GSMs']
   assert rows[j]['aggregate_current_GSMs']==rows[ix[0]]['aggregate_current_GSMs']
 assert {(r['background'],r['future_hours']) for r in rows}=={(bg,hr) for bg in backgrounds for hr in [168.,336.] if (bg,hr)!=('IES2',336.)}

 # Complete EVERY relationship/design guard before fitting any response slope.
 designs=[];folds=[]
 for j,r in enumerate(rows):
  bg=r['background'];hr=r['future_hours']
  other=[i for i,s in enumerate(rows) if s['background']!=bg]
  held=[i for i,s in enumerate(rows) if s['background']==bg]
  fit=[i for i in other if rows[i]['future_hours']==hr]
  assert len(fit)==(6 if hr==168 else 5) and len({rows[i]['background'] for i in fit})==len(fit)
  held_links=set().union(*(linksets[i] for i in held));train_links=set().union(*(linksets[i] for i in other))
  assert not(held_links&train_links),'Original identifiers cross a held-background fold; primary must repair, not drop the links.'
  xx=x[fit,:G];x0=x[fit,G:2*G]
  mx=xx.mean(axis=0);m0=x0.mean(axis=0);dx=xx-mx;d0=x0-m0
  cxx=float(np.sum(dx*dx));assert np.isfinite(cxx) and cxx>1e-12
  q=float(np.sum(dx*d0))/cxx;z0=d0-q*dx;energy=float(np.sum(z0*z0))
  design=np.column_stack([dx.ravel(),d0.ravel()]);sv=np.linalg.svd(design,compute_uv=False)
  assert np.linalg.matrix_rank(design)==2 and np.isfinite(sv).all() and sv[-1]>0
  assert np.isfinite(q) and np.isfinite(energy) and energy>1e-12
  designs.append(dict(mx=mx,m0=m0,dx=dx,z0=z0,cxx=cxx,q=q,energy=energy,condition=float(sv[0]/sv[1])))
  folds.append(dict(model_index=j,case_id=r['case_id'],target_id=r['target_id'],held_background=bg,endpoint_hours=hr,
   fit_row_indices=fit,all_horizon_training_row_indices=other,held_row_indices=held,
   fit_case_ids=[rows[i]['case_id'] for i in fit],excluded_case_ids=[rows[i]['case_id'] for i in held],
   training_backgrounds=[rows[i]['background'] for i in fit],all_horizon_training_rows=len(other),
   cross_partition_explicit_links=[],numeric_design_rank=2,design_condition_number=float(sv[0]/sv[1])))

 params={k:[] for k in ['mean_current','mean_initial','mean_future','pooled_mean','initial_projection','current_slope','initial_slope']}
 coefficients=[]
 for j,(f,d) in enumerate(zip(folds,designs)):
  yy=y[f['fit_row_indices']];my=yy.mean(axis=0);dy=yy-my
  a=float(np.sum(d['dx']*dy))/d['cxx']
  cross=float(np.sum(d['z0']*(dy-a*d['dx'])));b=cross/d['energy']
  assert np.isfinite([a,b,cross]).all()
  values=dict(mean_current=d['mx'],mean_initial=d['m0'],mean_future=my,
   pooled_mean=y[f['all_horizon_training_row_indices']].mean(axis=0),
   initial_projection=d['q'],current_slope=a,initial_slope=b)
  for key,val in values.items():params[key].append(val)
  coefficients.append(dict(model_index=j,held_background=f['held_background'],endpoint_hours=f['endpoint_hours'],training_backgrounds=len(f['fit_row_indices']),
   q=d['q'],a=a,b=b,b_sign=int(np.sign(b)),Cxx=d['cxx'],initial_residual_energy=d['energy'],initial_response_crossproduct=cross,
   orthogonality_inner_product=float(np.sum(d['dx']*d['z0'])),design_condition_number=d['condition']))
  completed.append(j)
 assert len(completed)==13
 arrays={k:np.asarray(v,dtype=np.float64) for k,v in params.items()}
 for k,v in arrays.items():
  assert v.shape==((13,G) if k in ['mean_current','mean_initial','mean_future','pooled_mean'] else (13,)) and np.isfinite(v).all()
 np.savez_compressed(O/'lobo_parameters2142.npz',**arrays)
 np.savez_compressed(O/'lobo_inputs_only2142.npz',x=x,model_index=np.arange(13,dtype=np.int64))
 requests=[dict(model_index=j,case_id=r['case_id'],target_id=r['target_id'],background=r['background'],current_hours=24,endpoint_hours=r['future_hours'],
  initial0_GSMs=r['aggregate_initial0_GSMs'],current24_GSMs=r['aggregate_current_GSMs'],future_GSMs=r['aggregate_future_GSMs']) for j,r in enumerate(rows)]
 out_schema=dict(features=schema['features'],targets=schema['targets'],axis_pin=schema['axis_pin'],current_hours=24,initial_context_hours=0,
  preprocessing=schema['preprocessing'],endpoint_hours=[168,336],input_dependency='observed_population_covariates',
  historical_partition_roles='Original9fit+4query union; each new fold excludes its background across all horizons. Adaptive retrospective, not renewed blindness.',
  gene_specific_slopes=False,serial_pairing=False,independent_cultures=None,study_count=1)
 put('lobo_schema2142.json',out_schema)
 put('lobo_requests2142.json',requests,True);put('lobo_fold_manifest2142.json',folds,True)
 put('lobo_coefficients2142.json',dict(scope='Actually fitted LOBO training coefficients, not old native weights or independent biology.',rows=coefficients),True)
 put('lobo_original_member_audit2142.json',members,True)
 dependency=dict(status='original_dependency_verified',original97_metadata_reconciled=True,rows=13,backgrounds=7,
  original_initial0_libraries=21,original_current24_libraries=21,original_future_libraries=39,related_studies=1,independent_cultures=None,verified_serial_pairs=0,
  future_library_IDs_disjoint_from_inputs=True,distinct_cultures_proven=False,cross_background_explicit_links=[],missing_endpoint='IES2 at336h not fabricated',
  held_background_excluded_all_horizons=True,fold_specific_means_projection_and_response=True,preprocessing='Fixed per-original-profile ranks inherited; no fold-spanning fitted standardization.',
  historical_truth_pool_mounted_for_retrospective_fit=True,held_truth_excluded_from_each_fold_coefficients=True,physical_absence_of_held_labels_from_fit_container_claim=False,
  target_population_means_and39_GSM_members_exact=True,all_inputs_unchanged=True,scientific_approval=False)
 assert {k:pin(v) for k,v in paths.items()}==before and pin(mp)==P['metadata_pin']
 put('lobo_dependency2142.json',dependency)
 model=dict(schema_version=1,algorithm='shared_scalar_current_and_nested_initial_population_LOBO',training_rows=13,background_folds=7,eligible_horizon_models=13,
  formula='current=muY+a*(x24-mu24);nested=current+b*((x0-mu0)-q*(x24-mu24));clock=muY;pooled=all-horizon training mean;P=x24. RAW and finalCLIP[0,1] at prediction.',
  parameters=pin(O/'lobo_parameters2142.npz'),schema=pin(O/'lobo_schema2142.json'),requests=pin(O/'lobo_requests2142.json'),inputs=pin(O/'lobo_inputs_only2142.npz'),fold_manifest=pin(O/'lobo_fold_manifest2142.json'),
  fit_completed=True,predictions_created=False,query_scores_computed=False,formal_native_experiment=False,native_fit_predict_calls=0,
  primary_summary='background-balanced means over available horizons; report counts and endpoint-weighted/horizon-stratified summaries too',
  scientific_approval=False,default_promotion=False)
 put('lobo_model2142.json',model)
 outputs={p.name:pin(p) for p in O.iterdir() if p.is_file()}
 put('lobo_fit_ready2142.json',dict(status='fit_complete_not_predicted',all_inputs_unchanged=True,original_dependencies_verified=True,
  actually_fitted_eligible_horizon_models=13,current_slopes=13,additional_initial_slopes=13,background_folds=7,native_fit_predict_calls=0,new_formal_experiments=0,
  predictions=0,query_scores=0,outputs=outputs,scientific_approval=False,
  next_step='Inspect readiness/dependency/coefficient reports; then separately predict from saved models and input-only requests without truth mounts or fitting; score in a later isolated step.'))
 put('metadata.json',dict(status='matrix_ready',task_ready=False,readiness='lobo_fit_ready2142.json',fitted_models=13,predictions=0,scientific_approval=False))

try:main()
except Exception as e:
 put('lobo_fit_ready2142.json',dict(status='not_ready',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc()[-6000:],
  completed_coefficient_models_before_failure=len(completed),predictions=0,query_scores=0,scientific_approval=False,
  next_step='Inspect and repair the concrete failed guard; no silent model, grouping or pairing changes. Do not use partial outputs as verified.'))
 put('metadata.json',dict(status='metadata_ready',task_ready=False,readiness='lobo_fit_ready2142.json',scientific_approval=False))

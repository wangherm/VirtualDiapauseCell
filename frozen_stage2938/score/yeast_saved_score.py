"""Remote-only scoring of immutable LOBO predictions against original13/39 targets. No fit."""
from pathlib import Path
from collections import Counter
import argparse,json,hashlib
import numpy as np
G=3598
def read(p):return json.loads(Path(p).read_text())
def pin(p):
 b=Path(p).read_bytes();return dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def put(n,v,compact=False):
 (O/n).write_text(json.dumps(v,indent=None if compact else 2,allow_nan=False)+'\n')
def score(p,y):
 d=p-y;return dict(mse=float(np.mean(d*d)),mae=float(np.mean(np.abs(d))))
def cmp(a,b):
 return dict(delta=a-b,relative_change=None if b==0 else (a-b)/b,
  status='equivalent' if np.isclose(a,b,rtol=1e-12,atol=1e-14) else 'gain' if a<b else 'harm')

def main():
 global O
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('config','input-map','output-dir'):
  p.add_argument('--'+name,type=Path,required=True)
 args=p.parse_args()
 O=args.output_dir
 P=read(args.config)
 paths={k:Path(v) for k,v in read(args.input_map).items()}
 if set(paths)!=set(P['references']) or len(paths)!=8:
  raise ValueError('Wrong yeast scoring input roles')
 for key,path in paths.items():
  if pin(path)!=P['references'][key]['pin']:
   raise ValueError('Yeast score input pin changed')
 before={k:pin(v) for k,v in paths.items()}
 ready=read(paths['ready']);manifest=read(paths['manifest']);dep=read(paths['fit_dependency']);coef=read(paths['coefficients'])['rows']
 assert ready['status']=='predictions_frozen_not_scored' and ready['all_inputs_unchanged']
 assert ready['population_requests']==13 and ready['backgrounds']==7 and ready['heads']==5 and ready['genes']==G
 assert ready['saved_model_parameters_reloaded'] and not ready['truth_value_arrays_mounted']
 assert ready['fit_calls']==ready['query_scores']==ready['native_fit_predict_calls']==ready['new_formal_experiments']==0
 assert manifest['operation']=='saved_parameter_prediction_only' and manifest['model_reloaded_from_saved_files']
 assert manifest['fit_calls']==0 and manifest['target_value_arrays_mounted'] is False
 for k in ['prediction','requests','manifest']:
  assert ready['outputs'][Path(P['references'][k]['ref']['path']).name]==before[k]
 assert manifest['prediction_file']==before['prediction'] and manifest['request_file']==before['requests']
 assert dep['status']=='original_dependency_verified' and dep['held_background_excluded_all_horizons']
 assert dep['original97_metadata_reconciled'] and dep['target_population_means_and39_GSM_members_exact']
 assert dep['independent_cultures'] is None and dep['verified_serial_pairs']==0 and dep['related_studies']==1
 assert dep['cross_background_explicit_links']==[]
 req=read(paths['requests']);assert len(req)==13
 assert len(coef)==13 and [r['model_index'] for r in coef]==list(range(13))
 with np.load(paths['prediction'],allow_pickle=False) as z:
  assert set(z.files)=={'raw','clipped','head_names','case_ids','target_ids','model_index'}
  raw=z['raw'].copy();clipped=z['clipped'].copy();heads=z['head_names'].tolist()
  cases=z['case_ids'].tolist();targets=z['target_ids'].tolist();mi=z['model_index'].copy()
 assert heads==['current24','nested_initial0','horizon_clock','pooled_train_mean','own24h_persistence']
 assert raw.shape==clipped.shape==(5,13,G) and np.isfinite(raw).all() and np.array_equal(np.clip(raw,0,1),clipped)
 assert np.array_equal(mi,np.arange(13)) and [r['case_id'] for r in req]==cases and [r['target_id'] for r in req]==targets
 assert len(set(cases))==len(set(targets))==13
 with np.load(paths['truth'],allow_pickle=False) as z:
  y=z['population_mean_rank'].copy();ay=z['original_assay_rank'].copy();gsms=z['original_future_GSM'].tolist()
  ai=z['assay_target_index'].copy();tids=z['target_ids'].tolist()
 assert y.shape==(13,G) and ay.shape==(39,G) and len(set(gsms))==39 and len(set(tids))==13
 assert all(np.isfinite(v).all() and ((v>=0)&(v<=1)).all() for v in [y,ay])
 assert set(tids)==set(targets) and ai.shape==(39,)
 assert Counter(r['endpoint_hours'] for r in req)=={168.:7,336.:6}
 assert Counter(r['background'] for r in req)=={'ARP42':2,'ASP1':2,'HHT2':2,'IEC1':2,'IES2':1,'PHT1':2,'SMT0':2}
 # Original frozen mean and persistence controls are reused; no old estimator is fitted again.
 with np.load(paths['old_baselines'],allow_pickle=False) as z:
  oldtargets=z['target_ids'].tolist();oldcases=z['request_ids'].tolist()
  oldclock=z['fold_later_mean'].copy();oldP=z['own24h_persistence'].copy()
 assert set(oldtargets)==set(targets) and len(oldtargets)==13
 control_parity={'horizon_clock_max_absolute_error':0.,'own24h_persistence_max_absolute_error':0.}
 pop=[];libraries=[]
 views={h+'_'+v:(raw if v=='RAW' else clipped)[k] for k,h in enumerate(heads) for v in ['RAW','CLIP']}
 for j,r in enumerate(req):
  tj=tids.index(r['target_id']);ix=np.flatnonzero(ai==tj)
  assert len(ix)==3 and [gsms[int(k)] for k in ix]==r['future_GSMs']
  assert np.array_equal(ay[ix].mean(axis=0),y[tj])
  assert len(r['initial0_GSMs'])==len(r['current24_GSMs'])==3
  assert len(set(r['initial0_GSMs']+r['current24_GSMs']+r['future_GSMs']))==9
  assert coef[j]['held_background']==r['background'] and coef[j]['endpoint_hours']==r['endpoint_hours']
  oj=oldtargets.index(r['target_id']);assert oldcases[oj]==r['case_id']
  for name,k,old in [('horizon_clock',2,oldclock),('own24h_persistence',4,oldP)]:
   delta=float(np.max(np.abs(raw[k,j]-old[oj])))
   control_parity[name+'_max_absolute_error']=max(control_parity[name+'_max_absolute_error'],delta)
   assert delta<=1e-13,'Historical equivalent control changed; repair before interpretation.'
  info=dict(case_id=r['case_id'],target_id=r['target_id'],background=r['background'],endpoint_hours=r['endpoint_hours'],
   initial0_GSMs=r['initial0_GSMs'],current24_GSMs=r['current24_GSMs'],future_GSMs=r['future_GSMs'])
  pop.append(dict(**info,scores={h:score(p[j],y[tj]) for h,p in views.items()}))
  for k in ix:
   libraries.append(dict(case_id=r['case_id'],target_id=r['target_id'],background=r['background'],endpoint_hours=r['endpoint_hours'],
    future_GSM=gsms[int(k)],scores={h:score(p[j],ay[k]) for h,p in views.items()}))
 assert len(libraries)==39 and len({r['future_GSM'] for r in libraries})==39
 backgrounds=sorted({r['background'] for r in req})
 def macro(rows):return {h:{m:float(np.mean([r['scores'][h][m] for r in rows])) for m in ['mse','mae']} for h in views}
 def group_rows(rows):
  return [dict(background=bg,observed_endpoint_hours=sorted({r['endpoint_hours'] for r in rows if r['background']==bg}),
   rows=sum(r['background']==bg for r in rows),scores=macro([r for r in rows if r['background']==bg])) for bg in backgrounds]
 pg=group_rows(pop);lg=group_rows(libraries)
 def comps(rows):
  summary=macro(rows);out={}
  for mode in ['RAW','CLIP']:
   for base in ['current24','horizon_clock','pooled_train_mean','own24h_persistence']:
    bk=base+'_'+mode;nk='nested_initial0_'+mode;out[bk]={}
    for metric in ['mse','mae']:
     signs=Counter(cmp(r['scores'][nk][metric],r['scores'][bk][metric])['status'] for r in rows)
     out[bk][metric]=dict(**cmp(summary[nk][metric],summary[bk][metric]),gain=signs['gain'],harm=signs['harm'],equivalent=signs['equivalent'])
  return out
 columns=[dict(head=h,metric=m) for h in views for m in ['mse','mae']]
 def table(rows):
  return dict(format='ordered_error_columns_v1',score_columns=columns,rows=[dict(**{k:v for k,v in r.items() if k!='scores'},scores=[r['scores'][c['head']][c['metric']] for c in columns]) for r in rows])
 put('lobo_all13_population_scores2144.json',table(pop),True)
 for part in range(3):put('lobo_original39_library_scores2144_part'+str(part+1)+'.json',table(libraries[part*13:(part+1)*13]),True)
 put('lobo_background_scores2144.json',dict(population=table(pg),libraries=table(lg)),True)
 put('lobo_horizon_scores2144.json',{str(hr):dict(population_count=sum(r['endpoint_hours']==hr for r in pop),library_count=sum(r['endpoint_hours']==hr for r in libraries),
  population=macro([r for r in pop if r['endpoint_hours']==hr]),libraries=macro([r for r in libraries if r['endpoint_hours']==hr]),
  population_comparisons=comps([r for r in pop if r['endpoint_hours']==hr])) for hr in [168,336]},True)
 # Primary equal-background aggregation is distinct from equal-endpoint aggregation.
 focus=dict(scope='Adaptive same-study population prediction; no independent cultures, serial trajectories, causal memory or function labels.',
  counts=dict(backgrounds=7,population_endpoints=13,future_libraries=39,distinct_initial0_libraries=21,distinct_current24_libraries=21,related_studies=1,verified_serial_pairs=0,independent_cultures=None),
  primary_background_balanced_population=macro(pg),secondary_endpoint_weighted_population=macro(pop),
  background_balanced_libraries=macro(lg),endpoint_weighted_libraries=macro(libraries),
  primary_background_comparisons=comps(pg),population_comparisons=comps(pop),library_comparisons=comps(libraries),
  per_background_nested_minus_current={r['background']:{mode:{m:cmp(r['scores']['nested_initial0_'+mode][m],r['scores']['current24_'+mode][m]) for m in ['mse','mae']} for mode in ['RAW','CLIP']} for r in pg},
  original_frozen_control_parity=control_parity,
  primary_weighting='Mean errors within each background over its observed horizons, then equal mean of7 backgrounds. IES2 has168h only, other backgrounds168/336h. Horizon-specific and endpoint-weighted summaries separate.',
  coefficient_stability={str(hr):dict(training_models=sum(r['endpoint_hours']==hr for r in coef),
   b_positive=sum(r['endpoint_hours']==hr and r['b']>0 for r in coef),b_negative=sum(r['endpoint_hours']==hr and r['b']<0 for r in coef),
   b_min=min(r['b'] for r in coef if r['endpoint_hours']==hr),b_max=max(r['b'] for r in coef if r['endpoint_hours']==hr)) for hr in [168,336]},
  no_IID_confidence_interval=True,numerical_equivalence=dict(rtol=1e-12,atol=1e-14),new_fit_calls=0,scientific_approval=False)
 put('lobo_score_focus2144.json',focus,True)
 assert {k:pin(v) for k,v in paths.items()}==before
 outputs={p.name:pin(p) for p in O.iterdir() if p.is_file()}
 put('lobo_scoring_ready2144.json',dict(status='scoring_complete',all_inputs_unchanged=True,outputs=outputs,
  original13target39library_mapping_verified=True,frozen_control_parity=control_parity,prediction_arrays_modified=False,
  new_fit_calls=0,new_prediction_calls=0,new_native_fit_predict_calls=0,new_formal_experiments=0,scientific_approval=False,
  next_step='Read actual scores before claims. Full ordered score columns and all cases retained; no winner or default promotion is automatic.'))
 put('metadata.json',dict(status='metadata_ready',task_ready=False,readiness='lobo_scoring_ready2144.json',new_fit_calls=0,scientific_approval=False))

 result=pin(O/'lobo_score_focus2144.json')
 put('adapter_receipt.json',dict(status='completed',phase='score',task='yeast_late_relative_rna',
  score_summary=result,exact_historical_score_summary_match=result==P['reference_score'],
  population_endpoints=13,future_libraries=39,background_groups=7,
  fit_calls=0,prediction_calls=0,scientific_approval=False))

if __name__=='__main__':
 main()

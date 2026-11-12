"""Remote-only prediction from pinned LOBO models and input-only requests; no fit or truth."""
from pathlib import Path
from collections import Counter
import argparse,json,hashlib
import numpy as np
G=3598
def read(p):return json.loads(Path(p).read_text())
def pin(p):
 b=Path(p).read_bytes();return dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def put(n,v):return (O/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

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
  raise ValueError('Wrong saved yeast inputs')
 for key,path in paths.items():
  if pin(path)!=P['references'][key]['pin']:
   raise ValueError('Yeast input pin changed')
 before={k:pin(v) for k,v in paths.items()}
 ready=read(paths['ready']);dep=read(paths['dependency']);model=read(paths['model'])
 assert ready['status']=='fit_complete_not_predicted' and ready['all_inputs_unchanged'] and ready['original_dependencies_verified']
 assert ready['actually_fitted_eligible_horizon_models']==13 and ready['background_folds']==7
 assert ready['predictions']==ready['query_scores']==ready['native_fit_predict_calls']==ready['new_formal_experiments']==0
 for k in ['parameters','schema','requests','inputs','fold_manifest']:
  assert ready['outputs'][P['references'][k]['ref']['path']]==before[k]
  assert model[k]==before[k]
 assert ready['outputs'][P['references']['model']['ref']['path']]==before['model']
 assert ready['outputs'][P['references']['dependency']['ref']['path']]==before['dependency']
 assert dep['status']=='original_dependency_verified' and dep['original97_metadata_reconciled']
 assert dep['rows']==13 and dep['backgrounds']==7 and dep['related_studies']==1
 assert [dep[k] for k in ['original_initial0_libraries','original_current24_libraries','original_future_libraries']]==[21,21,39]
 assert dep['independent_cultures'] is None and dep['verified_serial_pairs']==0 and not dep['distinct_cultures_proven']
 assert dep['cross_background_explicit_links']==[] and dep['held_background_excluded_all_horizons']
 assert dep['fold_specific_means_projection_and_response'] and dep['target_population_means_and39_GSM_members_exact'] and dep['all_inputs_unchanged']
 assert model['algorithm']=='shared_scalar_current_and_nested_initial_population_LOBO'
 assert model['fit_completed'] and not model['predictions_created'] and not model['query_scores_computed']
 assert model['eligible_horizon_models']==13 and model['background_folds']==7
 schema=read(paths['schema']);req=read(paths['requests']);fold=read(paths['fold_manifest'])
 assert len(req)==len(fold)==13
 genes=[s[len('future_rank__'):] for s in schema['targets']]
 assert len(genes)==len(set(genes))==G and schema['targets']==['future_rank__'+g for g in genes]
 assert schema['features']==['current24_rank__'+g for g in genes]+['initial0_rank__'+g for g in genes]+['requested_endpoint_is336h']
 with np.load(paths['inputs'],allow_pickle=False) as z:
  assert set(z.files)=={'x','model_index'};x=z['x'].copy();mi=z['model_index'].copy()
 assert x.shape==(13,7197) and np.array_equal(mi,np.arange(13))
 assert np.isfinite(x).all() and ((x>=0)&(x<=1)).all() and np.isin(x[:,-1],[0,1]).all()
 assert len({r['case_id'] for r in req})==len({r['target_id'] for r in req})==13
 assert Counter(r['endpoint_hours'] for r in req)=={168.:7,336.:6}
 assert Counter(r['background'] for r in req)=={'ARP42':2,'ASP1':2,'HHT2':2,'IEC1':2,'IES2':1,'PHT1':2,'SMT0':2}
 for j,(r,f) in enumerate(zip(req,fold)):
  assert r['model_index']==f['model_index']==j
  assert r['case_id']==f['case_id'] and r['target_id']==f['target_id']
  assert r['background']==f['held_background'] and r['endpoint_hours']==f['endpoint_hours']
  assert r['current_hours']==24 and x[j,-1]==float(r['endpoint_hours']==336)
  assert len(r['initial0_GSMs'])==len(r['current24_GSMs'])==len(r['future_GSMs'])==3
  assert len(set(r['initial0_GSMs']+r['current24_GSMs']+r['future_GSMs']))==9
  expected_other=[i for i,s in enumerate(req) if s['background']!=r['background']]
  expected_fit=[i for i in expected_other if req[i]['endpoint_hours']==r['endpoint_hours']]
  expected_held=[i for i,s in enumerate(req) if s['background']==r['background']]
  assert f['fit_row_indices']==expected_fit and f['all_horizon_training_row_indices']==expected_other
  assert f['held_row_indices']==expected_held and f['cross_partition_explicit_links']==[] and f['numeric_design_rank']==2
  assert len(expected_fit)==(6 if r['endpoint_hours']==168 else 5)
  assert f['fit_case_ids']==[req[i]['case_id'] for i in expected_fit]
  assert f['excluded_case_ids']==[req[i]['case_id'] for i in expected_held]
  assert f['training_backgrounds']==[req[i]['background'] for i in expected_fit]
 with np.load(paths['parameters'],allow_pickle=False) as z:
  assert set(z.files)=={'mean_current','mean_initial','mean_future','pooled_mean','initial_projection','current_slope','initial_slope'}
  pars={k:z[k].copy() for k in z.files}
 for k,v in pars.items():
  assert v.shape==((13,G) if k in ['mean_current','mean_initial','mean_future','pooled_mean'] else (13,)) and np.isfinite(v).all()
 # All values below are stored training parameters or declared inputs. No target arrays are mounted.
 dx=x[:,:G]-pars['mean_current'][mi];d0=x[:,G:2*G]-pars['mean_initial'][mi]
 current=pars['mean_future'][mi]+pars['current_slope'][mi,None]*dx
 nested=current+pars['initial_slope'][mi,None]*(d0-pars['initial_projection'][mi,None]*dx)
 heads=['current24','nested_initial0','horizon_clock','pooled_train_mean','own24h_persistence']
 raw=np.stack([current,nested,pars['mean_future'][mi],pars['pooled_mean'][mi],x[:,:G]])
 clipped=np.clip(raw,0,1)
 assert raw.shape==clipped.shape==(5,13,G) and np.isfinite(raw).all() and np.isfinite(clipped).all()
 np.savez_compressed(O/'lobo_frozen_predictions2143.npz',raw=raw,clipped=clipped,head_names=np.asarray(heads),
  case_ids=np.asarray([r['case_id'] for r in req]),target_ids=np.asarray([r['target_id'] for r in req]),model_index=mi)
 put('lobo_prediction_requests2143.json',req)
 put('lobo_prediction_manifest2143.json',dict(operation='saved_parameter_prediction_only',head_names=heads,prediction_shape=[5,13,G],
  prediction_file=pin(O/'lobo_frozen_predictions2143.npz'),request_file=pin(O/'lobo_prediction_requests2143.json'),
  model_ref=P['references']['model'],parameter_ref=P['references']['parameters'],schema_ref=P['references']['schema'],input_ref=P['references']['inputs'],
  all_input_pins=before,model_reloaded_from_saved_files=True,fit_calls=0,target_value_arrays_mounted=False,
  prediction_input_is_preprocessed_rankmean3=True,raw_matrix_new_request_verification=False,
  actual_old_native_or_export_package_verified=False,scientific_approval=False))
 assert {k:pin(v) for k,v in paths.items()}==before
 put('lobo_prediction_ready2143.json',dict(status='predictions_frozen_not_scored',all_inputs_unchanged=True,
  outputs={n:pin(O/n) for n in ['lobo_frozen_predictions2143.npz','lobo_prediction_requests2143.json','lobo_prediction_manifest2143.json']},
  population_requests=13,backgrounds=7,heads=5,views=['RAW','CLIP'],genes=G,
  saved_model_parameters_reloaded=True,truth_value_arrays_mounted=False,fit_calls=0,query_scores=0,native_fit_predict_calls=0,new_formal_experiments=0,
  actual_original_libraries_to_score_later=39,raw_matrix_new_request_verification=False,scientific_approval=False,
  next_step='Qualify frozen output pins; join13 target means and39 original future libraries only in a separate isolated scoring preparation.'))
 put('metadata.json',dict(status='matrix_ready',task_ready=False,readiness='lobo_prediction_ready2143.json',query_scores=0,fit_calls=0,scientific_approval=False))

 output_pin=pin(O/'lobo_frozen_predictions2143.npz')
 put('adapter_receipt.json',dict(status='completed',phase='predict',task='yeast_late_relative_rna',
  prediction=output_pin,exact_historical_frozen_packet_match=output_pin==P['reference_prediction'],
  folds=7,endpoints=13,future_libraries=39,fit_calls=0,score_calls=0,
  target_value_arrays_mounted=False,scientific_approval=False))

if __name__=='__main__':
 main()

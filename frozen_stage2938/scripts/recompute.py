"""Recompute frozen final results and saved-parameter predictions. Never fit a model."""
from pathlib import Path
import csv, json, hashlib, io, zipfile, collections
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; E=ROOT/'evidence'; OUT=ROOT/'source_data'; OUT.mkdir(exist_ok=True)
checks=[]; rows=[]; params=[]; used=set()
def rel(p):return p.relative_to(ROOT).as_posix()
def read(p):
 used.add(rel(p));return json.loads(p.read_text('utf-8'))
def arrays(p):
 used.add(rel(p))
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def check(name,actual,expected,tol=1e-11):
 a,b=np.asarray(actual),np.asarray(expected);delta=float(np.max(np.abs(a-b)))
 ok=a.shape==b.shape and bool(np.all(np.isfinite(a))) and delta<=tol
 checks.append(dict(check=name,max_absolute_difference=delta,tolerance=tol,passed=ok))
 if not ok:raise ValueError((name,delta,a.shape,b.shape))
def table(name,data):
 keys=list(dict.fromkeys(k for r in data for k in r))
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)
def metric(task,cut,level,group,model,p,y,predfile,truthfile,horizon=''):
 e=np.asarray(p)-np.asarray(y)
 for m,v in [('MAE',np.abs(e).mean()),('MSE',(e*e).mean())]:
  rows.append(dict(task=task,cut=cut,level=level,group=group,model=model,metric=m,value=float(v),horizon=horizon,prediction_file=predfile,target_file=truthfile,provenance='recomputed_from_saved_prediction_and_target'))
def aggregate(task,cut,from_level,to_level,expected=None):
 rr=[r for r in rows if r['task']==task and r['cut']==cut and r['level']==from_level]
 for model,m in dict.fromkeys((r['model'],r['metric']) for r in rr):
  vals=[r['value'] for r in rr if (r['model'],r['metric'])==(model,m)]
  value=float(np.mean(vals));rows.append(dict(task=task,cut=cut,level=to_level,group='ALL',model=model,metric=m,value=value,horizon='',prediction_file='see individual rows',target_file='see individual rows',provenance='recomputed_from_saved_prediction_and_target'))
  if expected is not None:check(f'{task}/{cut}/{to_level}/{model}/{m}',value,expected[model][m.lower()] if m.lower() in expected[model] else expected[model][m])

# Verify every supplied evidence file before using it.
manifest=json.loads((ROOT/'EVIDENCE_MANIFEST.json').read_text('utf-8'))
for r in manifest:
 p=ROOT/r['path'];b=p.read_bytes()
 if len(b)!=r['bytes'] or hashlib.sha256(b).hexdigest()!=r['sha256']:raise ValueError('Changed evidence: '+r['path'])

# Seed: five held related genotypes; centroid and per-array errors are distinct.
seed_contract=read(E/'predict_kit/seed_benchmark_contract.json');seed_summary=read(E/'score/seed_ar/benchmark_scores.json')
seed_groups=seed_contract['groups'];seed_archive=E/'predict_kit/inputs/seed_ar/archive/VDC_GSE90162_internal_research_package2887.zip';used.add(rel(seed_archive))
split=read(E/'score_kit/inputs/seed_ar/fold_manifest.json/fold_manifest.json')
foldmap={x['held_genotype']:x for x in split['folds']}
with zipfile.ZipFile(seed_archive) as z:
 for g in seed_groups:
  pf=E/f'prediction/seed_ar/{g}_package_reference.npz';cf=E/f'prediction/seed_ar/{g}_candidate.npz';tf=E/f'score_kit/inputs/seed_ar/outcome/outcome_{g}.npz/outcome_{g}.npz';bf=E/f'score_kit/inputs/seed_ar/control/controls_{g}.npz/frozen_primary_predictions.npz'
  truth=arrays(tf);base=arrays(bf);pred={'source_ar_mean':arrays(pf)['prediction'][0],'scalar_candidate':arrays(cf)['prediction'][0]}
  pred.update({k:base[k][0] for k in ['persistence','shifted_persistence','fixed_half_shift']})
  check('seed centroid '+g,truth['y'][0],truth['ar_arrays'].mean(0))
  with zipfile.ZipFile(io.BytesIO(z.read('checkpoints/'+g+'.zip'))) as ck:
   state=json.loads(ck.read('model/state.json'));pp=np.load(io.BytesIO(ck.read('model/parameters.npz')),allow_pickle=False)
   q=arrays(E/f'predict_kit/inputs/seed_ar/{g}/request_{g}.npz')['x']
   check('seed saved reference '+g,pred['source_ar_mean'],pp['mean_y'])
   check('seed saved scalar '+g,pred['scalar_candidate'],(pp['mean_y']+state['alpha']*(q-pp['mean_x']))[0])
   params.append(dict(task='seed_ar',fold=g,parameter='alpha',value=state['alpha'],state_step=state['step'],source_rows=4,source_file=rel(seed_archive)+'!checkpoints/'+g+'.zip'))
  for model,v in pred.items():
   pfname=rel(pf if model=='source_ar_mean' else cf if model=='scalar_candidate' else bf)
   metric('seed_ar','related_genotype_out','group',g,model,v,truth['y'][0],pfname,rel(tf))
   for j,gsm in enumerate(foldmap[g]['outcome_gsm']):metric('seed_ar','related_genotype_out','library',gsm,model,v,truth['ar_arrays'][j],pfname,rel(tf))
 seed_exp=dict(seed_summary['macro_genotype_centroid']);seed_exp['scalar_candidate']=seed_exp['candidate']
 aggregate('seed_ar','related_genotype_out','group','primary',seed_exp)
 seed_exp=dict(seed_summary['equal_AR_array']);seed_exp['scalar_candidate']=seed_exp['candidate']
 aggregate('seed_ar','related_genotype_out','library','secondary_library',seed_exp)

# Apricot: preserve both cuts; these are not independent datasets.
acfg=read(E/'score_kit/apricot_score_config.json');asummary=read(E/'score/apricot_native800/apricot_scores.json')
for fold in acfg['folds']:
 f,cut=fold['fold'],fold['cut'];pf=E/f'prediction/apricot_native800/{f}_predictions.npz';tf=E/f'score_kit/inputs/apricot_native800/{f}/truth/truth.npz'
 pred=arrays(pf);t=arrays(tf);libs=read(E/f'score_kit/inputs/apricot_native800/{f}/library_rows/truth_library_rows.json');held=pred['genotype_ids'].tolist();assert held==fold['held_genotypes']
 archive=E/f'predict_kit/inputs/apricot_native800/{f}/checkpoint/apricot_updated_checkpoint.zip';used.add(rel(archive))
 with zipfile.ZipFile(archive) as z:
  state=json.loads(z.read('model/state.json'));pp=np.load(io.BytesIO(z.read('model/parameters.npz')),allow_pickle=False);x=arrays(E/f'predict_kit/inputs/apricot_native800/{f}/query/query.npz')['x'];G=state['features'];current=x[:,G:2*G];dc=current-pp['mean_current'];de=(current-x[:,:G])-pp['mean_early'];anchor=current+state['half_mean_late']*pp['mean_late']
  reconstructed={'ridge_history':anchor+pp['history_coef'][0]*dc+pp['history_coef'][1]*de,'ridge_current':anchor+state['current_coef']*dc,'half_source_drift':anchor,'persistence':current,'full_source_drift':current+pp['mean_late'],'source_future_mean':np.broadcast_to(pp['mean_current']+pp['mean_late'],current.shape)}
  for k,v in reconstructed.items():check('apricot saved '+f+'/'+k,pred[k],v)
  for k,v in [('history_current_coef',pp['history_coef'][0]),('history_early_change_coef',pp['history_coef'][1]),('current_only_coef',state['current_coef'])]:params.append(dict(task='apricot_native800',fold=f,parameter=k,value=float(v),state_step=state['step'],source_rows=state['source_count'],source_file=rel(archive)))
 for j,g in enumerate(held):
  check('apricot centroid '+cut+'/'+g,t['y'][j],t['library_y'][t['library_query_index']==j].mean(0))
  for model in acfg['modes']:metric('apricot_native800',cut,'group',g,model,pred[model][j],t['y'][j],rel(pf),rel(tf))
 for j,lib in enumerate(libs):
  for model in acfg['modes']:metric('apricot_native800',cut,'library',lib['GSM'],model,pred[model][int(t['library_query_index'][j])],t['library_y'][j],rel(pf),rel(tf))
for cut,stats in asummary['summary'].items():
 aggregate('apricot_native800',cut,'group','primary',stats['fresh_centroid_equal4'])
 aggregate('apricot_native800',cut,'library','secondary_library',stats['fresh_library_equal11'])

# Yeast: join target IDs explicitly; average within background before across backgrounds.
pf=E/'prediction/yeast_late_relative_rna/lobo_frozen_predictions2143.npz';tf=E/'score_kit/inputs/yeast_late_relative_rna/truth/later_truth_scoring_only2121.npz';pred=arrays(pf);t=arrays(tf);req=read(E/'prediction/yeast_late_relative_rna/lobo_prediction_requests2143.json');ysummary=read(E/'score/yeast_late_relative_rna/lobo_score_focus2144.json');lookup={s:i for i,s in enumerate(t['target_ids'].tolist())}
par=arrays(E/'predict_kit/inputs/yeast_late_relative_rna/parameters/lobo_parameters2142.npz');inp=arrays(E/'predict_kit/inputs/yeast_late_relative_rna/inputs/lobo_inputs_only2142.npz');x=inp['x'];mi=inp['model_index'];G=3598;dx=x[:,:G]-par['mean_current'][mi];d0=x[:,G:2*G]-par['mean_initial'][mi];cur=par['mean_future'][mi]+par['current_slope'][mi,None]*dx;nest=cur+par['initial_slope'][mi,None]*(d0-par['initial_projection'][mi,None]*dx)
raw=np.stack([cur,nest,par['mean_future'][mi],par['pooled_mean'][mi],x[:,:G]])
check('yeast saved raw',raw,pred['raw']);check('yeast saved clipped',np.clip(raw,0,1),pred['clipped'])
for j,r in enumerate(req):
 assert r['target_id']==str(pred['target_ids'][j]);idx=lookup[r['target_id']];ix=np.flatnonzero(t['assay_target_index']==idx);check('yeast centroid '+r['target_id'],t['population_mean_rank'][idx],t['original_assay_rank'][ix].mean(0))
 for key in ['current_slope','initial_slope','initial_projection']:params.append(dict(task='yeast_late_relative_rna',fold=r['case_id'],parameter=key,value=float(par[key][j]),state_step='saved_fit',source_rows=6 if r['endpoint_hours']==168 else 5,source_file='evidence/predict_kit/inputs/yeast_late_relative_rna/parameters/lobo_parameters2142.npz'))
 for k,head in enumerate(pred['head_names'].tolist()):
  for view in ['RAW','CLIP']:
   v=pred['raw' if view=='RAW' else 'clipped'][k,j];model=head+'_'+view
   metric('yeast_late_relative_rna','background_out','endpoint',r['background'],model,v,t['population_mean_rank'][idx],rel(pf),rel(tf),r['endpoint_hours'])
   for i in ix:metric('yeast_late_relative_rna','background_out','library',str(t['original_future_GSM'][i]),model,v,t['original_assay_rank'][i],rel(pf),rel(tf),r['endpoint_hours'])
for bg in sorted({r['background'] for r in req}):
 subset=[r for r in rows if r['task']=='yeast_late_relative_rna' and r['level']=='endpoint' and r['group']==bg]
 for model,m in dict.fromkeys((r['model'],r['metric']) for r in subset):
  v=float(np.mean([r['value'] for r in subset if (r['model'],r['metric'])==(model,m)]));rows.append(dict(task='yeast_late_relative_rna',cut='background_out',level='group',group=bg,model=model,metric=m,value=v,horizon='observed horizons',prediction_file=rel(pf),target_file=rel(tf),provenance='recomputed_from_saved_prediction_and_target'))
aggregate('yeast_late_relative_rna','background_out','group','primary',ysummary['primary_background_balanced_population'])
aggregate('yeast_late_relative_rna','background_out','endpoint','secondary_endpoint',ysummary['secondary_endpoint_weighted_population'])
aggregate('yeast_late_relative_rna','background_out','library','secondary_library',ysummary['endpoint_weighted_libraries'])

# Functional panel: all 24 once-held observed means, all five fixed prediction heads.
schema=read(E/'predict_kit/inputs/chilling_functional/schema/functional_schema.json');func=[];seen=set()
for fold in range(4):
 pf=E/f'prediction/chilling_functional/fold_{fold}_predictions.json';tf=E/f'score_kit/inputs/chilling_functional/fold_{fold}_truth/fold_{fold}_truth.json';pp=read(pf);tt=read(tf);modelpath=E/f'predict_kit/inputs/chilling_functional/fold_{fold}_model/functional_model.json';model=read(modelpath);query=read(E/f'predict_kit/inputs/chilling_functional/fold_{fold}_query/fold_{fold}_query.json')['rows'];source=read(E/f'functional_roles/fold_{fold}_source.json')['rows'];assert len(source)==18 and len(query)==6 and not ({r['row_id'] for r in source}&{r['row_id'] for r in query})
 truth={r['row_id']:r for r in tt['rows']};assert list(pp['row_ids'])==[r['row_id'] for r in query]
 X=np.asarray([[1.]+[float(r['genotype']==g) for g in schema['genotypes'][1:]]+[float(r['condition_id']==a['id']) for a in schema['conditions'][1:]] for r in query])
 saved={'fractional_logit_additive':np.exp(-np.logaddexp(0,-X@np.asarray(model['fractional_logit_coefficients']))),'two_way_ols_clipped':np.clip(X@np.asarray(model['two_way_ols_coefficients']),0,1),'condition_only_median':[model['condition_medians'][r['condition_id']] for r in query],'line_only_median':[model['line_medians'][r['genotype']] for r in query],'grand_median':[model['grand_median']]*6}
 for head,v in saved.items():check(f'functional saved fold{fold}/{head}',v,pp['predictions'][head])
 for head in ['fractional_logit','two_way_ols']:
  for name,v in zip(model['feature_columns'],model[head+'_coefficients']):params.append(dict(task='chilling_functional',fold=fold,parameter=head+':'+name,value=v,state_step='saved_fit',source_rows=18,source_file=rel(modelpath)))
 for j,r in enumerate(query):
  assert r['row_id'] not in seen;seen.add(r['row_id']);tr=truth[r['row_id']];y=tr['germinated_fraction']
  for head,v in pp['predictions'].items():
   func.append(dict(row_id=r['row_id'],fold=fold,genotype=r['genotype'],condition=r['condition_id'],model=head,observed_fraction=y,predicted_fraction=v[j],observed_percent=100*y,predicted_percent=100*v[j],absolute_error_pp=100*abs(v[j]-y),signed_error_pp=100*(v[j]-y),prediction_file=rel(pf),target_file=rel(tf)))
assert len(seen)==24
fsummary=read(E/'score/chilling_functional/functional_score_summary.json')
for model in saved:
 ss=[r for r in func if r['model']==model];v=float(np.mean([r['absolute_error_pp'] for r in ss]));check('functional primary '+model,v,fsummary['metrics'][model]['mae_pp'])
 rows.append(dict(task='chilling_functional',cut='held_cell_calibration',level='primary',group='ALL',model=model,metric='MAE_pp',value=v,horizon='7 days',prediction_file='see FUNCTIONAL_CELLS.csv',target_file='see FUNCTIONAL_CELLS.csv',provenance='recomputed_from_saved_prediction_and_target'))
 for level,key in [('group','genotype'),('condition','condition')]:
  for g in dict.fromkeys(r[key] for r in ss):
   local=[r['absolute_error_pp'] for r in ss if r[key]==g];rows.append(dict(task='chilling_functional',cut='held_cell_calibration',level=level,group=g,model=model,metric='MAE_pp',value=float(np.mean(local)),horizon='7 days',prediction_file='see FUNCTIONAL_CELLS.csv',target_file='see FUNCTIONAL_CELLS.csv',provenance='recomputed_from_saved_prediction_and_target'))
 check('functional worst '+model,max(r['absolute_error_pp'] for r in ss),fsummary['metrics'][model]['worst_abs_pp'])

# Final serving record: direct numerical inheritance, no refit.
later=read(E/'later_use/delivered_package_prediction.json')
cached=[]
for cut,stats in asummary['summary'].items():
 for model,mm in stats['cached_historical_centroid_equal4'].items():
  for m,v in mm.items():cached.append(dict(task='apricot_native800',cut=cut,model=model,metric=m,value=v,evidence_status='saved_summary_only_in_this_paper_packet',source_file='evidence/score/apricot_native800/apricot_scores.json',excluded_from_primary_recomputed_panels=True))
table('ALL_RECOMPUTED_METRICS.csv',rows);table('FUNCTIONAL_CELLS.csv',func);table('SAVED_LEARNING_PARAMETERS.csv',params);table('CACHED_HISTORICAL_COMPARISONS.csv',cached);table('RECOMPUTATION_CHECKS.csv',checks)
primary=[r for r in rows if r['level']=='primary' and r['metric']==('MSE' if r['task']=='yeast_late_relative_rna' else 'MAE_pp' if r['task']=='chilling_functional' else 'MAE')]
table('FINAL_TASK_RESULTS.csv',primary)
report=dict(status='passed',checks=len(checks),max_absolute_difference=max(x['max_absolute_difference'] for x in checks),evidence_files_verified=len(manifest),metrics_rows=len(rows),functional_rows=len(func),saved_parameter_rows=len(params),new_training_calls=0,new_api_calls=0,new_model_searches=0,uncertainty_intervals_computed=False,used_files=sorted(used))
(ROOT/'RECOMPUTATION_REPORT.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report|{'used_files':len(used)}))
for r in primary:print(r['task'],r['cut'],r['model'],r['metric'],r['value'])

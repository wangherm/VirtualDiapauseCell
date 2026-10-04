"""Small frozen-model applications; all fitting precedes any reserved model query."""
from pathlib import Path
import copy,hashlib,shutil
import numpy as np
from .io import read_json,write_json,save_npz,sha256,object_hash
from .admission import role_manifest,audit_internal_task
from .contracts import ObservationBundle
from .state import StatePredictor
from .pk1_numeric import KillifishExitReference
from .pk1_assets import verify_files
from .pk2_numeric import dual_ridge,metrics,query
from .application_data import development_counts,non_target_input,projection,bulk_queries,sc_queries
from .observation import normalise_expression,score_programmes
from .waves import WaveReference


def inventory(root):
    root=Path(root)
    return {p.relative_to(root).as_posix():sha256(p) for p in sorted(root.rglob('*')) if p.is_file()}


def predict_ridge(path,x,n_outputs=None):
    with np.load(path,allow_pickle=False) as a:
        if 'coefficients' not in a:
            if n_outputs is None:raise ValueError('Partial-target ridge requires explicit output dimension')
            out=np.full((len(x),n_outputs),np.nan)
            for key in a.files:
                if not key.endswith('_columns'):continue
                prefix=key[:-len('columns')]
                out[:,a[key]]=((x-a[prefix+'mean'])/a[prefix+'scale'])@a[prefix+'coefficients']+a[prefix+'target_mean']
            return out
        return ((x-a['mean'])/a['scale'])@a['coefficients']+a['target_mean']


def zero_target_bundle(template,scored):
    b=copy.deepcopy(template);b.values=scored['values'];b.mask=scored['mask'];b.coverage=scored['coverage']
    b.clock[:]=0;b.clock_mask[:]=False
    return b.validate()


def prepare_freeze(source,private,run,config,code_identity,supplement=None):
    source,private,run=map(Path,(source,private,run));snapshot=run/'snapshot'
    if (run/'freeze.json').exists():
        freeze=verify_freeze(run)
        if freeze['config']!=config or freeze['code_identity']!=code_identity:raise ValueError('Frozen application configuration/code changed')
        return freeze
    if snapshot.exists():raise FileExistsError('Incomplete freeze preserved; use a new run rather than overwriting fitted artifacts')
    policy,_=role_manifest(private/'sample_roles.json')
    lock=read_json(private/'source_lock.json')
    if object_hash(lock)!=policy['source_lock_hash']:raise ValueError('Role policy source identity changed')
    verify_files(private/'input',lock['files'])
    if not (source/'repair_import.json').exists():raise ValueError('Use the completed corrected PK2 run, not the original faulty semantic run')
    statuses=read_json(source/'queue_status.json')['tasks'];plan=read_json(source/'plan.json')
    selected=[j for jobs in config['candidates'].values() for j in jobs]
    for job in selected:
        s=statuses[job]
        if s['status'] not in {'evaluated_new','reused_verified'}:raise ValueError('Candidate not evaluated: '+job)
        if plan[job]['params']['job'].get('seed')!=42:raise ValueError('Frozen common seed rule violated')
        verify_files(source/'tasks'/job,s['files'])
        meta=read_json(source/'tasks'/job/'model/run.json')
        if meta['scope']['context'].get('admission_hash')!=policy['approval']['content_hash']:raise ValueError('Candidate and query role policy differ')
        if meta['is_synthetic']:raise ValueError('Synthetic model cannot become a research application')
        if any(r['split']!='train' for r in meta['fit_rows']):raise ValueError('Candidate has non-training fitted rows')
        if 'new_domain' in job:
            contract=read_json(source/'tasks'/job/'semantic_input.json')
            if contract['condition']!='domain' or contract['forward_check']['status']!='passed':raise ValueError('Corrected domain semantics not verified')
    # Persist intent before any fitting; no held-out expression is consulted here.
    intent={'config':config,'source_signature':read_json(source/'run_manifest.json')['signature'],
        'role_hash':policy['approval']['content_hash'],'candidate_checkpoints':{j:sha256(source/'tasks'/j/'model/best.pt') for j in selected},
        'evaluation_scope':config['evaluation_scope'],'queries_started':False}
    write_json(run/'FREEZE_INTENT.json',intent);snapshot.mkdir(parents=True)
    shutil.copy2(private/'sample_roles.json',snapshot/'sample_roles.json')
    for view,jobs in config['candidates'].items():
        print('FREEZE_VIEW',view,flush=True)
        inp=source/'tasks'/('input_'+view);verify_files(inp,statuses['input_'+view]['files'])
        shutil.copytree(inp,snapshot/view/'input')
        prep=private/'prepared'/view;contract=read_json(prep/'expression_contract.json')
        if sha256(prep/'gene_expression.npz')!=contract['expression_sha256']:raise ValueError('Prepared development expression changed')
        shutil.copytree(prep,snapshot/view/'prepared')
        b=ObservationBundle.load(inp/'bundle');audit_internal_task(b.rows,'state')
        if any(r['split'] not in {'train','validation'} for r in b.rows):raise ValueError('Only development fits permitted')
        genes=contract['gene_ids'];defs=read_json(prep/'programmes.json');counts=development_counts(private,policy,view,b,genes)
        if object_hash(defs)!=contract['context']['programme_definition_id']:raise ValueError('Programme membership identity changed')
        expected_panel=object_hash({'parent':object_hash(defs),'mode':'programme','features':b.feature_ids})
        if b.context['programme_definition_id']!=expected_panel:raise ValueError('Frozen input is not the original programme panel')
        full=normalise_expression(counts,np.ones_like(counts,bool),'counts')
        scored=score_programmes(full,np.ones_like(full,bool),genes,defs)
        if scored['feature_ids']!=b.feature_ids or not np.array_equal(scored['mask'],b.mask) or not np.allclose(scored['values'],b.values,atol=1e-6):raise ValueError('Raw development counts do not reproduce frozen input')
        tr,va=b.indices('train'),b.indices('validation')
        hidden={g for g in genes if int(hashlib.sha256(g.encode()).hexdigest()[:8],16)%5==0}
        _,s=non_target_input(counts,genes,defs,hidden);safe=zero_target_bundle(b,s)
        targets=full
        eligible=np.array([i for i,g in enumerate(genes) if g in hidden and targets[tr,i].var()>1e-8])
        idx=eligible[np.argsort(-targets[tr][:,eligible].var(0),kind='stable')[:config['hidden_gene_limit']]]
        if not len(idx):raise ValueError('No train-variable hidden targets')
        target=targets[:,idx];train_rows=[b.rows[i] for i in tr]
        observed_features=np.column_stack([safe.values,safe.mask])
        direct=dual_ridge(observed_features[tr],target[tr],train_rows,config['readout_ridge_alpha'])
        save_npz(snapshot/view/'hidden/direct.npz',**direct.parameters)
        mean=target[tr].mean(0);save_npz(snapshot/view/'hidden/mean.npz',value=mean)
        write_json(snapshot/view/'hidden/contract.json',{'all_removed_genes':sorted(hidden),'target_gene_ids':[genes[i] for i in idx],
            'target_indices':idx.tolist(),'fit_ids':[b.rows[i]['observation_id'] for i in tr],
            'input_rule':'Remove hidden genes from all programme ranks/members AND input library denominator',
            'target_scale':'log1p relative count abundance using target sample full library; not absolute per-cell abundance',
            'input_distribution_shift':'frozen-state input genes withheld; separate development-fitted downstream readout',
            'baseline_validation':metrics(direct(observed_features[va]),target[va],np.ones_like(target[va],bool),[b.rows[i] for i in va])})
        for job in jobs:
            src=source/'tasks'/job
            shutil.copytree(src/'model',snapshot/'models'/job)
            model=StatePredictor.load(snapshot/'models'/job)
            latent=query(model,safe)['latent'];fn=dual_ridge(latent[tr],target[tr],train_rows,config['readout_ridge_alpha'])
            save_npz(snapshot/view/'hidden'/(job+'.npz'),**fn.parameters)
            save_npz(snapshot/view/'hidden'/(job+'_development.npz'),prediction=fn(latent[va]),target=target[va],baseline=direct(observed_features[va]))
            write_json(snapshot/view/'hidden'/(job+'.json'),{'checkpoint_sha256':sha256(snapshot/'models'/job/'best.pt'),
                'fit_scope':'original_train_only','model_selection':'none; fixed candidate and alpha',
                'validation':metrics(fn(latent[va]),target[va],np.ones_like(target[va],bool),[b.rows[i] for i in va])})
            print('READOUT_FITTED',job,flush=True)
        # Existing full-input ridge is kept with its actual normalisation checkpoint.
        ridge=source/'tasks'/jobs[0]/'evaluation/ridge_parameters'
        paths=list(ridge.glob('*.npz'))
        if len(paths)!=1:raise ValueError('One internal study ridge expected')
        shutil.copy2(paths[0],snapshot/view/'ridge.npz')
        wave='waves_'+view;verify_files(source/'tasks'/wave,statuses[wave]['files'])
        shutil.copytree(source/'tasks'/wave,snapshot/view/'waves')
        if view=='core_celltypes':
            verify_files(source/'tasks/type_waves',statuses['type_waves']['files'])
            shutil.copytree(source/'tasks/type_waves',snapshot/view/'type_waves')
    # Supplement identity is fixed before querying. Absence is recorded; it does not block bulk.
    supplement_identity=None
    if supplement:
        m=read_json(Path(supplement)/'manifest.json')
        if m['role_hash']!=policy['approval']['content_hash']:raise ValueError('Supplement role hash differs')
        verify_files(supplement,m['files']);supplement_identity={'manifest_sha256':sha256(Path(supplement)/'manifest.json'),'files':m['files']}
        from .application_stress import fit_reference
        write_json(snapshot/'stress/status.json',fit_reference(supplement,snapshot,policy,config))
    manifest={**intent,'code_identity':code_identity,'files':inventory(snapshot),'supplement_identity':supplement_identity,
        'input_source_lock_sha256':sha256(private/'source_lock.json'),'private_source_files':read_json(private/'source_lock.json')['files'],
        'new_training':'eight latent-to-hidden-gene ridge readouts plus three direct-programme baselines; no state/Qwen retraining',
        'original_role_manifest_changed':False,'future_prediction':config['future_prediction']}
    manifest['freeze_id']=object_hash(manifest);write_json(run/'freeze.json',manifest)
    return manifest


def verify_freeze(run):
    run=Path(run);m=read_json(run/'freeze.json')
    if object_hash({k:v for k,v in m.items() if k!='freeze_id'})!=m['freeze_id']:raise ValueError('Freeze manifest changed')
    verify_files(run/'snapshot',m['files'])
    return m


def apply_group(run,private,group,supplement=None):
    run,private=Path(run),Path(private);f=verify_freeze(run);snapshot=run/'snapshot';config=f['config']
    policy,_=role_manifest(snapshot/'sample_roles.json')
    if sha256(private/'source_lock.json')!=f['input_source_lock_sha256']:raise ValueError('Original query source lock changed')
    if group in {'bulk','single_exit'}:
        # Source hashes are checked before loading any query columns.
        needed=['bulk_counts.csv'] if group=='bulk' else ['single_exit_gene_ids.json']+['exit_counts_'+r['sample_id']+'.npz' for r in policy['samples'] if r['source']=='single_exit' and r['role']=='reserved_evaluation']
        verify_files(private/'input',{n:f['private_source_files'][n] for n in needed})
        counts,genes,entries=bulk_queries(private,policy,group);views=['bulk'];missing=[]
    else:
        if not supplement or not f['supplement_identity']:raise ValueError('unsupported_missing_frozen_SC_supplement')
        if sha256(Path(supplement)/'manifest.json')!=f['supplement_identity']['manifest_sha256']:raise ValueError('Supplement identity changed after freeze')
        counts,genes,entries,m=sc_queries(supplement,policy,group);views=[group];missing=m['unavailable']
    out=run/'applications'/group;out.mkdir(parents=True,exist_ok=True);reports=[]
    for view in views:
        vdir=snapshot/view
        # Isolate individual samples: an unavailable type/gene universe cannot hide other queries.
        for i,entry in enumerate(entries):
            name=object_hash([entry['sample_key'],entry.get('cell_type')])[:20];dest=out/name
            if (dest/'result.json').exists():
                previous=read_json(dest/'result.json')
                if previous.get('freeze_id')!=f['freeze_id']:raise ValueError('Recorded query used another freeze')
                if previous['files']:verify_files(dest,previous['files'])
                reports.append(previous);continue
            try:
                b,y,x,g,defs,scopes=projection(counts[i:i+1],genes,[entry],policy,vdir/'prepared',vdir/'input',config)
                ref=KillifishExitReference.load(vdir/'input/reference');direct_clock=ref.predict(y,g)
                hidden=read_json(vdir/'hidden/contract.json');_,s=non_target_input(x,g,defs,set(hidden['all_removed_genes']))
                safe=zero_target_bundle(b,s);target=y[:,hidden['target_indices']]
                direct_hidden=predict_ridge(vdir/'hidden/direct.npz',np.column_stack([safe.values,safe.mask]))
                with np.load(vdir/'hidden/mean.npz',allow_pickle=False) as a:mean_hidden=np.broadcast_to(a['value'],target.shape).copy()
                save_npz(dest/'observations.npz',observed=b.values,mask=b.mask,coverage=b.coverage,reference_clock=direct_clock,
                    gene_observed=y,hidden_gene_observed=target,hidden_gene_direct_ridge=direct_hidden,hidden_gene_train_mean=mean_hidden)
                write_json(dest/'projection.json',{'freeze_id':f['freeze_id'],'scope_projection':scopes,'rows':b.rows,
                    'feature_ids':b.feature_ids,'hidden_gene_ids':hidden['target_gene_ids']})
                models={}
                gene_index={fid:j for j,fid in enumerate(g)}
                for job in config['candidates'][view]:
                    predictor=StatePredictor.load(snapshot/'models'/job);pred=query(predictor,b)
                    arrays={'reconstructed':pred['programme'],'neural_clock':pred['clock'],'latent':pred['latent']}
                    for kind in ('programme','gene','TF'):
                        if not (vdir/'waves'/kind/'reference.json').exists():continue
                        wave=WaveReference.load(vdir/'waves'/kind)
                        observed=b.values if kind=='programme' else y[:,[gene_index[fid] for fid in wave.feature_ids]]
                        if kind=='TF':arrays['TF_observed']=observed
                        w=wave.residual(direct_clock,observed,b.mask if kind=='programme' else np.ones_like(observed,bool),b.clock_reference_id,object_hash(b.context))
                        arrays.update({kind+'_'+k:v for k,v in w.items() if isinstance(v,np.ndarray)})
                        chained=wave.predict(pred['clock'],b.clock_reference_id,object_hash(b.context))
                        arrays[kind+'_neural_clock_expected']=chained['expected']
                    if view=='core_celltypes':
                        tm=read_json(vdir/'type_waves/result.json');typ=entry.get('cell_type')
                        if typ in tm['types']:
                            train=ObservationBundle.load(vdir/'input/bundle');idx=[j for j,r in enumerate(train.rows) if r['split']=='train' and r['reference_eligible'] and r.get('cell_type')==typ]
                            supported=bool(idx and min(train.clock[idx])<=direct_clock[0]<=max(train.clock[idx]))
                            design=np.array([[float(t==typ) for t in tm['types']]+[float(direct_clock[0])]])
                            expected=predict_ridge(vdir/'type_waves/type_intercept_shared_clock/model.npz',design)
                            arrays['type_wave_expected']=expected if supported else np.full_like(expected,np.nan)
                            arrays['type_wave_residual']=b.values-arrays['type_wave_expected']
                    hp=predict_ridge(vdir/'hidden'/(job+'.npz'),query(predictor,safe)['latent'])
                    arrays['hidden_gene_prediction']=hp
                    save_npz(dest/(job+'.npz'),**arrays)
                    models[job]={'checkpoint_sha256':sha256(snapshot/'models'/job/'best.pt'),
                        'direct_reference_clock':float(direct_clock[0]),'neural_clock':float(pred['clock'][0]),
                        'programme_coverage':float(b.mask.mean()),'in_reference_wave_range':float(arrays['programme_residual_mask'].mean()),
                        'hidden_readout':metrics(hp,target,np.ones_like(target,bool),b.rows),
                        'hidden_direct_ridge':metrics(direct_hidden,target,np.ones_like(target,bool),b.rows),
                        'hidden_train_mean':metrics(mean_hidden,target,np.ones_like(target,bool),b.rows),
                        'type_wave_status':('available_or_outside_training_coordinate_range' if 'type_wave_expected' in arrays else 'unsupported_unmatched_archived_type') if view=='core_celltypes' else 'not_applicable',
                        'reconstruction_is_not_clean_truth':True}
                p=StatePredictor.load(snapshot/'models'/config['candidates'][view][0])
                ridge=predict_ridge(vdir/'ridge.npz',np.column_stack([(b.values-p.mean)/p.scale,b.mask]),len(b.feature_ids))
                save_npz(dest/'ridge.npz',reconstructed=ridge)
                report={'sample_key':entry['sample_key'],'cell_type':entry.get('cell_type'),'role':entry['role'],'status':'queried',
                    'freeze_id':f['freeze_id'],'models':models,'task':'current_expression_location_and_hidden_readout',
                    'future_prediction':'not_executed_no_compatible_development_transition','depth':None,
                    'failure_accuracy':None,'condition':entry['condition'],'scope':scopes[0]['qualification'],'result_folder':name}
            except ValueError as exc:
                # Only an explicit applicability rejection is an unsupported sample.
                # Missing model files, broken contracts and implementation failures must fail the task.
                if not str(exc).startswith('unsupported_'):raise
                report={'sample_key':entry['sample_key'],'cell_type':entry.get('cell_type'),'status':'unsupported','reason':str(exc),'result_folder':name,'freeze_id':f['freeze_id']}
            report['files']=inventory(dest) if dest.exists() else {}
            write_json(dest/'result.json',report);reports.append(report)
            if (i+1)%10==0 or i+1==len(entries):print(f'APPLICATION {group} {i+1}/{len(entries)}',flush=True)
    # Every requested SC query absent from the supplement remains visible in the report.
    reports.extend({'status':'unsupported',**r} for r in missing)
    write_json(out/'result.json',{'group':group,'freeze_id':f['freeze_id'],'rows':reports,
        'evaluation_scope':config['evaluation_scope'],'reserved_used_for_fitting':False,'winner_selection':False})
    return reports

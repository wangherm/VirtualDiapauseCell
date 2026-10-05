"""Read actual CW runs, never infer execution from a directory or ZIP timestamp."""
from pathlib import Path
import argparse, datetime, json
from vdc.io import read_json, write_json, sha256, object_hash
from vdc.pk1_assets import verify_files

PROTOCOL='VDC_CW1_targeted_revision_v2'


def optional(path, default):
    return read_json(path) if path.is_file() else default


def inspect_run(run):
    run=Path(run).resolve();c=optional(run/'config.json',{});plan=optional(run/'plan.json',[])
    queue=optional(run/'queue_status.json',{});states=queue.get('tasks',{})
    protocol=c.get('protocol',{});stage=protocol.get('protocol')=='VDC_CW_stage_1_experimental' and protocol.get('identity_protocol')=='fixed_evidence_v3'
    revision=(protocol.get('protocol')==PROTOCOL and protocol.get('identity_protocol')=='short_slots_v2') or stage
    expected={kind+'_'+view for kind in ('residual','support') for view in ('bulk','core','coarse')}|{'qwen_base','qwen_domain','chain_base','chain_domain'}
    if stage:expected={kind+'_'+view for kind in ('shrink','stage') for view in ('bulk','core','coarse')}|{'qwen_base','qwen_domain','chain_base','chain_domain'}
    jobs={j['id']:j for j in plan};valid_revision=revision and expected.issubset(jobs)
    categories={'added':[],'rerun':[],'reused_verified':[],'historical_completed':[],'unfinished':[]}
    errors=[];fresh_evidence={}
    for name,state in states.items():
        if state.get('status')!='completed':categories['unfinished'].append({'task':name,'status':state.get('status'),'reason':state.get('reason')});continue
        try:verify_files(run,state.get('files',{}))
        except (OSError,ValueError) as exc:
            errors.append({'task':name,'error':str(exc)});categories['unfinished'].append({'task':name,'status':'artifact_verification_failed'});continue
        if state.get('execution_kind')!='reused_verified' and state.get('task_signature') and state['task_signature']!=object_hash({'config':c,'job':jobs.get(name,{})}):
            errors.append({'task':name,'error':'Completion code/protocol/task signature mismatch'})
            categories['unfinished'].append({'task':name,'status':'signature_mismatch'});continue
        if state.get('execution_kind')=='reused_verified':categories['reused_verified'].append(name);continue
        if not valid_revision:categories['historical_completed'].append(name);continue
        job=jobs.get(name,{})
        kind='added' if job.get('kind') in {'residual','support','stage_view'} else 'rerun' if name in expected else None
        if kind is None:
            categories['unfinished'].append({'task':name,'status':'unclassified_completion'});continue
        try:
            result=read_json(run/'tasks'/name/'result.json')
            if name.startswith('qwen_'):
                if result.get('contract')!=protocol['identity_protocol']:raise ValueError('No matching identity-contract generation result')
                count=0;all_count=0
                for j in plan:
                    if j['kind']!='identity':continue
                    cards=sum(bool(s.strip()) for s in (run/'tasks'/j['id']/'cards.jsonl').read_text(encoding='utf-8').splitlines())
                    fields={'mapping','prompt_tokens','response_tokens','raw_answer','stop_observation','identity_status','json_valid','schema_valid','evidence_valid'}
                    for condition in (('original','no_numeric','challenge') if stage else ('original',)):
                        path=run/'tasks'/name/(j['id']+('' if condition=='original' else '_'+condition)+'.json');a=read_json(path)
                        if not a.get('complete') or len(a['responses'])!=cards:raise ValueError(condition+' responses incomplete')
                        if any(not fields.issubset(r) for r in a['responses']):raise ValueError('Missing fresh generation diagnostics')
                        all_count+=len(a['responses'])
                        if condition=='original':count+=len(a['responses'])
                if stage and all_count!=result.get('generation_count'):raise ValueError('Three-condition generation count mismatch')
                fresh_evidence[name]={'original_responses':count,'all_conditions_generations':result.get('generation_count'),
                                      'status_counts':result.get('status_counts')}
            elif job.get('kind')=='residual':
                if result.get('hidden_input_invariance')!='passed' or result.get('clock_refitted') is not False:
                    raise ValueError('Missing residual isolation/frozen reference evidence')
            elif job.get('kind')=='support':
                if 'support_by_condition_identity' not in result or result.get('range_changed') is not False:
                    raise ValueError('Missing unchanged-range support diagnosis')
            elif name.startswith('chain_'):
                if not (run/'tasks'/name/'stage_status.json').is_file():raise ValueError('No stage failure provenance')
            elif job.get('kind')=='stage_view':
                if result.get('actual_default_invocation')!='passed' or result.get('reload')!='passed':raise ValueError('No actual default route/reload evidence')
            categories[kind].append(name)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            errors.append({'task':name,'error':str(exc)});categories['unfinished'].append({'task':name,'status':'output_contract_unverified'})
    for name in jobs.keys()-states.keys():categories['unfinished'].append({'task':name,'status':'no_execution_record'})
    runtime=optional(run/'runtime_identity.json',{})
    runtime_verified=False
    if runtime.get('source_files') and runtime.get('release'):
        try:verify_files(Path(runtime['release']),runtime['source_files']);runtime_verified=object_hash(runtime['source_files'])==c.get('code')
        except (OSError,ValueError) as exc:errors.append({'runtime_source':str(exc)})
    log=(run/'console.log').read_text(encoding='utf-8',errors='replace') if (run/'console.log').exists() else ''
    total_fresh=len(categories['added'])+len(categories['rerun'])
    return {'run':str(run),'run_id':run.name,'protocol':protocol,'revision_config_and_plan':valid_revision,
            'config_sha256':sha256(run/'config.json') if c else None,'code_hash_recorded':c.get('code'),
            'input_version_hashes':{k:c.get(k) for k in ('role_manifest_sha256','source_config_sha256','annotation_sha256')},
            'runtime_source_verified':runtime_verified,'runtime_identity':runtime,
            'runtime_source_limit':'Recorded code hash only when the historical run has no runtime source manifest',
            'plan_sha256':sha256(run/'plan.json') if (run/'plan.json').exists() else None,
            'queue_sha256':sha256(run/'queue_status.json') if (run/'queue_status.json').exists() else None,
            'task_total':len(plan),'queue_status':queue.get('status'),'categories':categories,'fresh_tasks_verified':total_fresh,
            'fresh_generation_evidence':fresh_evidence,'errors':errors,'revision_parent':c.get('revision_parent'),
            'targeted_complete':valid_revision and expected.issubset(set(categories['added']+categories['rerun'])) and not categories['unfinished'] and not errors,
            'log_sha256':sha256(run/'console.log') if log else None,
            'launch_receipt':optional(run/'launch_receipt.json',{}),
            'log_signals':{k:log.count(k) for k in ('START qwen_base','START qwen_domain','START residual_','START support_','TASK_EXIT_CODE=')},
            'exit_code':(run/'exit_code.txt').read_text().strip() if (run/'exit_code.txt').exists() else None,
            'packaging_is_new_execution':False}


def scan(private, extra_roots=()):
    private=Path(private).resolve();pointers={}
    for name in ('LATEST_CLOCK_WAVE.txt','LATEST_CLOCK_WAVE_RELEASE.txt','LATEST_CLOCK_WAVE_REVISION.txt','LATEST_STAGE.txt','LATEST_STAGE_RELEASE.txt'):
        p=private/name;pointers[name]=p.read_text().strip() if p.exists() else None
    roots=[private/'runs',*map(Path,extra_roots)];paths=set()
    for root in roots:
        if root.exists():
            for c in root.glob('*/config.json'):
                try:
                    if str(read_json(c).get('protocol',{}).get('protocol','')).startswith(('VDC_CW1','VDC_CW_stage')):paths.add(c.parent.resolve())
                except (OSError,ValueError,AttributeError):continue
            paths.update(p.resolve() for p in root.glob('clock_wave*') if p.is_dir())
    for name in ('LATEST_CLOCK_WAVE.txt','LATEST_CLOCK_WAVE_REVISION.txt','LATEST_STAGE.txt'):
        if pointers[name] and Path(pointers[name]).is_dir():paths.add(Path(pointers[name]).resolve())
    runs=[]
    for p in sorted(paths):
        try:runs.append(inspect_run(p))
        except (OSError,ValueError,KeyError,TypeError) as exc:runs.append({'run':str(p),'audit_error':str(exc)})
    # Match historical config hashes to actual release files, not directory names or Git HEAD alone.
    releases={};release_root=private.parent/'vdc-releases'
    candidates=set(p for p in release_root.glob('*') if p.is_dir()) if release_root.exists() else set()
    if pointers['LATEST_CLOCK_WAVE_RELEASE.txt']:candidates.add(Path(pointers['LATEST_CLOCK_WAVE_RELEASE.txt']))
    for folder in sorted(candidates):
        if not (folder/'scripts/run_clock_wave.py').exists():continue
        files={p.relative_to(folder).as_posix():sha256(p) for sub in ('src','scripts','configs','knowledge','data/curated')
               for p in sorted((folder/sub).rglob('*')) if p.is_file() and p.suffix in {'.py','.json','.jsonl','.yaml','.sh'}}
        releases.setdefault(object_hash(files),[]).append(str(folder))
    for r in runs:r['matching_source_releases']=releases.get(r.get('code_hash_recorded'),[])
    revisions=[r for r in runs if r.get('revision_config_and_plan')]
    # Completed, verified revisions take priority over incomplete ones, independent of LATEST.
    complete=[r for r in revisions if r['targeted_complete']]
    evidence=[r for r in revisions if r['fresh_tasks_verified']]
    selected=sorted(complete or evidence,key=lambda r:Path(r['run'],'queue_status.json').stat().st_mtime)[-1] if (complete or evidence) else None
    action='export_completed_revision' if complete else 'export_partial_revision_do_not_restart' if evidence else 'inspect_existing_revision_do_not_duplicate' if revisions else 'targeted_revision_not_found'
    unresolved=[r['run'] for r in runs if r.get('audit_error') or not r.get('config_sha256')]
    if unresolved and not selected:action='inspect_uninitialized_or_unreadable_run'
    return {'scope':'local server filesystem scan only; no SSH or remote execution implied',
            'roots':[str(p) for p in roots],'pointers':pointers,'runs':runs,'selected_run':selected['run'] if selected else None,
            'next_action':action,'unresolved':unresolved,
            'latest_matches_selected':bool(selected and pointers['LATEST_CLOCK_WAVE.txt']==selected['run']),
            'pk2_execution_requested':False,'training_started_by_audit':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--private-root',type=Path,required=True)
    p.add_argument('--extra-run-root',action='append',default=[]);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--export-found',action='store_true');a=p.parse_args()
    result=scan(a.private_root,a.extra_run_root)
    write_json(a.out,result)
    if a.export_found and result['selected_run']:
        from pack_clock_wave_report import pack
        result['exported_report']=str(pack(result['selected_run'],require_revision=True))
    write_json(a.out,result)
    print(json.dumps({'audit':str(a.out),'next_action':result['next_action'],'selected_run':result['selected_run'],
                      'latest_matches_selected':result['latest_matches_selected'],'exported_report':result.get('exported_report')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()

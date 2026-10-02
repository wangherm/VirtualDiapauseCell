"""Thin, sequential task runner. Failures isolate branches; no performance gate or silent fallback."""
from pathlib import Path
import argparse
import contextlib
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from vdc.io import read_json, write_json, read_jsonl, object_hash, sha256

TASKS = {
    'data_dauer': ([], 'data/dauer'), 'data_ard': (['data_dauer'], 'data/ard'),
    'clock_reference': (['data_dauer'], 'clock_reference'),
    'state_base': (['clock_reference'], 'state_base'), 'state_ard': (['data_ard'], 'state_ard'),
    'state_readout': (['state_base'], 'state_readout'), 'waves': (['state_base'], 'waves'),
    'endpoint': (['data_ard'], 'endpoint'), 'endpoint_state_ard': (['state_ard'], 'endpoint_state_ard'),
    'transition': (['clock_reference'], 'transition'), 'transition_state_base': (['state_base'], 'transition_state_base'),
    'functional': ([], 'functional'), 'knowledge_corpus': ([], 'knowledge_corpus'),
    'qwen_smoke': (['knowledge_corpus'], 'qwen_smoke'), 'knowledge': (['qwen_smoke'], 'knowledge'),
    'knowledge_base_eval': (['knowledge_corpus'], 'knowledge_base_eval'),
    'knowledge_adapter_eval': (['knowledge'], 'knowledge_adapter_eval'),
    'semantics': (['knowledge', 'data_dauer'], 'semantics'),
    'state_matched_zero': (['semantics', 'clock_reference'], 'state_matched_zero'),
    'state_semantic': (['semantics', 'clock_reference'], 'state_semantic'),
    'transition_state_semantic': (['state_semantic'], 'transition_state_semantic'),
    'regulon': ([], 'regulon'), 'expression_depth': (['functional'], 'expression_depth'),
}
GPU_TASKS = {'qwen_smoke', 'knowledge', 'knowledge_base_eval', 'knowledge_adapter_eval', 'semantics'}
RELOAD_TASKS = {'clock_reference', 'state_base', 'state_ard', 'state_readout', 'waves', 'endpoint',
    'endpoint_state_ard', 'transition', 'transition_state_base', 'functional', 'state_matched_zero',
    'state_semantic', 'transition_state_semantic'}


def utc(): return dt.datetime.now(dt.timezone.utc).isoformat()


def source_fingerprint():
    files = [p for folder in ('src', 'scripts', 'configs', 'knowledge', 'data/curated') for p in (ROOT/folder).rglob('*')
             if p.is_file() and p.suffix in {'.py', '.json', '.jsonl', '.yaml', '.sh'}]
    return object_hash({p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(files)})


def inventory(path):
    return {p.relative_to(path).as_posix(): sha256(p) for p in sorted(path.rglob('*')) if p.is_file() and p.suffix != '.partial'}


def verify_inventory(path, files):
    if not files: return False
    return inventory(path)==files


def execution_metadata(task,path):
    result={'steps_or_epochs_completed':None,'evaluation_scope':'development_only','science_status':'unvalidated'}
    if task in {'state_base','state_ard','state_matched_zero','state_semantic'}:
        result.update(fit_kind='gradient_training',steps_or_epochs_completed=read_json(path/'metrics.json')['steps_completed'])
    elif task=='knowledge':
        m=read_json(path/'status.json');result.update(fit_kind='gradient_training',steps_or_epochs_completed={'epochs':m['epochs_completed'],'optimizer_steps':m['optimizer_steps']})
    elif task=='qwen_smoke':result.update(fit_kind='software_smoke_not_formal_training',steps_or_epochs_completed=2)
    elif task in {'clock_reference','waves'}:result['fit_kind']='reference_fit'
    elif task.startswith(('endpoint','transition')) or task in {'functional','state_readout'}:result['fit_kind']='regression_fit'
    else:result['fit_kind']='data_processing_or_inference'
    if (path/'reload.json').exists():result['save_reload']=read_json(path/'reload.json')
    if (path/'dataset/response.json').exists():
        m=read_json(path/'dataset/response.json');result['data_scope']={k:m[k] for k in ('context_id','representation_id','mode')}
    elif (path/'run.json').exists():result['data_scope']=read_json(path/'run.json').get('scope')
    return result


@contextlib.contextmanager
def exclusive_run(run):
    f = (run/'.runner.lock').open('a+b')
    try:
        f.seek(0); f.write(b'0'); f.flush(); f.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally: f.close()


def worker(args):
    import torch
    run = args.run; c = read_json(run/'config.json'); task = args.task
    torch.set_num_threads(c['cpu_threads'])
    from vdc.alpha_numeric import (reference_task, state_task, classification_task, waves_task,
        response_task, functional_task, reload_check)
    if args.reload:
        reload_check(run, task); return
    if task == 'data_dauer':
        from prepare_alpha_data import prepare_dauer
        prepare_dauer(run/'data/dauer', args.download)
    elif task == 'data_ard':
        from prepare_alpha_data import prepare_ard
        prepare_ard(run/'data/ard', args.download)
    elif task == 'clock_reference': reference_task(run)
    elif task in {'state_base','state_ard','state_matched_zero','state_semantic'}: state_task(run,task,c)
    elif task == 'state_readout': classification_task(run)
    elif task == 'waves': waves_task(run)
    elif task.startswith('endpoint'): response_task(run,'endpoint','state_ard' if task!='endpoint' else 'observed')
    elif task.startswith('transition'): response_task(run,'transition',task.removeprefix('transition_') if task!='transition' else 'observed')
    elif task == 'functional': functional_task(run,ROOT/'data/curated/functional_fig1b.json')
    elif task == 'knowledge_corpus':
        from vdc.knowledge import audit_knowledge
        records=read_jsonl(ROOT/'knowledge/alpha/corpus.jsonl')
        report=audit_knowledge(records,set(c['excluded_families']))
        report.update({'sources':read_json(ROOT/'knowledge/alpha/sources.json'), 'label_source':'source_curated_weak_reference',
            'corpus_hash':object_hash(records), 'train':sum(r['split']=='train' for r in records), 'validation':sum(r['split']=='validation' for r in records)})
        write_json(run/'knowledge_corpus/audit.json',report)
    elif task in GPU_TASKS:
        records=read_jsonl(ROOT/'knowledge/alpha/corpus.jsonl'); name=c['resolved_model']; revision=c['model_revision']
        from vdc.knowledge_train import smoke_knowledge, fit_knowledge, evaluate_knowledge
        if task not in {'qwen_smoke','knowledge'}:
            from vdc.knowledge_train import model_fingerprint
            write_json(run/task/'base_verification.json',model_fingerprint(name,revision))
        kwargs={'max_length':c['max_length'],'allow_download':args.allow_model_download,'quantized':c['quantized'],'excluded_families':c['excluded_families']}
        if task=='qwen_smoke': smoke_knowledge(records,run/task,name,revision,**kwargs)
        elif task=='knowledge': fit_knowledge(records,run/task,name,revision,epochs=c['knowledge_epochs'],smoke_dir=run/'qwen_smoke',**kwargs)
        elif task in {'knowledge_base_eval','knowledge_adapter_eval'}:
            evaluate_knowledge(records,run/task,name,revision,adapter=run/'knowledge/adapter' if task=='knowledge_adapter_eval' else None,allow_download=args.allow_model_download)
        else:
            from vdc.knowledge import embed_objects
            from vdc.contracts import ObservationBundle
            b=ObservationBundle.load(run/'data/dauer')
            go={r['id']:r for r in read_json(ROOT/'knowledge/snapshots/go_2026-10-01/quickgo_response.json')['results']}
            objects=[]
            for fid in b.feature_ids:
                term=go[fid]; desc=term['name']+'. '+term['definition']['text']
                objects.append({'record_id':fid,'object_id':fid,'study_family':'GO_20261001','source_ref':'https://www.ebi.ac.uk/QuickGO/term/'+fid,
                    'split':'train','reviewed':True,'kind':'object_description','text':desc})
            embed_objects(objects,run/task,name,revision,device='cuda',allow_download=args.allow_model_download,adapter=run/'knowledge/adapter')
    else: raise ValueError('Unknown runnable task '+task)


def summarize(run, statuses):
    missing = [k for k,v in statuses.items() if v['execution_status']!='completed']
    summary={'overall_status':'running' if any(v['execution_status']=='running' for v in statuses.values()) else 'partial' if missing else 'completed', 'updated':utc(), 'missing_modules':missing,
        'completed_modules':[k for k,v in statuses.items() if v['execution_status']=='completed'],
        'science_status':'unvalidated','internal_killifish_used':False,'old_test_used':False,'modules':statuses}
    write_json(run/'module_status.json',summary)
    import csv
    with (run/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['module','execution_status','artifact','reason','science_status'])
        for k,v in statuses.items(): w.writerow([k,v['execution_status'],v.get('artifact',''),v.get('reason',''),'unvalidated'])
    lines=['# All-Module Alpha','',f"Overall: **{summary['overall_status']}**. Scientific validation: **unvalidated**.",'',
        'Only development splits are used. Completion means real execution, not better-than-baseline biology.',
        'Source-curated knowledge pilot: 18 train questions / two families, 9 validation questions / one family.',
        'Functional model uses measured group counts and protocol/history only. Expression-to-depth is unavailable.','',
        '| Module | Execution | Reason |','|---|---|---|']
    for k,v in statuses.items(): lines.append(f"| {k} | {v['execution_status']} | {v.get('reason','')} |")
    lines += ['', '## Actual numeric reports', '']
    for name in ('state_base','state_ard','state_readout','waves','endpoint','endpoint_state_ard','transition','transition_state_base','functional','state_matched_zero','state_semantic','transition_state_semantic'):
        for leaf in ('result.json','diagnostics.json'):
            p=run/name/leaf
            if p.exists() and statuses.get(name,{}).get('execution_status')=='completed':
                lines += [f'### {name}', '```json',json.dumps(read_json(p),ensure_ascii=False,indent=2),'```','']
    (run/'REPORT_CN.md').write_text('\n'.join(lines),encoding='utf-8')
    return summary


def orchestrate(args):
    run=args.run.resolve();run.mkdir(parents=True,exist_ok=True)
    c=read_json(args.config)
    c['resolved_model']=args.model_path or c['model_name']
    signature=object_hash({'code':source_fingerprint(),'config':c})
    with exclusive_run(run):
        if (run/'run_manifest.json').exists():
            old=read_json(run/'run_manifest.json')
            if not args.resume: raise FileExistsError('Existing run: use --resume with identical code/config')
            if old['signature']!=signature: raise ValueError('Code/config changed: choose a NEW run directory; old artifacts preserved')
        else:
            write_json(run/'config.json',c)
            git=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True)
            dirty=subprocess.run(['git','status','--short'],cwd=ROOT,capture_output=True,text=True)
            freeze=subprocess.run([sys.executable,'-m','pip','freeze'],cwd=ROOT,capture_output=True,text=True)
            (run/'pip_freeze.txt').write_text(freeze.stdout,encoding='utf-8')
            write_json(run/'run_manifest.json',{'created':utc(),'signature':signature,'code_fingerprint':source_fingerprint(),
                'git_commit':git.stdout.strip(),'git_status_at_start':dirty.stdout,'python':sys.version,'platform':platform.platform(),'execution_mode':'experimental',
                'gpu_execution_requested':args.gpu,'data_roles':'development_only; original test/VAL/internal frozen'})
        statuses=read_json(run/'module_status.json')['modules'] if (run/'module_status.json').exists() else {
            k:{'execution_status':'pending','reason':'Not executed yet'} for k in TASKS}
        for task,(deps,relative) in TASKS.items():
            previous=statuses.get(task,{})
            if previous.get('execution_status')=='completed':
                if not verify_inventory(run/relative,previous.get('files',{})): raise ValueError('Completed artifact changed: '+task)
                print('VERIFIED RESUME '+task,flush=True);continue
            unmet=[d for d in deps if statuses.get(d,{}).get('execution_status')!='completed']
            if unmet or (task in GPU_TASKS and not args.gpu):
                statuses[task]={'execution_status':'blocked_dependency','reason':'Dependencies: '+','.join(unmet) if unmet else 'Real GPU execution pending; run --gpu on AutoDL'}
                summarize(run,statuses);continue
            if task in {'regulon','expression_depth'}:
                reason='No admitted versioned TF-target membership; GO programmes are not regulons' if task=='regulon' else 'Source S1 audited: functional histories 1/10/20d versus RNA 1/4/15/30d; no cohort mapping. Group functional baseline runs separately; no expression depth labels'
                statuses[task]={'execution_status':'blocked_data','reason':reason};summarize(run,statuses);continue
            target=run/relative
            if target.exists():
                archive=run/'failed_attempts'/f'{task}_{dt.datetime.now().strftime("%Y%m%dT%H%M%S_%f")}'
                archive.parent.mkdir(exist_ok=True);target.rename(archive)
            statuses[task]={'execution_status':'running','started':utc(),'artifact':relative};summarize(run,statuses)
            print('START '+task,flush=True)
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--run',str(run),'--task',task]
            if args.download:cmd.append('--download')
            if args.allow_model_download:cmd.append('--allow-model-download')
            logs=run/'logs';logs.mkdir(exist_ok=True)
            with (logs/(task+'.log')).open('a',encoding='utf-8') as f:
                f.write('\nSTART '+utc()+'\n');f.flush()
                proc=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
                if proc.returncode==0 and task in RELOAD_TASKS:
                    proc=subprocess.run(cmd+['--reload'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            if proc.returncode:
                statuses[task].update(execution_status='failed',exit_code=proc.returncode,reason='See logs/'+task+'.log')
            else:
                statuses[task].update(execution_status='completed',files=inventory(target),**execution_metadata(task,target))
            statuses[task]['finished']=utc();summarize(run,statuses)
            print(statuses[task]['execution_status'].upper()+' '+task,flush=True)
        # Internal registry is built from only completed, checksum-verified branches, even for a partial run.
        from vdc.experimental import build_registry, exercise_app
        try:
            build_registry(run);exercise_app(run)
            statuses['experimental_interface']={'execution_status':'completed','artifact':'experimental_release',
                'files':inventory(run/'experimental_release'),'reason':'Actual saved models invoked through local ASGI requests; no public server'}
        except Exception as exc:
            write_json(run/'experimental_error.json',{'error':str(exc),'traceback':traceback.format_exc()})
            statuses['experimental_interface']={'execution_status':'failed','reason':str(exc)}
        final=summarize(run,statuses)
        print(json.dumps({'overall_status':final['overall_status'],'report':str(run/'REPORT_CN.md'),'missing':final['missing_modules']},indent=2),flush=True)
        return 2 if final['missing_modules'] else 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/all_modules_alpha.json')
    p.add_argument('--download',action='store_true');p.add_argument('--gpu',action='store_true')
    p.add_argument('--allow-model-download',action='store_true');p.add_argument('--model-path');p.add_argument('--resume',action='store_true')
    p.add_argument('--task',choices=list(TASKS));p.add_argument('--reload',action='store_true')
    a=p.parse_args()
    if a.task:worker(a)
    else:sys.exit(orchestrate(a))

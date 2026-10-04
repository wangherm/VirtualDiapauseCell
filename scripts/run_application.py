"""Freeze once, then query approved reserved/exploratory samples and re-evaluate knowledge."""
from pathlib import Path
import argparse,concurrent.futures,json,os,subprocess,sys,traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_all_modules import utc,source_fingerprint,exclusive_run
from vdc.io import read_json,read_jsonl,write_json,object_hash,sha256
from vdc.application import prepare_freeze,apply_group,inventory,verify_freeze
from vdc.pk1_assets import verify_files


def worker(run,task):
    import torch
    torch.set_num_threads(2)
    run=Path(run);c=read_json(run/'config.json');source=Path(c['source']);private=Path(c['private_root'])
    os.environ['VDC_ROLE_MANIFEST']=str(private/'sample_roles.json');os.environ.pop('VDC_DEVELOPMENT_FOLD',None)
    if task=='freeze':
        prepare_freeze(source,private,run,c['protocol'],c['code'],c.get('supplement'))
    elif task in {'bulk','single_exit','core','core_celltypes'}:
        apply_group(run,private,task,c.get('supplement'))
    elif task=='stress':
        from vdc.application_stress import apply_stress
        if not c.get('supplement'):raise ValueError('Normalised stress supplement absent; raw counts pending')
        apply_stress(run,c['supplement'])
    elif task in {'knowledge_base','knowledge_domain'}:
        from vdc.application_knowledge import source_independence
        from vdc.knowledge_train import evaluate_knowledge
        old=read_jsonl(source/'tasks/corpus/corpus.jsonl');new=read_jsonl(ROOT/'knowledge/application/evaluation_only.jsonl')
        out=run/'knowledge'/task;write_json(out/'source_audit.json',source_independence(old,new))
        statuses=read_json(source/'queue_status.json')['tasks'];verify_files(source/'tasks/corpus',statuses['corpus']['files'])
        adapter=None
        if task=='knowledge_domain':
            selected=read_json(source/'tasks/select_adapter/selection.json')['selected']['job']
            verify_files(source/'tasks'/selected,statuses[selected]['files']);adapter=source/'tasks'/selected/'trained/adapter'
        prior=read_json(source/'config.json')
        evaluate_knowledge(old+new,out,prior['model_path'],prior['model_revision'],adapter=adapter,clean_evidence=True)
    else:raise ValueError('Unknown application task')


def report(run,statuses):
    run=Path(run);rows=[]
    for group in ('bulk','single_exit','core','core_celltypes','stress'):
        p=run/'applications'/group/'result.json'
        if p.exists():rows.extend(read_json(p)['rows'])
    result={'tasks':statuses,'queried_profiles':sum(r['status']=='queried' for r in rows),
        'unsupported_profiles':sum(r['status']=='unsupported' for r in rows),
        'profile_count_is_not_independent_sample_count':True,
        'new_state_or_qwen_training':False,'reserved_for_fitting':False,
        'stress_counts':'pending_user_source; archived normalised expression descriptive substitute only',
        'future_prediction':'unsupported: no development-fitted killifish longitudinal transition; late expression queries are location, not forecasts',
        'expression_depth':'unavailable_no_matched_functional_labels','science_status':'retrospective_experimental_not_validated',
        'updated':utc()}
    write_json(run/'application_status.json',result)
    lines=['# 冻结候选的回顾性应用','','使用当前表达进行定位，不宣称预测未来。预留结果不用于选模型；stalled 不计算恢复失败准确率。',
        f"已查询 profiles：{result['queried_profiles']}；不支持：{result['unsupported_profiles']}。profile 数不是独立 pool 数。",'',
        '|阶段|状态|说明|','|---|---|---|']
    for name,s in statuses.items():lines.append(f"|{name}|{s['status']}|{s.get('reason','')}|")
    lines+=['','## 样本覆盖','','|样本|细胞类型|状态|说明|','|---|---|---|---|']
    for r in rows:lines.append(f"|{r.get('sample_key','')}|{r.get('cell_type','')}|{r['status']}|{r.get('reason',r.get('scope',''))}|")
    (run/'REPORT_CN.md').write_text('\n'.join(lines),encoding='utf-8');return result


def orchestrate(a):
    c={'source':str(a.source.resolve()),'private_root':str(a.private_root.resolve()),
       'supplement':str(a.supplement.resolve()) if a.supplement else None,
       'protocol':read_json(ROOT/'configs/application_v1.json'),'code':source_fingerprint()}
    run=a.run.resolve();run.mkdir(parents=True,exist_ok=True)
    with exclusive_run(run):
        if (run/'config.json').exists():
            if read_json(run/'config.json')!=c:raise ValueError('Resume requires identical application configuration and code')
        else:
            from vdc.pk2_runtime import hardware
            write_json(run/'config.json',c);write_json(run/'hardware.json',hardware())
        groups=('bulk','single_exit','core','core_celltypes','stress')
        tasks=['freeze',*groups,'knowledge_base','knowledge_domain']
        states=read_json(run/'application_status.json')['tasks'] if (run/'application_status.json').exists() else {t:{'status':'pending'} for t in tasks}
        def execute(task):
            if states[task]['status']=='completed':
                verify_files(run,states[task]['files']);return states[task]
            folder=run/'logs';folder.mkdir(exist_ok=True)
            write_json(run/'task_progress'/(task+'.json'),{'status':'running','started':utc()})
            with (folder/(task+'.log')).open('a',encoding='utf-8') as log:
                print('START',task,flush=True)
                proc=subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--run',str(run),'--worker',task],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if proc.returncode:
                p=run/'failures'/(task+'.json');reason=read_json(p)['reason'] if p.exists() else 'See log'
                result={'status':'failed' if task=='freeze' else 'blocked_or_failed','reason':reason,'exit_code':proc.returncode}
                write_json(run/'task_progress'/(task+'.json'),result);return result
            paths=[run/'freeze.json'] if task=='freeze' else list((run/('knowledge' if task.startswith('knowledge') else 'applications')/task).rglob('*'))
            print('COMPLETED',task,flush=True)
            write_json(run/'task_progress'/(task+'.json'),{'status':'completed','finished':utc()})
            return {'status':'completed','finished':utc(),'files':{p.relative_to(run).as_posix():sha256(p) for p in paths if p.is_file()}}
        # Qwen inference queue is independent of freeze and CPU application work; no new LoRA fits.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as gpu:
            future=gpu.submit(execute,'knowledge_base')
            states['freeze']=execute('freeze');report(run,states)
            if states['freeze']['status']=='completed':
                verify_freeze(run)
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as cpu:
                    pending={cpu.submit(execute,t):t for t in groups}
                    for f in concurrent.futures.as_completed(pending):states[pending[f]]=f.result();report(run,states)
            else:
                for t in groups:states[t]={'status':'blocked','reason':'Freeze did not complete; no reserved model queries performed'}
            states['knowledge_base']=future.result();report(run,states)
            states['knowledge_domain']=gpu.submit(execute,'knowledge_domain').result()
        result=report(run,states)
        if (run/'freeze.json').exists():
            verify_freeze(run);write_json(run/'application_snapshot.json',{'freeze_id':read_json(run/'freeze.json')['freeze_id'],
                'files':{p.relative_to(run).as_posix():sha256(p) for folder in ('applications','knowledge') for p in (run/folder).rglob('*') if p.is_file()},
                'access':'loopback_only; private results; saved frozen queries, not arbitrary uploads'})
        return 0 if all(s['status']=='completed' for s in states.values()) else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--worker');p.add_argument('--source',type=Path);p.add_argument('--private-root',type=Path);p.add_argument('--supplement',type=Path)
    a=p.parse_args()
    if a.worker:
        try:worker(a.run,a.worker)
        except Exception as exc:
            traceback.print_exc();write_json(a.run/'failures'/(a.worker+'.json'),{'reason':str(exc),'type':type(exc).__name__});sys.exit(1)
    else:
        if not a.source or not a.private_root:p.error('--source and --private-root required')
        sys.exit(orchestrate(a))

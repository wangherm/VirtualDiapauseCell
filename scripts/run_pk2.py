"""PK2 full ready queue: isolated workers, per-task dependencies, real artifacts."""
from pathlib import Path
import argparse,collections,copy,importlib.metadata,json,os,shutil,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_all_modules import utc,inventory,verify_inventory,exclusive_run,source_fingerprint
from vdc.io import read_json,read_jsonl,write_json,object_hash,sha256
from vdc.pk2_data import DataUnavailable
VIEWS={'bulk':'bulk','core_pool':'core','core_pool_celltype':'core_celltypes'}

def build_plan(c):
    tasks={}
    def add(name,kind,deps=(),resource='cpu',priority=0,**params):
        tasks[name]={'id':name,'kind':kind,'depends':list(deps),'resource':resource,'priority':priority,'params':params}
    add('audit','audit');add('public_reuse','public_reuse');add('annotations','annotations',resource='cpu')
    add('corpus','corpus');add('benchmark','benchmark',['audit'],resource='profile',priority=-4)
    for view in VIEWS.values():add('input_'+view,'local_input',['audit'],view=view)
    for study in ['GSE202844','GSE124109','GSE221467','GSE3169']:
        add('data_'+study,'public_input',['audit','annotations'],study=study)
    for name,studies in c['source_sets'].items():
        add('source_'+name,'source_set',['audit','public_reuse']+['data_'+s for s in studies if s not in {'GSE288723','GSE291659'}],source_set=name)
    jobs=read_jsonl(ROOT/'configs/pk2_jobs.jsonl')
    routes={j['route']:j['route_config'] for j in jobs if j['family']=='killifish_adaptation'}
    add('embeddings_base','embeddings',['audit'],resource='gpu',priority=12,adapter=None)
    qwen=[j['job_id'] for j in jobs if j['family']=='qwen_domain']
    add('select_adapter','select_adapter',qwen,priority=12)
    add('embeddings_domain','embeddings',['audit','select_adapter'],resource='gpu',priority=12,adapter='selected')
    for job in jobs:
        j=copy.deepcopy(job);name=j['job_id'];family=j['family'];deps=[]
        if family=='public_pretraining':deps=['source_'+j['source_set'],'benchmark'];j['bundle_task']='source_'+j['source_set']
        elif family=='qwen_domain':
            add(name,'qwen',['corpus'],resource='gpu',priority=10,job=j);continue
        else:
            view=VIEWS.get(j.get('view'),'core');j['view_internal']=view
            if family in {'killifish_adaptation','semantic_increment'}:inp='input_'+view
            elif family=='pool_generalization':
                inp=f'input_poolcv_{j["fold_index"]}'
                if inp not in tasks:add(inp,'local_input',['audit'],view='core',fold_index=j['fold_index'])
            elif family=='bulk_learning_curve':
                inp=f'input_learning_{j["fraction"]}_{j["unit_subset_seed"]}'
                if inp not in tasks:add(inp,'local_input',['audit'],view='bulk',fraction=j['fraction'],subset_seed=j['unit_subset_seed'])
            elif family=='feature_resolution':
                inp=f'input_features_{j["feature_view"]}_{view}'
                if inp not in tasks:add(inp,'local_input',['audit'],view=view,mode='gene' if j['feature_view']=='local_gene_panel2000' else 'fine')
            else:raise ValueError('No executor for '+family)
            j['bundle_task']=inp;deps=[inp,'benchmark'];route=j.get('route','R0');j['route_config']=routes[route]
            parent=routes[route]['source']
            if parent:
                j['parent_task']=f'pretrain_{parent}_{j.get("seed",j.get("algorithm_seed",42))}';deps.append(j['parent_task'])
            if family=='semantic_increment':deps.append('embeddings_base' if j['semantics']=='base' else 'embeddings_domain')
        add(name,'numeric',deps,resource='numeric',priority=j['dispatch_priority'],job=j)
        if family=='bulk_learning_curve' and j['fraction']==1.0:
            tasks[name]['params']['equivalent_candidate']=f'adapt_{j["route"]}_bulk_42'
            tasks[name]['depends'].append(tasks[name]['params']['equivalent_candidate'])
    for view in VIEWS.values():
        logical=next(k for k,v in VIEWS.items() if v==view)
        add('waves_'+view,'waves',['input_'+view,f'adapt_R0_{logical}_42'],priority=18,view=view,state=f'adapt_R0_{logical}_42')
    add('type_waves','type_waves',['input_core_celltypes'],priority=18)
    for name in ['endpoint','transition','functional','public_regulon','local_regulon','observation_stress','public_generalisation']:
        add(name,'analysis',['public_reuse','audit','input_bulk','input_core','input_core_celltypes'],priority=19,analysis=name)
    tasks['observation_stress']['depends']+=['adapt_R0_bulk_42','adapt_R0_core_pool_42']
    add('history_future','analysis',['public_reuse'],priority=19,analysis='history_future')
    add('array_series','analysis',['data_GSE3169'],priority=19,analysis='array_series')
    add('source_gaps','analysis',priority=19,analysis='source_gaps')
    add('nhdf_comparison','analysis',priority=19,analysis='nhdf_comparison')
    for mode in ['endpoint','transition']:add('composed_'+mode,'analysis',['pretrain_P-R_42','source_P-R','public_reuse'],priority=20,analysis='composed_'+mode)
    add('cross_platform_waves','analysis',['data_GSE3169','public_reuse'],priority=20,analysis='cross_platform_waves')
    add('knowledge_base_evaluation','knowledge_eval',['corpus'],resource='gpu',priority=11)
    add('register_pk2','register',[j['job_id'] for j in jobs]+[k for k,v in tasks.items() if v['kind'] in {'waves','analysis','type_waves'}],priority=90)
    for task in tasks.values():
        if task['resource']=='gpu' and 'benchmark' not in task['depends']:task['depends'].append('benchmark')
    return tasks

def worker(run,name):
    import torch,numpy as np
    run=Path(run);c=read_json(run/'config.json');task=read_json(run/'plan.json')[name];p=task['params'];kind=task['kind'];out=run/'tasks'/name
    torch.set_num_threads(c['cpu_threads']);os.environ['VDC_ROLE_MANIFEST']=str(Path(c['private_root'])/'sample_roles.json');os.environ.pop('VDC_DEVELOPMENT_FOLD',None)
    private=Path(c['private_root']);pk1=Path(c['pk1_run']);acquired=Path(c['acquire_run']);base=run/'tasks'
    features=read_json(base/'audit/audit.json')['feature_ids'] if (base/'audit/audit.json').exists() else None
    from vdc.pk1_assets import private_audit,checked_member,verify_files
    from vdc.pk2_data import prepare_public,prepare_dauer_arrays,combine_sources,local_input
    from vdc.contracts import ObservationBundle
    from prepare_public_pilot import fetch_locked
    if kind=='audit':private_audit(private,out)
    elif kind=='public_reuse':
        m=read_json(pk1/'module_status.json')['modules']['public_source']
        if m['execution_status']!='completed':raise DataUnavailable('PK1 public source did not complete')
        source=checked_member(pk1,m['artifact']);verify_files(source,m['files']);shutil.copytree(source,out)
        write_json(out/'PK2_reuse.json',{'source':str(source),'inventory_hash':object_hash(m['files']),'is_training':False})
    elif kind=='annotations':
        for e in read_json(ROOT/'configs/pk2_annotations.json')['files']:fetch_locked(e,ROOT/'data/raw/pk2_annotations',True)
        entry=read_json(ROOT/'configs/public_pilot_sources.json')['files']['annotations'];fetch_locked(entry,ROOT/'data/raw/GSE288723',True)
        write_json(out/'result.json',{'status':'verified','lock_sha256':sha256(ROOT/'configs/pk2_annotations.json')})
    elif kind=='public_input':
        study=p['study'];gaf=ROOT/'data/raw/GSE288723'/read_json(ROOT/'configs/public_pilot_sources.json')['files']['annotations']['file']
        if study=='GSE3169':prepare_dauer_arrays(acquired,c['public_raw'],gaf,features,out/'bundle')
        else:prepare_public(study,acquired,ROOT/'data/raw/pk2_annotations'/('rnorvegicus_go.tsv' if study=='GSE124109' else 'mmusculus_go.tsv'),features,out/'bundle')
    elif kind=='source_set':
        paths=[base/'public_reuse'/('data/dauer' if s=='GSE288723' else 'data/ard') if s in {'GSE288723','GSE291659'} else base/('data_'+s)/'bundle' for s in c['source_sets'][p['source_set']]]
        combine_sources(paths,out/'bundle',features)
    elif kind=='local_input':
        local_input(private/'prepared'/p['view'],out,annotation=ROOT/'knowledge/snapshots/ensembl_2026-10-03/nfurzeri_go.tsv',**{k:v for k,v in p.items() if k!='view'})
    elif kind=='benchmark':
        from vdc.pk2_runtime import benchmark
        benchmark(private/'prepared/bulk',out,c)
    elif kind=='numeric':
        from vdc.pk2_numeric import train_job
        from vdc.pk1_numeric import semantic_cache
        job=p['job'];inp=base/job['bundle_task'];fold=inp/'fold.json'
        if fold.exists():os.environ['VDC_DEVELOPMENT_FOLD']=str(fold)
        sem=None;provenance=None
        if job.get('semantics') in {'base','domain','shuffle'}:
            which='base' if job['semantics']=='base' else 'domain'
            sem,provenance=semantic_cache(base/('embeddings_'+which),features,'frozen_base' if which=='base' else 'domain_adapter')
            if job['semantics']=='shuffle':
                permutation=np.random.default_rng(c['semantic_shuffle_seed']).permutation(len(sem));sem=sem[permutation];provenance={**provenance,'permutation':permutation.tolist(),'ablation':'shuffled_domain'}
        parent=base/job['parent_task']/'model' if job.get('parent_task') else None
        device=read_json(base/'benchmark/profile.json')['numeric_device']
        torch.set_num_threads(read_json(base/'benchmark/profile.json')['numeric_threads_requested'])
        if p.get('equivalent_candidate'):
            from vdc.pk2_numeric import reuse_equivalent
            candidate=p['equivalent_candidate'];spec=read_json(run/'plan.json')[candidate]['params']['job']
            if reuse_equivalent(inp/'bundle',base/spec['bundle_task']/'bundle',base/candidate,out,job,spec,c):return
        train_job(inp/'bundle',out,job,c,parent,sem,provenance,device,c.get('integration_steps'))
    elif kind=='corpus':
        from vdc.pk2_knowledge import corpus
        corpus(acquired,ROOT/'knowledge/alpha/corpus.jsonl',out)
    elif kind=='qwen':
        from vdc.knowledge_train import smoke_knowledge,fit_knowledge
        records=read_jsonl(base/'corpus/corpus.jsonl');j=p['job'];kw={'seed':j['seed'],'learning_rate':j['learning_rate'],'quantized':c['qwen_quantized']}
        smoke_knowledge(records,out/'smoke',c['model_path'],c['model_revision'],**kw)
        fit_knowledge(records,out/'trained',c['model_path'],c['model_revision'],epochs=3,smoke_dir=out/'smoke',**kw)
        import gc
        gc.collect();torch.cuda.empty_cache()
        # Fresh process evaluation must complete before this task unlocks adapter selection.
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--run',str(run),'--qwen-evaluate',name],check=True,cwd=ROOT)
        write_json(out/'result.json',{'training':read_json(out/'trained/status.json'),'evaluation':read_json(out/'evaluation/evaluation.json'),
            'adapter':str(out/'trained/adapter'),'fresh_adapter_training':True,'status':'evaluated_new'})
    elif kind=='select_adapter':
        candidates=[]
        for dep in task['depends']:
            if not (base/dep/'result.json').exists():continue
            status=read_json(base/dep/'trained/status.json');log=read_json(base/dep/'trained/training_log.json');loss=min(r['eval_loss'] for r in log if 'eval_loss' in r)
            candidates.append({'job':dep,'validation_token_loss':loss,'optimizer_steps':status['optimizer_steps']})
        if not candidates:raise DataUnavailable('No newly trained and evaluated domain adapter is available')
        selected=min(candidates,key=lambda r:(r['validation_token_loss'],r['job']))
        write_json(out/'selection.json',{'selected':selected,'candidates':candidates,'selection_rule':'knowledge development token loss only; no private evaluation selection'})
    elif kind=='embeddings':
        from vdc.pk2_knowledge import objects
        from vdc.knowledge import embed_objects
        adapter=None
        if p['adapter']=='selected':adapter=base/read_json(base/'select_adapter/selection.json')['selected']['job']/'trained/adapter'
        embed_objects(objects(features,ROOT),out,c['model_path'],c['model_revision'],device='cuda',adapter=adapter)
    elif kind=='knowledge_eval':
        from vdc.knowledge_train import evaluate_knowledge
        evaluate_knowledge(read_jsonl(base/'corpus/corpus.jsonl'),out,c['model_path'],c['model_revision'],evidence_modes=True)
    elif kind=='waves':
        from vdc.pk1_numeric import waves_task
        waves_task(private/'prepared'/p['view'],base/('input_'+p['view']),base/p['state']/'model',out,ROOT/'knowledge/snapshots/ensembl_2026-10-03/nfurzeri_go.tsv')
    elif kind in {'type_waves','analysis'}:
        from vdc.pk2_analysis import run_analysis
        run_analysis(p.get('analysis','type_waves'),run,out,c)
    elif kind=='register':
        from vdc.pk2_publish import register
        register(run,out)
    else:raise ValueError('No implementation for '+kind)

def status_report(run,plan,statuses):
    logical=[k for k,v in plan.items() if v['kind'] in {'numeric','qwen'}];counts=collections.Counter(statuses[k]['status'] for k in logical)
    active=any(s['status'] in {'ready','running','pending'} for s in statuses.values())
    failed=any(s['status']=='failed' for s in statuses.values())
    result={'planned_training':len(logical),'training_counts':dict(counts),'all_task_counts':dict(collections.Counter(s['status'] for s in statuses.values())),
        'status':'running' if active else 'partial_failed' if failed else 'partial_resource' if any(s['status']=='blocked_resource' for s in statuses.values()) else 'completed_executable_scope',
        'full_research_plan_completed':False,'tasks':statuses,'updated':utc(),'reserved_queries':False,'server_execution_claim':'Only this actual process and its recorded workers'}
    manifest=read_json(run/'run_manifest.json')
    result['integration_only']=manifest.get('integration_only',False)
    write_json(run/'queue_status.json',result)
    lines=['# PK2 全队列执行记录','',f'逻辑训练任务：{len(logical)}；状态：{result["status"]}。',
        '短预算工程联调，不是正式训练。' if result['integration_only'] else '本次使用任务清单的正式预算；是否完成以各工件为准。',
        '下载、旧权重、服务调用不计为新增训练；所有结果只涉及 development。',
        '完整研究目标仍未全部验证，具体缺口和执行失败保留在下面。','',json.dumps(result['training_counts'],ensure_ascii=False),'', '|任务|状态|原因|','|---|---|---|']
    for name,s in statuses.items():lines.append(f'|{name}|{s["status"]}|{s.get("reason","")}|')
    (run/'REPORT_CN.md').write_text('\n'.join(lines),encoding='utf-8');return result

def orchestrate(a):
    import torch
    from vdc.pk2_runtime import hardware
    c=read_json(a.config)
    if c['reserved_queries_enabled']:raise ValueError('This runner does not unlock reserved queries')
    c.update(private_root=str(a.private_root.resolve()),pk1_run=str(a.pk1_run.resolve()),acquire_run=str(a.acquire_run.resolve()),
        public_raw=str(a.public_raw.resolve()),model_path=str(a.model_path.resolve()) if a.model_path else '',integration_steps=a.integration_steps)
    run=a.run.resolve();run.mkdir(parents=True,exist_ok=True);plan=build_plan(c);hw=hardware();gpu=hw['cuda_available']
    for name,t in plan.items():t['argv']=[sys.executable,'-u',str(Path(__file__).resolve()),'--run',str(run),'--worker',name]
    signature=object_hash({'config':c,'code':source_fingerprint(),'roles':sha256(a.private_root/'sample_roles.json'),
        'pk1_manifest':sha256(a.pk1_run/'module_status.json'),'acquisition':sha256(a.acquire_run/'acquisition_status.json'),
        'dependencies':{n:importlib.metadata.version(n) for n in ['torch','numpy','scipy']}})
    with exclusive_run(run):
        if (run/'run_manifest.json').exists():
            if not a.resume or read_json(run/'run_manifest.json')['signature']!=signature:raise ValueError('Resume requires identical source/config/input/environment signature')
        else:
            write_json(run/'config.json',c);write_json(run/'plan.json',plan);write_json(run/'hardware.json',hw)
            write_json(run/'run_manifest.json',{'signature':signature,'started':utc(),'code':source_fingerprint(),'full_queue_logical_jobs':171,'integration_only':a.integration_steps is not None})
        selected=set(plan)
        if a.only:
            selected=set(a.only)
            if not selected.issubset(plan):raise ValueError('Unknown requested job')
            pending=list(selected)
            while pending:
                for dep in plan[pending.pop()]['depends']:
                    if dep not in selected:selected.add(dep);pending.append(dep)
        statuses=read_json(run/'queue_status.json')['tasks'] if (run/'queue_status.json').exists() else {n:{'status':'pending'} for n in plan}
        for n,s in statuses.items():
            if s['status'] in {'evaluated_new','completed','reused_verified'}:
                if not verify_inventory(run/'tasks'/n,s['files']):raise ValueError('Completed artifact changed: '+n)
            elif n not in selected:s.update(status='not_requested',reason='Explicit local/integration selection; not whole-queue completion')
            else:s.update(status='pending')
        active={};(run/'logs').mkdir(exist_ok=True);success={'completed','evaluated_new','reused_verified'};last_heartbeat=0
        try:
            while True:
                for name,(proc,stream,resource) in list(active.items()):
                    code=proc.poll()
                    if code is None:continue
                    stream.close();s=statuses[name];s.update(finished=utc(),exit_code=code)
                    if code==0:
                        s.update(status='evaluated_new' if plan[name]['kind'] in {'numeric','qwen'} else 'completed',files=inventory(run/'tasks'/name))
                        if (run/'tasks'/name/'reuse.json').exists() and plan[name]['kind']=='numeric':s['status']='reused_verified'
                    else:
                        error=run/'tasks'/name/'failure.json';failure=read_json(error) if error.exists() else {'reason':'See worker log'}
                        s.update(status='blocked_data' if code==3 else 'failed',reason=failure['reason'])
                    print(s['status'].upper(),name,flush=True);del active[name]
                for name,t in plan.items():
                    s=statuses[name]
                    if s['status'] not in {'pending','ready'}:continue
                    if t['kind'] in {'register','select_adapter'}:
                        if all(statuses[d]['status'] not in {'pending','ready','running'} for d in t['depends']):s['status']='ready'
                        continue
                    if any(statuses[d]['status'] in {'failed','blocked_data','blocked_dependency','blocked_resource','inapplicable'} for d in t['depends']):
                        s.update(status='blocked_dependency',reason='; '.join(d+':'+statuses[d]['status'] for d in t['depends'] if statuses[d]['status'] not in success));continue
                    if all(statuses[d]['status'] in success for d in t['depends']):s['status']='ready'
                    if t['resource']=='gpu' and not gpu:s.update(status='blocked_resource',reason='No actual CUDA GPU in this execution environment')
                ready=sorted([n for n,s in statuses.items() if s['status']=='ready'],key=lambda n:(plan[n]['priority'],not plan[n]['params'].get('job',{}).get('prefer_first_seed',True),n))
                for name in ready:
                    t=plan[name];resource=t['resource']
                    if resource=='profile':resource='gpu' if gpu else 'cpu'
                    if resource=='numeric':resource='gpu' if read_json(run/'tasks/benchmark/profile.json')['numeric_device'].startswith('cuda') else 'cpu'
                    used=sum(r==resource for _,_,r in active.values());limit=1 if resource=='gpu' else c['cpu_workers']
                    if used>=limit:continue
                    target=run/'tasks'/name
                    if target.exists():
                        archive=run/'failed_attempts'/(name+'_'+str(time.time_ns()));archive.parent.mkdir(exist_ok=True);target.rename(archive)
                    argv=[sys.executable,'-u',str(Path(__file__).resolve()),'--run',str(run),'--worker',name]
                    env=os.environ.copy();env.update(OMP_NUM_THREADS=str(c['cpu_threads']),OPENBLAS_NUM_THREADS=str(c['cpu_threads']),MKL_NUM_THREADS=str(c['cpu_threads']))
                    if resource=='gpu':env['CUDA_VISIBLE_DEVICES']='0'
                    stream=(run/'logs'/f'{name}.log').open('a',encoding='utf-8');proc=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                    statuses[name].update(status='running',started=utc(),pid=proc.pid,argv=argv,resource=resource,log=f'logs/{name}.log')
                    active[name]=(proc,stream,resource);print('START',name,'PID',proc.pid,'RESOURCE',resource,flush=True)
                status_report(run,plan,statuses)
                if time.monotonic()-last_heartbeat>15:
                    from vdc.pk2_runtime import resource_sample
                    write_json(run/'resources.json',{'time':utc(),'active':{n:{'pid':p.pid,'resource':r} for n,(p,_,r) in active.items()},'cpu_slots':c['cpu_workers'],'gpu_slots':int(gpu),
                        'gpu_idle_reason':'no ready GPU task' if not any(r=='gpu' for _,_,r in active.values()) else None,**resource_sample()})
                    with (run/'resource_history.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(read_json(run/'resources.json'))+'\n')
                    last_heartbeat=time.monotonic()
                if not active and not any(s['status']=='ready' for s in statuses.values()):break
                time.sleep(1)
        finally:
            for proc,stream,_ in active.values():
                proc.terminate()
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                stream.close()
        result=status_report(run,plan,statuses);print(json.dumps({k:v for k,v in result.items() if k!='tasks'},indent=2))
        return 2 if any(s['status'] in {'failed','blocked_data','blocked_dependency','blocked_resource','pending'} for s in statuses.values()) else 0

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--worker');p.add_argument('--qwen-evaluate')
    p.add_argument('--config',type=Path,default=ROOT/'configs/pk2_train.json');p.add_argument('--private-root',type=Path);p.add_argument('--pk1-run',type=Path);p.add_argument('--acquire-run',type=Path)
    p.add_argument('--public-raw',type=Path,default=ROOT/'data/raw/pk2_public');p.add_argument('--model-path',type=Path);p.add_argument('--resume',action='store_true');p.add_argument('--only',nargs='+');p.add_argument('--integration-steps',type=int)
    a=p.parse_args()
    if a.worker:
        try:worker(a.run,a.worker)
        except Exception as exc:
            traceback.print_exc();write_json(a.run/'tasks'/a.worker/'failure.json',{'reason':str(exc),'type':type(exc).__name__});sys.exit(3 if isinstance(exc,DataUnavailable) else 1)
    elif a.qwen_evaluate:
        from vdc.knowledge_train import evaluate_knowledge
        c=read_json(a.run/'config.json');out=a.run/'tasks'/a.qwen_evaluate
        evaluate_knowledge(read_jsonl(a.run/'tasks/corpus/corpus.jsonl'),out/'evaluation',c['model_path'],c['model_revision'],adapter=out/'trained/adapter',evidence_modes=True)
    else:
        if not all([a.private_root,a.pk1_run,a.acquire_run]):p.error('--private-root, --pk1-run and --acquire-run are required')
        sys.exit(orchestrate(a))

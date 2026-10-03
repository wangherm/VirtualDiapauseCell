"""PK1 training DAG reuses Alpha utilities and canonical models. No hidden fallback."""
from pathlib import Path
import argparse
import copy
import importlib.metadata
import json
import os
import subprocess
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_all_modules import utc,inventory,verify_inventory,exclusive_run,source_fingerprint
from vdc.io import read_json,write_json,object_hash,sha256


def tasks(c):
    t={
      'audit':([], 'audit'), 'public_source':([], 'public'),
      'knowledge_source':([], 'knowledge_source'),
      'semantics_base':(['audit'], 'semantics_base'),
      'semantics_domain':(['audit','knowledge_source'], 'semantics_domain'),
      'public_regulon':(['public_source'], 'public_regulon'),
      'endpoint':(['public_source'], 'endpoint'), 'transition':(['public_source'], 'transition'),
      'functional':([], 'functional'),
      'public_bridge':(['audit','public_source'], 'public_bridge'),
    }
    for view in c['modalities']:
        t['clock_'+view]=(['audit']+(['clock_core'] if view=='core_celltypes' else []),'clock_'+view)
    for seed in c['seeds']:
        init='initial_'+str(seed);public='pretrained_'+str(seed)
        t[init]=(['audit'],init)
        t[public]=(['public_bridge',init],public)
        for view in c['modalities']:
            for route in c['routes']:
                deps=['clock_'+view,init]
                if route in {'K1','K3','K4','K5'}:deps.append(public)
                if route in {'K2','K3','K5'}:deps.append('semantics_domain')
                if route=='K4':deps.append('semantics_base')
                name=f'state_{view}_{route}_{seed}';t[name]=(deps,name)
        if seed==c['seeds'][0]:
            for view in c['modalities']:
                for route in ('K0','K3'):
                    name=f'waves_{view}_{route}';t[name]=([f'state_{view}_{route}_{seed}'],name)
    return t


def worker(run,task):
    import torch
    run=Path(run);c=read_json(run/'config.json');private=Path(c['private_root']);alpha=Path(c['alpha_run']);out=run/task
    if source_fingerprint()!=read_json(run/'run_manifest.json')['code_fingerprint']:
        raise RuntimeError('Source changed during this run; start a new run with the new code')
    os.environ['VDC_ROLE_MANIFEST']=str(private/'sample_roles.json');torch.set_num_threads(c['cpu_threads'])
    from vdc.pk1_assets import private_audit,import_public,reuse_knowledge
    from vdc.pk1_numeric import clock_task,shared_public_bundle,save_initial_weights,state_route,waves_task
    if task=='audit':private_audit(private,out)
    elif task=='public_source':import_public(alpha,run/'public')
    elif task=='knowledge_source':reuse_knowledge(alpha,out)
    elif task.startswith('semantics_'):
        from vdc.knowledge_train import require_gpu,model_fingerprint
        from vdc.knowledge import embed_objects
        require_gpu();verification=model_fingerprint(c['model_path'],c['model_revision'])
        go={r['id']:r for r in read_json(ROOT/'knowledge/snapshots/go_2026-10-01/quickgo_response.json')['results']}
        ids=read_json(run/'audit/audit.json')['feature_ids'];objects=[]
        for fid in ids:
            term=go[fid]
            objects.append({'record_id':fid,'object_id':fid,'study_family':'GO_20261001','source_ref':'https://www.ebi.ac.uk/QuickGO/term/'+fid,
                'split':'train','reviewed':True,'kind':'object_description','text':term['name']+'. '+term['definition']['text']})
        adapter=run/'knowledge_source/adapter' if task=='semantics_domain' else None
        embed_objects(objects,out,c['model_path'],c['model_revision'],device='cuda',allow_download=False,adapter=adapter)
        write_json(out/'base_verification.json',verification)
    elif task=='public_regulon':
        from prepare_public_pilot import fetch_locked
        from vdc.regulon import public_regulon_task
        net=fetch_locked(read_json(ROOT/'configs/pk1_regulon_source.json'),ROOT/'data/raw/pk1_public',c['download'])
        gaf=fetch_locked(read_json(ROOT/'configs/public_pilot_sources.json')['files']['annotations'],ROOT/'data/raw/GSE288723',c['download'])
        public_regulon_task(run/'public',net,gaf,out)
    elif task in {'endpoint','transition'}:
        from vdc.pk1_baselines import response_with_controls
        response_with_controls(run/'public',task,out)
    elif task=='functional':
        from vdc.pk1_baselines import bounded_functional
        bounded_functional(ROOT/'data/curated/functional_fig1b.json',out)
    elif task=='public_bridge':
        shared_public_bundle(run/'public/data/dauer',read_json(run/'audit/audit.json')['feature_ids'],out)
    elif task.startswith('clock_'):
        view=task.removeprefix('clock_')
        clock_task(private/'prepared'/view,out,run/'clock_core/reference' if view=='core_celltypes' else None)
    elif task.startswith('initial_'):
        cfg={**c['state_config'],'seed':int(task.split('_')[-1])}
        save_initial_weights(read_json(run/'audit/audit.json')['feature_ids'],c['semantic_dim'],cfg,out/'initial.pt')
    elif task.startswith('pretrained_'):
        seed=int(task.split('_')[-1]);local=copy.deepcopy(c)
        local['state_config'].update(seed=seed,clock_weight=0);local['state_steps']=c['public_pretraining_steps']
        state_route(run/'public_bridge',out,local,run/f'initial_{seed}/initial.pt','public',c['semantic_dim'])
    elif task.startswith('state_'):
        body=task.removeprefix('state_');view,route,seed=body.rsplit('_',2);seed=int(seed)
        local=copy.deepcopy(c);local['state_config']['seed']=seed
        state_route(run/f'clock_{view}/bundle',out,local,run/f'initial_{seed}/initial.pt',route,c['semantic_dim'],
            domain=run/'semantics_domain',base=run/'semantics_base',pretrained=run/f'pretrained_{seed}' if route in {'K1','K3','K4','K5'} else None)
    elif task.startswith('waves_'):
        view,route=task.removeprefix('waves_').rsplit('_',1)
        waves_task(private/'prepared'/view,run/f'clock_{view}',run/f'state_{view}_{route}_{c["seeds"][0]}',out,
            ROOT/'knowledge/snapshots/ensembl_2026-10-03/nfurzeri_go.tsv')
    else:raise ValueError('Unknown PK1 task')


def summary(run,statuses):
    complete=[k for k,v in statuses.items() if v['execution_status']=='completed']
    missing=[k for k in statuses if k not in complete]
    result={'protocol':'PK1_first_numeric_training','updated':utc(),'overall_status':'running' if any(v['execution_status']=='running' for v in statuses.values()) else 'partial' if missing else 'completed_current_scope',
        'completed':len(complete),'total':len(statuses),'modules':statuses,'missing':missing,
        'full_PK1_research_plan_completed':False,'new_qwen_training_executed':False,
        'knowledge_strategy':'verified existing Alpha adapter; new base/domain vectors actually generated when GPU tasks complete',
        'deferred':{'expanded_knowledge_training':'not_run','GSE3169_GSE221467':'not_run',
            'killifish_regulon':'legacy_fit_units_not_established','functional_depth':'no_matched_expression_function_labels',
            'PK1_service':'not_run; saved-model batch execution only',
            'reserved_queries':'not_run; all frozen roles retained'},'science_status':'unvalidated'}
    write_json(run/'module_status.json',result)
    lines=['# PK1 首轮数值训练','',f"执行状态：{result['overall_status']}；完成 {len(complete)}/{len(statuses)}。",
        '复用已训练 Alpha adapter，不宣称新完成 Qwen 扩展训练；不读取私有留出用于拟合。',
        '该轮是可执行的数值迁移实验，不是完整研究计划已完成。','', '|任务|执行|原因|','|---|---|---|']
    for k,v in statuses.items():lines.append(f"|{k}|{v['execution_status']}|{v.get('reason','')}|")
    comparisons=[]
    for name in complete:
        p=run/name/'result.json'
        if p.exists() and name.startswith('state_'):
            m=read_json(p);comparisons.append({'task':name,**m})
    write_json(run/'comparison.json',comparisons)
    lines+=['','详细对比：comparison.json。每个任务目录保留实际模型、预测和评价；logs/ 保留真实运行日志。',
            '未完成的研究范围：扩大知识语料、新增公共时序/干预接入、本地 regulon 审定、冻结后查询、表达 depth。']
    (run/'REPORT_CN.md').write_text('\n'.join(lines),encoding='utf-8')
    return result


def orchestrate(a):
    c=read_json(a.config)
    if c['reserved_queries_enabled']:raise ValueError('This training runner never unlocks reserved queries')
    c.update(private_root=str(a.private_root.resolve()),alpha_run=str(a.alpha_run.resolve()),model_path=str(a.model_path.resolve()) if a.model_path else None,download=a.download,cpu_only=a.cpu_only)
    if c['knowledge_mode']!='reuse_verified_alpha_adapter':raise ValueError('Unsupported knowledge policy; no silent training substitute')
    if not c['seeds'] or 'K0' not in c['routes'] or 'K3' not in c['routes']:raise ValueError('Declare control routes and seeds')
    run=a.run.resolve();run.mkdir(parents=True,exist_ok=True)
    os.environ['VDC_ROLE_MANIFEST']=str(a.private_root.resolve()/'sample_roles.json')
    versions={}
    for name in ('torch','numpy','scipy','h5py','transformers','peft'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    signature=object_hash({'code':source_fingerprint(),'config':c,'versions':versions,
        'roles':sha256(a.private_root/'sample_roles.json'),'source_lock':sha256(a.private_root/'source_lock.json'),
        'alpha_status':sha256(a.alpha_run/'module_status.json'),
        'annotation':sha256(ROOT/'knowledge/snapshots/ensembl_2026-10-03/nfurzeri_go.tsv')})
    with exclusive_run(run):
        if (run/'run_manifest.json').exists():
            if not a.resume or read_json(run/'run_manifest.json')['signature']!=signature:raise ValueError('Existing run requires --resume and identical code/data/config/environment')
        else:
            write_json(run/'config.json',c);write_json(run/'run_manifest.json',{'signature':signature,'created':utc(),'versions':versions,
                'code_fingerprint':source_fingerprint(),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip() if (ROOT/'.git').exists() else 'offline_source_bundle',
                'cpu_only':a.cpu_only,'private_artifacts':True})
        plan=tasks(c);statuses=read_json(run/'module_status.json')['modules'] if (run/'module_status.json').exists() else {k:{'execution_status':'not_run'} for k in plan}
        for task,(deps,relative) in plan.items():
            old=statuses[task]
            if old['execution_status']=='completed':
                if not verify_inventory(run/relative,old['files']):raise ValueError('Completed artifact changed: '+task)
                print('VERIFIED RESUME '+task,flush=True);continue
            unmet=[d for d in deps if statuses[d]['execution_status']!='completed']
            if unmet or (a.cpu_only and task.startswith('semantics_')):
                statuses[task]={'execution_status':'blocked_dependency','reason':','.join(unmet) if unmet else 'GPU not executed in explicit CPU-only run'}
                summary(run,statuses);continue
            target=run/relative
            if target.exists():
                archive=run/'failed_attempts'/(task+'_'+utc().replace(':','').replace('.',''))
                archive.parent.mkdir(exist_ok=True);target.rename(archive)
            statuses[task]={'execution_status':'running','started':utc(),'artifact':relative};summary(run,statuses)
            print('START '+task,flush=True);(run/'logs').mkdir(exist_ok=True)
            with (run/'logs'/(task+'.log')).open('a',encoding='utf-8') as f:
                proc=subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--run',str(run),'--worker',task],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            statuses[task].update(execution_status='completed' if proc.returncode==0 else 'failed',finished=utc(),
                reason='' if proc.returncode==0 else 'See logs/'+task+'.log',exit_code=proc.returncode)
            if proc.returncode==0:statuses[task]['files']=inventory(target)
            summary(run,statuses);print(statuses[task]['execution_status'].upper()+' '+task,flush=True)
        final=summary(run,statuses)
        print(json.dumps({'status':final['overall_status'],'completed':final['completed'],'total':final['total'],'report':str(run/'REPORT_CN.md')},indent=2),flush=True)
        return 2 if final['missing'] else 0


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/pk1.json')
    p.add_argument('--private-root',type=Path);p.add_argument('--alpha-run',type=Path);p.add_argument('--model-path',type=Path)
    p.add_argument('--download',action='store_true');p.add_argument('--cpu-only',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--worker')
    a=p.parse_args()
    if a.worker:worker(a.run,a.worker)
    else:
        if not a.private_root or not a.alpha_run:p.error('--private-root and --alpha-run are required')
        sys.exit(orchestrate(a))

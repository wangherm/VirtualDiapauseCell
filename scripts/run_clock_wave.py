"""CW1 development queue. Real fits and evaluations; existing Qwen inference only."""
from pathlib import Path
import argparse, concurrent.futures, os, subprocess, sys, time, traceback, shutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_all_modules import utc, exclusive_run, source_fingerprint
from vdc.io import read_json, write_json, sha256, object_hash
from vdc.pk1_assets import verify_files


def compare_representations(run, states):
    import numpy as np
    from vdc.pk2_numeric import metrics
    result={}
    for view in ('bulk','core','coarse'):
        names=[f'numeric_{view}_{rep}' for rep in ('C0','C1','C2')]
        if not all(states[n]['status']=='completed' for n in names):continue
        arrays=[];statuses=[];contracts=[]
        for n in names:
            path=run/'tasks'/n
            with np.load(path/'validation/predictions.npz',allow_pickle=False) as a:arrays.append({k:a[k].copy() for k in a.files})
            statuses.append(read_json(path/'validation/query_status.json')['status'])
            contracts.append(read_json(path/'model/model.json')['target_genes'])
        if any(c!=contracts[0] for c in contracts):raise ValueError('Representation comparison target panels differ')
        rows=read_json(run/'tasks'/names[0]/'validation/rows.json');target=arrays[0]['hidden_target']
        if any(not np.array_equal(a['hidden_target'],target) for a in arrays):raise ValueError('Representation evaluation observations differ')
        common=np.ones_like(target,bool)
        for a,s in zip(arrays,statuses):common&=np.isfinite(a['hidden_prediction'])&np.array([x=='located' for x in s])[:,None]
        result[view]={n:{'model':metrics(a['hidden_prediction'],target,common,rows),
                        'ridge_same_identity':metrics(a['hidden_direct_ridge'],target,common,rows),
                        'identity_mean':metrics(a['hidden_identity_mean'],target,common,rows)} for n,a in zip(names,arrays)}
    write_json(run/'common_support_comparisons.json',result)
    return result


def build_plan(units):
    jobs=[{'id':'prepare','kind':'prepare','deps':[],'resource':'cpu'}]
    identities=[]
    for fold in [None,*range(units)]:
        name='identity_original' if fold is None else f'identity_fold_{fold:02d}'
        identities.append(name);jobs.append({'id':name,'kind':'identity','fold':fold,'deps':['prepare'],'resource':'cpu'})
    # Once all cards are ready, GPU inference can overlap the numerical fits.
    for mode in ('base','domain'):
        jobs.append({'id':'qwen_'+mode,'kind':'qwen','mode':mode,'deps':identities,'resource':'gpu'})
    for view in ('bulk','core','coarse'):
        for rep in ('C0','C1','C2'):
            jobs.append({'id':f'numeric_{view}_{rep}','kind':'numeric','view':view,'representation':rep,'deps':['prepare'],'resource':'cpu'})
    for fold in range(units):
        for rep in ('C0','C2'):
            jobs.append({'id':f'pool_{fold:02d}_{rep}','kind':'numeric','view':'core','fold':fold,'representation':rep,'deps':['prepare'],'resource':'cpu'})
    for mode in ('correct','shuffle'):
        jobs.append({'id':'semantic_'+mode,'kind':'numeric','view':'bulk','representation':'C2','semantic':mode,'deps':['numeric_bulk_C2'],'resource':'cpu'})
    for number in range(3):
        jobs.append({'id':f'family_{number}','kind':'numeric','view':'bulk','representation':'C2','family_index':number,'deps':['prepare'],'resource':'cpu'})
    for mode in ('numeric','base','domain'):
        jobs.append({'id':'chain_'+mode,'kind':'chain','mode':mode,'resource':'cpu',
                     'deps':['numeric_coarse_C2', 'identity_original' if mode=='numeric' else 'qwen_'+mode]})
    for view in ('bulk','core','coarse'):
        jobs.append({'id':'stress_'+view,'kind':'stress','view':view,'deps':['numeric_'+view+'_C2','identity_original'],'resource':'cpu'})
        jobs.append({'id':'gene_waves_'+view,'kind':'gene_waves','view':view,'deps':['numeric_'+view+'_C2'],'resource':'cpu'})
    jobs.append({'id':'cell_composition','kind':'cells','resource':'cpu','deps':['numeric_core_C2']})
    return jobs


def compare_identity_chains(run, states):
    import numpy as np
    from vdc.pk2_numeric import metrics
    names=['numeric_coarse_C2','chain_numeric','chain_base','chain_domain']
    missing=[n for n in names if states.get(n,{}).get('status')!='completed']
    if missing:
        result={'status':'unavailable','missing':missing,'reason':'Common support needs all named chains; no silent subset comparison'}
    else:
        arrays=[];inside=[];rows=None;target=None
        for name in names:
            folder=run/'tasks'/name/'validation';rr=read_json(folder/'rows.json')
            with np.load(folder/'predictions.npz',allow_pickle=False) as a:
                truth=a['hidden_target'].copy();pred=a['hidden_prediction'].copy()
            if rows is not None and (rr!=rows or not np.array_equal(truth,target)):
                raise ValueError('Chain comparison rows or targets differ')
            rows,target=rr,truth;arrays.append(pred)
            inside.append(np.isfinite(pred)&np.array([s=='located' for s in read_json(folder/'query_status.json')['status']])[:,None])
        common=np.logical_and.reduce(inside)
        result={'status':'evaluated','query_profiles':len(rows),
                'common_in_reference_coverage':float(common.mean()),
                'models':{n:{'own_support':metrics(a,target,m,rows),'common_support':metrics(a,target,common,rows)}
                          for n,a,m in zip(names,arrays,inside)},
                'interpretation':'Compare MSE only on common support and also report all-query coverage; lower acceptance is not improvement'}
    write_json(run/'identity_chain_comparisons.json',result)
    return result


def revision_plan(units):
    jobs=build_plan(units)
    for j in jobs:
        j['reuse_parent']=j['kind']!='qwen' and not (j['kind']=='chain' and j['mode']!='numeric')
    for view in ('bulk','core','coarse'):
        for kind in ('residual','support'):
            jobs.append({'id':kind+'_'+view,'kind':kind,'view':view,'resource':'cpu',
                         'deps':['numeric_'+view+'_'+rep for rep in ('C0','C1','C2')]})
    return jobs


def reuse_parent(parent, run, job, private):
    """Verify every reused byte before copying; no old answers become new inference."""
    old=read_json(parent/'queue_status.json')['tasks'][job['id']]
    if old['status']!='completed':raise ValueError('Parent task incomplete: '+job['id'])
    verify_files(parent,old['files'])
    if read_json(parent/'config.json')['role_manifest_sha256']!=sha256(private/'sample_roles.json'):
        raise ValueError('Approved roles changed since original CW1')
    for relative in old['files']:
        if not relative.startswith('tasks/'+job['id']+'/') and not (job['kind']=='prepare' and relative.startswith('data/')):
            raise ValueError('Unexpected file outside reused task scope: '+relative)
        destination=run/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(parent/relative,destination)
    verify_files(run,old['files'])
    return {**old,'execution_kind':'reused_verified','parent_task':job['id'],
            'parent_queue_sha256':sha256(parent/'queue_status.json')}


def report(run,plan,states):
    counts={s:sum(v['status']==s for v in states.values()) for s in sorted({v['status'] for v in states.values()})}
    terminal={'completed','not_applicable','blocked','failed'}
    finished=all(v['status'] in terminal for v in states.values())
    overall='completed_current_scope' if all(v['status']=='completed' for v in states.values()) else 'partial' if finished else 'running'
    result={'status':overall,'counts':counts,'total':len(plan),'tasks':states,'updated':utc(),
            'reused_verified':sum(v.get('execution_kind')=='reused_verified' for v in states.values()),
            'reserved_queries_executed':False,'new_qwen_training':False,'future_prediction':None,'depth':None,
            'optional_student':'disabled','composition_stress':states.get('cell_composition',{}).get('status','not_run'),
            'interpretation':'development only; weak models remain reported; no winner selected from prior reserved results'}
    write_json(run/'queue_status.json',result)
    lines=['# CW1 Clock–Identity–Wave 开发轮','','状态：'+overall,
           '本轮为数值拟合、现有 Qwen 推理和分组评价；未安排新 LoRA。没有查询预留样本。',
           'clock 是 Early/Developing 参考坐标，不是时间、恢复百分比或 depth。身份标签为源注释的粗分组参考。',
           '独立单元按 pool/生物学单位计；不把类型 profile 数算作独立重复。','',
           '|任务|状态|原因|','|---|---|---|']
    for j in plan:
        s=states[j['id']];lines.append(f"|{j['id']}|{s['status']}|{s.get('reason','')}|")
    lines+=['','## 数值结果（validation；各视图目标不可直接横比）','','|任务|模型 MSE，同参考支持|Ridge MSE，同支持|覆盖率|','|---|---|---|---|']
    for j in plan:
        p=run/'tasks'/j['id']/'result.json'
        if states[j['id']]['status']=='completed' and p.exists():
            m=read_json(p).get('validation')
            if m:
                a=m['model_in_reference'];b=m['ridge_same_support']
                lines.append(f"|{j['id']}|{a['macro_unit_mse']}|{b['macro_unit_mse']}|{a['coverage']}|")
    lines+=['','## 范围与解释','','语义零条件复用 numeric_bulk_C2；correct/shuffle 是固定图正则对照，不是新增跨物种同源关系。',
            'I0 为源标签映射一致性，不计作独立身份分类准确率。I2/I3 为数值支持后的 Qwen 证据确认或拒答。',
            '原 split 三视图九次拟合、整 pool 逐单元 C0/C2；三项预定义 GO 家族读出额外重拟合。',
            '身份原 split 一次及逐 pool 折：两种 Qwen 各一次加载，逐卡单进程，不是再次领域训练。',
            '计数压力目标为原始带噪观测，不是 clean truth；组成任务实际从开发 pool 抽取细胞，固定/偏移组成使用相同总细胞数。',
            '已查看预留材料保持原角色，后续如再查询应声明回顾性；本轮 runner 不提供预留查询开关。',
            '热应激 counts 仍待补；不造 depth、未来预测或缺失头输出。结果浏览器只展示本轮已保存开发预测。']
    if any(j['kind']=='residual' for j in plan):
        lines+=['','## CW1 定向修订',f"已校验复用旧任务：{result['reused_verified']}；其余状态见完整队列。",
                '旧数值模型冻结。Qwen 使用 short_slots_v2 重新生成原卡、无数值提示和证据挑战；不进行新 LoRA。',
                '原卡链保留数值支持门槛；无提示/挑战独立评分，不送入正式 chain。技术失败与 biological unknown 分开。',
                'residual_* 保存四模型共同支持、外推和匹配训练行 Ridge 对照；support_* 保存条件/身份支持与训练锚点敏感性。']
        for j in plan:
            if j['kind'] not in {'residual','qwen'} or states[j['id']]['status']!='completed':continue
            data=read_json(run/'tasks'/j['id']/'result.json')
            if j['kind']=='qwen':lines.append(f"{j['id']}: {data.get('status_counts',{})}")
            else:
                lines.append(j['id']+': '+str({k:v['metric']['macro_unit_mse'] for k,v in data['metrics']['in_reference'].items()}))
    (run/'REPORT_CN.md').write_text('\n'.join(lines),encoding='utf-8')
    return result


def orchestrate(a):
    run=a.run.resolve();run.mkdir(parents=True,exist_ok=True)
    private=a.private_root.resolve();source=a.source.resolve()
    protocol=read_json(ROOT/'configs/clock_wave.json');crosswalk=read_json(ROOT/'configs/identity_crosswalk.json')
    parent=a.revision_source.resolve() if a.revision_source else None
    if parent:
        if parent==run or run.is_relative_to(parent):raise ValueError('Revision must be outside original run')
        original_config=read_json(parent/'config.json')
        if Path(original_config['source']).resolve()!=source:raise ValueError('Use original CW1 PK2 source')
        if original_config['protocol']!=protocol or original_config['crosswalk']!=crosswalk:
            raise ValueError('Frozen CW1 protocol differs; do not silently revise old models')
        protocol={**protocol,**read_json(ROOT/'configs/clock_wave_revision.json')}
    # Read metadata only to enumerate the actual authorized units; do not inspect held expression.
    original=read_json(private/'prepared/core/manifest.json')
    units=len({r['biological_unit'] for r in original['rows']})
    c={'protocol':protocol,'crosswalk':crosswalk,'private_root':str(private),'source':str(source),
       'code':source_fingerprint(),'annotation_sha256':sha256(ROOT/'knowledge/snapshots/ensembl_2026-10-03/nfurzeri_go.tsv'),
       'role_manifest_sha256':sha256(private/'sample_roles.json'),'source_config_sha256':sha256(source/'config.json'),
       'qwen_enabled':not a.no_qwen}
    if parent:c['revision_parent']={'path':str(parent),'config_sha256':sha256(parent/'config.json'),'queue_sha256':sha256(parent/'queue_status.json')}
    plan=revision_plan(units) if parent else build_plan(units)
    with exclusive_run(run):
        if (run/'config.json').exists():
            if read_json(run/'config.json')!=c or read_json(run/'plan.json')!=plan:raise ValueError('Resume configuration/code/source changed; use a new run')
        else:
            from vdc.pk2_runtime import hardware
            write_json(run/'config.json',c);write_json(run/'plan.json',plan);write_json(run/'hardware.json',hardware())
        states=read_json(run/'queue_status.json')['tasks'] if (run/'queue_status.json').exists() else {}
        for j in plan:
            name=j['id'];prior=states.get(name,{})
            if prior.get('status')=='completed':verify_files(run,prior['files'])
            elif j.get('reuse_parent'):
                states[name]=reuse_parent(parent,run,j,private);print('REUSED_VERIFIED',name,flush=True)
            else:states[name]={'status':'pending'}
            if a.no_qwen and j['kind']=='qwen':states[name]={'status':'blocked','reason':'explicit_no_qwen_local_CPU_verification; not server execution'}
        def execute(job):
            name=job['id'];logs=run/'logs';logs.mkdir(exist_ok=True)
            with (logs/(name+'.log')).open('a',encoding='utf-8') as log:
                proc=subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--run',str(run),'--worker',name],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if proc.returncode:
                path=run/'tasks'/name/'failure.json';reason=read_json(path)['reason'] if path.exists() else 'See task log'
                return {'status':'not_applicable' if proc.returncode==3 else 'failed','reason':reason,'exit_code':proc.returncode,'finished':utc()}
            files={p.relative_to(run).as_posix():sha256(p) for p in (run/'tasks'/name).rglob('*') if p.is_file() and p.name!='failure.json'}
            if job['kind']=='prepare':files.update({p.relative_to(run).as_posix():sha256(p) for p in (run/'data').rglob('*') if p.is_file()})
            return {'status':'completed','files':files,'finished':utc()}
        active={};resources=set()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            while True:
                for future,j in list(active.items()):
                    if future.done():
                        states[j['id']]=future.result();resources.remove(j['resource']);del active[future]
                        print(states[j['id']]['status'].upper(),j['id'],flush=True)
                for j in plan:
                    if states[j['id']]['status']!='pending':continue
                    if any(states[d]['status'] in {'failed','blocked','not_applicable'} for d in j['deps']):
                        states[j['id']]={'status':'blocked','reason':'Dependency not completed: '+','.join(d for d in j['deps'] if states[d]['status']!='completed')};continue
                    if j['resource'] in resources or not all(states[d]['status']=='completed' for d in j['deps']):continue
                    states[j['id']]={'status':'running','started':utc()};resources.add(j['resource']);active[pool.submit(execute,j)]=j
                    print('START',j['id'],flush=True)
                result=report(run,plan,states)
                if not active:break
                concurrent.futures.wait(list(active),timeout=3,return_when=concurrent.futures.FIRST_COMPLETED)
        compare_representations(run,states)
        compare_identity_chains(run,states)
        snapshot={'protocol':c['protocol']['protocol'],'config_hash':object_hash(c),'files':
                  {p.relative_to(run).as_posix():sha256(p) for p in (run/'tasks').rglob('*')
                   if p.is_file() and (p.name in {'result.json','metrics.json','rows.json','query_status.json','predictions.npz','stage_status.json','preflight.json'} or (p.suffix=='.json' and p.name.startswith('identity_')))
                   and states[p.relative_to(run).parts[1]]['status']=='completed'}}
        for name in ('common_support_comparisons.json','identity_chain_comparisons.json'):
            snapshot['files'][name]=sha256(run/name)
        snapshot['snapshot_id']=object_hash(snapshot);write_json(run/'results_snapshot.json',snapshot)
        print({'status':result['status'],'report':str(run/'REPORT_CN.md')},flush=True)
        return 0 if result['status']=='completed_current_scope' else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--private-root',type=Path);p.add_argument('--source',type=Path);p.add_argument('--worker');p.add_argument('--no-qwen',action='store_true');p.add_argument('--revision-source',type=Path);a=p.parse_args()
    if a.worker:
        try:
            import torch
            from vdc.clock_wave_tasks import worker
            torch.set_num_threads(read_json(a.run/'config.json')['protocol']['cpu_threads'])
            spec=next(j for j in read_json(a.run/'plan.json') if j['id']==a.worker);worker(a.run,spec,ROOT)
        except Exception as exc:
            from vdc.clock_wave import NotIdentifiable
            traceback.print_exc();write_json(a.run/'tasks'/a.worker/'failure.json',{'type':type(exc).__name__,'reason':str(exc)})
            sys.exit(3 if isinstance(exc,NotIdentifiable) else 1)
    else:
        if not a.source or not a.private_root:p.error('--source and --private-root are required')
        sys.exit(orchestrate(a))

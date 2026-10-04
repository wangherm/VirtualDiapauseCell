"""Explicit, limited compatibility migration from the audited first PK2 release.

Never relax normal resume signatures. Completed source artifacts are copied,
verified, and attributed to their original run, not counted as new training.
"""
from pathlib import Path
import copy
import shutil
from .io import read_json, write_json, object_hash
from .pk2_semantics import semantic_mode

AUDITED_SOURCE_CODE = '01e6aec7c8b6646119d9ba82e932cbf9b95475167949322dfe14f91112b9ebf4'
SUCCESS = {'completed', 'evaluated_new', 'reused_verified'}


def repair_invalidations(plan):
    invalid = {'audit': 'Recheck current approved inputs', 'benchmark': 'Measure actual current hardware',
               'register_pk2': 'Publish corrected model identities and independently verify HTTP calls',
               'repair_summary': 'Aggregate corrected results',
               'knowledge_balanced_base': 'New mixed-answerability evidence evaluation',
               'knowledge_balanced_domain': 'Same evaluation with existing selected adapter'}
    for name, task in plan.items():
        job = task['params'].get('job', {})
        if job.get('family') == 'semantic_increment' and semantic_mode(job) in {'domain', 'shuffle'}:
            invalid[name] = 'Original domain/shuffle dispatch silently selected zero semantics'
        if job.get('family') == 'pool_generalization':
            invalid[name] = 'Rerun fixed final budget; original outer pool selected checkpoints'
    return invalid


def import_verified(source, run, plan, config, verify_inventory, inventory):
    source, run = Path(source).resolve(), Path(run).resolve()
    if source == run or source.is_relative_to(run) or run.is_relative_to(source):
        raise ValueError('Repair must use a separate sibling run; original run stays unchanged')
    manifest = read_json(source/'run_manifest.json')
    if manifest.get('code') != AUDITED_SOURCE_CODE or manifest.get('integration_only'):
        raise ValueError('Repair compatibility is restricted to the audited full-budget PK2 release')
    old_config = read_json(source/'config.json')
    if any(config.get(k) != v for k, v in old_config.items()):
        raise ValueError('Repair cannot silently change the source scientific configuration or input paths')
    old_plan = read_json(source/'plan.json')
    statuses = read_json(source/'queue_status.json')['tasks']
    if any(s['status'] in {'pending','ready','running'} for s in statuses.values()):
        raise ValueError('Repair source is not a finished queue')
    invalid = repair_invalidations(plan)
    required_bytes=sum((source/'tasks'/n/f).stat().st_size for n,t in plan.items()
        if n not in invalid and n in statuses and statuses[n]['status'] in SUCCESS
        for f in statuses[n]['files'])
    free=shutil.disk_usage(run).free
    print(f'REPAIR COPY: {required_bytes} bytes of verified artifacts; free={free}',flush=True)
    if free < required_bytes + 2*1024**3:
        raise ValueError('Insufficient space for independent verified copies plus repair outputs; original run remains unchanged')
    imported = {}
    for name, task in plan.items():
        if name in invalid or name not in old_plan: continue
        if task['params'] != old_plan[name]['params'] or task['kind'] != old_plan[name]['kind']:
            raise ValueError('Changed task is not approved for reuse: '+name)
        s = statuses[name]
        if s['status'] not in SUCCESS: continue
        src, dest = source/'tasks'/name, run/'tasks'/name
        if not verify_inventory(src, s['files']):
            raise ValueError('Source artifact inventory changed: '+name)
        if dest.exists():
            # Allows recovery of a copy interrupted before the import manifest was committed.
            for p in dest.rglob('*'):
                if p.is_file() and p.relative_to(dest).as_posix() not in s['files']:
                    raise ValueError('Unexpected existing repair import: '+str(p))
        shutil.copytree(src, dest, dirs_exist_ok=True)
        if not verify_inventory(dest, s['files']):
            raise ValueError('Copied artifact inventory differs: '+name)
        entry = copy.deepcopy(s)
        entry.update(status='reused_verified' if task['kind'] in {'numeric','qwen'} else 'completed',
                     reuse_source_run=str(source), reuse_source_signature=manifest['signature'],
                     reuse_source_inventory_hash=object_hash(s['files']), new_execution=False)
        imported[name] = entry
        print('VERIFIED REUSE', name, flush=True)
    # Absent Early/D7 anchors are a documented inapplicable subset, not a retryable training failure.
    missing = 'input_learning_0.25_42'
    old = statuses.get(missing, {})
    if old.get('status') == 'blocked_data' and 'Missing train Early/D7 anchors' in old.get('reason',''):
        for name in (missing, 'learning_0.25_R0_42', 'learning_0.25_R3_42'):
            imported[name] = {'status': 'inapplicable', 'reason': 'Prespecified 25% subset lacks train Early/D7 anchors; no validation anchors borrowed',
                              'reuse_source_signature': manifest['signature'], 'new_execution': False}
    report = {'source_signature': manifest['signature'], 'source_code': manifest['code'],
              'source_run': str(source), 'imported_tasks': imported, 'invalidated_tasks': invalid,
              'original_artifacts_modified': False, 'reserved_queries': False,
              'outer_cv_scope': 'Selection-free rerun of previously inspected development pools; not a new blind test'}
    write_json(run/'repair_import.json', report)
    return imported


def summary(run, out):
    import numpy as np
    from .pk2_numeric import dynamic_ratio
    run, out = Path(run), Path(out)
    queue=read_json(run/'queue_status.json');plan=read_json(run/'plan.json');rows=[]
    for name, status in queue['tasks'].items():
        if plan[name]['kind']!='numeric' or status['status'] not in SUCCESS: continue
        task=run/'tasks'/name;job=plan[name]['params']['job'];ev=task/'evaluation'
        masks=read_json(ev/'mask_metrics.json')
        means={k:float(np.mean([r[k]['macro_unit_mse'] for r in masks if r[k]['macro_unit_mse'] is not None]))
               for k in ('model','ridge','train_mean')}
        with np.load(ev/'predictions.npz', allow_pickle=False) as a:
            dynamic=dynamic_ratio(a['prediction'], a['observed'], a['mask'])
        rows.append({'job':name,'family':job['family'],'route':job.get('route'),'view':job.get('view'),
                     'semantics':job.get('semantics','zero'),'seed':job.get('seed',job.get('algorithm_seed')),
                     'execution':status['status'],'masked_macro_unit_mse':means, **dynamic,
                     'checkpoint_selection':read_json(task/'model/metrics.json')['checkpoint_selection']})
    knowledge={}
    for name in ('knowledge_balanced_base','knowledge_balanced_domain'):
        p=run/'tasks'/name/'evaluation.json'
        knowledge[name]=read_json(p) if p.exists() else {'status':queue['tasks'].get(name,{}).get('status','not_run')}
    write_json(out/'result.json', {'rows':rows,'knowledge_evaluation':knowledge,'science_status':'unvalidated','reserved_queries':False,
        'scope':'All results are development; masks and seeds do not create independent biological units',
        'known_inapplicable':[n for n,s in queue['tasks'].items() if s['status']=='inapplicable'],
        'selection':'No automatic scientific approval or reserved-sample query; choose and freeze after review'})
    lines=['# PK2 修复后总表','','复用结果与本轮重训分别标记。逐 pool 结果来自已查看过的开发样本，固定末步选择不等于全新盲测。',
           '单观测或无真实变化的动态幅度返回 null。不同视图、输入和评价任务分别比较；保留 Ridge 与类型条件 waves。',
           '预留评价未启动；先审阅本表，再冻结路线与输入规则。','',
           '|任务|执行|模型 MSE|Ridge MSE|幅度比|','|---|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"|{r['job']}|{r['execution']}|{r['masked_macro_unit_mse']['model']:.7g}|{r['masked_macro_unit_mse']['ridge']:.7g}|{r['dynamic_std_ratio_median']}|")
    lines+=['','## 知识评价','','按给定来源定位证据并成对移除证据；包括可答题与未知题。不是开放检索或闭卷知识验证。',
            'Adapter 仍由开发验证集选择，不能把本次重评称为独立知识测试。','',
            '```json',__import__('json').dumps({k:v.get('by_evidence_mode',v) for k,v in knowledge.items()},ensure_ascii=False,indent=2),'```']
    out.mkdir(parents=True,exist_ok=True);(out/'COMPARISON_CN.md').write_text('\n'.join(lines),encoding='utf-8')

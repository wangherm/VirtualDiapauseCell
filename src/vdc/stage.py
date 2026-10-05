"""Fixed stage routes over existing CW artifacts. No new model family or public inference API."""
from pathlib import Path
from collections import Counter
import numpy as np
from .io import read_json,write_json,save_npz,sha256,object_hash
from .clock_wave import ClockWave,identities
from .clock_wave_revision import residual_features
from .clock_wave_data import load_data
from .observation import normalise_expression
from .application import predict_ridge
from .pk2_numeric import metrics,dynamic_ratio
from .pk1_assets import verify_files


def predict_route(run,view,counts,types,route):
    model=ClockWave.load(Path(run)/'tasks'/('numeric_'+view+'_C2')/'model')
    q=model.hidden_predictions(counts,types)
    if route=='clock_identity':prediction=q['hidden_prediction']
    elif route=='clock_identity_residual':
        design,location=residual_features(model,counts,types);finite=np.isfinite(location['clock'])
        prediction=np.full_like(q['hidden_prediction'],np.nan)
        if finite.any():prediction[finite]=predict_ridge(Path(run)/'tasks'/('residual_'+view)/'readout.npz',design[finite])
    else:raise ValueError('Unknown stage route; no fallback')
    return model,q,prediction


def build_view(run,view,out,config):
    x,g,defs,rows=load_data(run/'data'/view);va=np.array([i for i,r in enumerate(rows) if r['split']=='validation'])
    rows=[rows[i] for i in va];x=x[va];types=identities(rows);route=config['stage_routes'][view]
    model,q,pred=predict_route(run,view,x,types,route)
    if model.meta['genes']!=g or model.meta['definitions']!=defs:raise ValueError('Stage input/model contract mismatch')
    _,q2,pred2=predict_route(run,view,x,types,route)
    np.testing.assert_allclose(pred,pred2,atol=0,rtol=0,equal_nan=True)
    target=normalise_expression(x,np.ones_like(x,bool),'counts')[:,model.meta['target_indices']]
    # Assert real default invocation matches the already saved declared candidate, never switch silently.
    with np.load(run/'tasks'/('residual_'+view)/'validation/predictions.npz',allow_pickle=False) as a:
        np.testing.assert_allclose(pred,a[route],atol=1e-10,rtol=1e-10,equal_nan=True)
        matched=a['matched_fit_direct'].copy()
    models={'stage_main':pred,'direct_ridge_matched_fit':matched,'direct_ridge_original':q['hidden_direct_ridge'],
            'identity_mean':q['hidden_identity_mean'],'clock_identity':q['hidden_prediction']}
    common=np.logical_and.reduce([np.isfinite(a) for a in models.values()]);inside=np.array([s=='located' for s in q['status']])[:,None]
    groups={'all_conditions':list(range(len(rows)))}
    for condition in sorted({r['condition'] for r in rows}):groups['condition:'+condition]=[i for i,r in enumerate(rows) if r['condition']==condition]
    table=[]
    for group,ii in groups.items():
        rr=[rows[i] for i in ii]
        for scope,mask in [('in_reference',common&inside),('extrapolation_only',common&~inside),('all_finite',common)]:
            for name,a in models.items():
                m=metrics(a[ii],target[ii],mask[ii],rr)
                table.append({'view':view,'group':group,'scope':scope,'model':name,'main_route':route,
                              'query_profiles':len(ii),'query_units':len({r['biological_unit'] for r in rr}),
                              'supported_units':len({rows[i]['biological_unit'] for i in ii if mask[i].any()}),
                              'metric':m,'dynamic':dynamic_ratio(a[ii],target[ii],mask[ii])})
    save_npz(out/'validation/predictions.npz',**models,hidden_target=target,clock=q['clock'],
             observed=q['observed'],reference_expected=q['reference_expected'],residual=q['residual'],residual_mask=q['residual_mask'])
    write_json(out/'validation/rows.json',rows);write_json(out/'validation/query_status.json',{'status':q['status'],'identity':types})
    stages=[{'observation_id':r['observation_id'],'identity':'inherited_source_annotation_not_new_classification',
             'clock':s,'waves':'within_reference' if s=='located' else 'linear_extrapolation_only' if s=='outside_reference_range' else 'unsupported',
             'readout':route,'Qwen_required':False} for r,s in zip(rows,q['status'])]
    write_json(out/'stage_status.json',stages)
    write_json(out/'result.json',{'status':'evaluated','view':view,'main_route':route,'table':table,
               'status_counts':dict(Counter(q['status'])),'actual_default_invocation':'passed','reload':'passed',
               'role':'post-development selected experimental route; not independent validation',
               'residual_sensitivity':'alpha=100 candidate remains auxiliary; never auto-select by its score',
               'clock_model_sha256':sha256(run/'tasks'/('numeric_'+view+'_C2')/'model/model.json'),
               'input_contract':'exact integer counts in frozen gene order, inherited declared identity; no new-input service exposed',
               'target_isolation_scope':model.meta['target_isolation_scope']})


def finalize(run,config,states):
    """Produce a complete or explicitly partial immutable stage manifest after the bounded queue ends."""
    run=Path(run);tables=[];views={};defaults=[]
    for view in config['protocol']['stage_routes']:
        name='stage_'+view
        if states.get(name,{}).get('status')=='completed':
            r=read_json(run/'tasks'/name/'result.json');tables+=r['table'];views[view]={k:r[k] for k in ('main_route','status_counts','actual_default_invocation','reload','clock_model_sha256')};defaults.append(name)
        else:views[view]={'status':'unavailable','task_state':states.get(name)}
    knowledge={}
    for mode in ('base','domain'):
        name='qwen_'+mode
        if states.get(name,{}).get('status')=='completed':
            r=read_json(run/'tasks'/name/'result.json');p=read_json(run/'tasks'/name/'provenance.json')
            knowledge[mode]={'status':'evaluation_executed','contract':r['contract'],'generation_count':r['generation_count'],
                             'counts':r['status_counts'],'folds':r['folds'],'actual_model_provenance':p,
                             'model_success_claim':False,'default_identity_source':False}
        else:knowledge[mode]={'status':'not_completed','task_state':states.get(name)}
    sensitivity={}
    for view in config['protocol']['stage_routes']:
        name='shrink_'+view
        if states.get(name,{}).get('status')=='completed':
            old=read_json(run/'tasks'/('residual_'+view)/'result.json')
            new=read_json(run/'tasks'/name/'result.json')
            sensitivity[view]={'alpha':config['protocol']['residual_sensitivity_alpha'],
                'default_changed':False,'scope':'fixed diagnostic; no automatic selection',
                'macro_unit_mse':{scope:{'original':old['metrics'][scope]['clock_identity_residual']['metric']['macro_unit_mse'],
                                       'stronger_shrinkage':new['metrics'][scope]['clock_identity_residual']['metric']['macro_unit_mse']}
                                  for scope in ('in_reference','extrapolation_only','all_finite')}}
        else:sensitivity[view]={'status':'not_completed','task_state':states.get(name)}
    complete=bool(states) and all(s['status']=='completed' for s in states.values())
    summary={'version':'CW-stage-1','status':'frozen_experimental' if complete else 'partial_not_closed',
             'default_tasks':defaults,'views':views,'knowledge':knowledge,'result_table':tables,'residual_sensitivity':sensitivity,
             'capability':'saved development predictions browser; internal default numerical route replay tested',
             'new_input_web_inference':False,'new_independent_validation':False,
             'reserved_queries_executed':False,'new_qwen_training':False,'depth':None,'future_prediction':None,
             'historical_parent':config.get('revision_parent'),'policy':config['protocol'],
             'unfinished':{k:v for k,v in states.items() if v['status']!='completed'},
             'stop_rule':'no further automatic tuning or model expansion after fixed queue'}
    write_json(run/'STAGE_SUMMARY.json',summary)
    lines=['# CW-stage-1 阶段实验模型卡','', '状态：'+summary['status'],
           '主线：rank＋本地相对表达幅度 → 已知身份/背景 → 参考 clock → gene/TF RNA/programme waves → 观测偏离。',
           '本版依据已查看的开发结果收拢；不是新的独立测试，不宣称因果恢复或未来预测。','',
           '|视图|固定默认路线|强基线|','|---|---|---|',
           '|bulk|clock＋身份＋可见 residual（原 alpha=1）|匹配训练行 direct Ridge，并列保留|',
           '|core 整 pool|clock＋身份|direct Ridge；residual 仅候选|',
           '|粗身份 profile|身份条件 clock/waves；关闭 residual 修正|匹配 direct Ridge|','',
           'alpha=100 为唯一追加收缩诊断，不影响本轮默认选择。训练 residual 是训练内参考，不是 cross-fit。',
           '已知身份优先继承；Qwen 是辅助证据评价，格式/证据/unknown/数值冲突分别计数，不证明 lineage。',
           'Early 范围外、blood 缺参考和单 pool 锚点重复不足保持原状态；不扩范围、不 clip。Late 维护不等同 Exit。',
           'TF 曲线是 TF RNA，不是经过扰动验证的 TF 活性。跨物种 mapping、depth、复杂未来响应暂不提供。',
           '浏览器只展示冻结的保存结果，没有上传新样本推理功能。权重与来源按 stage_snapshot.json 绑定。','',
           '## 主结果（生物学单位宏平均；范围外单列于 STAGE_SUMMARY.json）','',
           '|视图|模型|范围内 MSE|覆盖率|支持单位/请求单位|','|---|---|---|---|---|']
    for r in tables:
        if r['group']=='all_conditions' and r['scope']=='in_reference' and r['model'] in {'stage_main','direct_ridge_matched_fit'}:
            lines.append(f"|{r['view']}|{r['model']}|{r['metric']['macro_unit_mse']}|{r['metric']['coverage']}|{r['supported_units']}/{r['query_units']}|")
    lines+=['','## 唯一收缩诊断（所有有限预测，单位宏平均 MSE）','',
            '|视图|原残差 alpha=1|收缩 alpha=100|默认是否改变|','|---|---|---|---|']
    for view,r in sensitivity.items():
        m=r.get('macro_unit_mse',{}).get('all_finite',{})
        lines.append(f"|{view}|{m.get('original','未完成')}|{m.get('stronger_shrinkage','未完成')}|否|")
    lines+=['','## 身份分支','固定字段协议全量重新生成；slot 合法不等于生物学特异性，候选确认不等于独立注释。',
            '','|模型|状态|各类输出计数|','|---|---|---|']
    for mode,k in knowledge.items():lines.append(f"|{mode}|{k['status']}|{k.get('counts',{})}|")
    lines+=['','精确 base/adapter 版本、三类评价及逐折结果见 STAGE_SUMMARY.json 和 qwen_* 工件。',
            '','## 未完成',json_text(summary['unfinished'])]
    (run/'MODEL_CARD_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    files={}
    for name,s in states.items():
        if s['status']!='completed':continue
        for rel,h in s.get('files',{}).items():
            if rel.startswith('tasks/'):files[rel]=h
    for name in ('STAGE_SUMMARY.json','MODEL_CARD_CN.md','runtime_identity.json'):
        if (run/name).exists():files[name]=sha256(run/name)
    snap={'version':'CW-stage-1','status':summary['status'],'config_hash':object_hash(config),
          'code_hash':config['code'],'files':files,'default_routes':config['protocol']['stage_routes'],
          'external_llm_weights':'not copied; actual base/adapter fingerprints recorded in Qwen provenance',
          'source_counts':'not included in report package; cached authorized development inputs retained on server'}
    snap['snapshot_id']=object_hash(snap);write_json(run/'stage_snapshot.json',snap)
    verify_files(run,files)
    return summary


def json_text(value):
    import json
    return '```json\n'+json.dumps(value,ensure_ascii=False,indent=2)+'\n```'

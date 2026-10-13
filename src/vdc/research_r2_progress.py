"""Content-based tool reuse and a recoverable, science-agnostic no-progress stop."""
from collections import Counter
from .io import object_hash,sha256
from .research_r2_artifacts import checked_artifact

READ_TOOLS={'inspect_preparation','read_metadata','inspect','fetch','read_public','read_task','read_action','read_vdc','read_history','read_preparation_code'}


def request_fingerprint(engine,a):
    request={k:v for k,v in a.items() if k not in ('id','device_reason')}
    kind=a['type']; bindings={}
    if kind=='inspect_preparation':
        if a.get('path'):
            _,item,_=checked_artifact(engine.run,a['preparation'],a['path'],engine.inv)
            bindings['file']=item['sha256']
        else:
            from .research_r2_tools import tree_hash
            bindings=tree_hash(engine.run/'actions'/a['preparation']/'output')
    elif kind in ('inspect','read_metadata'):
        for key in a.get('asset_ids',[a.get('asset')]):
            asset=engine.inv['assets'][key]
            # Hash the content, not a new action name or a changed mtime.
            bindings[key]=sha256(asset['path'])
    elif kind=='read_public':bindings['document']=sha256(engine.run/'literature'/(a['document']+'.txt'))
    elif kind=='read_preparation_code':
        from .research_r2_tools import inside, identifier
        path=inside(engine.run/'actions'/identifier(a['preparation'])/'code',a['path'])
        bindings['code']=sha256(path) if path.is_file() else 'absent'
    elif kind=='read_task':bindings=engine.state['tasks'].get(a['task'],{})
    elif kind=='read_action':bindings=engine.state['actions'].get(a['action'],{})
    elif kind=='read_vdc':
        from .research_r2_tools import inside
        bindings['code']=sha256(inside(engine.repo,a['path']))
    return object_hash({'request':request,'content':bindings,'tool_protocol':'engine_v2'})


def progress_signature(engine):
    artifacts=[]
    for a in engine.state['actions'].values():
        if a['type']=='prepare' and a['status']=='completed':
            index=a.get('result',{}).get('artifacts',{})
            if index.get('status') not in (None,'no_artifacts'):
                artifacts.append(index.get('content_signature'))
    return object_hash({'public_assets':sorted(v.get('sha256','') for v in engine.inv['assets'].values() if v.get('download_provenance')),
        'artifacts':sorted(set(artifacts)),
        'tasks':sorted(v['signature'] for v in engine.state['tasks'].values()),
        'experiments':sorted(object_hash({'task':v['task_signature'],'code':v['code_files'],'metrics':v['metrics']}) for v in engine.state['experiments'].values()),
        'exports':sorted(engine.state['exports'])})


def observe_progress(engine,before):
    progress=engine.state.setdefault('engine_progress',{'stagnant_decisions':0,'diagnostic_mode':False})
    after=progress_signature(engine)
    progress['stagnant_decisions']=progress['stagnant_decisions']+1 if before==after else 0
    progress['signature']=after
    warning=engine.config.get('no_progress_warning_decisions',8)
    stop=engine.config.get('no_progress_stop_decisions',24)
    progress['diagnostic_mode']=progress['stagnant_decisions']>=warning
    if progress['diagnostic_mode']:
        counts=Counter(engine.state.get('request_observations',[])[-100:])
        progress['repeated_requests']=[{'fingerprint':k,'times':v} for k,v in counts.most_common(5)]
        progress['recent_obstacles']=[{'action':k,'tool':v['type'],'diagnostic':v.get('diagnostic'),
            'candidate_error_classes':v.get('candidate_error_classes',[])} for k,v in list(engine.state['actions'].items())[-40:] if v['status']=='failed'][-5:]
        progress['available_preparations']=[{'action':k,'state':v.get('result',{}).get('artifacts',{}).get('status')}
            for k,v in engine.state['actions'].items() if v['type']=='prepare' and v['status']=='completed'][-8:]
        progress['next_step']='No new usable artifacts/tasks/evaluated candidates/exports. Diagnose the concrete missing input or unreadable field, use artifact references or another executable branch. Do not reread an unchanged response.'
    if progress['stagnant_decisions']>=stop and engine.state['status']!='completed_declared_scope':
        engine.state['status']='stalled_no_progress'
        progress['resume']='Fix the recorded obstacle, then explicitly resume with --resume-after-stall. No automatic paid loop.'
    return progress

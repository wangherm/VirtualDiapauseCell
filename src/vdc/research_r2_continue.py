"""Fork a stopped, preparation-only research run without altering its record."""
import shutil
from pathlib import Path
from .io import read_json, write_json, sha256
from .research_r2 import Engine


def continue_preparation_run(repo, private, parent, target, config):
    parent, target = Path(parent).resolve(), Path(target).resolve()
    private = Path(private).resolve()
    if parent == target or parent in target.parents or not parent.is_relative_to(private) or not target.is_relative_to(private):
        raise ValueError('Choose distinct run directories inside the private root')
    if target.exists():
        raise ValueError('Continuation target must not exist')
    state = read_json(parent/'state.json')
    if state['status'] not in ('budget_exhausted_partial','paused_by_operator'):
        raise ValueError('Only a stopped, known-outcome run can be continued by this migration')
    if state['tasks'] or state['experiments'] or state['exports']:
        raise ValueError('This migration only preserves preparation; trained tasks need separate compatibility review')
    if any(v['status']=='running' for v in state['actions'].values()):
        raise ValueError('An action still has an uncertain outcome')
    old_manifest=read_json(parent/'manifest.json')
    sources=['api','decisions','actions','literature','wheels','inventory.json','state.json']
    fingerprints={}
    for name in sources:
        root=parent/name
        if not root.exists(): continue
        for p in ([root] if root.is_file() else root.rglob('*')):
            if p.is_symlink(): raise ValueError('Do not migrate linked artifacts')
            if p.is_file(): fingerprints[p.relative_to(parent).as_posix()]=sha256(p)
    target.mkdir(parents=True)
    write_json(target/'migration_status.json',{'status':'copying'})
    for name in sources:
        src=parent/name
        if src.is_dir(): shutil.copytree(src,target/name)
        elif src.is_file(): shutil.copy2(src,target/name)
    for rel, digest in fingerprints.items():
        if sha256(target/rel)!=digest or sha256(parent/rel)!=digest:
            raise ValueError('Artifact changed during migration; do not launch')
    write_json(target/'parent_manifest.json',old_manifest)
    write_json(target/'continuation.json',{
        'parent_run':str(parent),'parent_manifest_sha256':sha256(parent/'manifest.json'),
        'inherited_actions':len(state['actions']),'inherited_decisions':state['next_decision']-1,
        'inherited_api_requests':len(list((parent/'api').glob('*.json'))),
        'artifact_sha256':fingerprints,'scope':'preparation_only; not new training',
        'parent_modified':False})
    state['status']='ready'
    state['memories'].append('Continuation: previous decisions and preparations remain available through read_action/inspect_preparation. No tasks or models have yet been trained. Do not restart the inventory.')
    write_json(target/'state.json',state)
    engine=Engine(repo,private,target,config)
    if engine.config['runtime']['image_id'] != old_manifest['config']['runtime']['image_id']:
        raise ValueError('Runtime changed; inspect compatibility before continuing')
    engine.save(); engine.report()
    write_json(target/'migration_status.json',{'status':'verified','parent_unchanged':True})
    return engine


def continue_engine_phase(repo, private, parent, target, config):
    """New audited-contract phase; retain preparations and old science as history."""
    from .research_r2_tools import tree_hash
    from .research_r2_artifacts import artifact_index
    parent,target,private=map(lambda p:Path(p).resolve(),(parent,target,private))
    if parent==target or parent in target.parents or not parent.is_relative_to(private) or not target.is_relative_to(private):
        raise ValueError('Choose distinct runs inside the private root')
    if target.exists():raise ValueError('Continuation target must not exist')
    state=read_json(parent/'state.json')
    if state['status'] not in ('budget_exhausted_partial','paused_by_operator','stalled_no_progress','format_failures_partial'):
        raise ValueError('Parent must have a stopped known outcome')
    if any(a['status']=='running' for a in state['actions'].values()):
        raise ValueError('Resolve uncertain running actions before migration')
    target.mkdir(parents=True);write_json(target/'migration_status.json',{'status':'copying'})
    files={}
    for name in ('actions','literature','wheels','inventory.json'):
        src=parent/name
        if src.is_dir():
            hashes=tree_hash(src);shutil.copytree(src,target/name)
            if tree_hash(target/name)!=hashes or tree_hash(src)!=hashes:raise ValueError('Copy/source changed during migration')
            files.update({name+'/'+k:v for k,v in hashes.items()})
        elif src.is_file():
            digest=sha256(src);shutil.copy2(src,target/name)
            if sha256(src)!=digest or sha256(target/name)!=digest:raise ValueError('Source changed during migration')
            files[name]=digest
    history=target/'history';history.mkdir()
    for name in ('state.json','manifest.json','status.json','api','decisions','tasks','exports'):
        src=parent/name
        if src.is_dir():shutil.copytree(src,history/name)
        elif src.is_file():shutil.copy2(src,history/name)
    old_manifest=read_json(parent/'manifest.json')
    inv=read_json(target/'inventory.json')
    for source in inv['sources'].values():
        source['historical_disposition']=source.get('disposition')
        source['historical_experiments']=source.pop('experiments',[])
        source['disposition']='not_processed'
    write_json(target/'inventory.json',inv)
    inherited={k:dict(v) for k,v in state['actions'].items() if v['type'] in ('prepare','fetch','search','dependencies') and v['status']=='completed'}
    fresh={'status':'ready','actions':inherited,'tasks':{},'experiments':{},'exports':{},'next_decision':1,
        'memories':['New engine phase: reuse existing preparations via inspect_preparation and artifact_inputs. '
                    'Read source unit metadata; old tasks are historical only and need fresh contract registration. '
                    'read_history exposes previous candidate/metric evidence. Do not restart source discovery.']}
    write_json(target/'state.json',fresh)
    engine=Engine(repo,private,target,config)
    if engine.config['runtime']['image_id']!=old_manifest['config']['runtime']['image_id']:
        raise ValueError('Runtime differs from parent; perform an explicit compatibility review before migration')
    for key,entry in engine.state['actions'].items():
        if entry['type']=='prepare':entry['result']['artifacts']=artifact_index(target,key,engine.inv)
    records=[read_json(p) for p in (history/'api').glob('*.json')]
    lineage={'parent_run':str(parent),'parent_manifest_sha256':sha256(parent/'manifest.json'),
        'inherited_actions':len(inherited),'inherited_decisions':state['next_decision']-1,
        'inherited_api_requests':len(records),'historical_reported_tokens':sum((r.get('usage') or {}).get('total_tokens',0) for r in records),
        'historical_unknown_usage_requests':sum(not r.get('usage') for r in records),
        'historical_experiments':len(state['experiments']),'counter_mode':'new_phase',
        'historical_limitations':('killi_state_v1 used material types as units and all-gene CPM before target selection; independent-group/strict-target-isolation interpretations are not supported. Original scores preserved.'
            if 'killi_state_v1' in state['tasks'] else 'Historical contracts retain their original limitations; not revalidated by migration.'),
        'new_phase_request_limit':config['max_api_calls'],'new_phase_token_limit':config.get('max_token_budget'),
        'scope':'preparation reuse; historical task scores unchanged; new audited tasks not yet trained',
        'parent_modified':False,'artifact_sha256':files,'history_sha256':tree_hash(history)}
    write_json(target/'continuation.json',lineage)
    engine.save();engine.report()
    write_json(target/'migration_status.json',{'status':'verified','parent_unchanged':True,'new_training_executed':False})
    return engine

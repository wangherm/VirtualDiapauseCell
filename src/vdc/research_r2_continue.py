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

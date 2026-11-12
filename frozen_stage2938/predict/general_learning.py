"""Reusable numerical training driver for the EXISTING isolated VDC executor.

The primary supplies model.py: initialize(data, config, parent), step(state,data,
config), save(state,directory), load(directory), predict(state,data,config).
save/load must preserve parameters, optimiser and RNG state needed for resumption.
This driver does not choose a scientific model or decide adoption.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import numpy as np

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,sort_keys=True,indent=2),encoding='utf-8');os.replace(tmp,path)
def load_plugin(path):
    spec=importlib.util.spec_from_file_location('vdc_scientific_backend',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for name in ('initialize','step','save','load','predict'):
        if not callable(getattr(module,name,None)):raise ValueError('Missing backend function: '+name)
    return module
def arrays(path):
    with np.load(path,allow_pickle=False) as values:return {k:values[k] for k in values.files}
def verify(folder):
    folder=Path(folder);manifest=read(folder/'checkpoint.json')
    actual={p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*') if p.is_file() and p.name!='checkpoint.json'}
    if actual!=manifest['files']:raise ValueError('Checkpoint bytes changed')
    return manifest
def latest(folder):
    points=sorted(Path(folder).glob('checkpoint_[0-9][0-9][0-9][0-9][0-9][0-9]'))
    return points[-1] if points else None
def train(recipe_path,data_path,plugin_path,output,parent=None,stop_after=None):
    out=Path(output);recipe=read(recipe_path);plugin_path=Path(plugin_path)
    if set(recipe)!={'schema_version','steps','config','data_contract'} or recipe['schema_version']!=1:
        raise ValueError('Recipe needs schema_version,steps,config,data_contract')
    if type(recipe['steps']) is not int or not 1<=recipe['steps']<=100000:raise ValueError('Invalid steps')
    if not isinstance(recipe['config'],dict) or not isinstance(recipe['data_contract'],dict) or not recipe['data_contract']:
        raise ValueError('Explicit config and data contract required')
    parent=Path(parent) if parent else None
    pm=verify(parent) if parent else None
    if pm and pm['binding']['plugin_sha256']!=sha(plugin_path):raise ValueError('Parent backend differs; use a documented migration')
    binding={'recipe_sha256':sha(recipe_path),'data_sha256':sha(data_path),'plugin_sha256':sha(plugin_path),
             'driver_sha256':sha(__file__),'parent_checkpoint_sha256':sha(parent/'checkpoint.json') if parent else None}
    out.mkdir(parents=True,exist_ok=True)
    if (out/'session.json').exists():
        if read(out/'session.json')['binding']!=binding:raise ValueError('Session content changed; use a new candidate directory')
    elif list(out.iterdir()):raise ValueError('Unrecognised nonempty output directory')
    else:write(out/'session.json',{'binding':binding,'recipe':recipe,'adoption':'experimental_not_promoted'})
    lock=out/'.training.lock'
    # Kernel lock releases on process exit; a leftover lock file is not a live job.
    import fcntl
    with lock.open('a+b') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        checkpoint=latest(out);backend=load_plugin(plugin_path);data=arrays(data_path)
        if checkpoint:
            info=verify(checkpoint)
            if info['binding']!=binding:raise ValueError('Checkpoint session mismatch')
            state=backend.load(checkpoint/'model');start=info['completed_steps']
        else:
            parent_state=backend.load(parent/'model') if parent else None
            state=backend.initialize(data,recipe['config'],parent_state);start=0
        count=0
        for number in range(start+1,recipe['steps']+1):
            state=backend.step(state,data,recipe['config'])
            temp=out/f'.checkpoint_{number:06d}.pending'
            if temp.exists():
                # Preserve incomplete files; do not silently call them a valid checkpoint.
                archived=out/f'interrupted_{number:06d}_{len(list(out.glob("interrupted_*"))):06d}'
                temp.rename(archived)
            temp.mkdir();(temp/'model').mkdir()
            backend.save(state,temp/'model')
            files=list((temp/'model').rglob('*'))
            if not any(p.is_file() for p in files) or any(p.is_symlink() for p in files):raise ValueError('Backend must save real non-symlink state')
            shutil.copyfile(plugin_path,temp/'model.py')
            write(temp/'recipe.json',recipe)
            manifest={'binding':binding,'completed_steps':number,
                'files':{p.relative_to(temp).as_posix():sha(p) for p in temp.rglob('*') if p.is_file()},
                'parameters_semantics':'backend_defined; not proof of scientific improvement'}
            write(temp/'checkpoint.json',manifest)
            dest=out/f'checkpoint_{number:06d}';temp.rename(dest)
            write(out/'progress.json',{'completed_steps':number,'target_steps':recipe['steps'],'checkpoint':dest.name,'status':'training'})
            count+=1
            if stop_after and count>=stop_after and number<recipe['steps']:
                result={'status':'paused_at_checkpoint','completed_steps':number,'checkpoint':dest.name}
                write(out/'progress.json',result);return result
        point=latest(out)
        result={'status':'completed','completed_steps':recipe['steps'],'checkpoint':point.name,
                'checkpoint_sha256':sha(point/'checkpoint.json'),'adoption':'experimental_not_promoted'}
        write(out/'progress.json',result)
        return result

def predict(checkpoint,data_path,output):
    checkpoint=Path(checkpoint);verify(checkpoint)
    backend=load_plugin(checkpoint/'model.py');recipe=read(checkpoint/'recipe.json')
    state=backend.load(checkpoint/'model')
    prediction=np.asarray(backend.predict(state,arrays(data_path),recipe['config']))
    if prediction.ndim!=2 or not np.isfinite(prediction).all():raise ValueError('Expected finite2D predictions')
    output=Path(output)
    if output.exists():raise ValueError('Do not overwrite predictions')
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,prediction=prediction)
    result={'checkpoint_sha256':sha(checkpoint/'checkpoint.json'),'input_sha256':sha(data_path),
            'output_sha256':sha(output),'shape':list(prediction.shape),'training_during_predict':False}
    write(output.with_suffix('.receipt.json'),result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train')
    for name in ('recipe','data','plugin','output'):t.add_argument('--'+name,required=True)
    t.add_argument('--parent');t.add_argument('--stop-after',type=int)
    v=sub.add_parser('predict')
    for name in ('checkpoint','data','output'):v.add_argument('--'+name,required=True)
    args=vars(p.parse_args());command=args.pop('command')
    if command=='train':result=train(args['recipe'],args['data'],args['plugin'],args['output'],args['parent'],args['stop_after'])
    else:result=predict(args['checkpoint'],args['data'],args['output'])
    print(json.dumps(result))
if __name__=='__main__':main()

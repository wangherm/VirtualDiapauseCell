"""Record actual hardware and benchmark canonical numerical forward/backward work."""
from pathlib import Path
import os,platform,shutil,time,subprocess
import numpy as np
import torch
from .io import write_json

def resource_sample():
    memory={}
    if Path('/proc/meminfo').exists():
        memory={k:int(v.strip().split()[0])*1024 for k,v in (l.split(':',1) for l in Path('/proc/meminfo').read_text().splitlines()) if k in {'MemTotal','MemAvailable'}}
    gpu={'status':'nvidia-smi unavailable'}
    if shutil.which('nvidia-smi'):
        try:
            result=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
            gpu={'status':'measured' if result.returncode==0 else 'measurement_failed','csv':result.stdout.strip(),'units':['index','percent','MiB','MiB']}
        except (OSError,subprocess.TimeoutExpired) as exc:gpu={'status':'measurement_failed','reason':str(exc)}
    return {'memory_bytes':memory,'load_average':list(os.getloadavg()) if hasattr(os,'getloadavg') else None,'gpu_measurement':gpu}

def hardware():
    memory={}
    if Path('/proc/meminfo').exists():
        memory={k:int(v.strip().split()[0])*1024 for k,v in (l.split(':',1) for l in Path('/proc/meminfo').read_text().splitlines()) if k in {'MemTotal','MemAvailable'}}
    disk=shutil.disk_usage(Path.cwd())
    return {'platform':platform.platform(),'python':platform.python_version(),'cpu_count':os.cpu_count(),
        'memory_bytes':memory,'disk_free_bytes':disk.free,'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
        'cuda_available':torch.cuda.is_available(),'gpus':[{'index':i,'name':torch.cuda.get_device_name(i),
            'total_memory_bytes':torch.cuda.get_device_properties(i).total_memory,'capability':list(torch.cuda.get_device_capability(i))} for i in range(torch.cuda.device_count())]}

def benchmark(bundle,out,c):
    from .contracts import ObservationBundle
    from .state import ProgrammeStateModel,StateConfig
    b=ObservationBundle.load(bundle);idx=b.indices('train');results=[]
    for device,threads in [('cpu',1),('cpu',c['cpu_threads'])]+([('cuda',c['cpu_threads'])] if torch.cuda.is_available() else []):
        torch.set_num_threads(threads);torch.manual_seed(1941)
        model=ProgrammeStateModel(len(b.feature_ids),StateConfig(**c['state_config']),torch.zeros(len(b.feature_ids),c['semantic_dim'])).to(device)
        x=torch.tensor(b.values[idx],device=device);m=torch.tensor(b.mask[idx],device=device);cov=torch.tensor(b.coverage[idx],device=device)
        opt=torch.optim.AdamW(model.parameters());start=None
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        for step in range(35):
            if step==5:
                if device=='cuda':torch.cuda.synchronize()
                start=time.monotonic()
            opt.zero_grad();pred=model(x,m,cov)['programme'];loss=((pred-x)**2).mean();loss.backward();opt.step()
        if device=='cuda':torch.cuda.synchronize()
        elapsed=time.monotonic()-start;results.append({'device':device,'threads':threads,'steps_per_second':30/elapsed,
            'peak_cuda_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None})
        del model,opt,x,m,cov
        if device=='cuda':torch.cuda.empty_cache()
    best=min(results,key=lambda r:1/r['steps_per_second']);torch.set_num_threads(c['cpu_threads'])
    # CPU slots can overlap data work; GPU runs use one shared exclusive slot with Qwen.
    write_json(Path(out)/'profile.json',{'numeric_device':best['device'],'results':results,
        'input_shape':list(b.values[idx].shape),'benchmark_only_not_research_training':True,
        'policy':'fastest measured single-worker configuration; GPU Qwen and numeric do not overlap',
        'concurrent_gpu_jobs':1,'numeric_threads_requested':best['threads'],
        'limitation':'Representative programme model benchmark; not an exhaustive resource search'})

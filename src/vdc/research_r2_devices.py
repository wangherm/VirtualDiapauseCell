"""Measured accelerator capability and explicit per-action device selection."""
from pathlib import Path
import os
import subprocess
from .io import read_json


def gpu_inventory():
    import torch
    result={'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
            'cuda_available':torch.cuda.is_available(),'devices':[]}
    if result['cuda_available']:
        for i in range(torch.cuda.device_count()):
            p=torch.cuda.get_device_properties(i)
            result['devices'].append({'index':i,'name':p.name,'total_memory_bytes':p.total_memory,
                'compute_capability':[p.major,p.minor],'uuid':str(getattr(p,'uuid','unavailable'))})
    return result


def gpu_receipt(runtime):
    from .research_r2_linux import runtime_info
    expected=runtime_info(runtime)['runtime_id']
    p=Path(runtime['runtime_root'])/'gpu_probe_passed.json'
    if not p.is_file():raise ValueError('CUDA has not passed isolated forward/backward/reload; run setup_research_r2.sh')
    r=read_json(p)
    if (r.get('runtime_id')!=expected or r.get('status')!='passed' or not r.get('hardware',{}).get('devices')
            or r.get('forward_backward',{}).get('status')!='passed' or r.get('reload',{}).get('status')!='passed'):
        raise ValueError('CUDA probe is missing, stale or belongs to another runtime')
    # Actual host identity is checked once when starting a controller; free memory
    # is deliberately excluded from the frozen configuration.
    if gpu_inventory()!=r['hardware']:
        raise ValueError('GPU/driver runtime identity changed; rerun isolated GPU qualification')
    return r


def configure_devices(runtime,policy=None):
    r=dict(runtime)
    policy=policy or r.get('device_policy','cpu')
    if policy not in ('ai','cpu'):raise ValueError('Device policy must be ai or cpu')
    r.update(device_policy=policy,gpu=False,gpu_available=False,selected_device='cpu',available_devices=['cpu'])
    r.pop('gpu_hardware',None)
    if policy=='ai':
        receipt=gpu_receipt(r)
        r['available_devices'].append('cuda')
        r['gpu_hardware']=receipt['hardware']
        r['gpu_available']=True
    return r


def select_device(runtime,device):
    if device not in ('cpu','cuda'):raise ValueError('Choose device cpu or cuda explicitly')
    if device=='cuda' and (runtime.get('device_policy')!='ai' or not runtime.get('gpu_available')):
        raise ValueError('CUDA requested without a qualified GPU; no CPU fallback')
    return {**runtime,'gpu':device=='cuda','selected_device':device}


def process_group_usage(group,proc=Path('/proc')):
    """RSS includes shared pages per process; conservative sampled bound, not a cgroup."""
    pids=[];rss=0;page=os.sysconf('SC_PAGE_SIZE')
    for p in proc.iterdir():
        if not p.name.isdigit():continue
        try:
            fields=(p/'stat').read_text().rsplit(')',1)[1].split()
            if int(fields[2])!=group:continue
            pids.append(int(p.name));rss+=int((p/'statm').read_text().split()[1])*page
        except (OSError,ValueError,IndexError):continue
    return pids,rss


def gpu_process_memory(pids):
    try:
        r=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory',
            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
        if r.returncode:return None
        total=0
        for line in r.stdout.splitlines():
            values=line.split(',')
            if len(values)==2 and int(values[0].strip()) in pids:
                total+=int(values[1].strip())*1024**2
        return total
    except (OSError,ValueError,subprocess.TimeoutExpired):return None

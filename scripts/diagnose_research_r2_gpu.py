"""Fixed CUDA diagnostic only: no API, data, qualification receipts or research."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from vdc.io import read_json,write_json
from vdc.research_r2_linux import execute_linux

PROBE = '''import json,os,threading,sys
from pathlib import Path
import torch
print('TORCH_IMPORTED',torch.__version__,flush=True)
tid=threading.get_native_id()
p=Path('/proc/self/task')/str(tid)/'comm'
try:
    # Write the same name back; test the real interface, not a fake proc file.
    name=p.read_text();p.write_text(name)
    comm={'status':'passed'}
except OSError as e:
    comm={'status':'failed','errno':e.errno,'error_type':type(e).__name__}
print('PROC_THREAD_COMM',json.dumps(comm),flush=True)
print('CUDA_INIT_START',flush=True)
torch.cuda.init()
x=torch.randn(16,16,device='cuda',requires_grad=True)
(x@x.T).square().mean().backward()
torch.cuda.synchronize()
assert torch.isfinite(x.grad).all()
r={'status':'passed','torch':torch.__version__,'cuda':torch.version.cuda,
   'device':torch.cuda.get_device_name(0),'proc_thread_comm':comm,
   'forward_backward':True}
Path(sys.argv[1]).write_text(json.dumps(r,indent=2))
print('CUDA_FORWARD_BACKWARD_PASSED',flush=True)
'''


def trace_findings(path):
    if not path.exists():return []
    lines=path.read_text(errors='replace').splitlines()
    # Missing optional libraries are common; these are clues, not verdicts.
    denied=[s for s in lines if any(e in s for e in ('= -1 EPERM','= -1 EACCES'))]
    missing=[s for s in lines if '= -1 ENOENT' in s and any(p in s for p in ('/proc/','/dev/','/sys/'))]
    return (denied+missing)[-60:]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    code=a.out/'code';code.mkdir();(code/'gpu.py').write_text(PROBE)
    report={'operation':'gpu_isolation_diagnostic','api_calls_executed':0,
            'research_experiments_executed':0,'qualification_receipt_written':False}
    fields=('CapEff:','Seccomp:','NoNewPrivs:')
    report['host_process_controls']=[s for s in Path('/proc/self/status').read_text().splitlines() if s.startswith(fields)]
    print('DIAG_START host_cuda',flush=True)
    with (a.out/'host.log').open('w') as log:
        try:
            r=subprocess.run([sys.executable,str(code/'gpu.py'),str(a.out/'host.json')],
                             stdout=log,stderr=subprocess.STDOUT,timeout=120)
            report['host_exit_code']=r.returncode
        except subprocess.TimeoutExpired:report['host_exit_code']='timeout'
    print((a.out/'host.log').read_text(errors='replace')[-6000:],flush=True)
    report['host']=read_json(a.out/'host.json') if (a.out/'host.json').exists() else {'status':'failed'}
    if report['host'].get('status')=='passed':
        print('DIAG_START isolated_cuda_with_trace',flush=True)
        config={'runtime_root':str(a.runtime_root),'probe_mode':True,'gpu':True,
                'memory':'64g','cpus':2,'timeout_seconds':120,'diagnostic_trace':True}
        try:
            execute_linux(code,{},a.out/'isolated',['python','/code/gpu.py','/out/result.json'],config)
            report['isolated']=read_json(a.out/'isolated/result.json')
        except Exception as e:
            report['isolated']={'status':'failed','error_type':type(e).__name__,'diagnostic':str(e)}
        log=a.out/'isolated.log'
        if log.exists():print(log.read_text(errors='replace')[-6000:],flush=True)
        report['syscall_clues']=trace_findings(a.out/'isolated_syscalls.log')
    else:
        report['isolated']={'status':'not_run','reason':'host_cuda_failed'}
    report['out']=str(a.out)
    write_json(a.out/'diagnostic.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return 0 if report['isolated'].get('status')=='passed' else 2


if __name__=='__main__':sys.exit(main())

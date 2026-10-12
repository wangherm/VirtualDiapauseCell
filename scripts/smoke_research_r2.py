"""Actual isolated multi-file fit/reload test, explicitly synthetic."""
import argparse
import json
from pathlib import Path
import numpy as np
from vdc.io import write_json,save_npz
from vdc.research_r2_tools import write_code
from vdc.research_r2_runtime import docker_image,execute,seal_task,run_candidate


def execute_probe(code, inputs, out, command, runtime, **kwargs):
    """Only fixed software probes: surface their diagnostics in the setup log."""
    try:
        return execute(code,inputs,out,command,runtime,**kwargs)
    except Exception:
        log=out.parent/(out.name+'.log')
        if log.is_file():
            print('FAILED_PROBE_LOG '+str(log),flush=True)
            print(log.read_text(errors='replace')[-6000:],flush=True)
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path)
    p.add_argument('--backend',choices=['docker','linux_landlock_chroot'],default='docker')
    p.add_argument('--runtime-root',type=Path)
    p.add_argument('--gpu',action='store_true')
    p.add_argument('--image',default='vdc-research-r2:1');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    runtime={'memory':'4g','cpus':2,'timeout_seconds':120,'gpu':False}
    if a.backend=='linux_landlock_chroot':
        from vdc.research_r2_linux import runtime_info
        runtime.update(backend=a.backend,runtime_root=str(a.runtime_root),probe_mode=True)
        runtime['image_id']=runtime_info(runtime)['runtime_id']
    else: runtime['image_id']=docker_image(a.image)
    # Pure runtime probe checks that no host credentials/root data/network are mounted.
    print('SMOKE_START cpu_isolation_and_torch (RSS budget 4 GiB; separate address-space limit)',flush=True)
    write_code(a.out/'probe_code',{'probe.py':'''import os,socket,json,mmap,resource
from pathlib import Path
print('PROBE_STARTED',flush=True)
assert not os.environ.get("VDC_RESEARCH_API_KEY")
assert not Path("/root/autodl-tmp/vdc-private").exists()
assert os.geteuid()!=0
with open('/dev/null','wb') as sink:sink.write(b'probe')
try:
    Path('/input/sentinel.txt').write_text('corrupt')
except OSError: pass
else: raise RuntimeError('Input was writable')
try:
    socket.create_connection(("1.1.1.1",443),timeout=1)
except OSError: pass
else: raise RuntimeError("Unexpected outbound network")
print('ISOLATION_PASSED',flush=True)
# Reserve addresses without touching pages: this must not consume the RSS budget.
with mmap.mmap(-1,8*1024**3,prot=mmap.PROT_NONE if hasattr(mmap,'PROT_NONE') else 0):
    pass
print('TORCH_IMPORT_START',flush=True)
import torch
print('TORCH_IMPORT_PASSED '+torch.__version__,flush=True)
x=torch.randn(8,8,requires_grad=True);(x@x.T).sum().backward()
assert torch.isfinite(x.grad).all()
Path("/out/probe.json").write_text(json.dumps({"status":"passed","synthetic":True,"torch_cpu_backward":True,
    "address_space_soft_limit":resource.getrlimit(resource.RLIMIT_AS)[0],"virtual_reservation_test":True}))
'''})
    (a.out/'sentinel.txt').write_text('original')
    execute_probe(a.out/'probe_code',{'sentinel.txt':a.out/'sentinel.txt'},a.out/'probe',['python','/code/probe.py'],runtime)
    assert (a.out/'sentinel.txt').read_text()=='original'
    print('SMOKE_PASSED cpu_isolation_and_torch',flush=True)
    if a.gpu:
        print('SMOKE_START cuda_forward_backward_and_reload',flush=True)
        from vdc.research_r2_devices import gpu_inventory
        hardware=gpu_inventory()
        if not hardware['cuda_available']:raise RuntimeError('Host CUDA unavailable; AI GPU policy cannot start')
        write_json(a.out/'hardware.json',hardware)
        write_code(a.out/'gpu_code',{'gpu.py':'''import torch,json
from pathlib import Path
assert torch.cuda.is_available()
torch.manual_seed(42)
m=torch.nn.Linear(16,4).cuda();x=torch.randn(64,16,device='cuda')
o=torch.optim.AdamW(m.parameters(),lr=.001)
for _ in range(3):
    o.zero_grad();loss=m(x).square().mean();loss.backward();o.step()
assert torch.isfinite(loss)
bf16=torch.cuda.is_bf16_supported()
if bf16:
    b=torch.randn(32,32,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    (b@b.T).float().mean().backward();assert torch.isfinite(b.grad).all()
torch.cuda.synchronize()
m.eval()
torch.save({'weights':{k:v.cpu() for k,v in m.state_dict().items()},'x':x.cpu(),'prediction':m(x).detach().cpu()},'/out/probe.pt')
Path('/out/gpu.json').write_text(json.dumps({'status':'passed','torch':torch.__version__,
    'device':torch.cuda.get_device_name(0),'bf16_backward_passed':bf16,
    'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()}))
''','reload.py':'''import torch,json
from pathlib import Path
d=torch.load('/model/probe.pt',map_location='cpu',weights_only=True)
m=torch.nn.Linear(16,4).cuda();m.load_state_dict(d['weights']);m.eval()
p=m(d['x'].cuda()).detach().cpu()
delta=float((p-d['prediction']).abs().max());assert torch.allclose(p,d['prediction'],rtol=1e-5,atol=1e-6)
Path('/out/reload.json').write_text(json.dumps({'status':'passed','max_abs_delta':delta}))
'''})
        cuda={**runtime,'gpu':True,'memory':'64g'}
        execute_probe(a.out/'gpu_code',{},a.out/'gpu',['python','/code/gpu.py'],cuda)
        execute_probe(a.out/'gpu_code',{},a.out/'gpu_reload',['python','/code/reload.py'],cuda,model=a.out/'gpu')
        print('SMOKE_PASSED cuda_forward_backward_and_reload',flush=True)
    rng=np.random.default_rng(2);x=rng.normal(size=(12,3));y=x@np.array([[1.],[2.],[-1.]])
    prepared=a.out/'fixture';save_npz(prepared/'train.npz',x=x[:8],y=y[:8])
    save_npz(prepared/'query.npz',x=x[8:]);save_npz(prepared/'truth.npz',y=y[8:])
    for name,indices in [('train',range(8)),('query',range(8,12))]:
        # Unit-testing fixtures: never registered in an actual research inventory.
        write_json(prepared/(name+'_rows.json'),[{'study_family':'software_fixture','biological_unit':str(i),'origin':'software_fixture'} for i in indices])
    write_json(prepared/'schema.json',{'features':['a','b','c'],'targets':['y']})
    contract={k:'software fixture' for k in ('question','target_definition','target_scale','allowed_inputs','unit_definition','preprocessing','split_rationale')}
    contract.update(mission='state',source_ids=['fixture'])
    seal_task(prepared,a.out/'task',contract,['fixture'])
    write_code(a.out/'code',{
        'helper.py':'import numpy as np\ndef load(p): return np.load(p,allow_pickle=False)\n',
        'fit.py':'''from helper import *
d=load('/input/train.npz')
np.savez('/out/model.npz',beta=np.linalg.lstsq(d['x'],d['y'],rcond=None)[0])
''', 'predict.py':'''from helper import *
from pathlib import Path
assert not Path('/input/truth.npz').exists()
assert not Path('/input/train.npz').exists()
x=load('/input/query.npz')['x']; b=load('/model/model.npz')['beta']
np.savez('/out/predictions.npz',prediction=x@b)
'''})
    from vdc.io import read_json
    result=run_candidate(a.out/'task',a.out/'experiment',a.out/'code',runtime,
                         Path(__file__).resolve().parents[1],read_json(a.out/'task/baselines.json'),
                         {'reference':'train_mean'})
    report={'status':'passed','scope':'synthetic_software_only','real_research_executed':False,
            'gpu_qualification':'passed' if a.gpu else 'not_requested',
            'hardware':hardware if a.gpu else None,
            'reload_verified':result['reload_verified'],'mse':result['metrics']['macro_unit_mse'],
            'image_id':runtime['image_id']}
    write_json(a.out/'status.json',report);print(json.dumps(report,indent=2))
    if a.backend=='linux_landlock_chroot':
        write_json(a.runtime_root/'probe_passed.json',{'runtime_id':runtime['image_id'],**report})
        if a.gpu:write_json(a.runtime_root/'gpu_probe_passed.json',{'runtime_id':runtime['image_id'],
            'status':'passed','hardware':hardware,'forward_backward':read_json(a.out/'gpu/gpu.json'),
            'reload':read_json(a.out/'gpu_reload/reload.json')})


if __name__=='__main__':main()

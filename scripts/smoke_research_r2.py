"""Actual isolated multi-file fit/reload test, explicitly synthetic."""
import argparse
import json
from pathlib import Path
import numpy as np
from vdc.io import write_json,save_npz
from vdc.research_r2_tools import write_code
from vdc.research_r2_runtime import docker_image,execute,seal_task,run_candidate


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
    write_code(a.out/'probe_code',{'probe.py':'''import os,socket,json
from pathlib import Path
assert not os.environ.get("VDC_RESEARCH_API_KEY")
assert not Path("/root/autodl-tmp/vdc-private").exists()
assert os.geteuid()!=0
try:
    Path('/input/sentinel.txt').write_text('corrupt')
except OSError: pass
else: raise RuntimeError('Input was writable')
try:
    socket.create_connection(("1.1.1.1",443),timeout=1)
except OSError: pass
else: raise RuntimeError("Unexpected outbound network")
import torch
x=torch.randn(8,8,requires_grad=True);(x@x.T).sum().backward()
assert torch.isfinite(x.grad).all()
Path("/out/probe.json").write_text(json.dumps({"status":"passed","synthetic":True,"torch_cpu_backward":True}))
'''})
    (a.out/'sentinel.txt').write_text('original')
    execute(a.out/'probe_code',{'sentinel.txt':a.out/'sentinel.txt'},a.out/'probe',['python','/code/probe.py'],runtime)
    assert (a.out/'sentinel.txt').read_text()=='original'
    if a.gpu:
        write_code(a.out/'gpu_code',{'gpu.py':'''import torch,json
from pathlib import Path
assert torch.cuda.is_available()
x=torch.randn(16,16,device='cuda',requires_grad=True);(x@x.T).sum().backward()
assert torch.isfinite(x.grad).all()
Path('/out/gpu.json').write_text(json.dumps({'status':'passed','torch':torch.__version__,'device':torch.cuda.get_device_name(0)}))
'''})
        execute(a.out/'gpu_code',{},a.out/'gpu',['python','/code/gpu.py'],{**runtime,'gpu':True,'memory':'64g'})
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
            'reload_verified':result['reload_verified'],'mse':result['metrics']['macro_unit_mse'],
            'image_id':runtime['image_id']}
    write_json(a.out/'status.json',report);print(json.dumps(report,indent=2))
    if a.backend=='linux_landlock_chroot':
        write_json(a.runtime_root/'probe_passed.json',{'runtime_id':runtime['image_id'],**report})
        if a.gpu:write_json(a.runtime_root/'gpu_probe_passed.json',{'runtime_id':runtime['image_id'],'status':'passed'})


if __name__=='__main__':main()

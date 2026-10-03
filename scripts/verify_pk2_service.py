"""Independent HTTP client: actually start, query, stop and restart the snapshot."""
from pathlib import Path
import argparse,json,os,socket,subprocess,sys,time,urllib.request,urllib.error
import numpy as np
from vdc.io import write_json,read_json,object_hash
ROOT=Path(__file__).resolve().parents[1]

def request(url,payload=None):
    data=None if payload is None else json.dumps(payload).encode()
    req=urllib.request.Request(url,data=data,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=60) as response:return json.load(response)

def verify(snapshot,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);results=[]
    expected=read_json(Path(snapshot)/'service_manifest.json')['snapshot_id']
    for turn in range(2):
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        base=f'http://127.0.0.1:{port}'
        with (out/f'process_{turn}.log').open('w',encoding='utf-8') as log:
            process=subprocess.Popen([sys.executable,str(ROOT/'scripts/serve_pk2.py'),'serve','--snapshot',str(Path(snapshot).resolve()),'--port',str(port)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            try:
                deadline=time.monotonic()+60
                while True:
                    if process.poll() is not None:raise RuntimeError('Service exited during startup; inspect process log')
                    try:health=request(base+'/health');break
                    except (urllib.error.URLError,TimeoutError):
                        if time.monotonic()>deadline:raise RuntimeError('Service startup timed out')
                        time.sleep(.5)
                if health['snapshot_id']!=expected:raise ValueError('Unexpected service identity')
                caps=request(base+'/capabilities');predictions={}
                for name,entry in caps['models'].items():
                    rows=request(base+'/samples?view='+entry['view'])
                    sample=next((r for r in rows if r['split']=='validation'),rows[0])
                    predictions[name]=request(base+'/analyse',{'model':name,'observation_ids':[sample['observation_id']]})
                try:request(base+'/analyse',{'model':next(iter(caps['models'])),'observation_ids':['__unknown_reserved__']})
                except urllib.error.HTTPError as exc:
                    if exc.code!=422:raise
                else:raise AssertionError('Unknown sample was accepted')
                for module in ['endpoint','transition']:
                    if module in caps['public_models']:
                        rows=request(base+'/public_cases?module='+module)
                        sample=next((r for r in rows if r['split']=='validation'),rows[0])
                        predictions[module]=request(base+'/public_response',{'module':module,'observation_id':sample['observation_id']})
                if 'functional' in caps['public_models']:
                    with np.load(Path(snapshot)/'functional/model.npz',allow_pickle=False) as a:hour=float(a['hour_levels'][0])
                    predictions['functional']=request(base+'/functional',{'hours_after_release':hour,'history_days':4})
                results.append(predictions)
            finally:
                process.terminate()
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
    if object_hash(results[0])!=object_hash(results[1]):raise AssertionError('Outputs changed after process restart')
    write_json(out/'predictions.json',results[0])
    report={'status':'passed','snapshot_id':expected,'process_starts':2,'stopped_after_each_check':True,
        'model_queries_per_start':len(results[0]),'reload_outputs_identical':True,'unknown_sample_rejected':True,
        'scientific_validation':False,'query_scope':'development_only'}
    write_json(out/'verification.json',report);print(json.dumps(report,indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();verify(a.snapshot,a.out)

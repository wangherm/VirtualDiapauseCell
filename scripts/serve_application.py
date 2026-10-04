"""Private loopback browser for saved frozen application results; no arbitrary model queries."""
from pathlib import Path
import argparse,json,os,socket,subprocess,sys,time,urllib.request
import numpy as np
from fastapi import FastAPI,HTTPException
from fastapi.responses import HTMLResponse
from vdc.io import read_json,write_json,sha256
from vdc.pk1_assets import checked_member,verify_files
from vdc.experimental import jsonable


def create_app(run):
    run=Path(run).resolve();snapshot=read_json(run/'application_snapshot.json');verify_files(run,snapshot['files'])
    app=FastAPI(title='Frozen retrospective applications')
    @app.get('/health')
    def health():return {'status':'ready','freeze_id':snapshot['freeze_id'],'mode':'private_saved_application_results'}
    @app.get('/samples')
    def samples():
        out=[]
        for group in ('bulk','single_exit','core','core_celltypes','stress'):
            p=run/'applications'/group/'result.json'
            if p.exists():out.extend({'group':group,**r} for r in read_json(p)['rows'])
        return out
    @app.get('/sample/{group}/{identity}')
    def sample(group:str,identity:str):
        if group not in {'bulk','single_exit','core','core_celltypes','stress'} or len(identity)!=20 or any(c not in '0123456789abcdef' for c in identity):raise HTTPException(404)
        folder=checked_member(run,'applications/'+group+'/'+identity);p=folder/'result.json'
        if not p.exists():raise HTTPException(404)
        result=read_json(p);arrays={}
        for p in folder.glob('*.npz'):
            rel=p.relative_to(run).as_posix()
            if rel not in snapshot['files'] or sha256(p)!=snapshot['files'][rel]:raise HTTPException(409,'Saved query changed')
            with np.load(p,allow_pickle=False) as a:
                # Gene arrays remain downloadable in the private report; browser focuses on programmes.
                arrays[p.stem]={k:jsonable(a[k]) for k in a.files if not k.startswith(('gene_','hidden_gene_'))}
        return {'result':result,'arrays':arrays,'projection':read_json(folder/'projection.json') if (folder/'projection.json').exists() else None}
    @app.get('/',response_class=HTMLResponse)
    def home():return '''<!doctype html><meta charset="utf-8"><title>VDC frozen applications</title>
<style>body{font:16px/1.6 system-ui;margin:30px;max-width:1200px;color:#183542}select{max-width:95%;padding:8px}pre{white-space:pre-wrap;background:#f2f6f7;padding:18px}h1{font-size:25px}</style>
<h1>Virtual Diapause Cell · 冻结样本应用</h1><p>回顾性、实验性结果。当前表达定位不是未来预测；programme 重构不是 clean truth。结果不用于重新选择模型；不生成功能 depth。</p>
<select id="choice"><option>加载样本清单…</option></select><pre id="result"></pre>
<script>let rows=[];const s=document.getElementById('choice'),r=document.getElementById('result');fetch('samples').then(x=>x.json()).then(x=>{rows=x;s.replaceChildren();x.forEach((v,i)=>{let o=document.createElement('option');o.value=i;o.textContent=[v.group,v.sample_key,v.cell_type||'',v.status].join(' / ');s.appendChild(o)});s.onchange=async()=>{let v=rows[s.value];if(v.status!=='queried'){r.textContent=JSON.stringify(v,null,2);return}let data=await fetch('sample/'+v.group+'/'+v.result_folder).then(x=>x.json());r.textContent=JSON.stringify(data,null,2)};s.onchange()});</script>'''
    return app


def verify(run):
    run=Path(run)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    with (run/'browser_verify.log').open('w',encoding='utf-8') as log:
        p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--run',str(run.resolve()),'--port',str(port)],stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        try:
            deadline=time.monotonic()+60
            while True:
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=3) as f:health=json.load(f)
                    break
                except OSError:
                    if p.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Application browser failed to start')
                    time.sleep(.5)
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/samples',timeout=30) as f:rows=json.load(f)
            first=next((r for r in rows if r['status']=='queried'),None)
            if first:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/sample/{first['group']}/{first['result_folder']}",timeout=30) as f:response=json.load(f)
                if response['result']['freeze_id']!=health['freeze_id']:raise ValueError('Browser returned a different frozen run')
            write_json(run/'browser_verification.json',{'status':'passed','health':health,'listed_profiles':len(rows),
                'saved_query_response_checked':first is not None,'new_model_inference':False,'live_service_started_by_this_check':False})
        finally:
            p.terminate()
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait()


def check_running(run,port):
    run=Path(run);expected=read_json(run/'application_snapshot.json')['freeze_id'];deadline=time.monotonic()+45
    while True:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as f:health=json.load(f)
            if health.get('freeze_id')!=expected:raise ValueError('Port belongs to another frozen run; existing service was not stopped')
            write_json(run/'deployment.json',{'status':'running_verified','host':'127.0.0.1','port':port,'health':health});return
        except OSError:
            if time.monotonic()>deadline:
                write_json(run/'deployment.json',{'status':'failed','port':port,'reason':'health check timeout; inspect service.log'})
                raise RuntimeError('Application result browser failed health check')
            time.sleep(.5)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--port',type=int,default=8768);p.add_argument('--verify',action='store_true');p.add_argument('--check-running',action='store_true');a=p.parse_args()
    if a.verify:verify(a.run)
    elif a.check_running:check_running(a.run,a.port)
    else:
        import uvicorn
        uvicorn.run(create_app(a.run),host='127.0.0.1',port=a.port)

"""Loopback-only browser/API for saved CW1 development predictions."""
from pathlib import Path
import argparse,json,subprocess,sys,socket,time,urllib.request
import numpy as np
from fastapi import FastAPI,HTTPException
from fastapi.responses import HTMLResponse
from vdc.io import read_json,write_json,object_hash,sha256
from vdc.pk1_assets import verify_files
from vdc.experimental import jsonable


def create_app(run):
    run=Path(run).resolve();snapshot=read_json(run/'results_snapshot.json')
    if object_hash({k:v for k,v in snapshot.items() if k!='snapshot_id'})!=snapshot['snapshot_id']:raise ValueError('Snapshot identity changed')
    verify_files(run,snapshot['files']);app=FastAPI(title='CW1 development results')
    @app.get('/health')
    def health():return {'status':'ready','snapshot_id':snapshot['snapshot_id'],'mode':'experimental_saved_development_results','live_inference':False}
    @app.get('/tasks')
    def tasks():return read_json(run/'queue_status.json')
    @app.get('/task/{name}')
    def task(name:str):
        if name not in read_json(run/'queue_status.json')['tasks']:raise HTTPException(404)
        paths=[n for n in snapshot['files'] if n.startswith('tasks/'+name+'/')];files={}
        for n in paths:
            p=run/n
            if sha256(p)!=snapshot['files'][n]:raise HTTPException(409,'Saved result changed')
            if p.suffix=='.json':files[n]=read_json(p)
            elif p.name=='predictions.npz' and p.parent.name=='validation':
                with np.load(p,allow_pickle=False) as a:files[n]={k:jsonable(a[k]) for k in a.files}
        return {'snapshot_id':snapshot['snapshot_id'],'task_state':read_json(run/'queue_status.json')['tasks'][name],'files':files}
    @app.get('/',response_class=HTMLResponse)
    def home():return '''<!doctype html><meta charset="utf-8"><title>CW1 开发结果</title>
<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;color:#172333}select,button{padding:8px}pre{white-space:pre-wrap;background:#f5f7fa;padding:20px}canvas{border:1px solid #bbb}</style>
<h1>Clock–Identity–Wave · experimental</h1><p>开发集结果；clock 是参考位置，不是恢复百分比或 depth。显示保存的预测，不执行新查询。</p>
<select id="jobs"></select> <button id="load">查看</button><p id="caption"></p><canvas id="plot" width="1000" height="260"></canvas><pre id="result"></pre>
<script>
let state; const jobs=document.getElementById('jobs'),res=document.getElementById('result');
fetch('/tasks').then(r=>r.json()).then(s=>{state=s; for(const [k,v] of Object.entries(s.tasks)){let o=document.createElement('option');o.value=k;o.textContent=k+' — '+v.status;jobs.append(o)}res.textContent=JSON.stringify({status:s.status,counts:s.counts},null,2)});
document.getElementById('load').onclick=async()=>{
  let data=await(await fetch('/task/'+encodeURIComponent(jobs.value))).json();res.textContent=JSON.stringify(data,null,2);
  let entries=Object.entries(data.files||{}),a=entries.find(([k,v])=>k.endsWith('validation/predictions.npz'));
  let status=entries.find(([k,v])=>k.endsWith('validation/query_status.json'));
  let stages=entries.find(([k,v])=>k.endsWith('stage_status.json'));
  let caption=document.getElementById('caption'),c=document.getElementById('plot').getContext('2d');c.clearRect(0,0,1000,260);
  caption.textContent=stages?'身份确认、clock定位、表达预测的逐阶段状态见 stage_status；数值建议不算作 LLM 成功。':'';
  if(!a)return;let t=a[1].clock||[],v=t.filter(Number.isFinite);if(!v.length)return;
  let lo=Math.min(0,...v),hi=Math.max(1,...v);c.strokeStyle='#aaa';c.beginPath();c.moveTo(30,225);c.lineTo(980,225);c.stroke();
  t.forEach((z,i)=>{if(Number.isFinite(z)){c.fillStyle=status&&status[1].status[i]==='located'?'#176d9a':'#b54f24';c.beginPath();c.arc(40+i*920/Math.max(t.length-1,1),220-(z-lo)/(hi-lo)*190,4,0,7);c.fill()}});
  caption.textContent+=' 逐 profile 参考位置（不裁剪）：'+lo.toFixed(3)+' 至 '+hi.toFixed(3)+'。蓝色为训练参考范围内，橙色为范围外；各身份范围分别定义，不能用统一 0–1 判断。';
};
</script>'''
    return app


def verify(run):
    run=Path(run)
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    with (run/'browser_verify.log').open('w') as log:
        process=subprocess.Popen([sys.executable,__file__,'--run',str(run),'--port',str(port)],stdout=log,stderr=log)
        try:
            deadline=time.monotonic()+45
            while True:
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as f:health=json.load(f)
                    break
                except OSError:
                    if process.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Browser startup failed; see browser_verify.log')
                    time.sleep(.5)
            expected=read_json(run/'results_snapshot.json')['snapshot_id']
            if health['snapshot_id']!=expected:raise ValueError('Wrong browser snapshot')
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/tasks',timeout=5) as f:states=json.load(f)
            name=next((k for k,v in states['tasks'].items() if v['status']=='completed' and k.startswith('numeric_')),None)
            if name:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/task/{name}',timeout=30) as f:result=json.load(f)
                if result['snapshot_id']!=expected or not result['files']:raise ValueError('Saved prediction response invalid')
            write_json(run/'browser_verification.json',{'status':'passed','new_process_http':True,'saved_numeric_query':name,'snapshot_id':expected,'service_still_running':False})
        finally:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--port',type=int,default=8769);p.add_argument('--verify',action='store_true');a=p.parse_args()
    if a.verify:verify(a.run)
    else:
        import uvicorn
        uvicorn.run(create_app(a.run),host='127.0.0.1',port=a.port)

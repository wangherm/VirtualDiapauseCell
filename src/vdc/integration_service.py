"""Local application over the same VirtualDiapauseCell call used by Python and CLI."""
from pathlib import Path
import json,uuid,zipfile
from .integration import VirtualDiapauseCell,read_request
from .pk1_assets import checked_member
from .io import write_json


def create_app(model,output,base_path=None):
    from fastapi import FastAPI,HTTPException,Request
    from fastapi.responses import HTMLResponse,FileResponse
    engine=VirtualDiapauseCell(model,base_path);out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    app=FastAPI(title='Virtual Diapause Cell INT1')
    @app.get('/health')
    def health():return {'status':'ready','bundle_id':engine.manifest['bundle_id'],'full_artifacts_packaged':engine.manifest['full_ready'],'live_qwen_loaded':engine.engine is not None}
    @app.get('/capabilities')
    def capabilities():return {k:engine.manifest[k] for k in ('views','public_models','regulons','full_ready')}
    def run(body):
        if 'input' in body.get('request',{}):raise ValueError('HTTP JSON accepts inline arrays; use upload for files')
        token=uuid.uuid4().hex;dest=out/token
        r=engine.analyse(body['request'],body.get('mode','full'),body.get('variant','domain_clock_wave'),dest)
        with zipfile.ZipFile(out/(token+'.zip'),'x',zipfile.ZIP_DEFLATED) as z:
            for f in dest.rglob('*'):
                if f.is_file():z.write(f,f.relative_to(dest).as_posix())
        return {'result':r,'report_url':'/results/'+token+'/report.html','download_url':'/download/'+token}
    @app.post('/analyse')
    def analyse(body:dict):
        try:return run(body)
        except (ValueError,KeyError,TypeError) as e:raise HTTPException(422,str(e))
    @app.post('/upload')
    async def upload(request:Request):
        import tempfile
        from starlette.concurrency import run_in_threadpool
        form=await request.form()
        try:
            body=json.loads(str(form['metadata']))
            with tempfile.TemporaryDirectory(prefix='vdc_upload_',dir=out) as tmp:
                p=Path(tmp)
                for field,name in [('matrix','matrix'),('samples','samples.csv')]:
                    data=await form[field].read()
                    if len(data)>128*1024*1024:raise ValueError('Use CLI for files above 128 MiB')
                    (p/name).write_bytes(data)
                req=body['request'];fmt=str(form['format'])
                req['input']={'format':fmt,'path':'matrix','matrix':'matrix','samples':'samples.csv','counts_layer':str(form.get('counts_layer',''))}
                write_json(p/'request.json',req);body['request']=read_request(p/'request.json')
                return await run_in_threadpool(run,body)
        except (ValueError,KeyError,TypeError) as e:raise HTTPException(422,str(e))
    @app.get('/results/{token}/{file:path}')
    def result_file(token:str,file:str):
        if len(token)!=32 or any(c not in '0123456789abcdef' for c in token):raise HTTPException(404)
        try:p=checked_member(out/token,file)
        except ValueError:raise HTTPException(404)
        if not p.is_file():raise HTTPException(404)
        return FileResponse(p)
    @app.get('/download/{token}')
    def download(token:str):
        if len(token)!=32 or any(c not in '0123456789abcdef' for c in token):raise HTTPException(404)
        p=out/(token+'.zip')
        if not p.is_file():raise HTTPException(404)
        return FileResponse(p,filename='VDC_'+token+'.zip')
    @app.get('/',response_class=HTMLResponse)
    def home():return PAGE
    return app


PAGE='''<!doctype html><meta charset="utf-8"><title>Virtual Diapause Cell</title>
<style>body{font:16px system-ui;max-width:1080px;margin:36px auto;padding:16px;color:#173047}textarea{width:100%;height:300px}button,input,select{margin:8px;padding:8px}pre{white-space:pre-wrap}iframe{width:100%;height:700px;border:1px solid #ccc}</style>
<h1>Virtual Diapause Cell · INT1</h1><p>提交新表达输入，实际计算参考 clock、waves、偏离与基因预测。Full 同时调用领域 Qwen。研究实验版；缺参考的能力保持 unsupported。</p>
<p>上传包含 request/mode/variant 的 JSON；或在 JSON 中填写物种、背景与 view，再上传矩阵及样本 CSV。CSV 第一列 sample_id；样本表包含 sample_id、biological_unit、identity、identity_source。</p>
<input id="json" type="file" accept=".json"><textarea id="body">{"mode":"full","variant":"domain_clock_wave","request":{"request_id":"new-request","species":"Nothobranchius furzeri","context":"embryonic_diapause_exit","view":"bulk","scale":"counts","question":"Describe the observed deviations and their limitations"}}</textarea>
<p>矩阵 <input id="matrix" type="file"> 样本表 <input id="samples" type="file" accept=".csv"></p>
<select id="format"><option value="csv">CSV counts</option><option value="h5ad">H5AD</option></select><input id="layer" placeholder="H5AD counts layer (required)">
<button id="go">实际分析并导出</button><a id="download" hidden>下载完整结果</a><pre id="status"></pre><iframe id="report" hidden></iframe>
<script>const el=id=>document.getElementById(id);el('json').onchange=async()=>{el('body').value=await el('json').files[0].text()};
el('go').onclick=async()=>{el('go').disabled=true;el('status').textContent='正在实际计算；Full 需要加载 Qwen 并生成回答…';try{
let r;if(el('matrix').files.length){let f=new FormData();f.set('metadata',el('body').value);f.set('matrix',el('matrix').files[0]);f.set('samples',el('samples').files[0]);f.set('format',el('format').value);f.set('counts_layer',el('layer').value);r=await fetch('/upload',{method:'POST',body:f})}
else{r=await fetch('/analyse',{method:'POST',headers:{'Content-Type':'application/json'},body:el('body').value})}
let x=await r.json();if(!r.ok)throw Error(JSON.stringify(x));el('status').textContent=JSON.stringify({status:x.result.status,components:x.result.components},null,2);el('report').src=x.report_url;el('report').hidden=false;el('download').href=x.download_url;el('download').hidden=false;
}catch(e){el('status').textContent=String(e)}finally{el('go').disabled=false}};</script>'''

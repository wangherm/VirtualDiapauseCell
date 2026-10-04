"""Freeze newly evaluated PK2 models separately from the existing PK1 service."""
from pathlib import Path
import os,shutil,subprocess,sys,time,urllib.request,json,socket
from .io import read_json,write_json,sha256,object_hash
from .pk1_assets import verify_files
from .pk2_service import implementation_identity,runtime_identity
from .pk2_data import DataUnavailable

def register(run,out):
    run=Path(run);out=Path(out);c=read_json(run/'config.json');statuses=read_json(run/'queue_status.json')['tasks'];plan=read_json(run/'plan.json')
    snapshot=out/'snapshot';snapshot.mkdir(parents=True);files={};registry={};views={}
    def cp(source,relative):
        target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);files[relative]=sha256(target)
    for name,s in statuses.items():
        spec=plan[name];j=spec['params'].get('job',{})
        if s['status']!='evaluated_new' or j.get('family') not in {'killifish_adaptation','semantic_increment','feature_resolution'}:continue
        source=run/'tasks'/name;verify_files(source,s['files']);meta=read_json(source/'model/run.json')
        if meta['is_synthetic']:raise ValueError('Synthetic model cannot become a PK2 research snapshot')
        input_task=j['bundle_task'];view=input_task.removeprefix('input_');views[view]=input_task
        for file in ['run.json','best.pt','metrics.json']:cp(source/'model'/file,name+'/'+file)
        cp(source/'evaluation/result.json',name+'/result.json')
        registry[name]={'view':view,'route':j.get('route',j.get('feature_view')),'seed':j.get('seed',42),
            'checkpoint_sha256':sha256(source/'model/best.pt'),'training_steps':read_json(source/'model/metrics.json')['steps_completed'],
            'scope':meta['scope'],'science_status':'unvalidated','wave':None,'source_job':name,
            'semantic_provenance':meta.get('semantic_provenance'),'transfer':meta.get('transfer')}
        wave='waves_'+view
        if statuses.get(wave,{}).get('status')=='completed':
            w=read_json(run/'tasks'/wave/'result.json')
            if w['encoder_sha256']==registry[name]['checkpoint_sha256']:
                for file in ['programme/reference.json','programme/reference.npz','result.json']:cp(run/'tasks'/wave/file,wave+'/'+file)
                registry[name]['wave']=wave
    if not registry:raise DataUnavailable('No newly evaluated deployable PK2 development model')
    for view,task in views.items():
        for file in ['manifest.json','observations.npz']:cp(run/'tasks'/task/'bundle'/file,f'clock_{view}/bundle/{file}')
    cp(Path(c['private_root'])/'sample_roles.json','sample_roles.json')
    public={};analyses={}
    if statuses.get('endpoint',{}).get('status')=='completed':
        for file in ['response_model.json','response_model.npz','dataset/response.json','dataset/response.npz','result.json']:cp(run/'tasks/endpoint'/file,'endpoint/'+file)
        public['endpoint']={'kind':'new_PK2_response_fit','scope':read_json(snapshot/'endpoint/response_model.json')['scope']}
    if statuses.get('functional',{}).get('status')=='completed':
        for file in ['model.npz','result.json']:cp(run/'tasks/functional'/file,'functional/'+file)
        public['functional']={'kind':'group_level_binomial_readout','expression_depth':False}
    for name,s in statuses.items():
        if plan[name]['kind'] not in {'waves','analysis','type_waves'}:continue
        path=run/'tasks'/name/'result.json'
        analyses[name]={'execution_status':s['status'],'reason':s.get('reason'),'request_kind':'saved_development_evaluation_not_new_prediction'}
        if s['status']=='completed' and path.exists():cp(path,'analyses/'+name+'.json');analyses[name]['report']='analyses/'+name+'.json'
    manifest={'profile':'experimental','models':registry,'views':list(views),'public_models':public,'analyses':analyses,'files':files,
        'source_run_signature':read_json(run/'run_manifest.json')['signature'],'implementation':implementation_identity(),'runtime':runtime_identity(),
        'query_policy':'existing approved development observations only; original reserved roles retained',
        'qwen_live_generation':'not enabled; actual base/domain/shuffle buffers retained per numerical model',
        'new_PK2_training_executed':True,'scientific_approval':False,'PK1_snapshot_unchanged':True}
    manifest['integration_only']=c.get('integration_steps') is not None
    manifest['snapshot_id']=object_hash(manifest);write_json(snapshot/'service_manifest.json',manifest)
    root=Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable,str(root/'scripts/verify_pk2_service.py'),'--snapshot',str(snapshot),'--out',str(out/'http_verification')],check=True,cwd=root)
    deployment={'status':'not_started_in_this_environment'}
    if sys.platform.startswith('linux') and c.get('integration_steps') is None:
        if not shutil.which('screen'):raise RuntimeError('screen missing; PK2 snapshot verified but service not started')
        port=8766
        with socket.socket() as s:s.bind(('127.0.0.1',port))
        session='vdc_pk2_models_'+str(int(time.time()))
        subprocess.run(['screen','-L','-Logfile',str(out/'service.log'),'-dmS',session,sys.executable,
            str(root/'scripts/serve_pk2.py'),'serve','--snapshot',str(snapshot),'--port',str(port)],check=True,cwd=root)
        deadline=time.monotonic()+60
        while True:
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=3) as response:health=json.load(response)
                if health['snapshot_id']!=manifest['snapshot_id']:raise RuntimeError('Unexpected live snapshot identity')
                deployment={'status':'ready','session':session,'port':port,'health':health};break
            except (OSError,ValueError):
                if time.monotonic()>deadline:raise RuntimeError('PK2 screen service did not become healthy; see service.log')
                time.sleep(1)
    write_json(out/'result.json',{'snapshot_id':manifest['snapshot_id'],'models':len(registry),'deployment':deployment,
        'service_scope':'New PK2 state/endpoint/functional calls, checkpoint-matched programme waves; other module assessments served as explicitly saved development reports',
        'verification':read_json(out/'http_verification/verification.json')})

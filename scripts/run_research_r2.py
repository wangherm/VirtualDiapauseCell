"""Actual R2 controller. No silent R1/graph or unisolated execution fallback."""
import argparse
import json
import os
from pathlib import Path
import sys
from vdc.io import read_json, write_json
from vdc.research_r2 import Engine
from vdc.research_r2_tools import inventory, tree_hash
from vdc.research_r2_runtime import resolve_runtime, execute, numeric_npz
from vdc.research_api import verify_connection_check
from run_all_modules import exclusive_run

ROOT = Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['doctor','run','status','infer'])
    p.add_argument('--config',type=Path,default=os.environ.get('VDC_RESEARCH_CONFIG',ROOT/'configs/research_r2.json'))
    p.add_argument('--private-root',type=Path,default=os.environ.get('VDC_PRIVATE_ROOT','/root/autodl-tmp/vdc-private'))
    p.add_argument('--asset-report',default=os.environ.get('VDC_RESEARCH_ASSET_REPORT'))
    p.add_argument('--extra-data-root',type=Path,action='append',default=[])
    p.add_argument('--run',type=Path)
    p.add_argument('--retry-api-errors',action='store_true')
    p.add_argument('--bundle',type=Path); p.add_argument('--input',type=Path); p.add_argument('--out',type=Path)
    a=p.parse_args()
    if a.command=='status':
        run=a.run
        if run is None:
            ptr=a.private_root/'LATEST_RESEARCH_R2.txt'
            if not ptr.exists(): print('{"status":"not_started"}'); return 0
            run=Path(ptr.read_text().strip())
        print(json.dumps(read_json(run/'status.json') if (run/'status.json').exists() else {'status':'starting'},indent=2))
        return 0
    if a.command=='infer':
        if not a.bundle or not a.input or not a.out: p.error('infer requires --bundle --input --out')
        if a.out.exists(): raise ValueError('Choose a new request output directory')
        m=read_json(a.bundle/'model_manifest.json'); files=tree_hash(a.bundle)
        if any(files.get(k)!=v for k,v in m['files'].items()): raise ValueError('Exported model files changed')
        x=numeric_npz(a.input,('x',))['x']; schema=read_json(a.bundle/'schema.json')
        if x.shape[1]!=len(schema['features']): raise ValueError('Input features do not match exported schema')
        runtime=dict(m['runtime'])
        if os.environ.get('VDC_R2_RUNTIME_ROOT'): runtime['runtime_root']=os.environ['VDC_R2_RUNTIME_ROOT']
        if resolve_runtime(runtime)!=runtime['image_id']: raise ValueError('Pinned runtime unavailable')
        if (a.bundle/'wheels').exists(): runtime['wheels']=str(a.bundle/'wheels')
        execute(a.bundle/'code',{'query.npz':a.input,'schema.json':a.bundle/'schema.json'},a.out,
            ['python','/code/predict.py'],runtime,model=a.bundle/'model',repo=a.bundle/'runtime')
        pred=numeric_npz(a.out/'predictions.npz',('prediction',))['prediction']
        if pred.shape!=(len(x),len(schema['targets'])): raise ValueError('Output shape differs from schema')
        import csv,html
        with (a.out/'predictions.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.writer(f);w.writerow(['request_row',*schema['targets']]);w.writerows([i,*r] for i,r in enumerate(pred))
        summary={'status':'executed_new_request','rows':len(x),'targets':len(schema['targets']),
            'expert':m['expert'],'input_scale':m['task_contract']['allowed_inputs'],
            'independent_confirmation':False,'historical_validation_reused_as_request':False}
        write_json(a.out/'result.json',summary)
        (a.out/'report.html').write_text('<!doctype html><meta charset="utf-8"><h1>VDC R2 request</h1><pre>'+html.escape(json.dumps(summary,indent=2))+'</pre>',encoding='utf-8')
        print(json.dumps(summary));return 0
    config=read_json(a.config)
    if config['runtime'].get('backend')=='linux_landlock_chroot':
        from vdc.research_r2_devices import configure_devices
        config['runtime']['runtime_root']=os.environ.get('VDC_R2_RUNTIME_ROOT',str(a.private_root/'r2_runtime_devices_v2'))
        config['runtime']=configure_devices(config['runtime'],os.environ.get('VDC_R2_DEVICE_POLICY'))
    image=resolve_runtime(config['runtime'])
    if a.command=='doctor':
        inv=inventory(ROOT,a.private_root,a.asset_report,a.extra_data_root)
        print(json.dumps({'status':'structural_checks_passed','image_id':image,
            'device_policy':config['runtime'].get('device_policy'),
            'available_devices':config['runtime'].get('available_devices',['cpu']),
            'gpu_hardware':config['runtime'].get('gpu_hardware'),
            'private_assets':sum(not v['public'] for v in inv['assets'].values()),
            'public_assets_present':sum(v['public'] for v in inv['assets'].values()),
            'missing_sources':[k for k,v in inv['sources'].items() if not v['asset_ids']],
            'public_checksums_verified_here':False,'all_original_roles_open':True,
            'api_calls_executed':0,'runtime_probe':'run scripts/smoke_research_r2.py for actual isolation'},indent=2))
        return 0
    if a.run is None: p.error('run requires explicit --run')
    check=os.environ.get('VDC_RESEARCH_API_CHECK')
    if not check: raise ValueError('Set VDC_RESEARCH_API_CHECK to a passed API receipt')
    verify_connection_check(check,config)
    a.run.mkdir(parents=True,exist_ok=True)
    with exclusive_run(a.run):
        engine=Engine(ROOT,a.private_root,a.run,config,a.asset_report,a.extra_data_root)
        result=engine.run_loop(a.retry_api_errors)
    print(json.dumps({'status':result['status'],'report':str(a.run/'REPORT_CN.md')}))
    return 0 if result['status']=='completed_declared_scope' else 2


if __name__=='__main__':
    raise SystemExit(main())

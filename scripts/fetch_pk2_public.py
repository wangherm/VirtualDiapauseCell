"""Acquire pinned public files and sample observations, without training admission."""
from pathlib import Path
import argparse,concurrent.futures,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_public_pilot import fetch_locked
from run_all_modules import exclusive_run,inventory,verify_inventory,utc,source_fingerprint
from vdc.io import read_json,write_json,object_hash
from vdc.pk2_acquire import acquire_study

def main(a):
    config=read_json(ROOT/'configs/pk2_public_sources.json');studies=a.studies or list(config['studies'])
    if not set(studies).issubset(config['studies']):raise ValueError('Unknown accession; do not silently replace studies')
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    signature=object_hash({'config':config,'studies':studies,'code':source_fingerprint()})
    a.raw.mkdir(parents=True,exist_ok=True)
    with exclusive_run(out),exclusive_run(a.raw):
        if (out/'acquisition_manifest.json').exists():
            if read_json(out/'acquisition_manifest.json')['signature']!=signature:raise ValueError('Code/source selection changed; choose a new output directory')
        else:write_json(out/'acquisition_manifest.json',{'signature':signature,'started':utc(),'studies':studies,'training_executed':False})
        statuses=read_json(out/'acquisition_status.json')['studies'] if (out/'acquisition_status.json').exists() else {}
        def save():write_json(out/'acquisition_status.json',{'studies':statuses,'updated':utc(),'training_executed':False,'training_admitted':False})
        def job(acc):
            print('START '+acc,flush=True)
            result=acquire_study(acc,config['studies'][acc],a.raw,out/acc,a.download,fetch_locked)
            return {'execution_status':'completed','result':result,'files':inventory(out/acc)}
        todo=[]
        for acc in studies:
            old=statuses.get(acc,{})
            if old.get('execution_status')=='completed':
                if not verify_inventory(out/acc,old['files']):raise ValueError('Completed acquisition changed: '+acc)
                print('VERIFIED RESUME '+acc,flush=True)
            else:todo.append(acc);statuses[acc]={'execution_status':'queued'}
        save()
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures={pool.submit(job,acc):acc for acc in todo}
            for f in concurrent.futures.as_completed(futures):
                acc=futures[f]
                try:statuses[acc]=f.result();print('COMPLETED '+acc,flush=True)
                except Exception as exc:statuses[acc]={'execution_status':'failed','reason':str(exc)};print('FAILED '+acc+': '+str(exc),flush=True)
                save()
        summary={'acquisition_completed':sum(s['execution_status']=='completed' for s in statuses.values()),
            'requested':len(studies),'training_executed':False,'report':str(out/'acquisition_status.json')}
        write_json(out/'summary.json',summary);print(json.dumps(summary,indent=2))
        return 0 if summary['acquisition_completed']==len(studies) else 2

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--raw',type=Path,default=ROOT/'data/raw/pk2_public');p.add_argument('--studies',nargs='+')
    p.add_argument('--workers',type=int,choices=[1,2,3,4],default=2);p.add_argument('--download',action='store_true')
    sys.exit(main(p.parse_args()))

"""Private INT1 report; exclude raw input counts and large model/adapter copies."""
from pathlib import Path
import argparse,datetime,zipfile
from vdc.io import sha256,write_json

def pack(run):
    run=Path(run).resolve();stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out=run.parent/('VDC_INT1_PRIVATE_report_'+run.name+'_'+stamp+'.zip');files=[]
    with zipfile.ZipFile(out,'x',zipfile.ZIP_DEFLATED) as z:
        for p in run.rglob('*'):
            if not p.is_file():continue
            rel=p.relative_to(run)
            if any(k in rel.parts for k in ('work','inputs','adapter','model','portable_model','http_0','http_1')):continue
            if p.suffix not in {'.json','.jsonl','.md','.csv','.html','.log','.txt','.png','.npz'}:continue
            z.write(p,rel.as_posix());files.append({'file':rel.as_posix(),'sha256':sha256(p),'bytes':p.stat().st_size})
        import json
        z.writestr('PRIVATE_PACKAGE_MANIFEST.json',json.dumps({'private':True,'explicit_run':str(run),'raw_counts_included':False,'large_model_weights_included':False,'files':files}))
    with zipfile.ZipFile(out) as z:
        if z.testzip():raise ValueError('ZIP verification failed')
    print(out);return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();pack(a.run)

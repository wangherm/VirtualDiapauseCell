"""Explicit-run private report; omit input matrices, fitted weights and secrets."""
import argparse,datetime,json,zipfile
from pathlib import Path
from vdc.io import read_json,sha256

def pack(run):
    run=Path(run).resolve()
    if not (run/'manifest.json').is_file():raise ValueError('No research manifest in explicit run')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out=run.parent/f'VDC_RESEARCH_PRIVATE_report_{run.name}_{stamp}.zip';files=[]
    with zipfile.ZipFile(out,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(run.rglob('*')):
            if p.is_symlink():raise ValueError('Symlink not allowed in report')
            if not p.is_file():continue
            rel=p.relative_to(run)
            if any(k in rel.parts for k in ('data','release','sandbox','model')):continue
            if p.suffix not in {'.json','.jsonl','.md','.log','.txt','.npz'}:continue
            z.write(p,rel.as_posix());files.append({'file':rel.as_posix(),'sha256':sha256(p),'bytes':p.stat().st_size})
        status=read_json(run/'status.json') if (run/'status.json').exists() else {'status':'unknown'}
        z.writestr('PRIVATE_PACKAGE_MANIFEST.json',json.dumps({'run':str(run),'status':status['status'],
            'operation':'export_existing_results_not_new_experiments','contains_private_predictions':True,
            'raw_inputs_included':False,'files':files},indent=2))
    with zipfile.ZipFile(out) as z:
        if z.testzip():raise ValueError('ZIP verification failed')
    print(out);return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);pack(p.parse_args().run)

"""Explicit R2 archive includes fitted models and transforms; raw inputs stay local."""
import argparse
import datetime
from pathlib import Path
import json
import zipfile
from vdc.io import read_json, sha256


def pack(run):
    run=Path(run).resolve()
    if not (run/'manifest.json').exists(): raise ValueError('No R2 manifest in explicit run')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out=run.parent/f'VDC_RESEARCH_R2_PRIVATE_{run.name}_{stamp}.zip'
    records=[]
    with zipfile.ZipFile(out,'x',zipfile.ZIP_DEFLATED,allowZip64=True) as z:
        for p in sorted(run.rglob('*')):
            if p.is_symlink(): raise ValueError('Symlink in report tree')
            if not p.is_file(): continue
            rel=p.relative_to(run); parts=rel.parts
            if any(n in parts for n in ('release','__pycache__','refit_inputs')): continue
            if parts[0]=='wheels' or (parts[0]=='actions' and 'wheels' in parts):continue
            if parts[0]=='actions' and 'output' in parts: continue  # arbitrary data-preparation outputs
            if parts[0]=='tasks' and p.name in ('train.npz','query.npz','truth.npz'): continue
            if p.name in ('.env','inventory.json'): continue  # local raw source paths stay on server
            if p.suffix in ('.partial',): continue
            z.write(p,rel.as_posix())
            records.append({'file':rel.as_posix(),'bytes':p.stat().st_size,'sha256':sha256(p)})
        status=read_json(run/'status.json') if (run/'status.json').exists() else {'status':'unknown'}
        z.writestr('PRIVATE_PACKAGE_MANIFEST.json',json.dumps({'operation':'export_existing_results',
            'run':run.name,'status':status['status'],'contains_private_predictions_and_weights':True,
            'raw_inputs_included':False,'export_dependency_wheels_included':True,'files':records},indent=2))
    with zipfile.ZipFile(out) as z:
        if z.testzip(): raise ValueError('Archive integrity failed')
    print(out); return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path)
    pack(p.parse_args().run)

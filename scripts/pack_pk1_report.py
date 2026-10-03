"""Create a PRIVATE report package; never send private run artifacts to GitHub."""
from pathlib import Path
import argparse
import datetime
import json
import zipfile
from vdc.io import sha256

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',nargs='?');p.add_argument('--out',type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    run=Path(a.run or (root/'runs/LATEST_PK1.txt').read_text().strip()).resolve()
    if not (run/'module_status.json').exists():raise SystemExit('No PK1 task report exists yet')
    out=a.out or run.parent/f'VDC_PK1_PRIVATE_report_{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%SZ}.zip'
    out=out.resolve()
    if out.is_relative_to(run):raise SystemExit('Report ZIP must be outside the run directory')
    included=[];omitted=[]
    with zipfile.ZipFile(out,'x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(run.rglob('*')):
            if not f.is_file():continue
            rel=f.relative_to(run).as_posix()
            if 'failed_attempts/' in rel or f.suffix in {'.pt','.safetensors'} or f.name in {'.runner.lock'} or rel.startswith(('public/data/','public/clock_reference/')):
                omitted.append(rel);continue
            z.write(f,'pk1_report/'+rel);included.append({'file':rel,'sha256':sha256(f),'bytes':f.stat().st_size})
        z.writestr('PRIVATE_PACKAGE_MANIFEST.json',json.dumps({'private':True,'contains_internal_predictions_and_sample_ids':True,
            'weights_included':False,'files':included,'omitted':omitted},indent=2))
    print(out)

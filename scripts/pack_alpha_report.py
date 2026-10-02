"""Archive actual artifacts, explicitly list omitted large model files; never invent missing outputs."""
import argparse,datetime,json,zipfile,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run',nargs='?');p.add_argument('--weights',action='store_true');p.add_argument('--out',type=Path);a=p.parse_args()
run=Path(a.run or Path('runs/LATEST_ALPHA.txt').read_text().strip()).resolve()
if not (run/'module_status.json').exists():raise SystemExit('No module status; this is not a completed runner directory')
out=a.out or Path(f'VirtualDiapauseCell_Alpha_report_{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%SZ}.zip')
if out.exists():raise SystemExit('Refusing to overwrite an existing archive')
entries=[];omitted=[]
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for f in sorted(run.rglob('*')):
  if not f.is_file() or f.name=='.runner.lock' or f.suffix=='.partial':continue
  relative=f.relative_to(run).as_posix()
  if 'failed_attempts/' in relative or (not a.weights and (f.suffix in {'.pt','.safetensors'} or '/checkpoints/' in relative)):
   omitted.append(relative);continue
  data=f.read_bytes();z.writestr(run.name+'/'+relative,data)
  entries.append({'path':relative,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
 z.writestr('PACKAGE_MANIFEST.json',json.dumps({'run':run.name,'weights_included':a.weights,'files':entries,'omitted':omitted},indent=2))
print(out.resolve());print('Bytes',out.stat().st_size,'Files',len(entries),'Omitted',len(omitted))

"""Private CW1 report: evaluations and fitted parameters, never source counts or LLM weights."""
from pathlib import Path
import argparse,datetime,json,zipfile,hashlib


def pack(run,out=None):
    run=Path(run).resolve();stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=Path(out) if out else run.parent/('VDC_CLOCK_WAVE_PRIVATE_report_'+stamp+'.zip')
    if out.resolve().is_relative_to(run):raise ValueError('Package must be outside run')
    files=[]
    with zipfile.ZipFile(out,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(run.rglob('*')):
            if not p.is_file() or p.is_symlink():continue
            rel=p.relative_to(run)
            if rel.parts[0]=='data' or p.suffix not in {'.json','.jsonl','.npz','.md','.log','.txt'}:continue
            if p.name in {'config.json','counts.npz'}:continue
            payload=p.read_bytes();z.writestr(rel.as_posix(),payload)
            files.append({'file':rel.as_posix(),'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()})
        z.writestr('PRIVATE_PACKAGE_MANIFEST.json',json.dumps({'files':files,'private':True,'not_for_public_repository':True,
            'raw_counts_included':False,'llm_weights_included':False,'numeric_parameters_included':True,
            'note':'Contains development predictions and private metadata; keep local.'},indent=2))
    with zipfile.ZipFile(out) as z:
        if z.testzip():raise RuntimeError('Package CRC failed')
        for f in files:
            if hashlib.sha256(z.read(f['file'])).hexdigest()!=f['sha256']:raise RuntimeError('Package manifest mismatch')
    print(out);return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path);a=p.parse_args();pack(a.run,a.out)

"""Pack private query results and protocol evidence, never weights or source counts."""
from pathlib import Path
import argparse,datetime,hashlib,json,zipfile,io
import numpy as np

def pack(run,out=None,full_gene_arrays=False):
    run=Path(run).resolve();stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=Path(out) if out else run.parent/('VDC_APPLICATION_PRIVATE_report_'+stamp+'.zip')
    if out.resolve().is_relative_to(run):raise ValueError('Output must be outside run')
    files=[]
    with zipfile.ZipFile(out,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(run.rglob('*')):
            if not p.is_file() or p.is_symlink():continue
            rel=p.relative_to(run)
            if rel.parts[0]=='snapshot':
                if 'hidden' not in rel.parts or p.suffix not in {'.json','.npz'}:continue
            elif rel.parts[0] not in {'applications','knowledge','logs','task_progress','failures'} and len(rel.parts)>1:continue
            if p.name=='config.json' or p.suffix in {'.pt','.safetensors','.h5ad'}:continue
            data=p.read_bytes();original_hash=hashlib.sha256(data).hexdigest();omitted=[]
            if p.suffix=='.npz' and rel.parts[0]=='applications' and not full_gene_arrays:
                with np.load(io.BytesIO(data),allow_pickle=False) as a:
                    omitted=[k for k in a.files if k.startswith('gene_')]
                    if omitted:
                        buffer=io.BytesIO();np.savez_compressed(buffer,**{k:a[k] for k in a.files if k not in omitted});data=buffer.getvalue()
            z.writestr(rel.as_posix(),data)
            files.append({'file':rel.as_posix(),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
                'original_sha256':original_hash,'omitted_arrays':omitted})
        z.writestr('PRIVATE_PACKAGE_MANIFEST.json',json.dumps({'files':files,'private':True,'contains_reserved_query_results':True,
            'weights_included':False,'raw_counts_included':False,'full_gene_arrays':full_gene_arrays,
            'compact_npz_notice':'Original full arrays stay on server. Package manifest records transformed file hashes and omitted keys; original query inventories describe original files.',
            'not_for_public_repository':True},indent=2))
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None
        for f in files:assert hashlib.sha256(z.read(f['file'])).hexdigest()==f['sha256']
    print(out);return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path);p.add_argument('--full-gene-arrays',action='store_true');a=p.parse_args();pack(a.run,a.out,a.full_gene_arrays)

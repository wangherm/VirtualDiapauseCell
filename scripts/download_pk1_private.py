"""Fetch only explicitly named private archives. Never print Drive identifiers."""
from pathlib import Path
import argparse
import contextlib
import io
import json
import sys
from vdc.io import read_json, sha256, write_json


def resolve_files(config, records):
    result=[]
    for wanted in config['files']:
        name=wanted['name']
        if Path(name).name!=name or name in {'.','..'} or '/' in name or '\\' in name:
            raise ValueError('Expected a plain archive basename')
        if len(wanted['sha256'])!=64 or wanted['bytes']<=0:raise ValueError('Expected pinned bytes and SHA256')
        found=[r for r in records if Path(r.path).name==name]
        if len(found)!=1:raise ValueError('Required archive is missing or ambiguous')
        result.append((wanted,found[0]))
    if not result:raise ValueError('No approved downloads')
    return result


def download(config_path,out,list_only=False):
    import gdown
    config=read_json(config_path);out=Path(out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    # gdown exceptions can contain private URLs; neither those nor its console text
    # are allowed into shareable launcher logs.
    try:
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            records=gdown.download_folder(url=config['folder_url'],output=str(out),
                quiet=True,use_cookies=False,skip_download=True)
        if records is None:raise RuntimeError('listing failed')
        selected=resolve_files(config,records)
    except Exception:
        raise RuntimeError('Private folder listing failed or required archives are absent; check access locally') from None
    statuses={}
    for wanted,remote in selected:
        name=wanted['name'];target=out/name
        print(('FOUND ' if list_only else 'VERIFY/DOWNLOAD ')+name,flush=True)
        if list_only:
            statuses[name]={'status':'listed_not_downloaded'};continue
        if target.exists():
            if target.stat().st_size!=wanted['bytes'] or sha256(target)!=wanted['sha256']:
                raise ValueError('Existing archive differs from local source lock; preserve it and use a new directory')
        else:
            try:
                with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                    result=gdown.download(id=remote.id,output=str(target),quiet=True,use_cookies=False,resume=True)
                if result is None:raise RuntimeError('download failed')
            except Exception:
                raise RuntimeError('Private archive download interrupted; rerun to resume or upload the archive manually') from None
            if target.stat().st_size!=wanted['bytes'] or sha256(target)!=wanted['sha256']:
                raise ValueError('Downloaded bytes differ from the locally audited archive; do not import')
        statuses[name]={'status':'verified','bytes':wanted['bytes'],'sha256':wanted['sha256']}
        write_json(out/'download_status.json',statuses)
        print('VERIFIED '+name,flush=True)
    if list_only:write_json(out/'listing_status.json',statuses)
    return statuses


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--list-only',action='store_true');a=p.parse_args()
    try:download(a.config,a.out,a.list_only)
    except Exception as exc:
        # All external library exceptions have already been replaced above.
        print('FAILED: '+str(exc),file=sys.stderr);sys.exit(1)

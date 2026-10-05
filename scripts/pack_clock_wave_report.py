"""Private CW1 report: evaluations and fitted parameters, never source counts or LLM weights."""
from pathlib import Path
import argparse,datetime,json,zipfile,hashlib
from vdc.io import read_json,sha256,object_hash


def pack(run,out=None,require_revision=False):
    from audit_clock_wave_runs import inspect_run
    run=Path(run).resolve();stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    audit=inspect_run(run)
    if require_revision and not audit['revision_config_and_plan']:
        raise ValueError('Requested targeted revision, but explicit --run is not a revision; no package created')
    out=Path(out) if out else run.parent/('VDC_CLOCK_WAVE_PRIVATE_report_'+run.name+'_'+stamp+'.zip')
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
        payload_hash=object_hash({v['file']:v['sha256'] for v in files});matches=[]
        for old in run.parent.glob('*CLOCK_WAVE*report*.zip'):
            if old.resolve()==out.resolve():continue
            try:
                with zipfile.ZipFile(old) as previous:
                    manifest=json.loads(previous.read('PRIVATE_PACKAGE_MANIFEST.json'))
                    prior={v['file']:v['sha256'] for v in manifest['files'] if v['file']!='PACKAGING_PROVENANCE.json'}
                    if object_hash(prior)==payload_hash:matches.append(str(old))
            except (OSError,ValueError,KeyError,zipfile.BadZipFile):continue
        parent=audit.get('revision_parent') or {};delta={'status':'no_revision_parent'}
        if parent.get('path'):
            old=Path(parent['path']);old_queue=read_json(old/'queue_status.json') if (old/'queue_status.json').exists() else {}
            new_queue=read_json(run/'queue_status.json');before=old_queue.get('tasks',{});after=new_queue.get('tasks',{})
            delta={'status':'compared_task_inventories' if before else 'parent_inventory_unavailable',
                   'added_task_ids':sorted(after.keys()-before.keys()),
                   'changed_task_inventories':sorted(n for n in after.keys()&before.keys() if after[n].get('files')!=before[n].get('files')),
                   'identical_task_inventories':sorted(n for n in after.keys()&before.keys() if after[n].get('files')==before[n].get('files'))}
        pointers={}
        for name in ('LATEST_CLOCK_WAVE.txt','LATEST_CLOCK_WAVE_RELEASE.txt','LATEST_CLOCK_WAVE_REVISION.txt'):
            path=run.parent.parent/name;pointers[name]=path.read_text().strip() if path.exists() else None
        provenance={'packaged_at_utc':stamp,'explicit_run':str(run),'packager_sha256':sha256(Path(__file__)),
                    'payload_fingerprint':payload_hash,'identical_previous_packages':matches,
                    'operation':'repack_only' if matches else 'export_existing_targeted_results' if audit['fresh_tasks_verified'] else 'export_existing_results_no_targeted_execution_evidence',
                    'packaging_executed_training':False,'pointers_at_pack_time':pointers,
                    'latest_matches_explicit_run':pointers['LATEST_CLOCK_WAVE.txt']==str(run),
                    'run_audit':audit,'delta_from_parent':delta}
        payload=json.dumps(provenance,ensure_ascii=False,indent=2).encode('utf-8')
        z.writestr('PACKAGING_PROVENANCE.json',payload)
        files.append({'file':'PACKAGING_PROVENANCE.json','bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()})
        z.writestr('PRIVATE_PACKAGE_MANIFEST.json',json.dumps({'files':files,'private':True,'not_for_public_repository':True,
            'raw_counts_included':False,'llm_weights_included':False,'numeric_parameters_included':True,
            'note':'Contains development predictions and private metadata; keep local.'},indent=2))
    with zipfile.ZipFile(out) as z:
        if z.testzip():raise RuntimeError('Package CRC failed')
        for f in files:
            if hashlib.sha256(z.read(f['file'])).hexdigest()!=f['sha256']:raise RuntimeError('Package manifest mismatch')
    print(json.dumps({'package':str(out),'run':str(run),'operation':provenance['operation'],
                     'task_categories':{k:len(v) for k,v in audit['categories'].items()},
                     'targeted_complete':audit['targeted_complete']},ensure_ascii=False));return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path);p.add_argument('--require-revision',action='store_true');a=p.parse_args();pack(a.run,a.out,a.require_revision)

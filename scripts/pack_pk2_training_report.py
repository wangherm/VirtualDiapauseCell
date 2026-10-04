"""Private point-in-time diagnostics, never weights/raw expression/private download URLs."""
from pathlib import Path
import argparse,datetime,hashlib,json,zipfile

def pack(run,out=None):
    run=Path(run).resolve();stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=Path(out).resolve() if out else run.parent/f'VDC_PK2_TRAINING_PRIVATE_report_{stamp}.zip'
    if out.is_relative_to(run):raise ValueError('Archive must be outside run')
    if not (run/'queue_status.json').exists():raise ValueError('Not an initialized full training queue')
    files=[];omitted=[]
    root_names={'REPORT_CN.md','queue_status.json','run_manifest.json','hardware.json','resources.json','resource_history.jsonl','console.log','exit_code.txt','plan.json','repair_import.json'}
    report_names={'result.json','failure.json','phase.json','metrics.json','run.json','reload.json','profile.json','manifest.json','selection.json',
        'evaluation.json','evaluation_scope.json','reuse.json','verification.json','training_log.json','status.json','source_set.json','expression_contract.json','input.json','subset.json',
        'mask_metrics.json','feature_metrics.json','contrasts.json','rows.json','answers.json','steps.jsonl','audit.json','service_manifest.json',
        'predictions.npz','mask_predictions.npz','validation.npz','double_mutant.npz','simulated_deviation.npz',
        'semantic_input.json','embeddings.json','embeddings.npz','corpus.jsonl','predictions.json','process_0.log','process_1.log','service.log','COMPARISON_CN.md'}
    out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,'x',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(run.rglob('*')):
            if not p.is_file() or p.is_symlink() or not p.resolve().is_relative_to(run):continue
            rel=p.relative_to(run)
            if 'failed_attempts' in rel.parts or 'snapshot' in rel.parts or 'bundle' in rel.parts:continue
            if not (len(rel.parts)==1 and p.name in root_names or rel.parts[0]=='logs' and p.suffix=='.log' or rel.parts[0]=='tasks' and p.name in report_names):continue
            size=p.stat().st_size;limit=3*1024**2 if p.suffix=='.log' else 25*1024**2
            if size>limit and p.suffix!='.log':omitted.append({'file':rel.as_posix(),'bytes':size});continue
            with p.open('rb') as f:
                f.seek(max(0,size-limit));data=f.read(limit)
            archive.writestr(rel.as_posix(),data);files.append({'file':rel.as_posix(),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'tail_only':size>limit})
        archive.writestr('PRIVATE_PACKAGE_MANIFEST.json',json.dumps({'private':True,'files':files,'omitted':omitted,
            'contains_development_sample_ids_and_predictions':True,'weights_included':False,'raw_expression_included':False,
            'live_files_are_point_in_time_snapshots':True,'training_completion_claim':False},indent=2))
    with zipfile.ZipFile(out) as archive:
        if archive.testzip():raise RuntimeError('Archive integrity failure')
        for item in files:
            if hashlib.sha256(archive.read(item['file'])).hexdigest()!=item['sha256']:raise RuntimeError('Archive hash failure')
    print(json.dumps({'archive':str(out),'bytes':out.stat().st_size,'files':len(files),'verified':True}));return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
    run=a.run or Path((Path(__file__).resolve().parents[1]/'runs/LATEST_PK2_TRAINING.txt').read_text().strip());pack(run,a.out)

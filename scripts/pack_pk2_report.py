"""Standard-library-only PRIVATE report packer; does not stop or rerun services."""
from pathlib import Path
import argparse,datetime,hashlib,json,urllib.request,zipfile


def pack(repo,out=None,acquire=None,service=None):
    repo=Path(repo).resolve();now=datetime.datetime.now(datetime.timezone.utc)
    sources={};missing=[];errors=[];files=[];omitted=[];summaries={}
    for name,override in [('acquire',acquire),('service',service)]:
        pointer=repo/'runs'/f'LATEST_PK2_{name.upper()}.txt'
        try:
            value=str(override) if override else pointer.read_text(encoding='utf-8').strip()
            if not value:raise ValueError('Empty run pointer')
            p=Path(value);p=(repo/p).resolve() if not p.is_absolute() else p.resolve()
            if not p.is_dir():raise FileNotFoundError(str(p))
            sources[name]=p
        except (OSError,ValueError) as exc:missing.append({'branch':name,'reason':str(exc)})
    if not sources:raise ValueError('No PK2 run directories found; supply --repo or explicit --acquire/--service')
    out=Path(out).resolve() if out else repo.parent/'vdc-private/reports'/f'VDC_PK2_PRIVATE_report_{now:%Y%m%dT%H%M%S_%fZ}.zip'
    if any(out.is_relative_to(p) for p in sources.values()):raise ValueError('ZIP must be outside source run directories')
    out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,'x',zipfile.ZIP_DEFLATED) as z:
        def add(root,relative,label):
            p=root/relative
            if not p.exists():return
            if not p.is_file() or p.is_symlink() or not p.resolve().is_relative_to(root):
                omitted.append({'file':label+'/'+relative,'reason':'not a regular contained file'});return
            limit=5*1024*1024 if p.suffix=='.log' else 20*1024*1024
            try:
                with p.open('rb') as f:
                    f.seek(0,2);size=f.tell()
                    if size>limit and p.suffix!='.log':
                        omitted.append({'file':label+'/'+relative,'reason':'over 20 MiB report-file limit','bytes':size});return
                    f.seek(max(0,size-limit));data=f.read(limit)
                name=label+'/'+relative;z.writestr(name,data)
                files.append({'file':name,'bytes':len(data),'source_bytes_at_read':size,
                    'tail_only':size>limit,'sha256':hashlib.sha256(data).hexdigest()})
            except OSError as exc:errors.append({'file':label+'/'+relative,'reason':str(exc)})
        def read(p):
            try:return json.loads(p.read_text(encoding='utf-8'))
            except (OSError,ValueError) as exc:return {'read_error':str(exc)}
        for label,root in sources.items():
            for name in ['console.log','restart.log','exit_code.txt','finished_utc.txt','port.txt']:
                add(root,name,label)
            exit_file=root/'exit_code.txt'
            summaries[label]={'source_directory':str(root),'launcher_exit_code':exit_file.read_text().strip() if exit_file.is_file() else 'not_recorded'}
            if label=='acquire':
                for name in ['acquisition_manifest.json','acquisition_status.json','summary.json']:add(root,name,label)
                summaries[label]['summary']=read(root/'summary.json')
                summaries[label]['study_status']={k:{key:v.get(key) for key in ['execution_status','reason','result']}
                    for k,v in read(root/'acquisition_status.json').get('studies',{}).items()}
                for folder in sorted(root.glob('GSE*')):
                    if not folder.is_dir() or folder.is_symlink():continue
                    for name in ['status.json','samples.jsonl','dauer_mtc_samples.jsonl','differential_tables.json']:
                        add(root,folder.name+'/'+name,label)
            else:
                for name in ['verification.json','predictions.json','process_0.log','process_1.log']:
                    add(root,'http_verification/'+name,label)
                add(root,'snapshot/service_manifest.json',label)
                m=read(root/'snapshot/service_manifest.json')
                summaries[label]['snapshot_id']=m.get('snapshot_id')
                summaries[label]['registered_models']=len(m.get('models',{}))
                summaries[label]['reload_verification']=read(root/'http_verification/verification.json')
                for folder in sorted((root/'snapshot').glob('*')):
                    if not folder.is_dir() or folder.is_symlink():continue
                    for name in ['run.json','result.json','metrics.json','reference.json','response_model.json']:
                        add(root,'snapshot/'+folder.name+'/'+name,label)
                try:
                    port=int((root/'port.txt').read_text().strip())
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=3) as response:health=json.load(response)
                    summaries[label]['live_health']={'response':health,'snapshot_matches':health.get('snapshot_id')==m.get('snapshot_id')}
                except Exception as exc:summaries[label]['live_health']={'status':'not_reachable','reason':str(exc)}
        manifest={'private':True,'created_utc':now.isoformat(),'contains_development_predictions_or_sample_ids':True,
            'weights_included':False,'expression_arrays_included':False,'private_sample_role_manifest_included':False,
            'not_a_training_completion_claim':True,'live_logs_are_point_in_time_snapshots':True,
            'sources':summaries,'missing_branches':missing,'read_errors':errors,'omitted':omitted,'files':files}
        z.writestr('PRIVATE_PACKAGE_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        text=['# PK2 抓取与开发服务报告','',
            '这是私有诊断包，可能含开发样本标识和预测。请勿提交公开仓库。',
            '不包含原始表达矩阵、模型权重、完整样本角色表或虚拟环境。',
            '打包不会启动训练、重跑查询或停止服务。服务常驻不等于任务未完成；下载完成也不等于 PK2 扩训完成。','',
            '```json',json.dumps({'sources':summaries,'missing_branches':missing,'read_errors':errors},ensure_ascii=False,indent=2),'```']
        z.writestr('REPORT_CN.md','\n'.join(text))
    with zipfile.ZipFile(out) as z:
        if z.testzip() is not None:raise RuntimeError('ZIP integrity check failed')
        for entry in files:
            if hashlib.sha256(z.read(entry['file'])).hexdigest()!=entry['sha256']:raise RuntimeError('Package hash check failed')
    print(json.dumps({'package':str(out),'bytes':out.stat().st_size,'files':len(files),
        'missing_branches':missing,'read_errors':errors,'package_verified':True},ensure_ascii=False,indent=2))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--out',type=Path)
    p.add_argument('--acquire',type=Path);p.add_argument('--service',type=Path)
    a=p.parse_args();pack(a.repo,a.out,a.acquire,a.service)

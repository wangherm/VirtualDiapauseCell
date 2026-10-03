"""Show acquisition progress and verify the live service snapshot identity."""
from pathlib import Path
import json,urllib.request
ROOT=Path(__file__).resolve().parents[1]
for kind in ['ACQUIRE','SERVICE']:
    pointer=ROOT/'runs'/f'LATEST_PK2_{kind}.txt'
    if not pointer.exists():print(kind,'not_started');continue
    run=Path(pointer.read_text().strip());print('\n'+kind,str(run))
    status=run/'acquisition_status.json'
    if status.exists():
        for acc,s in json.loads(status.read_text(encoding='utf-8'))['studies'].items():
            print(acc,s['execution_status'],s.get('reason',''),s.get('result',{}).get('processing_status',''))
    if kind=='SERVICE':
        try:
            port=int((run/'port.txt').read_text());expected=json.loads((run/'snapshot/service_manifest.json').read_text(encoding='utf-8'))['snapshot_id']
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=3) as response:health=json.load(response)
            print('HTTP',health['status'],'snapshot_matches',health['snapshot_id']==expected,'port',port)
        except Exception as exc:print('HTTP not ready:',str(exc))
    exit_file=run/'exit_code.txt'
    if exit_file.exists():print('PROCESS_EXIT',exit_file.read_text().strip())
    print('LOG',run/'console.log')

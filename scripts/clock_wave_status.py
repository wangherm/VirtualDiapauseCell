"""Show terminal coverage and active CW1 jobs; blocked does not mean trained."""
from pathlib import Path
import argparse,os
from vdc.io import read_json
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);a=p.parse_args()
    run=a.run or Path((Path(os.environ.get('VDC_PRIVATE_ROOT','/root/autodl-tmp/vdc-private'))/'LATEST_CLOCK_WAVE.txt').read_text().strip())
    f=run/'queue_status.json'
    if not f.exists():print('Not initialized yet; check',run/'console.log');raise SystemExit(1)
    q=read_json(f);print('RUN',run);print(q['status'],q['counts'],'total',q['total'])
    if 'reused_verified' in q:print('Verified reuse:',q['reused_verified'],'; newly completed:',q['counts'].get('completed',0)-q['reused_verified'])
    for name,state in q['tasks'].items():
        if state['status']!='completed':print(name,state['status'],state.get('reason',''))
    print('Report:',run/'REPORT_CN.md')

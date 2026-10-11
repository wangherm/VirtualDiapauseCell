"""Print actual progress without loading any model or resubmitting jobs."""
import argparse,json,os
from pathlib import Path
from vdc.io import read_json

p=argparse.ArgumentParser();p.add_argument('--run',type=Path);a=p.parse_args()
run=a.run or Path((Path(os.environ.get('VDC_PRIVATE_ROOT','/root/autodl-tmp/vdc-private'))/'LATEST_RESEARCH.txt').read_text().strip())
status=read_json(run/'status.json') if (run/'status.json').exists() else {'status':'not_started'}
ledger=[read_json(f) for f in (run/'api').glob('*.json')]
print(json.dumps({'run':str(run),'status':status['status'],'completed_experiments':status.get('completed_experiments',0),
 'rounds_recorded':len(status.get('rounds',[])),'api_requests_submitted':len(ledger),
 'reported_tokens':sum((r.get('usage') or {}).get('total_tokens',0) for r in ledger),
 'budget_charge_upper_bound':sum(r['budget_charge'] for r in ledger),
 'exit_code':(run/'exit_code.txt').read_text().strip() if (run/'exit_code.txt').exists() else None,
 'error':read_json(run/'error.json') if (run/'error.json').exists() else None},ensure_ascii=False,indent=2))

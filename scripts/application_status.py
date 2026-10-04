"""Read progress without issuing any model query."""
from pathlib import Path
import argparse,json
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run
    result=json.loads((r/'application_status.json').read_text()) if (r/'application_status.json').exists() else {'status':'initializing'}
    result['progress']={p.stem:json.loads(p.read_text()) for p in (r/'task_progress').glob('*.json')}
    result['exit_code']=(r/'exit_code.txt').read_text().strip() if (r/'exit_code.txt').exists() else None
    print(json.dumps(result,ensure_ascii=False,indent=2))

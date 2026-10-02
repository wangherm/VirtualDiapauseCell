"""Read-only progress summary, including blocked and failed branches."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run',nargs='?');a=p.parse_args()
run=Path(a.run or Path('runs/LATEST_ALPHA.txt').read_text().strip())
print('RUN',run)
path=run/'module_status.json'
if not path.exists():print('Runner not reached yet; inspect console.log (dependency/GPU setup).')
else:
 m=json.loads(path.read_text(encoding='utf-8'));print('OVERALL',m['overall_status'])
 for name,s in m['modules'].items():print(f"{name:28} {s['execution_status']:20} {s.get('reason','')}")
 for f in run.glob('state_*/steps.jsonl'):
  lines=f.read_text().splitlines()
  if lines:print(f.parent.name,'LATEST_STEP',lines[-1])
 f=run/'logs/knowledge.log'
 if f.exists():print('QWEN_LOG',f,'\n','\n'.join(f.read_text(errors='replace').splitlines()[-6:]))

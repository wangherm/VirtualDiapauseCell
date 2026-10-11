"""Show actual public acquisition progress, including failures, without starting work."""
import argparse
import json
import os
from pathlib import Path
from vdc.io import read_json

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',type=Path)
p.add_argument('--private-root',type=Path,default=Path(os.environ.get('VDC_PRIVATE_ROOT','/root/autodl-tmp/vdc-private')))
a=p.parse_args()
pointer=a.private_root/'LATEST_RESEARCH_DATA.txt'
run=a.run or (Path(pointer.read_text().strip()) if pointer.is_file() else None)
if run is None:
    print(json.dumps({'status':'not_started'}));raise SystemExit(0)
state=read_json(run/'status.json') if (run/'status.json').is_file() else {'status':'starting'}
result={k:state.get(k) for k in ('status','verified','total','failed','updated')}
result.update(run=str(run),exit_code=(run/'exit_code.txt').read_text().strip() if (run/'exit_code.txt').exists() else None,
              failures={k:v for k,v in state.get('files',{}).items() if v['status']=='failed'})
print(json.dumps(result,ensure_ascii=False,indent=2))

"""Read real PK1 task states and the last numerical training step."""
from pathlib import Path
import argparse
import json

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',nargs='?');a=p.parse_args()
    pointer=Path(__file__).resolve().parents[1]/'runs/LATEST_PK1.txt'
    run=Path(a.run or pointer.read_text().strip())
    print('RUN',run)
    if not (run/'module_status.json').exists():print('SETUP: inspect console.log for dependencies, data or GPU checks')
    else:
        m=json.loads((run/'module_status.json').read_text(encoding='utf-8'))
        print('STATUS',m['overall_status'],'COMPLETED',m['completed'],'/',m['total'])
        for name,r in m['modules'].items():
            if r['execution_status']!='completed':print(name,r['execution_status'],r.get('reason',''))
        for name,r in m['modules'].items():
            if r['execution_status']=='running':
                steps=run/name/'steps.jsonl'
                if steps.exists():
                    lines=steps.read_text().splitlines()
                    if lines:print('LATEST_STEP',name,lines[-1])
    if (run/'exit_code.txt').exists():print('PROCESS_EXIT', (run/'exit_code.txt').read_text().strip())
    print('LOG',run/'console.log')

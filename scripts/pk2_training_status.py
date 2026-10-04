"""Read full queue progress without rerunning any evaluation."""
from pathlib import Path
import argparse,json

def status(run):
    run=Path(run);p=run/'queue_status.json'
    if not p.exists():return {'run':str(run),'status':'queue_not_initialized','inspect':str(run/'console.log')}
    result=json.loads(p.read_text(encoding='utf-8'));tasks=result.pop('tasks');active=[]
    for name,item in tasks.items():
        if item['status']!='running':continue
        row={'task':name,**item};phase=run/'tasks'/name/'phase.json'
        if phase.exists():row['phase']=json.loads(phase.read_text())
        steps=run/'tasks'/name/'model/steps.jsonl'
        if steps.exists():
            lines=steps.read_text().splitlines()
            if lines:
                try:row['latest_training_step']=json.loads(lines[-1])
                except ValueError:row['latest_training_step']='being written'
        active.append(row)
    result.update(run=str(run),active=active,issues={n:r.get('reason') for n,r in tasks.items() if r['status'].startswith(('blocked','failed'))})
    p=run/'resources.json'
    if p.exists():result['resources']=json.loads(p.read_text())
    p=run/'exit_code.txt';result['launcher_exit_code']=p.read_text().strip() if p.exists() else 'not_recorded'
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);a=p.parse_args()
    run=a.run or Path((Path(__file__).resolve().parents[1]/'runs/LATEST_PK2_TRAINING.txt').read_text().strip())
    print(json.dumps(status(run),ensure_ascii=False,indent=2))

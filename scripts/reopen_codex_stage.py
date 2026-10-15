"""Operator-only phase transition; execute with the run's ORIGINAL frozen engine."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone


def reopen(engine, expected, brief, snapshot_fn, apply=False):
    run=engine.run
    state=engine.state
    if state['next_decision'] != expected:
        raise ValueError('Next decision differs; do not rewind a run')
    if (run/'STOP').exists():
        raise ValueError('Operator STOP remains present')
    content=Path(brief).read_bytes()
    if not content.strip() or len(content)>100_000:
        raise ValueError('A bounded nonempty operator brief is required')
    digest=hashlib.sha256(content).hexdigest()
    folder=run/'stage_reviews'/f'before_{expected:06d}'
    receipt=folder/'transition.json'
    snap=json.loads((run/'controller_snapshot.json').read_text())
    if snap['next_decision'] != expected:
        raise ValueError('Snapshot and state do not match')
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if old['brief_sha256']!=digest or old['status']!='completed' or state['status']!='running':
            raise ValueError('Previous transition requires inspection')
        if snap['status']!='running':
            raise ValueError('Transition snapshot requires inspection')
        return {**old,'operation':'already_reopened_no_replay'}
    if state['status']!='completed_declared_scope' or snap['status']!='completed_declared_scope':
        raise ValueError('Only a reviewed completed scope can enter a new phase')
    queue=run/'mailbox'
    for p in queue.glob('*.request.json'):
        key=p.name.split('.')[0]
        if not (queue/(key+'.response.json')).exists():
            raise ValueError('Pending mailbox request')
    for p in queue.glob('*.running.json'):
        if not (queue/(p.name.split('.')[0]+'.response.json')).exists():
            raise ValueError('Uncertain worker result')
    if list(queue.glob(f'{expected:06d}.*')):
        raise ValueError('Next decision already has mailbox evidence')
    report={'operation':'operator_reviewed_next_phase','status':'validated_not_applied',
        'next_decision':expected,'brief_sha256':digest,'scientific_approval':False,
        'old_experiments_reexecuted':False,'resource_budgets_changed':False,
        'created_at':datetime.now(timezone.utc).isoformat()}
    if not apply:return report
    folder.mkdir(parents=True,exist_ok=False)
    for name in ('state.json','controller_snapshot.json','status.json','working_evidence.json'):
        if (run/name).exists():(folder/('before_'+name)).write_bytes((run/name).read_bytes())
    (folder/'operator_brief.md').write_bytes(content)
    receipt.write_text(json.dumps({**report,'status':'prepared'},indent=2))
    state['status']='running'
    state.setdefault('operator_stage_reviews',[]).append({
        'next_decision':expected,'brief_sha256':digest,'receipt':str(receipt.relative_to(run))})
    # No tasks, predictions, action IDs, counters, limits or cached results reset.
    engine.save()
    snapshot_fn(engine)
    report['status']='completed'
    receipt.write_text(json.dumps(report,indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--private-root',type=Path,required=True)
    p.add_argument('--brief',type=Path,required=True)
    p.add_argument('--expected-next',type=int,required=True)
    p.add_argument('--apply',action='store_true')
    a=p.parse_args()
    frozen=a.run.resolve()/'release'
    # Import from the original immutable release, never the new checkout.
    sys.path[:0]=[str(frozen/'src'),str(frozen/'scripts')]
    from vdc.research_r2 import Engine
    from vdc.research_bridge import snapshot
    from run_all_modules import exclusive_run
    with exclusive_run(a.run):
        config=json.loads((a.run/'manifest.json').read_text())['config']
        engine=Engine(frozen,a.private_root,a.run,config)
        print(json.dumps(reopen(engine,a.expected_next,a.brief,snapshot,a.apply)))


if __name__=='__main__':main()

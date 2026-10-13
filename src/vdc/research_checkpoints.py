"""Host-written research checkpoints. These report evidence, not scientific approval."""
from datetime import datetime, timezone
from pathlib import Path
from .io import read_json, write_json, object_hash, sha256

CONTINUATION = '''Continue towards a stable, usable diapause/dormancy model.
Completed task labels or a collection of narrow classifiers do not fulfil this goal.
Choose models, concepts, data, literature and scientific routes freely. Use positive
AND negative comparisons to select informative next experiments. Investigate recovery,
intervention and transfer to new backgrounds; test whether initial sample expression
adds information beyond time/history/treatment when legitimate pairing exists. Never
invent longitudinal pairs or use endpoint expression as an initial-state predictor.
All authorised sources remain available; historical reserved/test roles are provenance,
not renewed blindness. Report retrospective/adaptive results as such. Show actual
libraries/samples as well as conservative related-unit groups. Floating-point equality
to a baseline is numerical equivalence, not scientific improvement. Method learning
should lead to an implementation and comparison, not just a citation.
Read continuation_brief.md and latest_checkpoint.json when present. At checkpoints,
update research_notes.md: evidence learned, failed hypotheses, unresolved dependencies,
next discriminating experiment, and progress towards a usable model. Checkpoints are
automatic evidence snapshots, not requirements that every model improve.
Before requesting finish, document actual input/output, preprocessing/schema exported
with the predictor, reload/new-request verification, state-added-value comparisons,
background-transfer results and limitations. Shared models and scoped experts are both
allowed. A finish request pauses for human stage review; it is not scientific approval.
Missing pairing/labels can be a documented scientific limitation, never fabricated.
Do not purchase resources, change billing, bypass isolation or enlarge configured
resource budgets. Read/model-API counts are not successful scientific experiments.
'''


def usage_totals(workspace):
    workspace = Path(workspace)
    receipts = [read_json(p) for p in sorted((workspace/'turns').glob('*/codex_receipt.json'))]
    ledger = [read_json(p) for p in sorted((workspace/'advisor').glob('*/api/*.json'))]
    primary = {}
    for r in receipts:
        for u in r.get('usage', []):
            for k, v in u.items():
                if isinstance(v, (int, float)):
                    primary[k] = primary.get(k, 0) + v
    return {'codex_local_decision_calls': len(receipts),
            'codex_reported_usage': primary,
            'codex_usage_note': 'CLI-reported usage; local decisions are not underlying model request counts',
            'advisor_requests': len(ledger),
            'advisor_reserved_token_upper_bound': sum(r.get('budget_charge', 0) for r in ledger),
            'advisor_reported_tokens': sum((r.get('usage') or {}).get('total_tokens', 0) for r in ledger)}


def checkpoint(workspace, state, reason, config=None):
    """Unique completed action IDs since the previous checkpoint, including negatives."""
    workspace = Path(workspace)
    latest = workspace/'latest_checkpoint.json'
    prior = read_json(latest) if latest.exists() else {}
    last = state.get('next_turn', 1)-1
    previous = prior.get('through_local_turn', 0)
    seen = set()
    records = []
    for p in sorted((workspace/'turns').glob('*/response.json')):
        turn = int(p.parent.name)
        if turn > last:
            continue
        for action_id, item in read_json(p).get('results', {}).items():
            if action_id in seen:
                continue
            seen.add(action_id)
            if turn <= previous:
                continue
            r = item.get('result') or {}
            records.append({'turn': turn, 'action': action_id, 'type': item.get('type'),
                'status': item.get('status'), 'task': r.get('task'), 'parent': r.get('parent'),
                'metrics': r.get('metrics'), 'comparison': r.get('comparison'),
                'reload_verified': r.get('reload_verified'),
                'raw_matrix_inference': r.get('raw_matrix_inference'),
                'execution_seconds': r.get('execution_seconds'),
                'selected_device': r.get('selected_device'),
                'error_type': item.get('error_type'), 'diagnostic': item.get('diagnostic'),
                'receipt': str(p.relative_to(workspace)).replace('\\', '/'), 'receipt_sha256': sha256(p)})
    number = prior.get('checkpoint_number', 0)+1
    stem = workspace/'checkpoints'/f'{number:04d}_turn_{last:06d}'
    progress = state.get('worker_progress', {})
    report = {'checkpoint_number': number, 'created_at': datetime.now(timezone.utc).isoformat(),
        'reason': reason, 'status': state.get('status'), 'through_local_turn': last,
        'previous_through_local_turn': previous, 'next_local_turn': last+1,
        'cumulative_worker_progress': progress, 'new_actions': records,
        'new_completed_experiments': sum(r['type']=='experiment' and r['status']=='completed' for r in records),
        'usage': usage_totals(workspace), 'resources': config or {},
        'scientific_approval': False, 'evaluation_scope': 'adaptive_research_not_new_blind_test'}
    notes = workspace/'agent/research_notes.md'
    if notes.exists():
        report['model_written_notes_sha256'] = sha256(notes)
        stem.parent.mkdir(parents=True, exist_ok=True)
        stem.with_suffix('.notes.md').write_bytes(notes.read_bytes())
    write_json(stem.with_suffix('.json'), report)
    write_json(latest, report)
    lines = ['# Research checkpoint '+str(number), '',
        f'Reason: {reason}; through local turn {last}; next turn {last+1}.',
        'This is an evidence checkpoint, not scientific validation or approval.', '',
        f'New successful experiments: {report["new_completed_experiments"]}.',
        'Cumulative worker counters include inherited material; new actions are listed below.', '',
        '| Action | Type | Status | Task |', '| --- | --- | --- | --- |']
    for r in records:
        lines.append('| '+' | '.join(str(r.get(k) or '').replace('|','/') for k in ('action','type','status','task'))+' |')
    lines += ['', 'Full metrics, comparison scope, failures, source hashes and usage are in the paired JSON.',
              'Model-written research notes are a separate interpretation, not execution evidence.']
    stem.with_suffix('.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('CHECKPOINT '+str(stem.with_suffix('.json')), flush=True)
    return report


def due(workspace, state, every_experiments=5, every_turns=24):
    p = Path(workspace)/'latest_checkpoint.json'
    if not p.exists():
        return True
    old = read_json(p)
    progress = state.get('worker_progress', {})
    return (progress.get('experiments', 0)-old.get('cumulative_worker_progress', {}).get('experiments', 0) >= every_experiments
            or state.get('next_turn', 1)-1-old['through_local_turn'] >= every_turns)


def bind_controller(workspace, settings, state, snapshot, accept_update=False):
    """Migrate only a quiescent, aligned boundary; retain all prior receipts/session."""
    workspace = Path(workspace)
    path = workspace/'controller_binding.json'
    if path.exists():
        old = read_json(path)
        if old != settings:
            if not accept_update:
                raise ValueError('Controller workspace changed; explicitly accept controller update at an idle boundary')
            if any(old.get(k) != settings.get(k) for k in ('transport','model','protocol')):
                raise ValueError('Cannot migrate another run/transport/model/protocol')
            if 'pending' in state or (workspace/'turns'/f'{state["next_turn"]:06d}'/'codex_receipt.json').exists():
                raise ValueError('Pending or uncertain next turn; inspect before migrating controller')
            if snapshot['next_decision'] != state['next_turn']:
                raise ValueError('Local and remote next decisions do not match')
            target = workspace/'controller_versions'/('before_turn_'+str(state['next_turn'])+'_'+object_hash(old)[:12]+'.json')
            write_json(target, {'previous': old, 'current': settings, 'next_turn': state['next_turn'],
                'worker_snapshot_sha256': snapshot['snapshot_sha256'],
                'approved_explicitly': True, 'old_completed_decisions_reexecuted': False})
    write_json(path, settings)

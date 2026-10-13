"""Bounded joint evidence, with durable references rather than last-result eviction."""
import json
from .io import object_hash, write_json


def size(value):
    return len(json.dumps(value, ensure_ascii=False).encode('utf-8'))


def evidence_record(key, entry, limit):
    # Entries contain host-produced projections, never arbitrary candidate stdout.
    record = {k: entry[k] for k in ('type', 'status', 'result', 'diagnostic', 'reused_from') if k in entry}
    ref = {'type': 'read_action', 'action': key, 'offset': 0}
    if size(record) <= limit:
        return {**record, 'full_result_tool': ref}
    text = json.dumps(record, ensure_ascii=False)
    # Reserve room for JSON escaping and multibyte text. No claim that this is complete.
    excerpt = text.encode('utf-8')[:max(100, (limit-600)//2)].decode('utf-8', errors='ignore')
    return {'type': entry['type'], 'status': entry['status'], 'excerpt': excerpt,
            'excerpt_not_complete': True, 'full_result_tool': ref,
            'record_sha256': object_hash(record), 'record_bytes': size(record)}


def continuity_context(engine, context, maximum):
    """Reserve evidence space before inventory; all last-batch IDs survive trimming."""
    state = engine.state
    latest = state.get('last_batch_actions') or state.get('observations', [])[-4:]
    latest = list(dict.fromkeys(latest))[-4:]
    if not latest:
        latest = list(state['actions'])[-4:]
    # Keep code AND original metadata available across intervening decisions.
    anchors = {}
    for key in state.get('observations', list(state['actions'])):
        entry = state['actions'][key]
        if entry['status'] == 'completed' and entry['type'] in (
                'read_metadata', 'read_preparation_code', 'read_task', 'inspect_preparation'):
            anchors[entry['type']] = key
    keys = list(dict.fromkeys(latest + list(anchors.values())))
    safe = {key: {k: v for k, v in state['actions'][key].items()
                 if k in ('type', 'status', 'result', 'diagnostic', 'reused_from')} for key in keys}
    write_json(engine.run/'working_evidence.json', {
        'protocol': 'joint_evidence_v1', 'latest_batch': latest,
        'anchor_actions': anchors, 'records': safe,
        'agent_working_state': state.get('working_state', {}),
        'interpretations_are_not_validation': True})
    context['working_state'] = state.get('working_state', {})
    context['recent_actions'] = {}
    context['retained_evidence'] = {}
    context['evidence_policy'] = {'latest_batch': latest, 'anchors': anchors,
        'full_records': 'working_evidence.json; read_action(action, offset)',
        'budget_bytes': maximum, 'truncated_records_are_explicit': True}
    # Large inventory and old task summaries have paging tools; do not sacrifice
    # the just-requested code/metadata pair to repeat these indexes.
    context['inventory']['assets'] = {}
    context['inventory']['sources'] = [{k: s[k] for k in ('id', 'asset_count', 'disposition') if k in s}
                                      for s in context['inventory']['sources']]
    context['memory'] = [str(m)[:600] for m in state.get('memories', [])[-4:]]
    for field in ('tasks', 'experiments'):
        while context[field] and size(context) > maximum//3:
            del context[field][next(iter(context[field]))]
    available = maximum-size(context)-1500
    if available < 1000*max(1, len(keys)):
        raise ValueError('Context budget cannot retain joint tool evidence; enlarge request budget')
    per_record = min(11000, available//max(1, len(keys)))
    for key in keys:
        field = 'recent_actions' if key in latest else 'retained_evidence'
        context[field][key] = evidence_record(key, state['actions'][key], per_record)
    if size(context) > maximum:
        raise ValueError('Joint evidence exceeds context budget; no silent last-result fallback')
    return context


def update_working_state(engine, proposal):
    value = proposal.get('working_state')
    if value is None:
        return
    if not isinstance(value, dict) or size(value) > 6000:
        raise ValueError('working_state must be an object of at most 6000 UTF-8 bytes')
    refs = value.get('evidence_actions', [])
    if not isinstance(refs, list) or any(k not in engine.state['actions'] for k in refs):
        raise ValueError('working_state evidence_actions must reference recorded actions')
    engine.state['working_state'] = value

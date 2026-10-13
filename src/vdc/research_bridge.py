"""Durable single-writer mailbox; computation remains in Engine's isolated tools."""
from contextlib import contextmanager
import json
from pathlib import Path
import os
import time
from .io import read_json, write_json, object_hash
from .research_protocol import decision, SYSTEM, TOOLS

PROTOCOL = 'vdc_codex_worker_v1'


@contextmanager
def lock(directory, name='.bridge.lock'):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/name).open('a+b') as stream:
        stream.seek(0); stream.write(b'0'); stream.flush(); stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def snapshot(engine):
    context = engine.context()
    value = {'protocol': PROTOCOL, 'run_id': engine.run.name,
             'manifest_sha256': object_hash(read_json(engine.run/'manifest.json')),
             'next_decision': engine.state['next_decision'], 'status': engine.state['status'],
             'context': context, 'instructions': SYSTEM+'\n'+TOOLS}
    value['snapshot_sha256'] = object_hash(value)
    write_json(engine.run/'controller_snapshot.json', value)
    return value


def submit(run, request):
    run = Path(run)
    if request.get('protocol') != PROTOCOL:
        raise ValueError('Unsupported mailbox protocol')
    index = request['index']
    if type(index) is not int or not 1 <= index <= 1000000:
        raise ValueError('Invalid decision index')
    decision(request['proposal'])
    if len(json.dumps(request).encode()) > 2_000_000:
        raise ValueError('Decision is too large; use smaller code actions')
    key = f'{index:06d}'
    queue = run/'mailbox'
    with lock(queue):
        path = queue/(key+'.request.json')
        if path.exists():
            if read_json(path) != request:
                raise ValueError('Decision ID already has different content; no overwrite')
            return {'status': 'already_submitted', 'id': key, 'request_sha256': object_hash(request)}
        current = read_json(run/'controller_snapshot.json')
        if current['status'] in ('stalled_no_progress', 'completed_declared_scope', 'paused_by_operator'):
            raise ValueError('Worker is stopped; inspect status before starting a new phase')
        if request['snapshot_sha256'] != current['snapshot_sha256'] or index != current['next_decision']:
            raise ValueError('Stale snapshot; reload context and reconsider the decision')
        write_json(path, request)
    return {'status': 'submitted', 'id': key, 'request_sha256': object_hash(request)}


def process_one(engine):
    queue = engine.run/'mailbox'
    queue.mkdir(exist_ok=True)
    index = engine.state['next_decision']
    key = f'{index:06d}'
    path = queue/(key+'.request.json')
    # Detect a crash after action execution/advance but before response publication.
    for marker in queue.glob('*.running.json'):
        oldkey = marker.name.split('.')[0]
        if not (queue/(oldkey+'.response.json')).exists():
            raise RuntimeError('Uncertain worker outcome for '+oldkey+'; inspect actions before resubmission')
    if not path.exists():
        return False
    if (queue/(key+'.response.json')).exists():
        raise RuntimeError('Response exists but worker index did not advance')
    request = read_json(path)
    current = read_json(engine.run/'controller_snapshot.json')
    if request['snapshot_sha256'] != current['snapshot_sha256']:
        raise ValueError('Queued decision snapshot changed')
    write_json(queue/(key+'.running.json'), {'request_sha256': object_hash(request), 'started_at': time.time()})
    results = engine.execute_decision(request['proposal'], index)
    new_snapshot = snapshot(engine)
    write_json(queue/(key+'.response.json'), {'protocol': PROTOCOL, 'id': key,
        'request_sha256': object_hash(request), 'results': results,
        'snapshot': new_snapshot, 'working_evidence': read_json(engine.run/'working_evidence.json'),
        'finished_at': time.time(), 'new_api_calls_on_worker': 0})
    return True


def rpc(run, packet):
    run = Path(run)
    command = packet.get('operation')
    if command == 'snapshot':
        return read_json(run/'controller_snapshot.json')
    if command == 'submit':
        return submit(run, packet['request'])
    if command == 'poll':
        index = packet['index']
        if type(index) is not int or not 1 <= index <= 1000000:
            raise ValueError('Invalid decision index')
        key = f'{index:06d}'
        path = run/'mailbox'/(key+'.response.json')
        if path.exists():
            return read_json(path)
        heartbeat = read_json(run/'worker_status.json') if (run/'worker_status.json').exists() else {}
        return {'status': 'pending', 'worker': heartbeat,
                'running': (run/'mailbox'/(key+'.running.json')).exists()}
    if command == 'evidence':
        return read_json(run/'working_evidence.json')
    raise ValueError('Unknown RPC operation')

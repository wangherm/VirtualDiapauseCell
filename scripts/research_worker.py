"""AutoDL worker: no research LLM or OpenAI credentials required here."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from vdc.io import read_json, write_json
from vdc.research_bridge import rpc, snapshot, process_one
from vdc.research_r2 import Engine
from vdc.research_r2_continue import continue_engine_phase
from vdc.research_r2_devices import configure_devices
from run_all_modules import exclusive_run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['init', 'serve', 'resume', 'rpc'])
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--parent', type=Path)
    p.add_argument('--private-root', type=Path, default=os.environ.get('VDC_PRIVATE_ROOT', '/root/autodl-tmp/vdc-private'))
    a = p.parse_args()
    if a.command == 'rpc':
        raw = sys.stdin.buffer.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ValueError('RPC request too large')
        # Only protocol JSON goes to stdout; logs never contaminate the transport.
        with contextlib.redirect_stdout(sys.stderr):
            result = rpc(a.run, json.loads(raw))
        print(json.dumps(result, ensure_ascii=False)); return
    if a.command == 'init':
        if not a.parent:
            p.error('init requires --parent')
        c = read_json(ROOT/'configs/research_r2.json')
        c['runtime']['runtime_root'] = os.environ.get('VDC_R2_RUNTIME_ROOT', str(a.private_root/'r2_runtime_devices_v2'))
        c['runtime'] = configure_devices(c['runtime'], os.environ.get('VDC_R2_DEVICE_POLICY', 'cpu'))
        c['controller'] = 'local_codex_cli_with_optional_deepseek_advisor'
        with exclusive_run(a.parent):
            engine = continue_engine_phase(ROOT, a.private_root, a.parent, a.run, c)
        release = a.run/'release'
        release.mkdir()
        for name in ('src', 'scripts', 'configs', 'knowledge'):
            shutil.copytree(ROOT/name, release/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        if (ROOT/'RELEASE_COMMIT.txt').is_file():
            shutil.copyfile(ROOT/'RELEASE_COMMIT.txt', release/'RELEASE_COMMIT.txt')
        snapshot(engine)
        print(json.dumps({'status': 'ready_for_local_codex', 'run': str(a.run), 'api_calls_executed': 0}))
        return
    with exclusive_run(a.run):
        c = read_json(a.run/'manifest.json')['config']
        engine = Engine(ROOT, a.private_root, a.run, c)
        if a.command == 'resume':
            if engine.state['status'] != 'paused_by_operator' or (a.run/'STOP').exists():
                raise ValueError('Resume only an operator-paused run after removing STOP; stalled/uncertain outcomes require inspection')
            previous = read_json(a.run/'worker_pause.json')['resume_status']
            engine.state['status'] = previous
            engine.save()
        snapshot(engine)
        try:
            while engine.state['status'] not in ('stalled_no_progress', 'completed_declared_scope', 'paused_by_operator'):
                if (a.run/'STOP').exists():
                    write_json(a.run/'worker_pause.json', {'resume_status': engine.state['status']})
                    engine.state['status'] = 'paused_by_operator'; engine.save(); snapshot(engine); break
                write_json(a.run/'worker_status.json', {'status': 'waiting_or_executing', 'updated_at': time.time(),
                    'next_decision': engine.state['next_decision']})
                if not process_one(engine):
                    time.sleep(2)
            engine.report()
            write_json(a.run/'worker_status.json', {'status': engine.state['status'], 'updated_at': time.time()})
        except Exception as exc:
            write_json(a.run/'worker_status.json', {'status': 'failed_requires_inspection',
                'error_type': type(exc).__name__, 'updated_at': time.time()})
            raise


if __name__ == '__main__':
    main()

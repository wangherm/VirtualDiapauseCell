"""Run on the LOCAL computer: Codex primary, DeepSeek advisory, AutoDL compute."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from vdc.io import read_json
from vdc.research_codex import controller, transport

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['doctor', 'run'])
    p.add_argument('--connection', type=Path, required=True, help='Private *.local.json transport config')
    p.add_argument('--workspace', type=Path)
    p.add_argument('--turns', type=int, default=24, help='Local decision-turn budget for this invocation, not experiment count')
    p.add_argument('--model', help='Omit to use locally configured Codex model')
    p.add_argument('--continuous', action='store_true', help='Continue batches until STOP, worker stop, stage review or error; no resource increases')
    p.add_argument('--checkpoint-experiments', type=int, default=5)
    p.add_argument('--checkpoint-turns', type=int, default=24)
    p.add_argument('--accept-controller-update', action='store_true', help='Record source migration at a verified idle boundary without replaying old turns')
    p.add_argument('--brief', type=Path, help='Operator continuation brief; copied and fingerprinted, never applied to old decisions')
    p.add_argument('--advisor-config', type=Path, default=ROOT/'configs/research_r2.json')
    a = p.parse_args()
    if min(a.turns, a.checkpoint_experiments, a.checkpoint_turns) < 1:
        p.error('Turn and checkpoint intervals must be positive')
    connection = read_json(a.connection)
    if a.command == 'doctor':
        s = transport(connection, {'operation': 'snapshot'})
        print(json.dumps({k: s[k] for k in ('protocol', 'run_id', 'status', 'next_decision')}, indent=2))
    else:
        if not a.workspace:
            p.error('run requires --workspace (private local directory)')
        print(json.dumps(controller(connection, a.workspace, a.turns, a.model, read_json(a.advisor_config),
            continuous=a.continuous, checkpoint_experiments=a.checkpoint_experiments,
            checkpoint_turns=a.checkpoint_turns, accept_update=a.accept_controller_update, brief=a.brief), indent=2))

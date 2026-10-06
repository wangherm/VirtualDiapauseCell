#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
RELEASE="$PWD"
export VDC_PRIVATE_ROOT="${VDC_PRIVATE_ROOT:-/root/autodl-tmp/vdc-private}"
export VDC_PYTHON="${VDC_PYTHON:-/root/autodl-tmp/VirtualDiapauseCell/.venv-autodl-vdc/bin/python}"
export PYTHONPATH="$RELEASE/src"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
RUN="${VDC_INT1_RUN:-$(cat "$VDC_PRIVATE_ROOT/LATEST_INT1.txt")}"; PORT="${VDC_INT1_PORT:-8769}"
test -f "$RUN/model/bundle.json"
if [[ "${1:-}" == --worker ]]; then
  exec >> "$RUN/service.log" 2>&1
  exec "$VDC_PYTHON" -m vdc serve-app --model "$RUN/model" --output "$RUN/requests" --port "$PORT" --base-path "$VDC_QWEN_BASE"
fi
command -v screen >/dev/null
export VDC_QWEN_BASE="$("$VDC_PYTHON" - "$RUN" <<'PY'
import sys
from pathlib import Path
from vdc.io import read_json
from vdc.integration_bundle import load_manifest
run=Path(sys.argv[1]);load_manifest(run/'model')
status=read_json(run/'status.json')
if status['status'] not in {'full_verified','partial'}:raise SystemExit('Complete qualification before serving')
stage=Path(read_json(run/'assembly/assembly.json')['stage_path'])
source=Path(read_json(stage/'config.json')['source'])
print(read_json(source/'config.json')['model_path'])
PY
)"
"$VDC_PYTHON" - "$PORT" <<'PY'
import socket,sys
with socket.socket() as s:s.bind(('127.0.0.1',int(sys.argv[1])))
PY
export VDC_INT1_RUN="$RUN" VDC_INT1_PORT="$PORT"
screen -dmS "vdc_int1_app_$PORT" bash "$RELEASE/scripts/serve_int1.sh" --worker
printf 'Local app requested: http://127.0.0.1:%s\nLog: %s/service.log\nUse an SSH tunnel; inspect /health before use.\n' "$PORT" "$RUN"

#!/usr/bin/env bash
# Repair-phase entry: verify a real-data tool chain, preserve the parent, start autonomous research.
set -euo pipefail
cd "$(dirname "$0")/.."
export VDC_PRIVATE_ROOT="${VDC_PRIVATE_ROOT:-/root/autodl-tmp/vdc-private}"
export VDC_PYTHON="${VDC_PYTHON:-/root/autodl-tmp/VirtualDiapauseCell/.venv-autodl-vdc/bin/python}"
export VDC_R2_RUNTIME_ROOT="${VDC_R2_RUNTIME_ROOT:-$VDC_PRIVATE_ROOT/r2_runtime_devices_v2}"
export VDC_R2_DEVICE_POLICY="${VDC_R2_DEVICE_POLICY:-cpu}"
export PYTHONPATH="$PWD/src" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export VDC_RESEARCH_CONFIG="$PWD/configs/research_r2.json"
if [[ "${1:-}" == --worker ]]; then
  exec >> "$VDC_ENGINE_SETUP_LOG" 2>&1
  trap 'code=$?; echo "ENGINE_SETUP_EXIT_CODE=$code"' EXIT
  echo 'ENGINE_CHECK: real public data, no API calls; this is not autonomous research.'
  CHECK="$VDC_PRIVATE_ROOT/runs/engine_check_$(date -u +%Y%m%dT%H%M%SZ)_$$"
  "$VDC_PYTHON" scripts/smoke_research_engine.py --private-root "$VDC_PRIVATE_ROOT" \
    --runtime-root "$VDC_R2_RUNTIME_ROOT" --parent "$VDC_ENGINE_PARENT" --out "$CHECK"
  export VDC_R2_RUN="$VDC_PRIVATE_ROOT/runs/research_engine_$(date -u +%Y%m%dT%H%M%SZ)_$$"
  "$VDC_PYTHON" scripts/continue_research_r2.py --engine-phase --parent "$VDC_ENGINE_PARENT" --out "$VDC_R2_RUN"
  cp "$CHECK/ENGINE_CHECK.json" "$VDC_R2_RUN/engine_check_receipt.json"
  cp "$VDC_RESEARCH_API_CHECK" "$VDC_R2_RUN/api_check_receipt.json"
  export VDC_RESEARCH_API_CHECK="$VDC_R2_RUN/api_check_receipt.json" VDC_R2_RESUME=1
  unset VDC_R2_RETRY_API_ERRORS VDC_R2_RESUME_AFTER_STALL VDC_R2_GPU
  echo "NEW_RUN=$VDC_R2_RUN"
  bash "$VDC_R2_RUN/release/scripts/launch_research_r2.sh"
  exit 0
fi
test -x "$VDC_PYTHON"
command -v screen >/dev/null
export VDC_ENGINE_PARENT="${1:?Pass the stopped parent run directory}"
test -f "$VDC_ENGINE_PARENT/state.json"
export VDC_RESEARCH_API_BASE="$("$VDC_PYTHON" -c 'import json,os;print(json.load(open(os.environ["VDC_RESEARCH_CONFIG"]))["expected_api_base"])')"
export VDC_RESEARCH_MODEL="$("$VDC_PYTHON" -c 'import json,os;print(json.load(open(os.environ["VDC_RESEARCH_CONFIG"]))["expected_model"])')"
if [[ -z "${VDC_RESEARCH_API_KEY:-}" ]]; then
  read -rsp 'DeepSeek API key (hidden): ' VDC_RESEARCH_API_KEY; echo
fi
export VDC_RESEARCH_API_KEY
if [[ -z "${VDC_RESEARCH_API_CHECK:-}" ]]; then
  VDC_RESEARCH_API_CHECK="$("$VDC_PYTHON" - <<'PY'
import os
from pathlib import Path
from vdc.io import read_json
from vdc.research_api import verify_connection_check
c=read_json(os.environ['VDC_RESEARCH_CONFIG'])
for p in sorted((Path(os.environ['VDC_PRIVATE_ROOT'])/'runs').glob('research_api_check_*/status.json'),reverse=True):
    try: verify_connection_check(p,c)
    except (OSError,ValueError,KeyError): continue
    print(p);break
else:raise SystemExit('No matching API receipt. Run check_research_api.py once explicitly; no paid request sent here.')
PY
)"
fi
export VDC_RESEARCH_API_CHECK
"$VDC_PYTHON" scripts/check_research_api.py --check "$VDC_RESEARCH_API_CHECK"
mkdir -p "$VDC_PRIVATE_ROOT/runs"
export VDC_ENGINE_SETUP_LOG="$VDC_PRIVATE_ROOT/runs/engine_setup_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
printf '%s\n' "$VDC_ENGINE_SETUP_LOG" > "$VDC_PRIVATE_ROOT/LATEST_ENGINE_SETUP.txt"
SESSION="vdc_engine_setup_$$"
screen -dmS "$SESSION" bash "$PWD/scripts/start_research_engine.sh" --worker
printf 'SCREEN=%s\nSETUP_LOG=%s\n' "$SESSION" "$VDC_ENGINE_SETUP_LOG"

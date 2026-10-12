#!/usr/bin/env bash
# One foreground entry: qualify CPU/GPU, then dispatch the research queue in screen.
set -euo pipefail
cd "$(dirname "$0")/.."
export VDC_PRIVATE_ROOT="${VDC_PRIVATE_ROOT:-/root/autodl-tmp/vdc-private}"
export VDC_PYTHON="${VDC_PYTHON:-/root/autodl-tmp/VirtualDiapauseCell/.venv-autodl-vdc/bin/python}"
export PYTHONPATH="$PWD/src"
export VDC_RESEARCH_CONFIG="$PWD/configs/research_r2.json"
export VDC_R2_RUNTIME_ROOT="${VDC_R2_RUNTIME_ROOT:-$VDC_PRIVATE_ROOT/r2_runtime_devices_v2}"
export VDC_R2_DEVICE_POLICY="${VDC_R2_DEVICE_POLICY:-ai}"
if [[ "${1:-}" == --worker ]]; then
  exec >> "$VDC_R2_BOOT_LOG" 2>&1
  trap 'code=$?; echo "SETUP_LAUNCH_EXIT_CODE=$code"' EXIT
  echo "Device policy: $VDC_R2_DEVICE_POLICY; GPU must pass real qualification before AI can select it."
  bash scripts/setup_research_r2.sh
  bash scripts/launch_research_r2.sh
  exit 0
fi
test -x "$VDC_PYTHON"
command -v screen >/dev/null
if [[ -n "${VDC_R2_RUN:-}" || "${VDC_R2_RESUME:-0}" == 1 ]]; then
  echo 'This entry creates a NEW run. For resume use launch_research_r2.sh from the frozen release.'; exit 2
fi
export VDC_RESEARCH_API_BASE="$("$VDC_PYTHON" -c 'import json,os;print(json.load(open(os.environ["VDC_RESEARCH_CONFIG"]))["expected_api_base"])')"
export VDC_RESEARCH_MODEL="$("$VDC_PYTHON" -c 'import json,os;print(json.load(open(os.environ["VDC_RESEARCH_CONFIG"]))["expected_model"])')"
if [[ -z "${VDC_RESEARCH_API_KEY:-}" ]]; then
  read -rsp 'DeepSeek API key (hidden): ' VDC_RESEARCH_API_KEY
  echo
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
    try:verify_connection_check(p,c)
    except (ValueError,OSError,KeyError):continue
    print(p);break
else:raise SystemExit('No matching passed API receipt. Set VDC_RESEARCH_API_CHECK; no paid check was sent.')
PY
)"
fi
export VDC_RESEARCH_API_CHECK
"$VDC_PYTHON" scripts/check_research_api.py --check "$VDC_RESEARCH_API_CHECK"
mkdir -p "$VDC_PRIVATE_ROOT/runs"
export VDC_R2_BOOT_LOG="$VDC_PRIVATE_ROOT/runs/r2_setup_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
printf '%s\n' "$VDC_R2_BOOT_LOG" > "$VDC_PRIVATE_ROOT/LATEST_R2_SETUP.txt"
SESSION="vdc_r2_setup_$$"
screen -dmS "$SESSION" bash "$PWD/scripts/start_research_r2.sh" --worker
printf 'SCREEN=%s\nSETUP_LOG=%s\n' "$SESSION" "$VDC_R2_BOOT_LOG"

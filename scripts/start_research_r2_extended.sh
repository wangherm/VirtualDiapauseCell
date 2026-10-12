#!/usr/bin/env bash
# Continue the stopped preparation-only run; no overwrite, no new runtime install.
set -euo pipefail
cd "$(dirname "$0")/.."
export VDC_PRIVATE_ROOT="${VDC_PRIVATE_ROOT:-/root/autodl-tmp/vdc-private}"
export VDC_PYTHON="${VDC_PYTHON:-/root/autodl-tmp/VirtualDiapauseCell/.venv-autodl-vdc/bin/python}"
export VDC_R2_DEVICE_POLICY=cpu
export VDC_R2_RUNTIME_ROOT="${VDC_R2_RUNTIME_ROOT:-$VDC_PRIVATE_ROOT/r2_runtime_devices_v2}"
export PYTHONPATH="$PWD/src" PYTHONUNBUFFERED=1
export VDC_RESEARCH_CONFIG="$PWD/configs/research_r2.json"
command -v screen >/dev/null
PARENT="${1:?Pass the stopped parent run directory}"
export VDC_RESEARCH_API_BASE="$("$VDC_PYTHON" -c 'import json,os;print(json.load(open(os.environ["VDC_RESEARCH_CONFIG"]))["expected_api_base"])')"
export VDC_RESEARCH_MODEL="$("$VDC_PYTHON" -c 'import json,os;print(json.load(open(os.environ["VDC_RESEARCH_CONFIG"]))["expected_model"])')"
if [[ -z "${VDC_RESEARCH_API_KEY:-}" ]]; then
  read -rsp 'DeepSeek API key (hidden): ' VDC_RESEARCH_API_KEY; echo
fi
export VDC_RESEARCH_API_KEY
# Client implementation changed, so perform one fresh small connection check.
CHECK="$VDC_PRIVATE_ROOT/runs/research_api_check_$(date -u +%Y%m%dT%H%M%SZ)_$$"
"$VDC_PYTHON" scripts/check_research_api.py --config "$VDC_RESEARCH_CONFIG" --out "$CHECK"
export VDC_RESEARCH_API_CHECK="$CHECK/status.json"
export VDC_R2_RUN="$VDC_PRIVATE_ROOT/runs/research_r2_extended_$(date -u +%Y%m%dT%H%M%SZ)_$$"
"$VDC_PYTHON" scripts/continue_research_r2.py --parent "$PARENT" --out "$VDC_R2_RUN"
cp "$VDC_RESEARCH_API_CHECK" "$VDC_R2_RUN/api_check_receipt.json"
export VDC_RESEARCH_API_CHECK="$VDC_R2_RUN/api_check_receipt.json"
export VDC_R2_RESUME=1
unset VDC_R2_RETRY_API_ERRORS VDC_R2_GPU
bash "$VDC_R2_RUN/release/scripts/launch_research_r2.sh"

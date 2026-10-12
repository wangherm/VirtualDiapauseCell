#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SOURCE="$PWD"
export VDC_PRIVATE_ROOT="${VDC_PRIVATE_ROOT:-/root/autodl-tmp/vdc-private}"
export VDC_PYTHON="${VDC_PYTHON:-/root/autodl-tmp/VirtualDiapauseCell/.venv-autodl-vdc/bin/python}"
export PYTHONPATH="$SOURCE/src"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
export VDC_RESEARCH_CONFIG="${VDC_RESEARCH_CONFIG:-$SOURCE/configs/research_r2.json}"
if [[ "${1:-}" == --worker ]]; then
  RUN="$2"
  export VDC_RESEARCH_CONFIG="$SOURCE/configs/research_r2_selected.json"
  exec >> "$RUN/console.log" 2>&1
  trap 'code=$?; echo "TASK_EXIT_CODE=$code"; printf "%s\n" "$code" > "$RUN/exit_code.txt"' EXIT
  args=(run --run "$RUN")
  if [[ "${VDC_R2_RETRY_API_ERRORS:-0}" == 1 ]]; then args+=(--retry-api-errors); fi
  if [[ -n "${VDC_R2_EXTRA_DATA_ROOT:-}" ]]; then args+=(--extra-data-root "$VDC_R2_EXTRA_DATA_ROOT"); fi
  code=0
  "$VDC_PYTHON" -u scripts/run_research_r2.py "${args[@]}" || code=$?
  "$VDC_PYTHON" scripts/pack_research_r2.py --run "$RUN" || true
  exit "$code"
fi
command -v screen >/dev/null
: "${VDC_RESEARCH_API_KEY:?Set DeepSeek key in server environment; do not paste it into chat}"
: "${VDC_RESEARCH_API_CHECK:?Set passed API check receipt}"
"$VDC_PYTHON" scripts/check_research_api.py --check "$VDC_RESEARCH_API_CHECK"
if [[ -z "${VDC_RESEARCH_ASSET_REPORT:-}" && -f "$VDC_PRIVATE_ROOT/LATEST_RESEARCH_DATA.txt" ]]; then
  export VDC_RESEARCH_ASSET_REPORT="$(cat "$VDC_PRIVATE_ROOT/LATEST_RESEARCH_DATA.txt")/status.json"
fi
"$VDC_PYTHON" scripts/run_research_r2.py doctor
RUN="${VDC_R2_RUN:-$VDC_PRIVATE_ROOT/runs/research_r2_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
mkdir -p "$RUN"
RUN="$(realpath "$RUN")"
if [[ -f "$RUN/manifest.json" && "${VDC_R2_RESUME:-0}" != 1 ]]; then
  echo 'Existing run. Inspect status, then set VDC_R2_RESUME=1; no automatic API retries.'; exit 2
fi
if [[ ! -d "$RUN/release" ]]; then
  "$VDC_PYTHON" - "$SOURCE" "$RUN/release" "$VDC_RESEARCH_CONFIG" <<'PY'
import sys,shutil
from pathlib import Path
src,dst,config=map(Path,sys.argv[1:]);dst.mkdir()
for name in ('src','scripts','configs','knowledge'):
    shutil.copytree(src/name,dst/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copyfile(config,dst/'configs/research_r2_selected.json')
PY
fi
SESSION="vdc_r2_$(date -u +%H%M%S)_$$"
screen -dmS "$SESSION" bash "$RUN/release/scripts/launch_research_r2.sh" --worker "$RUN"
printf '%s\n' "$RUN" > "$VDC_PRIVATE_ROOT/LATEST_RESEARCH_R2.txt"
printf 'SCREEN=%s\nRUN=%s\n' "$SESSION" "$RUN"

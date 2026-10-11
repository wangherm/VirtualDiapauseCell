#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SOURCE="$PWD"
export VDC_RESEARCH_CONFIG="${VDC_RESEARCH_CONFIG:-$SOURCE/configs/research_round1.json}"
# Workers always use the frozen selected config, never the mutable source file.
if [[ "${1:-}" == --worker && -f "$SOURCE/configs/research_selected.json" ]]; then
  export VDC_RESEARCH_CONFIG="$SOURCE/configs/research_selected.json"
fi
export VDC_PRIVATE_ROOT="${VDC_PRIVATE_ROOT:-/root/autodl-tmp/vdc-private}"
export VDC_PYTHON="${VDC_PYTHON:-/root/autodl-tmp/VirtualDiapauseCell/.venv-autodl-vdc/bin/python}"
export PYTHONPATH="$SOURCE/src"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
if [[ "${1:-}" != --worker ]]; then
  command -v screen >/dev/null
  test -x "$VDC_PYTHON"
  : "${VDC_RESEARCH_API_BASE:?Set API base URL on server}"
  : "${VDC_RESEARCH_MODEL:?Set research model name}"
  : "${VDC_RESEARCH_API_KEY:?Set API key on server; never paste into chat}"
  : "${VDC_RESEARCH_API_CHECK:?Set path to passed check_research_api.py status.json}"
  "$VDC_PYTHON" scripts/check_research_api.py --check "$VDC_RESEARCH_API_CHECK"
  "$VDC_PYTHON" -c 'from vdc.research_api import endpoint; endpoint()'
  "$VDC_PYTHON" scripts/run_research.py doctor
  RUN="${VDC_RESEARCH_RUN:-$VDC_PRIVATE_ROOT/runs/research_r1_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
  mkdir -p "$RUN"
  RUN="$(realpath "$RUN")"
  if [[ -f "$RUN/manifest.json" && "${VDC_RESEARCH_RESUME:-0}" != 1 ]]; then
    echo 'Existing run: inspect then set VDC_RESEARCH_RESUME=1 for signature-checked resume'; exit 2
  fi
  RELEASE="$RUN/release"
  if [[ ! -f "$RUN/RELEASE.txt" ]]; then
    "$VDC_PYTHON" - "$SOURCE" "$RELEASE" "$VDC_RESEARCH_CONFIG" <<'PY'
import sys,shutil
from pathlib import Path
source,out,config=map(Path,sys.argv[1:]);out.mkdir(parents=True,exist_ok=False)
for name in ('src','scripts','configs','knowledge','data/curated'):
    shutil.copytree(source/name,out/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copyfile(config,out/'configs/research_selected.json')
PY
    printf '%s\n' "$RELEASE" > "$RUN/RELEASE.txt"
  else
    RELEASE="$(cat "$RUN/RELEASE.txt")"
  fi
  "$VDC_PYTHON" - "$VDC_RESEARCH_API_CHECK" "$RUN/api_check.json" <<'PY'
import shutil,sys
from pathlib import Path
if Path(sys.argv[1]).resolve() != Path(sys.argv[2]).resolve():
    shutil.copyfile(sys.argv[1],sys.argv[2])
PY
  export VDC_RESEARCH_API_CHECK="$RUN/api_check.json"
  SESSION="vdc_research_$(date -u +%H%M%S)_$$"
  screen -dmS "$SESSION" bash "$RELEASE/scripts/launch_research.sh" --worker "$RUN"
  printf '%s\n' "$RUN" > "$VDC_PRIVATE_ROOT/LATEST_RESEARCH.txt"
  printf 'SCREEN=%s\nRUN=%s\nRELEASE=%s\n' "$SESSION" "$RUN" "$RELEASE"
  exit 0
fi
RUN="$2"
exec >> "$RUN/console.log" 2>&1
trap 'code=$?; printf "TASK_EXIT_CODE=%s\n" "$code"; printf "%s\n" "$code" > "$RUN/exit_code.txt"' EXIT
rm -f -- "$RUN/exit_code.txt"
args=(run --run "$RUN")
if [[ "${VDC_RESEARCH_RETRY_API_ERRORS:-0}" == 1 ]]; then args+=(--retry-api-errors); fi
code=0
"$VDC_PYTHON" scripts/check_research_api.py --check "$VDC_RESEARCH_API_CHECK"
"$VDC_PYTHON" -u scripts/run_research.py "${args[@]}" || code=$?
"$VDC_PYTHON" scripts/pack_research_report.py --run "$RUN" || true
exit "$code"

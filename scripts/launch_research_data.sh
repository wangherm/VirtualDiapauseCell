#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SOURCE="$PWD"
export VDC_PRIVATE_ROOT="${VDC_PRIVATE_ROOT:-/root/autodl-tmp/vdc-private}"
export VDC_PYTHON="${VDC_PYTHON:-/root/autodl-tmp/VirtualDiapauseCell/.venv-autodl-vdc/bin/python}"
export PYTHONPATH="$SOURCE/src"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
if [[ "${1:-}" != --worker ]]; then
  command -v screen >/dev/null
  test -x "$VDC_PYTHON"
  export VDC_PUBLIC_CACHE="${VDC_PUBLIC_CACHE:-$SOURCE/data/raw}"
  RUN="${VDC_RESEARCH_DATA_RUN:-$VDC_PRIVATE_ROOT/runs/research_data_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
  mkdir -p "$RUN"
  RUN="$(realpath "$RUN")"
  RELEASE="$RUN/release"
  if [[ ! -d "$RELEASE" ]]; then
    "$VDC_PYTHON" - "$SOURCE" "$RELEASE" <<'PY'
from pathlib import Path
import shutil,sys
source,out=map(Path,sys.argv[1:]);out.mkdir()
for name in ('src','scripts','configs','knowledge'):
    shutil.copytree(source/name,out/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
PY
  fi
  SESSION="vdc_data_$(date -u +%H%M%S)_$$"
  screen -dmS "$SESSION" bash "$RELEASE/scripts/launch_research_data.sh" --worker "$RUN"
  printf '%s\n' "$RUN" > "$VDC_PRIVATE_ROOT/LATEST_RESEARCH_DATA.txt"
  printf 'SCREEN=%s\nRUN=%s\n' "$SESSION" "$RUN"
  exit 0
fi
RUN="$2"
exec >> "$RUN/console.log" 2>&1
trap 'code=$?; printf "TASK_EXIT_CODE=%s\n" "$code"; printf "%s\n" "$code" > "$RUN/exit_code.txt"' EXIT
rm -f -- "$RUN/exit_code.txt"
echo 'Public files only; no private downloads, API calls, training or role changes.'
args=(--out "$RUN" --raw "${VDC_RESEARCH_PUBLIC_RAW:-$VDC_PRIVATE_ROOT/public_assets}"
      --cache "$VDC_PUBLIC_CACHE" --workers "${VDC_DOWNLOAD_WORKERS:-2}" --download)
if [[ -n "${VDC_EXTRA_PUBLIC_CACHE:-}" ]]; then args+=(--cache "$VDC_EXTRA_PUBLIC_CACHE"); fi
"$VDC_PYTHON" scripts/acquire_research_data.py "${args[@]}"

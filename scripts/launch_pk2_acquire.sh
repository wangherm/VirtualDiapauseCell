#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_DIR="$PWD"
PY="${VDC_PYTHON:-$REPO_DIR/.venv-autodl-vdc/bin/python}"
test -x "$PY" || { echo 'Set VDC_PYTHON to the existing VDC environment'; exit 1; }
if [[ "${1:-}" != '--worker' ]]; then
  command -v screen >/dev/null || { echo 'Install screen first'; exit 1; }
  RUN_DIR="${VDC_ACQUIRE_RUN:-$REPO_DIR/runs/pk2_acquire_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
  mkdir -p "$RUN_DIR" runs
  RUN_DIR="$(realpath "$RUN_DIR")"
  printf '%s\n' "$RUN_DIR" > runs/LATEST_PK2_ACQUIRE.txt
  SESSION="vdc_pk2_data_$(date -u +%H%M%S)_$$"
  screen -dmS "$SESSION" bash "$REPO_DIR/scripts/launch_pk2_acquire.sh" --worker "$RUN_DIR"
  printf 'SCREEN=%s\nRUN_DIR=%s\nProgress: python scripts/pk2_status.py\n' "$SESSION" "$RUN_DIR"
  exit 0
fi
RUN_DIR="$2"
exec >> "$RUN_DIR/console.log" 2>&1
trap 'code=$?; printf "TASK_EXIT_CODE=%s\n" "$code"; printf "%s\n" "$code" > "$RUN_DIR/exit_code.txt"' EXIT
rm -f "$RUN_DIR/exit_code.txt"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
echo 'Public acquisition only. No PK2 training, private data or heldout queries.'
"$PY" scripts/fetch_pk2_public.py --out "$RUN_DIR" --raw "${VDC_PK2_RAW:-$REPO_DIR/data/raw/pk2_public}" --workers "${VDC_DOWNLOAD_WORKERS:-2}" --download

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_DIR="$PWD"
PY="${VDC_PYTHON:-$REPO_DIR/.venv-autodl-vdc/bin/python}"
test -x "$PY" || { echo 'Set VDC_PYTHON to the existing CUDA environment'; exit 1; }
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${1:-}" != '--worker' ]]; then
  command -v screen >/dev/null || { echo 'screen is required'; exit 1; }
  : "${VDC_PRIVATE_ROOT:?Set VDC_PRIVATE_ROOT}"
  : "${VDC_PK1_RUN:?Set VDC_PK1_RUN}"
  : "${VDC_ACQUIRE_RUN:?Set VDC_ACQUIRE_RUN to completed acquisition}"
  : "${VDC_PK2_RAW:?Set VDC_PK2_RAW to the verified raw cache}"
  : "${VDC_QWEN_MODEL:?Set VDC_QWEN_MODEL to the downloaded base snapshot}"
  test -f "$VDC_PRIVATE_ROOT/sample_roles.json"
  test -f "$VDC_PK1_RUN/module_status.json"
  test -f "$VDC_ACQUIRE_RUN/acquisition_status.json"
  test -f "$VDC_QWEN_MODEL/model.safetensors.index.json"
  RUN_DIR="${VDC_PK2_RUN:-$VDC_PRIVATE_ROOT/runs/pk2_training_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
  mkdir -p "$RUN_DIR" runs
  RUN_DIR="$(realpath "$RUN_DIR")"
  # Existing run is only resumed explicitly; do not overwrite an active run.
  if [[ -f "$RUN_DIR/run_manifest.json" && "${VDC_RESUME:-0}" != 1 ]]; then
    echo 'Existing run: set VDC_RESUME=1 for signature-checked resume'; exit 1
  fi
  printf '%s\n' "$RUN_DIR" > runs/LATEST_PK2_TRAINING.txt
  printf '%s\n' "$RUN_DIR" > "$VDC_PRIVATE_ROOT/LATEST_PK2_TRAINING.txt"
  SESSION="vdc_pk2_train_$(date -u +%H%M%S)_$$"
  screen -dmS "$SESSION" bash "$REPO_DIR/scripts/launch_pk2_training.sh" --worker "$RUN_DIR"
  printf 'SCREEN=%s\nRUN_DIR=%s\n' "$SESSION" "$RUN_DIR"
  exit 0
fi
RUN_DIR="$2"
exec >> "$RUN_DIR/console.log" 2>&1
trap 'code=$?; printf "TASK_EXIT_CODE=%s\n" "$code"; printf "%s\n" "$code" > "$RUN_DIR/exit_code.txt"' EXIT
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
echo 'PK2 full development queue; 171 logical slots. No reserved queries. PK1 service unchanged.'
args=(--run "$RUN_DIR" --private-root "$VDC_PRIVATE_ROOT" --pk1-run "$VDC_PK1_RUN"
  --acquire-run "$VDC_ACQUIRE_RUN" --public-raw "$VDC_PK2_RAW" --model-path "$VDC_QWEN_MODEL")
if [[ "${VDC_RESUME:-0}" == 1 ]]; then args+=(--resume); fi
"$PY" -u scripts/run_pk2.py "${args[@]}"

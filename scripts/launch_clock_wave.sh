#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
RELEASE="$PWD"
: "${VDC_PYTHON:?Set the existing CUDA Python executable}"
: "${VDC_PRIVATE_ROOT:?Set private data root}"
: "${VDC_CW_SOURCE:?Set completed corrected PK2 run}"
export PYTHONPATH="$RELEASE/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${1:-}" != --worker ]]; then
  command -v screen >/dev/null
  "$VDC_PYTHON" -c 'import vdc,torch,h5py,transformers,peft; print(torch.__version__, torch.cuda.is_available())'
  RUN="${VDC_CW_RUN:-$VDC_PRIVATE_ROOT/runs/clock_wave_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
  mkdir -p "$RUN"
  printf '%s\n' "$RUN" > "$VDC_PRIVATE_ROOT/LATEST_CLOCK_WAVE.txt"
  printf '%s\n' "$RELEASE" > "$VDC_PRIVATE_ROOT/LATEST_CLOCK_WAVE_RELEASE.txt"
  SESSION="vdc_clock_wave_$(date -u +%H%M%S)_$$"
  screen -dmS "$SESSION" bash "$RELEASE/scripts/launch_clock_wave.sh" --worker "$RUN"
  printf 'SCREEN=%s\nRUN=%s\nRELEASE=%s\n' "$SESSION" "$RUN" "$RELEASE"
  exit 0
fi
RUN="$2"
exec >> "$RUN/console.log" 2>&1
trap 'code=$?; printf "TASK_EXIT_CODE=%s\n" "$code"; printf "%s\n" "$code" > "$RUN/exit_code.txt"' EXIT
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
code=0
"$VDC_PYTHON" -u scripts/run_clock_wave.py --run "$RUN" --private-root "$VDC_PRIVATE_ROOT" --source "$VDC_CW_SOURCE" || code=$?
if [[ -f "$RUN/results_snapshot.json" ]]; then
  "$VDC_PYTHON" scripts/serve_clock_wave.py --run "$RUN" --verify
fi
exit "$code"

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
RELEASE="$PWD"
export VDC_PRIVATE_ROOT="${VDC_PRIVATE_ROOT:-/root/autodl-tmp/vdc-private}"
export VDC_PYTHON="${VDC_PYTHON:-/root/autodl-tmp/VirtualDiapauseCell/.venv-autodl-vdc/bin/python}"
export PYTHONPATH="$RELEASE/src"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
if [[ "${1:-}" != --worker ]]; then
  command -v screen >/dev/null
  STAGE="${VDC_STAGE_RUN:-$(cat "$VDC_PRIVATE_ROOT/LATEST_STAGE.txt")}"
  test -f "$STAGE/stage_snapshot.json"
  "$VDC_PYTHON" -c 'import torch,anndata,matplotlib,multipart,transformers,peft; print(torch.__version__,torch.cuda.is_available())'
  if [[ -f "$VDC_PRIVATE_ROOT/LATEST_INT1.txt" && -z "${VDC_INT1_RUN:-}" ]]; then
    printf 'Existing INT1 run: %s\nInspect status; explicitly set VDC_INT1_RUN to resume it.\n' "$(cat "$VDC_PRIVATE_ROOT/LATEST_INT1.txt")" >&2; exit 2
  fi
  RUN="${VDC_INT1_RUN:-$VDC_PRIVATE_ROOT/runs/int1_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
  mkdir -p "$RUN"
  if [[ -f "$RUN/RELEASE.txt" && "$(cat "$RUN/RELEASE.txt")" != "$RELEASE" ]]; then
    printf 'Resume using the original RELEASE.txt path; changed code requires a new run.\n' >&2; exit 2
  fi
  printf '%s\n' "$RELEASE" > "$RUN/RELEASE.txt"
  screen -dmS "vdc_int1_$(date -u +%H%M%S)_$$" bash "$RELEASE/scripts/launch_int1.sh" --worker "$STAGE" "$RUN"
  printf '%s\n' "$RUN" > "$VDC_PRIVATE_ROOT/LATEST_INT1.txt"
  printf 'RUN=%s\nRELEASE=%s\n' "$RUN" "$RELEASE"
  exit 0
fi
STAGE="$2"; RUN="$3"
exec >> "$RUN/console.log" 2>&1
trap 'code=$?; printf "TASK_EXIT_CODE=%s\n" "$code"; printf "%s\n" "$code" > "$RUN/exit_code.txt"' EXIT
code=0
"$VDC_PYTHON" -u scripts/run_int1.py --stage "$STAGE" --run "$RUN" || code=$?
"$VDC_PYTHON" scripts/pack_int1_report.py --run "$RUN"
exit "$code"

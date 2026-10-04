#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
RELEASE="$PWD"
: "${VDC_PYTHON:?Set existing CUDA Python}"
: "${VDC_APPLICATION_SOURCE:?Set completed PK2 repair run}"
: "${VDC_PRIVATE_ROOT:?Set private root}"
export PYTHONPATH="$RELEASE/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${1:-}" != --worker ]]; then
  command -v screen >/dev/null
  RUN="${VDC_APPLICATION_RUN:-$VDC_PRIVATE_ROOT/runs/application_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
  mkdir -p "$RUN"
  printf '%s\n' "$RUN" > "$VDC_PRIVATE_ROOT/LATEST_APPLICATION.txt"
  printf '%s\n' "$RELEASE" > "$VDC_PRIVATE_ROOT/LATEST_APPLICATION_RELEASE.txt"
  SESSION="vdc_application_$(date -u +%H%M%S)_$$"
  screen -dmS "$SESSION" bash "$RELEASE/scripts/launch_application.sh" --worker "$RUN"
  printf 'SCREEN=%s\nRUN=%s\n' "$SESSION" "$RUN"
  exit 0
fi
RUN="$2"
exec >> "$RUN/console.log" 2>&1
trap 'code=$?; printf "TASK_EXIT_CODE=%s\n" "$code"; printf "%s\n" "$code" > "$RUN/exit_code.txt"' EXIT
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
args=(--run "$RUN" --source "$VDC_APPLICATION_SOURCE" --private-root "$VDC_PRIVATE_ROOT")
if [[ -n "${VDC_APPLICATION_SC:-}" ]]; then args+=(--supplement "$VDC_APPLICATION_SC"); fi
echo 'Application: fixed seed 42, no state/Qwen retraining. Freeze first, then authorized retrospective queries.'
code=0
"$VDC_PYTHON" -u scripts/run_application.py "${args[@]}" || code=$?
if [[ -f "$RUN/application_snapshot.json" ]]; then
  "$VDC_PYTHON" scripts/serve_application.py --run "$RUN" --verify
  screen -L -Logfile "$RUN/service.log" -dmS "vdc_application_results_$$" "$VDC_PYTHON" scripts/serve_application.py --run "$RUN" --port "${VDC_APPLICATION_PORT:-8768}"
  "$VDC_PYTHON" scripts/serve_application.py --run "$RUN" --port "${VDC_APPLICATION_PORT:-8768}" --check-running
fi
exit "$code"

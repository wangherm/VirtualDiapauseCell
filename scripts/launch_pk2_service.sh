#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_DIR="$PWD"
PY="${VDC_PYTHON:-$REPO_DIR/.venv-autodl-vdc/bin/python}"
test -x "$PY" || { echo 'Set VDC_PYTHON to the existing VDC environment'; exit 1; }
export VDC_PRIVATE_ROOT="${VDC_PRIVATE_ROOT:-/root/autodl-tmp/vdc-private}"
if [[ "${1:-}" != '--worker' ]]; then
  command -v screen >/dev/null || { echo 'Install screen first'; exit 1; }
  if [[ -z "${VDC_PK1_RUN:-}" ]]; then
    test -f runs/LATEST_PK1.txt || { echo 'Set VDC_PK1_RUN to the completed PK1 run'; exit 1; }
    export VDC_PK1_RUN="$(cat runs/LATEST_PK1.txt)"
  fi
  export VDC_PK1_RUN="$(realpath "$VDC_PK1_RUN")"
  test -f "$VDC_PK1_RUN/module_status.json"
  test -f "$VDC_PRIVATE_ROOT/sample_roles.json"
  RUN_DIR="$VDC_PRIVATE_ROOT/services/pk2_$(date -u +%Y%m%dT%H%M%SZ)_$$"
  mkdir -p "$RUN_DIR" runs
  RUN_DIR="$(realpath "$RUN_DIR")"
  printf '%s\n' "$RUN_DIR" > runs/LATEST_PK2_SERVICE.txt
  SESSION="vdc_pk2_web_$(date -u +%H%M%S)_$$"
  screen -dmS "$SESSION" bash "$REPO_DIR/scripts/launch_pk2_service.sh" --worker "$RUN_DIR"
  printf 'SCREEN=%s\nRUN_DIR=%s\nLoopback URL: http://127.0.0.1:%s\n' "$SESSION" "$RUN_DIR" "${VDC_SERVICE_PORT:-8765}"
  exit 0
fi
RUN_DIR="$2"
exec >> "$RUN_DIR/console.log" 2>&1
trap 'code=$?; printf "TASK_EXIT_CODE=%s\n" "$code"; printf "%s\n" "$code" > "$RUN_DIR/exit_code.txt"' EXIT
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PORT="${VDC_SERVICE_PORT:-8765}"
"$PY" -c 'import socket,sys; s=socket.socket(); s.bind(("127.0.0.1",int(sys.argv[1]))); s.close()' "$PORT"
"$PY" scripts/serve_pk2.py freeze --pk1-run "$VDC_PK1_RUN" --roles "$VDC_PRIVATE_ROOT/sample_roles.json" --out "$RUN_DIR/snapshot"
"$PY" scripts/verify_pk2_service.py --snapshot "$RUN_DIR/snapshot" --out "$RUN_DIR/http_verification"
printf '%s\n' "$PORT" > "$RUN_DIR/port.txt"
echo 'HTTP reload checks passed. Starting frozen development service; not new PK2 training.'
"$PY" scripts/serve_pk2.py serve --snapshot "$RUN_DIR/snapshot" --port "$PORT"

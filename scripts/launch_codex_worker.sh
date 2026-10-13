#!/usr/bin/env bash
set -euo pipefail
RELEASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${VDC_PRIVATE_ROOT:-/root/autodl-tmp/vdc-private}"
PY="${VDC_PYTHON:-/root/autodl-tmp/VirtualDiapauseCell/.venv-autodl-vdc/bin/python}"
PARENT="${1:?Pass the stopped parent run path}"
RUN="${VDC_CODEX_RUN:-$ROOT/runs/research_codex_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
test -x "$PY"
command -v screen >/dev/null
"$PY" "$RELEASE/scripts/research_worker.py" init --parent "$PARENT" --run "$RUN" --private-root "$ROOT"
printf '%s\n' "$RUN" > "$ROOT/LATEST_CODEX_WORKER.txt"
cat > "$RUN/worker.sh" <<'SH'
#!/usr/bin/env bash
set -uo pipefail
PY="$1"; RUN="$2"; ROOT="$3"
"$PY" "$RUN/release/scripts/research_worker.py" serve --run "$RUN" --private-root "$ROOT" >> "$RUN/console.log" 2>&1
RC=$?
printf 'WORKER_EXIT_CODE=%s\n' "$RC" >> "$RUN/console.log"
exit "$RC"
SH
SCREEN="vdc_codex_worker_$(date -u +%H%M%S)_$$"
screen -dmS "$SCREEN" bash "$RUN/worker.sh" "$PY" "$RUN" "$ROOT"
printf 'SCREEN=%s\nRUN=%s\nReady for LOCAL Codex; no autonomous research started by this launcher.\n' "$SCREEN" "$RUN"

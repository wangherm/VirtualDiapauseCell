#!/usr/bin/env bash
# Audit/export first. Only --start-missing allows submitting a targeted run, never PK2.
set -euo pipefail
cd "$(dirname "$0")/.."
RELEASE="$PWD"
export VDC_SOURCE_REPO="${VDC_SOURCE_REPO:-/root/autodl-tmp/VirtualDiapauseCell}"
export VDC_PRIVATE_ROOT="${VDC_PRIVATE_ROOT:-/root/autodl-tmp/vdc-private}"
export VDC_PYTHON="${VDC_PYTHON:-$VDC_SOURCE_REPO/.venv-autodl-vdc/bin/python}"
export PYTHONPATH="$RELEASE/src${PYTHONPATH:+:$PYTHONPATH}"
AUDIT="$VDC_PRIVATE_ROOT/runs/cw_run_audit_$(date -u +%Y%m%dT%H%M%SZ)_$$.json"
"$VDC_PYTHON" scripts/audit_clock_wave_runs.py --private-root "$VDC_PRIVATE_ROOT" --extra-run-root "$VDC_SOURCE_REPO/runs" --out "$AUDIT" --export-found
ACTION="$("$VDC_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["next_action"])' "$AUDIT")"
if [[ "$ACTION" != targeted_revision_not_found ]]; then
  printf 'AUDIT_ACTION=%s\nNo new training submitted. Read %s\n' "$ACTION" "$AUDIT"
  exit 0
fi
if [[ "${1:-}" != --start-missing ]]; then
  printf 'No targeted revision found in audited roots. Audit only; no training submitted.\n'
  exit 0
fi
# Choose a fully inventoried original CW1 by configuration, never a stale shell RUN variable.
export VDC_CW_REVISION_SOURCE="$("$VDC_PYTHON" -c 'import json,sys; a=json.load(open(sys.argv[1])); r=[r for r in a["runs"] if r.get("protocol",{}).get("protocol")=="VDC_CW1_development_clock_identity_wave" and r.get("queue_status")=="completed_current_scope" and r.get("task_total",0)>0 and not r.get("errors") and not r.get("categories",{}).get("unfinished")]; assert r,"No verified complete original CW1; inspect audit"; print(sorted(r,key=lambda x:x["run"])[-1]["run"])' "$AUDIT")"
export VDC_CW_SOURCE="$("$VDC_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"])' "$VDC_CW_REVISION_SOURCE/config.json")"
export VDC_CW_RUN="$VDC_PRIVATE_ROOT/runs/clock_wave_revision_$(date -u +%Y%m%dT%H%M%SZ)_$$"
bash scripts/launch_clock_wave.sh

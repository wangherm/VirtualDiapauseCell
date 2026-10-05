#!/usr/bin/env bash
# The recommended stage entry. Existing runner, status, packager and browser remain canonical.
set -euo pipefail
cd "$(dirname "$0")/.."
RELEASE="$PWD"
export VDC_PRIVATE_ROOT="${VDC_PRIVATE_ROOT:-/root/autodl-tmp/vdc-private}"
export VDC_PYTHON="${VDC_PYTHON:-/root/autodl-tmp/VirtualDiapauseCell/.venv-autodl-vdc/bin/python}"
export PYTHONPATH="$RELEASE/src${PYTHONPATH:+:$PYTHONPATH}"
ACTION="${1:-status}"
case "$ACTION" in
  start)
    if [[ -f "$VDC_PRIVATE_ROOT/LATEST_STAGE.txt" ]]; then
      printf 'An existing stage run is recorded: %s\nUse status/resume/pack; do not submit a duplicate stage queue.\n' "$(cat "$VDC_PRIVATE_ROOT/LATEST_STAGE.txt")" >&2
      exit 2
    fi
    if [[ -z "${VDC_STAGE_SOURCE:-}" ]]; then
      POINTER="$VDC_PRIVATE_ROOT/LATEST_CLOCK_WAVE_REVISION.txt"
      [[ -f "$POINTER" ]] || POINTER="$VDC_PRIVATE_ROOT/LATEST_CLOCK_WAVE.txt"
      export VDC_STAGE_SOURCE="$(cat "$POINTER")"
    fi
    "$VDC_PYTHON" -c 'import json,sys; c=json.load(open(sys.argv[1])); assert c["protocol"]["identity_protocol"]=="short_slots_v2", "Stage source must be the existing v2 revision; do not use an old CW1 or stage run"' "$VDC_STAGE_SOURCE/config.json"
    export VDC_CW_SOURCE="$("$VDC_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"])' "$VDC_STAGE_SOURCE/config.json")"
    unset VDC_CW_REVISION_SOURCE
    export VDC_CW_RUN="$VDC_PRIVATE_ROOT/runs/cw_stage_$(date -u +%Y%m%dT%H%M%SZ)_$$"
    bash scripts/launch_clock_wave.sh
    ;;
  status|pack|browse|resume)
    RUN="${VDC_STAGE_RUN:-$(cat "$VDC_PRIVATE_ROOT/LATEST_STAGE.txt")}"
    if [[ -f "$RUN/runtime_identity.json" ]]; then
      ORIGINAL_RELEASE="$("$VDC_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["release"])' "$RUN/runtime_identity.json")"
    else
      ORIGINAL_RELEASE="$(cat "$VDC_PRIVATE_ROOT/LATEST_STAGE_RELEASE.txt")"
    fi
    export PYTHONPATH="$ORIGINAL_RELEASE/src"
    case "$ACTION" in
      status) "$VDC_PYTHON" "$ORIGINAL_RELEASE/scripts/clock_wave_status.py" --run "$RUN" ;;
      pack) "$VDC_PYTHON" "$ORIGINAL_RELEASE/scripts/pack_clock_wave_report.py" --run "$RUN" --require-revision ;;
      browse) "$VDC_PYTHON" "$ORIGINAL_RELEASE/scripts/serve_clock_wave.py" --run "$RUN" --port "${VDC_STAGE_PORT:-8769}" ;;
      resume)
        export VDC_STAGE_SOURCE="$("$VDC_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["revision_parent"]["path"])' "$RUN/config.json")"
        export VDC_CW_SOURCE="$("$VDC_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"])' "$RUN/config.json")"
        export VDC_CW_RUN="$RUN"
        unset VDC_CW_REVISION_SOURCE
        bash "$ORIGINAL_RELEASE/scripts/launch_clock_wave.sh"
        ;;
    esac
    ;;
  *) printf 'Usage: bash scripts/vdc_stage.sh {start|status|resume|pack|browse}\n' >&2; exit 2 ;;
esac

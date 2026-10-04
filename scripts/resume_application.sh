#!/usr/bin/env bash
set -euo pipefail
: "${VDC_PRIVATE_ROOT:?Set private root}"
RUN="${VDC_APPLICATION_RUN:-$(cat "$VDC_PRIVATE_ROOT/LATEST_APPLICATION.txt")}"
RELEASE="$(cat "$VDC_PRIVATE_ROOT/LATEST_APPLICATION_RELEASE.txt")"
: "${VDC_PYTHON:?Set original CUDA Python}"
export VDC_APPLICATION_RUN="$RUN"
export VDC_APPLICATION_SOURCE="$("$VDC_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"])' "$RUN/config.json")"
export VDC_APPLICATION_SC="$("$VDC_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["supplement"] or "")' "$RUN/config.json")"
bash "$RELEASE/scripts/launch_application.sh"

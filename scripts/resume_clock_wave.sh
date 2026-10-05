#!/usr/bin/env bash
set -euo pipefail
: "${VDC_PRIVATE_ROOT:?Set private root}"
: "${VDC_PYTHON:?Set original CUDA Python}"
export VDC_CW_RUN="${VDC_CW_RUN:-$(cat "$VDC_PRIVATE_ROOT/LATEST_CLOCK_WAVE.txt")}"
RELEASE="$(cat "$VDC_PRIVATE_ROOT/LATEST_CLOCK_WAVE_RELEASE.txt")"
export VDC_CW_SOURCE="$("$VDC_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"])' "$VDC_CW_RUN/config.json")"
bash "$RELEASE/scripts/launch_clock_wave.sh"

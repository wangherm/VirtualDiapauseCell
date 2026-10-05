#!/usr/bin/env bash
set -euo pipefail
SOURCE="${VDC_SOURCE_REPO:-$PWD}"
export VDC_PYTHON="${VDC_PYTHON:-$SOURCE/.venv-autodl-vdc/bin/python}"
export VDC_PRIVATE_ROOT="${VDC_PRIVATE_ROOT:-/root/autodl-tmp/vdc-private}"
export VDC_CW_REVISION_SOURCE="${VDC_CW_REVISION_SOURCE:-$(cat "$VDC_PRIVATE_ROOT/LATEST_CLOCK_WAVE.txt")}"
test -f "$VDC_CW_REVISION_SOURCE/config.json"
export VDC_CW_REVISION_SOURCE="$("$VDC_PYTHON" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); c=json.load(open(p/"config.json")); print(c.get("revision_parent",{}).get("path",str(p)))' "$VDC_CW_REVISION_SOURCE")"
export VDC_CW_SOURCE="$("$VDC_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"])' "$VDC_CW_REVISION_SOURCE/config.json")"
export VDC_CW_RUN="${VDC_CW_RUN:-$VDC_PRIVATE_ROOT/runs/clock_wave_revision_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
bash "$(dirname "$0")/install_clock_wave_release.sh"

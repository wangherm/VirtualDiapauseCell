#!/usr/bin/env bash
set -euo pipefail
SOURCE="${VDC_SOURCE_REPO:-$PWD}"
export VDC_PYTHON="${VDC_PYTHON:-$SOURCE/.venv-autodl-vdc/bin/python}"
export VDC_PRIVATE_ROOT="${VDC_PRIVATE_ROOT:-/root/autodl-tmp/vdc-private}"
test -x "$VDC_PYTHON"
if [[ -z "${VDC_CW_SOURCE:-}" ]]; then
  APP_RUN="$(cat "$VDC_PRIVATE_ROOT/LATEST_APPLICATION.txt")"
  export VDC_CW_SOURCE="$("$VDC_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"])' "$APP_RUN/config.json")"
fi
test -f "$VDC_CW_SOURCE/config.json"
REV="$(git -C "$SOURCE" rev-parse origin/main)"
RELEASE="$(dirname "$SOURCE")/vdc-releases/clock_wave_${REV:0:12}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$RELEASE"
git -C "$SOURCE" archive "$REV" | tar -x -C "$RELEASE"
printf '%s\n' "$REV" > "$RELEASE/RELEASE_COMMIT.txt"
bash "$RELEASE/scripts/launch_clock_wave.sh"

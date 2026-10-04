#!/usr/bin/env bash
set -euo pipefail
SOURCE="${VDC_SOURCE_REPO:-$PWD}"
: "${VDC_APPLICATION_SOURCE:?Set completed PK2 repair run}"
export VDC_PYTHON="${VDC_PYTHON:-$SOURCE/.venv-autodl-vdc/bin/python}"
test -x "$VDC_PYTHON"
export VDC_PRIVATE_ROOT="${VDC_PRIVATE_ROOT:-$("$VDC_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["private_root"])' "$VDC_APPLICATION_SOURCE/config.json")}"
if [[ -z "${VDC_APPLICATION_SC:-}" && -f "$VDC_PRIVATE_ROOT/application_sc/manifest.json" ]]; then
  export VDC_APPLICATION_SC="$VDC_PRIVATE_ROOT/application_sc"
fi
REV="$(git -C "$SOURCE" rev-parse origin/main)"
RELEASE="$(dirname "$SOURCE")/vdc-releases/application_${REV:0:12}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$RELEASE"
git -C "$SOURCE" archive "$REV" | tar -x -C "$RELEASE"
printf '%s\n' "$REV" > "$RELEASE/RELEASE_COMMIT.txt"
bash "$RELEASE/scripts/launch_application.sh"

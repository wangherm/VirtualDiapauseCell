#!/usr/bin/env bash
# Bootstrap from the original checkout without overwriting it or changing its venv.
set -euo pipefail
SOURCE="${VDC_SOURCE_REPO:-$PWD}"
: "${VDC_PK2_REPAIR_FROM:?Set VDC_PK2_REPAIR_FROM to the finished original PK2 training run}"
export VDC_PYTHON="${VDC_PYTHON:-$SOURCE/.venv-autodl-vdc/bin/python}"
test -x "$VDC_PYTHON"
SETTINGS="$("$VDC_PYTHON" - "$VDC_PK2_REPAIR_FROM" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]).resolve()
c=json.loads((p/'config.json').read_text())
for k in ('private_root','pk1_run','acquire_run','public_raw','model_path'):
    value=c[k]
    if not value or '\n' in value or '\r' in value or not Path(value).exists():
        raise ValueError('Missing or invalid original input path: '+k)
    print(value)
PY
)"
mapfile -t REPAIR_SETTINGS <<< "$SETTINGS"
[[ ${#REPAIR_SETTINGS[@]} == 5 ]]
export VDC_PRIVATE_ROOT="${REPAIR_SETTINGS[0]}" VDC_PK1_RUN="${REPAIR_SETTINGS[1]}"
export VDC_ACQUIRE_RUN="${REPAIR_SETTINGS[2]}" VDC_PK2_RAW="${REPAIR_SETTINGS[3]}" VDC_QWEN_MODEL="${REPAIR_SETTINGS[4]}"
export VDC_PK2_SERVICE_PORT="${VDC_PK2_SERVICE_PORT:-8767}"
export VDC_PK2_RUN="${VDC_PK2_RUN:-$VDC_PRIVATE_ROOT/runs/pk2_repair_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
# The same canonical release installer/runner handles both full and targeted queues.
git -C "$SOURCE" show origin/main:scripts/install_pk2_release.sh | bash

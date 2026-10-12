#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${VDC_PYTHON:-$PWD/.venv-autodl-vdc/bin/python}"
test -x "$PY"
if ! command -v cc >/dev/null; then
  apt-get update
  apt-get install -y gcc libc6-dev
fi
# Reuse the existing CUDA Torch; add parsers without installing/replacing torch.
"$PY" -m pip install 'pypdf>=5,<6' 'pandas>=2.2,<3' 'scikit-learn>=1.6,<2' \
  'anndata>=0.10,<0.13' 'openpyxl>=3.1,<4' 'xlrd>=2,<3' 'biopython>=1.85,<2'
export PYTHONPATH="$PWD/src"
export VDC_R2_RUNTIME_ROOT="${VDC_R2_RUNTIME_ROOT:-${VDC_PRIVATE_ROOT:-/root/autodl-tmp/vdc-private}/r2_runtime_devices_v1}"
if [[ ! -f "$VDC_R2_RUNTIME_ROOT/runtime.json" ]]; then
  "$PY" -c 'import os; from pathlib import Path; from vdc.research_r2_linux import build_runtime; print("Runtime snapshot:",build_runtime(Path.cwd(),os.environ["VDC_R2_RUNTIME_ROOT"])["runtime_id"])'
fi
"$PY" -c 'import os; from pathlib import Path; from vdc.io import read_json; m=read_json(Path(os.environ["VDC_R2_RUNTIME_ROOT"])/"runtime.json"); assert m.get("executor_version")=="r2_device_choice_v1", "Old runtime: select a new VDC_R2_RUNTIME_ROOT and repeat setup"'
args=()
POLICY="${VDC_R2_DEVICE_POLICY:-ai}"
case "$POLICY" in ai) args+=(--gpu);; cpu) ;; *) echo 'VDC_R2_DEVICE_POLICY must be ai or cpu'; exit 2;; esac
"$PY" scripts/smoke_research_r2.py --backend linux_landlock_chroot --runtime-root "$VDC_R2_RUNTIME_ROOT" \
  --out "${VDC_PRIVATE_ROOT:-/root/autodl-tmp/vdc-private}/runs/r2_runtime_smoke_$(date -u +%Y%m%dT%H%M%SZ)_$$" "${args[@]}"

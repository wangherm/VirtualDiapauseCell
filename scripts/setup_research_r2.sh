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
export VDC_R2_RUNTIME_ROOT="${VDC_R2_RUNTIME_ROOT:-${VDC_PRIVATE_ROOT:-/root/autodl-tmp/vdc-private}/r2_runtime}"
if [[ ! -f "$VDC_R2_RUNTIME_ROOT/runtime.json" ]]; then
  "$PY" -c 'import os; from pathlib import Path; from vdc.research_r2_linux import build_runtime; print("Runtime snapshot:",build_runtime(Path.cwd(),os.environ["VDC_R2_RUNTIME_ROOT"])["runtime_id"])'
fi
args=()
if [[ "${VDC_R2_GPU:-0}" == 1 ]]; then args+=(--gpu); fi
"$PY" scripts/smoke_research_r2.py --backend linux_landlock_chroot --runtime-root "$VDC_R2_RUNTIME_ROOT" \
  --out "${VDC_PRIVATE_ROOT:-/root/autodl-tmp/vdc-private}/runs/r2_runtime_smoke_$(date -u +%Y%m%dT%H%M%SZ)_$$" "${args[@]}"

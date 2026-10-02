#!/usr/bin/env bash
# Explicit optional setup, separate from launcher: safe for existing CUDA environments.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .venv-alpha/bin/python ]]; then python -m venv .venv-alpha; fi
PY="$PWD/.venv-alpha/bin/python"
"$PY" -m pip install --upgrade pip
# Official PyTorch 2.8.0 CUDA 12.8 wheel supports the Blackwell training setup.
# It is installed only in this new dedicated environment, not the archived project.
"$PY" -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install -e '.[alpha]'
"$PY" -c 'import torch; assert torch.cuda.is_available(); x=torch.ones((32,32),device="cuda",dtype=torch.bfloat16,requires_grad=True); (x@x).sum().backward(); print(torch.__version__,torch.cuda.get_device_name(0))'
printf 'Environment ready. Start with:\nVDC_PYTHON="%s" bash scripts/launch_alpha.sh\n' "$PY"

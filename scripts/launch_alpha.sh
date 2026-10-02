#!/usr/bin/env bash
# Start from a JupyterLab terminal. All long work lives inside screen.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT_DIR="$PWD"
if [[ "${1:-}" != "--worker" ]]; then
  command -v screen >/dev/null || { echo 'Install screen first: apt-get update && apt-get install -y screen'; exit 1; }
  RUN_DIR="${VDC_RUN_DIR:-$ROOT_DIR/runs/alpha_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
  mkdir -p "$RUN_DIR"
  printf '%s\n' "$RUN_DIR" > runs/LATEST_ALPHA.txt
  SESSION="vdc_alpha_$(date -u +%H%M%S)_$$"
  screen -dmS "$SESSION" bash "$ROOT_DIR/scripts/launch_alpha.sh" --worker "$RUN_DIR"
  printf 'SCREEN=%s\nRUN_DIR=%s\nProgress: tail -n 50 -F "%s/console.log"\n' "$SESSION" "$RUN_DIR" "$RUN_DIR"
  exit 0
fi
RUN_DIR="$2"
exec > >(tee -a "$RUN_DIR/console.log") 2>&1
trap 'code=$?; printf "TASK_EXIT_CODE=%s\n" "$code"; date -u +%FT%TZ > "$RUN_DIR/finished_utc.txt"' EXIT
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
echo 'All-Module Alpha — development only; overall partial is expected if capabilities lack labels.'
date -u
nvidia-smi
# Reuse a working CUDA environment by explicit choice; never replace an existing CUDA build with CPU torch.
PY="${VDC_PYTHON:-$ROOT_DIR/.venv-autodl-vdc/bin/python}"
if [[ ! -x "$PY" ]]; then
  echo 'No existing VDC environment. Create a dedicated one with scripts/setup_alpha.sh, then rerun.'
  exit 1
fi
"$PY" -m pip install -e '.[alpha]'
"$PY" -m pip freeze > "$RUN_DIR/pip_freeze.txt"
"$PY" -c 'import torch; print("torch",torch.__version__,"CUDA",torch.version.cuda); assert torch.cuda.is_available(), "No working CUDA torch. See docs/ALL_MODULE_ALPHA_CN.md; no CPU GPU-smoke fallback."; x=torch.ones((32,32),device="cuda",dtype=torch.bfloat16,requires_grad=True); (x@x).sum().backward(); torch.cuda.synchronize(); print(torch.cuda.get_device_name(0), "BF16 matmul/backward passed")'
REV=cdbee75f17c01a7cc42f958dc650907174af0554
MODEL="${VDC_MODEL_PATH:-$HF_HOME/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/$REV}"
if [[ ! -f "$MODEL/model.safetensors.index.json" ]]; then
  echo "Pinned base snapshot missing: $MODEL"
  echo 'Set VDC_MODEL_PATH to the verified Qwen base snapshot. No archived FactorBridge adapter is used.'
  exit 1
fi
ARGS=(--run "$RUN_DIR" --download --gpu --model-path "$MODEL")
if [[ -f "$RUN_DIR/run_manifest.json" ]]; then ARGS+=(--resume); fi
"$PY" scripts/run_all_modules.py "${ARGS[@]}"

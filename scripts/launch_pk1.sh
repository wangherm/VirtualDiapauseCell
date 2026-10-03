#!/usr/bin/env bash
# A private screen-backed training run, not a public deployment.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_DIR="$PWD"
export VDC_PRIVATE_ROOT="${VDC_PRIVATE_ROOT:-/root/autodl-tmp/vdc-private}"
if [[ "${1:-}" != "--worker" ]]; then
  command -v screen >/dev/null || { echo 'Install screen: apt-get update && apt-get install -y screen'; exit 1; }
  test -f "$VDC_PRIVATE_ROOT/sample_roles.json" || { echo 'Approved private preparation package has not been extracted'; exit 1; }
  if [[ -z "${VDC_ALPHA_RUN:-}" ]]; then
    test -f runs/LATEST_ALPHA.txt || { echo 'Set VDC_ALPHA_RUN to the previous successful Alpha run (including actual adapter weights)'; exit 1; }
    export VDC_ALPHA_RUN="$(cat runs/LATEST_ALPHA.txt)"
  fi
  export VDC_ALPHA_RUN="$(realpath "$VDC_ALPHA_RUN")"
  test -f "$VDC_ALPHA_RUN/module_status.json" || { echo 'Alpha run not found'; exit 1; }
  RUN_DIR="${VDC_RUN_DIR:-$VDC_PRIVATE_ROOT/runs/pk1_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
  mkdir -p "$RUN_DIR" runs
  RUN_DIR="$(realpath "$RUN_DIR")"
  printf '%s\n' "$RUN_DIR" > runs/LATEST_PK1.txt
  SESSION="vdc_pk1_$(date -u +%H%M%S)_$$"
  screen -dmS "$SESSION" bash "$REPO_DIR/scripts/launch_pk1.sh" --worker "$RUN_DIR"
  printf 'SCREEN=%s\nRUN_DIR=%s\nProgress: python scripts/pk1_status.py\n' "$SESSION" "$RUN_DIR"
  exit 0
fi
RUN_DIR="$2"
exec > >(tee -a "$RUN_DIR/console.log") 2>&1
trap 'code=$?; printf "TASK_EXIT_CODE=%s\n" "$code"; printf "%s\n" "$code" > "$RUN_DIR/exit_code.txt"; date -u +%FT%TZ > "$RUN_DIR/finished_utc.txt"' EXIT
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
export VDC_ROLE_MANIFEST="$VDC_PRIVATE_ROOT/sample_roles.json"
PY="${VDC_PYTHON:-$REPO_DIR/.venv-autodl-vdc/bin/python}"
test -x "$PY" || { echo 'Set VDC_PYTHON to your existing working CUDA environment'; exit 1; }
"$PY" -m pip install -e '.[pk1]'
"$PY" -m pip freeze > "$RUN_DIR/pip_freeze.txt"
"$PY" scripts/prepare_pk1_private.py verify --roles "$VDC_ROLE_MANIFEST"
"$PY" -c 'import torch; assert torch.cuda.is_available(), "Working CUDA environment required"; x=torch.ones((32,32),device="cuda",dtype=torch.bfloat16,requires_grad=True); (x@x).sum().backward(); torch.cuda.synchronize(); print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0),"BF16 forward/backward passed")'
REV=cdbee75f17c01a7cc42f958dc650907174af0554
MODEL="${VDC_MODEL_PATH:-$HF_HOME/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/$REV}"
test -f "$MODEL/model.safetensors.index.json" || { echo 'Set VDC_MODEL_PATH to the verified Qwen base snapshot'; exit 1; }
echo 'PK1 first numerical training: 3 seeds, K0-K5; existing Alpha adapter reused; no new Qwen training or private heldout queries.'
ARGS=(--run "$RUN_DIR" --private-root "$VDC_PRIVATE_ROOT" --alpha-run "$VDC_ALPHA_RUN" --model-path "$MODEL" --download)
if [[ -f "$RUN_DIR/run_manifest.json" ]]; then ARGS+=(--resume); fi
"$PY" scripts/run_pk1.py "${ARGS[@]}"

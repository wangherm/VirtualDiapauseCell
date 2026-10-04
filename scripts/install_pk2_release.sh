#!/usr/bin/env bash
# Run in the existing checkout, or via: git show origin/main:scripts/install_pk2_release.sh | bash
set -euo pipefail
SOURCE="${VDC_SOURCE_REPO:-$PWD}"
test -d "$SOURCE/.git" || { echo 'Run from the original Git checkout'; exit 1; }
: "${VDC_PRIVATE_ROOT:?Set VDC_PRIVATE_ROOT}"
: "${VDC_PK1_RUN:?Set VDC_PK1_RUN}"
export VDC_PYTHON="${VDC_PYTHON:-$SOURCE/.venv-autodl-vdc/bin/python}"
test -x "$VDC_PYTHON"
export VDC_ACQUIRE_RUN="${VDC_ACQUIRE_RUN:-$(cat "$SOURCE/runs/LATEST_PK2_ACQUIRE.txt")}"
export VDC_PK2_RAW="${VDC_PK2_RAW:-$SOURCE/data/raw/pk2_public}"
export VDC_QWEN_MODEL="${VDC_QWEN_MODEL:-/root/autodl-tmp/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}"
REVISION="$(git -C "$SOURCE" rev-parse origin/main)"
RELEASE="${VDC_RELEASE_PARENT:-$(dirname "$SOURCE")/vdc-releases}/pk2_${REVISION:0:12}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$RELEASE"
git -C "$SOURCE" archive "$REVISION" | tar -x -C "$RELEASE"
mkdir -p "$RELEASE/data/raw"
for cache in GSE288723 pk2_annotations; do
  if [[ -d "$SOURCE/data/raw/$cache" ]]; then ln -s "$SOURCE/data/raw/$cache" "$RELEASE/data/raw/$cache"; fi
done
printf '%s\n' "$REVISION" > "$RELEASE/RELEASE_COMMIT.txt"
mkdir -p "$VDC_PRIVATE_ROOT"
printf '%s\n' "$RELEASE" > "$VDC_PRIVATE_ROOT/LATEST_PK2_RELEASE.txt"
export PYTHONPATH="$RELEASE/src${PYTHONPATH:+:$PYTHONPATH}"
"$VDC_PYTHON" -c 'import torch,numpy,scipy,pandas,h5py,transformers,peft,datasets,fastapi,uvicorn; print("Existing environment imports passed; torch", torch.__version__)'
echo "PK2_RELEASE=$RELEASE"
echo 'Existing checkout and editable environment remain associated with PK1. New workers use explicit PYTHONPATH.'
bash "$RELEASE/scripts/launch_pk2_training.sh"

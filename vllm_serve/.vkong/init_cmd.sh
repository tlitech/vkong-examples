#!/usr/bin/env bash
# Prefetch model weights (optional — vLLM can also download on first start).
set -euo pipefail
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HUGGINGFACE_HUB_CACHE"
MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
echo "[init] prefetch $MODEL → $HUGGINGFACE_HUB_CACHE"
python3 - <<PY
from huggingface_hub import snapshot_download
import os
print(snapshot_download(os.environ.get("MODEL", "Qwen/Qwen3.5-4B")))
PY
echo "[init] done"

#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
mkdir -p "$HF_HOME"
export LLAMA_CACHE="${LLAMA_CACHE:-$PWD/.llama_cache}"
mkdir -p "$LLAMA_CACHE"

export LD_LIBRARY_PATH="/app:${LD_LIBRARY_PATH:-}"
exec /app/llama-server \
  --hf-repo unsloth/Qwen3.5-9B-GGUF \
  --hf-file Qwen3.5-9B-Q4_K_M.gguf \
  --host 0.0.0.0 --port "$PORT" \
  -ngl 999 -c 4096 --alias qwen3.5-9b

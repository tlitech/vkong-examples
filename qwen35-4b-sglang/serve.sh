#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
mkdir -p "$HF_HOME"

exec python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-4B \
  --host 0.0.0.0 --port "$PORT" \
  --context-length 2048 --mem-fraction-static 0.85 \
  --trust-remote-code

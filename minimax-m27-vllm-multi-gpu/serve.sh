#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
mkdir -p "$HF_HOME"

exec vllm serve MiniMaxAI/MiniMax-M2.7 \
  --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --tool-call-parser minimax_m2 --reasoning-parser minimax_m2 \
  --enable-auto-tool-choice --trust-remote-code \
  --max-model-len 32768 --gpu-memory-utilization 0.90

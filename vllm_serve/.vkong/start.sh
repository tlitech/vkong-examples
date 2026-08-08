#!/usr/bin/env bash
# OpenAI-compatible server. Image is vllm/vllm-openai — vLLM is already installed.
set -euo pipefail
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
echo "[start] vllm serve $MODEL :$PORT"
exec vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --max-model-len "${MAX_MODEL_LEN:-2048}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}" \
  --trust-remote-code

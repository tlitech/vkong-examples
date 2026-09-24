#!/usr/bin/env bash
set -euo pipefail

# VKong sets HF_HOME when the Hugging Face cache profile is enabled.
# This local folder is the fallback when HF_HOME is not already set.
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
MODEL="google/diffusiongemma-26B-A4B-it"
REVISION="f7f5b7f5fa82ffc52addd066915886d497f5517b"
mkdir -p "$HF_HOME"

# Run the model on a private port.
vllm serve "$MODEL" \
  --revision "$REVISION" \
  --served-model-name dgemma \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 8192 \
  --generation-config vllm \
  --diffusion-config '{"canvas_length":64}' \
  --max-logprobs 32 \
  --enable-prefix-caching &
vllm_pid=$!

until curl -fsS http://127.0.0.1:8000/health >/dev/null; do
  kill -0 "$vllm_pid" 2>/dev/null || wait "$vllm_pid"
  sleep 2
done

# Expose Jev's structured API through VKong.
exec python3 jev_server.py \
  --upstream http://127.0.0.1:8000 \
  --model dgemma \
  --tokenizer "$MODEL" \
  --tokenizer-revision "$REVISION" \
  --host 0.0.0.0 \
  --port 8011

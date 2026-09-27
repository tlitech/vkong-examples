#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
mkdir -p "$HF_HOME"

# A 24 GB or larger GPU keeps the DiT and VAE resident while the text encoder
# is streamed layer by layer from host memory.
exec sglang serve \
  --model-path Qwen/Qwen-Image-2.1 \
  --lora-path Viggle/Qwen-Image-2.1-viggle-turbo \
  --lora-weight-name Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors \
  --performance-mode manual \
  --component-residency text_encoder=layerwise-offload \
  --host 0.0.0.0 \
  --port "$PORT"

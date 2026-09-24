#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"

MODEL="${MODEL:-google/diffusiongemma-26B-A4B-it}"
MODEL_REVISION="${MODEL_REVISION:-f7f5b7f5fa82ffc52addd066915886d497f5517b}"
MODEL_DIR="${MODEL_DIR:-$HUGGINGFACE_HUB_CACHE/models--google--diffusiongemma-26B-A4B-it/snapshots/$MODEL_REVISION}"
UPSTREAM_PORT="${UPSTREAM_PORT:-8000}"
PORT="${PORT:-8011}"
CANVAS_LENGTH="${CANVAS_LENGTH:-64}"
STRUCTURED_SERVER="/vllm-workspace/examples/features/structured_diffusion/structured_server.py"

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "[start] pinned model snapshot is missing: $MODEL_DIR" >&2
  exit 1
fi
if [[ ! -f "$STRUCTURED_SERVER" ]]; then
  echo "[start] this image does not contain the pinned Jev structured server" >&2
  exit 1
fi

upstream_pid=""
gateway_pid=""
cleanup() {
  trap - EXIT INT TERM
  [[ -z "$gateway_pid" ]] || kill "$gateway_pid" 2>/dev/null || true
  [[ -z "$upstream_pid" ]] || kill "$upstream_pid" 2>/dev/null || true
  [[ -z "$gateway_pid" ]] || wait "$gateway_pid" 2>/dev/null || true
  [[ -z "$upstream_pid" ]] || wait "$upstream_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[start] DiffusionGemma upstream on 127.0.0.1:$UPSTREAM_PORT"
vllm serve "$MODEL_DIR" \
  --served-model-name dgemma \
  --host 127.0.0.1 \
  --port "$UPSTREAM_PORT" \
  --generation-config vllm \
  --diffusion-config "{\"canvas_length\":$CANVAS_LENGTH}" \
  --max-logprobs 32 \
  --enable-prefix-caching \
  --max-num-seqs "${MAX_NUM_SEQS:-4}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" &
upstream_pid=$!

for _ in $(seq 1 180); do
  if ! kill -0 "$upstream_pid" 2>/dev/null; then
    wait "$upstream_pid"
    exit $?
  fi
  if curl -fsS "http://127.0.0.1:$UPSTREAM_PORT/health" >/dev/null; then
    break
  fi
  sleep 2
done
if ! curl -fsS "http://127.0.0.1:$UPSTREAM_PORT/health" >/dev/null; then
  echo "[start] vLLM did not become ready within 6 minutes" >&2
  exit 1
fi

echo "[start] Jev gateway on 0.0.0.0:$PORT"
python3 "$STRUCTURED_SERVER" \
  --upstream "http://127.0.0.1:$UPSTREAM_PORT" \
  --model dgemma \
  --tokenizer "$MODEL_DIR" \
  --canvas "$CANVAS_LENGTH" \
  --host 0.0.0.0 \
  --port "$PORT" &
gateway_pid=$!
wait "$gateway_pid"

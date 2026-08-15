# llama-cpp-serve — RTX 4060 + Qwen3.5-9B GGUF

| | |
|--|--|
| **Image** | `ghcr.io/ggml-org/llama.cpp:server-cuda` (official) |
| **GPU** | RTX 3060 × 1 (12 GB VRAM) |
| **Model** | `unsloth/Qwen3.5-9B-GGUF` — Q4_K_M (5.68 GB) |

Uses the official **llama.cpp** server image with a GGUF-quantized model from Unsloth.

## Run

```bash
cd llama_cpp_serve
vkong run -C .
```

> **Note:** Local port changes each session. Check `vkong` output for the actual port (e.g. `http://127.0.0.1:50534`).

## Smoke test

```bash
# List models
curl -fsS http://127.0.0.1:<local-port>/v1/models | head -c 800

# Chat completion
curl -fsS http://127.0.0.1:<local-port>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-9b",
    "messages": [{"role": "user", "content": "Say hi in one short sentence."}],
    "max_tokens": 64,
    "temperature": 0.2
  }'
```

## Python client

```bash
pip install openai
python client.py
```

## Env (optional)

| Variable | Default |
|----------|---------|
| `REPO` | `unsloth/Qwen3.5-9B-GGUF` |
| `MODEL_FILE` | `Qwen3.5-9B-Q4_K_M.gguf` |
| `N_GPU_LAYERS` | `999` (offload all layers to GPU) |
| `CTX_SIZE` | `4096` |
| `PORT` | `8080` (set by vkong) |

# llama-cpp-serve — Qwen3.5-9B on a rented GPU

| | |
|--|--|
| **Image** | `ghcr.io/ggml-org/llama.cpp:server-cuda` (official) |
| **GPU** | RTX 4090 x1 (24 GB VRAM; available on both supported backends) |
| **Model** | `unsloth/Qwen3.5-9B-GGUF` — Q4_K_M (~5.7 GB) |
| **API** | OpenAI-compatible (`/v1/chat/completions`, `/v1/models`) |

## 1. Login (one-time)

```bash
vkong login --server https://vkong.tli-tech.com
```

## 2. Rent and run

```bash
cd vkong-examples/llama_cpp_serve

vkong run -C . --destroy=false --keep-alive
```

What happens:
- Searches for an RTX 4090 that matches the config on the selected backend
- Rents a matching machine
- Downloads the GGUF model (~5.7 GB, first time only)
- Starts `llama-server` on port 8080
- Opens a local tunnel: `http://127.0.0.1:<port>`

Note the local port from the output. vkong associates the rental with the configured `app: llama-cpp-serve`.

`--destroy=false` keeps the machine alive after Ctrl+C so you can re-attach later without re-downloading the model.

## 3. Test the API

```bash
# List models
curl -s http://127.0.0.1:<port>/v1/models | python3 -m json.tool

# Chat completion
curl -s http://127.0.0.1:<port>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-9b",
    "messages": [{"role":"user","content":"Say hi in one short sentence."}],
    "max_tokens": 64
  }'
```

## 4. Python client (streaming)

```bash
pip install requests
VKONG_URL=http://127.0.0.1:<local-port> python client.py
```

## 5. Dev workflow

After Ctrl+C, the machine is still running. You don't need to `vkong run` again.

```bash
# Re-open tunnel only (app still running on machine, instant)
vkong attach vk_xxxx -C . --skip-sync --skip-run

# Re-sync code + restart llama-server (~1-5 min model reload)
vkong attach vk_xxxx -C .

# Check what machines you have running
vkong instances
```

**Do not use `--live-mirror`** for LLM services — every file save restarts the server and reloads model weights (minutes).

## 6. Publish a public URL (optional)

```bash
vkong deploy -C .
# Output: public_url=https://<random>-deploy.tli-tech.com
```

## 7. Stop the App

```bash
vkong app stop llama-cpp-serve
```

The active rental and its data are removed and billing stops. App history remains visible in the dashboard. The next `vkong run` starts a new App lifecycle.

## Use a different model

```bash
REPO=TheBloke/Mistral-7B-Instruct-v0.2-GGUF \
MODEL_FILE=mistral-7b-instruct-v0.2.Q4_K_M.gguf \
vkong run -C . --destroy=false --keep-alive
```

Or edit the defaults directly in `.vkong.config`.

## Env reference

| Variable | Default | Description |
|----------|---------|-------------|
| `REPO` | `unsloth/Qwen3.5-9B-GGUF` | Hugging Face repo |
| `MODEL_FILE` | `Qwen3.5-9B-Q4_K_M.gguf` | GGUF filename |
| `N_GPU_LAYERS` | `999` | GPU layers (999 = all) |
| `CTX_SIZE` | `4096` | Context window |
| `PORT` | `8080` | Server port (set by vkong) |

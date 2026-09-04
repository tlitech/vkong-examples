# vkong-examples

Example projects for [vkong](https://vkong.tli-tech.com) — rent a GPU, sync your code, run your app.

## Install CLI

```bash
# Linux
curl -fsSL -o vkong https://vkong.tli-tech.com/download/latest/files/vkong-linux-amd64
chmod +x vkong && sudo mv vkong /usr/local/bin/

# macOS (Apple Silicon)
curl -fsSL -o vkong https://vkong.tli-tech.com/download/latest/files/vkong-darwin-arm64
chmod +x vkong && sudo mv vkong /usr/local/bin/

# macOS (Intel)
curl -fsSL -o vkong https://vkong.tli-tech.com/download/latest/files/vkong-darwin-amd64
chmod +x vkong && sudo mv vkong /usr/local/bin/

# Windows
# Download: https://vkong.tli-tech.com/download/latest/files/vkong-windows-amd64.exe
```

## Login

```bash
vkong login
```

## Examples

| Example | Description |
|---------|-------------|
| [hello_world](hello_world) | Minimal FastAPI app on CPU (`ubuntu:24.04`) |
| [vllm_serve](vllm_serve) | Serve Qwen3.5-4B with vLLM on RTX 4090 |
| [sglang_serve](sglang_serve) | Serve Qwen3.5-4B with SGLang on RTX 4090 |
| [triton_inference_server](triton_inference_server) | Triton Inference Server with MobileNetV2 (ONNX) on RTX 4090 |
| [llama_cpp_serve](llama_cpp_serve) | llama.cpp server with Qwen3.5-9B GGUF (Q4_K_M) on RTX 4090 |

## Usage

```bash
# Run an example (rent → sync → start → local port)
cd hello_world
vkong run -C . --destroy=false --keep-alive

# Or use the TUI
vkong
```

> **Note:** Local port changes each session. Check `vkong` output for the actual port.

`--destroy=false` keeps the rental after Ctrl+C. Stop the App with `vkong app stop <app-name>` when you are finished.

## Project layout

```text
my-app/
├── .vkong.config      # app, gpu, image, app_port, init_cmd, start
├── main.py
├── requirements.txt
└── …
```

All config and scripts go in `.vkong.config`. The `app:` field is required and names the App shown in the dashboard. Use `vkong new-config` to generate the file interactively.

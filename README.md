# vkong-examples

Example projects for [vkong](https://vkong.tli-tech.com) — rent a GPU, sync your code, run your app.

## Install CLI (v0.7.0)

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
| [vllm_serve](vllm_serve) | vLLM OpenAI-compatible server with Qwen3.5-4B on RTX 4090 |

## Usage

```bash
# Run an example (rent → sync → start → local port)
cd hello_world
vkong run -C .

# Or use the TUI
vkong
```

> **Note:** Local port changes each session. Check `vkong` output for the actual port.

## Project layout

```text
my-app/
├── .vkong.config      # gpu, image, app_port, init_cmd, start
├── main.py
├── requirements.txt
└── …
```

All config and scripts go in `.vkong.config`. Use `vkong new-config` to generate one interactively.

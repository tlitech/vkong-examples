<p align="center">
  <a href="https://vkong.tli-tech.com">
    <img src="https://vkong.tli-tech.com/logo-text.png" alt="VKong" height="72">
  </a>
</p>

<h1 align="center">VKong Recipes</h1>

<h3 align="center">Runnable recipes for training, generation, and GPU inference with VKong.</h3>

<p align="center">
  <a href="#-get-started">Get started</a> •
  <a href="#-choose-a-recipe">Recipes</a> •
  <a href="https://vkong.tli-tech.com/docs">Documentation</a> •
  <a href="https://vkong.tli-tech.com/download">Download VKong</a>
</p>

Bring your code. VKong finds matching compute, prepares the machine, syncs the
project, and runs the workload through one CLI workflow. Every folder in this
repository is a complete project with its own `vkong.yaml`.

## ⚡ Get started

Install the [`vkong` CLI](https://vkong.tli-tech.com/download), then run the
smallest example:

```bash
git clone https://github.com/tlitech/vkong-examples.git
cd vkong-examples/hello-world

vkong login
vkong run -C .
```

VKong prints the local URL when the service is ready. Open it in a browser or
test it from another terminal:

```bash
curl http://127.0.0.1:<local-port>/
```

The machine remains active when you disconnect so you can attach again. Stop
the App when you are finished to release its compute and stop billing:

```bash
vkong app stop hello-world
```

## 🧭 Choose a recipe

Recipes are grouped by workload so this catalog can grow across models and
frameworks without becoming one long mixed list.

### Start here

| Recipe | Framework | Compute | What you will run |
|---|---|---:|---|
| [Hello World](hello-world) | FastAPI | CPU | Minimal service and the shortest end-to-end VKong workflow |

### LLM inference

| Recipe | Model | Compute | Interface |
|---|---|---:|---|
| [Qwen3.5 9B with llama.cpp](qwen35-9b-llama-cpp) | Qwen3.5-9B GGUF | 1× RTX 4090 | OpenAI-compatible API |
| [Qwen3.5 4B with SGLang](qwen35-4b-sglang) | Qwen3.5-4B | 1× RTX 4090 | OpenAI-compatible API |
| [MiniMax-M2.7 with vLLM (multi-GPU)](minimax-m27-vllm-multi-gpu) | MiniMax-M2.7 | 4× H100 | OpenAI-compatible API, tensor and expert parallelism |

### Model serving

| Recipe | Framework | Model | Compute |
|---|---|---|---:|
| [MobileNetV2 with NVIDIA Triton](mobilenetv2-triton) | Triton + ONNX Runtime | MobileNetV2 | 1× RTX 4090 |

### Training

| Recipe | Framework | Workload | Compute |
|---|---|---|---:|
| [Fine-tune SmolLM2 with Unsloth](smollm2-unsloth-finetuning) | Unsloth | 10-step LoRA fine-tune with a reusable model cache and checkpoint | 1× GPU |
| [GPU training smoke test](pytorch-training-smoke-test) | PyTorch | Verify training, checkpoints, and task lifecycle with a small CNN | 1× GPU |

### Image generation

| Recipe | Framework | Model | Compute |
|---|---|---|---:|
| [SDXL image generation](sdxl-image-generation) | Diffusers | SSD-1B text-to-image task | 1× RTX 4090 |

Start with **Hello World** to verify login, provisioning, sync, and connectivity.
Use **llama.cpp** or **SGLang** for a practical single-GPU inference service.
The **vLLM** recipe requests four H100 GPUs and can be expensive; review its
`max_dph` and the displayed hourly price before starting it.

## How a recipe works

Each recipe keeps the workload contract in `vkong.yaml`:

```text
my-app/
├── vkong.yaml          # workload, compute requirements, image, and commands
├── README.md           # run, test, reconnect, and stop instructions
└── application files
```

The important fields are:

- `type: task|service` — whether the workload exits or keeps listening.
- `app` — the stable App name shown in the dashboard.
- `gpu`, `num_gpus`, `cpu_cores`, `ram_gb`, `disk_gb` — compute requirements.
- `max_dph` — the maximum hourly machine price you accept.
- `image` — the container image for the workload.
- `init_cmd` and `start` — setup and execution commands.

Run a recipe from its directory:

```bash
vkong run -C .
```

For a long-running task or service, detach after it starts:

```bash
vkong run -C . --detach
```

For a finite task that should release compute when it exits:

```bash
vkong run -C . --detach --auto-stop
```

See the [Getting Started guide](https://vkong.tli-tech.com/docs/getting-started)
for the complete first-run flow and the
[CLI reference](https://vkong.tli-tech.com/docs/cli/login) for command details.

## Bring your own project

From an existing project directory, generate a config and run it:

```bash
cd my-project
vkong new-config
vkong run -C .
```

Use these recipes as starting points: copy the closest `vkong.yaml`, then change
the App name, image, compute requirements, setup command, and start command for
your workload.

## Contributing a recipe

A useful recipe should:

- run from a fresh clone without committed credentials;
- pin a compatible image or dependency version;
- state the expected GPU count and VRAM;
- include one deterministic smoke test;
- explain how to stop the App and billing;
- keep secret values out of `vkong.yaml`, source files, and logs.

Open an issue or pull request with the workload, expected hardware, and the
command used to verify it.

# MiniMax-M2.7 with vLLM (multi-GPU)

This example serves `MiniMaxAI/MiniMax-M2.7` through an OpenAI-compatible API.
It follows the official vLLM TP4 recipe and keeps the model on one machine with four
H100 GPUs.

| | |
|--|--|
| **Image** | `vllm/vllm-openai:minimax27` |
| **GPU** | H100 × 4 on one machine |
| **Model** | `MiniMaxAI/MiniMax-M2.7` |
| **Parallelism** | tensor parallel 4 + expert parallel |
| **Context for this demo** | 32,768 tokens |

The model needs about 220 GB for weights. The project requests 500 GB disk so the
Hugging Face cache and image layers fit on a fresh rental.

## Run and deploy

```bash
cd minimax-m27-vllm-multi-gpu
vkong run -C . --detach
vkong deploy -C .
```

The first run rents one four-GPU machine, downloads the model, and starts vLLM. It can
take a while because the checkpoint is large and vLLM compiles GPU kernels on first
start. `--detach` exits the CLI after the service is ready and keeps the rental running.
The deploy command then publishes the stable App URL and exits after the URL is ready.

Follow setup and service output in the CLI or Dashboard → Apps → minimax-m27-vllm-multi-gpu → Logs.

## Smoke test

```bash
export VKONG_URL=https://<url-printed-by-deploy>

curl -fsS "$VKONG_URL/v1/models"

curl -fsS "$VKONG_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniMaxAI/MiniMax-M2.7",
    "messages": [{"role": "user", "content": "Reply with exactly: VKong MiniMax is ready"}],
    "max_tokens": 32,
    "temperature": 0
  }'
```

Or use the Python client:

```bash
pip install openai
VKONG_URL="$VKONG_URL" python client.py
```

Stop the App when testing is complete:

```bash
vkong app stop minimax-m27-vllm-multi-gpu
```

Stopping the App releases all four GPUs and stops billing.

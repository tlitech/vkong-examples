# Serve MiniMax-M2.7 with vLLM

Serve MiniMax-M2.7 on one machine with four H100 GPUs. vLLM downloads the weights and distributes inference across the GPUs.

## Run

From this repository's root, after `vkong login`:

```bash
cd minimax-m27-vllm-multi-gpu
vkong run
```

Keep the terminal open. Copy the local URL printed by VKong into a second terminal:

```bash
export VKONG_URL=http://127.0.0.1:<local-port>
python3 -m pip install openai
python3 client.py
```

## Run in the background

Use `vkong run --detach` when starting a service that should outlive the terminal.
To publish it at an HTTPS URL, run `vkong deploy` from this directory.
Use that URL as `VKONG_URL` with the same client.

## Stop

```bash
vkong app stop minimax-m27-vllm-multi-gpu
```

This releases the machine and stops compute billing. If storage is attached,
VKong saves the volume during a controlled stop; storage billing is separate.

## Keep the model cache

By default the cache lasts for this rental. To save it between machines, add
these fields to `vkong.yaml` before starting:

```yaml
storage: minimax-m27-vllm-multi-gpu-cache
cache: [huggingface]
```

VKong mounts the volume at `/data` and sets the framework's cache location.
The framework handles model downloads. A new machine restores the saved cache;
it does not mount it lazily. Saved cache bytes count toward storage usage.
Choose enough `disk_gb` for the runtime, restored data, and headroom.

## Customize

Edit `vkong.yaml` for compute requirements and the hourly price limit.
This example requests four GPUs and up to $16/hour for the machine. Review `max_dph` before starting. The 500 GB disk accommodates the large model and runtime. Multi-GPU availability and compatibility still need a live run; do not treat this recipe as a completed provider proof.

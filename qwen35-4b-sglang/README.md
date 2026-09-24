# Chat with Qwen3.5 4B and SGLang

Serve Qwen3.5-4B on one RTX 4090 through an OpenAI-compatible API. SGLang downloads the weights when it starts.

## Run

From this repository's root, after `vkong login`:

```bash
cd qwen35-4b-sglang
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
vkong app stop qwen35-4b-sglang
```

This releases the machine and stops compute billing. If storage is attached,
VKong saves the volume during a controlled stop; storage billing is separate.

## Keep the model cache

By default the cache lasts for this rental. To save it between machines, add
these fields to `vkong.yaml` before starting:

```yaml
storage: qwen35-4b-sglang-cache
cache: [huggingface]
```

VKong mounts the volume at `/data` and sets the framework's cache location.
The framework handles model downloads. A new machine restores the saved cache;
it does not mount it lazily. Saved cache bytes count toward storage usage.
Choose enough `disk_gb` for the runtime, restored data, and headroom.

## Customize

Edit `vkong.yaml` for compute requirements and the hourly price limit.
Change the model or context size in `serve.sh`. The example uses a 2,048-token context.

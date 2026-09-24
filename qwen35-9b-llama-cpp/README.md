# Chat with Qwen3.5 9B and llama.cpp

Serve the Q4_K_M version of Qwen3.5-9B on one RTX 4090. llama.cpp downloads the selected GGUF directly from Hugging Face.

## Run

From this repository's root, after `vkong login`:

```bash
cd qwen35-9b-llama-cpp
vkong run
```

Keep the terminal open. Copy the local URL printed by VKong into a second terminal:

```bash
export VKONG_URL=http://127.0.0.1:<local-port>
python3 -m pip install requests
python3 client.py
```

## Run in the background

Use `vkong run --detach` when starting a service that should outlive the terminal.
To publish it at an HTTPS URL, run `vkong deploy` from this directory.
Use that URL as `VKONG_URL` with the same client.

## Stop

```bash
vkong app stop qwen35-9b-llama-cpp
```

This releases the machine and stops compute billing. If storage is attached,
VKong saves the volume during a controlled stop; storage billing is separate.

## Keep the model cache

By default the cache lasts for this rental. To save it between machines, add
these fields to `vkong.yaml` before starting:

```yaml
storage: qwen35-9b-llama-cpp-cache
cache: [huggingface, llama-cpp]
```

VKong mounts the volume at `/data` and sets the framework's cache location.
The framework handles model downloads. A new machine restores the saved cache;
it does not mount it lazily. Saved cache bytes count toward storage usage.
Choose enough `disk_gb` for the runtime, restored data, and headroom.

## Customize

Edit `vkong.yaml` for compute requirements and the hourly price limit.
Change the repository, GGUF filename, or context size in `serve.sh`. The API model name is `qwen3.5-9b`.

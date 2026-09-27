# Qwen-Image 2.1 with Viggle Turbo on SGLang

Run a fast text-to-image and image-editing API on one GPU. This recipe serves
[Qwen-Image 2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) with the
[Viggle Turbo](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo) LoRA
through SGLang's OpenAI-compatible image endpoints. The Turbo LoRA reduces the
base model's 40-step flow to 6 steps without classifier-free guidance.

VKong selects and provisions the machine. The recipe has been exercised on real
Vast.ai and RunPod rentals; see [Measured provider proof](#measured-provider-proof)
for the reproducible results and important caveats.

## Run

Log in once, then start the service from this directory:

```bash
vkong login
cd qwen-image-21-viggle-turbo-sglang
vkong run --detach
```

The first start pulls the container and downloads the Qwen-Image checkpoint and
LoRA, so it can take several minutes. The command prints an attach command; use
it to reconnect to live startup logs without refreshing a page:

```bash
vkong attach <vk-instance-id> -C .
```

Publish the healthy service at an HTTPS URL:

```bash
vkong deploy
```

Set the printed URL and generate an image:

```bash
export VKONG_URL=https://<your-app-url>
python3 client.py \
  --prompt "A glass astronaut helmet on a wet basalt pedestal, neon reflections" \
  --size 1024x1024 \
  --output output.png
```

No third-party Python packages are required by the client.

## Edit an input image

The same client sends a multipart request to `/v1/images/edits` when
`--input-image` is present:

```bash
python3 client.py \
  --input-image output.png \
  --prompt "Keep the composition and add a paper crane reflected in the visor" \
  --size 1024x1024 \
  --output edited.png
```

Each invocation prints end-to-end latency, server inference time, transport and
queueing overhead, request/response sizes, peak GPU memory, and the output SHA-256.

## Benchmark several image sizes

Run four generation sizes followed by an image-edit request using the largest
successful output as input:

```bash
python3 benchmark.py --output-dir benchmark-output
```

All generated images and a machine-readable `results.json` are kept in the
output directory. Override `--sizes` to change the workload.

## Measured provider proof

The checked-in benchmark artifacts preserve every generated image, raw timing,
and startup evidence. Provider names appear here only as deployment evidence;
the recipe and client are provider-neutral.

| Test | GPU | Result | Evidence |
|---|---|---|---|
| Vast.ai, 2026-09-27 | RTX 4090 24 GB | Passed generation at four sizes and one input-image edit | [`benchmarks/2026-09-27-vast-rtx4090`](benchmarks/2026-09-27-vast-rtx4090) |
| Provider-forced startup checks, 2026-09-27 | Vast RTX 5090; RunPod RTX 5090 and A100 80 GB | RunPod A100 passed localhost and public generation; other attempts exposed readiness and CUDA compatibility failures | [`benchmarks/2026-09-27-provider-comparison`](benchmarks/2026-09-27-provider-comparison) |

The RTX 4090 run reached a healthy public API in about 9 minutes on a fresh
machine, including provisioning, image setup, model download, and server startup.
Once warm, a 1024×1024 request took 6.40 seconds end-to-end and a 1536×1536
request took 11.68 seconds. The large input-image edit took 27.29 seconds
end-to-end; most of that extra time was uploading its 5.1 MB source image through
the public endpoint.

The RunPod A100 proof reached agent Ready in about 5 minutes. A partially cached
base-model pull took 1 minute 55 seconds and the uncached LoRA pull took 17
seconds. A 1024×1024 request took 11.90 seconds through the localhost tunnel and
12.25 seconds through the deployed HTTPS endpoint; both responses had the same
SHA-256. A 2.05 MB multipart image upload was not reliable through the public
endpoint in this snapshot, so the RunPod input-edit result is recorded as failed.

Two RunPod RTX 5090 rentals reached Ready but could not initialize CUDA because
the pinned CUDA 13 image required forward compatibility that their host drivers
did not support. Use the default verified RTX 4090 recipe as the proven 24 GB
path; do not treat `verified_only` as a CUDA-driver compatibility guarantee.

These are point-in-time measurements, not capacity or latency guarantees.
Marketplace stock, host network, registry proximity, provider setup, and model
cache state can all change the result. VKong currently reports the combined
startup phase; it does not expose container-pull time separately.

## Stop

```bash
vkong app stop qwen-image-21-viggle-turbo-sglang
```

Stopping releases the rental and ends compute billing.

## Runtime notes

- The default asks for one verified RTX 4090, 64 GB system RAM, and 100 GB disk.
- `serve.sh` keeps the DiT and VAE on the GPU and streams text-encoder layers
  from host memory so the full pipeline can run on a 24 GB GPU.
- Keep `num_inference_steps` at `6` and `guidance_scale` at `1.0` for the Turbo
  LoRA. Larger step counts or CFG do not match its training setup.
- Models are cached only on the rental's local disk. A new machine downloads
  them again; this service intentionally does not use a persistent training volume.
- The pinned SGLang nightly includes Qwen-Image 2.1 support. See the
  [SGLang cookbook](https://docs.sglang.io/cookbook/diffusion/Qwen-Image/Qwen-Image-2.1)
  for other placement modes.

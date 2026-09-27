# Generate and edit images with Qwen-Image 2.1

Run a text-to-image and image-editing API on one RTX 4090. SGLang serves
[Qwen-Image 2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) with the
[Viggle Turbo](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo) LoRA,
which generates an image in six inference steps.

## Run

From this repository's root, after `vkong login`:

```bash
cd qwen-image-21-viggle-turbo-sglang
vkong run --detach
```

The first run downloads the model and LoRA, so startup can take several
minutes. Use the attach command printed by VKong to follow the logs.

Publish the service at a public HTTPS URL:

```bash
vkong deploy
```

## Generate an image

Use the URL printed by `vkong deploy`:

```bash
export VKONG_URL=https://<your-app-url>
python3 client.py \
  --prompt "A glass astronaut helmet on a wet basalt pedestal, neon reflections" \
  --size 1024x1024 \
  --output output.png
```

The client uses only the Python standard library. It saves the image and prints
request timing, response size, peak GPU memory, and the output SHA-256.

## Edit an image

```bash
python3 client.py \
  --input-image output.png \
  --prompt "Keep the composition and add a paper crane reflected in the visor" \
  --size 1024x1024 \
  --output edited.png
```

## Stop

```bash
vkong app stop qwen-image-21-viggle-turbo-sglang
```

This releases the machine and stops compute billing.

## Measured run

The image above comes from a real client request to the public HTTPS App URL
created by `vkong deploy`. The request therefore used the VKong-managed public
route to reach SGLang on the GPU machine; it did not call `localhost` or the
machine directly. All four generation requests completed successfully.

The measurements have two boundaries:

- **GPU inference** is reported by SGLang and measures model execution inside
  the GPU machine.
- **VKong URL end to end** is measured by `client.py`, from sending the HTTPS
  request until the complete JSON response is received. It includes GPU
  inference, the VKong-managed public route, request/response transfer, and
  server-side JSON/base64 encoding. The difference between this value and GPU
  inference is therefore **combined non-inference overhead**, not pure VKong
  network latency.

| Output size | VKong URL end to end | GPU inference | Combined non-inference overhead | HTTP response |
|---|---:|---:|---:|---:|
| 512×512 | 3.46 s | 1.71 s | 1.75 s | 0.66 MB |
| 1024×1024 | 6.40 s | 3.82 s | 2.59 s | 2.95 MB |
| 1536×1024 | 8.87 s | 6.18 s | 2.68 s | 4.14 MB |
| 1536×1536 | 11.68 s | 8.91 s | 2.77 s | 6.87 MB |

### Generated images

These are all five measured inference outputs from the RTX 4090 run. The first
four are text-to-image generations, and the last uses the 1536×1536 image as
input for an image-edit request.

<table>
  <tr>
    <td align="center">
      <a href="assets/generate_512x512.png"><img src="assets/generate_512x512.png" alt="Generated glass astronaut helmet at 512 by 512" width="360"></a><br>
      <strong>Generate · 512×512</strong>
    </td>
    <td align="center">
      <a href="assets/generate_1024x1024.png"><img src="assets/generate_1024x1024.png" alt="Generated glass astronaut helmet at 1024 by 1024" width="360"></a><br>
      <strong>Generate · 1024×1024</strong>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="assets/generate_1536x1024.png"><img src="assets/generate_1536x1024.png" alt="Generated glass astronaut helmet at 1536 by 1024" width="360"></a><br>
      <strong>Generate · 1536×1024</strong>
    </td>
    <td align="center">
      <a href="assets/generate_1536x1536.png"><img src="assets/generate_1536x1536.png" alt="Generated glass astronaut helmet at 1536 by 1536" width="360"></a><br>
      <strong>Generate · 1536×1536</strong>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="assets/generate_1536x1536.png"><img src="assets/generate_1536x1536.png" alt="1536 by 1536 input image used for image editing" width="360"></a><br>
      <strong>Edit input · 1536×1536</strong>
    </td>
    <td align="center">
      <a href="assets/edit_from_generate_1536x1536.png"><img src="assets/edit_from_generate_1536x1536.png" alt="Edited glass astronaut helmet with a paper crane reflected in the visor" width="360"></a><br>
      <strong>Edited output · 1024×1024</strong>
    </td>
  </tr>
</table>

These results prove that the VKong public URL completed every request in this
run and returned responses up to 6.87 MB without a transport failure. They do
not isolate how much of the combined overhead came from the VKong route versus
SGLang response encoding or the client connection, so they are not a standalone
network-latency benchmark.

To repeat the same workload:

```bash
python3 benchmark.py --output-dir benchmark-output
```

For a concurrent load test, `benchmark_concurrency.py` distributes a total request
count across sequential concurrency levels and records per-request timing plus
throughput summaries:

```bash
VKONG_URL=https://<your-app-url> python3 benchmark_concurrency.py \
  --total 100 --concurrencies 1,2,3,4,5,6,7,8 --size 512x512
```

A 100-request RTX 4090 run by **tsdocode** is documented in
[`benchmark-results-tsdocode-2026-09-27.md`](benchmark-results-tsdocode-2026-09-27.md),
with per-request measurements in the adjacent JSON file. It achieved 100/100
successful requests; throughput increased from 0.398 images/s at concurrency 1
to approximately 0.74–0.77 images/s at concurrency 2–8, while per-request latency
rose as concurrent requests queued. This is one warmed-up run, not an SLA.

## Runtime notes

- The recipe requests one verified RTX 4090, 64 GB system RAM, and 100 GB disk.
- Keep `num_inference_steps` at `6` and `guidance_scale` at `1.0` for the Turbo
  LoRA.
- The model cache lives on the rented machine and is removed when that machine
  is released. This service intentionally does not attach a workspace volume.
- `serve.sh` keeps the DiT and VAE on the GPU and offloads text-encoder layers
  to host memory so the pipeline fits a 24 GB GPU.

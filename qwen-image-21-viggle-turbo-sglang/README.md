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
four are text-to-image generations. The final row shows the image-edit input
again beside its edited output so the change can be compared directly.

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
      <a href="assets/edit_input_1536x1536.png"><img src="assets/edit_input_1536x1536.png" alt="1536 by 1536 input image used for image editing" width="360"></a><br>
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

### Measured concurrency run (2026-09-27)

One warmed-up run generated 100 total 512×512 images (six steps, guidance 1.0)
over the VKong public HTTPS route. The concurrency levels ran in sequential
batches; levels 1–4 had 13 requests each and levels 5–8 had 12 each. All 100
requests succeeded.

| Concurrency | Samples | Mean latency | Median | p95 | Mean SGLang inference | Mean combined overhead | Throughput |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 13 | 2.510 s | 2.564 s | 3.916 s | 1.169 s | 1.341 s | 0.398 img/s |
| 2 | 13 | 2.602 s | 2.496 s | 4.003 s | 1.155 s | 1.447 s | 0.740 img/s |
| 3 | 13 | 3.609 s | 3.712 s | 4.452 s | 1.159 s | 2.450 s | 0.768 img/s |
| 4 | 13 | 4.706 s | 5.035 s | 5.839 s | 1.160 s | 3.546 s | 0.766 img/s |
| 5 | 12 | 5.489 s | 6.231 s | 6.843 s | 1.157 s | 4.331 s | 0.767 img/s |
| 6 | 12 | 6.289 s | 7.432 s | 8.373 s | 1.163 s | 5.125 s | 0.758 img/s |
| 7 | 12 | 6.956 s | 8.443 s | 9.385 s | 1.159 s | 5.797 s | 0.761 img/s |
| 8 | 12 | 7.527 s | 8.744 s | 10.875 s | 1.159 s | 6.368 s | 0.766 img/s |

Throughput rose from 0.398 images/s at concurrency 1 to about 0.74–0.77 at
concurrency 2–8, then mostly plateaued, while per-request latency increased as
requests queued. p95 is nearest-rank (`ceil(0.95 × n)`), which is the maximum
observation for these small sample counts. Combined overhead is end-to-end time
minus SGLang inference; it includes queueing, response encoding, routing and
transfer, and is not VKong network-only latency. This single warmed-up run is
not an SLA.

## Runtime notes

- The recipe requests one verified RTX 4090, 64 GB system RAM, and 100 GB disk.
- Keep `num_inference_steps` at `6` and `guidance_scale` at `1.0` for the Turbo
  LoRA.
- The model cache lives on the rented machine and is removed when that machine
  is released. This service intentionally does not attach a workspace volume.
- `serve.sh` keeps the DiT and VAE on the GPU and offloads text-encoder layers
  to host memory so the pipeline fits a 24 GB GPU.

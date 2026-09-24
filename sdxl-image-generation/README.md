# Generate an image with SDXL

Generate a 768×768 image with `segmind/SSD-1B` on one RTX 4090. Diffusers downloads the model automatically. Edit `PROMPT` in `generate.py` to change the image.

## Run

From this repository's root, after `vkong login`:

```bash
cd sdxl-image-generation
vkong run
```

The task prints its result and exits. Its `output.png` stays in the remote
project directory while the machine is active.

## Keep the output

For durable output, add `storage: sdxl-image-generation-data` to `vkong.yaml` and
change the output path in the Python script to `/data/output.png` before running.
Allow enough `disk_gb` for the runtime and restored data. After a successful run,
stop the App to save the volume, then download the saved file:

```bash
vkong app stop sdxl-image-generation
vkong storage download sdxl-image-generation-data output.png -o output.png
```

Storage only saves files under `/data`. The default project-directory output
is temporary and is removed when the machine is destroyed.

## Stop

```bash
vkong app stop sdxl-image-generation
```

For a smoke run whose output can be discarded, use `vkong run --auto-stop`
to release compute when the task finishes.

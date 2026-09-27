# Generate an image with SDXL

Generate a 768×768 image with `segmind/SSD-1B` on one RTX 4090. Diffusers downloads the model automatically. Edit `PROMPT` in `generate.py` to change the image.

## Run

From this repository's root, after `vkong login`:

```bash
cd sdxl-image-generation
vkong run
```

The task prints its result and exits. Its `output.png` stays in the remote
project directory while the machine is active and is removed when the machine is
released. This example does not attach `storage`.

## Stop

```bash
vkong app stop sdxl-image-generation
```

For a smoke run whose output can be discarded, use `vkong run --auto-stop`
to release compute when the task finishes.

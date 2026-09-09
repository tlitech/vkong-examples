# Generate an image with SDXL

This finite GPU task loads the compact SDXL-based `segmind/SSD-1B` model, generates
one 768×768 image, writes it to `output.png`, prints its checksum, and exits.

```bash
cd vkong-examples/sdxl_image
vkong run -C .
```

To change the image, edit `PROMPT` in `generate.py`. If the rental is still active,
upload the change and run the task again:

```bash
vkong attach <instance-id> -C . --update
```

The current CLI streams and retains task logs but does not download remote output files
yet. Run without `--auto-stop` so the remote workspace and `output.png` remain available
while you inspect the Run. Output download is planned as a separate artifact workflow.

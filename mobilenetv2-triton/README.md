# Classify images with MobileNetV2 and Triton

Serve an ONNX image classifier on one RTX 4090. `prepare_model.py` downloads the weights into the directory layout required by Triton.

## Run

From this repository's root, after `vkong login`:

```bash
cd mobilenetv2-triton
vkong run
```

Keep the terminal open. Copy the local URL printed by VKong into a second terminal:

```bash
export VKONG_URL=http://127.0.0.1:<local-port>
python3 -m pip install requests numpy Pillow
python3 client.py test.jpg
```

## Run in the background

Use `vkong run --detach` when starting a service that should outlive the terminal.
To publish it at an HTTPS URL, run `vkong deploy` from this directory.
Use that URL as `VKONG_URL` with the same client.

## Stop

```bash
vkong app stop mobilenetv2-triton
```

This releases the machine and stops compute billing. If storage is attached,
VKong saves the volume during a controlled stop; storage billing is separate.

## Customize

Edit `vkong.yaml` for compute requirements and the hourly price limit.
`models/mobilenetv2/config.pbtxt` defines the inputs and outputs. Triton needs an explicit model repository, so this recipe has a short preparation script.

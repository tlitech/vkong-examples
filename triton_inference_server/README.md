# triton-serve — NVIDIA Triton Inference Server + MobileNetV2

| | |
|--|--|
| **Image** | `nvcr.io/nvidia/tritonserver:26.07-py3` |
| **GPU** | RTX 4090 × 1 |
| **Model** | MobileNetV2 (ONNX, ImageNet 1001-class) |

The start command uses `/opt/tritonserver/bin/tritonserver` explicitly because vkong's
SSH-less agent replaces the image entrypoint and provider environments do not guarantee
that Triton's bin directory remains on `PATH`.

MobileNetV2 image classification served via the NVIDIA Triton Inference Server ONNX Runtime backend.

## Run

```bash
cd triton_inference_server
vkong run -C . --destroy=false --keep-alive
```

> **Note:** The local tunnel URL changes each session. Check `vkong` output, for example `http://127.0.0.1:50534`. `--destroy=false` keeps the machine running after Ctrl+C, so destroy it when you finish.

## Smoke test

```bash
# Server health
curl -fsS http://127.0.0.1:<local-port>/v2/health/ready

# Model metadata
curl -fsS http://127.0.0.1:<local-port>/v2/models/mobilenetv2
```

## Python client

```bash
pip install requests numpy Pillow
VKONG_URL=http://127.0.0.1:<local-port> python client.py test.jpg
```

## Stop the App

```bash
vkong app stop triton-inference-server
```

### Demo

![test.jpg](test.jpg)

```
Top-5 predictions for test.jpg:
  1. gorilla                         97.2%
  2. gibbon                          0.2%
  3. siamang                         0.1%
  4. chimpanzee                      0.1%
  5. patas                           0.0%
```

## Model repository

```text
models/
└── mobilenetv2/
    ├── config.pbtxt     # onnxruntime, input [1,3,224,224], output [-1,1001]
    └── 1/
        └── model.onnx   # downloaded by init_cmd from HuggingFace (~14MB)
```

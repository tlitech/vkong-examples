# triton-serve — Triton Inference Server + MobileNetV2

| | |
|--|--|
| **Image** | `nvcr.io/nvidia/tritonserver:26.07-py3` |
| **GPU** | RTX 4090 × 1 |
| **Model** | MobileNetV2 (ONNX, ImageNet 1001-class) |

MobileNetV2 image classification served via Triton ONNX runtime backend.

## Run

```bash
cd triton_inference_server
vkong run -C .
```

> **Note:** Deploy URL changes each session. Check `vkong` output.

## Smoke test

```bash
# Server health
curl -fsS https://<deploy-url>/v2/health/ready

# Model metadata
curl -fsS https://<deploy-url>/v2/models/mobilenetv2
```

## Python client

```bash
pip install requests numpy Pillow
python client.py test.jpg
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

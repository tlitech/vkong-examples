"""NVIDIA Triton Inference Server client — MobileNetV2 ImageNet classification."""

import json
import os
import sys

import numpy as np
import requests
from PIL import Image

URL = os.environ.get("VKONG_URL", "").rstrip("/")
MODEL = "mobilenetv2"

# ImageNet class names (top ones for demo)
# Full list: https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
IMAGENET_CLASSES = None


def load_imagenet_classes():
    global IMAGENET_CLASSES
    if IMAGENET_CLASSES is None:
        resp = requests.get(
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
            timeout=10,
        )
        IMAGENET_CLASSES = resp.text.strip().split("\n")
    return IMAGENET_CLASSES


def preprocess(image_path):
    img = Image.open(image_path).convert("RGB")
    # Resize shortest edge to 256, then center crop 224x224
    w, h = img.size
    scale = 256 / min(w, h)
    img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    w, h = img.size
    left = (w - 224) // 2
    top = (h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    # Normalize: rescale to [0,1], then (x - 0.5) / 0.5 = 2x - 1 -> [-1, 1]
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    # HWC -> CHW, add batch dim
    blob = arr.transpose(2, 0, 1)[np.newaxis]
    return blob


def infer(image_path):
    blob = preprocess(image_path)
    tensor_bytes = blob.tobytes()

    header = {
        "inputs": [{
            "name": "pixel_values",
            "shape": list(blob.shape),
            "datatype": "FP32",
            "parameters": {"binary_data_size": len(tensor_bytes)},
        }],
        "outputs": [{"name": "logits", "parameters": {"binary_data": True}}],
    }
    header_bytes = json.dumps(header).encode()
    body = header_bytes + tensor_bytes

    resp = requests.post(
        f"{URL}/v2/models/{MODEL}/infer",
        data=body,
        headers={
            "Content-Type": "application/octet-stream",
            "Inference-Header-Content-Length": str(len(header_bytes)),
            "User-Agent": "curl/8.7.1",
        },
        timeout=30,
    )
    resp.raise_for_status()

    h_len = int(resp.headers["Inference-Header-Content-Length"])
    meta = json.loads(resp.content[:h_len])
    raw = resp.content[h_len:]
    nbytes = meta["outputs"][0]["parameters"]["binary_data_size"]
    logits = np.frombuffer(raw[:nbytes], dtype=np.float32)
    return logits


def main():
    if not URL:
        raise SystemExit(
            "Set VKONG_URL to the local URL printed by vkong, for example: "
            "VKONG_URL=http://127.0.0.1:<port> python client.py test.jpg"
        )
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    classes = load_imagenet_classes()
    logits = infer(image_path)

    # Softmax
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()

    # Skip index 0 (background class) — model has 1001 outputs, ImageNet has 1000 classes
    top5 = probs[1:].argsort()[::-1][:5]
    print(f"Top-5 predictions for {image_path}:")
    for i, idx in enumerate(top5):
        name = classes[idx] if idx < len(classes) else str(idx)
        print(f"  {i+1}. {name:30s}  {probs[idx+1]*100:.1f}%")


if __name__ == "__main__":
    main()

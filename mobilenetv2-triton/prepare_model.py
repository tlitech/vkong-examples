"""Put the ONNX weights into Triton's versioned model repository."""
from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download

source = hf_hub_download("onnx-community/mobilenet_v2_1.0_224", "onnx/model.onnx")
target = Path("models/mobilenetv2/1/model.onnx")
target.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(source, target)
print(f"Model ready: {target}")

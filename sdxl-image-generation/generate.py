import hashlib
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline


MODEL = "segmind/SSD-1B"
PROMPT = "a small red panda astronaut, soft studio light, detailed digital illustration"
OUTPUT = Path("output.png")

print(f"loading {MODEL}", flush=True)
pipeline = StableDiffusionXLPipeline.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to("cuda")
pipeline.enable_attention_slicing()

print(f"generating: {PROMPT}", flush=True)
image = pipeline(
    prompt=PROMPT,
    negative_prompt="blurry, low quality, distorted",
    width=768,
    height=768,
    num_inference_steps=20,
    guidance_scale=7.0,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]
image.save(OUTPUT)

digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
print(f"done: {OUTPUT} ({OUTPUT.stat().st_size} bytes, sha256={digest})", flush=True)

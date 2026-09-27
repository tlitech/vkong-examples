#!/usr/bin/env python3
"""Generate or edit an image through the SGLang OpenAI-compatible API."""

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


DEFAULT_PROMPT = (
    "A studio portrait of an old fisherman mending a net, warm rim light, 85mm."
)
USER_AGENT = "vkong-qwen-image-example/1.0 curl-compatible"


def _open(request, timeout):
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:1000]
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"connection failed: {error.reason}") from error
    return status, raw, time.perf_counter() - started


def generate(base_url, prompt, size, steps, guidance_scale, seed, timeout=900):
    body = json.dumps(
        {
            "prompt": prompt,
            "size": size,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "response_format": "b64_json",
            "output_format": "png",
        },
        separators=(",", ":"),
    ).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/images/generations",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    status, raw, elapsed = _open(request, timeout)
    return status, len(body), raw, elapsed


def _multipart(fields, image_path):
    boundary = "----vkong-qwen-image-" + uuid.uuid4().hex
    chunks = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                str(value).encode(),
                b"\r\n",
            ]
        )
    image_bytes = image_path.read_bytes()
    content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    chunks.extend(
        [
            f"--{boundary}\r\n".encode(),
            (
                'Content-Disposition: form-data; name="image"; '
                f'filename="{image_path.name}"\r\n'
            ).encode(),
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            image_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return boundary, b"".join(chunks)


def edit(base_url, input_image, prompt, size, steps, guidance_scale, seed, timeout=900):
    boundary, body = _multipart(
        {
            "prompt": prompt,
            "size": size,
            "response_format": "b64_json",
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
        },
        input_image,
    )
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/images/edits",
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    status, raw, elapsed = _open(request, timeout)
    return status, len(body), raw, elapsed


def save_response(output_path, status, request_bytes, raw, elapsed):
    response = json.loads(raw)
    image = base64.b64decode(response["data"][0]["b64_json"])
    output_path.write_bytes(image)
    server_seconds = response.get("inference_time_s")
    overhead = None
    if isinstance(server_seconds, (int, float)):
        overhead = max(0.0, elapsed - float(server_seconds))
    return {
        "http_status": status,
        "elapsed_seconds": round(elapsed, 3),
        "server_inference_seconds": server_seconds,
        "non_inference_overhead_seconds": round(overhead, 3) if overhead is not None else None,
        "request_bytes": request_bytes,
        "response_bytes": len(raw),
        "image_bytes": len(image),
        "sha256": hashlib.sha256(image).hexdigest(),
        "output": str(output_path),
        "peak_memory_mb": response.get("peak_memory_mb"),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("VKONG_URL", ""))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--size", default="1024x1024")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-image", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output.png"))
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if not args.url:
        parser.error("set VKONG_URL or pass --url")
    if args.input_image and not args.input_image.is_file():
        parser.error(f"input image not found: {args.input_image}")
    return args


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.input_image:
        result = edit(
            args.url,
            args.input_image,
            args.prompt,
            args.size,
            args.steps,
            args.guidance_scale,
            args.seed,
            args.timeout,
        )
    else:
        result = generate(
            args.url,
            args.prompt,
            args.size,
            args.steps,
            args.guidance_scale,
            args.seed,
            args.timeout,
        )
    summary = save_response(args.output, *result)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

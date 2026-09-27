#!/usr/bin/env python3
"""Measure Qwen Image generation latency across sequential concurrency levels.

Examples:
  VKONG_URL=https://your-app-url python3 benchmark_concurrency.py
  python3 benchmark_concurrency.py --total 100 --concurrencies 1,2,4,8

Each level is a separate batch. Requests within a batch run concurrently; batches
run sequentially so the reported wall time and throughput are easy to interpret.
"""

import argparse
import math
import base64
import concurrent.futures
import json
import os
import statistics
import time
import urllib.request
from pathlib import Path

DEFAULT_PROMPT = "A studio portrait of an old fisherman mending a net, warm rim light, 85mm."


def percentile(values, p):
    """Nearest-rank percentile (p in [0, 1])."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * p) - 1))
    return round(ordered[index], 3)


def one_request(url, prompt, size, steps, guidance_scale, seed, timeout, output_dir):
    payload = json.dumps({
        "prompt": prompt,
        "size": size,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "response_format": "b64_json",
        "output_format": "png",
    }, separators=(",", ":")).encode()
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/images/generations",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "vkong-qwen-image-concurrency-benchmark/1.0",
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = response.status
        elapsed = time.perf_counter() - started
        body = json.loads(raw)
        image = base64.b64decode(body["data"][0]["b64_json"], validate=True)
        if not image.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("response image was not PNG")
        (output_dir / f"image-{seed:03d}.png").write_bytes(image)
        inference = body.get("inference_time_s")
        return {
            "seed": seed,
            "status": status,
            "elapsed_s": elapsed,
            "inference_s": inference,
            "overhead_s": max(0.0, elapsed - inference)
            if isinstance(inference, (int, float)) else None,
            "response_bytes": len(raw),
            "image_bytes": len(image),
            "peak_memory_mb": body.get("peak_memory_mb"),
        }
    except Exception as error:  # retain per-request failures in the report
        return {
            "seed": seed,
            "error": f"{type(error).__name__}: {error}",
            "elapsed_s": time.perf_counter() - started,
        }


def summarize(level, results, wall_s):
    successful = [r for r in results if r.get("status") == 200 and "error" not in r]
    latency = [r["elapsed_s"] for r in successful]
    inference = [r["inference_s"] for r in successful
                 if isinstance(r.get("inference_s"), (int, float))]
    overhead = [r["overhead_s"] for r in successful
                if isinstance(r.get("overhead_s"), (int, float))]
    return {
        "concurrency": level,
        "requested": len(results),
        "success": len(successful),
        "errors": len(results) - len(successful),
        "wall_s": round(wall_s, 3),
        "throughput_img_s": round(len(successful) / wall_s, 3) if wall_s else None,
        "latency_mean_s": round(statistics.mean(latency), 3) if latency else None,
        "latency_median_s": round(statistics.median(latency), 3) if latency else None,
        "latency_p95_s": percentile(latency, 0.95),
        "inference_mean_s": round(statistics.mean(inference), 3) if inference else None,
        "overhead_mean_s": round(statistics.mean(overhead), 3) if overhead else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("VKONG_URL", ""))
    parser.add_argument("--total", type=int, default=100)
    parser.add_argument("--concurrencies", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--size", default="512x512")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-concurrency-output"))
    args = parser.parse_args()
    if not args.url:
        parser.error("set VKONG_URL or pass --url")
    if args.total <= 0 or args.steps <= 0:
        parser.error("--total and --steps must be positive")
    levels = [int(value) for value in args.concurrencies.split(",")]
    if not levels or any(value <= 0 for value in levels):
        parser.error("--concurrencies must be comma-separated positive integers")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    quotient, remainder = divmod(args.total, len(levels))
    summaries, all_results = [], []
    seed = 1
    for index, level in enumerate(levels):
        count = quotient + (1 if index < remainder else 0)
        seeds = list(range(seed, seed + count))
        seed += count
        print(f"Running concurrency={level}, requests={count}", flush=True)
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=level) as executor:
            results = list(executor.map(
                lambda n: one_request(args.url, args.prompt, args.size, args.steps,
                                      args.guidance_scale, n, args.timeout, args.output_dir),
                seeds,
            ))
        wall = time.perf_counter() - started
        all_results.extend({**result, "concurrency": level} for result in results)
        summary = summarize(level, results, wall)
        summaries.append(summary)
        print(json.dumps(summary), flush=True)

    report = {
        "total_requests": args.total,
        "successful": sum(r.get("status") == 200 and "error" not in r for r in all_results),
        "configuration": {
            "size": args.size,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "concurrencies": levels,
            "prompt": args.prompt,
        },
        "summary": summaries,
        "results": all_results,
    }
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output_dir / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Benchmark several output sizes, then edit the largest successful image."""

import argparse
import json
import os
from pathlib import Path

import client


PROMPT = (
    "A cinematic product photograph of a translucent glass astronaut helmet on a "
    "wet basalt pedestal, intricate condensation, neon reflections, sharp micro-detail."
)
EDIT_PROMPT = (
    "Keep the same helmet composition and add a tiny paper crane reflected in the visor."
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("VKONG_URL", ""))
    parser.add_argument(
        "--sizes",
        default="512x512,1024x1024,1536x1024,1536x1536",
        help="comma-separated WIDTHxHEIGHT values",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-output"))
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if not args.url:
        parser.error("set VKONG_URL or pass --url")
    return args


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    generated = []

    for index, size in enumerate(part.strip() for part in args.sizes.split(",") if part.strip()):
        label = f"generate_{size}"
        output = args.output_dir / f"{label}.png"
        try:
            raw_result = client.generate(args.url, PROMPT, size, 6, 1.0, index, args.timeout)
            result = {"label": label, **client.save_response(output, *raw_result)}
            generated.append(output)
        except Exception as error:
            result = {"label": label, "error": repr(error)}
        results.append(result)
        print(json.dumps(result), flush=True)

    if generated:
        source = generated[-1]
        label = f"edit_from_{source.stem}"
        output = args.output_dir / f"{label}.png"
        try:
            raw_result = client.edit(
                args.url, source, EDIT_PROMPT, "1024x1024", 6, 1.0, 41, args.timeout
            )
            result = {"label": label, **client.save_response(output, *raw_result)}
            result["input_image"] = source.name
            result["input_image_bytes"] = source.stat().st_size
        except Exception as error:
            result = {"label": label, "input_image": source.name, "error": repr(error)}
        results.append(result)
        print(json.dumps(result), flush=True)

    summary = args.output_dir / "results.json"
    summary.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"summary": str(summary), "tests": len(results)}))
    if any("error" in result for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

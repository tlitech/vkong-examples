#!/usr/bin/env python3
"""Deterministic smoke request for the DiffusionGemma Jev endpoint."""

import json
import os
import urllib.request


base = os.environ.get("VKONG_URL", "http://127.0.0.1:8011").rstrip("/")
payload = {
    "model": "jev-latest",
    "state": {
        "ticket": "Production checkout is unavailable and a launch starts in 30 minutes.",
        "customer_impact": "All purchases fail.",
    },
    "samples": 2,
    "questions": {
        "urgent": {
            "type": "noul",
            "instructions": "Does this need a response within the next hour?",
        },
        "owner": {
            "type": "choice",
            "instructions": "Which team should handle this first?",
            "criteria": {
                "payments": "Checkout and payment failures",
                "growth": "Campaign and acquisition issues",
                "data": "Analytics pipeline issues",
            },
        },
    },
}
headers = {"content-type": "application/json"}
if token := os.environ.get("API_KEY"):
    headers["authorization"] = f"Bearer {token}"
request = urllib.request.Request(
    f"{base}/v1/systemone",
    data=json.dumps(payload).encode(),
    headers=headers,
)
with urllib.request.urlopen(request, timeout=120) as response:
    result = json.load(response)

answers = result.get("answers", {})
urgent = answers.get("urgent", {})
owner = answers.get("owner", {})
if urgent.get("type") != "noul" or not 0 <= urgent.get("noul", -1) <= 1:
    raise SystemExit(f"invalid urgent answer: {urgent!r}")
if owner.get("type") != "choice" or owner.get("choice") not in {
    "payments",
    "growth",
    "data",
}:
    raise SystemExit(f"invalid owner answer: {owner!r}")
print(json.dumps(result, indent=2))

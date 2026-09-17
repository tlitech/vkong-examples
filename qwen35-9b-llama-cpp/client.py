"""OpenAI-compatible streaming client for llama.cpp serve."""

import json
import os
import requests

BASE_URL = os.environ.get("VKONG_URL", "").rstrip("/")
if not BASE_URL:
    raise SystemExit(
        "Set VKONG_URL to the URL printed by vkong, for example: "
        "VKONG_URL=http://127.0.0.1:<port> python client.py"
    )

resp = requests.post(
    f"{BASE_URL}/v1/chat/completions",
    json={
        "model": "qwen3.5-9b",
        "messages": [{"role": "user", "content": "Say hi in one short sentence."}],
        "max_tokens": 512,
        "temperature": 0.2,
        "stream": True,
    },
    stream=True,
)

phase = None
for line in resp.iter_lines(decode_unicode=True):
    if not line or not line.startswith("data: "):
        continue
    data = line[6:]
    if data == "[DONE]":
        break
    chunk = json.loads(data)
    delta = chunk["choices"][0]["delta"]

    if delta.get("reasoning_content"):
        if phase != "thinking":
            phase = "thinking"
            print("Thinking: ", end="", flush=True)
        print(delta["reasoning_content"], end="", flush=True)

    if delta.get("content"):
        if phase != "answer":
            if phase == "thinking":
                print()
            phase = "answer"
            print("Answer: ", end="", flush=True)
        print(delta["content"], end="", flush=True)

print()

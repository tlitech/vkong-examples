"""OpenAI-compatible client for SGLang serve."""

import os
from openai import OpenAI

base_url = os.environ.get("VKONG_URL", "").rstrip("/")
if not base_url:
    raise SystemExit(
        "Set VKONG_URL to the URL printed by vkong, for example: "
        "VKONG_URL=http://127.0.0.1:<port> python client.py"
    )

client = OpenAI(base_url=f"{base_url}/v1", api_key="unused")

response = client.chat.completions.create(
    model="Qwen/Qwen3.5-4B",
    messages=[{"role": "user", "content": "Say hi in one short sentence."}],
    max_tokens=64,
    temperature=0.2,
)

print(response.choices[0].message.content)

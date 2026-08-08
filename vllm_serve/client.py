"""OpenAI-compatible client for vLLM serve."""

from openai import OpenAI

BASE_URL = "http://127.0.0.1:50534/v1"  # port changes each session, check vkong output

client = OpenAI(base_url=BASE_URL, api_key="unused")

response = client.chat.completions.create(
    model="Qwen/Qwen3.5-4B",
    messages=[{"role": "user", "content": "Say hi in one short sentence."}],
    max_tokens=64,
    temperature=0.2,
)

print(response.choices[0].message.content)

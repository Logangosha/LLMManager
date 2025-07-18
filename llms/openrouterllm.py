# llms/openrouterllm.py

import httpx
from core.llm import LLM
from core.message import Message

class OpenRouterLLM(LLM):
    async def generate(self, messages: list[Message]) -> str:
        headers = {
            "Authorization": f"Bearer {self.config.get('api_key')}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.get("model", "openrouter/gpt-4o-mini"),
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self.config.get("temperature", 0.7),
            "max_tokens": self.config.get("max_tokens", 512),
        }
        async with httpx.AsyncClient() as client:
            response = await client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

import asyncio
import time
import aiohttp
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    api_key: Optional[str] = None
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "google/gemini-flash-1.5-free"
    is_local: bool = False
    rate_limit_per_min: int = 15  # Default for free OpenRouter models

class RateLimiter:
    def __init__(self, requests_per_minute: int):
        self.delay = 60.0 / requests_per_minute
        self.last_call = 0.0
        self.lock = asyncio.Lock()

    async def wait(self):
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.delay:
                wait_time = self.delay - elapsed
                await asyncio.sleep(wait_time)
            self.last_call = time.time()

class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.rate_limiter = RateLimiter(config.rate_limit_per_min) if not config.is_local else None
        self.headers = {
            "Content-Type": "application/json"
        }
        if config.api_key:
            if "openrouter" in config.base_url.lower():
                self.headers["Authorization"] = f"Bearer {config.api_key}"
                self.headers["HTTP-Referer"] = "https://github.com/pranay5255/ablationsTrainerLocal"
                self.headers["X-Title"] = "YudaiV3 Training"
            else:
                self.headers["Authorization"] = f"Bearer {config.api_key}"

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 4096) -> str:
        if self.rate_limiter:
            await self.rate_limiter.wait()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(f"{self.config.base_url}/chat/completions", headers=self.headers, json=payload) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        logger.error(f"Error from LLM API ({resp.status}): {text}")
                        return ""
                    
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
            except Exception as e:
                logger.error(f"LLM request failed: {e}")
                return ""

    async def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        tasks = [self.generate(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)


"""Thin provider-agnostic LLM client. Optional: nothing in the app *requires* a model."""
import time
from typing import Optional, Protocol

import httpx

from .config import Settings


class LLMError(Exception):
    pass


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class HTTPLLM:
    def __init__(self, s: Settings):
        self.s = s

    def complete(self, system: str, user: str) -> str:
        s = self.s
        try:
            if s.llm_provider == "anthropic":
                r = httpx.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": s.llm_api_key, "anthropic-version": "2023-06-01"},
                    json={"model": s.llm_model, "max_tokens": 1024, "temperature": 0, "system": system,
                          "messages": [{"role": "user", "content": user}]},
                    timeout=s.llm_timeout_s)
                r.raise_for_status()
                return r.json()["content"][0]["text"]
            r = httpx.post(
                f"{s.llm_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {s.llm_api_key}"},
                json={"model": s.llm_model, "temperature": 0,
                      "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
                timeout=s.llm_timeout_s)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
            raise LLMError(f"{type(e).__name__}: {e}") from e


def build_llm(s: Settings) -> Optional[LLM]:
    if s.llm_provider in ("openai", "anthropic") and s.llm_api_key and s.llm_model:
        return HTTPLLM(s)
    return None


def sleep_backoff(s: Settings, attempt: int):
    if s.llm_backoff_s > 0:
        time.sleep(min(s.llm_backoff_s * (2 ** attempt), 5))

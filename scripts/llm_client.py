"""Provider-agnostic LLM client over the OpenAI-compatible chat API.

Configured ENTIRELY by environment variables so any OpenAI-compatible provider
works (DeepSeek / Qwen / Kimi / Zhipu / OpenAI / a self-hosted gateway):
  LLM_API_KEY   (required)
  LLM_BASE_URL  (optional; provider's base url, e.g. https://api.deepseek.com/v1)
  LLM_MODEL     (optional; default 'gpt-4o-mini')
No provider is hard-coded. No network at import time."""
from __future__ import annotations

import json
import os

from openai import OpenAI


class LLMConfigError(RuntimeError):
    """Raised when the LLM is called without an API key configured."""


def is_configured() -> bool:
    return bool(os.environ.get("LLM_API_KEY"))


def _client() -> OpenAI:
    key = os.environ.get("LLM_API_KEY")
    if not key:
        raise LLMConfigError(
            "LLM_API_KEY 未配置。请设置 LLM_API_KEY（及可选 LLM_BASE_URL / LLM_MODEL）。"
        )
    base = os.environ.get("LLM_BASE_URL") or None
    return OpenAI(api_key=key, base_url=base)


def extract_json(system_prompt: str, user_text: str, model: str | None = None) -> dict:
    """Ask the model to return a single JSON object; parse and return it."""
    client = _client()
    model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or "{}"
    return json.loads(content)

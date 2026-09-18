"""One OpenAI-compatible client for every provider (Gemini, Ollama, Groq, ...).

Every provider in config.PROVIDERS speaks the OpenAI chat-completions dialect,
so the whole app has exactly one LLM code path; switching provider is two env
vars. Only the lowest-common-denominator parameters are sent (no temperature,
no max_tokens) because providers disagree on names and accepted values.
"""

from __future__ import annotations

import json
import re

from openai import AsyncOpenAI

from atlas import config as C

_client: AsyncOpenAI | None = None


def client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(base_url=C.LLM_BASE_URL, api_key=C.LLM_API_KEY or "none")
    return _client


def usage_of(u) -> dict:
    """Normalise a completion's usage; providers omit fields freely."""
    if u is None:
        return {"input": 0, "cache_read": 0, "output": 0}
    cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    return {"input": u.prompt_tokens or 0, "cache_read": cached, "output": u.completion_tokens or 0}


async def complete(model: str, messages: list[dict], json_mode: bool = False) -> tuple[str, dict]:
    kw = {"response_format": {"type": "json_object"}} if json_mode else {}
    r = await client().chat.completions.create(model=model, messages=messages, **kw)
    return (r.choices[0].message.content or "").strip(), usage_of(r.usage)


def parse_json(text: str) -> dict:
    """Some models fence JSON despite json mode; some add prose. Take the outermost object."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError(f"no JSON object in model output: {text[:120]!r}")
    return json.loads(text[start : end + 1])

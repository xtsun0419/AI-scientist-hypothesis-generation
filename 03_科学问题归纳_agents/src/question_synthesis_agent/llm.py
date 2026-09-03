from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    api_key: str
    model: str

    @classmethod
    def from_env(cls) -> "LLMSettings | None":
        api_key = os.environ.get("OPENAI_API_KEY")
        model = os.environ.get("OPENAI_MODEL")
        if not api_key or not model:
            return None
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        return cls(base_url=base_url.rstrip("/"), api_key=api_key, model=model)


class OpenAICompatibleChatClient:
    def __init__(self, settings: LLMSettings, *, timeout_seconds: int = 90):
        self.settings = settings
        self.timeout_seconds = timeout_seconds

    def chat(self, *, system_prompt: str, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        url = f"{self.settings.base_url}/chat/completions"
        payload = {
            "model": self.settings.model,
            "messages": [{"role": "system", "content": system_prompt}, *messages],
            "temperature": temperature,
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "question-synthesis-agent/0.1",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds, context=_ssl_context()) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc
        content: Any = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        if isinstance(content, list):
            return "\n".join(str(item.get("text") or item) for item in content)
        if isinstance(content, str) and content.strip():
            return content.strip()
        # 兼容推理模型：content 为空时回退 reasoning_content
        fallback = raw.get("choices", [{}])[0].get("message", {}).get("reasoning_content", "")
        return str(fallback).strip()


def _ssl_context() -> ssl.SSLContext | None:
    try:
        import certifi
    except ImportError:
        return None
    return ssl.create_default_context(cafile=certifi.where())

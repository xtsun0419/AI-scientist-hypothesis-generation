from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .http import urlopen


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


def _message_text(raw: dict[str, Any]) -> Any:
    """提取回复文本；兼容推理模型（content 为空时回退 reasoning_content）。"""
    message = raw.get("choices", [{}])[0].get("message", {}) or {}
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(str(item.get("text") or item) for item in content)
    if isinstance(content, str) and content.strip():
        return content
    fallback = message.get("reasoning_content")
    if isinstance(fallback, str) and fallback.strip():
        return fallback
    return content if content is not None else ""


def _loads_json(text: Any) -> Any:
    """严格解析失败时，从文本（代码块/推理内容）中提取 JSON 对象。"""
    if isinstance(text, (dict, list)):
        return text
    source = str(text)
    try:
        return json.loads(source)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    candidates: list[Any] = []
    for match in re.finditer(r"\{", source):
        try:
            obj, _ = decoder.raw_decode(source, match.start())
        except json.JSONDecodeError:
            continue
        candidates.append(obj)
    if candidates:
        return candidates[-1]
    raise ValueError("no JSON object found")


class OpenAICompatibleClient:
    def __init__(self, settings: LLMSettings, *, timeout_seconds: int = 60):
        self.settings = settings
        self.timeout_seconds = timeout_seconds

    def review_relevance(self, prompt: str) -> dict[str, Any]:
        return self.chat_json(
            (
                "You review whether a literature record belongs to the configured research scope. "
                "Return only strict JSON with keys: decision, confidence, reason, "
                "matched_domain_terms, exclude_reason."
            ),
            prompt,
        )

    def recommend_literature(self, prompt: str) -> dict[str, Any]:
        return self.chat_json(
            (
                "You rerank a fixed candidate pool for a scientific literature exploration round. "
                "Never invent papers, titles, DOIs, URLs, or candidate ids. Select only candidate "
                "paper_id values present in the user JSON. Return only strict JSON with key selected, "
                "where selected is a list of objects with paper_id, score, reason, topic_tags."
            ),
            prompt,
        )

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        url = f"{self.settings.base_url}/chat/completions"
        payload = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "lit-agent/0.2",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        content = _message_text(raw)
        if isinstance(content, dict):
            parsed = content
            parsed["_raw_completion"] = raw
            return parsed
        try:
            parsed = _loads_json(content)
        except ValueError as exc:
            raise ValueError(f"LLM returned invalid JSON: {str(content)[:200]}") from exc
        parsed["_raw_completion"] = raw
        return parsed

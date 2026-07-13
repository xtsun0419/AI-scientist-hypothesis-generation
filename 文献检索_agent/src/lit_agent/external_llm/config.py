from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from lit_agent.config import project_root


DEFAULT_BASE_URL = "https://api.openai.com/v1"
MAX_LLM_APIS = 2
LLM_CONFIG_VERSION = 1


def default_external_llm_config_path() -> Path:
    return project_root() / "configs" / "external_llm_apis.json"


def load_llm_api_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or default_external_llm_config_path()
    empty = {"version": LLM_CONFIG_VERSION, "apis": []}
    if not config_path.exists():
        return empty
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty
    apis = payload.get("apis", [])
    if not isinstance(apis, list):
        apis = []
    cleaned = [_clean_api_entry(item) for item in apis[:MAX_LLM_APIS] if isinstance(item, dict)]
    return {"version": LLM_CONFIG_VERSION, "apis": cleaned}


def save_llm_api_from_form(fields: dict[str, str], path: Path | None = None) -> dict[str, Any]:
    config = load_llm_api_config(path)
    apis = list(config.get("apis", []))
    slot = _slot_index(fields.get("slot"))
    while len(apis) <= slot:
        apis.append(_blank_api_entry(len(apis)))

    previous = apis[slot]
    api_key = (fields.get("api_key") or "").strip() or str(previous.get("api_key") or "")
    entry = _clean_api_entry(
        {
            "name": fields.get("name") or f"LLM API {slot + 1}",
            "base_url": fields.get("base_url") or DEFAULT_BASE_URL,
            "api_key": api_key,
            "model": fields.get("model") or "",
            "enabled": fields.get("enabled") == "1",
        }
    )
    if not entry["model"]:
        raise ValueError("请填写模型名称。")
    if not entry["api_key"]:
        raise ValueError("请填写 API Key。")

    apis[slot] = entry
    if entry["enabled"]:
        for index, item in enumerate(apis):
            item["enabled"] = index == slot
    for index, item in enumerate(apis):
        item["priority"] = index + 1
    saved = {"version": LLM_CONFIG_VERSION, "apis": apis[:MAX_LLM_APIS]}
    config_path = path or default_external_llm_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    apply_active_llm_to_env(saved, override=True)
    return saved


def active_llm_api(config: dict[str, Any] | None = None) -> dict[str, Any] | None:
    payload = config if config is not None else load_llm_api_config()
    apis = payload.get("apis", [])
    if not isinstance(apis, list):
        return None
    for item in apis:
        if not isinstance(item, dict):
            continue
        entry = _clean_api_entry(item)
        if entry["enabled"] and entry["api_key"] and entry["model"]:
            return entry
    return None


def apply_active_llm_to_env(config: dict[str, Any] | None = None, *, override: bool = False) -> bool:
    active = active_llm_api(config)
    if not active:
        return False
    values = {
        "OPENAI_BASE_URL": active["base_url"],
        "OPENAI_API_KEY": active["api_key"],
        "OPENAI_MODEL": active["model"],
        "LIT_AGENT_SELECTION_MODE": "llm",
    }
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = str(value)
    return True


def _slot_index(value: str | None) -> int:
    try:
        slot = int(value or "0")
    except ValueError:
        slot = 0
    return max(0, min(slot, MAX_LLM_APIS - 1))


def _blank_api_entry(index: int) -> dict[str, Any]:
    return {
        "name": f"LLM API {index + 1}",
        "base_url": DEFAULT_BASE_URL,
        "api_key": "",
        "model": "",
        "enabled": index == 0,
        "priority": index + 1,
    }


def _clean_api_entry(item: dict[str, Any]) -> dict[str, Any]:
    base_url = str(item.get("base_url") or DEFAULT_BASE_URL).strip().rstrip("/") or DEFAULT_BASE_URL
    return {
        "name": str(item.get("name") or "LLM API").strip(),
        "base_url": base_url,
        "api_key": str(item.get("api_key") or "").strip(),
        "model": str(item.get("model") or "").strip(),
        "enabled": bool(item.get("enabled")),
        "priority": int(item.get("priority") or 1),
    }

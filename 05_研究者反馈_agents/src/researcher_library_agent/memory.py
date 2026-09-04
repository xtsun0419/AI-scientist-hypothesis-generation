from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any


MAX_MEMORY_RUNS = 12
MAX_MEMORY_CHARS = 12_000


def rebuild_memory(
    path: Path,
    runs: list[dict[str, Any]],
    items_by_id: dict[int, dict[str, Any]],
    reviews_by_run: dict[int, dict[str, Any]] | None = None,
) -> str:
    lines = [
        "# Researcher-Hypothesis Dialogue Memory",
        "",
        "This file contains only public dialogue outputs and cited library records.",
        "Private system prompts, hidden reasoning, route contexts, and critic contexts are excluded.",
    ]
    review_map = reviews_by_run or {}
    for run in reversed(runs[:MAX_MEMORY_RUNS]):
        item_ids = _item_ids(run.get("item_ids_json"))
        lines.extend(["", f"## Run {run.get('id')} · {run.get('created_at')}", "", "### Sources"])
        for item_id in item_ids:
            item = items_by_id.get(item_id)
            if item:
                lines.append(f"- [{item.get('source_type')} #{item_id}] {item.get('title')}")
        lines.extend(
            [
                "",
                "### Researcher Question",
                str(run.get("researcher_question") or ""),
                "",
                "### Hypothesis",
                str(run.get("hypothesis") or ""),
                "",
                "### Validation",
                _format_value(run.get("validation")),
            ]
        )
        review = review_map.get(int(run.get("id") or 0))
        if review:
            lines.extend(
                [
                    "",
                    "### Watchdog Review",
                    f"status: {review.get('status')} · issue: {review.get('issue')}",
                    str(review.get("rationale") or ""),
                ]
            )
            if review.get("restart_instruction"):
                lines.extend(["", "### Public Reset Instruction", str(review["restart_instruction"])])
    content = "\n".join(lines).strip() + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return content


def read_memory(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    return content[-MAX_MEMORY_CHARS:]


def _item_ids(value: Any) -> list[int]:
    try:
        values = json.loads(str(value or "[]"))
    except json.JSONDecodeError:
        return []
    return [int(item) for item in values if str(item).isdigit()]


def _format_value(value: Any) -> str:
    if not isinstance(value, str):
        return _format_structured(value)
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value
    return _format_structured(parsed)


def _format_structured(value: Any) -> str:
    if isinstance(value, dict):
        return "\n".join(f"{key}：{_format_structured(item)}" for key, item in value.items())
    if isinstance(value, list):
        return "；".join(_format_structured(item) for item in value)
    return str(value or "")

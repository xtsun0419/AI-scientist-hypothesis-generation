from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import QueryPlan


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    return project_root() / "configs" / "domain_general_research.yaml"


def default_db_path() -> Path:
    return project_root() / "data" / "literature.sqlite"


def default_pdf_dir() -> Path:
    return project_root() / "data" / "pdfs"


def default_report_dir() -> Path:
    return project_root() / "reports"


def default_parsed_dir() -> Path:
    return project_root().parent / "文献分析_agent" / "data" / "parsed_papers"


def load_domain_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or default_config_path()
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        return json.loads(text)
    return _parse_simple_yaml(text)


def query_plan_from_config(
    config: dict[str, Any],
    *,
    from_year: int | None = None,
    to_year: int | None = None,
    sources: list[str] | None = None,
) -> QueryPlan:
    queries_cfg = config.get("queries", {})
    base_terms = [str(item) for item in queries_cfg.get("base_terms", [])]
    performance_terms = [str(item) for item in queries_cfg.get("performance_terms", [])]
    process_terms = [str(item) for item in queries_cfg.get("process_terms", [])]
    templates = [str(item) for item in queries_cfg.get("query_templates", ["{base}"])]

    queries: list[str] = []
    for base in base_terms:
        for template in templates:
            if "{performance}" in template:
                for performance in performance_terms:
                    queries.append(template.format(base=base, performance=performance))
            elif "{process}" in template:
                for process in process_terms:
                    queries.append(template.format(base=base, process=process))
            else:
                queries.append(template.format(base=base))

    deduped_queries = list(dict.fromkeys(query.strip() for query in queries if query.strip()))
    limits = config.get("limits", {})
    return QueryPlan(
        domain=str(config.get("domain", "general_research")),
        queries=deduped_queries,
        include_terms=[str(item).lower() for item in config.get("include_terms", [])],
        exclude_terms=[str(item).lower() for item in config.get("exclude_terms", [])],
        sources=sources or [str(item) for item in config.get("sources", [])],
        from_year=from_year or int(config.get("from_year", 1900)),
        to_year=to_year or int(config.get("to_year", 2026)),
        max_results_per_query=int(limits.get("max_results_per_query", 20)),
        request_timeout_seconds=int(limits.get("request_timeout_seconds", 30)),
        polite_delay_seconds=float(limits.get("polite_delay_seconds", 0.2)),
    )


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    key_stack: list[tuple[int, str]] = []

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if line.startswith("- "):
            item_text = line[2:].strip()
            if not isinstance(parent, list):
                raise ValueError(f"YAML list item has non-list parent: {raw_line}")
            parent.append(_parse_scalar(item_text))
            continue

        if ":" not in line:
            raise ValueError(f"Unsupported YAML line: {raw_line}")

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if value:
            parent[key] = _parse_scalar(value)
            continue

        next_container: Any = {}
        parent[key] = next_container
        key_stack.append((indent, key))
        stack.append((indent, next_container))

        # Convert empty mapping to list when the next indented non-empty line is a list.
        # This lightweight parser handles only the repository config shape.
        lines_after = text.splitlines()
        current_index = lines_after.index(raw_line)
        for future_line in lines_after[current_index + 1 :]:
            if not future_line.strip() or future_line.lstrip().startswith("#"):
                continue
            future_indent = len(future_line) - len(future_line.lstrip(" "))
            if future_indent <= indent:
                break
            if future_line.strip().startswith("- "):
                parent[key] = []
                stack[-1] = (indent, parent[key])
            break

    return root


def _parse_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip('"').strip("'")

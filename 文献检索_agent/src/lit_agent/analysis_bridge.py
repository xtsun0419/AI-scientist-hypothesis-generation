from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import project_root
from .db import LiteratureDB


def analysis_agent_dir() -> Path:
    return project_root().parent / "文献分析_agent"


def analysis_parsed_dir() -> Path:
    return analysis_agent_dir() / "data" / "parsed_papers"


def analysis_data_dir() -> Path:
    return analysis_agent_dir() / "data"


def analysis_graph_path() -> Path:
    return analysis_data_dir() / "graph" / "graph.json"


def convert_pdfs_with_analysis_agent(
    *,
    round_id: int | None = None,
    limit: int | None = None,
    force: bool = False,
) -> dict[str, int]:
    run_py = analysis_agent_dir() / "run.py"
    if not run_py.exists():
        raise RuntimeError(f"文献分析_agent 不存在：{analysis_agent_dir()}")
    cmd = [sys.executable, str(run_py), "convert-pdfs"]
    if round_id is not None:
        cmd.extend(["--round-id", str(round_id)])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if force:
        cmd.append("--force")
    result = subprocess.run(cmd, cwd=analysis_agent_dir(), text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "文献分析_agent 执行失败").strip())
    return _parse_conversion_output(result.stdout)


def build_all_with_analysis_agent() -> dict[str, Any]:
    result = _run_analysis_agent(["build-all", "--json"])
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return _parse_key_value_output(result.stdout)


def search_with_analysis_agent(query: str, *, limit: int = 10) -> list[dict[str, Any]]:
    if not query.strip():
        return []
    result = _run_analysis_agent(["search", query, "--limit", str(limit), "--json"])
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def conversion_queue_count(db: LiteratureDB) -> int:
    seen: set[str] = set()
    for row in db.pdf_assets_for_conversion(None):
        if row["pdf_path"]:
            seen.add(str(_resolve_pdf_path(row["pdf_path"])))
    for root in (project_root() / "data" / "pdfs", project_root() / "data" / "goal_pdfs"):
        if not root.exists():
            continue
        for path in root.rglob("*.pdf"):
            seen.add(str(path.resolve()))
    return len(seen)


def conversion_metrics(db: LiteratureDB) -> dict[str, int]:
    row = dict(db.paper_conversion_metrics())
    return {
        "pdf_total": conversion_queue_count(db),
        "converted": int(row.get("converted") or 0),
        "failed": int(row.get("failed") or 0),
        "skipped": int(row.get("skipped") or 0),
    }


def _parse_conversion_output(output: str) -> dict[str, int]:
    stats = {"total": 0, "converted": 0, "skipped": 0, "failed": 0}
    parsed = _parse_key_value_output(output)
    for name in stats:
        if name in parsed:
            stats[name] = int(parsed[name])
    return stats


def _parse_key_value_output(output: str) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith("- ") or ":" not in line:
            continue
        name, value = line[2:].split(":", 1)
        name = name.strip()
        value = value.strip()
        if value.isdigit():
            stats[name] = int(value)
        else:
            try:
                stats[name] = float(value)
            except ValueError:
                stats[name] = value
    return stats


def _run_analysis_agent(args: list[str]) -> subprocess.CompletedProcess[str]:
    run_py = analysis_agent_dir() / "run.py"
    if not run_py.exists():
        raise RuntimeError(f"文献分析_agent 不存在：{analysis_agent_dir()}")
    result = subprocess.run([sys.executable, str(run_py), *args], cwd=analysis_agent_dir(), text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "文献分析_agent 执行失败").strip())
    return result


def _resolve_pdf_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = project_root() / candidate
    return candidate.resolve()

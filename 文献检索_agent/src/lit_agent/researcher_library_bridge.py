from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .config import project_root
from .external_llm import apply_active_llm_to_env


def researcher_library_agent_dir() -> Path:
    return project_root().parent / "05_研究者反馈_agents"


def researcher_library_state() -> dict[str, Any]:
    agent = _agent()
    try:
        return agent.state()
    finally:
        agent.close()


def researcher_library_set_enabled(enabled: bool) -> dict[str, Any]:
    agent = _agent()
    try:
        return agent.set_enabled(enabled)
    finally:
        agent.close()


def researcher_library_import_path(path: str) -> dict[str, int]:
    agent = _agent()
    try:
        return agent.import_path(path)
    finally:
        agent.close()


def researcher_library_sync(retrieval_db_path: Path | None = None) -> dict[str, int]:
    agent = _agent()
    try:
        return agent.sync_agent_literature(retrieval_db_path)
    finally:
        agent.close()


def researcher_library_import_pubmed(ids: list[str], *, email: str | None = None) -> dict[str, int]:
    agent = _agent()
    try:
        return agent.import_pubmed(ids, email=email)
    finally:
        agent.close()


def researcher_library_ask() -> dict[str, Any]:
    apply_active_llm_to_env()
    agent = _agent()
    try:
        return agent.ask_researcher_question()
    finally:
        agent.close()


def researcher_library_run_hypothesis_dialogue() -> dict[str, Any]:
    apply_active_llm_to_env()
    agent = _agent()
    try:
        return agent.run_hypothesis_dialogue()
    finally:
        agent.close()


def _agent() -> Any:
    src_dir = researcher_library_agent_dir() / "src"
    if not src_dir.exists():
        raise RuntimeError(f"研究者文献库模块不存在：{researcher_library_agent_dir()}")
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from researcher_library_agent.agent import ResearcherLibraryAgent

    return ResearcherLibraryAgent()

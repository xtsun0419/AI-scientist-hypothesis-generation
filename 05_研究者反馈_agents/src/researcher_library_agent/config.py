from __future__ import annotations

from pathlib import Path


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def agent_dir() -> Path:
    return workspace_root() / "05_研究者反馈_agents"


def default_db_path() -> Path:
    return agent_dir() / "data" / "researcher_library.sqlite"


def default_retrieval_db_path() -> Path:
    return workspace_root() / "文献检索_agent" / "data" / "literature.sqlite"

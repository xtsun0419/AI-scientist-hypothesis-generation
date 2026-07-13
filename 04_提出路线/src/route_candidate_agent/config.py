from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def workspace_root() -> Path:
    return project_root().parent


def default_data_dir() -> Path:
    return project_root() / "data"


def default_output_path() -> Path:
    return default_data_dir() / "route_candidates.json"


def default_retrieval_agent_dir() -> Path:
    return workspace_root() / "文献检索_agent"


def default_question_synthesis_agent_dir() -> Path:
    return workspace_root() / "03_科学问题归纳_agents"

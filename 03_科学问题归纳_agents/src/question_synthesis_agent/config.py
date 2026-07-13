from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def workspace_root() -> Path:
    return project_root().parent


def default_data_dir() -> Path:
    return project_root() / "data"


def default_db_path() -> Path:
    return default_data_dir() / "question_synthesis.sqlite"


def default_retrieval_agent_dir() -> Path:
    return workspace_root() / "文献检索_agent"


def default_retrieval_db_path() -> Path:
    return default_retrieval_agent_dir() / "data" / "literature.sqlite"


def default_analysis_agent_dir() -> Path:
    return workspace_root() / "文献分析_agent"


def default_analysis_data_dir() -> Path:
    return default_analysis_agent_dir() / "data"


def default_local_env_path() -> Path:
    return default_retrieval_agent_dir() / "configs" / "local.env"

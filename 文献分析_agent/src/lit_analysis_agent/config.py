from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def workspace_root() -> Path:
    return project_root().parent


def default_retrieval_agent_dir() -> Path:
    return workspace_root() / "文献检索_agent"


def default_db_path() -> Path:
    return default_retrieval_agent_dir() / "data" / "literature.sqlite"


def default_pdf_dir() -> Path:
    return default_retrieval_agent_dir() / "data" / "pdfs"


def default_goal_pdf_dir() -> Path:
    return default_retrieval_agent_dir() / "data" / "goal_pdfs"


def default_parsed_dir() -> Path:
    return project_root() / "data" / "parsed_papers"


def default_index_dir() -> Path:
    return project_root() / "data" / "index"


def default_cards_dir() -> Path:
    return project_root() / "data" / "cards"


def default_graph_dir() -> Path:
    return project_root() / "data" / "graph"


def default_wiki_dir() -> Path:
    return project_root() / "data" / "wiki"

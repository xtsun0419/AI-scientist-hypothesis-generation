from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .config import project_root


def question_synthesis_agent_dir() -> Path:
    return project_root().parent / "03_科学问题归纳_agents"


def question_synthesis_data_dir() -> Path:
    return question_synthesis_agent_dir() / "data"


def question_synthesis_state() -> dict[str, Any]:
    QuestionSynthesisAgent, QuestionSynthesisDB, default_db_path = _load_question_synthesis()
    db = QuestionSynthesisDB(default_db_path())
    db.init_schema()
    try:
        return QuestionSynthesisAgent(db).state()
    finally:
        db.close()


def question_synthesis_chat(message: str) -> dict[str, Any]:
    QuestionSynthesisAgent, QuestionSynthesisDB, default_db_path = _load_question_synthesis()
    db = QuestionSynthesisDB(default_db_path())
    db.init_schema()
    try:
        return QuestionSynthesisAgent(db).chat(message)
    finally:
        db.close()


def question_synthesis_reset() -> dict[str, Any]:
    QuestionSynthesisAgent, QuestionSynthesisDB, default_db_path = _load_question_synthesis()
    db = QuestionSynthesisDB(default_db_path())
    db.init_schema()
    try:
        return QuestionSynthesisAgent(db).reset()
    finally:
        db.close()


def _load_question_synthesis() -> tuple[Any, Any, Any]:
    src_dir = question_synthesis_agent_dir() / "src"
    if not src_dir.exists():
        raise RuntimeError(f"03_科学问题归纳_agents 不存在：{question_synthesis_agent_dir()}")
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from question_synthesis_agent.agent import QuestionSynthesisAgent
    from question_synthesis_agent.config import default_db_path
    from question_synthesis_agent.db import QuestionSynthesisDB

    return QuestionSynthesisAgent, QuestionSynthesisDB, default_db_path

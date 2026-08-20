from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .config import project_root
from .external_llm import apply_active_llm_to_env


def route_candidate_agent_dir() -> Path:
    return project_root().parent / "04_提出路线"


def route_candidate_data_dir() -> Path:
    return route_candidate_agent_dir() / "data"


def route_candidate_state(selected_question_id: int | None = None) -> dict[str, Any]:
    RouteCandidateAgent = _load_route_candidate_agent()
    apply_active_llm_to_env()
    return RouteCandidateAgent().state(selected_question_id=selected_question_id)


def route_candidate_generate(
    *,
    question_id: int | None,
    route_count: int,
    emphasis: str,
) -> dict[str, Any]:
    RouteCandidateAgent = _load_route_candidate_agent()
    apply_active_llm_to_env()
    return RouteCandidateAgent().generate(question_id=question_id, route_count=route_count, emphasis=emphasis)


def route_candidate_critique(*, run_id: str | None = None) -> dict[str, Any]:
    RouteCandidateAgent = _load_route_candidate_agent()
    apply_active_llm_to_env()
    return RouteCandidateAgent().critique(run_id=run_id)


def route_candidate_evolve(*, run_id: str | None = None) -> dict[str, Any]:
    RouteCandidateAgent = _load_route_candidate_agent()
    apply_active_llm_to_env()
    return RouteCandidateAgent().evolve(run_id=run_id)


def _load_route_candidate_agent() -> Any:
    src_dir = route_candidate_agent_dir() / "src"
    if not src_dir.exists():
        raise RuntimeError(f"04_提出路线 不存在：{route_candidate_agent_dir()}")
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from route_candidate_agent.agent import RouteCandidateAgent

    return RouteCandidateAgent

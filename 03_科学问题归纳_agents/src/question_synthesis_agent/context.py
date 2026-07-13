from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from .config import default_analysis_data_dir, default_retrieval_db_path


def collect_context(
    *,
    retrieval_db_path: Path | None = None,
    analysis_data_dir: Path | None = None,
) -> dict[str, Any]:
    retrieval_db = retrieval_db_path or default_retrieval_db_path()
    analysis_dir = analysis_data_dir or default_analysis_data_dir()
    context: dict[str, Any] = {
        "latest_goal": None,
        "latest_round": None,
        "retrieval_questions": [],
        "evidence_gaps": [],
        "candidate_papers": [],
        "paper_cards": [],
        "wiki_pages": [],
        "metrics": {},
        "paths": {
            "retrieval_db": str(retrieval_db),
            "analysis_data": str(analysis_dir),
        },
    }
    if retrieval_db.exists():
        with sqlite3.connect(retrieval_db) as conn:
            conn.row_factory = sqlite3.Row
            context.update(_collect_retrieval(conn))
            context["paper_cards"] = _paper_cards_from_db(conn)
            context["wiki_pages"] = _wiki_pages_from_db(conn)
    if not context["paper_cards"]:
        context["paper_cards"] = _paper_cards_from_files(analysis_dir / "cards")
    if not context["wiki_pages"]:
        context["wiki_pages"] = _wiki_pages_from_files(analysis_dir / "wiki")
    context["metrics"] = {
        "retrieval_questions": len(context["retrieval_questions"]),
        "evidence_gaps": len(context["evidence_gaps"]),
        "candidate_papers": len(context["candidate_papers"]),
        "paper_cards": len(context["paper_cards"]),
        "wiki_pages": len(context["wiki_pages"]),
    }
    context["fingerprint"] = context_fingerprint(context)
    return context


def context_fingerprint(context: dict[str, Any]) -> str:
    slim = {
        "goal": context.get("latest_goal"),
        "round": context.get("latest_round"),
        "questions": context.get("retrieval_questions", [])[:12],
        "gaps": context.get("evidence_gaps", [])[:12],
        "cards": [
            {"paper_id": item.get("paper_id"), "title": item.get("title"), "summary": item.get("summary")}
            for item in context.get("paper_cards", [])[:10]
        ],
        "wiki": [
            {"title": item.get("title"), "summary": item.get("summary"), "open_questions": item.get("open_questions", [])[:3]}
            for item in context.get("wiki_pages", [])[:12]
        ],
    }
    payload = json.dumps(slim, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _collect_retrieval(conn: sqlite3.Connection) -> dict[str, Any]:
    latest_goal = _first_dict(conn, "SELECT * FROM scientific_goals ORDER BY updated_at DESC, id DESC LIMIT 1")
    latest_round = _first_dict(conn, "SELECT * FROM exploration_rounds ORDER BY updated_at DESC, id DESC LIMIT 1")
    latest_synthesis = _first_dict(
        conn,
        """
        SELECT rs.*, er.round_index, er.goal_id
        FROM round_syntheses rs
        JOIN exploration_rounds er ON er.id = rs.round_id
        ORDER BY rs.created_at DESC, rs.id DESC
        LIMIT 1
        """,
    )
    questions: list[str] = []
    evidence_gaps: list[str] = []
    if latest_goal:
        questions.append(str(latest_goal.get("title") or "").strip())
        if latest_goal.get("description"):
            questions.append(str(latest_goal["description"]).strip())
    if latest_synthesis:
        evidence_gaps.extend(_json_list(latest_synthesis.get("evidence_gaps_json"))[:8])
        questions.extend(_json_list(latest_synthesis.get("next_queries_json"))[:8])
    if not questions:
        questions.extend(_queries_from_latest_run(conn)[:8])
    candidate_papers = _candidate_papers(conn, int(latest_round["id"])) if latest_round else []
    return {
        "latest_goal": latest_goal,
        "latest_round": latest_round,
        "latest_synthesis": latest_synthesis,
        "retrieval_questions": [item for item in dict.fromkeys(questions) if item],
        "evidence_gaps": [item for item in dict.fromkeys(evidence_gaps) if item],
        "candidate_papers": candidate_papers,
    }


def _candidate_papers(conn: sqlite3.Connection, round_id: int) -> list[dict[str, Any]]:
    if not _table_exists(conn, "round_candidates"):
        return []
    rows = conn.execute(
        """
        SELECT rc.rank, rc.selection_score, rc.selection_reason, p.title, p.year, p.venue, p.doi
        FROM round_candidates rc
        JOIN papers p ON p.id = rc.paper_id
        WHERE rc.round_id = ?
        ORDER BY rc.rank, rc.id
        LIMIT 10
        """,
        (round_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def _paper_cards_from_db(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    if not _table_exists(conn, "paper_cards"):
        return []
    rows = conn.execute("SELECT card_json FROM paper_cards ORDER BY updated_at DESC, paper_id LIMIT 20").fetchall()
    return [_compact_card(_loads(row["card_json"], {})) for row in rows]


def _wiki_pages_from_db(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    if not _table_exists(conn, "wiki_pages"):
        return []
    rows = conn.execute("SELECT page_json FROM wiki_pages ORDER BY updated_at DESC, title LIMIT 24").fetchall()
    return [_compact_wiki(_loads(row["page_json"], {})) for row in rows]


def _paper_cards_from_files(cards_dir: Path) -> list[dict[str, Any]]:
    if not cards_dir.exists():
        return []
    cards = []
    for path in sorted(cards_dir.glob("*.json"))[:20]:
        cards.append(_compact_card(_loads(path.read_text(encoding="utf-8"), {})))
    return cards


def _wiki_pages_from_files(wiki_dir: Path) -> list[dict[str, Any]]:
    if not wiki_dir.exists():
        return []
    pages = []
    for path in sorted(wiki_dir.glob("*.json"))[:24]:
        pages.append(_compact_wiki(_loads(path.read_text(encoding="utf-8"), {})))
    return pages


def _compact_card(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": card.get("paper_id"),
        "title": card.get("title"),
        "doi": card.get("doi"),
        "year": card.get("year"),
        "summary": _truncate(str(card.get("summary") or ""), 520),
        "research_object": card.get("research_object"),
        "materials": list(card.get("materials") or [])[:8],
        "methods": list(card.get("methods") or [])[:8],
        "properties": list(card.get("properties") or [])[:8],
        "claims": [
            {
                "text": _truncate(str(item.get("text") or ""), 360),
                "evidence_ids": list(item.get("evidence_ids") or [])[:4],
            }
            for item in list(card.get("claims") or [])[:4]
        ],
        "limitations": [_truncate(str(item), 260) for item in list(card.get("limitations") or [])[:4]],
        "evidence_ids": list(card.get("evidence_ids") or [])[:8],
    }


def _compact_wiki(page: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": page.get("title"),
        "summary": _truncate(str(page.get("summary") or ""), 360),
        "known_findings": [
            {
                "title": item.get("title"),
                "claim": _truncate(str(item.get("claim") or ""), 320),
                "evidence_ids": list(item.get("evidence_ids") or [])[:4],
            }
            for item in list(page.get("known_findings") or [])[:4]
        ],
        "limitations": [_truncate(str(item), 220) for item in list(page.get("limitations") or [])[:4]],
        "open_questions": [_truncate(str(item), 220) for item in list(page.get("open_questions") or [])[:4]],
        "evidence_ids": list(page.get("evidence_ids") or [])[:8],
        "needs_evidence": list(page.get("needs_evidence") or [])[:4],
    }


def _queries_from_latest_run(conn: sqlite3.Connection) -> list[str]:
    if not _table_exists(conn, "search_runs"):
        return []
    row = conn.execute("SELECT query_plan_json FROM search_runs ORDER BY id DESC LIMIT 1").fetchone()
    if row is None:
        return []
    plan = _loads(row["query_plan_json"], {})
    return [str(item) for item in list(plan.get("queries") or [])]


def _first_dict(conn: sqlite3.Connection, query: str) -> dict[str, Any] | None:
    try:
        row = conn.execute(query).fetchone()
    except sqlite3.Error:
        return None
    return dict(row) if row is not None else None


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?", (table,)).fetchone()
    return row is not None


def _json_list(value: Any) -> list[str]:
    data = _loads(str(value or "[]"), [])
    if not isinstance(data, list):
        return []
    return [str(item) for item in data if str(item).strip()]


def _loads(text: str, default: Any) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return default


def _truncate(value: str, limit: int) -> str:
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "..."

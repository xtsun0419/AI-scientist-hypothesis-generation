from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable

from lit_agent.config import load_domain_config, query_plan_from_config
from lit_agent.constants import ACCESS_DOWNLOADED_OA_PDF, ACCESS_PREPRINT_PDF
from lit_agent.db import LiteratureDB
from lit_agent.llm import LLMSettings, OpenAICompatibleClient
from lit_agent.models import (
    ManualDownloadTask,
    PaperAnalysis,
    QueryPlan,
    RoundCandidate,
    RoundSynthesis,
    ScientificGoal,
)
from lit_agent.text import doi_url

from .dedup import DeduplicationAgent
from .discovery import SourceDiscoveryAgent
from .download import PdfDownloadAgent
from .normalize import MetadataNormalizeAgent
from .oa_resolver import OAResolverAgent
from .relevance import RelevanceJudgeAgent


ROUND_PLANNED = "planned"
ROUND_AWAITING_APPROVAL = "awaiting_user_approval"
ROUND_APPROVED = "approved"
ROUND_ACQUIRING = "acquiring_pdfs"
ROUND_AWAITING_MANUAL = "awaiting_manual_pdfs"
ROUND_ANALYZING = "analyzing"
ROUND_SYNTHESIZED = "synthesized"
ROUND_NEXT_PROPOSED = "next_round_proposed"
ROUND_NEEDS_RETRY = "needs_retry"

DEFAULT_TARGET_COUNT = 20
DEFAULT_MANUAL_ROOT = Path("data/goal_pdfs")
LLM_SELECTION_ENV = "LIT_AGENT_SELECTION_MODE"
LLM_SELECTION_MODES = {"llm", "hybrid", "external_llm"}
MAX_LLM_CANDIDATES = 60

_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]{2,}|[\u4e00-\u9fff]{2,}")
_GENERIC_STOPWORDS = {"about", "analysis", "and", "for", "from", "into", "research", "study", "the", "with"}


class ScientificGoalAgent:
    """Creates a structured scientific goal without requiring LangGraph."""

    def __init__(self, db: LiteratureDB):
        self.db = db

    def create(
        self,
        *,
        title: str,
        description: str | None = None,
        domain: str = "general_research",
        target_count: int = DEFAULT_TARGET_COUNT,
    ) -> int:
        terms = _goal_terms(title, description)
        goal = ScientificGoal(
            id=None,
            title=title,
            description=description,
            domain=domain,
            include_terms=terms["include"],
            exclude_terms=terms["exclude"],
            default_target_count=target_count,
        )
        return self.db.create_scientific_goal(goal)


class RoundPlanningAgent:
    """Builds a small candidate pool for one human-approved exploration round."""

    def __init__(self, db: LiteratureDB):
        self.db = db

    def plan(
        self,
        *,
        goal_id: int,
        target_count: int | None = None,
        max_results_per_query: int = 8,
        query_limit: int = 4,
        sources: list[str] | None = None,
    ) -> dict[str, int]:
        goal = self.db.scientific_goal(goal_id)
        target = target_count or int(goal["default_target_count"])
        round_id = self.db.create_exploration_round(goal_id, target, ROUND_PLANNED)
        plan = _query_plan_for_goal(goal, target, max_results_per_query, query_limit, sources)
        run_id = self.db.create_search_run(plan.domain, plan.from_year, plan.to_year, asdict(plan))
        try:
            discovery_counts = SourceDiscoveryAgent(self.db).run(plan, run_id=run_id)
            normalized = MetadataNormalizeAgent(self.db).run()
            papers = DeduplicationAgent(self.db).run()
            RelevanceJudgeAgent(self.db).run(plan)
            OAResolverAgent(self.db).run()
            selected = LiteratureSelectionAgent(self.db).select(round_id=round_id, goal=goal, target_count=target)
            if selected == 0:
                self.db.finish_search_run(run_id, "finished")
                self.db.update_round_status(round_id, ROUND_NEEDS_RETRY)
                return {
                    "round_id": round_id,
                    "search_run_id": run_id,
                    "raw_records": sum(discovery_counts.values()),
                    "normalized": normalized,
                    "papers": papers,
                    "selected": selected,
                }
            self.db.finish_search_run(run_id, "finished")
            self.db.update_round_status(round_id, ROUND_AWAITING_APPROVAL)
            return {
                "round_id": round_id,
                "search_run_id": run_id,
                "raw_records": sum(discovery_counts.values()),
                "normalized": normalized,
                "papers": papers,
                "selected": selected,
            }
        except Exception:
            self.db.finish_search_run(run_id, "failed")
            self.db.update_round_status(round_id, ROUND_NEEDS_RETRY)
            raise


class LiteratureSelectionAgent:
    """Selects a diverse, core set of papers for a round."""

    def __init__(
        self,
        db: LiteratureDB,
        *,
        settings: LLMSettings | None = None,
        client_factory: Callable[[LLMSettings], OpenAICompatibleClient] = OpenAICompatibleClient,
        selection_mode: str | None = None,
    ):
        self.db = db
        self.settings = settings if settings is not None else LLMSettings.from_env()
        self.client_factory = client_factory
        self.selection_mode = (selection_mode or os.environ.get(LLM_SELECTION_ENV, "rules")).strip().lower()

    def select(self, *, round_id: int, goal: Any, target_count: int) -> int:
        self.db.clear_round_candidates(round_id)
        rows = self.db.rows(
            """
            SELECT p.*, ar.access_status, ar.pdf_url, ar.is_oa
            FROM papers p
            LEFT JOIN access_records ar ON ar.paper_id = p.id
            ORDER BY p.id DESC
            """
        )
        scored = []
        for row in rows:
            score, reason, tags = _selection_score(row, goal)
            if score <= 0:
                continue
            scored.append((score, row, reason, tags))
        selected = self._select_candidates(scored, goal, target_count)
        for rank, (score, row, reason, tags) in enumerate(selected, start=1):
            self.db.upsert_round_candidate(
                RoundCandidate(
                    round_id=round_id,
                    paper_id=row["id"],
                    rank=rank,
                    selection_score=score,
                    selection_reason=reason,
                    material_tags=tags,
                    evidence_level=_evidence_level(row),
                )
            )
        return len(selected)

    def _select_candidates(
        self,
        scored: list[tuple[float, Any, str, list[str]]],
        goal: Any,
        target_count: int,
    ) -> list[tuple[float, Any, str, list[str]]]:
        if self.selection_mode not in LLM_SELECTION_MODES:
            return _diverse_top(scored, target_count)
        if not self.settings:
            selected = _diverse_top(scored, target_count)
            return [
                (score, row, f"规则推荐（LLM 未配置）: {reason}", tags)
                for score, row, reason, tags in selected
            ]
        try:
            return self._llm_rerank(scored, goal, target_count)
        except Exception as exc:
            selected = _diverse_top(scored, target_count)
            return [
                (score, row, f"规则兜底（LLM 推荐失败: {_short_error(exc)}）: {reason}", tags)
                for score, row, reason, tags in selected
            ]

    def _llm_rerank(
        self,
        scored: list[tuple[float, Any, str, list[str]]],
        goal: Any,
        target_count: int,
    ) -> list[tuple[float, Any, str, list[str]]]:
        candidate_pool = _diverse_top(scored, max(target_count * 3, min(MAX_LLM_CANDIDATES, len(scored))))
        candidate_pool = candidate_pool[:MAX_LLM_CANDIDATES]
        if not candidate_pool:
            return []
        by_id = {int(row["id"]): (score, row, reason, tags) for score, row, reason, tags in candidate_pool}
        prompt = _llm_selection_prompt(goal, target_count, candidate_pool)
        response = self.client_factory(self.settings).recommend_literature(prompt)
        selected: list[tuple[float, Any, str, list[str]]] = []
        used: set[int] = set()
        raw_items = response.get("selected", [])
        if not isinstance(raw_items, list):
            raw_items = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            paper_id = _safe_int(item.get("paper_id"))
            if paper_id is None or paper_id not in by_id or paper_id in used:
                continue
            rule_score, row, rule_reason, rule_tags = by_id[paper_id]
            score = _safe_float(item.get("score"), rule_score)
            reason = _clean_text(item.get("reason")) or rule_reason
            tags = _clean_list(item.get("topic_tags") or item.get("material_tags")) or rule_tags
            selected.append((score, row, f"LLM推荐: {reason}", tags))
            used.add(paper_id)
            if len(selected) >= target_count:
                break
        if len(selected) < target_count:
            for score, row, reason, tags in _diverse_top(scored, target_count):
                paper_id = int(row["id"])
                if paper_id in used:
                    continue
                selected.append((score, row, f"规则补足: {reason}", tags))
                used.add(paper_id)
                if len(selected) >= target_count:
                    break
        return selected[:target_count]


class RoundApprovalAgent:
    def __init__(self, db: LiteratureDB):
        self.db = db

    def approve(self, round_id: int) -> None:
        round_row = self.db.exploration_round(round_id)
        if round_row["status"] != ROUND_AWAITING_APPROVAL:
            raise ValueError(f"Round {round_id} is not awaiting approval: {round_row['status']}")
        self.db.update_round_status(round_id, ROUND_APPROVED, approved=True)


class RoundAcquisitionAgent:
    """Downloads approved round PDFs and creates manual DOI tasks for the rest."""

    def __init__(self, db: LiteratureDB, pdf_dir: Path, manual_root: Path | None = None):
        self.db = db
        self.pdf_dir = pdf_dir
        self.manual_root = manual_root or DEFAULT_MANUAL_ROOT

    def acquire(self, round_id: int) -> dict[str, int]:
        round_row = self.db.exploration_round(round_id)
        if round_row["status"] != ROUND_APPROVED:
            raise ValueError(f"Round {round_id} must be approved before acquisition.")
        self.db.update_round_status(round_id, ROUND_ACQUIRING)
        before = _downloaded_paper_ids(self.db)
        records = self.db.access_records_for_round_download(round_id)
        PdfDownloadAgent(self.db, self.pdf_dir).run_records(records, limit=int(round_row["target_count"]))
        copied = self._sync_available_pdfs_to_round_folder(round_id)
        after = _downloaded_paper_ids(self.db)
        created_tasks = self._create_manual_tasks(round_id)
        next_status = ROUND_AWAITING_MANUAL if created_tasks else ROUND_ANALYZING
        self.db.update_round_status(round_id, next_status)
        round_downloaded = _downloaded_paper_ids_for_round(self.db, round_id)
        return {
            "downloaded": len(after - before),
            "round_downloaded": len(round_downloaded),
            "copied_to_round_folder": copied,
            "round_pdf_dir": str(self._round_pdf_dir(round_id)),
            "manual_tasks": created_tasks,
        }

    def _sync_available_pdfs_to_round_folder(self, round_id: int) -> int:
        copied = 0
        round_dir = self._round_pdf_dir(round_id)
        round_dir.mkdir(parents=True, exist_ok=True)
        for row in self.db.round_candidates(round_id):
            source = _latest_pdf_asset_for_paper(self.db, int(row["paper_id"]))
            if source is None:
                continue
            source_path = Path(str(source["file_path"]))
            if not source_path.exists() or source_path.stat().st_size <= 0:
                continue
            target_path = round_dir / _manual_filename(row)
            if source_path.resolve() != target_path.resolve() and not target_path.exists():
                shutil.copy2(source_path, target_path)
                copied += 1
            elif target_path.exists() and target_path.stat().st_size > 0:
                copied += 1
            self.db.upsert_pdf_asset(
                paper_id=row["paper_id"],
                pdf_url=f"round-copy://round_{round_id}/{target_path.name}",
                file_path=str(target_path),
                sha256=source["sha256"],
                file_size=target_path.stat().st_size,
                status=source["status"],
                error_message=None,
            )
        return copied

    def _create_manual_tasks(self, round_id: int) -> int:
        created = 0
        manual_dir = self._round_pdf_dir(round_id)
        manual_dir.mkdir(parents=True, exist_ok=True)
        downloaded = _downloaded_paper_ids(self.db)
        for row in self.db.round_candidates(round_id):
            if row["paper_id"] in downloaded:
                continue
            reason = "PDF was not downloaded automatically; use DOI or publisher page and place the PDF at target_path."
            target_path = manual_dir / _manual_filename(row)
            self.db.upsert_manual_download_task(
                ManualDownloadTask(
                    round_id=round_id,
                    paper_id=row["paper_id"],
                    doi=row["doi"],
                    doi_url=row["doi_url"] or doi_url(row["doi"]),
                    publisher_url=row["publisher_url"],
                    target_path=str(target_path),
                    status="pending",
                    reason=reason,
                )
            )
            created += 1
        return created

    def _round_pdf_dir(self, round_id: int) -> Path:
        round_row = self.db.exploration_round(round_id)
        return self.manual_root / f"goal_{round_row['goal_id']}" / f"round_{round_id}"


class ManualPdfIntakeAgent:
    """Binds manually downloaded PDFs back to round papers."""

    def __init__(self, db: LiteratureDB):
        self.db = db

    def run(self, round_id: int) -> int:
        count = 0
        tasks = self.db.manual_download_tasks(round_id)
        used_paths: set[Path] = set()
        for task in tasks:
            matched_path = _match_manual_pdf(task, tasks, used_paths)
            if matched_path is None:
                continue
            canonical_path = Path(task["target_path"])
            path = _canonicalize_manual_pdf(matched_path, canonical_path)
            used_paths.add(path.resolve())
            self.db.upsert_pdf_asset(
                paper_id=task["paper_id"],
                pdf_url=f"manual://{path.name}",
                file_path=str(path),
                sha256=None,
                file_size=path.stat().st_size,
                status=ACCESS_DOWNLOADED_OA_PDF,
                error_message=None,
            )
            self.db.update_manual_download_task_status(task["id"], "completed")
            count += 1
        return count


class PdfAnalysisAgent:
    """Creates deterministic first-pass analyses from PDFs or metadata."""

    def __init__(self, db: LiteratureDB):
        self.db = db

    def analyze(self, round_id: int) -> int:
        self.db.update_round_status(round_id, ROUND_ANALYZING)
        count = 0
        downloaded = _downloaded_paper_ids(self.db)
        for row in self.db.round_candidates(round_id):
            analysis_type = "pdf" if row["paper_id"] in downloaded else "metadata"
            evidence_level = "high" if analysis_type == "pdf" else "low"
            text = " ".join(str(row[key] or "") for key in ["title", "abstract", "venue"])
            tags = _topic_tags(text, _goal_terms(str(row["title"] or ""), str(row["abstract"] or ""))["include"])
            findings = _findings_from_text(text, tags)
            summary = f"{row['title']} is treated as {evidence_level}-evidence for this round based on {analysis_type}."
            self.db.upsert_paper_analysis(
                PaperAnalysis(
                    round_id=round_id,
                    paper_id=row["paper_id"],
                    analysis_type=analysis_type,
                    evidence_level=evidence_level,
                    summary=summary,
                    key_findings=findings,
                    limitations=[] if analysis_type == "pdf" else ["Only metadata was available; do not use as strong evidence."],
                    next_search_terms=tags[:4],
                )
            )
            count += 1
        return count


class EvidenceSynthesisAgent:
    def __init__(self, db: LiteratureDB):
        self.db = db

    def synthesize(self, round_id: int) -> RoundSynthesis:
        analyses = self.db.paper_analyses(round_id)
        high = sum(1 for row in analyses if row["evidence_level"] == "high")
        low = sum(1 for row in analyses if row["evidence_level"] == "low")
        terms = _collect_terms(analyses)
        gaps = [
            "More full-text PDFs are needed for low-evidence papers.",
            "Search should diversify across materials and mechanisms not yet covered.",
        ]
        next_queries = _next_queries_from_terms(terms)
        synthesis = RoundSynthesis(
            round_id=round_id,
            summary=f"Round {round_id} analyzed {len(analyses)} papers: {high} high-evidence PDFs and {low} metadata-only records.",
            evidence_gaps=gaps,
            next_queries=next_queries,
        )
        self.db.upsert_round_synthesis(synthesis)
        self.db.update_round_status(round_id, ROUND_SYNTHESIZED)
        return synthesis


class NextQueryProposalAgent:
    def __init__(self, db: LiteratureDB):
        self.db = db

    def propose(self, round_id: int) -> dict[str, Any]:
        synthesis = self.db.round_synthesis(round_id)
        if synthesis is None:
            synthesis_obj = EvidenceSynthesisAgent(self.db).synthesize(round_id)
            queries = synthesis_obj.next_queries
        else:
            queries = json.loads(synthesis["next_queries_json"])
        self.db.update_round_status(round_id, ROUND_NEXT_PROPOSED)
        return {"round_id": round_id, "next_queries": queries}


def _goal_terms(title: str, description: str | None) -> dict[str, list[str]]:
    text = f"{title} {description or ''}"
    include = []
    for match in _TERM_RE.finditer(text):
        term = match.group(0).strip()
        if len(term) >= 3 and term.lower() not in _GENERIC_STOPWORDS and term not in include:
            include.append(term)
    return {"include": include[:12], "exclude": []}


def _query_plan_for_goal(
    goal: Any,
    target_count: int,
    max_results_per_query: int,
    query_limit: int,
    sources: list[str] | None,
) -> QueryPlan:
    base = query_plan_from_config(load_domain_config())
    include_terms = json.loads(goal["include_terms_json"])
    title = str(goal["title"])
    queries = [title]
    for term in include_terms:
        queries.append(f"{title} {term}")
    if len(queries) < query_limit:
        queries.extend(base.queries)
    return replace(
        base,
        domain=goal["domain"],
        queries=list(dict.fromkeys(queries))[:query_limit],
        include_terms=include_terms,
        exclude_terms=json.loads(goal["exclude_terms_json"]),
        sources=sources or base.sources,
        max_results_per_query=max(max_results_per_query, min(target_count, 20)),
    )


def _selection_score(row: Any, goal: Any) -> tuple[float, str, list[str]]:
    text = " ".join(str(row[key] or "") for key in ["title", "abstract", "venue", "document_type"]).lower()
    include_terms = json.loads(goal["include_terms_json"])
    matched = [term for term in include_terms if term.lower() in text]
    tags = _topic_tags(text, include_terms)
    score = float(row["relevance_score"] or 0.2)
    score += 0.12 * len(matched)
    score += 0.08 * min(len(tags), 3)
    if row["doi"]:
        score += 0.05
    if row["pdf_url"] or row["access_status"] in {ACCESS_DOWNLOADED_OA_PDF, ACCESS_PREPRINT_PDF, "oa_pdf_available"}:
        score += 0.08
    if row["year"]:
        score += max(0, min(0.08, (int(row["year"]) - 2000) / 400))
    reason = "Matched goal terms: " + ", ".join(matched[:5]) if matched else "Selected as a diversity/core candidate."
    return score, reason, tags or ["General"]


def _llm_selection_prompt(
    goal: Any,
    target_count: int,
    candidate_pool: list[tuple[float, Any, str, list[str]]],
) -> str:
    payload = {
        "task": "Select the best literature records for the next small, human-approved exploration round.",
        "constraints": [
            "Use only paper_id values in candidates.",
            "Do not invent missing metadata, DOI, or PDF availability.",
            "Prefer a useful mix of core papers, recent papers, and topic diversity.",
            "Prefer papers that directly inform the scientific goal over generic records.",
            "Return at most target_count records.",
        ],
        "output_schema": {
            "selected": [
                {
                    "paper_id": "integer candidate id",
                    "score": "0-1 relevance/usefulness score",
                    "reason": "concise reason grounded in title/abstract/metadata",
                    "topic_tags": ["short topic or method tags"],
                }
            ]
        },
        "scientific_goal": {
            "title": goal["title"],
            "description": goal["description"],
            "include_terms": json.loads(goal["include_terms_json"]),
            "exclude_terms": json.loads(goal["exclude_terms_json"]),
            "target_count": target_count,
        },
        "candidates": [_candidate_for_llm(item) for item in candidate_pool],
    }
    return json.dumps(payload, ensure_ascii=False)


def _candidate_for_llm(item: tuple[float, Any, str, list[str]]) -> dict[str, Any]:
    score, row, reason, tags = item
    return {
        "paper_id": int(row["id"]),
        "title": row["title"],
        "year": row["year"],
        "venue": row["venue"],
        "doi": row["doi"],
        "abstract": _truncate(str(row["abstract"] or ""), 700),
        "document_type": row["document_type"],
        "rule_score": round(float(score), 4),
        "rule_reason": reason,
        "topic_tags": tags,
        "access_status": row["access_status"],
        "has_pdf_url": bool(row["pdf_url"]),
        "is_oa": bool(row["is_oa"]) if row["is_oa"] is not None else None,
    }


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if number < 0:
        return 0.0
    if number > 1:
        return min(number, float(default))
    return number


def _clean_text(value: Any, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return _truncate(text, limit)


def _clean_list(value: Any) -> list[str]:
    if isinstance(value, list):
        values = value
    elif value:
        values = [value]
    else:
        values = []
    cleaned = [_clean_text(item, 40) for item in values]
    return [item for item in cleaned if item][:8]


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _short_error(exc: Exception) -> str:
    return _clean_text(str(exc), 90)


def _diverse_top(scored: list[tuple[float, Any, str, list[str]]], target_count: int):
    scored = sorted(scored, key=lambda item: item[0], reverse=True)
    selected = []
    tag_counts: dict[str, int] = {}
    for item in scored:
        primary = item[3][0] if item[3] else "General"
        if tag_counts.get(primary, 0) >= max(3, target_count // 4) and len(selected) < target_count // 2:
            continue
        selected.append(item)
        tag_counts[primary] = tag_counts.get(primary, 0) + 1
        if len(selected) >= target_count:
            break
    if len(selected) < target_count:
        selected_ids = {item[1]["id"] for item in selected}
        for item in scored:
            if item[1]["id"] not in selected_ids:
                selected.append(item)
            if len(selected) >= target_count:
                break
    return selected


def _evidence_level(row: Any) -> str:
    if row["access_status"] in {ACCESS_DOWNLOADED_OA_PDF, ACCESS_PREPRINT_PDF}:
        return "high"
    if row["pdf_url"]:
        return "medium"
    return "low"


def _topic_tags(text: str, terms: list[str]) -> list[str]:
    lower = text.lower()
    return [term for term in terms if term.lower() in lower][:8]


def _findings_from_text(text: str, tags: list[str]) -> list[str]:
    findings = []
    for tag in tags[:4]:
        findings.append(f"Mentions {tag}.")
    return findings or ["Relevant metadata was captured for follow-up."]


def _collect_terms(analyses: list[Any]) -> list[str]:
    terms = []
    for row in analyses:
        terms.extend(json.loads(row["next_search_terms_json"]))
    return list(dict.fromkeys(term for term in terms if term))


def _next_queries_from_terms(terms: list[str]) -> list[str]:
    return list(dict.fromkeys(term for term in terms if term))[:8]


def _downloaded_paper_ids(db: LiteratureDB) -> set[int]:
    rows = db.rows(
        """
        SELECT DISTINCT paper_id
        FROM pdf_assets
        WHERE status IN (?, ?) AND file_path IS NOT NULL
        """,
        (ACCESS_DOWNLOADED_OA_PDF, ACCESS_PREPRINT_PDF),
    )
    return {int(row["paper_id"]) for row in rows}


def _downloaded_paper_ids_for_round(db: LiteratureDB, round_id: int) -> set[int]:
    rows = db.rows(
        """
        SELECT DISTINCT rc.paper_id
        FROM round_candidates rc
        JOIN pdf_assets pa ON pa.paper_id = rc.paper_id
        WHERE rc.round_id = ?
          AND pa.status IN (?, ?)
          AND pa.file_path IS NOT NULL
        """,
        (round_id, ACCESS_DOWNLOADED_OA_PDF, ACCESS_PREPRINT_PDF),
    )
    return {int(row["paper_id"]) for row in rows}


def _latest_pdf_asset_for_paper(db: LiteratureDB, paper_id: int) -> Any | None:
    rows = db.rows(
        """
        SELECT *
        FROM pdf_assets
        WHERE paper_id = ?
          AND status IN (?, ?)
          AND file_path IS NOT NULL
        ORDER BY downloaded_at DESC, id DESC
        LIMIT 1
        """,
        (paper_id, ACCESS_DOWNLOADED_OA_PDF, ACCESS_PREPRINT_PDF),
    )
    return rows[0] if rows else None


def _match_manual_pdf(task: Any, all_tasks: list[Any], used_paths: set[Path]) -> Path | None:
    target = Path(task["target_path"])
    if target.exists() and target.stat().st_size > 0:
        return target
    folder = target.parent
    if not folder.exists():
        return None
    available = [
        path
        for path in sorted(folder.glob("*.pdf"))
        if path.stat().st_size > 0 and path.resolve() not in used_paths
    ]
    if not available:
        return None
    expected_stem = target.stem.lower()
    for path in available:
        if path.stem.lower() == expected_stem:
            return path
    doi = str(task["doi"] or "")
    doi_tokens = _doi_tokens(doi)
    for path in available:
        name = path.stem.lower()
        if doi_tokens and all(token in name for token in doi_tokens[:2]):
            return path
    title_terms = _title_terms(str(task["title"] or ""))
    if title_terms:
        scored = sorted(
            ((len(title_terms & _title_terms(path.stem)), path) for path in available),
            key=lambda item: item[0],
            reverse=True,
        )
        if scored and scored[0][0] >= max(2, min(4, len(title_terms) // 3)):
            return scored[0][1]
    pending_count = sum(1 for item in all_tasks if item["status"] != "completed")
    if len(available) == 1 and pending_count == 1:
        return available[0]
    return None


def _canonicalize_manual_pdf(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == target.resolve():
        return target
    if target.exists():
        return target
    shutil.copy2(source, target)
    return target


def _manual_filename(row: Any) -> str:
    ident = row["doi"] or str(row["paper_id"])
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", ident).strip("_")
    return f"paper_{row['paper_id']}_{safe}.pdf"


def _doi_tokens(doi: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", doi.lower()) if len(token) >= 3]


def _title_terms(text: str) -> set[str]:
    stop = {"the", "and", "for", "with", "from", "into", "study", "effect", "using", "based"}
    return {
        token
        for token in re.split(r"[^a-z0-9]+", text.lower())
        if len(token) >= 4 and token not in stop
    }

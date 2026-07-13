from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class QueryPlan:
    domain: str
    queries: list[str]
    include_terms: list[str]
    exclude_terms: list[str]
    sources: list[str]
    from_year: int
    to_year: int
    max_results_per_query: int = 20
    request_timeout_seconds: int = 30
    polite_delay_seconds: float = 0.2


@dataclass(frozen=True)
class RawSourceRecord:
    source: str
    source_id: str
    query: str
    raw_payload: dict[str, Any]
    retrieved_at: str = field(default_factory=utc_now_iso)


@dataclass
class PaperCandidate:
    source_record_id: int | None
    source: str
    source_id: str
    query: str
    doi: str | None
    title: str
    authors: list[str]
    year: int | None
    venue: str | None
    abstract: str | None
    publisher: str | None
    publisher_url: str | None
    source_url: str | None
    pdf_url: str | None
    is_oa: bool | None
    document_type: str | None
    raw_payload: dict[str, Any]


@dataclass
class PaperRecord:
    id: int | None
    doi: str | None
    title: str
    normalized_title: str
    authors: list[str]
    year: int | None
    venue: str | None
    abstract: str | None
    publisher: str | None
    publisher_url: str | None
    source_url: str | None
    document_type: str | None
    relevance_score: float | None = None
    relevance_terms: list[str] = field(default_factory=list)
    relevance_reason: str | None = None


@dataclass
class AccessRecord:
    paper_id: int
    doi: str | None
    doi_url: str | None
    is_oa: bool | None
    pdf_url: str | None
    publisher_url: str | None
    source_url: str | None
    access_status: str
    resolved_at: str = field(default_factory=utc_now_iso)


@dataclass
class AuditFinding:
    paper_id: int | None
    severity: str
    issue_type: str
    message: str
    suggestion: str
    created_at: str = field(default_factory=utc_now_iso)


@dataclass
class SourceFailure:
    run_id: int | None
    source: str
    query: str
    failure_type: str
    message: str
    attempt: int
    created_at: str = field(default_factory=utc_now_iso)


@dataclass
class LLMRelevanceReview:
    paper_id: int
    provider: str
    model: str
    decision: str
    confidence: float | None
    reason: str
    matched_domain_terms: list[str]
    exclude_reason: str | None
    raw_response: dict[str, Any]
    reviewed_at: str = field(default_factory=utc_now_iso)


@dataclass
class ScientificGoal:
    id: int | None
    title: str
    description: str | None
    domain: str
    include_terms: list[str]
    exclude_terms: list[str]
    default_target_count: int
    status: str = "active"
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)


@dataclass
class ExplorationRound:
    id: int | None
    goal_id: int
    round_index: int
    status: str
    target_count: int
    approved_at: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)


@dataclass
class RoundCandidate:
    round_id: int
    paper_id: int
    rank: int
    selection_score: float
    selection_reason: str
    material_tags: list[str]
    evidence_level: str
    status: str = "selected"
    created_at: str = field(default_factory=utc_now_iso)


@dataclass
class ManualDownloadTask:
    round_id: int
    paper_id: int
    doi: str | None
    doi_url: str | None
    publisher_url: str | None
    target_path: str
    status: str
    reason: str
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)


@dataclass
class PaperAnalysis:
    round_id: int
    paper_id: int
    analysis_type: str
    evidence_level: str
    summary: str
    key_findings: list[str]
    limitations: list[str]
    next_search_terms: list[str]
    created_at: str = field(default_factory=utc_now_iso)


@dataclass
class RoundSynthesis:
    round_id: int
    summary: str
    evidence_gaps: list[str]
    next_queries: list[str]
    created_at: str = field(default_factory=utc_now_iso)

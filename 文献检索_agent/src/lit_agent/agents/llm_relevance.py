from __future__ import annotations

import json

from lit_agent.db import LiteratureDB
from lit_agent.llm import LLMSettings, OpenAICompatibleClient
from lit_agent.models import LLMRelevanceReview, QueryPlan


class LLMRelevanceReviewAgent:
    """Reviews only boundary relevance cases with an OpenAI-compatible LLM."""

    def __init__(
        self,
        db: LiteratureDB,
        *,
        settings: LLMSettings | None = None,
        low_threshold: float = 0.3,
        high_threshold: float = 0.7,
    ):
        self.db = db
        self.settings = settings if settings is not None else LLMSettings.from_env()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def run(self, plan: QueryPlan | None = None) -> dict[str, int]:
        candidates = self.boundary_candidates()
        if not self.settings:
            for row in candidates:
                self.db.upsert_llm_relevance_review(
                    LLMRelevanceReview(
                        paper_id=row["id"],
                        provider="openai_compatible",
                        model="not_configured",
                        decision="skipped_no_api_key",
                        confidence=None,
                        reason="OPENAI_API_KEY or OPENAI_MODEL is not set; skipped LLM relevance review.",
                        matched_domain_terms=[],
                        exclude_reason=None,
                        raw_response={"skipped": True, "reason": "missing_openai_env"},
                    )
                )
            return {"reviewed": 0, "skipped_no_api_key": len(candidates), "candidates": len(candidates)}
        client = OpenAICompatibleClient(self.settings)
        reviewed = 0
        failed = 0
        for row in candidates:
            try:
                response = client.review_relevance(_prompt_for(row, plan))
                review = _review_from_response(row["id"], self.settings.model, response)
                self.db.upsert_llm_relevance_review(review)
                reviewed += 1
            except Exception as exc:
                self.db.upsert_llm_relevance_review(
                    LLMRelevanceReview(
                        paper_id=row["id"],
                        provider="openai_compatible",
                        model=self.settings.model,
                        decision="review_failed",
                        confidence=None,
                        reason=str(exc),
                        matched_domain_terms=[],
                        exclude_reason="LLM review failed",
                        raw_response={"error": str(exc)},
                    )
                )
                failed += 1
        return {"reviewed": reviewed, "failed": failed, "candidates": len(candidates)}

    def boundary_candidates(self):
        return self.db.rows(
            """
            SELECT p.*, ar.access_status
            FROM papers p
            LEFT JOIN access_records ar ON ar.paper_id = p.id
            WHERE (
                p.relevance_score BETWEEN ? AND ?
                OR p.abstract IS NULL
                OR p.relevance_reason LIKE '%manual review%'
            )
            ORDER BY p.relevance_score ASC, p.id
            """,
            (self.low_threshold, self.high_threshold),
        )


def _prompt_for(row, plan: QueryPlan | None = None) -> str:
    scope = {
        "domain": plan.domain if plan else "general_research",
        "include_terms": plan.include_terms if plan else [],
        "exclude_terms": plan.exclude_terms if plan else [],
        "queries": plan.queries[:8] if plan else [],
    }
    payload = {
        "research_scope": scope,
        "title": row["title"],
        "abstract": row["abstract"],
        "venue": row["venue"],
        "year": row["year"],
        "doi": row["doi"],
        "rule_relevance_score": row["relevance_score"],
        "rule_relevance_reason": row["relevance_reason"],
    }
    return (
        "Decide whether this paper belongs to the configured research scope. "
        "decision must be one of include, exclude, uncertain. confidence must be 0-1. "
        "Use concise reasons and do not invent facts.\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def _review_from_response(paper_id: int, model: str, response: dict) -> LLMRelevanceReview:
    decision = str(response.get("decision", "uncertain")).lower()
    if decision not in {"include", "exclude", "uncertain"}:
        decision = "uncertain"
    confidence = response.get("confidence")
    try:
        confidence = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None
    matched_terms = response.get("matched_domain_terms") or []
    if not isinstance(matched_terms, list):
        matched_terms = [str(matched_terms)]
    return LLMRelevanceReview(
        paper_id=paper_id,
        provider="openai_compatible",
        model=model,
        decision=decision,
        confidence=confidence,
        reason=str(response.get("reason", "")),
        matched_domain_terms=[str(term) for term in matched_terms],
        exclude_reason=response.get("exclude_reason"),
        raw_response=response,
    )

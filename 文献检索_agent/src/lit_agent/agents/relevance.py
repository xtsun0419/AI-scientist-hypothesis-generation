from __future__ import annotations

import json

from lit_agent.db import LiteratureDB
from lit_agent.models import QueryPlan


class RelevanceJudgeAgent:
    """Rules-first relevance scorer with a reserved hook for future LLM review."""

    def __init__(self, db: LiteratureDB):
        self.db = db

    def run(self, plan: QueryPlan) -> int:
        updated = 0
        include_terms = [term.lower() for term in plan.include_terms]
        exclude_terms = [term.lower() for term in plan.exclude_terms]
        for row in self.db.papers():
            text = " ".join(
                str(value or "")
                for value in [
                    row["title"],
                    row["abstract"],
                    row["venue"],
                    row["publisher"],
                    row["document_type"],
                ]
            ).lower()
            matched = [term for term in include_terms if term in text]
            excluded = [term for term in exclude_terms if term in text]
            score = min(1.0, 0.25 + 0.2 * len(matched))
            if not matched:
                score = 0.2
            if excluded:
                score = min(score, 0.35)
            reason = _reason(matched, excluded)
            self.db.update_relevance(row["id"], score, matched, reason)
            updated += 1
        return updated


def _reason(matched: list[str], excluded: list[str]) -> str:
    if excluded:
        return "Excluded or down-weighted by terms: " + ", ".join(excluded)
    if matched:
        return "Matched configured research terms: " + ", ".join(matched[:8])
    return "No configured research term was found; needs manual review."

from __future__ import annotations

import time

from lit_agent.db import LiteratureDB
from lit_agent.errors import classify_failure, retryable_failure
from lit_agent.models import QueryPlan
from lit_agent.models import SourceFailure
from lit_agent.sources import SourceConnector, build_connectors


SOURCE_POLITE_DELAYS = {
    "semantic_scholar": 2.0,
    "arxiv": 3.0,
    "crossref": 0.7,
    "openalex": 0.4,
    "doaj": 0.7,
    "core": 1.5,
    "europe_pmc": 0.7,
}


class SourceDiscoveryAgent:
    """Runs source connectors and persists raw source records."""

    def __init__(
        self,
        db: LiteratureDB,
        connectors: dict[str, SourceConnector] | None = None,
        *,
        retry_attempts: int = 3,
        base_backoff_seconds: float = 1.0,
    ):
        self.db = db
        self.connectors = connectors
        self.retry_attempts = retry_attempts
        self.base_backoff_seconds = base_backoff_seconds

    def run(self, plan: QueryPlan, *, run_id: int | None = None) -> dict[str, int]:
        connectors = self.connectors or build_connectors(plan)
        counts: dict[str, int] = {name: 0 for name in connectors}
        for query in plan.queries:
            for name, connector in connectors.items():
                self._source_delay(name)
                records = self._search_with_retry(connector, query, plan, run_id=run_id)
                for record in records:
                    self.db.insert_source_record(run_id, record)
                    counts[name] = counts.get(name, 0) + 1
        return counts

    def _search_with_retry(self, connector: SourceConnector, query: str, plan: QueryPlan, *, run_id: int | None):
        for attempt in range(1, self.retry_attempts + 1):
            try:
                return connector.search(query, plan)
            except Exception as exc:  # pragma: no cover - network failures vary
                failure_type = classify_failure(exc)
                self.db.insert_source_failure(
                    SourceFailure(
                        run_id=run_id,
                        source=connector.name,
                        query=query,
                        failure_type=failure_type,
                        message=str(exc),
                        attempt=attempt,
                    )
                )
                if attempt >= self.retry_attempts:
                    print(f"[SourceDiscoveryAgent] {connector.name} failed for query '{query}': {exc}")
                    return []
                if not retryable_failure(failure_type):
                    print(f"[SourceDiscoveryAgent] {connector.name} non-retryable failure for query '{query}': {exc}")
                    return []
                time.sleep(self._backoff_seconds(attempt, failure_type))
        return []

    def _source_delay(self, source_name: str) -> None:
        delay = SOURCE_POLITE_DELAYS.get(source_name, 0.5)
        if delay:
            time.sleep(delay)

    def _backoff_seconds(self, attempt: int, failure_type: str) -> float:
        multiplier = 3.0 if failure_type == "rate_limited" else 1.0
        return self.base_backoff_seconds * multiplier * (2 ** (attempt - 1))

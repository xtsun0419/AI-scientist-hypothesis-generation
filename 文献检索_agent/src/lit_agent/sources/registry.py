from __future__ import annotations

from lit_agent.models import QueryPlan

from .base import SourceConnector
from .connectors import (
    ArxivConnector,
    CoreConnector,
    CrossrefConnector,
    DoajConnector,
    EuropePmcConnector,
    OpenAlexConnector,
    SemanticScholarConnector,
)

CONNECTOR_TYPES = {
    "openalex": OpenAlexConnector,
    "crossref": CrossrefConnector,
    "semantic_scholar": SemanticScholarConnector,
    "arxiv": ArxivConnector,
    "doaj": DoajConnector,
    "core": CoreConnector,
    "europe_pmc": EuropePmcConnector,
}


def build_connectors(plan: QueryPlan) -> dict[str, SourceConnector]:
    connectors: dict[str, SourceConnector] = {}
    for source_name in plan.sources:
        connector_type = CONNECTOR_TYPES.get(source_name)
        if connector_type is None:
            continue
        connectors[source_name] = connector_type(
            timeout_seconds=plan.request_timeout_seconds,
            polite_delay_seconds=plan.polite_delay_seconds,
        )
    return connectors

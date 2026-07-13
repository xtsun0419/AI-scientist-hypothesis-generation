from __future__ import annotations

from pathlib import Path

from lit_agent.config import load_domain_config, query_plan_from_config
from lit_agent.models import QueryPlan


class DomainQueryAgent:
    """Builds auditable query plans from a domain configuration."""

    def __init__(self, config_path: Path | None = None):
        self.config_path = config_path

    def build_plan(
        self,
        *,
        from_year: int | None = None,
        to_year: int | None = None,
        sources: list[str] | None = None,
    ) -> QueryPlan:
        config = load_domain_config(self.config_path)
        return query_plan_from_config(config, from_year=from_year, to_year=to_year, sources=sources)

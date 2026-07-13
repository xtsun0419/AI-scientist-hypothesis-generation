from lit_agent.agents.domain_query import DomainQueryAgent


def test_domain_query_plan_contains_permanent_magnet_terms() -> None:
    plan = DomainQueryAgent().build_plan(from_year=2000, to_year=2026, sources=["openalex"])
    assert plan.domain == "permanent_magnets"
    assert plan.from_year == 2000
    assert plan.to_year == 2026
    assert plan.sources == ["openalex"]
    assert "NdFeB coercivity" in plan.queries
    assert "permanent magnet" in plan.include_terms

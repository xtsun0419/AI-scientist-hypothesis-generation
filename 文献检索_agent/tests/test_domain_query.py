from lit_agent.agents.domain_query import DomainQueryAgent


def test_domain_query_plan_uses_the_general_default() -> None:
    plan = DomainQueryAgent().build_plan(from_year=2000, to_year=2026, sources=["openalex"])
    assert plan.domain == "general_research"
    assert plan.from_year == 2000
    assert plan.to_year == 2026
    assert plan.sources == ["openalex"]
    assert plan.queries == []
    assert plan.include_terms == []

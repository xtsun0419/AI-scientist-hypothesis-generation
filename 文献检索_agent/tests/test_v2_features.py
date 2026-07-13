from pathlib import Path

from lit_agent.agents.discovery import SourceDiscoveryAgent
from lit_agent.agents.llm_relevance import LLMRelevanceReviewAgent, _review_from_response
from lit_agent.agents.orchestrator import OrchestratorAgent
from lit_agent.agents.relevance import RelevanceJudgeAgent
from lit_agent.db import LiteratureDB
from lit_agent.errors import classify_failure
from lit_agent.models import QueryPlan, RawSourceRecord


class FailingConnector:
    name = "semantic_scholar"

    def search(self, query, plan):
        raise RuntimeError("semantic_scholar HTTP 429: Too Many Requests")


def make_db(tmp_path: Path) -> LiteratureDB:
    db = LiteratureDB(tmp_path / "literature.sqlite")
    db.init_schema()
    return db


def test_failure_classification() -> None:
    assert classify_failure(RuntimeError("HTTP 429")) == "rate_limited"
    assert classify_failure(RuntimeError("The read operation timed out")) == "timeout"
    assert classify_failure(RuntimeError("PDF HTTP 502: Bad Gateway")) == "server_error"


def test_source_failure_is_recorded(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    try:
        plan = QueryPlan(
            domain="permanent_magnets",
            queries=["NdFeB coercivity"],
            include_terms=[],
            exclude_terms=[],
            sources=["semantic_scholar"],
            from_year=2020,
            to_year=2026,
            max_results_per_query=1,
        )
        run_id = db.create_search_run("permanent_magnets", 2020, 2026, {"queries": plan.queries})
        counts = SourceDiscoveryAgent(
            db,
            connectors={"semantic_scholar": FailingConnector()},
            retry_attempts=1,
            base_backoff_seconds=0,
        ).run(plan, run_id=run_id)
        failures = db.source_failures()
        assert counts["semantic_scholar"] == 0
        assert len(failures) == 1
        assert failures[0]["failure_type"] == "rate_limited"
    finally:
        db.close()


def test_llm_review_skips_without_api_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    db = make_db(tmp_path)
    try:
        db.insert_source_record(
            None,
            RawSourceRecord(
                source="crossref",
                source_id="10.1000/boundary",
                query="permanent magnet",
                raw_payload={
                    "DOI": "10.1000/boundary",
                    "title": ["Predictive screening of rare-earth-free magnetic compounds"],
                    "issued": {"date-parts": [[2024]]},
                    "URL": "https://doi.org/10.1000/boundary",
                },
            ),
        )
        from lit_agent.agents.normalize import MetadataNormalizeAgent
        from lit_agent.agents.dedup import DeduplicationAgent

        MetadataNormalizeAgent(db).run()
        DeduplicationAgent(db).run()
        plan = QueryPlan(
            domain="permanent_magnets",
            queries=["permanent magnet"],
            include_terms=["permanent magnet", "rare-earth-free"],
            exclude_terms=[],
            sources=["crossref"],
            from_year=2020,
            to_year=2026,
        )
        RelevanceJudgeAgent(db).run(plan)
        result = LLMRelevanceReviewAgent(db).run()
        assert result["reviewed"] == 0
        assert result["skipped_no_api_key"] >= 1
        reviews = db.llm_relevance_reviews()
        assert reviews
        assert reviews[0]["decision"] == "skipped_no_api_key"
    finally:
        db.close()


def test_llm_response_parser() -> None:
    review = _review_from_response(
        1,
        "test-model",
        {
            "decision": "include",
            "confidence": 0.82,
            "reason": "Mentions permanent magnets and coercivity.",
            "matched_domain_terms": ["permanent magnets", "coercivity"],
            "exclude_reason": None,
        },
    )
    assert review.decision == "include"
    assert review.confidence == 0.82
    assert "coercivity" in review.matched_domain_terms


def test_mode_presets_limit_query_plan(tmp_path: Path) -> None:
    orchestrator = OrchestratorAgent(db_path=tmp_path / "db.sqlite")
    smoke = orchestrator.plan_queries(mode="smoke")
    pilot = orchestrator.plan_queries(mode="pilot")
    assert len(smoke.queries) == 2
    assert smoke.max_results_per_query == 5
    assert len(pilot.queries) == 4
    assert pilot.max_results_per_query == 5

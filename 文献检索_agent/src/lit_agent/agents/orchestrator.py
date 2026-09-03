from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

from lit_agent.analysis_bridge import convert_pdfs_with_analysis_agent
from lit_agent.config import default_db_path, default_parsed_dir, default_pdf_dir, default_report_dir
from lit_agent.db import LiteratureDB
from lit_agent.researcher_library_bridge import researcher_library_sync

from .dashboard import HtmlDashboardAgent
from .dedup import DeduplicationAgent
from .discovery import SourceDiscoveryAgent
from .domain_query import DomainQueryAgent
from .download import PdfDownloadAgent
from .llm_relevance import LLMRelevanceReviewAgent
from .normalize import MetadataNormalizeAgent
from .oa_resolver import OAResolverAgent
from .quality import QualityAuditAgent
from .relevance import RelevanceJudgeAgent
from .report import ReportExportAgent
from .rounds import (
    EvidenceSynthesisAgent,
    ManualPdfIntakeAgent,
    NextQueryProposalAgent,
    PdfAnalysisAgent,
    RoundAcquisitionAgent,
    RoundApprovalAgent,
    RoundPlanningAgent,
    ScientificGoalAgent,
)

MODE_PRESETS = {
    "smoke": {"query_limit": 2, "max_results_per_query": 5},
    "pilot": {"query_limit": 4, "max_results_per_query": 5},
    "full": {"query_limit": None, "max_results_per_query": None},
}


class OrchestratorAgent:
    """Single-process coordinator for the literature pipeline."""

    def __init__(
        self,
        *,
        db_path: Path | None = None,
        config_path: Path | None = None,
        pdf_dir: Path | None = None,
        report_dir: Path | None = None,
        parsed_dir: Path | None = None,
    ):
        self.db_path = db_path or default_db_path()
        self.config_path = config_path
        self.pdf_dir = pdf_dir or default_pdf_dir()
        self.report_dir = report_dir or default_report_dir()
        self.parsed_dir = parsed_dir or default_parsed_dir()

    def init(self) -> dict[str, str]:
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_dir.mkdir(parents=True, exist_ok=True)
        db = LiteratureDB(self.db_path)
        try:
            db.init_schema()
        finally:
            db.close()
        return {
            "db": str(self.db_path),
            "pdf_dir": str(self.pdf_dir),
            "report_dir": str(self.report_dir),
            "parsed_dir": str(self.parsed_dir),
        }

    def search(
        self,
        *,
        from_year: int | None = None,
        to_year: int | None = None,
        sources: list[str] | None = None,
        query_limit: int | None = None,
        max_results_per_query: int | None = None,
        mode: str = "full",
        llm_review: bool = True,
    ) -> dict[str, int]:
        db = LiteratureDB(self.db_path)
        try:
            db.init_schema()
            query_limit, max_results_per_query = _resolve_mode_limits(
                mode,
                query_limit=query_limit,
                max_results_per_query=max_results_per_query,
            )
            plan = self.plan_queries(
                from_year=from_year,
                to_year=to_year,
                sources=sources,
                query_limit=query_limit,
                max_results_per_query=max_results_per_query,
            )
            run_id = db.create_search_run(plan.domain, plan.from_year, plan.to_year, asdict(plan))
            try:
                discovery_counts = SourceDiscoveryAgent(db).run(plan, run_id=run_id)
                candidates = MetadataNormalizeAgent(db).run()
                papers = DeduplicationAgent(db).run()
                relevance = RelevanceJudgeAgent(db).run(plan)
                access = OAResolverAgent(db).run()
                llm_stats = LLMRelevanceReviewAgent(db).run(plan) if llm_review else {"reviewed": 0}
                self.record_metrics(db, run_id)
                db.finish_search_run(run_id, "finished")
                researcher_library_sync(self.db_path)
            except Exception:
                db.finish_search_run(run_id, "failed")
                raise
            return {
                "run_id": run_id,
                "raw_records": sum(discovery_counts.values()),
                "candidates": candidates,
                "papers": papers,
                "relevance_scored": relevance,
                "access_resolved": access,
                "llm_reviewed": int(llm_stats.get("reviewed", 0)),
            }
        finally:
            db.close()

    def plan_queries(
        self,
        *,
        from_year: int | None = None,
        to_year: int | None = None,
        sources: list[str] | None = None,
        query_limit: int | None = None,
        max_results_per_query: int | None = None,
        mode: str = "full",
    ):
        query_limit, max_results_per_query = _resolve_mode_limits(
            mode,
            query_limit=query_limit,
            max_results_per_query=max_results_per_query,
        )
        plan = DomainQueryAgent(self.config_path).build_plan(from_year=from_year, to_year=to_year, sources=sources)
        queries = plan.queries
        if query_limit is not None:
            queries = plan.queries[:query_limit]
        per_query = plan.max_results_per_query
        if max_results_per_query is not None:
            per_query = max_results_per_query
        return replace(plan, queries=queries, max_results_per_query=per_query)

    def discover(
        self,
        *,
        from_year: int | None = None,
        to_year: int | None = None,
        sources: list[str] | None = None,
        query_limit: int | None = None,
        max_results_per_query: int | None = None,
        mode: str = "full",
    ) -> dict[str, int]:
        db = LiteratureDB(self.db_path)
        try:
            db.init_schema()
            plan = self.plan_queries(
                from_year=from_year,
                to_year=to_year,
                sources=sources,
                query_limit=query_limit,
                max_results_per_query=max_results_per_query,
                mode=mode,
            )
            run_id = db.create_search_run(plan.domain, plan.from_year, plan.to_year, asdict(plan))
            counts = SourceDiscoveryAgent(db).run(plan, run_id=run_id)
            db.finish_search_run(run_id, "finished")
            return {"run_id": run_id, **counts}
        finally:
            db.close()

    def normalize(self) -> int:
        db = LiteratureDB(self.db_path)
        try:
            db.init_schema()
            return MetadataNormalizeAgent(db).run()
        finally:
            db.close()

    def dedup(self) -> int:
        db = LiteratureDB(self.db_path)
        try:
            db.init_schema()
            return DeduplicationAgent(db).run()
        finally:
            db.close()

    def judge_relevance(
        self,
        *,
        from_year: int | None = None,
        to_year: int | None = None,
        sources: list[str] | None = None,
    ) -> int:
        db = LiteratureDB(self.db_path)
        try:
            db.init_schema()
            plan = self.plan_queries(from_year=from_year, to_year=to_year, sources=sources)
            return RelevanceJudgeAgent(db).run(plan)
        finally:
            db.close()

    def resolve_oa(self) -> int:
        db = LiteratureDB(self.db_path)
        try:
            db.init_schema()
            return OAResolverAgent(db).run()
        finally:
            db.close()

    def review_relevance(self) -> dict[str, int]:
        db = LiteratureDB(self.db_path)
        try:
            db.init_schema()
            return LLMRelevanceReviewAgent(db).run(self.plan_queries())
        finally:
            db.close()

    def create_goal(
        self,
        *,
        title: str,
        description: str | None = None,
        domain: str = "general_research",
        target_count: int = 20,
    ) -> int:
        db = LiteratureDB(self.db_path)
        try:
            db.init_schema()
            return ScientificGoalAgent(db).create(
                title=title,
                description=description,
                domain=domain,
                target_count=target_count,
            )
        finally:
            db.close()

    def list_goals(self) -> list[dict[str, object]]:
        db = LiteratureDB(self.db_path)
        try:
            db.init_schema()
            return [dict(row) for row in db.scientific_goals()]
        finally:
            db.close()

    def delete_goal(self, goal_id: int) -> None:
        db = LiteratureDB(self.db_path)
        try:
            db.init_schema()
            db.delete_scientific_goal(goal_id)
        finally:
            db.close()

    def plan_round(
        self,
        *,
        goal_id: int,
        target_count: int | None = None,
        max_results_per_query: int = 8,
        query_limit: int = 4,
        sources: list[str] | None = None,
    ) -> dict[str, int]:
        db = LiteratureDB(self.db_path)
        try:
            db.init_schema()
            result = RoundPlanningAgent(db).plan(
                goal_id=goal_id,
                target_count=target_count,
                max_results_per_query=max_results_per_query,
                query_limit=query_limit,
                sources=sources,
            )
            researcher_library_sync(self.db_path)
            return result
        finally:
            db.close()

    def approve_round(self, round_id: int) -> None:
        db = LiteratureDB(self.db_path)
        try:
            db.init_schema()
            RoundApprovalAgent(db).approve(round_id)
        finally:
            db.close()

    def acquire_round(self, round_id: int) -> dict[str, int]:
        db = LiteratureDB(self.db_path)
        try:
            db.init_schema()
            return RoundAcquisitionAgent(db, self.pdf_dir).acquire(round_id)
        finally:
            db.close()

    def intake_manual_round(self, round_id: int) -> int:
        db = LiteratureDB(self.db_path)
        try:
            db.init_schema()
            return ManualPdfIntakeAgent(db).run(round_id)
        finally:
            db.close()

    def analyze_round(self, round_id: int) -> int:
        db = LiteratureDB(self.db_path)
        try:
            db.init_schema()
            count = PdfAnalysisAgent(db).analyze(round_id)
            EvidenceSynthesisAgent(db).synthesize(round_id)
            return count
        finally:
            db.close()

    def propose_next_round(self, round_id: int) -> dict[str, object]:
        db = LiteratureDB(self.db_path)
        try:
            db.init_schema()
            return NextQueryProposalAgent(db).propose(round_id)
        finally:
            db.close()

    def round_report(self, round_id: int) -> dict[str, object]:
        db = LiteratureDB(self.db_path)
        try:
            db.init_schema()
            round_row = dict(db.exploration_round(round_id))
            goal = dict(db.scientific_goal(round_row["goal_id"]))
            candidates = [dict(row) for row in db.round_candidates(round_id)]
            manual_tasks = [dict(row) for row in db.manual_download_tasks(round_id)]
            analyses = [dict(row) for row in db.paper_analyses(round_id)]
            synthesis = db.round_synthesis(round_id)
            return {
                "goal": goal,
                "round": round_row,
                "candidates": candidates,
                "manual_tasks": manual_tasks,
                "analyses": analyses,
                "synthesis": dict(synthesis) if synthesis else None,
            }
        finally:
            db.close()

    def record_metrics(self, db: LiteratureDB, run_id: int | None = None) -> None:
        metrics = _compute_metrics(db)
        for name, value in metrics.items():
            if isinstance(value, dict):
                numeric = float(value.get("total", 0))
                db.upsert_pipeline_metric(run_id=run_id, metric_name=name, metric_value=numeric, metric_json=value)
            else:
                db.upsert_pipeline_metric(run_id=run_id, metric_name=name, metric_value=float(value))

    def download(self, *, limit: int | None = None) -> int:
        db = LiteratureDB(self.db_path)
        try:
            db.init_schema()
            return PdfDownloadAgent(db, self.pdf_dir).run(limit=limit)
        finally:
            db.close()

    def convert_pdfs(
        self,
        *,
        round_id: int | None = None,
        limit: int | None = None,
        force: bool = False,
    ) -> dict[str, int]:
        _ = self.parsed_dir
        return convert_pdfs_with_analysis_agent(round_id=round_id, limit=limit, force=force)

    def audit(self) -> int:
        db = LiteratureDB(self.db_path)
        try:
            db.init_schema()
            return QualityAuditAgent(db).run()
        finally:
            db.close()

    def report(self) -> dict[str, Path]:
        db = LiteratureDB(self.db_path)
        try:
            db.init_schema()
            self.record_metrics(db, run_id=None)
            outputs = ReportExportAgent(db, self.report_dir).report()
            outputs["dashboard"] = HtmlDashboardAgent(db, self.report_dir).build()
            return outputs
        finally:
            db.close()

    def dashboard(self) -> Path:
        db = LiteratureDB(self.db_path)
        try:
            db.init_schema()
            self.record_metrics(db, run_id=None)
            return HtmlDashboardAgent(db, self.report_dir).build()
        finally:
            db.close()

    def export(self, fmt: str) -> Path:
        db = LiteratureDB(self.db_path)
        try:
            db.init_schema()
            return ReportExportAgent(db, self.report_dir).export(fmt)
        finally:
            db.close()


def _resolve_mode_limits(
    mode: str,
    *,
    query_limit: int | None,
    max_results_per_query: int | None,
) -> tuple[int | None, int | None]:
    if mode not in MODE_PRESETS:
        raise ValueError(f"Unsupported mode: {mode}")
    preset = MODE_PRESETS[mode]
    return (
        query_limit if query_limit is not None else preset["query_limit"],
        max_results_per_query if max_results_per_query is not None else preset["max_results_per_query"],
    )


def _compute_metrics(db: LiteratureDB) -> dict[str, float | dict[str, float]]:
    counts = db.rows(
        """
        SELECT
            (SELECT COUNT(*) FROM source_records) AS source_records,
            (SELECT COUNT(*) FROM paper_candidates) AS candidates,
            (SELECT COUNT(*) FROM papers) AS papers,
            (SELECT COUNT(*) FROM papers WHERE doi IS NOT NULL) AS doi_count,
            (SELECT COUNT(*) FROM access_records WHERE is_oa = 1) AS oa_count,
            (SELECT COUNT(*) FROM access_records WHERE pdf_url IS NOT NULL) AS pdf_url_count,
            (SELECT COUNT(*) FROM pdf_assets WHERE status IN ('downloaded_oa_pdf', 'preprint_pdf')) AS downloaded_count,
            (SELECT COUNT(*) FROM llm_relevance_reviews) AS llm_reviews,
            (SELECT COUNT(*) FROM source_failures) AS source_failures
        """
    )[0]
    source_records = counts["source_records"] or 0
    papers = counts["papers"] or 0
    return {
        "source_records": source_records,
        "paper_candidates": counts["candidates"] or 0,
        "papers": papers,
        "dedup_compression_ratio": (1 - papers / source_records) if source_records else 0,
        "doi_coverage": (counts["doi_count"] or 0) / papers if papers else 0,
        "oa_coverage": (counts["oa_count"] or 0) / papers if papers else 0,
        "pdf_url_coverage": (counts["pdf_url_count"] or 0) / papers if papers else 0,
        "downloaded_pdf_coverage": (counts["downloaded_count"] or 0) / papers if papers else 0,
        "llm_reviews": counts["llm_reviews"] or 0,
        "source_failures": counts["source_failures"] or 0,
    }

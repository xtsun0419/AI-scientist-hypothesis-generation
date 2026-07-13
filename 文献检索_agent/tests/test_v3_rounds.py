from pathlib import Path

import pytest

from lit_agent.agents.dedup import DeduplicationAgent
from lit_agent.agents.normalize import MetadataNormalizeAgent
from lit_agent.agents.oa_resolver import OAResolverAgent
from lit_agent.agents.rounds import (
    EvidenceSynthesisAgent,
    LiteratureSelectionAgent,
    ManualPdfIntakeAgent,
    PdfAnalysisAgent,
    RoundAcquisitionAgent,
    RoundApprovalAgent,
    RoundPlanningAgent,
    ScientificGoalAgent,
)
from lit_agent.constants import ACCESS_OA_PDF_AVAILABLE
from lit_agent.db import LiteratureDB
from lit_agent.llm import LLMSettings
from lit_agent.models import AccessRecord, RawSourceRecord, RoundCandidate


def make_db(tmp_path: Path) -> LiteratureDB:
    db = LiteratureDB(tmp_path / "literature.sqlite")
    db.init_schema()
    return db


def seed_papers(db: LiteratureDB, count: int = 30) -> None:
    for index in range(count):
        db.insert_source_record(
            None,
            RawSourceRecord(
                source="crossref",
                source_id=f"10.1000/v3.{index}",
                query="NdFeB grain boundary diffusion",
                raw_payload={
                    "DOI": f"10.1000/v3.{index}",
                    "title": [f"NdFeB grain boundary diffusion coercivity study {index}"],
                    "issued": {"date-parts": [[2020 + index % 5]]},
                    "container-title": ["Journal of Permanent Magnets"],
                    "URL": f"https://doi.org/10.1000/v3.{index}",
                    "abstract": "Permanent magnet coercivity and grain boundary diffusion.",
                },
            ),
        )
    MetadataNormalizeAgent(db).run()
    DeduplicationAgent(db).run()
    OAResolverAgent(db).run()


def test_goal_defaults_to_twenty(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    try:
        goal_id = ScientificGoalAgent(db).create(title="NdFeB coercivity mechanisms")
        goal = db.scientific_goal(goal_id)
        assert goal["default_target_count"] == 20
    finally:
        db.close()


def test_round_selection_and_approval_gate(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    try:
        seed_papers(db, 30)
        goal_id = ScientificGoalAgent(db).create(title="NdFeB grain boundary diffusion")
        round_id = db.create_exploration_round(goal_id, 20, "planned")
        goal = db.scientific_goal(goal_id)
        selected = LiteratureSelectionAgent(db).select(round_id=round_id, goal=goal, target_count=20)
        assert selected == 20
        assert len(db.round_candidates(round_id)) == 20
        with pytest.raises(ValueError):
            RoundAcquisitionAgent(db, tmp_path / "pdfs", tmp_path / "manual").acquire(round_id)
        db.update_round_status(round_id, "awaiting_user_approval")
        RoundApprovalAgent(db).approve(round_id)
        assert db.exploration_round(round_id)["status"] == "approved"
    finally:
        db.close()


def test_literature_selection_can_use_external_llm_rerank(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    try:
        seed_papers(db, 5)
        papers = db.rows("SELECT * FROM papers ORDER BY id")
        requested_id = papers[0]["id"]
        goal_id = ScientificGoalAgent(db).create(title="NdFeB grain boundary diffusion")
        round_id = db.create_exploration_round(goal_id, 1, "planned")

        class FakeClient:
            def __init__(self, settings: LLMSettings):
                self.settings = settings

            def recommend_literature(self, prompt: str) -> dict:
                assert str(requested_id) in prompt
                return {
                    "selected": [
                        {
                            "paper_id": requested_id,
                            "score": 0.93,
                            "reason": "Directly targets NdFeB grain-boundary diffusion and coercivity.",
                            "material_tags": ["NdFeB", "Grain boundary", "Coercivity"],
                        }
                    ]
                }

        selected = LiteratureSelectionAgent(
            db,
            settings=LLMSettings(base_url="https://example.test/v1", api_key="test", model="fake"),
            client_factory=FakeClient,
            selection_mode="llm",
        ).select(round_id=round_id, goal=db.scientific_goal(goal_id), target_count=1)

        rows = db.round_candidates(round_id)
        assert selected == 1
        assert rows[0]["paper_id"] == requested_id
        assert rows[0]["selection_reason"].startswith("LLM推荐:")
    finally:
        db.close()


def test_literature_selection_ignores_unknown_llm_ids_and_fills_with_rules(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    try:
        seed_papers(db, 5)
        goal_id = ScientificGoalAgent(db).create(title="NdFeB grain boundary diffusion")
        round_id = db.create_exploration_round(goal_id, 2, "planned")

        class FakeClient:
            def __init__(self, settings: LLMSettings):
                self.settings = settings

            def recommend_literature(self, prompt: str) -> dict:
                return {"selected": [{"paper_id": 999999, "score": 1.0, "reason": "Not in pool", "material_tags": []}]}

        selected = LiteratureSelectionAgent(
            db,
            settings=LLMSettings(base_url="https://example.test/v1", api_key="test", model="fake"),
            client_factory=FakeClient,
            selection_mode="llm",
        ).select(round_id=round_id, goal=db.scientific_goal(goal_id), target_count=2)

        rows = db.round_candidates(round_id)
        assert selected == 2
        assert {row["paper_id"] for row in rows} != {999999}
        assert all(row["selection_reason"].startswith("规则补足:") for row in rows)
    finally:
        db.close()


def test_manual_tasks_intake_and_analysis(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    try:
        seed_papers(db, 3)
        goal_id = ScientificGoalAgent(db).create(title="NdFeB grain boundary diffusion")
        round_id = db.create_exploration_round(goal_id, 3, "awaiting_user_approval")
        LiteratureSelectionAgent(db).select(round_id=round_id, goal=db.scientific_goal(goal_id), target_count=3)
        RoundApprovalAgent(db).approve(round_id)
        result = RoundAcquisitionAgent(db, tmp_path / "pdfs", tmp_path / "manual").acquire(round_id)
        assert result["manual_tasks"] == 3
        task = db.manual_download_tasks(round_id)[0]
        path = Path(task["target_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF-1.4 manual\n")
        assert ManualPdfIntakeAgent(db).run(round_id) == 1
        assert PdfAnalysisAgent(db).analyze(round_id) == 3
        synthesis = EvidenceSynthesisAgent(db).synthesize(round_id)
        assert synthesis.next_queries
        assert db.round_synthesis(round_id) is not None
    finally:
        db.close()


def test_manual_intake_matches_arbitrary_download_name_and_canonicalizes(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    try:
        seed_papers(db, 1)
        goal_id = ScientificGoalAgent(db).create(title="NdFeB grain boundary diffusion")
        round_id = db.create_exploration_round(goal_id, 1, "awaiting_user_approval")
        LiteratureSelectionAgent(db).select(round_id=round_id, goal=db.scientific_goal(goal_id), target_count=1)
        RoundApprovalAgent(db).approve(round_id)
        RoundAcquisitionAgent(db, tmp_path / "pdfs", tmp_path / "manual").acquire(round_id)
        task = db.manual_download_tasks(round_id)[0]
        canonical_path = Path(task["target_path"])
        arbitrary_path = canonical_path.parent / "publisher_default_download_name.pdf"
        arbitrary_path.write_bytes(b"%PDF-1.4 arbitrary\n")

        assert ManualPdfIntakeAgent(db).run(round_id) == 1

        assert canonical_path.exists()
        assets = db.rows("SELECT file_path, status FROM pdf_assets WHERE paper_id = ?", (task["paper_id"],))
        assert assets[0]["file_path"] == str(canonical_path)
        assert db.manual_download_tasks(round_id)[0]["status"] == "completed"
    finally:
        db.close()


def test_round_acquisition_downloads_only_current_round(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = make_db(tmp_path)
    try:
        seed_papers(db, 2)
        papers = db.papers()
        for row in papers:
            db.upsert_access_record(
                AccessRecord(
                    paper_id=row["id"],
                    doi=row["doi"],
                    doi_url=f"https://doi.org/{row['doi']}",
                    is_oa=True,
                    pdf_url=f"https://example.org/{row['id']}.pdf",
                    publisher_url=row["publisher_url"],
                    source_url=row["source_url"],
                    access_status=ACCESS_OA_PDF_AVAILABLE,
                )
            )

        goal_id = ScientificGoalAgent(db).create(title="NdFeB grain boundary diffusion")
        round_one = db.create_exploration_round(goal_id, 1, "awaiting_user_approval")
        round_two = db.create_exploration_round(goal_id, 1, "awaiting_user_approval")
        db.upsert_round_candidate(
            RoundCandidate(round_id=round_one, paper_id=papers[0]["id"], rank=1, selection_score=1.0, selection_reason="test", material_tags=["NdFeB"], evidence_level="medium")
        )
        db.upsert_round_candidate(
            RoundCandidate(round_id=round_two, paper_id=papers[1]["id"], rank=1, selection_score=1.0, selection_reason="test", material_tags=["NdFeB"], evidence_level="medium")
        )

        def fake_fetch_pdf(self, url: str) -> tuple[bytes, str]:
            return f"%PDF-1.4\n{url}\n".encode("utf-8"), "application/pdf"

        monkeypatch.setattr("lit_agent.agents.download.PdfDownloadAgent._fetch_pdf", fake_fetch_pdf)
        RoundApprovalAgent(db).approve(round_one)
        result = RoundAcquisitionAgent(db, tmp_path / "pdfs", tmp_path / "manual").acquire(round_one)

        downloaded_paper_ids = {
            row["paper_id"]
            for row in db.rows("SELECT paper_id FROM pdf_assets WHERE status IN ('downloaded_oa_pdf', 'preprint_pdf')")
        }
        assert result["downloaded"] == 1
        assert downloaded_paper_ids == {papers[0]["id"]}
    finally:
        db.close()


def test_round_acquisition_reports_reused_existing_pdf(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    try:
        seed_papers(db, 2)
        papers = db.papers()
        existing_pdf = tmp_path / "pdfs" / "existing.pdf"
        existing_pdf.parent.mkdir(parents=True, exist_ok=True)
        existing_pdf.write_bytes(b"%PDF-1.4 existing\n")
        db.upsert_pdf_asset(
            paper_id=papers[0]["id"],
            pdf_url="https://example.org/existing.pdf",
            file_path=str(existing_pdf),
            sha256="abc",
            file_size=existing_pdf.stat().st_size,
            status="downloaded_oa_pdf",
            error_message=None,
        )

        goal_id = ScientificGoalAgent(db).create(title="NdFeB grain boundary diffusion")
        round_id = db.create_exploration_round(goal_id, 2, "awaiting_user_approval")
        db.upsert_round_candidate(
            RoundCandidate(round_id=round_id, paper_id=papers[0]["id"], rank=1, selection_score=1.0, selection_reason="test", material_tags=["NdFeB"], evidence_level="high")
        )
        db.upsert_round_candidate(
            RoundCandidate(round_id=round_id, paper_id=papers[1]["id"], rank=2, selection_score=0.9, selection_reason="test", material_tags=["NdFeB"], evidence_level="low")
        )

        RoundApprovalAgent(db).approve(round_id)
        result = RoundAcquisitionAgent(db, tmp_path / "pdfs", tmp_path / "manual").acquire(round_id)

        assert result["downloaded"] == 0
        assert result["round_downloaded"] == 1
        assert result["copied_to_round_folder"] == 1
        assert result["manual_tasks"] == 1
        assert Path(result["round_pdf_dir"]).exists()
        rows = db.round_candidates(round_id)
        assert Path(rows[0]["local_pdf_path"]).parent == Path(result["round_pdf_dir"])
        assert Path(rows[0]["local_pdf_path"]).exists()
    finally:
        db.close()


def test_delete_goal_removes_v3_round_state_not_papers(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    try:
        seed_papers(db, 1)
        goal_id = ScientificGoalAgent(db).create(title="NdFeB grain boundary diffusion")
        round_id = db.create_exploration_round(goal_id, 1, "awaiting_user_approval")
        LiteratureSelectionAgent(db).select(round_id=round_id, goal=db.scientific_goal(goal_id), target_count=1)
        assert len(db.round_candidates(round_id)) == 1

        db.delete_scientific_goal(goal_id)

        assert db.rows("SELECT COUNT(*) AS n FROM scientific_goals")[0]["n"] == 0
        assert db.rows("SELECT COUNT(*) AS n FROM exploration_rounds")[0]["n"] == 0
        assert db.rows("SELECT COUNT(*) AS n FROM round_candidates")[0]["n"] == 0
        assert db.rows("SELECT COUNT(*) AS n FROM papers")[0]["n"] == 1
    finally:
        db.close()


def test_round_plan_without_candidates_needs_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = make_db(tmp_path)
    try:
        goal_id = ScientificGoalAgent(db).create(title="A very narrow permanent magnet question")

        monkeypatch.setattr("lit_agent.agents.rounds.SourceDiscoveryAgent.run", lambda self, plan, run_id=None: {})
        monkeypatch.setattr("lit_agent.agents.rounds.MetadataNormalizeAgent.run", lambda self: 0)
        monkeypatch.setattr("lit_agent.agents.rounds.DeduplicationAgent.run", lambda self: 0)
        monkeypatch.setattr("lit_agent.agents.rounds.RelevanceJudgeAgent.run", lambda self, plan: 0)
        monkeypatch.setattr("lit_agent.agents.rounds.OAResolverAgent.run", lambda self: 0)

        result = RoundPlanningAgent(db).plan(goal_id=goal_id, target_count=5, query_limit=1, max_results_per_query=1)

        assert result["selected"] == 0
        assert db.exploration_round(result["round_id"])["status"] == "needs_retry"
    finally:
        db.close()

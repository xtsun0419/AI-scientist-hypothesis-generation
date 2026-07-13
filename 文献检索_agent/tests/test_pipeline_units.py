from pathlib import Path

from lit_agent.agents.dashboard import HtmlDashboardAgent
from lit_agent.agents.dedup import DeduplicationAgent
from lit_agent.agents.download import PdfDownloadAgent
from lit_agent.agents.normalize import MetadataNormalizeAgent
from lit_agent.agents.oa_resolver import OAResolverAgent
from lit_agent.agents.quality import QualityAuditAgent
from lit_agent.agents.relevance import RelevanceJudgeAgent
from lit_agent.agents.report import ReportExportAgent
from lit_agent.db import LiteratureDB
from lit_agent.models import QueryPlan, RawSourceRecord


def make_db(tmp_path: Path) -> LiteratureDB:
    db = LiteratureDB(tmp_path / "literature.sqlite")
    db.init_schema()
    return db


def test_dedup_uses_doi_and_preserves_closed_access_doi(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    try:
        db.insert_source_record(
            None,
            RawSourceRecord(
                source="crossref",
                source_id="10.1016/j.jmmm.2020.166970",
                query="NdFeB coercivity",
                raw_payload={
                    "DOI": "10.1016/j.jmmm.2020.166970",
                    "title": ["Coercivity in NdFeB permanent magnets"],
                    "author": [{"given": "A", "family": "Researcher"}],
                    "issued": {"date-parts": [[2020]]},
                    "container-title": ["Journal of Magnetism and Magnetic Materials"],
                    "publisher": "Elsevier",
                    "URL": "https://doi.org/10.1016/j.jmmm.2020.166970",
                    "type": "journal-article",
                },
            ),
        )
        db.insert_source_record(
            None,
            RawSourceRecord(
                source="semantic_scholar",
                source_id="paper-1",
                query="NdFeB coercivity",
                raw_payload={
                    "paperId": "paper-1",
                    "externalIds": {"DOI": "10.1016/J.JMMM.2020.166970"},
                    "title": "Coercivity in NdFeB permanent magnets",
                    "year": 2020,
                    "venue": "Journal of Magnetism and Magnetic Materials",
                    "authors": [{"name": "A Researcher"}],
                    "isOpenAccess": False,
                },
            ),
        )
        assert MetadataNormalizeAgent(db).run() == 2
        assert DeduplicationAgent(db).run() == 1
        assert OAResolverAgent(db).run() == 1
        papers = db.papers()
        access = db.access_records()
        assert len(papers) == 1
        assert papers[0]["doi"] == "10.1016/j.jmmm.2020.166970"
        assert access[0]["access_status"] == "closed_access_has_doi"
        assert access[0]["doi_url"] == "https://doi.org/10.1016/j.jmmm.2020.166970"
    finally:
        db.close()


def test_relevance_judges_permanent_magnet_and_noise(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    try:
        source_id = db.insert_source_record(
            None,
            RawSourceRecord(
                source="crossref",
                source_id="10.1000/pm",
                query="permanent magnet",
                raw_payload={
                    "DOI": "10.1000/pm",
                    "title": ["Grain boundary diffusion in NdFeB permanent magnets"],
                    "issued": {"date-parts": [[2022]]},
                    "URL": "https://doi.org/10.1000/pm",
                },
            ),
        )
        MetadataNormalizeAgent(db).run()
        DeduplicationAgent(db).run()
        plan = QueryPlan(
            domain="permanent_magnets",
            queries=["permanent magnet"],
            include_terms=["permanent magnet", "ndfeb", "grain boundary diffusion"],
            exclude_terms=["geomagnetic"],
            sources=["crossref"],
            from_year=2000,
            to_year=2026,
        )
        RelevanceJudgeAgent(db).run(plan)
        row = db.papers()[0]
        assert row["relevance_score"] >= 0.8
        assert "permanent magnet" in row["relevance_terms_json"]
        _ = source_id
    finally:
        db.close()


def test_reports_include_missing_doi(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    try:
        db.insert_source_record(
            None,
            RawSourceRecord(
                source="arxiv",
                source_id="http://arxiv.org/abs/2401.00001v1",
                query="rare-earth-free permanent magnet",
                raw_payload={
                    "id": "http://arxiv.org/abs/2401.00001v1",
                    "title": "Rare-earth-free permanent magnet candidate",
                    "summary": "A preprint on permanent magnets.",
                    "published": "2024-01-01T00:00:00Z",
                    "authors": ["A Researcher"],
                    "links": [{"title": "pdf", "href": "https://arxiv.org/pdf/2401.00001", "type": "application/pdf"}],
                },
            ),
        )
        MetadataNormalizeAgent(db).run()
        DeduplicationAgent(db).run()
        OAResolverAgent(db).run()
        QualityAuditAgent(db).run()
        outputs = ReportExportAgent(db, tmp_path / "reports").report()
        dashboard = HtmlDashboardAgent(db, tmp_path / "reports").build()
        assert outputs["all_papers"].exists()
        assert outputs["missing_doi"].exists()
        assert outputs["source_failures"].exists()
        assert outputs["llm_reviews"].exists()
        assert outputs["pipeline_metrics"].exists()
        assert dashboard.exists()
        assert "Rare-earth-free permanent magnet candidate" in outputs["all_papers"].read_text()
        assert "文献检索 Agent 工作台" in dashboard.read_text()
    finally:
        db.close()


def test_open_pdf_available_enters_download_queue(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    try:
        db.insert_source_record(
            None,
            RawSourceRecord(
                source="doaj",
                source_id="doaj-1",
                query="NdFeB permanent magnet",
                raw_payload={
                    "id": "doaj-1",
                    "bibjson": {
                        "title": "Open access NdFeB permanent magnet study",
                        "year": "2023",
                        "identifier": [{"type": "doi", "id": "10.1234/open.ndfeb"}],
                        "author": [{"name": "A Researcher"}],
                        "journal": {"title": "Open Magnetics", "publisher": "OA Publisher"},
                        "link": [
                            {
                                "type": "fulltext",
                                "content_type": "application/pdf",
                                "url": "https://example.org/open.pdf",
                            }
                        ],
                    },
                },
            ),
        )
        MetadataNormalizeAgent(db).run()
        DeduplicationAgent(db).run()
        OAResolverAgent(db).run()
        access = db.access_records()
        queue = db.access_records_for_download()
        assert access[0]["access_status"] == "oa_pdf_available"
        assert len(queue) == 1
    finally:
        db.close()


def test_pdf_download_tries_alternate_candidates(tmp_path: Path) -> None:
    class FakeDownloadAgent(PdfDownloadAgent):
        def _fetch_pdf(self, url: str):
            if url.endswith("bad"):
                return b"<html>not a pdf</html>", "text/html"
            return b"%PDF-1.4\ncontent\n", "application/pdf"

    db = make_db(tmp_path)
    try:
        db.insert_source_record(
            None,
            RawSourceRecord(
                source="doaj",
                source_id="doaj-alt",
                query="NdFeB permanent magnet",
                raw_payload={
                    "id": "doaj-alt",
                    "bibjson": {
                        "title": "Open access NdFeB alternate PDF study",
                        "year": "2023",
                        "identifier": [{"type": "doi", "id": "10.1234/open.alt"}],
                        "author": [{"name": "A Researcher"}],
                        "journal": {"title": "Open Magnetics", "publisher": "OA Publisher"},
                        "link": [
                            {
                                "type": "fulltext",
                                "content_type": "application/pdf",
                                "url": "https://example.org/bad",
                            }
                        ],
                    },
                },
            ),
        )
        MetadataNormalizeAgent(db).run()
        DeduplicationAgent(db).run()
        OAResolverAgent(db).run()
        paper_id = db.papers()[0]["id"]
        db.upsert_pdf_candidate(
            paper_id=paper_id,
            pdf_url="https://example.org/good.pdf",
            source="test",
            priority=2,
            reason="alternate",
        )
        assert FakeDownloadAgent(db, tmp_path / "pdfs").run() == 1
        assets = db.rows("SELECT * FROM pdf_assets ORDER BY id")
        assert any(row["status"] == "download_failed" for row in assets)
        assert any(row["status"] == "downloaded_oa_pdf" for row in assets)
    finally:
        db.close()

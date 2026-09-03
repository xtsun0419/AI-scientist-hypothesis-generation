from pathlib import Path

import json

from lit_analysis_agent.db import AnalysisDB
from lit_analysis_agent.pdf_convert import (
    PdfConversionAgent,
    PdfRecord,
    extract_citations,
    split_reference_entries,
    structure_blocks,
)


def make_db(tmp_path: Path) -> AnalysisDB:
    db = AnalysisDB(tmp_path / "literature.sqlite")
    db.init_schema()
    db.conn.executescript(
        """
        CREATE TABLE papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            doi TEXT,
            year INTEGER,
            venue TEXT,
            authors_json TEXT NOT NULL DEFAULT '[]',
            abstract TEXT
        );
        CREATE TABLE pdf_assets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id INTEGER NOT NULL,
            file_path TEXT,
            status TEXT NOT NULL,
            downloaded_at TEXT
        );
        CREATE TABLE round_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            round_id INTEGER NOT NULL,
            paper_id INTEGER NOT NULL,
            rank INTEGER NOT NULL
        );
        CREATE TABLE exploration_rounds (
            id INTEGER PRIMARY KEY AUTOINCREMENT
        );
        """
    )
    db.conn.commit()
    return db


def seed_paper_pdf(db: AnalysisDB, pdf_path: Path, *, title: str = "TopicAlpha conversion test") -> int:
    cur = db.conn.execute(
        """
        INSERT INTO papers(title, doi, year, venue, authors_json, abstract)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (title, "10.1000/test", 2024, "Test Journal", json.dumps(["A Researcher"]), "A test abstract."),
    )
    paper_id = int(cur.lastrowid)
    db.conn.execute(
        """
        INSERT INTO pdf_assets(paper_id, file_path, status, downloaded_at)
        VALUES (?, ?, ?, ?)
        """,
        (paper_id, str(pdf_path), "downloaded_oa_pdf", "2026-01-01T00:00:00Z"),
    )
    db.conn.commit()
    return paper_id


def test_structure_blocks_extracts_sections_captions_references_and_citations() -> None:
    blocks = [
        {"text": "A Study of TopicAlpha Grain Boundary Diffusion", "page": 1, "font_size": 18},
        {"text": "Abstract", "page": 1, "font_size": 12},
        {"text": "This work studies outcome metric in TopicAlpha magnets [1].", "page": 1, "font_size": 10},
        {"text": "1. Introduction", "page": 2, "font_size": 14},
        {"text": "Grain-boundary diffusion improves outcome metric (Smith et al., 2020).", "page": 2, "font_size": 10},
        {"text": "Figure 1. Microstructure after diffusion.", "page": 3, "font_size": 9},
        {"text": "Table 1. Magnetic properties summary.", "page": 4, "font_size": 9},
        {"text": "References", "page": 5, "font_size": 14},
        {"text": "[1] Smith A. TopicAlpha diffusion. Journal 2020. doi:10.1000/test.doi", "page": 5, "font_size": 10},
    ]

    result = structure_blocks(blocks, metadata_title=None)

    assert result["title"] == "A Study of TopicAlpha Grain Boundary Diffusion"
    assert result["abstract"] == "This work studies outcome metric in TopicAlpha magnets [1]."
    assert [section["heading"] for section in result["sections"]] == ["Abstract", "1. Introduction", "References"]
    assert result["figures"][0]["label"].lower().startswith("figure")
    assert result["tables"][0]["extraction_quality"] == "text_table"
    assert result["paragraphs"][0]["citations"] == ["[1]"]
    assert result["paragraphs"][1]["citations"] == ["(Smith et al., 2020)"]
    assert result["references"][0]["doi"] == "10.1000/test.doi"
    assert result["references"][0]["year"] == 2020


def test_reference_and_citation_helpers() -> None:
    references = split_reference_entries(
        [
            "[1] First paper. https://example.org/a 2019.",
            "[2] Second paper. doi:10.5555/example.2020",
        ]
    )

    assert len(references) == 2
    assert references[0]["url"] == "https://example.org/a"
    assert references[1]["doi"] == "10.5555/example.2020"
    assert extract_citations("Prior work [1, 2] and (Brown et al., 2021) showed this.") == [
        "[1, 2]",
        "(Brown et al., 2021)",
    ]


def test_pdf_conversion_agent_writes_outputs_skips_and_forces(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    try:
        pdf_path = tmp_path / "pdfs" / "paper.pdf"
        pdf_path.parent.mkdir(parents=True)
        pdf_path.write_bytes(b"%PDF-1.4\nfake\n")
        seed_paper_pdf(db, pdf_path)

        calls = {"count": 0}

        def fake_parser(path: Path, record: PdfRecord) -> dict:
            calls["count"] += 1
            return {
                "metadata": {
                    "title": record.title,
                    "authors": record.authors,
                    "year": record.year,
                    "doi": record.doi,
                    "venue": record.venue,
                    "pdf_path": str(path),
                    "source": record.source,
                    "page_count": 1,
                    "parser_version": "fake",
                    "converted_at": "now",
                },
                "sections": [{"heading": "Abstract", "level": 1, "page_start": 1, "page_end": 1, "paragraphs": []}],
                "paragraphs": [],
                "figures": [],
                "tables": [],
                "references": [{"raw_text": "Reference", "doi": None, "url": None, "year": None}],
                "quality": {
                    "status": "converted",
                    "parser": "fake",
                    "warnings": [],
                    "text_chars": 10,
                    "section_count": 1,
                    "figure_count": 0,
                    "table_count": 0,
                    "reference_count": 1,
                },
            }

        agent = PdfConversionAgent(db, tmp_path / "parsed", parser=fake_parser, scan_roots=[])

        first = agent.run()
        second = agent.run()
        forced = agent.run(force=True)

        conversion = db.paper_conversions(limit=1)[0]
        assert first == {"total": 1, "converted": 1, "skipped": 0, "failed": 0}
        assert second == {"total": 1, "converted": 0, "skipped": 1, "failed": 0}
        assert forced == {"total": 1, "converted": 1, "skipped": 0, "failed": 0}
        assert calls["count"] == 2
        assert Path(conversion["markdown_path"]).exists()
        assert Path(conversion["json_path"]).exists()
        assert conversion["status"] == "converted"
        assert conversion["section_count"] == 1
        assert conversion["reference_count"] == 1
    finally:
        db.close()


def test_pdf_conversion_agent_records_missing_pdf_failure(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    try:
        missing = tmp_path / "missing.pdf"
        seed_paper_pdf(db, missing, title="Missing PDF")

        result = PdfConversionAgent(db, tmp_path / "parsed", parser=lambda path, record: {}, scan_roots=[]).run()

        conversion = db.paper_conversions(limit=1)[0]
        assert result == {"total": 1, "converted": 0, "skipped": 0, "failed": 1}
        assert conversion["status"] == "failed"
        assert "missing or empty" in conversion["error_message"]
    finally:
        db.close()

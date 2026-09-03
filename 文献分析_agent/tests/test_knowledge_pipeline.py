import json
from pathlib import Path

from lit_analysis_agent.db import AnalysisDB
from lit_analysis_agent.knowledge import (
    CorpusIndexAgent,
    HybridSearchAgent,
    KnowledgeGraphAgent,
    PaperCardAgent,
    WikiAgent,
    load_corpus_documents,
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
        """
    )
    db.conn.commit()
    return db


def seed_parsed_paper(db: AnalysisDB, tmp_path: Path) -> Path:
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    cur = db.conn.execute(
        """
        INSERT INTO papers(title, doi, year, venue, authors_json, abstract)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "Outcome Metric optimization in TopicAlpha magnets",
            "10.1000/knowledge",
            2025,
            "Magnetics Test",
            json.dumps(["A Researcher"]),
            "This study analyzes grain boundary diffusion for outcome metric optimization.",
        ),
    )
    paper_id = int(cur.lastrowid)
    json_path = parsed_dir / f"paper_{paper_id}_outcome metric.json"
    json_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "title": "Outcome Metric optimization in TopicAlpha magnets",
                    "doi": "10.1000/knowledge",
                    "year": 2025,
                    "venue": "Magnetics Test",
                    "page_count": 2,
                },
                "sections": [],
                "paragraphs": [
                    {
                        "id": "p00001",
                        "text": "Grain boundary diffusion improves outcome metric in TopicAlpha magnets to 20 kOe while preserving outcome retention.",
                        "page": 1,
                        "section_path": ["Abstract"],
                        "citations": ["[1]"],
                    },
                    {
                        "id": "p00002",
                        "text": "However, the process has a trade-off between outcome metric and energy product in thick samples.",
                        "page": 2,
                        "section_path": ["Discussion"],
                        "citations": [],
                    },
                ],
                "figures": [{"label": "Figure 1", "caption": "Figure 1. Diffusion microstructure.", "page": 2}],
                "tables": [{"label": "Table 1", "caption": "Table 1. Magnetic properties.", "text": "Hcj 20 kOe", "rows": ["Hcj 20 kOe"]}],
                "references": [{"raw_text": "[1] Prior outcome metric study. doi:10.1000/ref 2024.", "doi": "10.1000/ref", "url": None, "year": 2024}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    db.upsert_paper_conversion(
        paper_id=paper_id,
        pdf_asset_id=None,
        pdf_path=str(tmp_path / "paper.pdf"),
        markdown_path=str(parsed_dir / "paper.md"),
        json_path=str(json_path),
        status="converted",
        parser="test",
        error_message=None,
        page_count=2,
        text_chars=200,
        section_count=2,
        figure_count=1,
        table_count=1,
        reference_count=1,
    )
    return parsed_dir


def test_corpus_index_searches_paragraphs_tables_and_references(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    try:
        parsed_dir = seed_parsed_paper(db, tmp_path)

        result = CorpusIndexAgent(db, parsed_dir, tmp_path / "index").run()
        rows = db.search_chunks_bm25("outcome metric", limit=5)

        assert result["documents"] == 1
        assert result["chunks"] == 5
        assert rows
        assert any(row["chunk_type"] == "paragraph" for row in rows)
        assert HybridSearchAgent(db).search("outcome metric", limit=3)
    finally:
        db.close()


def test_cards_graph_and_wiki_keep_evidence_ids(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    try:
        parsed_dir = seed_parsed_paper(db, tmp_path)
        CorpusIndexAgent(db, parsed_dir, tmp_path / "index").run()

        card_result = PaperCardAgent(db, parsed_dir, tmp_path / "cards").run()
        card = json.loads(db.paper_cards()[0]["card_json"])
        graph_result = KnowledgeGraphAgent(db, tmp_path / "graph").run()
        wiki_result = WikiAgent(db, tmp_path / "wiki", tmp_path / "graph").run()
        wiki_pages = [json.loads(row["page_json"]) for row in db.wiki_pages()]

        assert card_result == {"cards": 1}
        assert "TopicAlpha" in card["materials"]
        assert "quantitative result" in card["properties"]
        assert card["claims"][0]["evidence_ids"][0].startswith(f"p:{card['paper_id']}:")
        assert graph_result["nodes"] > 0
        assert (tmp_path / "graph" / "graph.json").exists()
        assert wiki_result["wiki_pages"] > 0
        assert all(page["evidence_ids"] or page["needs_evidence"] for page in wiki_pages)
    finally:
        db.close()


def test_load_corpus_documents_deduplicates_same_paper(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    try:
        parsed_dir = seed_parsed_paper(db, tmp_path)
        original = db.paper_conversions(limit=1)[0]
        db.upsert_paper_conversion(
            paper_id=original["paper_id"],
            pdf_asset_id=None,
            pdf_path=str(tmp_path / "copy.pdf"),
            markdown_path=original["markdown_path"],
            json_path=original["json_path"],
            status="converted",
            parser="test",
            error_message=None,
        )

        docs = load_corpus_documents(db, parsed_dir)

        assert len(docs) == 1
    finally:
        db.close()

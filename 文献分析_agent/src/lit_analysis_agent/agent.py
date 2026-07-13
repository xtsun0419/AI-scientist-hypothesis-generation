from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import default_cards_dir, default_db_path, default_graph_dir, default_index_dir, default_parsed_dir, default_wiki_dir
from .db import AnalysisDB
from .knowledge import CorpusIndexAgent, HybridSearchAgent, KnowledgeGraphAgent, PaperCardAgent, WikiAgent
from .pdf_convert import PdfConversionAgent


class LiteratureAnalysisAgent:
    """Coordinator for the literature-analysis agent project."""

    def __init__(self, *, db_path: Path | None = None, parsed_dir: Path | None = None):
        self.db_path = db_path or default_db_path()
        self.parsed_dir = parsed_dir or default_parsed_dir()

    def init(self) -> dict[str, str]:
        self.parsed_dir.mkdir(parents=True, exist_ok=True)
        for path in (default_index_dir(), default_cards_dir(), default_graph_dir(), default_wiki_dir()):
            path.mkdir(parents=True, exist_ok=True)
        db = AnalysisDB(self.db_path)
        try:
            db.init_schema()
        finally:
            db.close()
        return {
            "db": str(self.db_path),
            "parsed_dir": str(self.parsed_dir),
            "index_dir": str(default_index_dir()),
            "cards_dir": str(default_cards_dir()),
            "graph_dir": str(default_graph_dir()),
            "wiki_dir": str(default_wiki_dir()),
        }

    def convert_pdfs(
        self,
        *,
        round_id: int | None = None,
        limit: int | None = None,
        force: bool = False,
    ) -> dict[str, int]:
        db = AnalysisDB(self.db_path)
        try:
            db.init_schema()
            return PdfConversionAgent(db, self.parsed_dir).run(round_id=round_id, limit=limit, force=force)
        finally:
            db.close()

    def index_corpus(self, *, force: bool = True) -> dict[str, int | str]:
        db = AnalysisDB(self.db_path)
        try:
            db.init_schema()
            return CorpusIndexAgent(db, self.parsed_dir).run(force=force)
        finally:
            db.close()

    def search(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        db = AnalysisDB(self.db_path)
        try:
            db.init_schema()
            return HybridSearchAgent(db).search(query, limit=limit)
        finally:
            db.close()

    def build_paper_cards(self) -> dict[str, int]:
        db = AnalysisDB(self.db_path)
        try:
            db.init_schema()
            return PaperCardAgent(db, self.parsed_dir).run()
        finally:
            db.close()

    def build_graph(self) -> dict[str, int | str]:
        db = AnalysisDB(self.db_path)
        try:
            db.init_schema()
            return KnowledgeGraphAgent(db).run()
        finally:
            db.close()

    def build_wiki(self) -> dict[str, int]:
        db = AnalysisDB(self.db_path)
        try:
            db.init_schema()
            return WikiAgent(db).run()
        finally:
            db.close()

    def build_all(self) -> dict[str, int | str]:
        result: dict[str, int | str] = {}
        result.update({f"index_{key}": value for key, value in self.index_corpus(force=True).items()})
        result.update(self.build_paper_cards())
        result.update(self.build_graph())
        result.update(self.build_wiki())
        return result

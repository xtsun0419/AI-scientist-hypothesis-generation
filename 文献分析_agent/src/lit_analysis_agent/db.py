from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AnalysisDB:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def init_schema(self) -> None:
        self.conn.executescript(
            """
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS paper_conversions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER,
                pdf_asset_id INTEGER,
                pdf_path TEXT NOT NULL UNIQUE,
                markdown_path TEXT,
                json_path TEXT,
                status TEXT NOT NULL,
                parser TEXT NOT NULL,
                error_message TEXT,
                page_count INTEGER NOT NULL DEFAULT 0,
                text_chars INTEGER NOT NULL DEFAULT 0,
                section_count INTEGER NOT NULL DEFAULT 0,
                figure_count INTEGER NOT NULL DEFAULT 0,
                table_count INTEGER NOT NULL DEFAULT 0,
                reference_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS rag_chunks (
                id TEXT PRIMARY KEY,
                paper_id INTEGER,
                source_json_path TEXT NOT NULL,
                chunk_type TEXT NOT NULL,
                title TEXT,
                section_path TEXT NOT NULL,
                text TEXT NOT NULL,
                page INTEGER,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS rag_chunks_fts USING fts5(
                id UNINDEXED,
                title,
                section_path,
                text
            );

            CREATE TABLE IF NOT EXISTS rag_embeddings (
                chunk_id TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(chunk_id) REFERENCES rag_chunks(id)
            );

            CREATE TABLE IF NOT EXISTS paper_cards (
                paper_id INTEGER PRIMARY KEY,
                source_json_path TEXT NOT NULL,
                card_json TEXT NOT NULL,
                markdown_path TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS knowledge_nodes (
                id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                label TEXT NOT NULL,
                summary TEXT,
                payload_json TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS knowledge_edges (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1,
                evidence_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS wiki_pages (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                node_id TEXT NOT NULL,
                page_json TEXT NOT NULL,
                markdown_path TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def pdf_assets_for_conversion(self, round_id: int | None = None) -> list[sqlite3.Row]:
        if round_id is None:
            return self._rows(
                """
                SELECT pa.id AS pdf_asset_id, pa.paper_id, pa.file_path AS pdf_path,
                       pa.status AS pdf_status, p.title, p.doi, p.year, p.venue,
                       p.authors_json, p.abstract
                FROM pdf_assets pa
                LEFT JOIN papers p ON p.id = pa.paper_id
                WHERE pa.file_path IS NOT NULL
                  AND pa.status IN ('downloaded_oa_pdf', 'preprint_pdf')
                ORDER BY pa.downloaded_at DESC, pa.id DESC
                """
            )
        return self._rows(
            """
            SELECT pa.id AS pdf_asset_id, pa.paper_id, pa.file_path AS pdf_path,
                   pa.status AS pdf_status, p.title, p.doi, p.year, p.venue,
                   p.authors_json, p.abstract
            FROM round_candidates rc
            JOIN pdf_assets pa ON pa.paper_id = rc.paper_id
            LEFT JOIN papers p ON p.id = pa.paper_id
            WHERE rc.round_id = ?
              AND pa.file_path IS NOT NULL
              AND pa.status IN ('downloaded_oa_pdf', 'preprint_pdf')
            ORDER BY rc.rank, pa.downloaded_at DESC, pa.id DESC
            """,
            (round_id,),
        )

    def paper_conversion_for_path(self, pdf_path: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM paper_conversions WHERE pdf_path = ?", (pdf_path,)).fetchone()

    def upsert_paper_conversion(
        self,
        *,
        paper_id: int | None,
        pdf_asset_id: int | None,
        pdf_path: str,
        markdown_path: str | None,
        json_path: str | None,
        status: str,
        parser: str,
        error_message: str | None,
        page_count: int = 0,
        text_chars: int = 0,
        section_count: int = 0,
        figure_count: int = 0,
        table_count: int = 0,
        reference_count: int = 0,
    ) -> None:
        now = utc_now_iso()
        self.conn.execute(
            """
            INSERT INTO paper_conversions(
                paper_id, pdf_asset_id, pdf_path, markdown_path, json_path, status, parser,
                error_message, page_count, text_chars, section_count, figure_count,
                table_count, reference_count, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(pdf_path) DO UPDATE SET
                paper_id = excluded.paper_id,
                pdf_asset_id = excluded.pdf_asset_id,
                markdown_path = excluded.markdown_path,
                json_path = excluded.json_path,
                status = excluded.status,
                parser = excluded.parser,
                error_message = excluded.error_message,
                page_count = excluded.page_count,
                text_chars = excluded.text_chars,
                section_count = excluded.section_count,
                figure_count = excluded.figure_count,
                table_count = excluded.table_count,
                reference_count = excluded.reference_count,
                updated_at = excluded.updated_at
            """,
            (
                paper_id,
                pdf_asset_id,
                pdf_path,
                markdown_path,
                json_path,
                status,
                parser,
                error_message,
                page_count,
                text_chars,
                section_count,
                figure_count,
                table_count,
                reference_count,
                now,
                now,
            ),
        )
        self.conn.commit()

    def paper_conversion_metrics(self) -> sqlite3.Row:
        return self.conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM pdf_assets WHERE status IN ('downloaded_oa_pdf', 'preprint_pdf') AND file_path IS NOT NULL) AS pdf_total,
                (SELECT COUNT(*) FROM paper_conversions WHERE status = 'converted') AS converted,
                (SELECT COUNT(*) FROM paper_conversions WHERE status = 'failed') AS failed,
                (SELECT COUNT(*) FROM paper_conversions WHERE status = 'skipped') AS skipped
            """
        ).fetchone()

    def paper_conversions(self, limit: int = 50) -> list[sqlite3.Row]:
        return self._rows(
            """
            SELECT pc.*, p.title, p.doi, p.year, p.venue
            FROM paper_conversions pc
            LEFT JOIN papers p ON p.id = pc.paper_id
            ORDER BY pc.updated_at DESC, pc.id DESC
            LIMIT ?
            """,
            (limit,),
        )

    def converted_papers(self) -> list[sqlite3.Row]:
        return self._rows(
            """
            SELECT pc.*, p.title, p.doi, p.year, p.venue
            FROM paper_conversions pc
            LEFT JOIN papers p ON p.id = pc.paper_id
            WHERE pc.status = 'converted'
              AND pc.json_path IS NOT NULL
            ORDER BY pc.updated_at DESC, pc.id DESC
            """
        )

    def clear_rag_index(self) -> None:
        self.conn.execute("DELETE FROM rag_chunks")
        self.conn.execute("DELETE FROM rag_chunks_fts")
        self.conn.execute("DELETE FROM rag_embeddings")
        self.conn.commit()

    def insert_rag_chunk(
        self,
        *,
        chunk_id: str,
        paper_id: int | None,
        source_json_path: str,
        chunk_type: str,
        title: str | None,
        section_path: str,
        text: str,
        page: int | None,
        metadata_json: str,
    ) -> None:
        now = utc_now_iso()
        self.conn.execute("DELETE FROM rag_chunks_fts WHERE id = ?", (chunk_id,))
        self.conn.execute(
            """
            INSERT OR REPLACE INTO rag_chunks(
                id, paper_id, source_json_path, chunk_type, title, section_path,
                text, page, metadata_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (chunk_id, paper_id, source_json_path, chunk_type, title, section_path, text, page, metadata_json, now),
        )
        self.conn.execute(
            "INSERT INTO rag_chunks_fts(id, title, section_path, text) VALUES (?, ?, ?, ?)",
            (chunk_id, title or "", section_path, text),
        )
        self.conn.commit()

    def upsert_embedding(self, *, chunk_id: str, model: str, vector_json: str) -> None:
        now = utc_now_iso()
        self.conn.execute(
            """
            INSERT INTO rag_embeddings(chunk_id, model, vector_json, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                model = excluded.model,
                vector_json = excluded.vector_json,
                created_at = excluded.created_at
            """,
            (chunk_id, model, vector_json, now),
        )
        self.conn.commit()

    def rag_chunks(self, limit: int | None = None) -> list[sqlite3.Row]:
        query = "SELECT * FROM rag_chunks ORDER BY id"
        if limit is not None:
            query += " LIMIT ?"
            return self._rows(query, (limit,))
        return self._rows(query)

    def search_chunks_bm25(self, query: str, limit: int = 12) -> list[sqlite3.Row]:
        return self._rows(
            """
            SELECT rc.*, bm25(rag_chunks_fts) AS bm25_score
            FROM rag_chunks_fts
            JOIN rag_chunks rc ON rc.id = rag_chunks_fts.id
            WHERE rag_chunks_fts MATCH ?
            ORDER BY bm25_score
            LIMIT ?
            """,
            (query, limit),
        )

    def clear_paper_cards(self) -> None:
        self.conn.execute("DELETE FROM paper_cards")
        self.conn.commit()

    def upsert_paper_card(self, *, paper_id: int, source_json_path: str, card_json: str, markdown_path: str | None) -> None:
        now = utc_now_iso()
        self.conn.execute(
            """
            INSERT INTO paper_cards(paper_id, source_json_path, card_json, markdown_path, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET
                source_json_path = excluded.source_json_path,
                card_json = excluded.card_json,
                markdown_path = excluded.markdown_path,
                updated_at = excluded.updated_at
            """,
            (paper_id, source_json_path, card_json, markdown_path, now, now),
        )
        self.conn.commit()

    def paper_cards(self, limit: int | None = None) -> list[sqlite3.Row]:
        query = "SELECT * FROM paper_cards ORDER BY paper_id"
        if limit is not None:
            query += " LIMIT ?"
            return self._rows(query, (limit,))
        return self._rows(query)

    def clear_graph(self) -> None:
        self.conn.execute("DELETE FROM knowledge_edges")
        self.conn.execute("DELETE FROM knowledge_nodes")
        self.conn.commit()

    def upsert_knowledge_node(
        self,
        *,
        node_id: str,
        node_type: str,
        label: str,
        summary: str | None,
        payload_json: str,
        weight: float = 1,
    ) -> None:
        now = utc_now_iso()
        self.conn.execute(
            """
            INSERT INTO knowledge_nodes(id, node_type, label, summary, payload_json, weight, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                node_type = excluded.node_type,
                label = excluded.label,
                summary = excluded.summary,
                payload_json = excluded.payload_json,
                weight = excluded.weight,
                updated_at = excluded.updated_at
            """,
            (node_id, node_type, label, summary, payload_json, weight, now, now),
        )
        self.conn.commit()

    def upsert_knowledge_edge(
        self,
        *,
        edge_id: str,
        source_id: str,
        target_id: str,
        edge_type: str,
        evidence_json: str,
        weight: float = 1,
    ) -> None:
        now = utc_now_iso()
        self.conn.execute(
            """
            INSERT INTO knowledge_edges(id, source_id, target_id, edge_type, weight, evidence_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                source_id = excluded.source_id,
                target_id = excluded.target_id,
                edge_type = excluded.edge_type,
                weight = excluded.weight,
                evidence_json = excluded.evidence_json,
                updated_at = excluded.updated_at
            """,
            (edge_id, source_id, target_id, edge_type, weight, evidence_json, now, now),
        )
        self.conn.commit()

    def knowledge_nodes(self) -> list[sqlite3.Row]:
        return self._rows("SELECT * FROM knowledge_nodes ORDER BY node_type, label")

    def knowledge_edges(self) -> list[sqlite3.Row]:
        return self._rows("SELECT * FROM knowledge_edges ORDER BY edge_type, id")

    def clear_wiki_pages(self) -> None:
        self.conn.execute("DELETE FROM wiki_pages")
        self.conn.commit()

    def upsert_wiki_page(
        self,
        *,
        page_id: str,
        title: str,
        node_id: str,
        page_json: str,
        markdown_path: str | None,
    ) -> None:
        now = utc_now_iso()
        self.conn.execute(
            """
            INSERT INTO wiki_pages(id, title, node_id, page_json, markdown_path, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                node_id = excluded.node_id,
                page_json = excluded.page_json,
                markdown_path = excluded.markdown_path,
                updated_at = excluded.updated_at
            """,
            (page_id, title, node_id, page_json, markdown_path, now, now),
        )
        self.conn.commit()

    def wiki_pages(self) -> list[sqlite3.Row]:
        return self._rows("SELECT * FROM wiki_pages ORDER BY title")

    def exploration_rounds(self, limit: int = 1) -> list[sqlite3.Row]:
        return self._rows("SELECT * FROM exploration_rounds ORDER BY id DESC LIMIT ?", (limit,))

    def _rows(self, query: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        return list(self.conn.execute(query, params))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

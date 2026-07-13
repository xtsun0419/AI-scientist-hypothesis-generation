from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .models import (
    AccessRecord,
    AuditFinding,
    LLMRelevanceReview,
    ManualDownloadTask,
    PaperCandidate,
    PaperAnalysis,
    PaperRecord,
    RawSourceRecord,
    RoundCandidate,
    RoundSynthesis,
    ScientificGoal,
    SourceFailure,
    utc_now_iso,
)
from .text import normalize_title


class LiteratureDB:
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

            CREATE TABLE IF NOT EXISTS search_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT NOT NULL,
                from_year INTEGER NOT NULL,
                to_year INTEGER NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                query_plan_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS source_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                source TEXT NOT NULL,
                source_id TEXT NOT NULL,
                query TEXT NOT NULL,
                raw_payload_json TEXT NOT NULL,
                retrieved_at TEXT NOT NULL,
                UNIQUE(source, source_id, query),
                FOREIGN KEY(run_id) REFERENCES search_runs(id)
            );

            CREATE TABLE IF NOT EXISTS paper_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_record_id INTEGER NOT NULL UNIQUE,
                source TEXT NOT NULL,
                source_id TEXT NOT NULL,
                query TEXT NOT NULL,
                doi TEXT,
                title TEXT NOT NULL,
                normalized_title TEXT NOT NULL,
                authors_json TEXT NOT NULL,
                year INTEGER,
                venue TEXT,
                abstract TEXT,
                publisher TEXT,
                publisher_url TEXT,
                source_url TEXT,
                pdf_url TEXT,
                is_oa INTEGER,
                document_type TEXT,
                raw_payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(source_record_id) REFERENCES source_records(id)
            );

            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doi TEXT UNIQUE,
                title TEXT NOT NULL,
                normalized_title TEXT NOT NULL,
                authors_json TEXT NOT NULL,
                year INTEGER,
                venue TEXT,
                abstract TEXT,
                publisher TEXT,
                publisher_url TEXT,
                source_url TEXT,
                document_type TEXT,
                relevance_score REAL,
                relevance_terms_json TEXT NOT NULL DEFAULT '[]',
                relevance_reason TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_papers_normalized_title
                ON papers(normalized_title, year);

            CREATE TABLE IF NOT EXISTS paper_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER NOT NULL,
                source_record_id INTEGER NOT NULL,
                source TEXT NOT NULL,
                source_id TEXT NOT NULL,
                query TEXT NOT NULL,
                UNIQUE(paper_id, source_record_id),
                FOREIGN KEY(paper_id) REFERENCES papers(id),
                FOREIGN KEY(source_record_id) REFERENCES source_records(id)
            );

            CREATE TABLE IF NOT EXISTS access_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER NOT NULL UNIQUE,
                doi TEXT,
                doi_url TEXT,
                is_oa INTEGER,
                pdf_url TEXT,
                publisher_url TEXT,
                source_url TEXT,
                access_status TEXT NOT NULL,
                resolved_at TEXT NOT NULL,
                FOREIGN KEY(paper_id) REFERENCES papers(id)
            );

            CREATE TABLE IF NOT EXISTS pdf_assets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER NOT NULL,
                pdf_url TEXT NOT NULL,
                file_path TEXT,
                sha256 TEXT,
                file_size INTEGER,
                status TEXT NOT NULL,
                error_message TEXT,
                downloaded_at TEXT,
                UNIQUE(paper_id, pdf_url),
                FOREIGN KEY(paper_id) REFERENCES papers(id)
            );

            CREATE TABLE IF NOT EXISTS pdf_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER NOT NULL,
                pdf_url TEXT NOT NULL,
                source TEXT NOT NULL,
                priority INTEGER NOT NULL,
                reason TEXT NOT NULL,
                last_status TEXT,
                last_error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(paper_id, pdf_url),
                FOREIGN KEY(paper_id) REFERENCES papers(id)
            );

            CREATE TABLE IF NOT EXISTS audit_findings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER,
                severity TEXT NOT NULL,
                issue_type TEXT NOT NULL,
                message TEXT NOT NULL,
                suggestion TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(paper_id) REFERENCES papers(id)
            );

            CREATE TABLE IF NOT EXISTS source_failures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                source TEXT NOT NULL,
                query TEXT NOT NULL,
                failure_type TEXT NOT NULL,
                message TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES search_runs(id)
            );

            CREATE TABLE IF NOT EXISTS llm_relevance_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                decision TEXT NOT NULL,
                confidence REAL,
                reason TEXT NOT NULL,
                matched_domain_terms_json TEXT NOT NULL,
                exclude_reason TEXT,
                raw_response_json TEXT NOT NULL,
                reviewed_at TEXT NOT NULL,
                UNIQUE(paper_id, provider, model),
                FOREIGN KEY(paper_id) REFERENCES papers(id)
            );

            CREATE TABLE IF NOT EXISTS pipeline_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_json TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(run_id, metric_name),
                FOREIGN KEY(run_id) REFERENCES search_runs(id)
            );

            CREATE TABLE IF NOT EXISTS scientific_goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                domain TEXT NOT NULL,
                include_terms_json TEXT NOT NULL,
                exclude_terms_json TEXT NOT NULL,
                default_target_count INTEGER NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS exploration_rounds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goal_id INTEGER NOT NULL,
                round_index INTEGER NOT NULL,
                status TEXT NOT NULL,
                target_count INTEGER NOT NULL,
                approved_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(goal_id, round_index),
                FOREIGN KEY(goal_id) REFERENCES scientific_goals(id)
            );

            CREATE TABLE IF NOT EXISTS round_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                round_id INTEGER NOT NULL,
                paper_id INTEGER NOT NULL,
                rank INTEGER NOT NULL,
                selection_score REAL NOT NULL,
                selection_reason TEXT NOT NULL,
                material_tags_json TEXT NOT NULL,
                evidence_level TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(round_id, paper_id),
                FOREIGN KEY(round_id) REFERENCES exploration_rounds(id),
                FOREIGN KEY(paper_id) REFERENCES papers(id)
            );

            CREATE TABLE IF NOT EXISTS manual_download_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                round_id INTEGER NOT NULL,
                paper_id INTEGER NOT NULL,
                doi TEXT,
                doi_url TEXT,
                publisher_url TEXT,
                target_path TEXT NOT NULL,
                status TEXT NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(round_id, paper_id),
                FOREIGN KEY(round_id) REFERENCES exploration_rounds(id),
                FOREIGN KEY(paper_id) REFERENCES papers(id)
            );

            CREATE TABLE IF NOT EXISTS paper_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                round_id INTEGER NOT NULL,
                paper_id INTEGER NOT NULL,
                analysis_type TEXT NOT NULL,
                evidence_level TEXT NOT NULL,
                summary TEXT NOT NULL,
                key_findings_json TEXT NOT NULL,
                limitations_json TEXT NOT NULL,
                next_search_terms_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(round_id, paper_id),
                FOREIGN KEY(round_id) REFERENCES exploration_rounds(id),
                FOREIGN KEY(paper_id) REFERENCES papers(id)
            );

            CREATE TABLE IF NOT EXISTS round_syntheses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                round_id INTEGER NOT NULL UNIQUE,
                summary TEXT NOT NULL,
                evidence_gaps_json TEXT NOT NULL,
                next_queries_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(round_id) REFERENCES exploration_rounds(id)
            );

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
                updated_at TEXT NOT NULL,
                FOREIGN KEY(paper_id) REFERENCES papers(id),
                FOREIGN KEY(pdf_asset_id) REFERENCES pdf_assets(id)
            );
            """
        )
        self.conn.commit()

    def create_search_run(self, domain: str, from_year: int, to_year: int, query_plan: dict[str, Any]) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO search_runs(domain, from_year, to_year, status, started_at, query_plan_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (domain, from_year, to_year, "running", utc_now_iso(), json.dumps(query_plan, ensure_ascii=False)),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def finish_search_run(self, run_id: int, status: str = "finished") -> None:
        self.conn.execute(
            "UPDATE search_runs SET status = ?, finished_at = ? WHERE id = ?",
            (status, utc_now_iso(), run_id),
        )
        self.conn.commit()

    def insert_source_record(self, run_id: int | None, record: RawSourceRecord) -> int:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO source_records(run_id, source, source_id, query, raw_payload_json, retrieved_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                record.source,
                record.source_id,
                record.query,
                json.dumps(record.raw_payload, ensure_ascii=False),
                record.retrieved_at,
            ),
        )
        row = self.conn.execute(
            "SELECT id FROM source_records WHERE source = ? AND source_id = ? AND query = ?",
            (record.source, record.source_id, record.query),
        ).fetchone()
        self.conn.commit()
        return int(row["id"])

    def source_records(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM source_records ORDER BY id"))

    def insert_candidate(self, candidate: PaperCandidate) -> int:
        if candidate.source_record_id is None:
            raise ValueError("source_record_id is required to insert a candidate")
        self.conn.execute(
            """
            INSERT OR REPLACE INTO paper_candidates(
                source_record_id, source, source_id, query, doi, title, normalized_title,
                authors_json, year, venue, abstract, publisher, publisher_url, source_url,
                pdf_url, is_oa, document_type, raw_payload_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                candidate.source_record_id,
                candidate.source,
                candidate.source_id,
                candidate.query,
                candidate.doi,
                candidate.title,
                normalize_title(candidate.title),
                json.dumps(candidate.authors, ensure_ascii=False),
                candidate.year,
                candidate.venue,
                candidate.abstract,
                candidate.publisher,
                candidate.publisher_url,
                candidate.source_url,
                candidate.pdf_url,
                _bool_to_int(candidate.is_oa),
                candidate.document_type,
                json.dumps(candidate.raw_payload, ensure_ascii=False),
                utc_now_iso(),
            ),
        )
        row = self.conn.execute(
            "SELECT id FROM paper_candidates WHERE source_record_id = ?",
            (candidate.source_record_id,),
        ).fetchone()
        self.conn.commit()
        return int(row["id"])

    def candidates(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM paper_candidates ORDER BY id"))

    def upsert_paper(self, record: PaperRecord) -> int:
        now = utc_now_iso()
        existing = None
        if record.doi:
            existing = self.conn.execute("SELECT id FROM papers WHERE doi = ?", (record.doi,)).fetchone()
        if existing is None:
            existing = self.conn.execute(
                "SELECT id FROM papers WHERE normalized_title = ? AND COALESCE(year, 0) = COALESCE(?, 0)",
                (record.normalized_title, record.year),
            ).fetchone()
        if existing:
            paper_id = int(existing["id"])
            current = self.conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
            self.conn.execute(
                """
                UPDATE papers SET
                    doi = COALESCE(doi, ?),
                    title = COALESCE(NULLIF(title, ''), ?),
                    authors_json = CASE WHEN authors_json = '[]' THEN ? ELSE authors_json END,
                    year = COALESCE(year, ?),
                    venue = COALESCE(venue, ?),
                    abstract = COALESCE(abstract, ?),
                    publisher = COALESCE(publisher, ?),
                    publisher_url = COALESCE(publisher_url, ?),
                    source_url = COALESCE(source_url, ?),
                    document_type = COALESCE(document_type, ?),
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    record.doi,
                    record.title,
                    json.dumps(record.authors, ensure_ascii=False),
                    record.year,
                    record.venue,
                    record.abstract,
                    record.publisher,
                    record.publisher_url,
                    record.source_url,
                    record.document_type,
                    now,
                    paper_id,
                ),
            )
            _ = current
            self.conn.commit()
            return paper_id

        cur = self.conn.execute(
            """
            INSERT INTO papers(
                doi, title, normalized_title, authors_json, year, venue, abstract,
                publisher, publisher_url, source_url, document_type, relevance_score,
                relevance_terms_json, relevance_reason, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.doi,
                record.title,
                record.normalized_title,
                json.dumps(record.authors, ensure_ascii=False),
                record.year,
                record.venue,
                record.abstract,
                record.publisher,
                record.publisher_url,
                record.source_url,
                record.document_type,
                record.relevance_score,
                json.dumps(record.relevance_terms, ensure_ascii=False),
                record.relevance_reason,
                now,
                now,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def link_paper_source(self, paper_id: int, source_record_id: int, source: str, source_id: str, query: str) -> None:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO paper_sources(paper_id, source_record_id, source, source_id, query)
            VALUES (?, ?, ?, ?, ?)
            """,
            (paper_id, source_record_id, source, source_id, query),
        )
        self.conn.commit()

    def papers(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM papers ORDER BY id"))

    def update_relevance(self, paper_id: int, score: float, terms: list[str], reason: str) -> None:
        self.conn.execute(
            """
            UPDATE papers
            SET relevance_score = ?, relevance_terms_json = ?, relevance_reason = ?, updated_at = ?
            WHERE id = ?
            """,
            (score, json.dumps(terms, ensure_ascii=False), reason, utc_now_iso(), paper_id),
        )
        self.conn.commit()

    def upsert_access_record(self, record: AccessRecord) -> None:
        self.conn.execute(
            """
            INSERT INTO access_records(
                paper_id, doi, doi_url, is_oa, pdf_url, publisher_url, source_url, access_status, resolved_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET
                doi = excluded.doi,
                doi_url = excluded.doi_url,
                is_oa = excluded.is_oa,
                pdf_url = excluded.pdf_url,
                publisher_url = excluded.publisher_url,
                source_url = excluded.source_url,
                access_status = excluded.access_status,
                resolved_at = excluded.resolved_at
            """,
            (
                record.paper_id,
                record.doi,
                record.doi_url,
                _bool_to_int(record.is_oa),
                record.pdf_url,
                record.publisher_url,
                record.source_url,
                record.access_status,
                record.resolved_at,
            ),
        )
        self.conn.commit()

    def access_records(self) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                """
                SELECT ar.*, p.title, p.year, p.venue
                FROM access_records ar
                JOIN papers p ON p.id = ar.paper_id
                ORDER BY ar.paper_id
                """
            )
        )

    def access_records_for_download(self) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                """
                SELECT ar.*, p.title, p.year
                FROM access_records ar
                JOIN papers p ON p.id = ar.paper_id
                WHERE ar.pdf_url IS NOT NULL
                  AND ar.access_status IN ('oa_pdf_available', 'preprint_pdf', 'download_failed')
                  AND NOT EXISTS (
                    SELECT 1
                    FROM pdf_assets pa
                    WHERE pa.paper_id = ar.paper_id
                      AND pa.status IN ('downloaded_oa_pdf', 'preprint_pdf')
                      AND pa.file_path IS NOT NULL
                  )
                ORDER BY ar.paper_id
                """
            )
        )

    def access_records_for_round_download(self, round_id: int) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                """
                SELECT ar.*, p.title, p.year
                FROM round_candidates rc
                JOIN access_records ar ON ar.paper_id = rc.paper_id
                JOIN papers p ON p.id = ar.paper_id
                WHERE rc.round_id = ?
                  AND ar.pdf_url IS NOT NULL
                  AND ar.access_status IN ('oa_pdf_available', 'preprint_pdf', 'download_failed')
                  AND NOT EXISTS (
                    SELECT 1
                    FROM pdf_assets pa
                    WHERE pa.paper_id = ar.paper_id
                      AND pa.status IN ('downloaded_oa_pdf', 'preprint_pdf')
                      AND pa.file_path IS NOT NULL
                  )
                ORDER BY rc.rank, rc.id
                """,
                (round_id,),
            )
        )

    def pdf_candidates_for_paper(self, paper_id: int) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                """
                SELECT *
                FROM pdf_candidates
                WHERE paper_id = ?
                ORDER BY priority, id
                """,
                (paper_id,),
            )
        )

    def upsert_pdf_candidate(
        self,
        *,
        paper_id: int,
        pdf_url: str,
        source: str,
        priority: int,
        reason: str,
    ) -> None:
        now = utc_now_iso()
        self.conn.execute(
            """
            INSERT INTO pdf_candidates(
                paper_id, pdf_url, source, priority, reason, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(paper_id, pdf_url) DO UPDATE SET
                source = excluded.source,
                priority = MIN(pdf_candidates.priority, excluded.priority),
                reason = excluded.reason,
                updated_at = excluded.updated_at
            """,
            (paper_id, pdf_url, source, int(priority), reason, now, now),
        )
        self.conn.commit()

    def update_pdf_candidate_status(
        self,
        *,
        paper_id: int,
        pdf_url: str,
        status: str,
        error_message: str | None,
    ) -> None:
        self.conn.execute(
            """
            UPDATE pdf_candidates
            SET last_status = ?, last_error = ?, updated_at = ?
            WHERE paper_id = ? AND pdf_url = ?
            """,
            (status, error_message, utc_now_iso(), paper_id, pdf_url),
        )
        self.conn.commit()

    def upsert_pdf_asset(
        self,
        *,
        paper_id: int,
        pdf_url: str,
        file_path: str | None,
        sha256: str | None,
        file_size: int | None,
        status: str,
        error_message: str | None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO pdf_assets(
                paper_id, pdf_url, file_path, sha256, file_size, status, error_message, downloaded_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(paper_id, pdf_url) DO UPDATE SET
                file_path = excluded.file_path,
                sha256 = excluded.sha256,
                file_size = excluded.file_size,
                status = excluded.status,
                error_message = excluded.error_message,
                downloaded_at = excluded.downloaded_at
            """,
            (paper_id, pdf_url, file_path, sha256, file_size, status, error_message, utc_now_iso()),
        )
        if status in {"downloaded_oa_pdf", "preprint_pdf"}:
            self.conn.execute(
                "UPDATE access_records SET access_status = ?, resolved_at = ? WHERE paper_id = ?",
                (status, utc_now_iso(), paper_id),
            )
        elif status == "download_failed":
            self.conn.execute(
                "UPDATE access_records SET access_status = ?, resolved_at = ? WHERE paper_id = ?",
                (status, utc_now_iso(), paper_id),
            )
        self.conn.commit()

    def clear_audit_findings(self) -> None:
        self.conn.execute("DELETE FROM audit_findings")
        self.conn.commit()

    def insert_audit_finding(self, finding: AuditFinding) -> None:
        self.conn.execute(
            """
            INSERT INTO audit_findings(paper_id, severity, issue_type, message, suggestion, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                finding.paper_id,
                finding.severity,
                finding.issue_type,
                finding.message,
                finding.suggestion,
                finding.created_at,
            ),
        )
        self.conn.commit()

    def audit_findings(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM audit_findings ORDER BY severity, id"))

    def insert_source_failure(self, failure: SourceFailure) -> None:
        self.conn.execute(
            """
            INSERT INTO source_failures(run_id, source, query, failure_type, message, attempt, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                failure.run_id,
                failure.source,
                failure.query,
                failure.failure_type,
                failure.message,
                failure.attempt,
                failure.created_at,
            ),
        )
        self.conn.commit()

    def source_failures(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM source_failures ORDER BY id"))

    def upsert_llm_relevance_review(self, review: LLMRelevanceReview) -> None:
        self.conn.execute(
            """
            INSERT INTO llm_relevance_reviews(
                paper_id, provider, model, decision, confidence, reason,
                matched_domain_terms_json, exclude_reason, raw_response_json, reviewed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(paper_id, provider, model) DO UPDATE SET
                decision = excluded.decision,
                confidence = excluded.confidence,
                reason = excluded.reason,
                matched_domain_terms_json = excluded.matched_domain_terms_json,
                exclude_reason = excluded.exclude_reason,
                raw_response_json = excluded.raw_response_json,
                reviewed_at = excluded.reviewed_at
            """,
            (
                review.paper_id,
                review.provider,
                review.model,
                review.decision,
                review.confidence,
                review.reason,
                json.dumps(review.matched_domain_terms, ensure_ascii=False),
                review.exclude_reason,
                json.dumps(review.raw_response, ensure_ascii=False),
                review.reviewed_at,
            ),
        )
        self.conn.commit()

    def llm_relevance_reviews(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM llm_relevance_reviews ORDER BY id"))

    def upsert_pipeline_metric(
        self,
        *,
        run_id: int | None,
        metric_name: str,
        metric_value: float,
        metric_json: dict[str, Any] | None = None,
    ) -> None:
        if run_id is None:
            self.conn.execute(
                "DELETE FROM pipeline_metrics WHERE run_id IS NULL AND metric_name = ?",
                (metric_name,),
            )
        self.conn.execute(
            """
            INSERT INTO pipeline_metrics(run_id, metric_name, metric_value, metric_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(run_id, metric_name) DO UPDATE SET
                metric_value = excluded.metric_value,
                metric_json = excluded.metric_json,
                created_at = excluded.created_at
            """,
            (
                run_id,
                metric_name,
                metric_value,
                json.dumps(metric_json, ensure_ascii=False) if metric_json is not None else None,
                utc_now_iso(),
            ),
        )
        self.conn.commit()

    def pipeline_metrics(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM pipeline_metrics ORDER BY id"))

    def create_scientific_goal(self, goal: ScientificGoal) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO scientific_goals(
                title, description, domain, include_terms_json, exclude_terms_json,
                default_target_count, status, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                goal.title,
                goal.description,
                goal.domain,
                json.dumps(goal.include_terms, ensure_ascii=False),
                json.dumps(goal.exclude_terms, ensure_ascii=False),
                goal.default_target_count,
                goal.status,
                goal.created_at,
                goal.updated_at,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def scientific_goals(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM scientific_goals ORDER BY id"))

    def scientific_goal(self, goal_id: int) -> sqlite3.Row:
        row = self.conn.execute("SELECT * FROM scientific_goals WHERE id = ?", (goal_id,)).fetchone()
        if row is None:
            raise ValueError(f"Scientific goal not found: {goal_id}")
        return row

    def delete_scientific_goal(self, goal_id: int) -> None:
        round_ids = [row["id"] for row in self.exploration_rounds(goal_id)]
        for round_id in round_ids:
            self.conn.execute("DELETE FROM round_syntheses WHERE round_id = ?", (round_id,))
            self.conn.execute("DELETE FROM paper_analyses WHERE round_id = ?", (round_id,))
            self.conn.execute("DELETE FROM manual_download_tasks WHERE round_id = ?", (round_id,))
            self.conn.execute("DELETE FROM round_candidates WHERE round_id = ?", (round_id,))
        self.conn.execute("DELETE FROM exploration_rounds WHERE goal_id = ?", (goal_id,))
        self.conn.execute("DELETE FROM scientific_goals WHERE id = ?", (goal_id,))
        self.conn.commit()

    def create_exploration_round(self, goal_id: int, target_count: int, status: str) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(round_index), 0) + 1 AS next_index FROM exploration_rounds WHERE goal_id = ?",
            (goal_id,),
        ).fetchone()
        now = utc_now_iso()
        cur = self.conn.execute(
            """
            INSERT INTO exploration_rounds(goal_id, round_index, status, target_count, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (goal_id, int(row["next_index"]), status, target_count, now, now),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def update_round_status(self, round_id: int, status: str, *, approved: bool = False) -> None:
        now = utc_now_iso()
        if approved:
            self.conn.execute(
                "UPDATE exploration_rounds SET status = ?, approved_at = ?, updated_at = ? WHERE id = ?",
                (status, now, now, round_id),
            )
        else:
            self.conn.execute(
                "UPDATE exploration_rounds SET status = ?, updated_at = ? WHERE id = ?",
                (status, now, round_id),
            )
        self.conn.commit()

    def exploration_round(self, round_id: int) -> sqlite3.Row:
        row = self.conn.execute("SELECT * FROM exploration_rounds WHERE id = ?", (round_id,)).fetchone()
        if row is None:
            raise ValueError(f"Exploration round not found: {round_id}")
        return row

    def exploration_rounds(self, goal_id: int | None = None) -> list[sqlite3.Row]:
        if goal_id is None:
            return list(self.conn.execute("SELECT * FROM exploration_rounds ORDER BY id"))
        return list(self.conn.execute("SELECT * FROM exploration_rounds WHERE goal_id = ? ORDER BY id", (goal_id,)))

    def clear_round_candidates(self, round_id: int) -> None:
        self.conn.execute("DELETE FROM round_candidates WHERE round_id = ?", (round_id,))
        self.conn.commit()

    def upsert_round_candidate(self, candidate: RoundCandidate) -> None:
        self.conn.execute(
            """
            INSERT INTO round_candidates(
                round_id, paper_id, rank, selection_score, selection_reason,
                material_tags_json, evidence_level, status, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(round_id, paper_id) DO UPDATE SET
                rank = excluded.rank,
                selection_score = excluded.selection_score,
                selection_reason = excluded.selection_reason,
                material_tags_json = excluded.material_tags_json,
                evidence_level = excluded.evidence_level,
                status = excluded.status
            """,
            (
                candidate.round_id,
                candidate.paper_id,
                candidate.rank,
                candidate.selection_score,
                candidate.selection_reason,
                json.dumps(candidate.material_tags, ensure_ascii=False),
                candidate.evidence_level,
                candidate.status,
                candidate.created_at,
            ),
        )
        self.conn.commit()

    def round_candidates(self, round_id: int) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                """
                SELECT rc.*, p.title, p.doi, p.year, p.venue, p.abstract, ar.access_status, ar.pdf_url,
                       ar.doi_url, ar.publisher_url, pa.file_path AS local_pdf_path,
                       pa.file_size AS local_pdf_size, pa.status AS local_pdf_status
                FROM round_candidates rc
                JOIN papers p ON p.id = rc.paper_id
                LEFT JOIN access_records ar ON ar.paper_id = p.id
                LEFT JOIN pdf_assets pa ON pa.id = (
                    SELECT pa2.id
                    FROM pdf_assets pa2
                    WHERE pa2.paper_id = p.id
                      AND pa2.status IN ('downloaded_oa_pdf', 'preprint_pdf')
                      AND pa2.file_path IS NOT NULL
                    ORDER BY pa2.downloaded_at DESC, pa2.id DESC
                    LIMIT 1
                )
                WHERE rc.round_id = ?
                ORDER BY rc.rank, rc.id
                """,
                (round_id,),
            )
        )

    def upsert_manual_download_task(self, task: ManualDownloadTask) -> None:
        self.conn.execute(
            """
            INSERT INTO manual_download_tasks(
                round_id, paper_id, doi, doi_url, publisher_url, target_path,
                status, reason, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(round_id, paper_id) DO UPDATE SET
                doi = excluded.doi,
                doi_url = excluded.doi_url,
                publisher_url = excluded.publisher_url,
                target_path = excluded.target_path,
                status = excluded.status,
                reason = excluded.reason,
                updated_at = excluded.updated_at
            """,
            (
                task.round_id,
                task.paper_id,
                task.doi,
                task.doi_url,
                task.publisher_url,
                task.target_path,
                task.status,
                task.reason,
                task.created_at,
                task.updated_at,
            ),
        )
        self.conn.commit()

    def manual_download_tasks(self, round_id: int | None = None) -> list[sqlite3.Row]:
        query = """
            SELECT mdt.*, p.title, p.year, p.venue
            FROM manual_download_tasks mdt
            JOIN papers p ON p.id = mdt.paper_id
        """
        if round_id is None:
            return list(self.conn.execute(query + " ORDER BY mdt.id"))
        return list(self.conn.execute(query + " WHERE mdt.round_id = ? ORDER BY mdt.id", (round_id,)))

    def update_manual_download_task_status(self, task_id: int, status: str) -> None:
        self.conn.execute(
            "UPDATE manual_download_tasks SET status = ?, updated_at = ? WHERE id = ?",
            (status, utc_now_iso(), task_id),
        )
        self.conn.commit()

    def upsert_paper_analysis(self, analysis: PaperAnalysis) -> None:
        self.conn.execute(
            """
            INSERT INTO paper_analyses(
                round_id, paper_id, analysis_type, evidence_level, summary,
                key_findings_json, limitations_json, next_search_terms_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(round_id, paper_id) DO UPDATE SET
                analysis_type = excluded.analysis_type,
                evidence_level = excluded.evidence_level,
                summary = excluded.summary,
                key_findings_json = excluded.key_findings_json,
                limitations_json = excluded.limitations_json,
                next_search_terms_json = excluded.next_search_terms_json,
                created_at = excluded.created_at
            """,
            (
                analysis.round_id,
                analysis.paper_id,
                analysis.analysis_type,
                analysis.evidence_level,
                analysis.summary,
                json.dumps(analysis.key_findings, ensure_ascii=False),
                json.dumps(analysis.limitations, ensure_ascii=False),
                json.dumps(analysis.next_search_terms, ensure_ascii=False),
                analysis.created_at,
            ),
        )
        self.conn.commit()

    def paper_analyses(self, round_id: int) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                """
                SELECT pa.*, p.title, p.year, p.venue, p.doi
                FROM paper_analyses pa
                JOIN papers p ON p.id = pa.paper_id
                WHERE pa.round_id = ?
                ORDER BY pa.id
                """,
                (round_id,),
            )
        )

    def upsert_round_synthesis(self, synthesis: RoundSynthesis) -> None:
        self.conn.execute(
            """
            INSERT INTO round_syntheses(
                round_id, summary, evidence_gaps_json, next_queries_json, created_at
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(round_id) DO UPDATE SET
                summary = excluded.summary,
                evidence_gaps_json = excluded.evidence_gaps_json,
                next_queries_json = excluded.next_queries_json,
                created_at = excluded.created_at
            """,
            (
                synthesis.round_id,
                synthesis.summary,
                json.dumps(synthesis.evidence_gaps, ensure_ascii=False),
                json.dumps(synthesis.next_queries, ensure_ascii=False),
                synthesis.created_at,
            ),
        )
        self.conn.commit()

    def round_synthesis(self, round_id: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM round_syntheses WHERE round_id = ?", (round_id,)).fetchone()

    def pdf_assets_for_conversion(self, round_id: int | None = None) -> list[sqlite3.Row]:
        if round_id is None:
            return list(
                self.conn.execute(
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
            )
        return list(
            self.conn.execute(
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
        return list(
            self.conn.execute(
                """
                SELECT pc.*, p.title, p.doi, p.year, p.venue
                FROM paper_conversions pc
                LEFT JOIN papers p ON p.id = pc.paper_id
                ORDER BY pc.updated_at DESC, pc.id DESC
                LIMIT ?
                """,
                (limit,),
            )
        )

    def rows(self, query: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        return list(self.conn.execute(query, tuple(params)))


def _bool_to_int(value: bool | None) -> int | None:
    if value is None:
        return None
    return 1 if value else 0

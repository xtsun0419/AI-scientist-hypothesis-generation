from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class ResearcherLibraryDB:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS library_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                external_key TEXT NOT NULL UNIQUE,
                source_type TEXT NOT NULL CHECK(source_type IN ('personal', 'agent')),
                title TEXT NOT NULL,
                abstract TEXT,
                content TEXT,
                doi TEXT,
                venue TEXT,
                year INTEGER,
                source_path TEXT,
                imported_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS researcher_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                rationale TEXT NOT NULL,
                item_ids_json TEXT NOT NULL,
                mode TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS hypothesis_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                researcher_question TEXT NOT NULL,
                hypothesis TEXT NOT NULL,
                rationale TEXT NOT NULL,
                validation TEXT NOT NULL,
                item_ids_json TEXT NOT NULL,
                mode TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS hypothesis_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL UNIQUE,
                status TEXT NOT NULL CHECK(status IN ('continue', 'warning', 'interrupt')),
                issue TEXT NOT NULL,
                rationale TEXT NOT NULL,
                restart_instruction TEXT,
                evidence_ids_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES hypothesis_runs(id)
            );
            """
        )
        if self.get_setting("enabled") is None:
            self.set_setting("enabled", False)
        self.conn.commit()

    def get_setting(self, key: str) -> Any | None:
        row = self.conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value"]) if row else None

    def set_setting(self, key: str, value: Any) -> None:
        self.conn.execute(
            """
            INSERT INTO settings(key, value, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (key, json.dumps(value, ensure_ascii=False), utc_now_iso()),
        )
        self.conn.commit()

    def upsert_item(self, item: dict[str, Any]) -> None:
        now = utc_now_iso()
        self.conn.execute(
            """
            INSERT INTO library_items(
                external_key, source_type, title, abstract, content, doi, venue, year, source_path, imported_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(external_key) DO UPDATE SET
                source_type = excluded.source_type, title = excluded.title, abstract = excluded.abstract,
                content = excluded.content, doi = excluded.doi, venue = excluded.venue, year = excluded.year,
                source_path = excluded.source_path, updated_at = excluded.updated_at
            """,
            (
                item["external_key"], item["source_type"], item["title"], item.get("abstract"), item.get("content"),
                item.get("doi"), item.get("venue"), item.get("year"), item.get("source_path"), now, now,
            ),
        )
        self.conn.commit()

    def recent_items(self, limit: int = 40) -> list[dict[str, Any]]:
        return [dict(row) for row in self.conn.execute("SELECT * FROM library_items ORDER BY updated_at DESC, id DESC LIMIT ?", (limit,))]

    def metrics(self) -> dict[str, int]:
        rows = self.conn.execute("SELECT source_type, COUNT(*) AS n FROM library_items GROUP BY source_type").fetchall()
        counts = {str(row["source_type"]): int(row["n"]) for row in rows}
        questions = self.conn.execute("SELECT COUNT(*) AS n FROM researcher_questions").fetchone()
        hypotheses = self.conn.execute("SELECT COUNT(*) AS n FROM hypothesis_runs").fetchone()
        return {"total": sum(counts.values()), "personal": counts.get("personal", 0), "agent": counts.get("agent", 0), "questions": int(questions["n"]), "hypotheses": int(hypotheses["n"])}

    def add_question(self, *, question: str, rationale: str, item_ids: list[int], mode: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO researcher_questions(question, rationale, item_ids_json, mode, created_at) VALUES (?, ?, ?, ?, ?)",
            (question, rationale, json.dumps(item_ids), mode, utc_now_iso()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def recent_questions(self, limit: int = 8) -> list[dict[str, Any]]:
        return [dict(row) for row in self.conn.execute("SELECT * FROM researcher_questions ORDER BY id DESC LIMIT ?", (limit,))]

    def add_hypothesis_run(
        self,
        *,
        researcher_question: str,
        hypothesis: str,
        rationale: str,
        validation: str,
        item_ids: list[int],
        mode: str,
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO hypothesis_runs(researcher_question, hypothesis, rationale, validation, item_ids_json, mode, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (researcher_question, hypothesis, rationale, validation, json.dumps(item_ids), mode, utc_now_iso()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def recent_hypotheses(self, limit: int = 8) -> list[dict[str, Any]]:
        return [dict(row) for row in self.conn.execute("SELECT * FROM hypothesis_runs ORDER BY id DESC LIMIT ?", (limit,))]

    def add_hypothesis_review(
        self,
        *,
        run_id: int,
        status: str,
        issue: str,
        rationale: str,
        restart_instruction: str | None,
        evidence_ids: list[str],
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO hypothesis_reviews(run_id, status, issue, rationale, restart_instruction, evidence_ids_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                status = excluded.status, issue = excluded.issue, rationale = excluded.rationale,
                restart_instruction = excluded.restart_instruction, evidence_ids_json = excluded.evidence_ids_json,
                created_at = excluded.created_at
            """,
            (run_id, status, issue, rationale, restart_instruction, json.dumps(evidence_ids), utc_now_iso()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def recent_hypothesis_reviews(self, limit: int = 24) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.conn.execute("SELECT * FROM hypothesis_reviews ORDER BY run_id DESC LIMIT ?", (limit,))
        ]

    def review_for_runs(self, run_ids: list[int]) -> dict[int, dict[str, Any]]:
        if not run_ids:
            return {}
        placeholders = ", ".join("?" for _ in run_ids)
        rows = self.conn.execute(f"SELECT * FROM hypothesis_reviews WHERE run_id IN ({placeholders})", run_ids).fetchall()
        return {int(row["run_id"]): dict(row) for row in rows}

    def items_by_ids(self, ids: list[int]) -> dict[int, dict[str, Any]]:
        if not ids:
            return {}
        placeholders = ", ".join("?" for _ in ids)
        rows = self.conn.execute(f"SELECT * FROM library_items WHERE id IN ({placeholders})", ids).fetchall()
        return {int(row["id"]): dict(row) for row in rows}

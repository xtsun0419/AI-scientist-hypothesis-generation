from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class QuestionSynthesisDB:
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
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_key TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                context_fingerprint TEXT NOT NULL,
                model TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                speaker TEXT NOT NULL,
                content TEXT NOT NULL,
                model TEXT,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS confirmed_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                problem_statement TEXT NOT NULL,
                variables_json TEXT NOT NULL,
                mechanism_hypothesis TEXT NOT NULL,
                validation_criteria_json TEXT NOT NULL,
                evidence_ids_json TEXT NOT NULL,
                source_message_id INTEGER,
                mode TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            );
            """
        )
        self.conn.commit()

    def get_session(self, session_key: str = "latest") -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM sessions WHERE session_key = ?", (session_key,)).fetchone()

    def create_or_update_session(
        self,
        *,
        session_key: str,
        title: str,
        context_fingerprint: str,
        model: str,
    ) -> int:
        now = utc_now_iso()
        self.conn.execute(
            """
            INSERT INTO sessions(session_key, title, context_fingerprint, model, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_key) DO UPDATE SET
                title = excluded.title,
                context_fingerprint = excluded.context_fingerprint,
                model = excluded.model,
                updated_at = excluded.updated_at
            """,
            (session_key, title, context_fingerprint, model, now, now),
        )
        row = self.get_session(session_key)
        self.conn.commit()
        return int(row["id"])

    def clear_messages(self, session_id: int) -> None:
        self.conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        self.conn.commit()

    def add_message(
        self,
        *,
        session_id: int,
        role: str,
        speaker: str,
        content: str,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO messages(session_id, role, speaker, content, model, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, role, speaker, content, model, json.dumps(metadata or {}, ensure_ascii=False), utc_now_iso()),
        )
        self.conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (utc_now_iso(), session_id))
        self.conn.commit()
        return int(cur.lastrowid)

    def messages(self, session_id: int) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM messages WHERE session_id = ? ORDER BY id", (session_id,)))

    def has_human_messages(self, session_id: int) -> bool:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE session_id = ? AND role = 'user'",
            (session_id,),
        ).fetchone()
        return int(row["n"] or 0) > 0

    def reset(self, session_key: str = "latest") -> None:
        session = self.get_session(session_key)
        if session is not None:
            self.clear_messages(int(session["id"]))

    def add_confirmed_question(
        self,
        *,
        session_id: int,
        problem_statement: str,
        variables: list[str],
        mechanism_hypothesis: str,
        validation_criteria: list[str],
        evidence_ids: list[str],
        source_message_id: int | None = None,
        mode: str = "llm",
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO confirmed_questions(
                session_id, problem_statement, variables_json, mechanism_hypothesis,
                validation_criteria_json, evidence_ids_json, source_message_id, mode, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                problem_statement,
                json.dumps(variables, ensure_ascii=False),
                mechanism_hypothesis,
                json.dumps(validation_criteria, ensure_ascii=False),
                json.dumps(evidence_ids, ensure_ascii=False),
                source_message_id,
                mode,
                utc_now_iso(),
            ),
        )
        self.conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (utc_now_iso(), session_id))
        self.conn.commit()
        return int(cur.lastrowid)

    def confirmed_questions(self, session_key: str = "latest") -> list[sqlite3.Row]:
        session = self.get_session(session_key)
        if session is None:
            return []
        return list(
            self.conn.execute(
                "SELECT * FROM confirmed_questions WHERE session_id = ? ORDER BY id DESC",
                (int(session["id"]),),
            )
        )

    def human_intervention_count(self, session_id: int) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE session_id = ? AND role = 'user'",
            (session_id,),
        ).fetchone()
        return int(row["n"] or 0)


def confirmed_question_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    for key, default in (
        ("variables_json", []),
        ("validation_criteria_json", []),
        ("evidence_ids_json", []),
    ):
        if key in data:
            try:
                data[key.removesuffix("_json")] = json.loads(data.pop(key) or "[]")
            except json.JSONDecodeError:
                data[key.removesuffix("_json")] = default
    return data


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    if "metadata_json" in data:
        try:
            data["metadata"] = json.loads(data.pop("metadata_json") or "{}")
        except json.JSONDecodeError:
            data["metadata"] = {}
    return data


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

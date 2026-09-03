from __future__ import annotations

import sqlite3
from pathlib import Path

from researcher_library_agent.agent import ResearcherLibraryAgent
from researcher_library_agent.db import ResearcherLibraryDB


def make_agent(tmp_path: Path) -> ResearcherLibraryAgent:
    return ResearcherLibraryAgent(ResearcherLibraryDB(tmp_path / "library.sqlite"))


def test_personal_import_and_disabled_context(tmp_path: Path) -> None:
    source = tmp_path / "personal_notes.md"
    source.write_text("A personal research note about a falsifiable mechanism.", encoding="utf-8")
    agent = make_agent(tmp_path)
    try:
        assert agent.import_path(str(source))["imported"] == 1
        state = agent.state()
        assert state["metrics"]["personal"] == 1
        assert agent.design_context()["enabled"] is False
    finally:
        agent.close()


def test_sync_and_question_use_only_library_items(tmp_path: Path, monkeypatch) -> None:
    retrieval = tmp_path / "literature.sqlite"
    with sqlite3.connect(retrieval) as conn:
        conn.execute("CREATE TABLE papers (id INTEGER, doi TEXT, title TEXT, abstract TEXT, venue TEXT, year INTEGER)")
        conn.execute(
            "INSERT INTO papers VALUES (1, '10.1/example', 'Agent paper', 'Evidence from the agent corpus.', 'Journal', 2026)"
        )
    agent = make_agent(tmp_path)
    try:
        assert agent.sync_agent_literature(retrieval)["synced"] == 1
        assert agent.state()["metrics"]["agent"] == 1
        agent.set_enabled(True)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        response = agent.ask_researcher_question()
        assert response["mode"] == "fallback_no_api_key"
        assert response["item_ids"]
        context = agent.design_context()
        assert context["enabled"] is True
        assert context["cards"][0]["evidence_ids"] == ["library:1"]
        assert "questions" not in context
    finally:
        agent.close()


def test_pubmed_import_and_isolated_hypothesis_dialogue(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "researcher_paper.md"
    source.write_text("Personal paper: variable A may influence outcome B.", encoding="utf-8")
    agent = make_agent(tmp_path)
    try:
        agent.import_path(str(source))
        monkeypatch.setattr(
            "researcher_library_agent.agent.fetch_pubmed_records",
            lambda ids, email: [
                {
                    "requested_id": "123",
                    "pmid": "123",
                    "pmcid": "",
                    "doi": "10.1/example",
                    "title": "Agent paper",
                    "abstract": "Agent evidence links variable A to outcome B.",
                    "venue": "Journal",
                    "year": "2026",
                }
            ],
        )
        assert agent.import_pubmed(["123"])["imported"] == 1
        agent.set_enabled(True)
        monkeypatch.setattr(
            "researcher_library_agent.agent._ask_llm",
            lambda snapshot: ({"question": "Does A cause B?", "rationale": "Compare both records.", "item_ids": [1, 2]}, "llm"),
        )
        monkeypatch.setattr(
            "researcher_library_agent.agent._design_hypothesis",
            lambda payload: ({"hypothesis": "A causes B under condition C.", "rationale": "Both records support testing.", "validation": "Use a controlled comparison."}, "llm"),
        )
        run = agent.run_hypothesis_dialogue()
        assert run["researcher_question"] == "Does A cause B?"
        assert run["hypothesis"] == "A causes B under condition C."
        assert agent.state()["metrics"]["hypotheses"] == 1
    finally:
        agent.close()

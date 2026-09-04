from __future__ import annotations

import sqlite3
from pathlib import Path

from researcher_library_agent.agent import ResearcherLibraryAgent
from researcher_library_agent.db import ResearcherLibraryDB


def make_agent(tmp_path: Path) -> ResearcherLibraryAgent:
    return ResearcherLibraryAgent(ResearcherLibraryDB(tmp_path / "library.sqlite"), memory_path=tmp_path / "MEMORY.md")


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
        monkeypatch.setattr(
            "researcher_library_agent.agent.assess_hypothesis",
            lambda public_run, recent_runs, items: {"issue": "none", "rationale": "Independent evidence is sufficient.", "restart_instruction": "", "evidence_ids": ["library:1", "library:2"]},
        )
        run = agent.run_hypothesis_dialogue()
        assert run["researcher_question"] == "Does A cause B?"
        assert run["hypothesis"] == "A causes B under condition C."
        assert agent.state()["metrics"]["hypotheses"] == 1
        memory = (tmp_path / "MEMORY.md").read_text(encoding="utf-8")
        assert "Does A cause B?" in memory
        assert "A causes B under condition C." in memory

        captured = {}
        monkeypatch.setattr(
            "researcher_library_agent.agent._ask_llm",
            lambda snapshot: (captured.update(snapshot) or {"question": "What should be tested next?", "rationale": "Memory-aware follow-up.", "item_ids": [1, 2]}, "llm"),
        )
        agent.ask_researcher_question()
        assert "A causes B under condition C." in captured["public_dialogue_memory"]
    finally:
        agent.close()


def test_watchdog_interrupts_after_three_repeated_narrow_runs(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "personal.md"
    source.write_text("Personal evidence about variable A and outcome B.", encoding="utf-8")
    agent = make_agent(tmp_path)
    try:
        agent.import_path(str(source))
        agent.db.upsert_item(
            {"external_key": "agent:test", "source_type": "agent", "title": "Agent evidence", "abstract": "Evidence about variable A and outcome B.", "content": "Evidence about variable A and outcome B."}
        )
        agent.set_enabled(True)
        monkeypatch.setattr(
            "researcher_library_agent.agent._ask_llm",
            lambda snapshot: ({"question": "Does A cause B?", "rationale": "Repeat check.", "item_ids": [1, 2]}, "llm"),
        )
        monkeypatch.setattr(
            "researcher_library_agent.agent._design_hypothesis",
            lambda payload: (
                {"hypothesis": "Alternative mechanism C should be tested.", "rationale": "Reset result.", "validation": "Compare C against A."}
                if payload.get("published_watchdog_instruction")
                else {"hypothesis": "A causes B under condition C.", "rationale": "Repeated result.", "validation": "Measure A and B."},
                "llm",
            ),
        )
        monkeypatch.setattr(
            "researcher_library_agent.agent.assess_hypothesis",
            lambda public_run, recent_runs, items: {"issue": "narrow", "rationale": "The public trajectory is repeatedly narrow.", "restart_instruction": "Compare alternative mechanism C.", "evidence_ids": ["library:1", "library:2"]},
        )

        first = agent.run_hypothesis_dialogue()
        second = agent.run_hypothesis_dialogue()
        third = agent.run_hypothesis_dialogue()

        assert first["watchdog"]["status"] == "warning"
        assert second["watchdog"]["status"] == "warning"
        assert third["watchdog"]["status"] == "interrupt"
        assert third["rethink"]["hypothesis"] == "Alternative mechanism C should be tested."
        memory = (tmp_path / "MEMORY.md").read_text(encoding="utf-8")
        assert "### Watchdog Review" in memory
        assert "### Public Reset Instruction" in memory
    finally:
        agent.close()

from pathlib import Path

from question_synthesis_agent.agent import QuestionSynthesisAgent
from question_synthesis_agent.db import QuestionSynthesisDB
from question_synthesis_agent.llm import LLMSettings


def make_db(tmp_path: Path) -> QuestionSynthesisDB:
    db = QuestionSynthesisDB(tmp_path / "question_synthesis.sqlite")
    db.init_schema()
    return db


def test_initial_state_uses_fallback_without_llm(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    db = make_db(tmp_path)
    try:
        state = QuestionSynthesisAgent(db, settings=None).state()
        assert state["model_name"] == "LLM 未配置 - 本地规则草稿"
        assert len(state["messages"]) == 2
        assert state["messages"][0]["role"] == "retrieval"
        assert state["messages"][1]["role"] == "assistant"
    finally:
        db.close()


def test_chat_persists_fake_llm_response(tmp_path: Path) -> None:
    class FakeClient:
        def __init__(self, settings: LLMSettings):
            self.settings = settings

        def chat(self, *, system_prompt: str, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
            assert "科学问题归纳 Agent" in system_prompt
            assert messages
            return "建议聚焦到一个可验证的机制问题。"

    db = make_db(tmp_path)
    try:
        agent = QuestionSynthesisAgent(
            db,
            settings=LLMSettings(base_url="https://example.test/v1", api_key="test", model="fake-model"),
            client_factory=FakeClient,
        )
        state = agent.chat("我想偏机制型问题")
        assert state["model_name"] == "fake-model"
        assert state["messages"][-2]["role"] == "user"
        assert state["messages"][-1]["content"] == "建议聚焦到一个可验证的机制问题。"
    finally:
        db.close()

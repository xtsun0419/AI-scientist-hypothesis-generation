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


def test_confirm_requires_prior_human_message(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    db = make_db(tmp_path)
    try:
        agent = QuestionSynthesisAgent(db, settings=None)
        agent.state()  # 初始化后只有检索+初始回复，没有 human 消息
        try:
            agent.confirm()
            raise AssertionError("应当抛出异常")
        except ValueError as exc:
            assert "暂无可确认内容" in str(exc)
    finally:
        db.close()


def test_confirm_fallback_persists_structured_question(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    db = make_db(tmp_path)
    try:
        agent = QuestionSynthesisAgent(db, settings=None)
        agent.chat("我想研究晶界相厚度对矫顽力的影响机制")
        state = agent.confirm()
        confirmed = state["confirmed_questions"]
        assert len(confirmed) == 1
        item = confirmed[0]
        assert item["problem_statement"].startswith("我想研究")
        assert item["variables"]
        assert item["mechanism_hypothesis"]
        assert item["validation_criteria"]
        assert item["mode"] == "fallback_no_api_key"
        assert item["source_message_id"] is not None
        assert state["interventions"]["human_messages"] == 1
        # 确认结果也写入对话记录
        roles = [message["role"] for message in state["messages"]]
        assert "confirm" in roles
    finally:
        db.close()


def test_confirm_via_fake_llm_filters_hallucinated_evidence_ids(tmp_path: Path) -> None:
    import json

    class FakeConfirmClient:
        def __init__(self, settings: LLMSettings):
            self.settings = settings

        def chat(self, *, system_prompt: str, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
            assert "科学问题确认器" in system_prompt
            return json.dumps(
                {
                    "problem_statement": "测试问题：变量 X 如何影响性能 Y",
                    "variables": ["X"],
                    "mechanism_hypothesis": "待验证机制",
                    "validation_criteria": ["用实验验证"],
                    "evidence_ids": ["幻觉id1", "幻觉id2"],
                },
                ensure_ascii=False,
            )

    db = make_db(tmp_path)
    try:
        agent = QuestionSynthesisAgent(
            db,
            settings=LLMSettings(base_url="https://example.test/v1", api_key="test", model="fake-model"),
            client_factory=FakeConfirmClient,
        )
        agent.chat("请确认：变量 X 如何影响性能 Y")
        state = agent.confirm()
        confirmed = state["confirmed_questions"][0]
        assert confirmed["mode"] == "llm"
        assert confirmed["problem_statement"] == "测试问题：变量 X 如何影响性能 Y"
        assert confirmed["evidence_ids"] == []  # 幻觉 id 被过滤
    finally:
        db.close()

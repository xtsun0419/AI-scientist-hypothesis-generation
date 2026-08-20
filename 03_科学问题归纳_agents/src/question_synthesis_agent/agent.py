from __future__ import annotations

import json
from typing import Any, Callable

from .context import collect_context
from .db import QuestionSynthesisDB, confirmed_question_to_dict, row_to_dict
from .llm import LLMSettings, OpenAICompatibleChatClient


SYSTEM_PROMPT = """你是 AI Scientist 工作流第 3 模块的“科学问题归纳 Agent”。
你的任务是把文献检索提出的问题、下一轮 query、证据缺口，以及文献分析模块生成的 Paper Cards / Wiki 证据，收敛成可研究、可验证、可继续进入路线设计的科学问题。
请始终用中文回答。不要编造文献、DOI 或实验结果；如果证据不足，要明确说“证据不足”。优先引用上下文里的 evidence ids、topic、材料体系、方法和性能指标。
回答要帮助研究者逐步决策，而不是泛泛综述。
每次回答最后必须附加一个独立小节，格式严格如下：
后续可追问：
- 问题1
- 问题2
- 问题3
这 3 个问题要是你认为最有助于细化研究方向的下一步追问。"""


CONFIRM_SYSTEM_PROMPT = """你是 AI Scientist 工作流第 3 模块的“科学问题确认器”。
研究者已经就某个科学问题完成了讨论，现在要把收敛结果结构化保存，供第 4 模块生成研究路线使用。
请基于给定的上下文 JSON 和对话记录，输出严格 JSON，字段必须为：
- problem_statement: 一句话问题陈述（“在 X 体系中，研究 Y 如何通过 Z 影响 W”）
- variables: 关键变量数组（材料/结构/工艺变量）
- mechanism_hypothesis: 机制假设（因果链描述，尚无证据也要写清“待验证”字样）
- validation_criteria: 验证判据数组（用什么计算/实验指标判断假设成立）
- evidence_ids: 证据 id 数组，只能从上下文提供的 evidence_ids 池中选择，找不到就写 []
不要编造文献、DOI、实验结果或池中不存在的 evidence id。"""


_UNSET = object()


class QuestionSynthesisAgent:
    def __init__(
        self,
        db: QuestionSynthesisDB,
        *,
        settings: LLMSettings | None | object = _UNSET,
        client_factory: Callable[[LLMSettings], Any] = OpenAICompatibleChatClient,
    ):
        self.db = db
        self.settings = LLMSettings.from_env() if settings is _UNSET else settings
        self.client_factory = client_factory

    def state(self, *, session_key: str = "latest") -> dict[str, Any]:
        context = collect_context()
        session_id = self.ensure_initialized(context=context, session_key=session_key)
        session = self.db.get_session(session_key)
        messages = [row_to_dict(row) for row in self.db.messages(session_id)]
        confirmed = [confirmed_question_to_dict(row) for row in self.db.confirmed_questions(session_key)]
        return {
            "session": dict(session) if session is not None else None,
            "messages": messages,
            "confirmed_questions": confirmed,
            "interventions": {"human_messages": self.db.human_intervention_count(session_id)},
            "context": context,
            "model_name": self.model_name,
            "llm_configured": self.settings is not None,
        }

    def chat(self, message: str, *, session_key: str = "latest") -> dict[str, Any]:
        message = message.strip()
        if not message:
            raise ValueError("请输入要讨论的问题或方向。")
        context = collect_context()
        session_id = self.ensure_initialized(context=context, session_key=session_key)
        self.db.add_message(
            session_id=session_id,
            role="user",
            speaker="研究者",
            content=message,
            metadata={"source": "human_input"},
        )
        prior = [row_to_dict(row) for row in self.db.messages(session_id)][-12:]
        response, metadata = self._respond(context=context, prior_messages=prior, latest_user_message=message)
        self.db.add_message(
            session_id=session_id,
            role="assistant",
            speaker="科学问题归纳 LLM",
            content=response,
            model=self.model_name,
            metadata=metadata,
        )
        return self.state(session_key=session_key)

    def reset(self, *, session_key: str = "latest") -> dict[str, Any]:
        self.db.reset(session_key)
        return self.state(session_key=session_key)

    def confirm(self, *, session_key: str = "latest") -> dict[str, Any]:
        """把当前对话收敛结果结构化为一条确认的科学问题，存入 confirmed_questions。"""
        context = collect_context()
        session_id = self.ensure_initialized(context=context, session_key=session_key)
        messages = [row_to_dict(row) for row in self.db.messages(session_id)]
        anchor = _latest_human_message(messages)
        if anchor is None:
            raise ValueError("暂无可确认内容：请先输入你想确认的科学问题或方向。")
        evidence_pool = collect_evidence_ids(context)
        if self.settings is not None:
            try:
                structured, mode = self._confirm_via_llm(context, messages, evidence_pool), "llm"
            except Exception as exc:
                structured, mode = _fallback_confirm(context, anchor, evidence_pool), "fallback_llm_error"
        else:
            structured, mode = _fallback_confirm(context, anchor, evidence_pool), "fallback_no_api_key"
        structured["evidence_ids"] = _filter_evidence_ids(structured.get("evidence_ids"), evidence_pool)
        question_id = self.db.add_confirmed_question(
            session_id=session_id,
            problem_statement=structured["problem_statement"],
            variables=structured["variables"],
            mechanism_hypothesis=structured["mechanism_hypothesis"],
            validation_criteria=structured["validation_criteria"],
            evidence_ids=structured["evidence_ids"],
            source_message_id=int(anchor["id"]) if anchor.get("id") is not None else None,
            mode=mode,
        )
        self.db.add_message(
            session_id=session_id,
            role="confirm",
            speaker="科学问题确认器",
            content=_confirm_summary(structured),
            model=self.model_name,
            metadata={"source": "confirm_action", "confirmed_question_id": question_id, "mode": mode},
        )
        return self.state(session_key=session_key)

    def _confirm_via_llm(
        self,
        context: dict[str, Any],
        messages: list[dict[str, Any]],
        evidence_pool: list[str],
    ) -> dict[str, Any]:
        assert self.settings is not None
        prompt = (
            "请把对话中最新确认的研究方向结构化为可进入路线设计的科学问题。\n"
            f"证据 id 池：{json.dumps(evidence_pool, ensure_ascii=False)}\n"
            f"最近对话（新→旧，最多 10 条）：\n"
            + _recent_messages_text(messages)
            + "\n输出严格 JSON，不要包含解释文字。"
        )
        content = self.client_factory(self.settings).chat(system_prompt=CONFIRM_SYSTEM_PROMPT, messages=[{"role": "user", "content": prompt}])
        return parse_confirm_json(content, context, messages)

    def ensure_initialized(self, *, context: dict[str, Any], session_key: str = "latest") -> int:
        title = _session_title(context)
        fingerprint = str(context.get("fingerprint") or "")
        session = self.db.get_session(session_key)
        session_id = self.db.create_or_update_session(
            session_key=session_key,
            title=title,
            context_fingerprint=fingerprint,
            model=self.model_name,
        )
        if session is not None and session["context_fingerprint"] != fingerprint and not self.db.has_human_messages(session_id):
            self.db.clear_messages(session_id)
        if self.db.messages(session_id):
            return session_id
        retrieval_message = retrieval_question_message(context)
        self.db.add_message(
            session_id=session_id,
            role="retrieval",
            speaker="文献检索 Agent",
            content=retrieval_message,
            metadata={"source": "literature_retrieval"},
        )
        response, metadata = self._initial_response(context)
        self.db.add_message(
            session_id=session_id,
            role="assistant",
            speaker="科学问题归纳 LLM",
            content=response,
            model=self.model_name,
            metadata=metadata,
        )
        return session_id

    @property
    def model_name(self) -> str:
        if self.settings is None:
            return "LLM 未配置 - 本地规则草稿"
        return self.settings.model

    def _initial_response(self, context: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        prompt = (
            "请基于以下 AI Scientist 上下文，生成模块打开时右侧 LLM 对话框的初始回答。\n"
            "输出结构：1) 文献分析给出的核心结果；2) 可收敛的科学问题候选；3) 建议优先细化方向；4) 下一步想问研究者的 2-3 个问题。\n"
            f"上下文 JSON：\n{_context_json(context)}"
        )
        return self._call_llm_or_fallback(prompt=prompt, context=context, latest_user_message=None)

    def _respond(
        self,
        *,
        context: dict[str, Any],
        prior_messages: list[dict[str, Any]],
        latest_user_message: str,
    ) -> tuple[str, dict[str, Any]]:
        messages = [{"role": "user", "content": "当前证据上下文 JSON：\n" + _context_json(context)}]
        for item in prior_messages:
            role = "assistant" if item["role"] == "assistant" else "user"
            messages.append({"role": role, "content": f"{item['speaker']}：{item['content']}"})
        if self.settings is None:
            return finalize_response(
                fallback_chat_response(context, latest_user_message),
                {"mode": "fallback_no_api_key"},
                context,
            )
        try:
            content = self.client_factory(self.settings).chat(system_prompt=SYSTEM_PROMPT, messages=messages)
            return finalize_response(content or fallback_chat_response(context, latest_user_message), {"mode": "llm"}, context)
        except Exception as exc:
            return finalize_response(
                fallback_chat_response(context, latest_user_message),
                {"mode": "fallback_llm_error", "error": str(exc)},
                context,
            )

    def _call_llm_or_fallback(
        self,
        *,
        prompt: str,
        context: dict[str, Any],
        latest_user_message: str | None,
    ) -> tuple[str, dict[str, Any]]:
        if self.settings is None:
            return finalize_response(fallback_initial_response(context), {"mode": "fallback_no_api_key"}, context)
        try:
            content = self.client_factory(self.settings).chat(
                system_prompt=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return finalize_response(content or fallback_initial_response(context), {"mode": "llm"}, context)
        except Exception as exc:
            fallback = fallback_initial_response(context) if latest_user_message is None else fallback_chat_response(context, latest_user_message)
            return finalize_response(fallback, {"mode": "fallback_llm_error", "error": str(exc)}, context)


def retrieval_question_message(context: dict[str, Any]) -> str:
    goal = context.get("latest_goal") or {}
    questions = context.get("retrieval_questions") or []
    gaps = context.get("evidence_gaps") or []
    lines = ["文献检索阶段提出的问题和下一步线索："]
    if goal.get("title"):
        lines.append(f"- 当前科学问题：{goal['title']}")
    if goal.get("description"):
        lines.append(f"- 补充说明：{goal['description']}")
    if questions:
        lines.append("- 检索提出的 query / 问题：")
        lines.extend(f"  {index}. {item}" for index, item in enumerate(questions[:8], start=1))
    if gaps:
        lines.append("- 上一轮综合暴露的证据缺口：")
        lines.extend(f"  {index}. {item}" for index, item in enumerate(gaps[:8], start=1))
    if len(lines) == 1:
        lines.append("- 暂未发现已保存的检索问题。请先在 01 模块创建科学问题并完成一轮文献检索。")
    return "\n".join(lines)


def fallback_initial_response(context: dict[str, Any]) -> str:
    cards = context.get("paper_cards") or []
    wikis = context.get("wiki_pages") or []
    topics = ", ".join(str(item.get("title")) for item in wikis[:6] if item.get("title")) or "尚未形成稳定 topic"
    materials = _unique(item for card in cards for item in card.get("materials", []))[:6]
    methods = _unique(item for card in cards for item in card.get("methods", []))[:5]
    properties = _unique(item for card in cards for item in card.get("properties", []))[:5]
    gaps = context.get("evidence_gaps") or []
    questions = context.get("retrieval_questions") or []
    lines = [
        "根据上一轮文献分析，当前可以先把科学问题收敛到“材料体系 - 微结构/方法 - 性能指标 - 可验证机制”的形式。",
        "",
        f"核心证据主题：{topics}。",
    ]
    if materials or methods or properties:
        lines.append(f"已出现的材料/方法/性能线索：{', '.join(materials) or '待补充'}；{', '.join(methods) or '待补充'}；{', '.join(properties) or '待补充'}。")
    if gaps:
        lines.append("证据缺口显示需要优先澄清：")
        lines.extend(f"- {item}" for item in gaps[:4])
    if questions:
        lines.append("候选科学问题可以从这些检索问题继续压缩：")
        lines.extend(f"- {item}" for item in questions[:4])
    lines.extend(
        [
            "",
            "建议先选一个主方向：",
            "1. 机制型：解释某一微结构变量为何改变关键磁性能。",
            "2. 设计型：寻找材料成分和微结构约束下的性能上限。",
            "3. 方法型：建立计算/数据驱动流程来筛选候选体系。",
            "",
            "你可以直接告诉我更偏向机制、设计还是方法，我会继续把问题压成可执行的研究题目和验证路线。",
        ]
    )
    return "\n".join(lines)


def fallback_chat_response(context: dict[str, Any], latest_user_message: str) -> str:
    wikis = context.get("wiki_pages") or []
    open_questions = _unique(item for page in wikis for item in page.get("open_questions", []))[:4]
    evidence_ids = _unique(item for page in wikis for item in page.get("evidence_ids", []))[:6]
    lines = [
        f"我会按你的输入继续收敛：{latest_user_message}",
        "",
        "基于当前证据，建议把表述压成：",
        "“在给定材料体系中，哪些可控结构/成分变量通过何种机制影响目标性能，并如何用计算或实验验证？”",
    ]
    if open_questions:
        lines.append("")
        lines.append("可接上的开放问题：")
        lines.extend(f"- {item}" for item in open_questions)
    if evidence_ids:
        lines.append("")
        lines.append("当前可追溯证据：")
        lines.append(", ".join(evidence_ids))
    lines.append("")
    lines.append("下一步请确认目标性能、材料边界和验证手段中的一个，我再把它整理成 2-3 个精确研究方向。")
    return "\n".join(lines)


def finalize_response(content: str, metadata: dict[str, Any], context: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    answer, questions = split_suggested_questions(content)
    metadata["suggested_questions"] = questions[:3] if questions else fallback_suggested_questions(context)
    return answer, metadata


def split_suggested_questions(content: str) -> tuple[str, list[str]]:
    markers = ["后续可追问：", "后续可追问:", "建议继续追问：", "建议继续追问:"]
    marker_index = -1
    marker_text = ""
    for marker in markers:
        index = content.rfind(marker)
        if index > marker_index:
            marker_index = index
            marker_text = marker
    if marker_index < 0:
        return content.strip(), []
    answer = content[:marker_index].strip()
    tail = content[marker_index + len(marker_text) :].strip()
    questions: list[str] = []
    for raw_line in tail.splitlines():
        line = raw_line.strip().lstrip("-*").strip()
        if len(line) > 2 and line[0].isdigit() and line[1] in {".", "、", ")"}:
            line = line[2:].strip()
        if line.startswith("问题") and ":" in line[:5]:
            line = line.split(":", 1)[1].strip()
        if line.startswith("问题") and "：" in line[:5]:
            line = line.split("：", 1)[1].strip()
        if line:
            questions.append(line)
        if len(questions) == 3:
            break
    return answer or content.strip(), questions


def fallback_suggested_questions(context: dict[str, Any]) -> list[str]:
    wikis = context.get("wiki_pages") or []
    open_questions = _unique(item for page in wikis for item in page.get("open_questions", []))
    if len(open_questions) >= 3:
        return open_questions[:3]
    return (open_questions + [
        "这个问题最适合聚焦到哪一个材料体系？",
        "哪些局域环境或微结构变量最可能改变目标性能？",
        "用什么计算或实验指标验证这个机制？",
    ])[:3]


def _session_title(context: dict[str, Any]) -> str:
    goal = context.get("latest_goal") or {}
    return str(goal.get("title") or "科学问题归纳对话")


def _context_json(context: dict[str, Any]) -> str:
    payload = {
        "latest_goal": context.get("latest_goal"),
        "latest_round": context.get("latest_round"),
        "retrieval_questions": context.get("retrieval_questions", [])[:10],
        "evidence_gaps": context.get("evidence_gaps", [])[:10],
        "candidate_papers": context.get("candidate_papers", [])[:8],
        "paper_cards": context.get("paper_cards", [])[:8],
        "wiki_pages": context.get("wiki_pages", [])[:12],
        "metrics": context.get("metrics", {}),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _unique(items: Any) -> list[str]:
    values: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in values:
            values.append(text)
    return values


def _latest_human_message(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in reversed(messages):
        if item.get("role") == "user":
            return item
    return None


def collect_evidence_ids(context: dict[str, Any]) -> list[str]:
    """从 Paper Cards / Wiki 中收集全部可追溯证据 id，作为确认动作的证据池。"""
    ids: list[str] = []
    for card in context.get("paper_cards") or []:
        for item in list(card.get("evidence_ids") or []):
            if item and item not in ids:
                ids.append(item)
        for claim in list(card.get("claims") or []):
            for item in list(claim.get("evidence_ids") or []):
                if item and item not in ids:
                    ids.append(item)
    for page in context.get("wiki_pages") or []:
        for item in list(page.get("evidence_ids") or []):
            if item and item not in ids:
                ids.append(item)
        for finding in list(page.get("known_findings") or []):
            for item in list(finding.get("evidence_ids") or []):
                if item and item not in ids:
                    ids.append(item)
    return ids


def parse_confirm_json(content: str, context: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
    anchor = _latest_human_message(messages)
    data = _extract_json_object(content)
    if not data:
        raise ValueError(f"确认器未返回有效 JSON：{str(content)[:200]}")
    evidence_pool = collect_evidence_ids(context)
    return _fallback_confirm(context, anchor, evidence_pool) if not isinstance(data, dict) else {
        "problem_statement": str(data.get("problem_statement") or "").strip(),
        "variables": _string_list(data.get("variables")),
        "mechanism_hypothesis": str(data.get("mechanism_hypothesis") or "").strip(),
        "validation_criteria": _string_list(data.get("validation_criteria")),
        "evidence_ids": _filter_evidence_ids(data.get("evidence_ids"), evidence_pool),
    }


def _extract_json_object(content: str) -> dict[str, Any] | None:
    text = (content or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def _fallback_confirm(
    context: dict[str, Any],
    anchor: dict[str, Any] | None,
    evidence_pool: list[str],
) -> dict[str, Any]:
    cards = context.get("paper_cards") or []
    wikis = context.get("wiki_pages") or []
    materials = _unique(item for card in cards for item in list(card.get("materials") or []))[:5]
    properties = _unique(item for card in cards for item in list(card.get("properties") or []))[:5]
    open_questions = _unique(item for page in wikis for item in list(page.get("open_questions") or []))[:3]
    statement = str((anchor or {}).get("content") or "").strip()[:200] or (
        open_questions[0] if open_questions else "基于当前文献证据收敛出的核心科学问题"
    )
    return {
        "problem_statement": statement,
        "variables": materials or ["待确认的材料/结构变量"],
        "mechanism_hypothesis": "待验证：关键结构/成分变量通过尚不明确的机制影响目标性能。",
        "validation_criteria": (
            ["用计算或实验手段验证变量-机制-性能因果链"] + ([f"目标指标：{', '.join(properties)}"] if properties else [])
        ),
        "evidence_ids": evidence_pool[:8],
    }


def _filter_evidence_ids(value: Any, evidence_pool: list[str]) -> list[str]:
    if not isinstance(value, list):
        return []
    pool = {str(item).strip() for item in evidence_pool}
    filtered: list[str] = []
    for item in value:
        text = str(item).strip()
        if text in pool and text not in filtered:
            filtered.append(text)
    return filtered


def _recent_messages_text(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in list(messages)[-10:]:
        speaker = item.get("speaker") or item.get("role")
        lines.append(f"[{speaker}] {item.get('content')}")
    return "\n".join(lines)


def _confirm_summary(structured: dict[str, Any]) -> str:
    lines = [
        "已确认科学问题（结构化保存）：",
        f"- 问题陈述：{structured['problem_statement']}",
        f"- 关键变量：{', '.join(structured['variables']) or '待补充'}",
        f"- 机制假设：{structured['mechanism_hypothesis']}",
        f"- 验证判据：{'; '.join(structured['validation_criteria']) or '待补充'}",
        f"- 证据：{', '.join(structured['evidence_ids']) or '无可追溯证据，标记为待补充'}",
    ]
    return "\n".join(lines)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()][:8]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import (
    default_graph_path,
    default_output_path,
    default_question_synthesis_agent_dir,
    default_retrieval_agent_dir,
)
from .critic import (
    build_evolve_prompt,
    critique_routes,
    elo_arena,
    fallback_evolve,
    independent_evidence_search,
)
from .graph import (
    KnowledgeGraph,
    analogy_candidates,
    constraint_list,
    feasibility_map,
    gap_candidates,
    graph_context,
    graph_novelty,
    graph_stats,
    load_knowledge_graph,
)


SYSTEM_PROMPT = """你是 AI Scientist 工作流第 4 模块的“提出路线/候选 Agent”。
你的任务是把 01 文献检索提出的科学问题、03 科学问题归纳的方向，以及 02 文献分析得到的 Paper Cards / Wiki 证据转化成若干条可执行研究路线。
请始终用中文回答。不要编造文献、DOI、实验结果或已验证性能。路线必须包含：核心思路、候选材料/结构、关键变量、验证方式、证据依据、主要风险、优先级理由。
输出只能是严格 JSON，对象顶层必须包含 routes 数组。"""


_UNSET = object()


@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    api_key: str
    model: str

    @classmethod
    def from_env(cls) -> "LLMSettings | None":
        api_key = os.environ.get("OPENAI_API_KEY")
        model = os.environ.get("OPENAI_MODEL")
        if not api_key or not model:
            return None
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        return cls(base_url=base_url.rstrip("/"), api_key=api_key, model=model)


class OpenAICompatibleJsonClient:
    def __init__(self, settings: LLMSettings, *, timeout_seconds: int = 90):
        self.settings = settings
        self.timeout_seconds = timeout_seconds

    def route_candidates(self, prompt: str) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.settings.base_url}/chat/completions",
            data=json.dumps(
                {
                    "model": self.settings.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.2,
                    "response_format": {"type": "json_object"},
                },
                ensure_ascii=False,
            ).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "route-candidate-agent/0.1",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc
        content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        if isinstance(content, dict):
            return content
        try:
            return json.loads(str(content))
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned invalid JSON: {str(content)[:200]}") from exc


class RouteCandidateAgent:
    def __init__(
        self,
        *,
        output_path: Path | None = None,
        settings: LLMSettings | None | object = _UNSET,
        client_factory: Any = OpenAICompatibleJsonClient,
    ):
        self.output_path = output_path or default_output_path()
        self.settings = LLMSettings.from_env() if settings is _UNSET else settings
        self.client_factory = client_factory

    @property
    def model_name(self) -> str:
        if self.settings is None:
            return "LLM 未配置 - 本地规则草稿"
        return self.settings.model

    def state(self, *, selected_question_id: int | None = None) -> dict[str, Any]:
        context = collect_context()
        questions = context.get("question_options", [])
        selected = select_question(questions, selected_question_id)
        saved = self._load_saved()
        latest_run = saved.get("runs", [])[0] if saved.get("runs") else None
        selected_run = next(
            (
                item
                for item in saved.get("runs", [])
                if selected is not None and str(item.get("question_id")) == str(selected.get("id"))
            ),
            latest_run,
        )
        metrics = {
            "questions": len(questions),
            "routes": len((selected_run or {}).get("routes") or []),
            "saved_runs": len(saved.get("runs", [])),
            "evidence_gaps": len(context.get("evidence_gaps") or []),
            "paper_cards": len(context.get("paper_cards") or []),
            "wiki_pages": len(context.get("wiki_pages") or []),
        }
        return {
            "context": context,
            "questions": questions,
            "selected_question": selected,
            "latest_run": selected_run,
            "all_runs": saved.get("runs", []),
            "metrics": metrics,
            "model_name": self.model_name,
            "llm_configured": self.settings is not None,
            "output_path": str(self.output_path),
        }

    def generate(
        self,
        *,
        question_id: int | None = None,
        route_count: int = 3,
        emphasis: str = "",
        with_critique: bool = True,
    ) -> dict[str, Any]:
        route_count = max(1, min(int(route_count or 3), 8))
        context = collect_context()
        questions = context.get("question_options", [])
        selected = select_question(questions, question_id)
        if selected is None:
            raise ValueError("暂无可用科学问题。请先在 01 模块创建科学问题，或在 03 模块完成问题归纳。")
        graph = load_knowledge_graph()
        evidence_pool = collect_evidence_pool(context, selected)
        prompt = build_prompt(
            context=context,
            question=selected,
            route_count=route_count,
            emphasis=emphasis,
            evidence_pool=evidence_pool,
            graph=graph,
        )
        metadata: dict[str, Any] = {"mode": "fallback_no_api_key"}
        if self.settings is not None:
            try:
                payload = self.client_factory(self.settings).route_candidates(prompt)
                routes = normalize_routes(payload.get("routes"), route_count=route_count)
                metadata = {"mode": "llm"}
            except Exception as exc:
                routes = fallback_routes(context=context, question=selected, route_count=route_count, emphasis=emphasis)
                metadata = {"mode": "fallback_llm_error", "error": str(exc)}
        else:
            routes = fallback_routes(context=context, question=selected, route_count=route_count, emphasis=emphasis)
        routes = annotate_evidence(routes, evidence_pool)
        routes = annotate_graph_novelty(routes, graph)
        snapshot = {
            "question": selected,
            "evidence_pool": evidence_pool,
            "graph": _graph_prompt_payload(graph, selected),
            "emphasis": emphasis.strip(),
        }
        run = {
            "id": utc_now_iso().replace(":", "").replace("+", "Z"),
            "created_at": utc_now_iso(),
            "question_id": selected.get("id"),
            "question_title": selected.get("title"),
            "question_source": selected.get("source"),
            "route_count": route_count,
            "emphasis": emphasis.strip(),
            "context_fingerprint": context_fingerprint(context),
            "model": self.model_name,
            "metadata": metadata,
            "evidence_pool_size": len(evidence_pool),
            "graph": graph_stats(graph),
            "snapshot": snapshot,
            "routes": routes,
        }
        if with_critique:
            run = self._apply_critique(run, context)
        self._save_runs([run] + [item for item in self._load_saved().get("runs", []) if item.get("id") != run["id"]])
        return self.state(selected_question_id=int(selected["id"]))

    def _apply_critique(self, run: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
        """批判 + Elo Arena：独立证据检索 grounding（防幻觉传播），结果写回 routes 与 critique 块。

        评判者不读取生成器选择的证据池与图倾向性摘要（这些只存于 snapshot 供审计）；
        而是基于路线自身内容在 02 语料（Paper Cards / Wiki）上独立检索证据。
        """
        if not run.get("snapshot"):
            run["snapshot"] = _rebuild_snapshot(run)
        if context is None:
            context = collect_context()
        cards = list(context.get("paper_cards") or [])
        wikis = list(context.get("wiki_pages") or [])
        independent = {
            route.get("rank"): independent_evidence_search(route, cards, wikis)
            for route in run.get("routes") or []
        }
        critique = critique_routes(run, self.settings, independent)
        arena = elo_arena(run, critique, self.settings, independent)
        crit_by_rank = {item.get("rank"): item for item in critique.get("routes") or []}
        elo_rank_by_route = {
            item["rank"]: index + 1 for index, item in enumerate(arena.get("rankings") or [])
        }
        for route in run.get("routes") or []:
            item = crit_by_rank.get(route.get("rank")) or {}
            route["critique"] = {
                "dimensions": item.get("dimensions") or {},
                "total": item.get("total"),
                "weaknesses": item.get("weaknesses") or [],
                "improvements": item.get("improvements") or [],
            }
            route["elo_rating"] = (arena.get("ratings") or {}).get(str(route.get("rank")))
            route["elo_rank"] = elo_rank_by_route.get(route.get("rank"))
            search = independent.get(route.get("rank")) or {}
            route["independent_verification"] = {
                "queries": search.get("queries") or [],
                "hit_count": len(search.get("hits") or []),
                "hit_evidence_ids": search.get("hit_evidence_ids") or [],
            }
        run["critique"] = {
            "mode": critique.get("mode"),
            "error": critique.get("error"),
            "created_at": utc_now_iso(),
            "model": self.model_name if critique.get("mode") == "llm" else None,
            "arena": arena,
            "independent_search": {
                "corpus_size": len(cards) + len(wikis),
                "per_route": {
                    str(rank): {
                        "queries": (independent.get(rank) or {}).get("queries") or [],
                        "hit_count": len((independent.get(rank) or {}).get("hits") or []),
                    }
                    for rank in independent
                },
            },
        }
        return run

    def critique(self, *, run_id: str | None = None) -> dict[str, Any]:
        """对已有 run 重新做批判评审（适用于旧 run 或手动触发）。"""
        saved = self._load_saved()
        run = _find_run(saved, run_id)
        if run is None:
            raise ValueError("还没有生成路线，请先点击“生成候选路线”。")
        run = self._apply_critique(run)
        self._save_runs([run] + [item for item in saved.get("runs", []) if item.get("id") != run["id"]])
        return self.state(selected_question_id=run.get("question_id"))

    def evolve(self, *, run_id: str | None = None) -> dict[str, Any]:
        """一轮演化：基于批判意见生成 v2 路线，保留 lineage（parent → child）。"""
        saved = self._load_saved()
        run = _find_run(saved, run_id)
        if run is None:
            raise ValueError("还没有生成路线，请先点击“生成候选路线”。")
        if not run.get("snapshot"):
            run["snapshot"] = _rebuild_snapshot(run)
        if not run.get("critique"):
            run = self._apply_critique(run)
        snapshot = run.get("snapshot") or {}
        evidence_pool = list(snapshot.get("evidence_pool") or [])
        graph = load_knowledge_graph()
        metadata: dict[str, Any]
        if self.settings is not None:
            try:
                payload = self.client_factory(self.settings).route_candidates(build_evolve_prompt(run))
                routes = normalize_routes(payload.get("routes"), route_count=len(run.get("routes") or []))
                if not routes:
                    raise ValueError("演化输出无有效路线")
                metadata = {"mode": "evolve_llm"}
            except Exception as exc:
                routes = fallback_evolve(run)
                metadata = {"mode": "evolve_fallback_llm_error", "error": str(exc)[:300]}
        else:
            routes = fallback_evolve(run)
            metadata = {"mode": "evolve_fallback"}
        routes = annotate_evidence(routes, evidence_pool)
        routes = annotate_graph_novelty(routes, graph)
        new_run = {
            "id": utc_now_iso().replace(":", "").replace("+", "Z"),
            "created_at": utc_now_iso(),
            "question_id": run.get("question_id"),
            "question_title": run.get("question_title"),
            "question_source": run.get("question_source"),
            "route_count": len(routes),
            "emphasis": run.get("emphasis", ""),
            "context_fingerprint": run.get("context_fingerprint"),
            "model": self.model_name,
            "metadata": metadata,
            "evidence_pool_size": len(evidence_pool),
            "graph": graph_stats(graph),
            "snapshot": snapshot,
            "lineage": [
                {
                    "parent_run_id": run.get("id"),
                    "parent_created_at": run.get("created_at"),
                    "critique_mode": (run.get("critique") or {}).get("mode"),
                }
            ],
            "routes": routes,
        }
        new_run = self._apply_critique(new_run)
        self._save_runs([new_run] + [item for item in saved.get("runs", []) if item.get("id") != new_run["id"]])
        return self.state(selected_question_id=new_run.get("question_id"))

    def _save_runs(self, runs: list[dict[str, Any]]) -> None:
        saved = {"version": 1, "runs": runs[:20]}
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _load_saved(self) -> dict[str, Any]:
        if not self.output_path.exists():
            return {"version": 1, "runs": []}
        try:
            payload = json.loads(self.output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"version": 1, "runs": []}
        runs = payload.get("runs", [])
        if not isinstance(runs, list):
            runs = []
        return {"version": 1, "runs": [item for item in runs if isinstance(item, dict)]}


def _find_run(saved: dict[str, Any], run_id: str | None) -> dict[str, Any] | None:
    runs = list(saved.get("runs") or [])
    if run_id:
        return next((item for item in runs if item.get("id") == run_id), None)
    return runs[0] if runs else None


def _rebuild_snapshot(run: dict[str, Any]) -> dict[str, Any]:
    """旧 run 未保存快照时的重建：用当前上下文重建生成时近似快照（标记 rebuilt）。"""
    context = collect_context()
    selected = select_question(context.get("question_options", []), run.get("question_id")) or {}
    graph = load_knowledge_graph()
    return {
        "question": selected,
        "evidence_pool": collect_evidence_pool(context, selected),
        "graph": _graph_prompt_payload(graph, selected),
        "emphasis": str(run.get("emphasis") or ""),
        "rebuilt": True,
    }


def collect_context() -> dict[str, Any]:
    question_context = _question_synthesis_context()
    retrieval_context = _retrieval_question_options()
    confirmed_options = _confirmed_question_options()
    question_options = merge_question_options(retrieval_context, question_context, confirmed_options)
    selected = select_question(question_options, None)
    graph = load_knowledge_graph()
    context = {
        **question_context,
        "question_options": question_options,
        "confirmed_questions": confirmed_options,
        "graph": {
            "stats": graph_stats(graph),
            "context": graph_context(graph, selected or {}),
            "gap_candidates": gap_candidates(graph),
            "analogy_candidates": analogy_candidates(graph),
            "constraints": constraint_list(graph),
            "feasibility": feasibility_map(graph, selected or {}),
        },
        "paths": {
            "route_output": str(default_output_path()),
            "retrieval_db": str(default_retrieval_agent_dir() / "data" / "literature.sqlite"),
            "question_synthesis": str(default_question_synthesis_agent_dir()),
            "graph": str(default_graph_path()),
        },
    }
    return context


def _question_synthesis_context() -> dict[str, Any]:
    src_dir = default_question_synthesis_agent_dir() / "src"
    if not src_dir.exists():
        return _empty_context()
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    try:
        from question_synthesis_agent.context import collect_context as collect_question_context

        context = collect_question_context()
    except Exception:
        return _empty_context()
    return {
        "latest_goal": context.get("latest_goal"),
        "latest_round": context.get("latest_round"),
        "latest_synthesis": context.get("latest_synthesis"),
        "retrieval_questions": list(context.get("retrieval_questions") or []),
        "evidence_gaps": list(context.get("evidence_gaps") or []),
        "candidate_papers": list(context.get("candidate_papers") or []),
        "paper_cards": list(context.get("paper_cards") or []),
        "wiki_pages": list(context.get("wiki_pages") or []),
        "metrics": dict(context.get("metrics") or {}),
    }


def _empty_context() -> dict[str, Any]:
    return {
        "latest_goal": None,
        "latest_round": None,
        "latest_synthesis": None,
        "retrieval_questions": [],
        "evidence_gaps": [],
        "candidate_papers": [],
        "paper_cards": [],
        "wiki_pages": [],
        "metrics": {},
    }


def _retrieval_question_options() -> list[dict[str, Any]]:
    db_path = default_retrieval_agent_dir() / "data" / "literature.sqlite"
    if not db_path.exists():
        return []
    options: list[dict[str, Any]] = []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            for row in conn.execute("SELECT * FROM scientific_goals ORDER BY updated_at DESC, id DESC LIMIT 20"):
                options.append(
                    {
                        "id": int(row["id"]),
                        "source": "01 文献获取",
                        "title": str(row["title"] or "").strip(),
                        "description": str(row["description"] or "").strip(),
                        "goal_id": int(row["id"]),
                    }
                )
    except sqlite3.Error:
        return options
    return options


def _confirmed_question_options() -> list[dict[str, Any]]:
    """读取 03 模块确认过的结构化科学问题（question_synthesis.sqlite），
    作为 04 的问题来源之一，且优先级最高。"""
    db_path = default_question_synthesis_agent_dir() / "data" / "question_synthesis.sqlite"
    if not db_path.exists():
        return []
    options: list[dict[str, Any]] = []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            for row in conn.execute(
                "SELECT * FROM confirmed_questions ORDER BY created_at DESC, id DESC LIMIT 20"
            ):
                options.append(
                    {
                        "id": 20000 + int(row["id"]),
                        "source": "03 确认问题",
                        "title": str(row["problem_statement"] or "").strip(),
                        "description": str(row["mechanism_hypothesis"] or "").strip(),
                        "variables": _json_list(row["variables_json"]),
                        "validation_criteria": _json_list(row["validation_criteria_json"]),
                        "evidence_ids": _json_list(row["evidence_ids_json"]),
                        "confirmed_at": str(row["created_at"] or ""),
                        "mode": str(row["mode"] or ""),
                    }
                )
    except sqlite3.Error:
        return options
    return options


def _json_list(value: Any) -> list[str]:
    if not value:
        return []
    try:
        data = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [str(item).strip() for item in data if str(item).strip()]


def merge_question_options(
    retrieval_options: list[dict[str, Any]],
    context: dict[str, Any],
    confirmed_options: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    seen: set[str] = set()
    # 03 模块确认的结构化问题优先级最高
    for item in confirmed_options or []:
        key = normalize_key(item.get("title"))
        if key and key not in seen:
            seen.add(key)
            options.append(item)
    for item in retrieval_options:
        key = normalize_key(item.get("title"))
        if key and key not in seen:
            seen.add(key)
            options.append(item)
    start = 10000
    for question in context.get("retrieval_questions") or []:
        text = str(question).strip()
        key = normalize_key(text)
        if key and key not in seen:
            seen.add(key)
            options.append({"id": start + len(options), "source": "03 科学问题归纳", "title": text, "description": ""})
    for page in context.get("wiki_pages") or []:
        for question in list(page.get("open_questions") or [])[:2]:
            text = str(question).strip()
            key = normalize_key(text)
            if key and key not in seen:
                seen.add(key)
                options.append(
                    {
                        "id": start + len(options),
                        "source": "02/03 开放问题",
                        "title": text,
                        "description": str(page.get("title") or ""),
                    }
                )
    return options[:24]


def select_question(options: list[dict[str, Any]], question_id: int | None) -> dict[str, Any] | None:
    if not options:
        return None
    if question_id is not None:
        for item in options:
            if int(item.get("id") or -1) == int(question_id):
                return item
    return options[0]


def build_prompt(
    *,
    context: dict[str, Any],
    question: dict[str, Any],
    route_count: int,
    emphasis: str,
    evidence_pool: list[str] | None = None,
    graph: KnowledgeGraph | None = None,
) -> str:
    graph_payload = _graph_prompt_payload(graph, question)
    payload = {
        "selected_question": question,
        "route_count": route_count,
        "researcher_emphasis": emphasis,
        "latest_goal": context.get("latest_goal"),
        "retrieval_questions": context.get("retrieval_questions", [])[:10],
        "evidence_gaps": context.get("evidence_gaps", [])[:10],
        "candidate_papers": context.get("candidate_papers", [])[:8],
        "paper_cards": context.get("paper_cards", [])[:8],
        "wiki_pages": context.get("wiki_pages", [])[:12],
        "evidence_pool": (evidence_pool or [])[:40],
        "knowledge_graph": graph_payload,
    }
    if question.get("source") == "03 确认问题":
        payload["confirmed_question"] = {
            "variables": question.get("variables"),
            "validation_criteria": question.get("validation_criteria"),
            "evidence_ids": question.get("evidence_ids"),
        }
    return (
        f"请为所选科学问题生成 {route_count} 条互相区分的可行研究路线。\n"
        "每条路线 JSON 字段必须为：title, rationale, candidates, variables, validation, evidence, risks, priority, next_step。\n"
        "candidates、variables、validation、evidence、risks 必须是字符串数组；priority 是 高/中/低之一。\n"
        "evidence 字段中尽量直接引用 evidence_pool 里的 evidence id，无法引用时明确写成推测。\n"
        "knowledge_graph 中的 gap_candidates（方法尚未用于某材料的组合）是候选新颖方向的线索，可以优先考虑；"
        "constraints 是已知限制，路线设计应避免与限制冲突。\n"
        "上下文 JSON：\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def _graph_prompt_payload(graph: KnowledgeGraph | None, question: dict[str, Any]) -> dict[str, Any]:
    if graph is None or graph.is_empty():
        return {"loaded": False}
    graph_ctx = graph_context(graph, question)
    gaps = gap_candidates(graph)
    analogies = analogy_candidates(graph)
    constraints = constraint_list(graph)
    return {
        "loaded": True,
        "matched_entities": graph_ctx.get("matched_entities", []),
        "related_nodes": [
            {"id": item.get("id"), "type": item.get("type"), "label": item.get("label")}
            for item in graph_ctx.get("nodes", [])[:24]
        ],
        "related_evidence_ids": graph_ctx.get("evidence_ids", [])[:20],
        "gap_candidates": gaps[:8],
        "analogy_candidates": analogies[:6],
        "constraints": [item.get("limitation") for item in constraints[:8]],
    }


def normalize_routes(value: Any, *, route_count: int) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    routes = []
    for index, item in enumerate(value[:route_count], start=1):
        if not isinstance(item, dict):
            continue
        routes.append(
            {
                "rank": index,
                "title": str(item.get("title") or f"路线 {index}").strip(),
                "rationale": str(item.get("rationale") or "").strip(),
                "candidates": string_list(item.get("candidates"))[:8],
                "variables": string_list(item.get("variables"))[:8],
                "validation": string_list(item.get("validation"))[:8],
                "evidence": string_list(item.get("evidence"))[:8],
                "risks": string_list(item.get("risks"))[:8],
                "priority": normalize_priority(item.get("priority"), index),
                "next_step": str(item.get("next_step") or "").strip(),
            }
        )
    return routes


def fallback_routes(
    *,
    context: dict[str, Any],
    question: dict[str, Any],
    route_count: int,
    emphasis: str,
) -> list[dict[str, Any]]:
    cards = list(context.get("paper_cards") or [])
    wikis = list(context.get("wiki_pages") or [])
    materials = unique(item for card in cards for item in list(card.get("materials") or []))[:8]
    methods = unique(item for card in cards for item in list(card.get("methods") or []))[:8]
    properties = unique(item for card in cards for item in list(card.get("properties") or []))[:8]
    gaps = unique(context.get("evidence_gaps") or [])[:6]
    open_questions = unique(item for page in wikis for item in list(page.get("open_questions") or []))[:6]
    evidence_ids = unique(
        item
        for page in wikis
        for finding in list(page.get("known_findings") or [])
        for item in list(finding.get("evidence_ids") or [])
    )[:8]
    if not evidence_ids:
        evidence_ids = unique(item for card in cards for item in list(card.get("evidence_ids") or []))[:8]
    material_text = ", ".join(materials[:4]) or "当前文献中反复出现的候选材料体系"
    method_text = ", ".join(methods[:4]) or "第一性原理、微磁模拟、结构表征或小样本实验"
    property_text = ", ".join(properties[:4]) or "矫顽力、剩磁、磁各向异性、能量积等目标指标"
    route_templates = [
        {
            "title": "机制优先路线：锁定性能退化的关键变量",
            "rationale": f"围绕“{question.get('title')}”先解释变量到性能的因果链，避免直接扩展过大的候选空间。",
            "candidates": materials[:5] or [material_text],
            "variables": ["晶粒尺寸/取向", "晶界相磁性与厚度", "缺陷、孪晶或反相边界", "成分替换比例"],
            "validation": [method_text, f"把输出指标固定为 {property_text}", "对照有/无关键变量的计算或实验结果"],
            "evidence": (evidence_ids[:5] or open_questions[:3] or gaps[:3] or ["当前证据不足，需要先补充可追溯 evidence ids"]),
            "risks": gaps[:3] or ["现有证据可能只支持相关性，尚不足以证明机制"],
            "priority": "高",
            "next_step": "把问题改写成一个明确的变量-机制-性能假设，并列出最小验证样本。",
        },
        {
            "title": "候选体系路线：成分/结构空间筛选",
            "rationale": "把 02/03 中出现的材料、方法和开放问题转为有限候选集合，用统一指标做第一轮排序。",
            "candidates": materials[:6] or ["Fe-rich 稀土减量体系", "L10 FeNi / MnAl / ThMn12 类候选体系"],
            "variables": ["主相稳定性", "磁晶各向异性", "饱和磁化强度", "交换刚度", "热稳定性"],
            "validation": ["高通量 DFT 或数据库筛选", "微磁模拟估计上限", "与文献 Paper Cards 中的基准值对比"],
            "evidence": evidence_ids[:5] or ["来自 Paper Cards / Wiki 的材料和性能线索"],
            "risks": ["筛选命中不等于可合成", "0 K 计算和室温性能之间可能存在偏差"],
            "priority": "中",
            "next_step": "确定候选体系白名单和必须满足的硬约束，再生成计算/实验矩阵。",
        },
        {
            "title": "工艺-微结构路线：从可制备性反推路线",
            "rationale": "优先选择能被实验工艺控制的变量，让路线可以进入后续计算/实验规划。",
            "candidates": materials[:4] or [material_text],
            "variables": ["退火温度/时间", "扩散源或添加元素", "晶界连续性", "织构与致密度"],
            "validation": ["小矩阵工艺实验", "EBSD/TEM 等微结构表征", f"用 {property_text} 做性能闭环"],
            "evidence": evidence_ids[:4] or open_questions[:4] or ["证据不足，需要从已下载 PDF 中补充工艺-性能关系"],
            "risks": ["工艺变量耦合较强，单因素解释可能不稳", "样品制备成本高于纯计算路线"],
            "priority": "中",
            "next_step": "选择 2-3 个最容易控制的工艺变量，设计最小正交矩阵。",
        },
        {
            "title": "数据驱动路线：构建小样本可解释排序器",
            "rationale": "当文献证据分散时，先把候选材料、变量和性能指标结构化，生成可迭代的路线评分表。",
            "candidates": materials[:6] or ["已抽取 Paper Cards 中的候选体系"],
            "variables": ["材料标签", "结构描述符", "工艺标签", "性能指标", "证据等级"],
            "validation": ["用已知文献记录做留一验证", "人工复核高分路线", "把不确定项转为下一轮检索 query"],
            "evidence": evidence_ids[:5] or ["Paper Cards、Wiki open_questions 和 evidence_gaps"],
            "risks": ["样本量不足会导致评分不稳定", "缺失数据需要显式标注，不能用模型补假数据"],
            "priority": "低",
            "next_step": "定义路线评分字段，并把当前候选路线导出为可人工修订的表格。",
        },
    ]
    if emphasis.strip():
        for item in route_templates:
            item["rationale"] += f" 研究者偏好：{emphasis.strip()}。"
    return [{**item, "rank": index} for index, item in enumerate(route_templates[:route_count], start=1)]


def context_fingerprint(context: dict[str, Any]) -> str:
    slim = {
        "questions": context.get("question_options", [])[:10],
        "gaps": context.get("evidence_gaps", [])[:10],
        "cards": [
            {"title": item.get("title"), "summary": item.get("summary")}
            for item in context.get("paper_cards", [])[:8]
        ],
        "wiki": [
            {"title": item.get("title"), "open_questions": item.get("open_questions", [])[:3]}
            for item in context.get("wiki_pages", [])[:10]
        ],
    }
    payload = json.dumps(slim, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    return [str(value).strip()] if str(value).strip() else []


def collect_evidence_pool(context: dict[str, Any], question: dict[str, Any]) -> list[str]:
    """收集可追溯证据 id 池：确认问题自带的证据优先，再补 Paper Cards / Wiki 的证据。"""
    ids: list[str] = []
    for item in list(question.get("evidence_ids") or []):
        text = str(item).strip()
        if text and text not in ids:
            ids.append(text)
    for card in context.get("paper_cards") or []:
        for item in list(card.get("evidence_ids") or []):
            text = str(item).strip()
            if text and text not in ids:
                ids.append(text)
    for page in context.get("wiki_pages") or []:
        for item in list(page.get("evidence_ids") or []):
            text = str(item).strip()
            if text and text not in ids:
                ids.append(text)
        for finding in list(page.get("known_findings") or []):
            for item in list(finding.get("evidence_ids") or []):
                text = str(item).strip()
                if text and text not in ids:
                    ids.append(text)
    return ids


def annotate_evidence(routes: list[dict[str, Any]], evidence_pool: list[str]) -> list[dict[str, Any]]:
    """对每条路线的 evidence 文本做确定性标注：引用了真实证据 id → 证据支撑；否则 → 推测。"""
    pool = [text for text in evidence_pool if text]
    for route in routes:
        annotations = []
        for text in route.get("evidence") or []:
            matched = [eid for eid in pool if eid in text]
            annotations.append(
                {
                    "text": text,
                    "kind": "证据支撑" if matched else "推测",
                    "matched_ids": matched,
                }
            )
        route["evidence_annotations"] = annotations
    return routes


def annotate_graph_novelty(routes: list[dict[str, Any]], graph: KnowledgeGraph | None) -> list[dict[str, Any]]:
    """对每条路线附加确定性图新颖性评分（高/中/低 + 理由）。"""
    for route in routes:
        route["graph_novelty"] = graph_novelty(graph, route)
    return routes


def normalize_priority(value: Any, index: int) -> str:
    text = str(value or "").strip()
    if text in {"高", "中", "低"}:
        return text
    return "高" if index == 1 else "中" if index <= 3 else "低"


def unique(items: Any) -> list[str]:
    values: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in values:
            values.append(text)
    return values


def normalize_key(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

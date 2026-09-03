"""P2：路线批判者（Critic）+ Elo Arena + 一轮演化。

关键设计：Critic 的评判依据是生成路线时使用的**上下文快照**（run["snapshot"]），
而不是事后重新收集的上下文。快照包含：
- question：所选科学问题的完整结构化字段（含 03 确认问题的 variables / validation_criteria / evidence_ids）
- evidence_pool：生成时可引用的真实证据 id 池
- graph：注入生成提示词的图谱摘要（匹配实体 / 邻域 / gap / 类比 / 限制）

所有函数都不依赖 LLM：无 LLM 时走确定性兜底（fallback_critique / 确定性 Elo），
保证流程在任何环境下都能产出可审计的评判结果。
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

CRITIC_SYSTEM_PROMPT = """你是 AI Scientist 工作流第 4 模块的"路线批判者（Critic）"。
你的任务是对候选研究路线做严格、可审计的批判性评审。

铁律：
1. 评判的唯一证据依据是"独立检索证据"块中给出的证据（带真实 evidence id 与来源）。
2. 路线里的 rationale、evidence 等字段是**生成器的声称，未经验证**：只有在独立检索证据能支持它时才能采信；否则必须在评判中明确指出"无证据支撑"。
3. 若生成器声称引用的证据 id 未出现在独立检索证据块中，视为未验证，不得采信。
4. 禁止凭借"看起来合理"给高分；禁止脑补文献结论。
5. 对每条路线输出四维评分（1-5 整数）：新颖性 novelty / 可行性 feasibility / 证据充分性 evidence / 可证伪性 falsifiability。
6. 输出严格 JSON，顶层必须是 {"routes": [...]}，routes 元素字段：rank, dimensions, weaknesses, improvements。
   dimensions 为 {"novelty": {"score": n, "reason": "..."}, "feasibility": {...}, "evidence": {...}, "falsifiability": {...}}。
   weaknesses / improvements 为字符串数组（各 2-4 条）。"""

DIMENSION_LABELS = {
    "novelty": "新颖性",
    "feasibility": "可行性",
    "evidence": "证据充分性",
    "falsifiability": "可证伪性",
}


def _message_text(raw: dict[str, Any]) -> Any:
    """提取回复文本；兼容推理模型（content 为空时回退 reasoning_content）。"""
    message = raw.get("choices", [{}])[0].get("message", {}) or {}
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(str(item.get("text") or item) for item in content)
    if isinstance(content, str) and content.strip():
        return content
    fallback = message.get("reasoning_content")
    if isinstance(fallback, str) and fallback.strip():
        return fallback
    return content if content is not None else ""


def _loads_json(text: Any) -> Any:
    """严格解析失败时，从文本（代码块/推理内容）中提取 JSON 对象。"""
    if isinstance(text, (dict, list)):
        return text
    source = str(text)
    try:
        return json.loads(source)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    candidates: list[Any] = []
    for match in re.finditer(r"\{", source):
        try:
            obj, _ = decoder.raw_decode(source, match.start())
        except json.JSONDecodeError:
            continue
        candidates.append(obj)
    if candidates:
        return candidates[-1]
    raise ValueError("no JSON object found")


def _llm_json(settings: Any, *, system_prompt: str, user_prompt: str, timeout_seconds: int = 120) -> dict[str, Any]:
    """与 OpenAICompatibleJsonClient 同协议的通用 JSON 调用（duck typing settings）。"""
    request = urllib.request.Request(
        f"{settings.base_url}/chat/completions",
        data=json.dumps(
            {
                "model": settings.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
            },
            ensure_ascii=False,
        ).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "route-critic-agent/0.1",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc
    content = _message_text(raw)
    if isinstance(content, dict):
        return content
    try:
        return _loads_json(content)
    except ValueError as exc:
        raise ValueError(f"LLM returned invalid JSON: {str(content)[:200]}") from exc


def _route_text(route: dict[str, Any], *, with_critique: bool = False) -> str:
    """把一条路线压缩为可读文本（含证据标注与图新颖性）。"""
    lines = [
        f"[rank={route.get('rank')}] {route.get('title') or ''}",
        f"思路: {route.get('rationale') or ''}",
        f"候选: {'、'.join(route.get('candidates') or [])}",
        f"变量: {'、'.join(route.get('variables') or [])}",
        f"验证: {'、'.join(route.get('validation') or [])}",
        f"风险: {'、'.join(route.get('risks') or [])}",
    ]
    for annotation in route.get("evidence_annotations") or []:
        ids = ", ".join(annotation.get("matched_ids") or [])
        mark = f" [引用: {ids}]" if ids else " [推测]"
        lines.append(f"证据({annotation.get('kind')}): {annotation.get('text')}{mark}")
    novelty = route.get("graph_novelty") or {}
    if novelty:
        lines.append(f"图新颖性: {novelty.get('level')} - {novelty.get('reason') or ''}")
    if with_critique:
        critique = route.get("critique") or {}
        for key in ("weaknesses", "improvements"):
            items = critique.get(key) or []
            if items:
                lines.append(f"{key}: {'；'.join(str(i) for i in items)}")
    return "\n".join(lines)


def _snapshot_text(run: dict[str, Any]) -> str:
    """生成上下文快照文本：仅供演化（生成器侧）与审计使用，不注入批判者提示词。"""
    snapshot = run.get("snapshot") or {}
    question = snapshot.get("question") or {}
    graph = snapshot.get("graph") or {}
    parts: list[str] = []
    parts.append(
        "【研究问题（生成时使用）】\n"
        + json.dumps(
            {
                "id": question.get("id"),
                "title": question.get("title"),
                "source": question.get("source"),
                "description": question.get("description"),
                "variables": question.get("variables"),
                "validation_criteria": question.get("validation_criteria"),
                "mechanism_hypothesis": question.get("mechanism_hypothesis"),
                "evidence_ids": question.get("evidence_ids"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    evidence_pool = list(snapshot.get("evidence_pool") or [])
    parts.append(f"【证据池（{len(evidence_pool)} 条可引用证据 id）】\n" + ", ".join(evidence_pool))
    parts.append("【知识图谱摘要（生成时注入）】\n" + json.dumps(graph, ensure_ascii=False, indent=2))
    if snapshot.get("rebuilt"):
        parts.append("注意：本快照为历史 run 重建（原 run 未保存快照），内容可能不完全等于生成时上下文。")
    if snapshot.get("emphasis"):
        parts.append(f"【研究者偏好】{snapshot.get('emphasis')}")
    return "\n\n".join(parts)


_LATIN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-\u2080-\u2089]{2,}")
_CHUNK_SPLIT_RE = re.compile(r'[，。、；：:（）()\[\]“”"\'\s,;\n\t]+')
_LATIN_STOPWORDS = {"the", "and", "for", "with", "not"}


def _item_atoms(text: str) -> tuple[list[str], list[str]]:
    """把一条查询文本拆为检索原子：(英文/材料 token, 中文 n-gram)。"""
    latin: list[str] = []
    for match in _LATIN_TOKEN_RE.finditer(text):
        token = match.group(0)
        if token.lower() not in _LATIN_STOPWORDS and token not in latin:
            latin.append(token)
    grams: list[str] = []
    for chunk in _CHUNK_SPLIT_RE.split(text):
        chunk = chunk.strip()
        if len(chunk) == 2:
            grams.append(chunk)
        elif len(chunk) >= 3:
            grams.extend(chunk[i : i + 3] for i in range(len(chunk) - 2))
    return latin, grams


def _item_matches(text: str, latin: list[str], grams: list[str]) -> bool:
    """一条查询是否命中文档：英文 token 任一命中（忽略连字符差异），
    或中文 n-gram 命中 ≥ 2。"""
    normalized = text.replace("-", "").replace("–", "").replace("—", "")
    if any(token.replace("-", "").replace("–", "").replace("—", "") in normalized for token in latin):
        return True
    hits = sum(1 for gram in grams if gram in text)
    return hits >= 2


def _route_query_terms(route: dict[str, Any]) -> list[str]:
    """从路线自身内容构建独立检索词（不使用生成器的证据池或图摘要）。"""
    terms: list[str] = []
    for field in ("candidates", "variables"):
        for item in route.get(field) or []:
            text = str(item).strip()
            if len(text) >= 2 and text not in terms:
                terms.append(text)
    title = str(route.get("title") or "").strip()
    if len(title) >= 4 and title not in terms:
        terms.append(title[:24])
    # 过滤纯停用词/纯标点
    filtered = []
    for term in terms:
        core = term.strip(" ，。、：；（）()[]“”'\"\n\t")
        if len(core) >= 2:
            filtered.append(term)
    return filtered[:12]


def _doc_snippet(text: str, query: str, *, width: int = 90) -> str:
    index = text.find(query)
    if index < 0:
        return text[:width]
    start = max(0, index - width // 3)
    return "…" + text[start : start + width].replace("\n", " ") + "…"


def independent_evidence_search(route: dict[str, Any], cards: list[Any], wikis: list[Any]) -> dict[str, Any]:
    """独立证据检索：基于路线自身内容（非生成器选择的证据池）在 02 语料中检索。

    返回命中证据及其来源，供批判者 grounding。这是批判者唯一的证据依据，
    防止生成器的幻觉（幻觉引用/误读）被评判环节吸收复用。
    """
    queries = _route_query_terms(route)
    docs: list[dict[str, Any]] = []
    for card in cards or []:
        if not isinstance(card, dict):
            continue
        findings_text = " ".join(str(finding) for finding in card.get("findings") or [])
        docs.append(
            {
                "source": "paper_card",
                "title": str(card.get("title") or card.get("id") or ""),
                "evidence_ids": [str(item).strip() for item in card.get("evidence_ids") or []],
                "text": " ".join(
                    str(part)
                    for part in (card.get("summary"), findings_text, card.get("concepts"))
                    if part
                ),
            }
        )
    for page in wikis or []:
        if not isinstance(page, dict):
            continue
        findings_text = " ".join(str(finding) for finding in page.get("known_findings") or [])
        docs.append(
            {
                "source": "wiki",
                "title": str(page.get("title") or page.get("id") or ""),
                "evidence_ids": [str(item).strip() for item in page.get("evidence_ids") or []],
                "text": " ".join(str(part) for part in (page.get("summary"), findings_text) if part),
            }
        )
    hits: list[dict[str, Any]] = []
    atoms = [_item_atoms(query) for query in queries]
    for doc in docs:
        matched = [
            query
            for query, (latin, grams) in zip(queries, atoms)
            if _item_matches(doc["text"], latin, grams)
        ]
        if not matched:
            continue
        ids = [item for item in doc["evidence_ids"] if item]
        hits.append(
            {
                "evidence_ids": ids,
                "source": doc["source"],
                "title": doc["title"],
                "snippet": _doc_snippet(doc["text"], matched[0]),
                "matched_queries": matched[:4],
                "match_count": len(matched),
            }
        )
    hits.sort(key=lambda item: item["match_count"], reverse=True)
    hit_ids: list[str] = []
    for hit in hits:
        for item in hit["evidence_ids"]:
            if item not in hit_ids:
                hit_ids.append(item)
    return {"queries": queries, "hits": hits[:12], "hit_evidence_ids": hit_ids}


def _independent_text(independent: dict[str, Any] | None) -> str:
    """把独立检索结果渲染成批判者的证据依据块。"""
    if not independent:
        return "（无独立检索结果：02 语料未加载或检索未命中）"
    queries = independent.get("queries") or []
    hits = independent.get("hits") or []
    lines = [f"检索词：{'、'.join(queries)}", f"命中 {len(hits)} 条证据来源："]
    for hit in hits:
        ids = ", ".join(hit.get("evidence_ids") or []) or "（无 evidence id）"
        lines.append(
            f"- [{hit.get('source')}] {hit.get('title')} | evidence: {ids}\n  命中词：{'、'.join(hit.get('matched_queries') or [])}\n  片段：{hit.get('snippet')}"
        )
    return "\n".join(lines)


def _question_text(run: dict[str, Any]) -> str:
    """评判对象：仅问题标题与描述（不给 03 模块的预设变量/机制假设/证据）。"""
    question = (run.get("snapshot") or {}).get("question") or {}
    title = str(question.get("title") or run.get("question_title") or "")
    description = str(question.get("description") or "")
    return f"研究问题：{title}\n问题描述：{description or '（无）'}"


def build_critique_prompt(run: dict[str, Any], independent: dict[int, Any] | None = None) -> str:
    """批判提示词：独立 grounding，不注入生成器的证据池 / 图倾向性摘要 / 机制假设。

    输入 = 研究问题（评判对象）+ 路线内容（声称标注）+ 独立检索证据（唯一证据依据）+ 客观已知限制。
    生成上下文快照只用于审计，不进入本提示词。
    """
    snapshot = run.get("snapshot") or {}
    graph = snapshot.get("graph") or {}
    constraints = [
        str(item.get("limitation") or "").strip()
        for item in graph.get("constraints") or []
        if isinstance(item, dict)
    ]
    sections = ["【研究问题（评判对象）】\n" + _question_text(run)]
    if constraints:
        sections.append("【客观已知限制（来自 02 语料，非生成器产出）】\n" + "\n".join(f"- {c}" for c in constraints[:8]))
    by_rank = independent or {}
    route_texts = []
    for route in run.get("routes") or []:
        rank = route.get("rank")
        lines = [
            f"[rank={rank}] {route.get('title') or ''}",
            f"生成器思路（声称，待检验）: {route.get('rationale') or ''}",
            f"候选: {'、'.join(route.get('candidates') or [])}",
            f"变量: {'、'.join(route.get('variables') or [])}",
            f"验证: {'、'.join(route.get('validation') or [])}",
            f"风险: {'、'.join(route.get('risks') or [])}",
        ]
        claimed = []
        for annotation in route.get("evidence_annotations") or []:
            ids = ", ".join(annotation.get("matched_ids") or [])
            mark = f" [声称引用: {ids}]" if ids else " [声称推测]"
            claimed.append(f"- {annotation.get('text')}{mark}")
        if claimed:
            lines.append("生成器声称的证据（未经验证，须与独立检索证据对照）:\n" + "\n".join(claimed))
        lines.append("独立检索证据（唯一证据依据）:\n" + _independent_text(by_rank.get(rank)))
        route_texts.append("\n".join(lines))
    sections.append("【候选路线与独立证据】\n\n" + "\n\n".join(route_texts))
    sections.append(
        "请基于独立检索证据逐条评判。生成器声称的证据只有与独立检索证据一致时才能算作证据支撑；"
        "声称引用但独立检索未命中的 id 一律视为未验证。"
    )
    return "\n\n".join(sections)


def _constraint_hit(constraint: str, text: str) -> bool:
    """已知限制是否命中路线的候选/变量文本（支持英文分词与中文短语）。"""
    if not constraint:
        return False
    if constraint in text:
        return True
    if " " in constraint:
        words = [w for w in constraint.split() if len(w) >= 2][:6]
        if words:
            return any(w in text for w in words)
        return False
    # 中文短语：检查路线候选/变量里的词是否出现在限制描述里
    for token in text.split():
        if len(token) >= 2 and token in constraint:
            return True
    return False


def _term_hit(term: str, text: str) -> bool:
    """语料未覆盖实体是否与路线候选/变量相关（短词直接匹配，长串取关键词）。"""
    if not term:
        return False
    if len(term) <= 12:
        return term in text
    words = [w for w in term.split() if len(w) >= 2][:5]
    return bool(words) and any(w in text for w in words)


def fallback_critique(run: dict[str, Any], independent: dict[int, Any] | None = None) -> dict[str, Any]:
    """确定性四维评分（无 LLM 或 LLM 失败时兜底）。

    证据分除声称标注外，还做"声称 vs 独立检索"交叉验证：生成器声称引用的 id
    若无法通过独立检索复现，视为未验证，证据分降级并记录弱点（防幻觉传播）。
    """
    snapshot = run.get("snapshot") or {}
    graph = snapshot.get("graph") or {}
    constraints = [
        str(item.get("limitation") or "").strip()
        for item in graph.get("constraints") or []
        if isinstance(item, dict)
    ]
    unknown_entities = [
        str(item).strip()
        for item in (graph.get("feasibility") or {}).get("unknown") or []
    ]
    by_rank = independent or {}
    results: list[dict[str, Any]] = []
    for route in run.get("routes") or []:
        annotations = list(route.get("evidence_annotations") or [])
        supported = sum(1 for a in annotations if a.get("kind") == "证据支撑")
        matched_total = sum(len(a.get("matched_ids") or []) for a in annotations)
        claimed_ids: list[str] = []
        for a in annotations:
            for item in a.get("matched_ids") or []:
                if item not in claimed_ids:
                    claimed_ids.append(item)
        hit_ids = (by_rank.get(route.get("rank")) or {}).get("hit_evidence_ids") or []
        verified_ids = [item for item in claimed_ids if item in hit_ids]
        unverified_count = len(claimed_ids) - len(verified_ids)
        if not annotations:
            evidence_score, evidence_reason = 1, "路线没有证据依据条目，全部内容无法追溯"
        elif supported == len(annotations) and matched_total >= 3 and unverified_count == 0:
            evidence_score, evidence_reason = 5, f"全部 {len(annotations)} 条证据引用真实 id（共 {matched_total} 处），且均被独立检索复现"
        elif supported == len(annotations) and unverified_count == 0:
            evidence_score, evidence_reason = 4, f"全部 {len(annotations)} 条证据有引用且被独立检索复现，但 id 数量较少（{matched_total} 处）"
        elif supported:
            evidence_score, evidence_reason = 3, f"{supported}/{len(annotations)} 条有证据支撑，其余为推测"
        else:
            evidence_score, evidence_reason = 1, "全部证据均为推测，无真实证据 id 引用"
        if unverified_count > 0:
            evidence_score = min(evidence_score, 2)
            evidence_reason += f"；{unverified_count}/{len(claimed_ids)} 条声称引用未能被独立检索复现（幻觉风险）"

        validation = list(route.get("validation") or [])
        quant_marks = ("%", "倍", "阈值", "提升", "降低", "对比", "±", "定量", "指标", "基准", "判据", "误差")
        quantitative = any(mark in str(text) for text in validation for mark in quant_marks)
        if validation and quantitative:
            falsifiability_score, falsifiability_reason = 5, "验证方式含定量/对比判据，可判定假设成立与否"
        elif validation:
            falsifiability_score, falsifiability_reason = 3, "有验证方式但缺少定量判据，证伪强度有限"
        else:
            falsifiability_score, falsifiability_reason = 1, "没有验证方式，假设不可证伪"

        novelty = route.get("graph_novelty") or {}
        novelty_level = str(novelty.get("level") or "")
        if novelty_level == "高":
            novelty_score, novelty_reason = 5, f"图中未发现该组合被探索：{novelty.get('reason') or ''}"
        elif novelty_level == "中":
            novelty_score, novelty_reason = 3, f"组合仅有部分共现证据：{novelty.get('reason') or ''}"
        elif novelty_level == "低":
            novelty_score, novelty_reason = 2, f"组合已在图中被直接探索过：{novelty.get('reason') or ''}"
        else:
            novelty_score, novelty_reason = 3, "图新颖性未知（图谱未加载或无匹配实体）"

        candidates = [str(item).strip() for item in route.get("candidates") or []]
        variables = [str(item).strip() for item in route.get("variables") or []]
        route_terms_text = " ".join(candidates + variables)
        hit_constraints = [c for c in constraints if _constraint_hit(c, route_terms_text)]
        hit_unknown = [u for u in unknown_entities if u and _term_hit(u, route_terms_text)]
        if hit_constraints:
            feasibility_score, feasibility_reason = 2, f"候选与已知限制冲突：{'；'.join(hit_constraints[:2])}"
        elif hit_unknown:
            feasibility_score, feasibility_reason = 3, f"候选实体语料未覆盖：{'、'.join(hit_unknown[:3])}，可行性无法评估"
        else:
            feasibility_score, feasibility_reason = 4, "候选未触发已知限制，语料覆盖范围内可行"

        scores = [novelty_score, feasibility_score, evidence_score, falsifiability_score]
        weaknesses: list[str] = []
        improvements: list[str] = []
        if unverified_count > 0:
            weaknesses.append(
                f"声称引用的证据中 {unverified_count}/{len(claimed_ids)} 条未能被独立检索复现，存在幻觉引用风险"
            )
            improvements.append("删除无法独立验证的引用，回到 01/02 模块重新检索该主张的证据")
        if evidence_score <= 2:
            weaknesses.append("证据链薄弱：路线关键主张缺少可独立复现的证据")
            improvements.append("把推测性主张拆分为可检索的子命题，回到 01/02 模块补充证据")
        if falsifiability_score <= 2:
            weaknesses.append("验证判据不可证伪或缺失，无法判定路线成败")
            improvements.append("补充可量化的验证判据（如性能阈值、对照实验设计）")
        if feasibility_score <= 2:
            weaknesses.append(f"可行性受已知限制约束：{feasibility_reason}")
            improvements.append("调整候选体系或变量，避开已知限制后再评估")
        if novelty_score >= 4 and evidence_score <= 3:
            weaknesses.append("新颖性与证据支撑不匹配：方向新但证据不足")
            improvements.append("为新颖组合寻找间接证据（类比候选 / gap 候选），或标注为高风险探索")
        if not weaknesses:
            weaknesses.append("未发现明显结构性问题，需结合实验可行性专家复核")
        if not improvements:
            improvements.append("保持方向，进入最小验证实验设计")

        results.append(
            {
                "rank": route.get("rank"),
                "dimensions": {
                    "novelty": {"score": novelty_score, "reason": novelty_reason},
                    "feasibility": {"score": feasibility_score, "reason": feasibility_reason},
                    "evidence": {"score": evidence_score, "reason": evidence_reason},
                    "falsifiability": {"score": falsifiability_score, "reason": falsifiability_reason},
                },
                "total": sum(scores),
                "weaknesses": weaknesses,
                "improvements": improvements,
            }
        )
    return {"mode": "fallback", "routes": results}


def _normalize_critique(value: Any, route_count: int) -> dict[str, Any]:
    """校验 LLM 输出的批判结果结构，坏条目回退到 fallback 的对应路线。"""
    if not isinstance(value, dict) or not isinstance(value.get("routes"), list):
        raise ValueError("critique 输出缺少 routes 数组")
    routes = []
    seen: set[int] = set()
    for item in value["routes"]:
        if not isinstance(item, dict):
            continue
        rank = int(item.get("rank") or 0)
        if rank in seen or not 1 <= rank <= route_count:
            continue
        seen.add(rank)
        dimensions: dict[str, Any] = {}
        for key in ("novelty", "feasibility", "evidence", "falsifiability"):
            raw = item.get("dimensions", {}).get(key) if isinstance(item.get("dimensions"), dict) else None
            if not isinstance(raw, dict):
                continue
            score = max(1, min(5, int(raw.get("score") or 3)))
            dimensions[key] = {"score": score, "reason": str(raw.get("reason") or "").strip()}
        if len(dimensions) != 4:
            continue
        routes.append(
            {
                "rank": rank,
                "dimensions": dimensions,
                "total": sum(d["score"] for d in dimensions.values()),
                "weaknesses": [str(x) for x in item.get("weaknesses") or [] if str(x).strip()][:4],
                "improvements": [str(x) for x in item.get("improvements") or [] if str(x).strip()][:4],
            }
        )
    if not routes:
        raise ValueError("critique 输出无有效路线条目")
    routes.sort(key=lambda item: item["rank"])
    return {"mode": "llm", "routes": routes}


def critique_routes(run: dict[str, Any], settings: Any, independent: dict[int, Any] | None = None) -> dict[str, Any]:
    """对 run 内全部路线做四维批判（独立 grounding）。settings 为 None 或调用失败时走确定性兜底。"""
    routes = list(run.get("routes") or [])
    fallback = fallback_critique(run, independent)
    if settings is None:
        return fallback
    try:
        payload = _llm_json(settings, system_prompt=CRITIC_SYSTEM_PROMPT, user_prompt=build_critique_prompt(run, independent))
        result = _normalize_critique(payload, route_count=len(routes))
        by_rank = {item["rank"]: item for item in result["routes"]}
        # LLM 漏掉的路线用兜底补齐，保证每条路线都有评判
        for item in fallback["routes"]:
            by_rank.setdefault(item["rank"], item)
        result["routes"] = [by_rank[rank] for rank in sorted(by_rank)]
        return result
    except Exception as exc:
        fallback["mode"] = "fallback_llm_error"
        fallback["error"] = str(exc)[:300]
        return fallback


def _llm_compare(
    run: dict[str, Any],
    route_a: dict[str, Any],
    route_b: dict[str, Any],
    settings: Any,
    independent: dict[int, Any] | None = None,
) -> dict[str, Any] | None:
    """Arena 单场裁决：基于独立检索证据（非生成上下文）对比两条路线；失败返回 None。"""
    by_rank = independent or {}
    prompt = (
        "你是一名科研路线仲裁者。基于独立检索证据比较两条候选路线哪个更值得优先推进。\n"
        "评判只能依据独立检索证据；两条路线的 rationale 与声称引用均未验证，不得直接采信。\n\n"
        "===== 路线 A =====\n"
        + _route_text(route_a)
        + "\n\n独立检索证据（A）：\n"
        + _independent_text(by_rank.get(route_a.get("rank")))
        + "\n\n===== 路线 B =====\n"
        + _route_text(route_b)
        + "\n\n独立检索证据（B）：\n"
        + _independent_text(by_rank.get(route_b.get("rank")))
        + "\n\n输出严格 JSON：{\"winner_rank\": <A 的 rank 或 B 的 rank>, \"reason\": \"...\"}。"
    )
    try:
        payload = _llm_json(
            settings,
            system_prompt="你是科研路线仲裁者，只输出严格 JSON，winner_rank 必须是两条候选路线之一的 rank。",
            user_prompt=prompt,
        )
    except Exception:
        return None
    winner = payload.get("winner_rank") if isinstance(payload, dict) else None
    valid = {route_a.get("rank"), route_b.get("rank")}
    if winner not in valid:
        return None
    return {"winner_rank": int(winner), "reason": str(payload.get("reason") or "").strip()}


def elo_arena(
    run: dict[str, Any],
    critique: dict[str, Any],
    settings: Any,
    independent: dict[int, Any] | None = None,
) -> dict[str, Any]:
    """两两比较 + Elo 排序（相邻对复核），替代绝对打分排序，规避 LLM 乐观偏差。

    初始 Elo 由确定性四维总分映射（1000 + 40 × (总分 − 12)），
    LLM 可用时对总分排序后的相邻对做仲裁复核（n−1 场，基于独立检索证据），胜者 +K/2、败者 −K/2（K=32）。
    """
    routes = list(run.get("routes") or [])
    route_by_rank = {route.get("rank"): route for route in routes}
    crit_by_rank = {item.get("rank"): item for item in critique.get("routes") or []}
    ranks = sorted(route_by_rank)
    ratings = {
        rank: 1000 + 40 * (crit_by_rank.get(rank, {}).get("total", 12) - 12)
        for rank in ranks
    }
    battles: list[dict[str, Any]] = []
    K = 32
    order = sorted(ranks, key=lambda r: ratings[r], reverse=True)
    for index in range(len(order) - 1):
        rank_a, rank_b = order[index], order[index + 1]
        route_a, route_b = route_by_rank[rank_a], route_by_rank[rank_b]
        mode = "deterministic"
        winner = rank_a if ratings[rank_a] >= ratings[rank_b] else rank_b
        reason = "按确定性四维总分裁决"
        if settings is not None and ratings[rank_a] != ratings[rank_b]:
            decision = _llm_compare(run, route_a, route_b, settings, independent)
            if decision is not None:
                mode = "llm"
                winner = int(decision["winner_rank"])
                reason = str(decision.get("reason") or "LLM 仲裁")[:300]
            else:
                mode = "llm_failed"
        half = K / 2
        loser = rank_b if winner == rank_a else rank_a
        ratings[winner] += half
        ratings[loser] -= half
        battles.append(
            {
                "a": rank_a,
                "b": rank_b,
                "winner": winner,
                "reason": reason,
                "mode": mode,
            }
        )
    rankings = [
        {"rank": rank, "elo": round(ratings[rank], 1)}
        for rank in sorted(ratings, key=lambda r: ratings[r], reverse=True)
    ]
    return {
        "battles": battles,
        "ratings": {str(rank): round(ratings[rank], 1) for rank in ranks},
        "rankings": rankings,
        "arena_mode": "llm" if any(b["mode"] == "llm" for b in battles) else "deterministic",
    }


def build_evolve_prompt(run: dict[str, Any]) -> str:
    """一轮演化提示词：父本路线 + 批判意见 + 同一快照 → v2 路线。"""
    parent_lines = [f"父本 run: {run.get('id')} · {run.get('question_title') or ''}"]
    for route in run.get("routes") or []:
        parent_lines.append(_route_text(route, with_critique=True))
    return (
        "你是路线生成器的演化版。基于同一份生成上下文快照和批判者的意见，"
        "对每条父本路线生成改进后的 v2 路线。\n"
        "要求：\n"
        "1. 保留父本路线的核心方向，针对 weaknesses 逐条回应。\n"
        "2. 不得凭空新增证据 id：只能引用快照 evidence_pool 中已有的 id，否则写成推测。\n"
        "3. 每条 v2 路线输出与生成器相同的字段：title, rationale, candidates, variables, validation, evidence, risks, priority, next_step。\n"
        "4. title 加前缀 'v2·'；rationale 开头说明相对父本路线的改进点。\n\n"
        "===== 生成上下文快照 =====\n"
        + _snapshot_text(run)
        + "\n\n===== 父本路线与批判意见 =====\n"
        + "\n\n".join(parent_lines)
        + "\n\n输出严格 JSON：{\"routes\": [...]}。"
    )


def fallback_evolve(run: dict[str, Any]) -> list[dict[str, Any]]:
    """无 LLM 时的确定性演化：把批判者的改进建议并入父本路线，标注 v2 修订。"""
    evolved: list[dict[str, Any]] = []
    for route in run.get("routes") or []:
        critique = route.get("critique") or {}
        improvements = [str(item).strip() for item in critique.get("improvements") or [] if str(item).strip()]
        revised = dict(route)
        revised["title"] = "v2·" + str(route.get("title") or "候选路线")
        revised["rationale"] = (
            "（v2 修订，依据批判意见："
            + ("；".join(improvements[:2]) if improvements else "保持原方向")
            + "）"
            + str(route.get("rationale") or "")
        )
        if improvements:
            revised["next_step"] = str(route.get("next_step") or "") + " ｜ v2 改进方向：" + "；".join(improvements[:2])
        evolved.append(revised)
    return evolved

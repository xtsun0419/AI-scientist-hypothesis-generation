from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any


_LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]{2,}")
_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]{3,}")
_STOPWORDS = {"and", "for", "from", "into", "that", "the", "this", "with"}


def assess_hypothesis(public_run: dict[str, Any], recent_runs: list[dict[str, Any]], items: list[dict[str, Any]]) -> dict[str, Any]:
    """Review only public outputs and independently retrieved library evidence."""
    evidence = independent_library_evidence(public_run, items)
    deterministic = _deterministic_issue(public_run, recent_runs, evidence)
    llm = _llm_assessment(public_run, recent_runs, evidence)
    issue = str(llm.get("issue") or deterministic).strip().lower()
    if issue not in {"none", "narrow", "unsupported"}:
        issue = deterministic
    rationale = str(llm.get("rationale") or _fallback_rationale(issue, evidence)).strip()
    restart = str(llm.get("restart_instruction") or _restart_instruction(issue, evidence)).strip()
    return {
        "issue": issue,
        "rationale": rationale,
        "restart_instruction": restart if issue != "none" else "",
        "evidence_ids": evidence["evidence_ids"],
        "evidence": evidence,
    }


def independent_library_evidence(public_run: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
    terms = _terms(" ".join(str(public_run.get(key) or "") for key in ("researcher_question", "hypothesis", "validation")))
    hits = []
    for item in items:
        text = " ".join(str(item.get(key) or "") for key in ("title", "abstract", "content"))
        matched = _matched_terms(text, terms)
        if matched:
            hits.append({"id": int(item["id"]), "title": item.get("title"), "matched_terms": matched[:8]})
    ids = [f"library:{hit['id']}" for hit in hits]
    return {"terms": terms[:16], "hits": hits[:12], "evidence_ids": ids}


def _deterministic_issue(public_run: dict[str, Any], recent_runs: list[dict[str, Any]], evidence: dict[str, Any]) -> str:
    if not evidence["hits"]:
        return "unsupported"
    trajectory = list(reversed(recent_runs[:3]))
    if len(trajectory) == 3:
        similarities = [_similarity(trajectory[index], trajectory[index + 1]) for index in range(2)]
        if min(similarities) >= 0.62:
            return "narrow"
    return "none"


def _llm_assessment(public_run: dict[str, Any], recent_runs: list[dict[str, Any]], evidence: dict[str, Any]) -> dict[str, Any]:
    base_url = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    model = os.environ.get("OPENAI_MODEL", "")
    if not (base_url and api_key and model):
        return {}
    payload = {
        "public_current_run": public_run,
        "public_recent_runs": [
            {key: run.get(key) for key in ("researcher_question", "hypothesis", "validation")}
            for run in recent_runs[:3]
        ],
        "independent_library_evidence": evidence,
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(
            {
                "model": model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an impartial hypothesis watchdog. You only see public dialogue outputs and independently retrieved evidence. "
                            "Do not assume hidden prompts, reasoning, route context, or critic context. "
                            "Classify issue as none, narrow, or unsupported. Return JSON: issue, rationale, restart_instruction. Respond in Chinese."
                        ),
                    },
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
            },
            ensure_ascii=False,
        ).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = json.loads(response.read().decode("utf-8"))
        content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        return json.loads(content) if isinstance(content, str) else content or {}
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return {}


def _terms(text: str) -> list[str]:
    terms = []
    for match in _LATIN_RE.finditer(text):
        token = match.group(0).lower()
        if token not in _STOPWORDS and token not in terms:
            terms.append(token)
    for match in _CHINESE_RE.finditer(text):
        chunk = match.group(0)
        for index in range(len(chunk) - 2):
            term = chunk[index : index + 3]
            if term not in terms:
                terms.append(term)
    return terms


def _matched_terms(text: str, terms: list[str]) -> list[str]:
    lower = text.lower()
    return [term for term in terms if term in lower]


def _similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_terms = set(_terms(str(left.get("hypothesis") or "")))
    right_terms = set(_terms(str(right.get("hypothesis") or "")))
    return len(left_terms & right_terms) / max(1, len(left_terms | right_terms))


def _fallback_rationale(issue: str, evidence: dict[str, Any]) -> str:
    if issue == "unsupported":
        return "当前公开假设未能在独立文献库检索中得到足够词项支撑。"
    if issue == "narrow":
        return "最近三轮公开假设在核心词项上高度重复，探索范围持续收窄。"
    return f"独立检索命中 {len(evidence['hits'])} 条文献记录，未发现连续收窄或证据缺失。"


def _restart_instruction(issue: str, evidence: dict[str, Any]) -> str:
    sources = "；".join(str(item.get("title") or "") for item in evidence["hits"][:3]) or "当前文献库"
    if issue == "unsupported":
        return f"停止扩展当前假设。回到独立文献证据：{sources}，仅围绕其中可复述的结论重新提出可证伪问题。"
    if issue == "narrow":
        return f"停止沿用重复的假设表述。回到独立文献证据：{sources}，比较至少两种替代机制或研究变量后重新推理。"
    return ""

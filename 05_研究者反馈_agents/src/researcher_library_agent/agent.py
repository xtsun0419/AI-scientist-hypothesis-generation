from __future__ import annotations

import json
import os
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .config import default_db_path, default_memory_path, default_retrieval_db_path
from .db import ResearcherLibraryDB
from .hypothesis_critic import assess_hypothesis
from .memory import read_memory, rebuild_memory
from .pubmed import fetch_pubmed_records


SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".json"}


class ResearcherLibraryAgent:
    """Maintains a provenance-preserving library for optional researcher simulation."""

    def __init__(self, db: ResearcherLibraryDB | None = None, *, memory_path: Path | None = None):
        self.db = db or ResearcherLibraryDB(default_db_path())
        self.memory_path = memory_path or default_memory_path()
        self.db.init_schema()
        self._refresh_memory()

    def close(self) -> None:
        self.db.close()

    def state(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.db.get_setting("enabled")),
            "metrics": self.db.metrics(),
            "items": self.db.recent_items(20),
            "questions": self.db.recent_questions(),
            "hypotheses": self.db.recent_hypotheses(),
            "reviews": self.db.recent_hypothesis_reviews(),
            "memory": read_memory(self.memory_path),
            "memory_path": str(self.memory_path),
        }

    def set_enabled(self, enabled: bool) -> dict[str, Any]:
        self.db.set_setting("enabled", bool(enabled))
        return self.state()

    def import_path(self, source: str) -> dict[str, int]:
        path = Path(source).expanduser().resolve()
        if not path.exists():
            raise ValueError("导入路径不存在。")
        paths = [path] if path.is_file() else [item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES]
        imported = 0
        for item in paths:
            self.db.upsert_item(_personal_item(item))
            imported += 1
        return {"imported": imported, "scanned": len(paths)}

    def sync_agent_literature(self, retrieval_db_path: Path | None = None) -> dict[str, int]:
        path = retrieval_db_path or default_retrieval_db_path()
        if not path.exists():
            return {"synced": 0}
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT id, doi, title, abstract, venue, year FROM papers ORDER BY id DESC").fetchall()
        for row in rows:
            key = f"agent:doi:{row['doi']}" if row["doi"] else f"agent:id:{row['id']}"
            self.db.upsert_item(
                {
                    "external_key": key,
                    "source_type": "agent",
                    "title": str(row["title"] or f"Agent paper {row['id']}"),
                    "abstract": str(row["abstract"] or ""),
                    "content": str(row["abstract"] or ""),
                    "doi": row["doi"],
                    "venue": row["venue"],
                    "year": row["year"],
                    "source_path": None,
                }
            )
        return {"synced": len(rows)}

    def import_pubmed(self, ids: list[str], *, email: str | None = None) -> dict[str, int]:
        records = fetch_pubmed_records(ids, email=email or os.environ.get("PUBMED_EMAIL", "researcher@example.com"))
        for record in records:
            key = f"agent:pubmed:{record['pmid']}"
            self.db.upsert_item(
                {
                    "external_key": key,
                    "source_type": "agent",
                    "title": record["title"] or f"PubMed {record['pmid']}",
                    "abstract": record["abstract"],
                    "content": record["abstract"],
                    "doi": record["doi"],
                    "venue": record["venue"],
                    "year": _year_or_none(record["year"]),
                    "source_path": f"pubmed:{record['pmid']}",
                }
            )
        return {"requested": len(ids), "imported": len(records)}

    def ask_researcher_question(self, *, memory: str | None = None) -> dict[str, Any]:
        if not self.db.get_setting("enabled"):
            raise ValueError("研究者模拟功能尚未启用。")
        items = self.db.recent_items(24)
        if not items:
            raise ValueError("更新后文献库为空，请先导入个人文献或同步检索结果。")
        item_ids = [int(item["id"]) for item in items]
        answer, mode = _ask_llm(_library_snapshot(items, memory=memory if memory is not None else read_memory(self.memory_path)))
        question = str(answer.get("question") or "当前证据中，哪项关键假设最需要通过可比较的验证来区分？").strip()
        rationale = str(answer.get("rationale") or "该问题基于文献库中的研究对象、方法和结果差异生成。")
        cited = [int(item) for item in answer.get("item_ids", []) if str(item).isdigit() and int(item) in item_ids]
        cited = cited or item_ids[:3]
        question_id = self.db.add_question(question=question, rationale=rationale, item_ids=cited, mode=mode)
        return {"id": question_id, "question": question, "rationale": rationale, "item_ids": cited, "mode": mode}

    def run_hypothesis_dialogue(self) -> dict[str, Any]:
        """Exchange only a public question between isolated researcher and designer calls."""
        if not self.db.get_setting("enabled"):
            raise ValueError("研究者模拟功能尚未启用。")
        items = self.db.recent_items(24)
        if len(items) < 2:
            raise ValueError("请至少准备 1 篇个人文献和 1 篇 AI 文献后再运行自动对话。")
        memory = read_memory(self.memory_path)
        question_run = self.ask_researcher_question(memory=memory)
        payload = {
            "library": _library_snapshot(items, memory=memory),
            "published_researcher_question": question_run["question"],
        }
        answer, mode = _design_hypothesis(payload)
        hypothesis = str(answer.get("hypothesis") or "待验证假设：文献库中反复出现的关键变量与目标结果之间存在可检验的因果关系。").strip()
        rationale = _structured_text(answer.get("rationale")) or "该假设仅基于更新后文献库中的证据摘要和公开提问生成。"
        validation = _structured_text(answer.get("validation")) or "使用预先定义的对照、可量化指标和独立复现实验检验该假设。"
        item_ids = [int(item["id"]) for item in items]
        run_id = self.db.add_hypothesis_run(
            researcher_question=question_run["question"],
            hypothesis=hypothesis,
            rationale=rationale,
            validation=validation,
            item_ids=item_ids,
            mode=f"researcher:{question_run['mode']};designer:{mode}",
        )
        public_run = {
            "id": run_id,
            "researcher_question": question_run["question"],
            "hypothesis": hypothesis,
            "validation": validation,
        }
        assessment = assess_hypothesis(public_run, self.db.recent_hypotheses(3), items)
        status = _watchdog_status(assessment["issue"], self.db.recent_hypothesis_reviews(2))
        self.db.add_hypothesis_review(
            run_id=run_id,
            status=status,
            issue=assessment["issue"],
            rationale=assessment["rationale"],
            restart_instruction=assessment["restart_instruction"] if status == "interrupt" else None,
            evidence_ids=assessment["evidence_ids"],
        )
        self._refresh_memory()
        result = {
            "id": run_id,
            "researcher_question": question_run["question"],
            "hypothesis": hypothesis,
            "rationale": rationale,
            "validation": validation,
            "mode": f"researcher:{question_run['mode']};designer:{mode}",
            "watchdog": {"status": status, **assessment},
        }
        if status == "interrupt":
            result["rethink"] = self._rethink_after_interrupt(items, question_run["question"], assessment["restart_instruction"])
            self._refresh_memory()
        return result

    def design_context(self) -> dict[str, Any]:
        """Only exposes evidence records, never simulated questions or workflow conversations."""
        if not self.db.get_setting("enabled"):
            return {"enabled": False, "cards": [], "metrics": self.db.metrics()}
        items = self.db.recent_items(24)
        cards = []
        for item in items:
            evidence_id = f"library:{item['id']}"
            summary = str(item.get("abstract") or item.get("content") or "")[:1200]
            cards.append(
                {
                    "paper_id": evidence_id,
                    "title": item["title"],
                    "summary": summary,
                    "materials": [],
                    "methods": [],
                    "properties": [],
                    "claims": [{"id": f"claim:{evidence_id}", "text": summary, "evidence_ids": [evidence_id]}] if summary else [],
                    "evidence_ids": [evidence_id],
                    "source_type": item["source_type"],
                }
            )
        return {"enabled": True, "cards": cards, "metrics": self.db.metrics()}

    def _refresh_memory(self) -> str:
        runs = self.db.recent_hypotheses()
        item_ids = []
        for run in runs:
            try:
                item_ids.extend(int(item) for item in json.loads(run.get("item_ids_json") or "[]"))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        return rebuild_memory(
            self.memory_path,
            runs,
            self.db.items_by_ids(list(dict.fromkeys(item_ids))),
            self.db.review_for_runs([int(run["id"]) for run in runs]),
        )

    def _rethink_after_interrupt(self, items: list[dict[str, Any]], question: str, restart_instruction: str) -> dict[str, Any]:
        answer, mode = _design_hypothesis(
            {
                "library": _library_snapshot(items, memory=read_memory(self.memory_path)),
                "published_researcher_question": question,
                "published_watchdog_instruction": restart_instruction,
            }
        )
        hypothesis = str(answer.get("hypothesis") or "待验证假设：应从独立文献证据重新选择研究变量和可证伪机制。").strip()
        rationale = _structured_text(answer.get("rationale")) or "该假设由公开重置指令和独立文献快照重新推理。"
        validation = _structured_text(answer.get("validation")) or "使用独立证据支持的对照设计重新检验。"
        item_ids = [int(item["id"]) for item in items]
        run_id = self.db.add_hypothesis_run(
            researcher_question=question,
            hypothesis=hypothesis,
            rationale=rationale,
            validation=validation,
            item_ids=item_ids,
            mode=f"watchdog_rethink:{mode}",
        )
        return {"id": run_id, "hypothesis": hypothesis, "rationale": rationale, "validation": validation, "mode": f"watchdog_rethink:{mode}"}


def _personal_item(path: Path) -> dict[str, Any]:
    content = _extract_text(path)
    return {
        "external_key": f"personal:path:{path}",
        "source_type": "personal",
        "title": path.stem.replace("_", " "),
        "abstract": content[:2000],
        "content": content[:12000],
        "doi": None,
        "venue": None,
        "year": None,
        "source_path": str(path),
    }


def _extract_text(path: Path) -> str:
    if path.suffix.lower() in {".txt", ".md", ".json"}:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return ""
    if path.suffix.lower() == ".pdf":
        try:
            import fitz

            with fitz.open(path) as document:
                return "\n".join(page.get_text() for page in list(document)[:5])
        except Exception:
            return ""
    return ""


def _library_snapshot(items: list[dict[str, Any]], *, memory: str = "") -> dict[str, Any]:
    return {
        "instruction": "Use only these library records. Do not invent papers, authors, identifiers, findings, or prior conversation.",
        "public_dialogue_memory": memory,
        "items": [
            {"id": item["id"], "source": item["source_type"], "title": item["title"], "abstract": str(item.get("abstract") or "")[:900]}
            for item in items
        ],
    }


def _ask_llm(snapshot: dict[str, Any]) -> tuple[dict[str, Any], str]:
    return _chat_json(
        system_prompt="Act as a careful researcher. Based only on the supplied library snapshot, ask one specific, falsifiable next question. Respond in Chinese. Return JSON: question, rationale, item_ids.",
        payload=snapshot,
    )


def _design_hypothesis(payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    return _chat_json(
        system_prompt=(
            "Act as an independent hypothesis designer. You receive only a literature snapshot and a published researcher question. "
            "Do not assume access to any prior conversation, hidden reasoning, routes, or critiques. Use only the supplied records. "
            "Respond in Chinese. Return JSON: hypothesis, rationale, validation."
        ),
        payload=payload,
    )


def _chat_json(*, system_prompt: str, payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    base_url = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    model = os.environ.get("OPENAI_MODEL", "")
    if not (base_url and api_key and model):
        return {}, "fallback_no_api_key"
    payload = {
        "model": model,
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = json.loads(response.read().decode("utf-8"))
        content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        return json.loads(content) if isinstance(content, str) else content, "llm"
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return {}, "fallback_llm_error"


def _year_or_none(value: Any) -> int | None:
    try:
        return int(str(value)[:4])
    except (TypeError, ValueError):
        return None


def _structured_text(value: Any) -> str:
    if isinstance(value, dict):
        return "\n".join(f"{key}：{_structured_text(item)}" for key, item in value.items())
    if isinstance(value, list):
        return "；".join(_structured_text(item) for item in value)
    return str(value or "").strip()


def _watchdog_status(issue: str, prior_reviews: list[dict[str, Any]]) -> str:
    if issue == "none":
        return "continue"
    same_issue = [item for item in prior_reviews if item.get("issue") == issue and item.get("status") != "continue"]
    return "interrupt" if len(same_issue) >= 2 else "warning"

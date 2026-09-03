from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import default_cards_dir, default_graph_dir, default_index_dir, default_parsed_dir, default_wiki_dir
from .db import AnalysisDB
from .text import normalize_whitespace


METHOD_TERMS = {
    "computational modeling": ["simulation", "modeling", "model", "algorithm"],
    "machine learning": ["machine learning", "neural network", "classification", "regression"],
    "statistical analysis": ["statistical", "regression analysis", "confidence interval"],
    "experimental study": ["experiment", "experimental", "measurement", "survey"],
    "systematic review": ["systematic review", "meta-analysis", "literature review"],
}
LIMITATION_TERMS = ["limit", "limitation", "challenge", "however", "but", "uncertain", "trade-off"]
STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "has", "have",
    "using", "into", "its", "can", "will", "not", "their", "these", "those", "study", "paper",
}
TOPIC_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]{2,}|[\u4e00-\u9fff]{2,}")


@dataclass(frozen=True)
class CorpusDocument:
    source_json_path: Path
    paper_id: int | None
    metadata: dict[str, Any]
    paragraphs: list[dict[str, Any]]
    figures: list[dict[str, Any]]
    tables: list[dict[str, Any]]
    references: list[dict[str, Any]]


class CorpusIndexAgent:
    def __init__(self, db: AnalysisDB, parsed_dir: Path | None = None, index_dir: Path | None = None):
        self.db = db
        self.parsed_dir = parsed_dir or default_parsed_dir()
        self.index_dir = index_dir or default_index_dir()

    def run(self, *, force: bool = True) -> dict[str, int | str]:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        if force:
            self.db.clear_rag_index()
        docs = load_corpus_documents(self.db, self.parsed_dir)
        chunk_count = 0
        embedding_count = 0
        model = os.environ.get("OPENAI_EMBEDDING_MODEL") or "local-hash-v1"
        for doc in docs:
            for chunk in chunks_for_document(doc):
                self.db.insert_rag_chunk(**chunk)
                vector = local_embedding(chunk["text"])
                self.db.upsert_embedding(chunk_id=chunk["chunk_id"], model=model, vector_json=json.dumps(vector))
                chunk_count += 1
                embedding_count += 1
        (self.index_dir / "index_status.json").write_text(
            json.dumps(
                {
                    "documents": len(docs),
                    "chunks": chunk_count,
                    "embeddings": embedding_count,
                    "embedding_mode": "configured" if os.environ.get("OPENAI_EMBEDDING_MODEL") else "local_hash_fallback",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return {"documents": len(docs), "chunks": chunk_count, "embeddings": embedding_count, "embedding_mode": "configured" if os.environ.get("OPENAI_EMBEDDING_MODEL") else "local_hash_fallback"}


class HybridSearchAgent:
    def __init__(self, db: AnalysisDB):
        self.db = db

    def search(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        query = normalize_whitespace(query) or ""
        if not query:
            return []
        bm25_rows = self._safe_bm25(query, limit=max(limit * 3, 20))
        bm25_by_id = {row["id"]: row for row in bm25_rows}
        query_vector = local_embedding(query)
        vector_scores: dict[str, float] = {}
        for row in self.db.rag_chunks():
            vector_scores[row["id"]] = cosine(query_vector, local_embedding(row["text"]))
        candidate_ids = set(bm25_by_id) | set(sorted(vector_scores, key=vector_scores.get, reverse=True)[: max(limit * 3, 20)])
        ranked = []
        for chunk_id in candidate_ids:
            row = bm25_by_id.get(chunk_id) or self._chunk_by_id(chunk_id)
            if row is None:
                continue
            bm25_score = float(row["bm25_score"]) if "bm25_score" in row.keys() and row["bm25_score"] is not None else 0.0
            bm25_norm = 1.0 / (1.0 + max(bm25_score, 0.0)) if chunk_id in bm25_by_id else 0.0
            vector_norm = max(0.0, vector_scores.get(chunk_id, 0.0))
            score = 0.55 * bm25_norm + 0.45 * vector_norm
            ranked.append((score, row, bm25_norm, vector_norm))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "id": row["id"],
                "paper_id": row["paper_id"],
                "chunk_type": row["chunk_type"],
                "title": row["title"],
                "section_path": row["section_path"],
                "source_json_path": row["source_json_path"],
                "text": row["text"],
                "page": row["page"],
                "score": round(score, 4),
                "bm25_score": round(bm25_norm, 4),
                "embedding_score": round(vector_norm, 4),
            }
            for score, row, bm25_norm, vector_norm in ranked[:limit]
        ]

    def _safe_bm25(self, query: str, limit: int) -> list[Any]:
        terms = " OR ".join(_query_terms(query)[:8]) or query
        try:
            return self.db.search_chunks_bm25(terms, limit=limit)
        except Exception:
            return []

    def _chunk_by_id(self, chunk_id: str) -> Any | None:
        rows = [row for row in self.db.rag_chunks() if row["id"] == chunk_id]
        return rows[0] if rows else None


class PaperCardAgent:
    def __init__(self, db: AnalysisDB, parsed_dir: Path | None = None, cards_dir: Path | None = None):
        self.db = db
        self.parsed_dir = parsed_dir or default_parsed_dir()
        self.cards_dir = cards_dir or default_cards_dir()

    def run(self) -> dict[str, int]:
        self.cards_dir.mkdir(parents=True, exist_ok=True)
        self.db.clear_paper_cards()
        docs = load_corpus_documents(self.db, self.parsed_dir)
        count = 0
        for doc in docs:
            card = paper_card_for_document(doc)
            paper_id = int(card["paper_id"])
            stem = _safe_slug(card["title"])[:80] or f"paper_{paper_id}"
            json_path = self.cards_dir / f"paper_{paper_id}_{stem}.json"
            md_path = self.cards_dir / f"paper_{paper_id}_{stem}.md"
            json_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
            md_path.write_text(render_card_markdown(card), encoding="utf-8")
            self.db.upsert_paper_card(
                paper_id=paper_id,
                source_json_path=str(doc.source_json_path),
                card_json=json.dumps(card, ensure_ascii=False),
                markdown_path=str(md_path),
            )
            count += 1
        return {"cards": count}


class KnowledgeGraphAgent:
    def __init__(self, db: AnalysisDB, graph_dir: Path | None = None):
        self.db = db
        self.graph_dir = graph_dir or default_graph_dir()

    def run(self) -> dict[str, int | str]:
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self.db.clear_graph()
        nodes: dict[str, dict[str, Any]] = {}
        edges: dict[str, dict[str, Any]] = {}
        for row in self.db.paper_cards():
            card = json.loads(row["card_json"])
            paper_node = _node("paper", f"paper:{card['paper_id']}", card["title"], card.get("summary"), {"paper_id": card["paper_id"]}, weight=2)
            nodes[paper_node["id"]] = paper_node
            for category, edge_type in [("materials", "mentions"), ("methods", "uses_method"), ("properties", "has_property")]:
                for label in card.get(category, []):
                    topic_type = "material" if category == "materials" else "method" if category == "methods" else "property"
                    topic = _node(topic_type, label, label, _topic_summary(label, [card]), {"label": label}, weight=3)
                    nodes[topic["id"]] = _merge_node(nodes.get(topic["id"]), topic)
                    edges[_edge_id(paper_node["id"], topic["id"], edge_type)] = _edge(paper_node["id"], topic["id"], edge_type, card.get("evidence_ids", []))
                    wiki = _node("wiki_topic", label, label, _topic_summary(label, [card]), {"topic": label}, weight=4)
                    nodes[wiki["id"]] = _merge_node(nodes.get(wiki["id"]), wiki)
                    edges[_edge_id(wiki["id"], topic["id"], "mentions")] = _edge(wiki["id"], topic["id"], "mentions", card.get("evidence_ids", []))
            for claim in card.get("claims", [])[:4]:
                claim_node = _node("claim", claim["id"], claim["text"][:96], claim["text"], claim, weight=1.5)
                nodes[claim_node["id"]] = claim_node
                edges[_edge_id(paper_node["id"], claim_node["id"], "supports")] = _edge(paper_node["id"], claim_node["id"], "supports", claim.get("evidence_ids", []))
                for evidence_id in claim.get("evidence_ids", [])[:3]:
                    evidence_node = _node("evidence", evidence_id, evidence_id, "Evidence paragraph", {"evidence_id": evidence_id}, weight=1)
                    nodes[evidence_node["id"]] = evidence_node
                    edges[_edge_id(claim_node["id"], evidence_node["id"], "supports")] = _edge(claim_node["id"], evidence_node["id"], "supports", [evidence_id])
        for node in nodes.values():
            self.db.upsert_knowledge_node(
                node_id=node["id"],
                node_type=node["type"],
                label=node["label"],
                summary=node.get("summary"),
                payload_json=json.dumps(node.get("payload", {}), ensure_ascii=False),
                weight=float(node.get("weight", 1)),
            )
        for edge in edges.values():
            self.db.upsert_knowledge_edge(
                edge_id=edge["id"],
                source_id=edge["source"],
                target_id=edge["target"],
                edge_type=edge["type"],
                evidence_json=json.dumps(edge.get("evidence_ids", []), ensure_ascii=False),
                weight=float(edge.get("weight", 1)),
            )
        graph = {
            "nodes": list(nodes.values()),
            "edges": list(edges.values()),
        }
        graph_path = self.graph_dir / "graph.json"
        graph_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"nodes": len(nodes), "edges": len(edges), "graph_path": str(graph_path)}


class WikiAgent:
    def __init__(self, db: AnalysisDB, wiki_dir: Path | None = None, graph_dir: Path | None = None):
        self.db = db
        self.wiki_dir = wiki_dir or default_wiki_dir()
        self.graph_dir = graph_dir or default_graph_dir()

    def run(self) -> dict[str, int]:
        self.wiki_dir.mkdir(parents=True, exist_ok=True)
        self.db.clear_wiki_pages()
        cards = [json.loads(row["card_json"]) for row in self.db.paper_cards()]
        topics = [dict(row) for row in self.db.knowledge_nodes() if row["node_type"] == "wiki_topic"]
        count = 0
        for topic in topics:
            topic_cards = [card for card in cards if _card_mentions_topic(card, topic["label"])]
            page = wiki_page_for_topic(topic, topic_cards)
            page_id = topic["id"].replace(":", "_")
            md_path = self.wiki_dir / f"{page_id}.md"
            json_path = self.wiki_dir / f"{page_id}.json"
            md_path.write_text(render_wiki_markdown(page), encoding="utf-8")
            json_path.write_text(json.dumps(page, ensure_ascii=False, indent=2), encoding="utf-8")
            self.db.upsert_wiki_page(
                page_id=page_id,
                title=page["title"],
                node_id=topic["id"],
                page_json=json.dumps(page, ensure_ascii=False),
                markdown_path=str(md_path),
            )
            count += 1
        self._write_graph_with_wiki()
        return {"wiki_pages": count}

    def _write_graph_with_wiki(self) -> None:
        graph_path = self.graph_dir / "graph.json"
        if not graph_path.exists():
            return
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        pages = {row["node_id"]: json.loads(row["page_json"]) for row in self.db.wiki_pages()}
        for node in graph.get("nodes", []):
            if node["id"] in pages:
                node["wiki"] = pages[node["id"]]
        graph_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")


def load_corpus_documents(db: AnalysisDB, parsed_dir: Path | None = None) -> list[CorpusDocument]:
    docs: list[CorpusDocument] = []
    seen: set[str] = set()
    for row in db.converted_papers():
        path = Path(str(row["json_path"]))
        if not path.exists() and parsed_dir is not None:
            alt = parsed_dir / path.name
            path = alt if alt.exists() else path
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        metadata = data.get("metadata", {})
        paper_id = row["paper_id"] if row["paper_id"] is not None else _paper_id_from_path(path)
        dedupe_key = f"paper:{paper_id}" if paper_id is not None else f"path:{path.resolve()}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        docs.append(
            CorpusDocument(
                source_json_path=path,
                paper_id=int(paper_id) if paper_id is not None else None,
                metadata=metadata,
                paragraphs=list(data.get("paragraphs") or []),
                figures=list(data.get("figures") or []),
                tables=list(data.get("tables") or []),
                references=list(data.get("references") or []),
            )
        )
    return docs


def chunks_for_document(doc: CorpusDocument) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    title = str(doc.metadata.get("title") or doc.source_json_path.stem)
    for paragraph in doc.paragraphs:
        text = normalize_whitespace(paragraph.get("text"))
        if not text:
            continue
        chunk_id = f"p:{doc.paper_id}:{paragraph.get('id')}"
        chunks.append(_chunk(doc, chunk_id, "paragraph", title, paragraph.get("section_path") or [], text, paragraph.get("page"), paragraph))
    for index, figure in enumerate(doc.figures, start=1):
        text = normalize_whitespace(figure.get("caption"))
        if text:
            chunks.append(_chunk(doc, f"fig:{doc.paper_id}:{index}", "caption", title, ["Figures"], text, figure.get("page"), figure))
    for index, table in enumerate(doc.tables, start=1):
        text = normalize_whitespace(" ".join([str(table.get("caption") or ""), str(table.get("text") or "")]))
        if text:
            chunks.append(_chunk(doc, f"tbl:{doc.paper_id}:{index}", "table", title, ["Tables"], text, table.get("page"), table))
    for index, reference in enumerate(doc.references, start=1):
        text = normalize_whitespace(reference.get("raw_text"))
        if text:
            chunks.append(_chunk(doc, f"ref:{doc.paper_id}:{index}", "reference", title, ["References"], text, None, reference))
    return chunks


def paper_card_for_document(doc: CorpusDocument) -> dict[str, Any]:
    text = " ".join([str(doc.metadata.get("title") or ""), str(doc.metadata.get("abstract") or "")] + [str(p.get("text") or "") for p in doc.paragraphs[:16]])
    materials = _research_topics(str(doc.metadata.get("title") or ""), str(doc.metadata.get("abstract") or ""))
    methods = _matches(text, METHOD_TERMS)
    properties = _metric_labels(doc.paragraphs)
    evidence_paragraphs = _best_evidence(doc.paragraphs, materials + methods + properties)
    paper_id = doc.paper_id or _paper_id_from_path(doc.source_json_path) or 0
    claims = [
        {
            "id": f"claim:{paper_id}:{index}",
            "text": paragraph["text"],
            "evidence_ids": [_paragraph_evidence_id(paper_id, paragraph)],
            "needs_evidence": False,
        }
        for index, paragraph in enumerate(evidence_paragraphs[:5], start=1)
    ]
    limitations = [p["text"] for p in doc.paragraphs if any(term in str(p.get("text", "")).lower() for term in LIMITATION_TERMS)][:3]
    metrics = _metric_sentences(doc.paragraphs)
    return {
        "paper_id": paper_id,
        "title": doc.metadata.get("title") or doc.source_json_path.stem,
        "doi": doc.metadata.get("doi"),
        "year": doc.metadata.get("year"),
        "venue": doc.metadata.get("venue"),
        "source_json_path": str(doc.source_json_path),
        "summary": doc.metadata.get("abstract") or (evidence_paragraphs[0]["text"] if evidence_paragraphs else ""),
        "research_object": _research_object(materials, properties),
        "materials": materials,
        "methods": methods,
        "properties": properties,
        "key_metrics": metrics,
        "claims": claims,
        "limitations": limitations,
        "reusable_data": metrics + [table.get("caption") for table in doc.tables if table.get("caption")],
        "evidence_ids": [_paragraph_evidence_id(paper_id, paragraph) for paragraph in evidence_paragraphs],
    }


def wiki_page_for_topic(topic: dict[str, Any], cards: list[dict[str, Any]]) -> dict[str, Any]:
    claims = []
    evidence_ids = []
    for card in cards:
        for claim in card.get("claims", [])[:3]:
            if claim.get("evidence_ids"):
                claims.append({"paper_id": card["paper_id"], "title": card["title"], "claim": claim["text"], "evidence_ids": claim["evidence_ids"]})
                evidence_ids.extend(claim["evidence_ids"])
    return {
        "title": topic["label"],
        "node_id": topic["id"],
        "summary": _topic_summary(topic["label"], cards),
        "known_findings": claims[:8],
        "evidence_ids": list(dict.fromkeys(evidence_ids)),
        "papers": [{"paper_id": card["paper_id"], "title": card["title"]} for card in cards],
        "limitations": list(dict.fromkeys(limit for card in cards for limit in card.get("limitations", [])))[:5],
        "open_questions": _open_questions(topic["label"], cards),
        "needs_evidence": [] if evidence_ids else [f"{topic['label']} needs direct evidence before being treated as a stable wiki topic."],
    }


def render_card_markdown(card: dict[str, Any]) -> str:
    lines = [f"# {card['title']}", "", f"- DOI: {card.get('doi') or ''}", f"- Year: {card.get('year') or ''}", f"- Research object: {card.get('research_object') or ''}", ""]
    for label, key in [("Materials", "materials"), ("Methods", "methods"), ("Properties", "properties")]:
        lines.extend([f"## {label}", ""])
        lines.extend(f"- {item}" for item in card.get(key, []))
        lines.append("")
    lines.extend(["## Claims", ""])
    lines.extend(f"- {claim['text']} (`{', '.join(claim.get('evidence_ids', []))}`)" for claim in card.get("claims", []))
    return "\n".join(lines).strip() + "\n"


def render_wiki_markdown(page: dict[str, Any]) -> str:
    lines = [f"# {page['title']}", "", page.get("summary") or "", "", "## Known Findings", ""]
    lines.extend(f"- {item['claim']} (`{', '.join(item.get('evidence_ids', []))}`)" for item in page.get("known_findings", []))
    lines.extend(["", "## Open Questions", ""])
    lines.extend(f"- {item}" for item in page.get("open_questions", []))
    return "\n".join(lines).strip() + "\n"


def local_embedding(text: str, dims: int = 64) -> list[float]:
    vector = [0.0] * dims
    for token in _tokens(text):
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % dims
        sign = 1 if digest[2] % 2 == 0 else -1
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [round(value / norm, 6) for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def _chunk(doc: CorpusDocument, chunk_id: str, chunk_type: str, title: str, section_path: list[Any], text: str, page: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "paper_id": doc.paper_id,
        "source_json_path": str(doc.source_json_path),
        "chunk_type": chunk_type,
        "title": title,
        "section_path": " / ".join(str(item) for item in section_path) if section_path else "",
        "text": text,
        "page": int(page) if page else None,
        "metadata_json": json.dumps(metadata, ensure_ascii=False),
    }


def _matches(text: str, term_map: dict[str, list[str]]) -> list[str]:
    lower = text.lower()
    return [label for label, terms in term_map.items() if any(term in lower for term in terms)]


def _best_evidence(paragraphs: list[dict[str, Any]], labels: list[str]) -> list[dict[str, Any]]:
    terms = [term.lower() for label in labels for term in re.split(r"[\s/-]+", label) if len(term) > 2]
    scored = []
    for paragraph in paragraphs:
        text = str(paragraph.get("text") or "")
        lower = text.lower()
        score = sum(1 for term in terms if term and term in lower)
        if any(char.isdigit() for char in text):
            score += 1
        if score > 0 and len(text) > 40:
            scored.append((score, paragraph))
    scored.sort(key=lambda item: item[0], reverse=True)
    if scored:
        return [paragraph for _, paragraph in scored[:8]]
    return [paragraph for paragraph in paragraphs if len(str(paragraph.get("text") or "")) > 40][:8]


def _metric_sentences(paragraphs: list[dict[str, Any]]) -> list[str]:
    hits = []
    for paragraph in paragraphs:
        text = str(paragraph.get("text") or "")
        if re.search(r"\d", text):
            hits.append(text)
    return hits[:5]


def _research_topics(title: str, abstract: str) -> list[str]:
    topics: list[str] = []
    for match in TOPIC_TOKEN_RE.finditer(f"{title} {abstract}"):
        token = match.group(0).strip()
        if token.lower() in STOPWORDS or token.lower() in {"analysis", "approach", "method", "methods", "results"}:
            continue
        if token not in topics:
            topics.append(token)
    return topics[:8]


def _metric_labels(paragraphs: list[dict[str, Any]]) -> list[str]:
    return ["quantitative result"] if _metric_sentences(paragraphs) else []


def _research_object(materials: list[str], properties: list[str]) -> str:
    left = ", ".join(materials[:3]) or "the research topic"
    right = ", ".join(properties[:3]) or "the reported outcomes"
    return f"{left} for {right}"


def _node(node_type: str, key: str, label: str, summary: str | None, payload: dict[str, Any], weight: float = 1) -> dict[str, Any]:
    node_id = key if ":" in key else f"{node_type}:{_safe_slug(key)}"
    return {"id": node_id, "type": node_type, "label": label, "summary": summary, "payload": payload, "weight": weight}


def _merge_node(existing: dict[str, Any] | None, new: dict[str, Any]) -> dict[str, Any]:
    if existing is None:
        return new
    existing["weight"] = float(existing.get("weight", 1)) + float(new.get("weight", 1))
    if not existing.get("summary") and new.get("summary"):
        existing["summary"] = new["summary"]
    return existing


def _edge(source_id: str, target_id: str, edge_type: str, evidence_ids: list[str]) -> dict[str, Any]:
    return {"id": _edge_id(source_id, target_id, edge_type), "source": source_id, "target": target_id, "type": edge_type, "weight": max(1, len(evidence_ids)), "evidence_ids": list(dict.fromkeys(evidence_ids))}


def _edge_id(source_id: str, target_id: str, edge_type: str) -> str:
    return f"{edge_type}:{source_id}->{target_id}"


def _topic_summary(label: str, cards: list[dict[str, Any]]) -> str:
    if not cards:
        return f"{label} is a candidate wiki topic that still needs direct evidence."
    titles = "; ".join(card["title"] for card in cards[:3])
    return f"{label} appears across {len(cards)} paper card(s), including {titles}."


def _card_mentions_topic(card: dict[str, Any], label: str) -> bool:
    label_lower = label.lower()
    fields = card.get("materials", []) + card.get("methods", []) + card.get("properties", [])
    return any(str(item).lower() == label_lower for item in fields)


def _open_questions(label: str, cards: list[dict[str, Any]]) -> list[str]:
    if not cards:
        return [f"What direct evidence supports {label}?"]
    return [
        f"Which experimental or computational evidence most strongly validates {label}?",
        f"Where do reported conclusions about {label} conflict or remain under-specified?",
    ]


def _safe_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower() or "untitled"


def _paper_id_from_path(path: Path) -> int | None:
    match = re.search(r"paper_(\d+)", path.name)
    return int(match.group(1)) if match else None


def _tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_]+", text.lower()) if token not in STOPWORDS]


def _query_terms(query: str) -> list[str]:
    return [token for token in _tokens(query) if len(token) > 2]


def _paragraph_evidence_id(paper_id: int, paragraph: dict[str, Any]) -> str:
    return f"p:{paper_id}:{paragraph.get('id')}"

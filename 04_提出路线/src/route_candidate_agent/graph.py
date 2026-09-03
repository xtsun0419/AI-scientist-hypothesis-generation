from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

from .config import default_graph_path


class KnowledgeGraph:
    """加载 02 模块生成的 graph.json，提供图上下文与确定性图指标。

    图 schema（由 02 模块生成）：
    - 节点类型：paper / material / method / property / wiki_topic / claim / evidence
    - 边类型：mentions（paper→material/wiki_topic）、uses_method（paper→method）、
      has_property（paper→property）、supports（paper→claim）
    """

    def __init__(self, data: dict[str, Any]):
        nodes = [item for item in data.get("nodes") or [] if isinstance(item, dict)]
        edges = [item for item in data.get("edges") or [] if isinstance(item, dict)]
        self.nodes = nodes
        self.edges = edges
        self.nodes_by_id = {str(item.get("id")): item for item in nodes if item.get("id")}
        self.adjacency: dict[str, set[str]] = {}
        for edge in edges:
            source = str(edge.get("source") or "")
            target = str(edge.get("target") or "")
            if not source or not target:
                continue
            self.adjacency.setdefault(source, set()).add(target)
            self.adjacency.setdefault(target, set()).add(source)

    def is_empty(self) -> bool:
        return not self.nodes

    def node(self, node_id: str) -> dict[str, Any] | None:
        return self.nodes_by_id.get(node_id)

    def nodes_of_type(self, node_type: str) -> list[dict[str, Any]]:
        return [item for item in self.nodes if item.get("type") == node_type]

    def neighbor_ids(self, node_id: str) -> set[str]:
        return set(self.adjacency.get(node_id) or set())

    def shortest_path(self, start: str, end: str) -> list[str]:
        """无向 BFS 最短路径，返回节点 id 序列；不可达返回空列表。"""
        if start not in self.nodes_by_id or end not in self.nodes_by_id:
            return []
        if start == end:
            return [start]
        queue: deque[tuple[str, list[str]]] = deque([(start, [start])])
        visited = {start}
        while queue:
            current, path = queue.popleft()
            for neighbor in sorted(self.adjacency.get(current, set())):
                if neighbor == end:
                    return path + [neighbor]
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
        return []

    def match_entities(self, texts: list[str]) -> list[dict[str, Any]]:
        """在文本中查找图中实体节点（material/method/property，label 子串匹配，忽略大小写）。"""
        matched: list[dict[str, Any]] = []
        seen: set[str] = set()
        lowered_text = " ".join(texts).lower()
        for node in self.nodes:
            if node.get("type") not in {"material", "method", "property"}:
                continue
            label = str(node.get("label") or "").strip()
            if not label:
                continue
            if label.lower() in lowered_text and node.get("id") not in seen:
                seen.add(str(node.get("id")))
                matched.append({"id": str(node.get("id")), "type": node.get("type"), "label": label})
        return matched


def load_knowledge_graph(graph_path: Path | None = None) -> KnowledgeGraph | None:
    path = graph_path or default_graph_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return KnowledgeGraph(data)


def graph_stats(graph: KnowledgeGraph | None) -> dict[str, Any]:
    if graph is None or graph.is_empty():
        return {"loaded": False, "nodes": 0, "edges": 0}
    return {
        "loaded": True,
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "materials": len(graph.nodes_of_type("material")),
        "methods": len(graph.nodes_of_type("method")),
        "properties": len(graph.nodes_of_type("property")),
        "papers": len(graph.nodes_of_type("paper")),
    }


def graph_context(graph: KnowledgeGraph | None, question: dict[str, Any]) -> dict[str, Any]:
    """为所选问题生成图局部上下文：命中实体 + 邻居节点 + 相关边 + 相关证据 id。"""
    if graph is None or graph.is_empty():
        return {"loaded": False, "matched_entities": [], "nodes": [], "edges": [], "evidence_ids": []}
    texts = [
        str(question.get("title") or ""),
        str(question.get("description") or ""),
        *[str(item) for item in (question.get("variables") or [])],
        *[str(item) for item in (question.get("candidates") or [])],
    ]
    matched = graph.match_entities(texts)
    involved: set[str] = set()
    for item in matched:
        involved.add(item["id"])
        involved.update(graph.neighbor_ids(item["id"]))
    nodes = [graph.nodes_by_id[node_id] for node_id in sorted(involved) if node_id in graph.nodes_by_id]
    edges = [
        edge
        for edge in graph.edges
        if str(edge.get("source") or "") in involved and str(edge.get("target") or "") in involved
    ]
    evidence_ids: list[str] = []
    for edge in edges:
        for item in edge.get("evidence_ids") or []:
            text = str(item).strip()
            if text and text not in evidence_ids:
                evidence_ids.append(text)
    return {
        "loaded": True,
        "matched_entities": matched,
        "nodes": nodes,
        "edges": edges,
        "evidence_ids": evidence_ids[:40],
    }


def gap_candidates(graph: KnowledgeGraph | None) -> list[dict[str, Any]]:
    """Gap detection：方法 X 已用于材料 A，但从未用于材料 B。

    规则：paper→(uses_method)→method X 且 paper→(mentions)→material A ⇒ A 用过 X；
    图中存在材料 B 没有对应的 paper 同时指向 X ⇒ "X 尚未用于 B" 的候选 gap。
    """
    if graph is None or graph.is_empty():
        return []
    method_to_materials: dict[str, dict[str, list[str]]] = {}
    for edge in graph.edges:
        if edge.get("type") != "uses_method":
            continue
        method_id = str(edge.get("source") or "")  # paper id
        method_target = str(edge.get("target") or "")
        paper_id = method_id
        materials = materials_mentioned_by(graph, paper_id)
        for material_id in materials:
            entry = method_to_materials.setdefault(method_target, {})
            entry.setdefault(material_id, [])
            for item in edge.get("evidence_ids") or []:
                text = str(item)
                if text not in entry[material_id]:
                    entry[material_id].append(text)
    gaps: list[dict[str, Any]] = []
    for method_id, used_by in method_to_materials.items():
        method = graph.node(method_id)
        for material_node in graph.nodes_of_type("material"):
            material_id = str(material_node.get("id"))
            if material_id in used_by:
                continue
            sources = [
                {"material": str(graph.node(mid).get("label") or mid) if graph.node(mid) else mid, "evidence_ids": eids}
                for mid, eids in used_by.items()
            ]
            gaps.append(
                {
                    "material": str(material_node.get("label") or material_id),
                    "material_id": material_id,
                    "method": str((method or {}).get("label") or method_id),
                    "method_id": method_id,
                    "used_on": sources,
                    "evidence_ids": [eid for item in sources for eid in item["evidence_ids"]][:8],
                }
            )
    return gaps


def analogy_candidates(graph: KnowledgeGraph | None) -> list[dict[str, Any]]:
    """Analogy search：材料 A 与 B 共享性能关注点（同一 property），A 用过方法 X 而 B 没用 ⇒ X 可迁移到 B。"""
    if graph is None or graph.is_empty():
        return []
    property_to_materials: dict[str, set[str]] = {}
    for edge in graph.edges:
        if edge.get("type") != "has_property":
            continue
        property_id = str(edge.get("target") or "")
        material_ids = materials_mentioned_by(graph, str(edge.get("source") or ""))
        property_to_materials.setdefault(property_id, set()).update(material_ids)
    material_methods: dict[str, set[str]] = {}
    for edge in graph.edges:
        if edge.get("type") != "uses_method":
            continue
        material_ids = materials_mentioned_by(graph, str(edge.get("source") or ""))
        method_id = str(edge.get("target") or "")
        for material_id in material_ids:
            material_methods.setdefault(material_id, set()).add(method_id)
    analogies: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for property_id, materials in property_to_materials.items():
        material_list = sorted(materials)
        property_node = graph.node(property_id)
        for index, material_a in enumerate(material_list):
            for material_b in material_list[index + 1 :]:
                for method_id in sorted(material_methods.get(material_a, set())):
                    if method_id in material_methods.get(material_b, set()):
                        continue
                    key = (material_a, material_b, method_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    analogies.append(
                        {
                            "source_material": _label_of(graph, material_a),
                            "target_material": _label_of(graph, material_b),
                            "method": _label_of(graph, method_id),
                            "shared_property": str((property_node or {}).get("label") or property_id),
                        }
                    )
    return analogies


def constraint_list(graph: KnowledgeGraph | None) -> list[dict[str, Any]]:
    """Constraint matching：从 wiki_topic 节点的 limitations 收集已知限制，供生成时过滤。
    过滤过长条目（02 模块可能把论文摘要误存入 limitations），只保留像限制的短句。"""
    if graph is None or graph.is_empty():
        return []
    constraints: list[dict[str, Any]] = []
    seen: set[str] = set()
    for node in graph.nodes_of_type("wiki_topic"):
        wiki = node.get("wiki") or {}
        topic = str((wiki.get("title") or node.get("label") or node.get("id"))).strip()
        for limitation in list(wiki.get("limitations") or []):
            text = str(limitation).strip()
            if not text or len(text) > 300:
                continue
            if text in seen:
                continue
            seen.add(text)
            constraints.append({"topic": topic, "limitation": text})
    return constraints


def feasibility_map(graph: KnowledgeGraph | None, question: dict[str, Any]) -> dict[str, Any]:
    """Feasibility mapping：候选方向提到的实体在图语料中是否存在对应节点。"""
    if graph is None or graph.is_empty():
        return {"loaded": False, "known": [], "unknown": []}
    texts = [
        str(question.get("title") or ""),
        str(question.get("description") or ""),
        *[str(item) for item in (question.get("variables") or [])],
        *[str(item) for item in (question.get("candidates") or [])],
    ]
    matched = graph.match_entities(texts)
    return {
        "loaded": True,
        "known": [item for item in matched],
        "unknown": _unknown_terms(texts, matched),
    }


def graph_novelty(graph: KnowledgeGraph | None, route: dict[str, Any]) -> dict[str, Any]:
    """确定性图新颖性评分：路线中的实体组合在图中的最短路径越短，越可能已被探索。

    - 直接相连（1 跳）：已被明确记录 ⇒ 低
    - 2 跳（通常经 paper 相连）：同一文献已组合 ⇒ 中
    - 3 跳及以上：间接关联 ⇒ 高（潜在新组合）
    - 无路径：两个实体从未同现 ⇒ 高（gap 信号）
    - 实体不在图中：语料未覆盖 ⇒ 高（标注"超出语料"）
    """
    if graph is None or graph.is_empty():
        return {"level": "高", "reason": "知识图谱未加载，无法排除已探索组合", "matched_entities": [], "pairs": []}
    texts = [
        str(route.get("title") or ""),
        str(route.get("rationale") or ""),
        *[str(item) for item in (route.get("candidates") or [])],
        *[str(item) for item in (route.get("variables") or [])],
    ]
    matched = graph.match_entities(texts)
    pairs: list[dict[str, Any]] = []
    for index, entity_a in enumerate(matched):
        for entity_b in matched[index + 1 :]:
            if entity_a["type"] == entity_b["type"]:
                continue
            path = graph.shortest_path(entity_a["id"], entity_b["id"])
            pairs.append(
                {
                    "from": entity_a["label"],
                    "to": entity_b["label"],
                    "path_length": len(path) - 1 if path else None,
                    "path": path,
                }
            )
    if not pairs:
        if matched:
            return {
                "level": "高",
                "reason": "路线只涉及单一类实体或实体间无对比组合，图语料无法证明该组合已被探索",
                "matched_entities": matched,
                "pairs": [],
            }
        return {
            "level": "高",
            "reason": "路线未匹配到图中的材料/方法/性能实体（超出语料），无法排除已探索组合",
            "matched_entities": [],
            "pairs": [],
        }
    distances = [pair["path_length"] for pair in pairs if pair["path_length"] is not None]
    unreachable = [pair for pair in pairs if pair["path_length"] is None]
    if unreachable:
        level = "高"
        pair_texts = "、".join(
            str(pair.get("from") or "") + "×" + str(pair.get("to") or "") for pair in unreachable[:3]
        )
        reason = f"存在图中从未同现的实体对：{pair_texts}"
    elif distances and min(distances) <= 1:
        level = "低"
        reason = "核心实体组合在图中有直接关系，已被明确记录"
    elif distances and min(distances) <= 2:
        level = "中"
        reason = "核心实体组合经同一文献相连，属于已出现的组合"
    else:
        level = "高"
        reason = "核心实体组合仅存在间接关联，潜在新组合"
    return {"level": level, "reason": reason, "matched_entities": matched, "pairs": pairs}


def materials_mentioned_by(graph: KnowledgeGraph, paper_id: str) -> list[str]:
    """某 paper 通过 mentions 边关联的全部 material 节点 id。"""
    result: list[str] = []
    for edge in graph.edges:
        if edge.get("type") != "mentions":
            continue
        if str(edge.get("source") or "") != paper_id:
            continue
        target = str(edge.get("target") or "")
        node = graph.nodes_by_id.get(target) or {}
        if node.get("type") == "material" and target not in result:
            result.append(target)
    return result


def _label_of(graph: KnowledgeGraph, node_id: str) -> str:
    node = graph.node(node_id)
    return str((node or {}).get("label") or node_id)


def _unknown_terms(texts: list[str], matched: list[dict[str, Any]]) -> list[str]:
    """粗略提示：问题文本中未被图覆盖的长词（>4 字符，排除通用词）。"""
    known_labels = {str(item["label"]).lower() for item in matched}
    generic = {
        "研究", "影响", "机制", "性能", "体系", "材料", "结构", "方法",
        "计算", "实验", "验证", "分析",
    }
    unknown: list[str] = []
    for text in texts:
        for token in text.replace(",", " ").replace(";", " ").split():
            word = token.strip()
            if len(word) < 5 or len(word) > 40 or word.lower() in known_labels:
                continue
            if any(known in word.lower() for known in known_labels):
                continue
            if word in unknown or word in generic:
                continue
            unknown.append(word)
    return unknown[:12]

from lit_agent.route_candidate_bridge import _load_route_candidate_agent


def _graph_module():
    _load_route_candidate_agent()
    import route_candidate_agent.graph as module

    return module


def make_synthetic_graph():
    """合成图：
    paper:1 mentions A、B；uses_method X；has_property P
    paper:2 mentions A；uses_method Y；has_property P
    """
    module = _graph_module()
    data = {
        "nodes": [
            {"id": "paper:1", "type": "paper", "label": "paper one"},
            {"id": "paper:2", "type": "paper", "label": "paper two"},
            {"id": "material:A", "type": "material", "label": "Alpha"},
            {"id": "material:B", "type": "material", "label": "Beta"},
            {"id": "method:X", "type": "method", "label": "X-method"},
            {"id": "method:Y", "type": "method", "label": "Y-method"},
            {"id": "property:P", "type": "property", "label": "coercivity"},
        ],
        "edges": [
            {"id": "e1", "source": "paper:1", "target": "material:A", "type": "mentions", "evidence_ids": ["p:1:e1"]},
            {"id": "e2", "source": "paper:1", "target": "material:B", "type": "mentions", "evidence_ids": ["p:1:e2"]},
            {"id": "e3", "source": "paper:2", "target": "material:A", "type": "mentions", "evidence_ids": ["p:2:e1"]},
            {"id": "e4", "source": "paper:1", "target": "method:X", "type": "uses_method", "evidence_ids": ["p:1:e3"]},
            {"id": "e5", "source": "paper:2", "target": "method:Y", "type": "uses_method", "evidence_ids": ["p:2:e2"]},
            {"id": "e6", "source": "paper:1", "target": "property:P", "type": "has_property", "evidence_ids": []},
            {"id": "e7", "source": "paper:2", "target": "property:P", "type": "has_property", "evidence_ids": []},
        ],
    }
    return module.KnowledgeGraph(data)


def test_gap_candidates_detects_unused_methods() -> None:
    module = _graph_module()
    graph = make_synthetic_graph()
    gaps = module.gap_candidates(graph)
    # method:X 已用于 A、B；method:Y 只用于 A → 唯一 gap：Y 尚未用于 B
    assert len(gaps) == 1
    gap = gaps[0]
    assert gap["method"] == "Y-method"
    assert gap["material"] == "Beta"
    assert gap["used_on"]
    assert any(item["material"] == "Alpha" for item in gap["used_on"])


def test_analogy_candidates_transfers_method_between_materials() -> None:
    module = _graph_module()
    graph = make_synthetic_graph()
    analogies = module.analogy_candidates(graph)
    # A、B 共享 property:P；A 用 Y 而 B 不用 → 类比：Y 可迁移到 B
    assert analogies
    target = [item for item in analogies if item["target_material"] == "Beta"]
    assert target
    assert target[0]["method"] == "Y-method"
    assert target[0]["source_material"] == "Alpha"
    assert target[0]["shared_property"] == "coercivity"


def test_graph_novelty_uses_path_distances() -> None:
    module = _graph_module()
    graph = make_synthetic_graph()
    # A 与 X 经 paper:1 相连（2 跳）→ 中
    route_known = {"title": "Alpha 用 X-method", "rationale": "", "candidates": ["Alpha"], "variables": []}
    novelty_known = module.graph_novelty(graph, route_known)
    assert novelty_known["level"] == "中"
    # B 与 Y 无路径 → 高
    route_gap = {"title": "Beta 用 Y-method", "rationale": "", "candidates": ["Beta"], "variables": []}
    novelty_gap = module.graph_novelty(graph, route_gap)
    assert novelty_gap["level"] == "高"
    # 图中不存在的实体 → 高
    route_outside = {"title": "Gamma 用 X-method", "rationale": "", "candidates": ["Gamma"], "variables": []}
    novelty_outside = module.graph_novelty(graph, route_outside)
    assert novelty_outside["level"] == "高"


def test_graph_novelty_on_empty_graph() -> None:
    module = _graph_module()
    result = module.graph_novelty(None, {"title": "任意路线", "rationale": "", "candidates": [], "variables": []})
    assert result["level"] == "高"
    assert "未加载" in result["reason"]


def test_graph_context_collects_neighbor_evidence() -> None:
    module = _graph_module()
    graph = make_synthetic_graph()
    question = {"title": "Alpha 的 coercivity 问题", "description": "", "variables": [], "candidates": []}
    context = module.graph_context(graph, question)
    assert context["loaded"]
    matched_labels = {item["label"] for item in context["matched_entities"]}
    assert "Alpha" in matched_labels
    assert "coercivity" in matched_labels
    # 邻居边上的证据应被收集
    assert any("p:1:e" in eid or "p:2:e" in eid for eid in context["evidence_ids"])

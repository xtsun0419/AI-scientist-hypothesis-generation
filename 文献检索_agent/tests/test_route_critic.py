"""P2：Critic + Elo Arena + 一轮演化的单元测试。"""

from pathlib import Path

from lit_agent.route_candidate_bridge import _load_route_candidate_agent


def _critic_module():
    _load_route_candidate_agent()
    import route_candidate_agent.critic as module

    return module


def make_run_with_snapshot() -> dict:
    """带生成上下文快照的合成 run（模拟 generate 的输出结构）。"""
    return {
        "id": "test-run-1",
        "created_at": "2026-08-20T00:00:00+00:00",
        "question_id": 1,
        "question_title": "测试问题",
        "question_source": "01 文献获取",
        "emphasis": "偏向可验证路线",
        "metadata": {"mode": "fallback_no_api_key"},
        "snapshot": {
            "question": {
                "id": 1,
                "title": "测试问题",
                "source": "01 文献获取",
                "description": "",
                "variables": [],
                "validation_criteria": [],
                "evidence_ids": ["p:1:e1"],
            },
            "evidence_pool": ["p:1:e1", "p:1:e2"],
            "graph": {
                "loaded": True,
                "matched_entities": [{"id": "material:ndfeb", "label": "NdFeB"}],
                "related_nodes": [],
                "related_evidence_ids": ["p:1:e1"],
                "gap_candidates": [],
                "analogy_candidates": [],
                "constraints": [{"topic": "NdFeB", "limitation": "晶界相薄于阈值时矫顽力骤降"}],
                "feasibility": {"loaded": True, "known": [], "unknown": []},
            },
            "emphasis": "偏向可验证路线",
        },
        "routes": [
            {
                "rank": 1,
                "title": "路线一",
                "rationale": "机制优先",
                "candidates": ["NdFeB", "晶界相"],
                "variables": ["晶界相厚度"],
                "validation": ["对比有/无晶界相的性能阈值"],
                "evidence": ["p:1:e1 支撑该机制", "纯推测内容"],
                "risks": [],
                "priority": "高",
                "next_step": "",
                "evidence_annotations": [
                    {"text": "p:1:e1 支撑该机制", "kind": "证据支撑", "matched_ids": ["p:1:e1"]},
                    {"text": "纯推测内容", "kind": "推测", "matched_ids": []},
                ],
                "graph_novelty": {"level": "高", "reason": "组合未共现", "matched_entities": []},
            },
            {
                "rank": 2,
                "title": "路线二",
                "rationale": "候选筛选",
                "candidates": ["FeNi"],
                "variables": ["成分"],
                "validation": [],
                "evidence": [],
                "risks": [],
                "priority": "中",
                "next_step": "",
                "evidence_annotations": [],
                "graph_novelty": {"level": "低", "reason": "图中已有直接路径", "matched_entities": []},
            },
        ],
    }


def test_fallback_critique_scores_dimensions() -> None:
    module = _critic_module()
    result = module.fallback_critique(make_run_with_snapshot())
    assert result["mode"] == "fallback"
    assert len(result["routes"]) == 2
    route1 = result["routes"][0]
    dims = route1["dimensions"]
    # 证据：无独立检索证据 → 声称引用 p:1:e1 视为未验证（防幻觉传播）→ 降级为 2 分
    assert dims["evidence"]["score"] == 2
    assert "幻觉风险" in dims["evidence"]["reason"]
    assert any("幻觉引用" in w for w in route1["weaknesses"])
    # 可证伪：validation 含“对比”与“阈值” → 5 分
    assert dims["falsifiability"]["score"] == 5
    # 新颖性：图新颖性高 → 5 分
    assert dims["novelty"]["score"] == 5
    # 可行性：candidates 命中已知限制（晶界相）→ 2 分
    assert dims["feasibility"]["score"] == 2
    route2 = result["routes"][1]
    dims2 = route2["dimensions"]
    assert dims2["evidence"]["score"] == 1
    assert dims2["falsifiability"]["score"] == 1
    assert dims2["novelty"]["score"] == 2
    assert dims2["feasibility"]["score"] == 4


def test_fallback_critique_independent_verification_restores_score() -> None:
    """独立检索证据能复现声称引用时，证据分不再被降级（防幻觉传播机制的两向性）。"""
    module = _critic_module()
    independent = {1: {"queries": ["NdFeB"], "hits": [], "hit_evidence_ids": ["p:1:e1"]}}
    result = module.fallback_critique(make_run_with_snapshot(), independent)
    route1 = result["routes"][0]
    dims = route1["dimensions"]
    # 声称引用 p:1:e1 被独立检索复现 → 恢复 3 分（1/2 条支撑），无幻觉风险标注
    assert dims["evidence"]["score"] == 3
    assert "幻觉风险" not in dims["evidence"]["reason"]
    assert not any("幻觉引用" in w for w in route1["weaknesses"])


def test_elo_arena_deterministic_rankings() -> None:
    module = _critic_module()
    run = make_run_with_snapshot()
    critique = module.fallback_critique(run)
    arena = module.elo_arena(run, critique, settings=None)
    # n-1 场相邻对决
    assert len(arena["battles"]) == 1
    assert arena["arena_mode"] == "deterministic"
    # 总分高的路线 1 排第一，且胜者 Elo 上升
    rankings = arena["rankings"]
    assert rankings[0]["rank"] == 1
    assert arena["ratings"]["1"] > 1000
    assert arena["ratings"]["2"] < arena["ratings"]["1"]


def test_critique_routes_no_settings_uses_fallback() -> None:
    module = _critic_module()
    result = module.critique_routes(make_run_with_snapshot(), settings=None)
    assert result["mode"] == "fallback"
    assert all(item.get("dimensions") for item in result["routes"])


def test_fallback_evolve_keeps_lineage_semantics() -> None:
    module = _critic_module()
    run = make_run_with_snapshot()
    critique = module.fallback_critique(run)
    for route in run["routes"]:
        route["critique"] = next(
            (item for item in critique["routes"] if item["rank"] == route["rank"]), {}
        )
    evolved = module.fallback_evolve(run)
    assert len(evolved) == 2
    assert evolved[0]["title"].startswith("v2·")
    # 改进建议被并入 rationale / next_step
    assert "v2" in evolved[0]["rationale"]


def test_apply_critique_writes_back_to_routes(tmp_path: Path) -> None:
    RouteCandidateAgent = _load_route_candidate_agent()
    agent = RouteCandidateAgent(output_path=tmp_path / "route_candidates.json", settings=None)
    run = make_run_with_snapshot()
    run = agent._apply_critique(run)
    assert run["critique"]["mode"] == "fallback"
    assert run["critique"]["arena"]["rankings"]
    for route in run["routes"]:
        assert "critique" in route
        assert route["critique"]["dimensions"]
        assert route["elo_rating"] is not None
        assert route["elo_rank"] is not None


def test_generate_persists_snapshot_and_critique(tmp_path: Path) -> None:
    RouteCandidateAgent = _load_route_candidate_agent()
    agent = RouteCandidateAgent(output_path=tmp_path / "route_candidates.json", settings=None)
    state = agent.state()
    question_id = state["questions"][0]["id"]

    result = agent.generate(question_id=question_id, route_count=2, emphasis="偏向可验证路线")
    run = result["latest_run"]
    snapshot = run["snapshot"]
    assert snapshot["question"]["id"] == question_id
    assert snapshot["evidence_pool"]
    assert "graph" in snapshot
    assert run["critique"]["mode"] == "fallback"
    for route in run["routes"]:
        assert route["critique"]["dimensions"]


def test_generate_without_critique_skips_critic(tmp_path: Path) -> None:
    RouteCandidateAgent = _load_route_candidate_agent()
    agent = RouteCandidateAgent(output_path=tmp_path / "route_candidates.json", settings=None)
    state = agent.state()
    question_id = state["questions"][0]["id"]

    result = agent.generate(question_id=question_id, route_count=1, with_critique=False)
    run = result["latest_run"]
    assert "critique" not in run
    assert "critique" not in run["routes"][0]
    # 后续可手动补批判
    result2 = agent.critique()
    assert result2["latest_run"]["critique"]["mode"] == "fallback"


def test_critique_and_evolve_post_actions(monkeypatch) -> None:
    from lit_agent.web_v3 import V3WebHandler

    for action in ("/route-candidates/critique", "/route-candidates/evolve"):
        redirects: list[str] = []

        class FakeHandler:
            path = action

            def _read_form(self) -> dict[str, str]:
                return {}

            def _redirect(self, location: str) -> None:
                redirects.append(location)

            def send_error(self, code: int) -> None:
                raise AssertionError(f"unexpected error response: {code}")

        fake_result = {
            "latest_run": {
                "question_id": 1,
                "critique": {"mode": "fallback"},
                "routes": [{"rank": 1}],
            }
        }
        monkeypatch.setattr("lit_agent.web_v3.route_candidate_critique", lambda **kwargs: fake_result)
        monkeypatch.setattr("lit_agent.web_v3.route_candidate_evolve", lambda **kwargs: fake_result)

        V3WebHandler.do_POST(FakeHandler())

        assert redirects
        assert redirects[0].startswith("/route-candidates?message=")


def test_route_arena_block_renders() -> None:
    from lit_agent.web_v3 import route_arena_block

    html = route_arena_block(
        {
            "critique": {
                "mode": "fallback",
                "created_at": "2026-08-20T00:00:00+00:00",
                "arena": {
                    "battles": [{"a": 1, "b": 2, "winner": 1, "reason": "证据更充分", "mode": "deterministic"}],
                    "rankings": [{"rank": 1, "elo": 1280.0}, {"rank": 2, "elo": 1232.0}],
                },
            }
        }
    )
    assert "Elo 排名" in html
    assert "胜者路线1" in html
    assert "证据更充分" in html

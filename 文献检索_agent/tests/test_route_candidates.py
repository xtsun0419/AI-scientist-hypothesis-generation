from pathlib import Path

from lit_agent.route_candidate_bridge import _load_route_candidate_agent
from lit_agent.web_v3 import V3WebHandler, render_route_candidates


def test_route_candidate_agent_generates_fallback_routes(tmp_path: Path) -> None:
    RouteCandidateAgent = _load_route_candidate_agent()
    output_path = tmp_path / "route_candidates.json"
    agent = RouteCandidateAgent(output_path=output_path, settings=None)

    state = agent.state()
    assert state["questions"]

    result = agent.generate(
        question_id=state["questions"][0]["id"],
        route_count=2,
        emphasis="偏向可验证路线",
    )

    assert result["metrics"]["routes"] == 2
    assert result["latest_run"]["metadata"]["mode"] == "fallback_no_api_key"
    assert output_path.exists()


def test_route_candidate_page_renders() -> None:
    html = render_route_candidates()
    assert "提出路线 / 候选" in html
    assert "第 1 模块提出的问题" in html
    assert "生成候选路线" in html


def test_route_candidate_post_redirects_after_generation(monkeypatch) -> None:
    redirects: list[str] = []

    class FakeHandler:
        path = "/route-candidates/generate"

        def _read_form(self) -> dict[str, str]:
            return {"question_id": "5", "route_count": "2", "emphasis": "偏向可验证路线"}

        def _redirect(self, location: str) -> None:
            redirects.append(location)

        def send_error(self, code: int) -> None:
            raise AssertionError(f"unexpected error response: {code}")

    monkeypatch.setattr(
        "lit_agent.web_v3.route_candidate_generate",
        lambda **kwargs: {
            "metrics": {"routes": 2},
            "selected_question": {"id": kwargs["question_id"], "title": "测试科学问题"},
        },
    )

    V3WebHandler.do_POST(FakeHandler())

    assert redirects
    assert redirects[0].startswith("/route-candidates?message=")
    assert "question_id=5" in redirects[0]

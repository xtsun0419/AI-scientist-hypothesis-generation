from __future__ import annotations

import html
import json
import os
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .analysis_bridge import (
    analysis_agent_dir,
    analysis_data_dir,
    analysis_graph_path,
    analysis_parsed_dir,
    build_all_with_analysis_agent,
    conversion_metrics,
    search_with_analysis_agent,
)
from .agents import OrchestratorAgent
from .config import default_db_path, default_pdf_dir, default_report_dir
from .db import LiteratureDB
from .env import load_local_env
from .external_llm import (
    active_llm_api,
    default_external_llm_config_path,
    load_llm_api_config,
    save_llm_api_from_form,
)
from .llm import LLMSettings
from .question_synthesis_bridge import (
    question_synthesis_agent_dir,
    question_synthesis_chat,
    question_synthesis_data_dir,
    question_synthesis_reset,
    question_synthesis_state,
)
from .route_candidate_bridge import (
    route_candidate_agent_dir,
    route_candidate_data_dir,
    route_candidate_generate,
    route_candidate_state,
)

AI_WORKFLOW_MODULES = [
    {
        "path": "/literature",
        "stage": "01",
        "title": "文献获取",
        "status": "已接入",
        "summary": "围绕科学问题做小批量检索、人工确认、PDF 获取、手动 DOI 清单与下一轮建议。",
        "metrics": ("目标问题", "轮次", "PDF"),
        "enabled": True,
    },
    {
        "path": "/paper-analysis",
        "stage": "02",
        "title": "分析文献",
        "status": "已接入",
        "summary": "将 PDF 转成结构化语料，并生成 RAG 索引、Paper Cards、轻量图谱和证据化 Wiki。",
        "metrics": ("PDF", "卡片", "Wiki"),
        "enabled": True,
    },
    {
        "path": "/question-synthesis",
        "stage": "03",
        "title": "科学问题归纳",
        "status": "已接入",
        "summary": "以对话方式读取检索问题和文献分析证据，逐步收敛可研究、可验证的科学问题与细致方向。",
        "metrics": ("问题", "假设", "缺口"),
        "enabled": True,
    },
    {
        "path": "/route-candidates",
        "stage": "04",
        "title": "提出路线 / 候选",
        "status": "已接入",
        "summary": "从科学问题与归纳方向生成多条可行路线，整理候选体系、变量空间、验证方式和优先级。",
        "metrics": ("路线", "候选", "变量"),
        "enabled": True,
    },
    {
        "path": "/planning",
        "stage": "05",
        "title": "计算 / 实验规划",
        "status": "待建设",
        "summary": "生成计算任务、实验矩阵、约束条件和预期观测指标。",
        "metrics": ("任务", "参数", "指标"),
        "enabled": False,
    },
    {
        "path": "/feedback",
        "stage": "06",
        "title": "结果反馈",
        "status": "待建设",
        "summary": "接收计算/实验结果，更新证据、修正路线，并触发下一轮问题驱动探索。",
        "metrics": ("结果", "更新", "下一轮"),
        "enabled": False,
    },
]
MODULE_PAGES = {item["path"]: item for item in AI_WORKFLOW_MODULES if item["path"] != "/literature"}


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    load_local_env()
    server = ThreadingHTTPServer((host, port), V3WebHandler)
    print(f"v3 web UI: http://{host}:{port}")
    server.serve_forever()


class V3WebHandler(BaseHTTPRequestHandler):
    def do_HEAD(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path.startswith("/static/dashboard"):
            self._redirect("/reports/dashboard.html")
            return
        if self.path.startswith("/reports/"):
            self._serve_report_file()
            return
        if self.path.startswith("/parsed/"):
            self._serve_parsed_file()
            return
        if self.path.startswith("/analysis-artifacts/"):
            self._serve_analysis_artifact_file()
            return
        path = urllib.parse.urlparse(self.path).path.rstrip("/") or "/"
        if path == "/":
            self._send_html(render_ai_home(message=self._query_param("message"), error=self._query_param("error")))
            return
        if path == "/literature":
            self._send_html(render_home(message=self._query_param("message")))
            return
        if path == "/paper-analysis":
            self._send_html(
                render_paper_analysis(
                    message=self._query_param("message"),
                    error=self._query_param("error"),
                    query=self._query_param("q") or "",
                )
            )
            return
        if path == "/question-synthesis":
            self._send_html(
                render_question_synthesis(
                    message=self._query_param("message"),
                    error=self._query_param("error"),
                )
            )
            return
        if path == "/route-candidates":
            self._send_html(
                render_route_candidates(
                    message=self._query_param("message"),
                    error=self._query_param("error"),
                    selected_question_id=_int(self._query_param("question_id"), None),
                )
            )
            return
        module = MODULE_PAGES.get(path)
        if module:
            self._send_html(render_empty_module(module))
            return
        self.send_error(404)

    def do_POST(self) -> None:
        fields = self._read_form()
        action = self.path.rstrip("/")
        orch = OrchestratorAgent()
        try:
            if action == "/llm-api/save":
                config = save_llm_api_from_form(fields)
                active = active_llm_api(config)
                active_name = active.get("name") if active else None
                detail = f"当前启用：{active_name}" if active_name else "已保存，尚未启用"
                self._redirect(_ai_home_message_location(f"外部 LLM API 已保存。{detail}"))
                return
            if action == "/goal/create":
                goal_id = orch.create_goal(
                    title=_required(fields, "title"),
                    description=fields.get("description") or None,
                    target_count=_int(fields.get("target_count"), 20),
                )
                self._redirect(_message_location(f"科学问题已创建：{goal_id}"))
                return
            if action == "/goal/delete":
                goal_id = _int_required(fields, "goal_id")
                orch.delete_goal(goal_id)
                self._redirect(_message_location(f"科学问题已删除：{goal_id}"))
                return
            if action == "/round/plan":
                result = orch.plan_round(
                    goal_id=_int_required(fields, "goal_id"),
                    target_count=_int(fields.get("target_count"), None),
                    query_limit=_int(fields.get("query_limit"), 4) or 4,
                    max_results_per_query=_int(fields.get("max_results_per_query"), 8) or 8,
                )
                self._redirect(_message_location(f"轮次已规划：{result['round_id']}"))
                return
            if action == "/round/approve":
                orch.approve_round(_int_required(fields, "round_id"))
                self._redirect(_message_location("本轮已确认"))
                return
            if action == "/round/acquire":
                result = orch.acquire_round(_int_required(fields, "round_id"))
                self._redirect(
                    _message_location(
                        f"本次新增下载 {result['downloaded']} 个 PDF；本轮阅读文件夹已有 PDF {result.get('round_downloaded', result['downloaded'])} 个；"
                        f"手动下载任务 {result['manual_tasks']} 条；文件夹：{result.get('round_pdf_dir', '')}"
                    )
                )
                return
            if action == "/round/intake":
                count = orch.intake_manual_round(_int_required(fields, "round_id"))
                self._redirect(_message_location(f"已绑定手动 PDF：{count}"))
                return
            if action == "/round/analyze":
                count = orch.analyze_round(_int_required(fields, "round_id"))
                self._redirect(_message_location(f"已生成本轮分析：{count}"))
                return
            if action == "/round/propose-next":
                result = orch.propose_next_round(_int_required(fields, "round_id"))
                self._redirect(_message_location(f"已生成下一轮 query：{len(result['next_queries'])} 条"))
                return
            if action == "/paper-analysis/convert":
                result = orch.convert_pdfs(
                    round_id=_int(fields.get("round_id"), None),
                    limit=_int(fields.get("limit"), None),
                    force=fields.get("force") == "1",
                )
                self._redirect(
                    _paper_analysis_message_location(
                        f"PDF 转换完成：总计 {result['total']}，新增/重转 {result['converted']}，跳过 {result['skipped']}，失败 {result['failed']}"
                    )
                )
                return
            if action == "/paper-analysis/build-all":
                result = build_all_with_analysis_agent()
                self._redirect(
                    _paper_analysis_message_location(
                        "知识库构建完成："
                        f"chunks {result.get('index_chunks', 0)}，cards {result.get('cards', 0)}，"
                        f"nodes {result.get('nodes', 0)}，wiki {result.get('wiki_pages', 0)}"
                    )
                )
                return
            if action == "/question-synthesis/chat":
                question_synthesis_chat(_required(fields, "message"))
                self._redirect(_question_synthesis_message_location("已收到，LLM 已基于当前证据继续归纳。"))
                return
            if action == "/question-synthesis/reset":
                question_synthesis_reset()
                self._redirect(_question_synthesis_message_location("对话已重置，并重新载入检索问题与文献分析结果。"))
                return
            if action == "/route-candidates/generate":
                result = route_candidate_generate(
                    question_id=_int(fields.get("question_id"), None),
                    route_count=_int(fields.get("route_count"), 3) or 3,
                    emphasis=fields.get("emphasis", ""),
                )
                routes = int((result.get("metrics") or {}).get("routes") or 0)
                selected = result.get("selected_question") or {}
                self._redirect(
                    _route_candidates_message_location(
                        f"已生成 {routes} 条候选路线：{selected.get('title', '')}",
                        question_id=selected.get("id"),
                    )
                )
                return
        except Exception as exc:
            if action.startswith("/llm-api/"):
                self._redirect(_ai_home_message_location(_friendly_error(str(exc)), error=True))
                return
            if action.startswith("/paper-analysis/"):
                self._redirect(_paper_analysis_message_location(_friendly_error(str(exc)), error=True))
                return
            if action.startswith("/question-synthesis/"):
                self._redirect(_question_synthesis_message_location(_friendly_error(str(exc)), error=True))
                return
            if action.startswith("/route-candidates/"):
                self._redirect(_route_candidates_message_location(_friendly_error(str(exc)), error=True))
                return
            self._send_html(render_home(error=_friendly_error(str(exc))))
            return
        self.send_error(404)

    def _read_form(self) -> dict[str, str]:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        parsed = urllib.parse.parse_qs(body)
        return {key: values[-1] for key, values in parsed.items()}

    def _query_param(self, name: str) -> str | None:
        parsed = urllib.parse.urlparse(self.path)
        values = urllib.parse.parse_qs(parsed.query).get(name)
        return values[-1] if values else None

    def _send_html(self, content: str) -> None:
        payload = content.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _redirect(self, location: str) -> None:
        self.send_response(303)
        self.send_header("Location", location)
        self.end_headers()

    def _serve_report_file(self) -> None:
        rel = self.path.removeprefix("/reports/").split("?", 1)[0]
        path = (default_report_dir() / rel).resolve()
        if default_report_dir().resolve() not in path.parents and path != default_report_dir().resolve():
            self.send_error(403)
            return
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        content_type = "text/html; charset=utf-8" if path.suffix == ".html" else "text/plain; charset=utf-8"
        payload = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _serve_parsed_file(self) -> None:
        rel = self.path.removeprefix("/parsed/").split("?", 1)[0]
        path = (analysis_parsed_dir() / rel).resolve()
        if analysis_parsed_dir().resolve() not in path.parents and path != analysis_parsed_dir().resolve():
            self.send_error(403)
            return
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        content_type = "application/json; charset=utf-8" if path.suffix == ".json" else "text/markdown; charset=utf-8"
        payload = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _serve_analysis_artifact_file(self) -> None:
        rel = self.path.removeprefix("/analysis-artifacts/").split("?", 1)[0]
        root = analysis_data_dir().resolve()
        path = (root / rel).resolve()
        if root not in path.parents and path != root:
            self.send_error(403)
            return
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        content_type = "application/json; charset=utf-8" if path.suffix == ".json" else "text/markdown; charset=utf-8"
        payload = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def render_ai_home(*, message: str | None = None, error: str | None = None) -> str:
    state = load_state()
    metrics = state["metrics"]
    llm_config = load_llm_api_config()
    active_llm = active_llm_api(llm_config)
    llm_status = f"已配置：{active_llm['name']}" if active_llm else state["selection_mode"]
    buttons = "".join(home_module_button(item) for item in AI_WORKFLOW_MODULES)
    worklog = "".join(
        f"<li><b>{html.escape(title)}</b><span>{html.escape(detail)}</span></li>"
        for title, detail in [
            ("文献获取 v3", "科学问题、轮次规划、人工确认、PDF 获取和下一轮建议已接入。"),
            ("外部 LLM 推荐", f"{llm_status}。"),
            ("独立文献文件夹", "每个科学问题/轮次会形成自己的 PDF 阅读文件夹。"),
            ("手动 DOI 清单", "下载失败或非开放文献会生成 DOI、出版商页面和放置目录。"),
            ("本地数据", f"当前已有 {metrics['goals']} 个科学问题、{metrics['rounds']} 个轮次、{metrics['downloaded']} 个本轮可用 PDF。"),
        ]
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Scientist 工作流</title>
  <style>{base_css()}</style>
</head>
<body class="ai-home-body">
  <input class="sidebar-toggle" id="llm-toggle" type="checkbox">
  <label class="llm-button" for="llm-toggle">LLM API</label>
  {llm_api_panel(llm_config)}
  <input class="sidebar-toggle" id="worklog-toggle" type="checkbox">
  <label class="worklog-button" for="worklog-toggle">工作记录</label>
  <aside class="worklog-panel" aria-label="当前完成的工作">
    <label class="worklog-close" for="worklog-toggle">隐藏</label>
    <div class="brand-kicker">Current Build</div>
    <h2>已完成的工作</h2>
    <ul>{worklog}</ul>
    <a class="button secondary" href="/reports/dashboard.html">静态仪表盘</a>
  </aside>
  <main class="home-minimal">
    {notice(message, "ok") if message else ""}
    {notice(error, "err") if error else ""}
    <div class="home-brand">
      <span>AI SCIENTIST WORKFLOW</span>
      <b>Literature · Analysis · Hypothesis · Route · Plan · Feedback</b>
    </div>
    <nav class="home-button-grid" aria-label="AI Scientist 工作流入口">
      {buttons}
    </nav>
  </main>
</body>
</html>"""


def render_empty_module(module: dict[str, Any]) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(module["title"])} · AI Scientist</title>
  <style>{base_css()}</style>
</head>
<body>
  <header>
    <div class="topbar">
      <div>
        <div class="brand-kicker">AI Scientist Module</div>
        <h1>{html.escape(module["title"])}</h1>
        <div class="status-line">
          <span class="badge badge-neutral">尚未建设</span>
          <span class="subtle">Stage {html.escape(module["stage"])}</span>
        </div>
      </div>
      <div class="header-actions">
        <a class="button secondary" href="/">返回首页</a>
        <a class="button secondary" href="/literature">文献获取</a>
      </div>
    </div>
  </header>
  <main class="home-shell">
    <section class="empty-module">
      <div class="brand-kicker">Reserved Workspace</div>
      <h2>{html.escape(module["title"])}</h2>
      <p class="subtle">{html.escape(module["summary"])}</p>
      <div class="empty-grid">
        <span>{html.escape(module["metrics"][0])}</span>
        <span>{html.escape(module["metrics"][1])}</span>
        <span>{html.escape(module["metrics"][2])}</span>
      </div>
    </section>
  </main>
</body>
</html>"""


def render_route_candidates(
    *,
    message: str | None = None,
    error: str | None = None,
    selected_question_id: int | None = None,
) -> str:
    try:
        state = route_candidate_state(selected_question_id=selected_question_id)
        state_error = None
    except Exception as exc:
        state = _empty_route_candidate_state()
        state_error = _friendly_error(str(exc))
    metrics = state.get("metrics", {})
    context = state.get("context", {})
    selected = state.get("selected_question") or {}
    latest_run = state.get("latest_run") or {}
    routes = list(latest_run.get("routes") or [])
    model_name = state.get("model_name") or "LLM 未配置"
    llm_configured = bool(state.get("llm_configured"))
    selected_id = selected.get("id") or ""
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>提出路线 / 候选 · AI Scientist</title>
  <style>{base_css()}{route_candidate_css()}</style>
  <script>
    window.addEventListener('DOMContentLoaded', () => {{
      const busy = document.querySelector('[data-busy]');
      document.querySelectorAll('form[method="post"]').forEach((form) => {{
        form.addEventListener('submit', () => {{
          const button = form.querySelector('button[type="submit"], button:not([type])');
          if (busy && button) {{
            busy.textContent = button.dataset.busy || '正在生成候选路线。';
            busy.style.display = 'block';
          }}
          form.querySelectorAll('button').forEach((item) => {{
            item.disabled = true;
            if (!item.classList.contains('secondary')) item.textContent = '处理中...';
          }});
        }});
      }});
      const selector = document.querySelector('[data-question-selector]');
      if (selector) {{
        selector.addEventListener('change', () => {{
          const url = new URL(window.location.href);
          url.searchParams.set('question_id', selector.value);
          window.location.href = url.toString();
        }});
      }}
    }});
  </script>
</head>
<body>
  <header>
    <div class="topbar">
      <div>
        <div class="brand-kicker">Stage 04 · Route Candidates</div>
        <h1>提出路线 / 候选</h1>
        <div class="status-line">
          <span class="badge badge-teal">已接入</span>
          <span class="badge {'badge-blue' if llm_configured else 'badge-amber'}">LLM：{html.escape(str(model_name))}</span>
          <span class="subtle">模块：{html.escape(str(route_candidate_agent_dir()))}</span>
        </div>
      </div>
      <div class="header-actions">
        <a class="button secondary" href="/">返回首页</a>
        <a class="button secondary" href="/literature">文献获取</a>
        <a class="button secondary" href="/question-synthesis">科学问题归纳</a>
      </div>
    </div>
  </header>
  <main class="route-shell">
    <div class="notice busy full" data-busy></div>
    {notice(message, "ok") if message else ""}
    {notice(error or state_error, "err") if (error or state_error) else ""}
    <section class="full hero-strip route-hero">
      <div>
        <div class="brand-kicker">Selected Research Question</div>
        <h2>{html.escape(str(selected.get("title") or "暂无科学问题"))}</h2>
        {route_question_description(selected)}
      </div>
      <div class="hero-counters">
        <span><b>{html.escape(str(metrics.get("questions", 0)))}</b> questions</span>
        <span><b>{html.escape(str(metrics.get("routes", 0)))}</b> routes</span>
        <span><b>{html.escape(str(metrics.get("saved_runs", 0)))}</b> runs</span>
      </div>
    </section>
    <section class="full metrics">
      {metric_card("可选问题", metrics.get("questions", 0))}
      {metric_card("候选路线", metrics.get("routes", 0))}
      {metric_card("证据缺口", metrics.get("evidence_gaps", 0))}
      {metric_card("Paper Cards", metrics.get("paper_cards", 0))}
      {metric_card("Wiki 条目", metrics.get("wiki_pages", 0))}
      {metric_card("生成模式", "外部 LLM" if llm_configured else "本地草稿")}
    </section>
    <section class="route-workspace">
      <aside class="stack route-side">
        <section class="panel">
          <div class="panel-head"><h2>路线生成</h2>{small_id("问题", selected_id)}</div>
          <form method="post" action="/route-candidates/generate">
            <label>第 1 模块提出的问题</label>
            {route_question_select(state.get("questions", []), selected_id)}
            <label>生成路线数量</label>
            <input name="route_count" type="number" value="{html.escape(str(latest_run.get("route_count") or 3))}" min="1" max="8">
            <label>偏好补充</label>
            <textarea name="emphasis" placeholder="例如：偏向计算筛选 / 偏向可合成实验 / 重点看矫顽力">{html.escape(str(latest_run.get("emphasis") or ""))}</textarea>
            <button data-busy="正在根据 01 问题和 03 归纳方向生成候选路线。" type="submit">生成候选路线</button>
          </form>
        </section>
        <section class="panel">
          <div class="panel-head"><h2>03 归纳方向</h2></div>
          {route_direction_block(context)}
        </section>
        <section class="panel">
          <div class="panel-head"><h2>数据文件</h2></div>
          <code>{html.escape(str(state.get("output_path") or route_candidate_data_dir()))}</code>
        </section>
      </aside>
      <section class="stack route-main">
        <section class="panel">
          <div class="panel-head">
            <div>
              <h2>候选路线</h2>
              {route_run_meta(latest_run)}
            </div>
            <span class="badge {'badge-teal' if routes else 'badge-neutral'}">{html.escape(str(len(routes)))} 条</span>
          </div>
          {route_cards(routes)}
        </section>
        <section class="split">
          <div class="panel">
            <div class="panel-head"><h2>检索阶段输入</h2></div>
            {retrieval_context_block(context)}
          </div>
          <div class="panel">
            <div class="panel-head"><h2>分析证据摘要</h2></div>
            {analysis_context_block(context)}
          </div>
        </section>
        <section class="panel">
          <div class="panel-head"><h2>历史生成</h2></div>
          {route_run_table(state.get("all_runs", []))}
        </section>
      </section>
    </section>
  </main>
</body>
</html>"""


def route_question_description(selected: dict[str, Any]) -> str:
    if not selected:
        return '<p class="subtle">请先在 01 模块创建科学问题，或在 03 模块完成科学问题归纳。</p>'
    description = str(selected.get("description") or "").strip()
    source = str(selected.get("source") or "问题来源")
    detail = f"{source}"
    if description:
        detail += f" · {description}"
    return f'<p class="subtle">{html.escape(detail)}</p>'


def route_question_select(questions: list[dict[str, Any]], selected_id: object) -> str:
    if not questions:
        return '<select name="question_id" disabled><option>暂无科学问题</option></select>'
    options = []
    for item in questions:
        value = str(item.get("id") or "")
        selected = " selected" if str(selected_id) == value else ""
        label = f"{item.get('source') or '问题'} · {item.get('title') or ''}"
        options.append(f'<option value="{html.escape(value)}"{selected}>{html.escape(_truncate(label, 120))}</option>')
    return '<select name="question_id" data-question-selector required>' + "".join(options) + "</select>"


def route_direction_block(context: dict[str, Any]) -> str:
    synthesis = context.get("latest_synthesis") or {}
    parts: list[str] = []
    if synthesis.get("summary"):
        parts.append(
            '<article class="context-card">'
            '<h3>上一轮综合</h3>'
            f'<p>{html.escape(str(synthesis.get("summary") or ""))}</p>'
            '</article>'
        )
    gaps = route_json_list(synthesis.get("evidence_gaps_json")) or list(context.get("evidence_gaps") or [])
    queries = route_json_list(synthesis.get("next_queries_json"))
    if gaps:
        parts.append("<h3>证据缺口</h3>" + compact_list(gaps[:6]))
    if queries:
        parts.append("<h3>建议方向 / Query</h3>" + compact_list(queries[:6]))
    if not parts:
        questions = list(context.get("retrieval_questions") or [])
        if questions:
            parts.append("<h3>可用问题线索</h3>" + compact_list(questions[:6]))
    if not parts:
        return '<div class="subtle">暂无 03 模块归纳方向。可以先用 01 的科学问题生成本地草稿。</div>'
    return "".join(parts)


def route_run_meta(run: dict[str, Any]) -> str:
    if not run:
        return '<div class="subtle">尚未生成路线。选择问题和数量后点击生成。</div>'
    mode = str((run.get("metadata") or {}).get("mode") or "")
    mode_label = {
        "llm": "外部 LLM",
        "fallback_no_api_key": "本地规则草稿",
        "fallback_llm_error": "LLM 失败后本地草稿",
    }.get(mode, mode or "未知模式")
    return (
        '<div class="subtle">'
        f'{html.escape(str(run.get("created_at") or ""))} · {html.escape(mode_label)} · {html.escape(str(run.get("model") or ""))}'
        '</div>'
    )


def route_cards(routes: list[dict[str, Any]]) -> str:
    if not routes:
        return '<div class="route-empty">选择一个科学问题，设定要生成的路线数量，然后生成候选路线。</div>'
    return '<div class="route-grid">' + "".join(route_card(route) for route in routes) + "</div>"


def route_card(route: dict[str, Any]) -> str:
    priority = str(route.get("priority") or "中")
    priority_class = {"高": "badge-teal", "中": "badge-blue", "低": "badge-neutral"}.get(priority, "badge-neutral")
    return (
        '<article class="route-card">'
        '<div class="route-card-head">'
        f'<span class="route-rank">Route {html.escape(str(route.get("rank") or ""))}</span>'
        f'<span class="badge {priority_class}">优先级 {html.escape(priority)}</span>'
        '</div>'
        f'<h3>{html.escape(str(route.get("title") or "候选路线"))}</h3>'
        f'<p>{html.escape(str(route.get("rationale") or ""))}</p>'
        f'{route_list_block("候选材料 / 结构", route.get("candidates"))}'
        f'{route_list_block("关键变量", route.get("variables"))}'
        f'{route_list_block("验证方式", route.get("validation"))}'
        f'{route_list_block("证据依据", route.get("evidence"))}'
        f'{route_list_block("主要风险", route.get("risks"))}'
        f'<div class="route-next"><b>下一步</b><span>{html.escape(str(route.get("next_step") or "待细化"))}</span></div>'
        '</article>'
    )


def route_list_block(title: str, items: Any) -> str:
    values = [str(item).strip() for item in (items or []) if str(item).strip()]
    if not values:
        return ""
    return (
        '<section class="route-block">'
        f'<h4>{html.escape(title)}</h4>'
        '<ul>'
        + "".join(f"<li>{html.escape(item)}</li>" for item in values[:8])
        + "</ul></section>"
    )


def route_run_table(runs: list[dict[str, Any]]) -> str:
    if not runs:
        return '<div class="subtle">暂无历史生成。</div>'
    body = "".join(
        "<tr>"
        f'<td>{html.escape(str(row.get("created_at") or ""))}</td>'
        f'<td>{html.escape(_truncate(str(row.get("question_title") or ""), 80))}</td>'
        f'<td class="num">{html.escape(str(len(row.get("routes") or [])))}</td>'
        f'<td>{html.escape(str((row.get("metadata") or {}).get("mode") or ""))}</td>'
        "</tr>"
        for row in runs[:8]
    )
    return (
        '<div class="table-wrap"><table><thead><tr><th>时间</th><th>问题</th><th>路线</th><th>模式</th></tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def route_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if str(item).strip()]


def route_candidate_css() -> str:
    return """
    .route-shell { display:grid; gap:10px; margin:0 auto; max-width:1560px; padding:12px 18px 42px; }
    .route-hero { background:linear-gradient(135deg,#0f172a,#12363f 58%,#166534); min-height:84px; padding:14px 16px; }
    .route-hero h2 { font-size:19px; margin-bottom:4px; }
    .route-workspace { align-items:start; display:grid; gap:10px; grid-template-columns:360px minmax(0,1fr); }
    .route-side .panel { background:var(--nav-2); border-color:var(--nav-line); color:#e5eefb; }
    .route-side h2 { color:#f8fafc; }
    .route-side h3 { color:#93c5fd; margin:10px 0 7px; }
    .route-side .subtle, .route-side li { color:#cbd5e1; }
    .route-side code { background:#0b1220; border-color:#334155; color:#dbeafe; white-space:normal; }
    .route-side form { display:grid; gap:7px; }
    .route-main { min-width:0; }
    .route-grid { display:grid; gap:10px; grid-template-columns:repeat(2,minmax(0,1fr)); }
    .route-card {
      background:linear-gradient(180deg,#ffffff,#f8fbff);
      border:1px solid var(--line);
      border-radius:8px;
      display:grid;
      gap:10px;
      min-width:0;
      padding:12px;
    }
    .route-card * { min-width:0; overflow-wrap:anywhere; }
    .route-card-head { align-items:center; display:flex; gap:8px; justify-content:space-between; }
    .route-rank { color:var(--primary); font-family:"Fira Code", ui-monospace, SFMono-Regular, Menlo, monospace; font-size:11px; font-weight:800; text-transform:uppercase; }
    .route-card h3 { color:#0f172a; font-size:16px; line-height:1.35; margin:0; text-transform:none; }
    .route-card p { color:#334155; line-height:1.5; margin:0; }
    .route-block { border-top:1px solid #e2e8f0; display:grid; gap:6px; padding-top:8px; }
    .route-block h4 { color:#475569; font-size:12px; margin:0; }
    .route-block ul { display:grid; gap:5px; margin:0; padding-left:18px; }
    .route-block li { line-height:1.45; }
    .route-next { background:#f0fdf4; border:1px solid #bbf7d0; border-radius:8px; display:grid; gap:4px; padding:9px; }
    .route-next b { color:#166534; }
    .route-next span { color:#14532d; line-height:1.45; }
    .route-empty {
      align-content:center;
      background:#f8fafc;
      border:1px dashed var(--line-strong);
      border-radius:8px;
      color:var(--muted);
      display:grid;
      min-height:260px;
      padding:24px;
      text-align:center;
    }
    @media (max-width:1100px) { .route-workspace { grid-template-columns:1fr; } .route-grid { grid-template-columns:1fr; } }
    @media (max-width:640px) { .route-shell { padding:10px 10px 30px; } }
    """


def _empty_route_candidate_state() -> dict[str, Any]:
    return {
        "context": {"metrics": {}, "retrieval_questions": [], "evidence_gaps": [], "paper_cards": [], "wiki_pages": []},
        "questions": [],
        "selected_question": None,
        "latest_run": None,
        "all_runs": [],
        "metrics": {},
        "model_name": "LLM 未配置",
        "llm_configured": False,
        "output_path": str(route_candidate_data_dir() / "route_candidates.json"),
    }


def render_question_synthesis(*, message: str | None = None, error: str | None = None) -> str:
    try:
        state = question_synthesis_state()
        state_error = None
    except Exception as exc:
        state = _empty_question_synthesis_state()
        state_error = _friendly_error(str(exc))
    messages = state.get("messages", [])
    context = state.get("context", {})
    metrics = context.get("metrics", {})
    model_name = state.get("model_name") or "LLM 未配置"
    llm_configured = bool(state.get("llm_configured"))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>科学问题归纳 · AI Scientist</title>
  <style>{base_css()}{question_synthesis_css()}</style>
  <script>
    window.addEventListener('DOMContentLoaded', () => {{
      const log = document.querySelector('.chat-log');
      if (log) log.scrollTop = log.scrollHeight;
      const busy = document.querySelector('[data-busy]');
      document.querySelectorAll('form[method="post"]').forEach((form) => {{
        form.addEventListener('submit', () => {{
          const button = form.querySelector('button[type="submit"], button:not([type])');
          if (busy && button) {{
            busy.textContent = button.dataset.busy || '正在请 LLM 基于当前证据继续归纳，请稍等。';
            busy.style.display = 'block';
          }}
          form.querySelectorAll('button').forEach((item) => {{
            item.disabled = true;
            if (!item.classList.contains('secondary')) item.textContent = '处理中...';
          }});
        }});
      }});
    }});
  </script>
</head>
<body>
  <header>
    <div class="topbar">
      <div>
        <div class="brand-kicker">Stage 03 · Scientific Question Synthesis</div>
        <h1>科学问题归纳</h1>
        <div class="status-line">
          <span class="badge badge-teal">已接入</span>
          <span class="badge badge-blue">LLM：{html.escape(str(model_name))}</span>
          <span class="subtle">模块：{html.escape(str(question_synthesis_agent_dir()))}</span>
        </div>
      </div>
      <div class="header-actions">
        <a class="button secondary" href="/">返回首页</a>
        <a class="button secondary" href="/literature">文献获取</a>
        <a class="button secondary" href="/paper-analysis">文献分析</a>
      </div>
    </div>
  </header>
  <main class="question-shell">
    <div class="notice busy full" data-busy></div>
    {notice(message, "ok") if message else ""}
    {notice(error or state_error, "err") if (error or state_error) else ""}
    <section class="full hero-strip question-hero">
      <div>
        <div class="brand-kicker">Question Refinement Workspace</div>
        <h2>{html.escape(str((state.get("session") or {}).get("title") or "从证据中收敛科学问题"))}</h2>
        <p class="subtle">检索问题、证据缺口、Paper Cards 与 Wiki 结果在同一个上下文中汇合，用于形成材料体系、机制变量、验证方法和评价指标明确的研究问题。</p>
      </div>
      <div class="hero-counters">
        <span><b>{html.escape(str(metrics.get("retrieval_questions", 0)))}</b> questions</span>
        <span><b>{html.escape(str(metrics.get("paper_cards", 0)))}</b> cards</span>
        <span><b>{html.escape(str(metrics.get("wiki_pages", 0)))}</b> wiki</span>
      </div>
    </section>
    <section class="full metrics">
      {metric_card("检索问题", metrics.get("retrieval_questions", 0))}
      {metric_card("证据缺口", metrics.get("evidence_gaps", 0))}
      {metric_card("候选文献", metrics.get("candidate_papers", 0))}
      {metric_card("Paper Cards", metrics.get("paper_cards", 0))}
      {metric_card("Wiki 条目", metrics.get("wiki_pages", 0))}
      {metric_card("LLM 状态", "已配置" if llm_configured else "本地草稿")}
    </section>
    <section class="chat-workspace">
      <aside class="stack question-context">
        <section class="panel">
          <div class="panel-head"><h2>检索阶段输入</h2></div>
          {retrieval_context_block(context)}
        </section>
        <section class="panel">
          <div class="panel-head"><h2>分析证据摘要</h2></div>
          {analysis_context_block(context)}
        </section>
        <section class="panel">
          <div class="panel-head"><h2>会话操作</h2></div>
          <code>{html.escape(str(question_synthesis_data_dir()))}</code>
          <form method="post" action="/question-synthesis/reset">
            <button class="secondary" data-busy="正在重置并重新载入上下文。" type="submit">重置对话</button>
          </form>
        </section>
      </aside>
      <section class="panel chat-panel">
        <div class="panel-head">
          <div>
            <h2>科学问题归纳对话</h2>
            <div class="subtle">模型名称：{html.escape(str(model_name))}</div>
          </div>
          <span class="badge {'badge-teal' if llm_configured else 'badge-amber'}">{'外部 LLM' if llm_configured else 'Fallback'}</span>
        </div>
        <div class="chat-log" aria-live="polite">
          {question_chat_messages(messages)}
        </div>
        <form class="chat-input" method="post" action="/question-synthesis/chat">
          <textarea name="message" required placeholder="输入你的追问或研究偏好"></textarea>
          <button data-busy="正在请 LLM 基于当前证据继续归纳，请稍等。" type="submit">发送</button>
        </form>
      </section>
    </section>
  </main>
</body>
</html>"""


def render_paper_analysis(*, message: str | None = None, error: str | None = None, query: str = "") -> str:
    state = load_paper_analysis_state_for_query(query)
    metrics = state["metrics"]
    latest_round = state["latest_round"]
    latest_round_id = latest_round["id"] if latest_round else ""
    graph_script = knowledge_graph_script(state)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>文献知识工作台</title>
  <style>{base_css()}</style>
  <script>
    window.addEventListener('DOMContentLoaded', () => {{
      const busy = document.querySelector('[data-busy]');
      document.querySelectorAll('form[method="post"]').forEach((form) => {{
        form.addEventListener('submit', () => {{
          if (busy) {{
            const button = form.querySelector('button[type="submit"], button:not([type])');
            const label = button ? button.textContent.trim() : '后台任务';
            busy.textContent = `正在执行：${{label}}。转换、索引和图谱构建可能需要几十秒，请不要重复点击。`;
            busy.style.display = 'block';
          }}
          form.querySelectorAll('button').forEach((button) => {{
            button.disabled = true;
            if (!button.classList.contains('danger')) button.textContent = '执行中...';
          }});
        }});
      }});
    }});
  </script>
</head>
<body>
  <header>
    <div class="topbar">
      <div>
        <div class="brand-kicker">Stage 02 · RAG + Cards + Graph + Wiki</div>
        <h1>文献知识工作台</h1>
        <div class="status-line">
          <span class="badge badge-teal">已接入</span>
          <span class="subtle">分析项目：{html.escape(str(analysis_agent_dir()))}</span>
          <span class="badge badge-blue">{embedding_mode_label(state)}</span>
        </div>
      </div>
      <div class="header-actions">
        <a class="button secondary" href="/">返回首页</a>
        <a class="button secondary" href="/literature">文献获取</a>
      </div>
    </div>
  </header>
  <main class="shell">
    <div class="notice busy full" data-busy></div>
    {notice(message, "ok") if message else ""}
    {notice(error, "err") if error else ""}
    <section class="full hero-strip">
      <div>
        <div class="brand-kicker">Literature Knowledge Workspace</div>
        <h2>RAG 找证据，Paper Cards 摘要论文，轻量图谱理解关系，Wiki 沉淀知识</h2>
        <p class="subtle">第 2 板块读取同级“文献分析_agent”的产物：parsed_papers、FTS/embedding 索引、cards、graph.json 和 wiki 页面。所有结论都保留 evidence ids，方便追溯回段落和源 JSON。</p>
      </div>
      <div class="hero-counters">
        <span><b>{metrics["converted"]}</b> parsed</span>
        <span><b>{metrics["cards"]}</b> cards</span>
        <span><b>{metrics["nodes"]}</b> nodes</span>
      </div>
    </section>
    <section class="full metrics">
      {metric_card("PDF 队列", metrics["pdf_total"])}
      {metric_card("已转换", metrics["converted"])}
      {metric_card("RAG Chunks", metrics["chunks"])}
      {metric_card("Paper Cards", metrics["cards"])}
      {metric_card("Graph Nodes", metrics["nodes"])}
      {metric_card("Wiki Pages", metrics["wiki_pages"])}
    </section>
    <aside class="stack knowledge-side">
      <section class="panel">
        <div class="panel-head"><h2>语料与构建</h2></div>
        <form method="post" action="/paper-analysis/convert">
          <input type="hidden" name="force" value="0">
          <button type="submit">转换全部 PDF</button>
        </form>
        <form method="post" action="/paper-analysis/convert">
          <label>轮次 ID</label>
          <input name="round_id" type="number" value="{html.escape(str(latest_round_id))}" required>
          <button type="submit">转换当前轮次 PDF</button>
        </form>
        <form method="post" action="/paper-analysis/convert">
          <input type="hidden" name="force" value="1">
          <button type="submit">强制重转全部 PDF</button>
        </form>
        <form method="post" action="/paper-analysis/build-all">
          <button type="submit">构建知识库</button>
        </form>
      </section>
      <section class="panel">
        <div class="panel-head"><h2>检索证据</h2></div>
        <form method="get" action="/paper-analysis">
          <label>关键词</label>
          <input name="q" value="{html.escape(query)}" placeholder="coercivity / NdFeB / micromagnetic">
          <button type="submit">RAG 搜索</button>
        </form>
        <p class="subtle">检索 paragraph、section、caption、table、reference；有本地向量索引时使用 hybrid rank。</p>
      </section>
      <section class="panel">
        <div class="panel-head"><h2>小批量试跑</h2></div>
        <form method="post" action="/paper-analysis/convert">
          <label>最多转换</label>
          <input name="limit" type="number" value="2" min="1" max="100">
          <button type="submit">转换前 N 个</button>
        </form>
        <p class="subtle">解析由同级“文献分析_agent”执行；建议先用 1-2 篇检查 Markdown 和 JSON 质量，再构建知识库。</p>
      </section>
      <section class="panel">
        <div class="panel-head"><h2>输出位置</h2></div>
        <code>{html.escape(str(analysis_data_dir()))}</code>
        {conversion_summary(state)}
      </section>
    </aside>
    <section class="stack knowledge-main">
      <section class="panel">
        <div class="panel-head"><h2>RAG 证据</h2><span class="subtle">{len(state["rag_results"])} 条结果</span></div>
        {rag_results_block(state)}
      </section>
      <section class="panel">
        <div class="panel-head"><h2>Paper Cards</h2><span class="subtle">{metrics["cards"]} 张卡片</span></div>
        {paper_cards_block(state["cards"])}
      </section>
      <section class="panel graph-panel">
        <div class="panel-head"><h2>轻量知识图谱</h2><span class="subtle">{len(state["graph"].get("edges", []))} 条关系</span></div>
        {knowledge_graph_block(state)}
      </section>
      <section class="panel">
        <div class="panel-head"><h2>Wiki 条目</h2><span class="subtle">{metrics["wiki_pages"]} 个主题</span></div>
        {wiki_pages_block(state["wiki_pages"])}
      </section>
      <section class="panel">
        <div class="panel-head"><h2>最近转换</h2><span class="subtle">{len(state["conversions"])} 条</span></div>
        {conversion_table(state["conversions"][:12])}
      </section>
    </section>
  </main>
  {graph_script}
</body>
</html>"""


def workflow_module_card(item: dict[str, Any], metrics: dict[str, int]) -> str:
    is_ready = bool(item["enabled"])
    klass = "module-card ready" if is_ready else "module-card disabled"
    badge = "badge-teal" if is_ready else "badge-neutral"
    if item["path"] == "/literature":
        metric_values = (metrics["goals"], metrics["rounds"], metrics["downloaded"])
    elif item["path"] == "/paper-analysis":
        metric_values = (metrics.get("analysis_pdf_total", 0), metrics.get("analysis_converted", 0), metrics.get("analysis_failed", 0))
    else:
        metric_values = (0, 0, 0)
    metric_row = "".join(
        f"<span><b>{value}</b>{html.escape(label)}</span>"
        for value, label in zip(metric_values, item["metrics"])
    )
    return (
        f'<a class="{klass}" href="{html.escape(item["path"])}">'
        f'<div class="module-top"><span class="stage">Stage {html.escape(item["stage"])}</span>'
        f'<span class="badge {badge}">{html.escape(item["status"])}</span></div>'
        f'<h2>{html.escape(item["title"])}</h2>'
        f'<p>{html.escape(item["summary"])}</p>'
        f'<div class="module-metrics">{metric_row}</div>'
        "</a>"
    )


def home_module_button(item: dict[str, Any]) -> str:
    klass = "home-module-button ready" if item["enabled"] else "home-module-button"
    return (
        f'<a class="{klass}" href="{html.escape(item["path"])}" data-stage="{html.escape(item["stage"])}">'
        '<span class="home-stage">'
        f'<small>{html.escape(item["stage"])}</small>'
        f'<b>{html.escape(item["title"])}</b>'
        '</span>'
        f'<em>{html.escape(item["summary"])}</em>'
        "</a>"
    )


def llm_api_panel(config: dict[str, Any]) -> str:
    apis = list(config.get("apis", []))
    first = apis[0] if len(apis) > 0 else {}
    second = apis[1] if len(apis) > 1 else {}
    active = active_llm_api(config)
    status = f"当前启用：{active['name']} · {active['model']}" if active else "尚未启用外部 LLM"
    second_configured = bool(second.get("model") or second.get("api_key") or second.get("base_url"))
    checked = " checked" if second_configured else ""
    return (
        '<aside class="llm-panel" aria-label="外部 LLM API 配置">'
        '<label class="llm-close" for="llm-toggle">隐藏</label>'
        '<div class="brand-kicker">External LLM</div>'
        '<h2>外部 LLM API</h2>'
        f'<p class="subtle">{html.escape(status)}</p>'
        f'<code>{html.escape(str(default_external_llm_config_path()))}</code>'
        f'{llm_api_form(0, first)}'
        f'<input class="llm-extra-toggle" id="llm-extra-toggle" type="checkbox"{checked}>'
        '<label class="llm-add-button" for="llm-extra-toggle" title="添加第二个 LLM API">+</label>'
        f'<div class="llm-extra-slot">{llm_api_form(1, second)}</div>'
        "</aside>"
    )


def llm_api_form(slot: int, entry: dict[str, Any]) -> str:
    name = str(entry.get("name") or f"LLM API {slot + 1}")
    base_url = str(entry.get("base_url") or "https://api.openai.com/v1")
    model = str(entry.get("model") or "")
    checked = " checked" if entry.get("enabled") or (slot == 0 and not entry) else ""
    key_hint = "已保存，留空表示沿用" if entry.get("api_key") else "请输入 API Key"
    return f"""
        <form class="llm-api-form" method="post" action="/llm-api/save">
          <input type="hidden" name="slot" value="{slot}">
          <div class="panel-head"><h3>{html.escape(name)}</h3><span class="badge badge-neutral">API {slot + 1}</span></div>
          <label>名称</label>
          <input name="name" value="{html.escape(name)}" placeholder="OpenAI / DeepSeek / Qwen">
          <label>Base URL</label>
          <input name="base_url" value="{html.escape(base_url)}" placeholder="https://api.openai.com/v1">
          <label>API Key</label>
          <input name="api_key" type="password" autocomplete="new-password" placeholder="{html.escape(key_hint)}">
          <label>模型</label>
          <input name="model" value="{html.escape(model)}" placeholder="gpt-4.1 / deepseek-chat / qwen-plus">
          <label class="llm-check"><input name="enabled" type="checkbox" value="1"{checked}>设为当前使用</label>
          <button type="submit">保存 API {slot + 1}</button>
        </form>
    """


def base_css() -> str:
    return """
    :root {
      color-scheme: light;
      --bg:#eef3f8;
      --surface:#ffffff;
      --surface-soft:#f8fafc;
      --line:#c9d7e8;
      --line-strong:#8ea3bc;
      --text:#0f172a;
      --muted:#64748b;
      --nav:#0f172a;
      --nav-2:#111c31;
      --nav-line:#243247;
      --primary:#1e40af;
      --primary-soft:#dbeafe;
      --secondary:#3b82f6;
      --accent:#d97706;
      --amber-soft:#fff7e5;
      --ok:#16a34a;
      --ok-soft:#dcfce7;
      --cyan:#0891b2;
      --cyan-soft:#cffafe;
      --red:#dc2626;
      --red-soft:#fff1f1;
      --shadow:0 12px 30px rgba(15,23,42,.08);
    }
    * { box-sizing:border-box; }
    body {
      background-color:var(--bg);
      background-image:linear-gradient(rgba(30,64,175,.045) 1px, transparent 1px), linear-gradient(90deg, rgba(30,64,175,.045) 1px, transparent 1px);
      background-size:28px 28px;
      color:var(--text);
      font-family:"Fira Sans", Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size:14px;
      margin:0;
      overflow-x:hidden;
    }
    header { background:var(--nav); border-bottom:1px solid #1d4ed8; box-shadow:0 12px 34px rgba(15,23,42,.22); position:sticky; top:0; z-index:10; }
    .topbar { align-items:center; display:flex; gap:16px; justify-content:space-between; margin:0 auto; max-width:1560px; padding:12px 18px; }
    .brand-kicker { color:#93c5fd; font-family:"Fira Code", ui-monospace, SFMono-Regular, Menlo, monospace; font-size:11px; font-weight:700; letter-spacing:.08em; margin-bottom:4px; text-transform:uppercase; }
    h1 { color:#f8fafc; font-size:22px; letter-spacing:0; line-height:1.2; margin:0; }
    h2 { font-size:15px; letter-spacing:0; margin:0; }
    h3 { color:var(--muted); font-size:12px; font-weight:650; letter-spacing:.02em; margin:0 0 8px; text-transform:uppercase; }
    a { color:var(--primary); }
    .subtle { color:var(--muted); font-size:12px; line-height:1.45; }
    .header-actions { align-items:center; display:flex; flex-wrap:wrap; gap:8px; justify-content:flex-end; }
    header .subtle { color:#94a3b8; }
    .shell { display:grid; gap:10px; grid-template-columns:320px minmax(0,1fr); margin:0 auto; max-width:1560px; padding:12px 18px 42px; }
    .home-shell { display:grid; gap:12px; margin:0 auto; max-width:1320px; padding:14px 18px 42px; }
    .full { grid-column:1 / -1; }
    .home-hero, .hero-strip {
      align-items:center;
      background:linear-gradient(135deg,#0f172a,#12284d 62%,#1e40af);
      border:1px solid #274b8d;
      border-radius:8px;
      box-shadow:0 18px 42px rgba(15,23,42,.22);
      color:#f8fafc;
      display:flex;
      gap:20px;
      justify-content:space-between;
      min-height:104px;
      overflow:hidden;
      padding:18px;
      position:relative;
    }
    .home-hero::after, .hero-strip::after {
      background:linear-gradient(90deg, transparent, rgba(255,255,255,.12), transparent);
      content:"";
      height:100%;
      position:absolute;
      right:18%;
      top:0;
      transform:skewX(-18deg);
      width:120px;
    }
    .home-hero h2, .hero-strip h2 { color:#f8fafc; font-size:22px; line-height:1.25; margin:0 0 6px; }
    .home-hero .subtle, .hero-strip .subtle { color:#cbd5e1; max-width:760px; }
    .hero-counters { display:grid; gap:8px; grid-template-columns:repeat(3,minmax(96px,1fr)); position:relative; z-index:1; }
    .hero-counters span { background:rgba(15,23,42,.42); border:1px solid rgba(147,197,253,.28); border-radius:8px; color:#cbd5e1; font-size:11px; padding:10px; text-transform:uppercase; }
    .hero-counters b { color:#f8fafc; display:block; font-family:"Fira Code", ui-monospace, SFMono-Regular, Menlo, monospace; font-size:22px; margin-bottom:2px; }
    .panel { background:rgba(255,255,255,.96); border:1px solid var(--line); border-radius:8px; box-shadow:var(--shadow); min-width:0; padding:12px; }
    aside .panel { background:var(--nav-2); border-color:var(--nav-line); color:#e5eefb; box-shadow:0 14px 32px rgba(15,23,42,.18); }
    aside .subtle, aside label { color:#94a3b8; }
    aside h2 { color:#f8fafc; }
    .panel-head { align-items:center; border-bottom:1px solid rgba(148,163,184,.22); display:flex; gap:10px; justify-content:space-between; margin:-2px 0 10px; padding-bottom:9px; }
    .stack { display:grid; gap:10px; }
    .metrics { display:grid; gap:8px; grid-template-columns:repeat(6,minmax(0,1fr)); }
    .metric { background:linear-gradient(180deg,#ffffff,#f8fbff); border:1px solid var(--line); border-top:4px solid var(--primary); border-radius:8px; min-height:78px; padding:10px 11px; position:relative; }
    .metric:nth-child(2) { border-top-color:var(--secondary); }
    .metric:nth-child(3) { border-top-color:var(--cyan); }
    .metric:nth-child(4) { border-top-color:var(--ok); }
    .metric:nth-child(5) { border-top-color:var(--accent); }
    .metric:nth-child(6) { border-top-color:var(--red); }
    .metric-value { font-family:"Fira Code", ui-monospace, SFMono-Regular, Menlo, monospace; font-size:27px; font-weight:760; line-height:1; margin-bottom:6px; }
    .metric-label { color:var(--muted); font-size:12px; }
    .metric-rule { background:#e2e8f0; border-radius:999px; bottom:9px; height:3px; left:11px; overflow:hidden; position:absolute; right:11px; }
    .metric-rule span { background:var(--primary); display:block; height:100%; width:62%; }
    .metric:nth-child(4) .metric-rule span { background:var(--ok); width:76%; }
    .metric:nth-child(5) .metric-rule span { background:var(--accent); width:42%; }
    .workflow, .flow-rail { background:var(--nav); border:1px solid var(--nav-line); border-radius:8px; display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); overflow:hidden; }
    .step, .flow-node { border-right:1px solid var(--nav-line); color:#cbd5e1; display:grid; gap:5px; min-height:58px; padding:9px 11px; position:relative; }
    .step:last-child, .flow-node:last-child { border-right:0; }
    .step.active, .flow-node.ready { background:#102a43; color:#f8fafc; }
    .step.active::before, .flow-node.ready::before { background:var(--ok); bottom:0; content:""; left:0; position:absolute; top:0; width:3px; }
    .step.done { background:#111f35; color:#dbeafe; }
    .step-num, .flow-node span { color:#93c5fd; font-family:"Fira Code", ui-monospace, SFMono-Regular, Menlo, monospace; font-size:10px; font-weight:700; }
    .step-name, .flow-node b { font-size:13px; font-weight:750; }
    .notice { border-radius:8px; font-size:13px; padding:10px 12px; }
    .busy { background:#eff6ff; border:1px solid #93c5fd; color:#1e3a8a; display:none; }
    .ok { background:var(--ok-soft); border:1px solid #86efac; }
    .err { background:var(--red-soft); border:1px solid #e1a3a0; }
    label { color:var(--muted); display:block; font-size:12px; font-weight:650; margin:8px 0 4px; }
    input, textarea, select { background:#fff; border:1px solid #c9d3df; border-radius:7px; color:var(--text); font:inherit; min-height:34px; padding:7px 9px; width:100%; }
    aside input, aside textarea, aside select { background:#0b1220; border-color:#334155; color:#f8fafc; }
    textarea { min-height:76px; resize:vertical; }
    input:focus, textarea:focus, select:focus { border-color:var(--secondary); box-shadow:0 0 0 3px rgba(59,130,246,.18); outline:none; }
    button, a.button { align-items:center; background:var(--primary); border:1px solid var(--primary); border-radius:7px; color:#fff; cursor:pointer; display:inline-flex; font:inherit; font-size:13px; font-weight:700; justify-content:center; min-height:34px; padding:7px 10px; text-decoration:none; transition:background .16s ease, border-color .16s ease, box-shadow .16s ease, transform .16s ease; }
    button:hover, a.button:hover { background:#1d4ed8; border-color:#1d4ed8; transform:translateY(-1px); }
    button:focus-visible, a.button:focus-visible { box-shadow:0 0 0 3px rgba(59,130,246,.25); outline:none; }
    button.danger { background:var(--red); border-color:var(--red); min-height:28px; padding:4px 8px; }
    button.danger:hover { background:#b91c1c; border-color:#b91c1c; }
    button[disabled] { cursor:wait; opacity:.72; transform:none; }
    .secondary { background:#f8fafc; border-color:#475569; color:#0f172a; }
    .compact { min-height:28px; padding:4px 8px; }
    header .secondary { background:#172033; border-color:#334155; color:#e2e8f0; }
    .secondary:hover { background:#eff6ff; border-color:var(--secondary); }
    .action-row { display:grid; gap:8px; grid-template-columns:1fr; }
    .action-form { display:grid; gap:7px; grid-template-columns:minmax(0,1fr) auto; }
    .action-form input { min-width:0; }
    .status-line { align-items:center; display:flex; flex-wrap:wrap; gap:8px; margin-top:6px; }
    .badge { align-items:center; border-radius:999px; display:inline-flex; font-size:12px; font-weight:700; line-height:1; min-height:24px; padding:5px 8px; white-space:nowrap; }
    .badge-neutral { background:#e2e8f0; color:#334155; }
    header .badge-neutral { background:#1e293b; color:#cbd5e1; }
    .badge-teal { background:var(--ok-soft); color:#166534; }
    .badge-blue { background:var(--primary-soft); color:#1e40af; }
    .badge-amber { background:var(--amber-soft); color:var(--amber); }
    .badge-red { background:var(--red-soft); color:var(--red); }
    .table-wrap { border:1px solid var(--line); border-radius:8px; max-height:440px; overflow:auto; }
    table { border-collapse:collapse; font-size:12.5px; width:100%; }
    th,td { border-bottom:1px solid var(--line); padding:7px 8px; text-align:left; vertical-align:top; }
    th { background:#eaf1fb; color:#1e3a8a; font-size:11px; font-weight:800; letter-spacing:.02em; position:sticky; text-transform:uppercase; top:0; z-index:1; }
    tr:last-child td { border-bottom:0; }
    tr:hover td { background:#f0f7ff; }
    .title-cell { min-width:260px; }
    .num { font-family:"Fira Code", ui-monospace, SFMono-Regular, Menlo, monospace; font-variant-numeric:tabular-nums; text-align:right; white-space:nowrap; }
    code { background:#f4f6f8; border:1px solid #e0e5eb; border-radius:6px; color:#314154; display:block; font-size:12px; overflow-x:auto; padding:7px 8px; white-space:nowrap; }
    .list { display:grid; gap:8px; margin:0; padding:0; }
    .list li { border-bottom:1px solid var(--line); list-style:none; padding:8px 0; }
    .list li:last-child { border-bottom:0; }
    .split { display:grid; gap:10px; grid-template-columns:minmax(0,1.25fr) minmax(260px,.75fr); }
    .knowledge-side .panel form { display:grid; gap:7px; margin-bottom:8px; }
    .knowledge-main { min-width:0; }
    .rag-list { display:grid; gap:8px; min-width:0; }
    .rag-result, .paper-card, .wiki-item {
      background:linear-gradient(180deg,#ffffff,#f8fbff);
      border:1px solid var(--line);
      border-radius:8px;
      display:grid;
      gap:8px;
      max-width:100%;
      min-width:0;
      padding:10px;
    }
    .rag-result *, .paper-card *, .wiki-item * { min-width:0; overflow-wrap:anywhere; }
    .rag-result h3, .paper-card h3, .wiki-item h3 { color:var(--text); font-size:13px; line-height:1.35; margin:0; text-transform:none; }
    .rag-result p, .paper-card p, .wiki-item p { color:#334155; line-height:1.5; margin:0; }
    .paper-card-grid { display:grid; gap:10px; grid-template-columns:repeat(2,minmax(0,1fr)); }
    .wiki-grid { display:grid; gap:10px; grid-template-columns:repeat(2,minmax(0,1fr)); }
    .chip-row { display:flex; flex-wrap:wrap; gap:6px; }
    .topic-chip {
      background:#eef2ff;
      border:1px solid #c7d2fe;
      border-radius:999px;
      color:#3730a3;
      font-size:11px;
      font-weight:750;
      line-height:1;
      padding:5px 7px;
    }
    .chip-material { background:#dcfce7; border-color:#86efac; color:#166534; }
    .chip-method { background:#ede9fe; border-color:#c4b5fd; color:#5b21b6; }
    .chip-property { background:#fee2e2; border-color:#fecaca; color:#991b1b; }
    .mini-list { display:grid; gap:6px; margin:0; padding:0; }
    .mini-list li { border-top:1px solid rgba(148,163,184,.22); list-style:none; padding-top:6px; }
    .mini-list li:first-child { border-top:0; padding-top:0; }
    .graph-panel { min-height:560px; }
    .graph-workspace { display:grid; gap:10px; grid-template-columns:minmax(0,1fr) 280px; }
    .graph-controls { align-items:center; display:flex; flex-wrap:wrap; gap:7px; grid-column:1 / -1; }
    .filter-check {
      align-items:center;
      background:#f8fafc;
      border:1px solid var(--line);
      border-radius:999px;
      color:#334155;
      display:inline-flex;
      font-size:12px;
      font-weight:700;
      gap:5px;
      margin:0;
      min-height:28px;
      padding:4px 8px;
    }
    .filter-check input { min-height:0; width:auto; }
    #knowledge-graph {
      background:#f8fafc;
      border:1px solid var(--line);
      border-radius:8px;
      min-height:500px;
      width:100%;
    }
    .graph-edge { stroke:#94a3b8; stroke-opacity:.42; stroke-width:1.2; }
    .graph-node { cursor:pointer; outline:none; }
    .graph-node circle {
      stroke:#fff;
      stroke-width:2;
      transition:filter .16s ease, stroke .16s ease, transform .16s ease;
    }
    .graph-node text {
      fill:#0f172a;
      font-size:11px;
      font-weight:750;
      paint-order:stroke;
      pointer-events:none;
      stroke:#f8fafc;
      stroke-width:3px;
    }
    .graph-node:hover circle, .graph-node:focus-visible circle { filter:drop-shadow(0 3px 6px rgba(15,23,42,.25)); stroke:#111827; }
    .graph-highlight circle { stroke:#facc15; stroke-width:4; }
    .graph-detail {
      background:#ffffff;
      border:1px solid var(--line);
      border-radius:8px;
      max-height:500px;
      overflow:auto;
      padding:10px;
    }
    .graph-detail h3 { color:var(--muted); font-size:12px; line-height:1.4; margin:0 0 8px; text-transform:uppercase; }
    .graph-detail p { line-height:1.5; margin:8px 0; }
    .evidence-wrap { overflow-wrap:anywhere; }
    .graph-empty {
      align-content:center;
      background:#f8fafc;
      border:1px dashed var(--line-strong);
      border-radius:8px;
      color:var(--muted);
      display:grid;
      min-height:360px;
      padding:24px;
      text-align:center;
    }
    .module-grid { display:grid; gap:10px; grid-template-columns:repeat(3,minmax(0,1fr)); }
    .module-card { background:rgba(255,255,255,.96); border:1px solid var(--line); border-radius:8px; box-shadow:var(--shadow); color:var(--text); display:grid; gap:10px; min-height:190px; padding:14px; text-decoration:none; transition:border-color .16s ease, box-shadow .16s ease, transform .16s ease; }
    .module-card:hover { border-color:var(--secondary); box-shadow:0 16px 36px rgba(15,23,42,.12); transform:translateY(-1px); }
    .module-card.disabled { color:#334155; }
    .module-card.disabled:hover { border-color:var(--line-strong); }
    .module-top { align-items:center; display:flex; justify-content:space-between; gap:8px; }
    .stage { color:var(--primary); font-family:"Fira Code", ui-monospace, SFMono-Regular, Menlo, monospace; font-size:11px; font-weight:800; text-transform:uppercase; }
    .module-card h2 { font-size:18px; }
    .module-card p { color:var(--muted); line-height:1.55; margin:0; }
    .module-metrics { border-top:1px solid var(--line); display:grid; gap:8px; grid-template-columns:repeat(3,minmax(0,1fr)); padding-top:10px; }
    .module-metrics span { color:var(--muted); font-size:11px; }
    .module-metrics b { color:var(--text); display:block; font-family:"Fira Code", ui-monospace, SFMono-Regular, Menlo, monospace; font-size:20px; }
    .empty-module { align-items:flex-start; background:linear-gradient(135deg,#ffffff,#f8fbff); border:1px dashed var(--line-strong); border-radius:8px; box-shadow:var(--shadow); display:grid; gap:12px; min-height:360px; padding:22px; }
    .empty-module h2 { font-size:28px; }
    .empty-grid { display:grid; gap:10px; grid-template-columns:repeat(3,minmax(120px,1fr)); margin-top:8px; max-width:560px; }
    .empty-grid span { background:#f8fafc; border:1px solid var(--line); border-radius:8px; color:var(--muted); font-weight:700; padding:12px; }
    .ai-home-body {
      background:
        radial-gradient(circle at 16% 18%, rgba(59,130,246,.36), transparent 26%),
        radial-gradient(circle at 84% 28%, rgba(8,145,178,.24), transparent 25%),
        linear-gradient(135deg,#08111f 0%,#0f172a 48%,#102a43 100%);
      min-height:100vh;
      overflow-x:hidden;
    }
    .ai-home-body::before {
      background-image:linear-gradient(rgba(255,255,255,.055) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.055) 1px, transparent 1px);
      background-size:40px 40px;
      bottom:0;
      content:"";
      left:0;
      mask-image:linear-gradient(to bottom, rgba(0,0,0,.78), transparent 86%);
      pointer-events:none;
      position:fixed;
      right:0;
      top:0;
    }
    .home-minimal {
      align-content:center;
      display:grid;
      gap:28px;
      min-height:100vh;
      padding:42px clamp(18px,4vw,64px);
      position:relative;
      z-index:1;
    }
    .home-brand { color:#e2e8f0; display:grid; gap:12px; justify-items:center; text-align:center; }
    .home-brand span {
      color:#f8fafc;
      font-family:"Avenir Next Condensed","DIN Alternate","Futura","Fira Code",ui-monospace,SFMono-Regular,Menlo,monospace;
      font-size:clamp(42px,8vw,104px);
      font-weight:900;
      letter-spacing:.08em;
      line-height:.95;
      text-shadow:0 0 34px rgba(147,197,253,.32);
      text-transform:uppercase;
    }
    .home-brand b { color:#bfdbfe; font-family:"Fira Code", ui-monospace, SFMono-Regular, Menlo, monospace; font-size:clamp(13px,1.35vw,18px); font-weight:700; letter-spacing:.03em; line-height:1.35; max-width:980px; }
    .home-button-grid {
      display:grid;
      gap:10px;
      grid-template-columns:repeat(6,minmax(120px,1fr));
      justify-self:center;
      max-width:1180px;
      perspective:1000px;
      width:100%;
    }
    .home-module-button {
      align-content:end;
      background:rgba(15,23,42,.58);
      border:1px solid rgba(148,163,184,.28);
      border-radius:8px;
      color:#dbeafe;
      display:grid;
      gap:10px;
      min-height:300px;
      overflow:hidden;
      padding:18px 14px;
      position:relative;
      text-decoration:none;
      transform-origin:center;
      transition:background .16s ease, border-color .16s ease, box-shadow .16s ease, opacity .16s ease, transform .18s ease;
    }
    .home-module-button.ready { background:rgba(30,64,175,.62); border-color:rgba(147,197,253,.9); }
    .home-button-grid:has(.home-module-button:hover) .home-module-button:not(:hover) {
      opacity:.68;
      transform:scale(.96);
    }
    .home-module-button:hover, .home-module-button:focus-visible {
      background:rgba(30,64,175,.42);
      border-color:rgba(147,197,253,.75);
      box-shadow:0 18px 46px rgba(59,130,246,.22);
      transform:translateY(-4px) scale(1.08);
      z-index:2;
    }
    .home-module-button.ready:hover, .home-module-button.ready:focus-visible {
      background:rgba(59,130,246,.82);
      border-color:#bfdbfe;
      box-shadow:0 18px 42px rgba(59,130,246,.34), inset 0 0 0 1px rgba(255,255,255,.12);
      transform:translateY(-4px) scale(1.08);
    }
    .home-module-button::before {
      background-repeat:no-repeat;
      background-size:100% 100%;
      content:"";
      inset:16px auto auto 50%;
      opacity:.3;
      pointer-events:none;
      position:absolute;
      transform:translateX(-50%);
      width:118px;
      height:68px;
      transition:opacity .16s ease, transform .18s ease;
    }
    .home-module-button:hover::before, .home-module-button:focus-visible::before { opacity:.42; transform:translateX(-50%) translateY(-4px); }
    .home-module-button[data-stage="01"]::before {
      background-image:url("data:image/svg+xml,%3Csvg width='180' height='92' viewBox='0 0 180 92' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' stroke='%2393c5fd' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M24 18h54l12 12h66v48H24z'/%3E%3Cpath d='M38 43h48M38 57h76M38 71h56'/%3E%3Ccircle cx='137' cy='52' r='13'/%3E%3Cpath d='m147 62 15 15'/%3E%3C/g%3E%3C/svg%3E");
    }
    .home-module-button[data-stage="02"]::before {
      background-image:url("data:image/svg+xml,%3Csvg width='180' height='92' viewBox='0 0 180 92' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' stroke='%2367e8f9' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='46' cy='46' r='13'/%3E%3Ccircle cx='92' cy='24' r='10'/%3E%3Ccircle cx='130' cy='52' r='14'/%3E%3Ccircle cx='84' cy='72' r='9'/%3E%3Cpath d='M58 40 82 29M101 29l18 14M116 58 93 68M59 51l17 14M46 20v13M130 18v20M154 52h14'/%3E%3C/g%3E%3C/svg%3E");
    }
    .home-module-button[data-stage="03"]::before {
      background-image:url("data:image/svg+xml,%3Csvg width='180' height='92' viewBox='0 0 180 92' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' stroke='%23fde68a' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M48 67c0-18 30-18 30-42 0-17-35-17-40 1'/%3E%3Cpath d='M62 79h1'/%3E%3Cpath d='M98 26h48M98 42h34M98 58h54'/%3E%3Cpath d='M25 20 14 10M26 72 15 82M151 18l12-10'/%3E%3C/g%3E%3C/svg%3E");
    }
    .home-module-button[data-stage="04"]::before {
      background-image:url("data:image/svg+xml,%3Csvg width='180' height='92' viewBox='0 0 180 92' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' stroke='%2386efac' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M24 68c26-36 48-36 74-12s45 12 58-22'/%3E%3Cpath d='M24 68h28M74 45h28M120 58h36'/%3E%3Cpath d='m145 25 12 9-15 5'/%3E%3Ccircle cx='52' cy='44' r='6'/%3E%3Ccircle cx='101' cy='58' r='6'/%3E%3Ccircle cx='140' cy='41' r='6'/%3E%3C/g%3E%3C/svg%3E");
    }
    .home-module-button[data-stage="05"]::before {
      background-image:url("data:image/svg+xml,%3Csvg width='180' height='92' viewBox='0 0 180 92' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' stroke='%23c4b5fd' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='30' y='18' width='118' height='58' rx='5'/%3E%3Cpath d='M30 34h118M54 18v58M78 18v58M102 18v58M126 18v58'/%3E%3Cpath d='M42 48h11M66 62h11M91 48h10M114 62h11M138 48h11'/%3E%3Cpath d='M22 76h136'/%3E%3C/g%3E%3C/svg%3E");
    }
    .home-module-button[data-stage="06"]::before {
      background-image:url("data:image/svg+xml,%3Csvg width='180' height='92' viewBox='0 0 180 92' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' stroke='%23fca5a5' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M55 26a32 32 0 1 0 28 48'/%3E%3Cpath d='M75 16h32v32'/%3E%3Cpath d='M106 17 73 50'/%3E%3Cpath d='M92 66h58M106 50h30M116 34h40'/%3E%3Ccircle cx='52' cy='58' r='8'/%3E%3C/g%3E%3C/svg%3E");
    }
    .home-stage { display:grid; gap:8px; min-width:0; padding-bottom:78px; position:relative; z-index:1; }
    .home-stage small { color:#93c5fd; font-family:"Fira Code", ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; font-weight:800; letter-spacing:0; }
    .home-stage b { color:#f8fafc; font-size:22px; line-height:1.18; }
    .home-module-button em {
      color:#dbeafe;
      bottom:16px;
      left:14px;
      right:14px;
      font-size:12px;
      font-style:normal;
      line-height:1.45;
      opacity:0;
      overflow:hidden;
      position:absolute;
      transform:translateY(8px);
      transition:opacity .16s ease, transform .18s ease;
      z-index:1;
    }
    .home-module-button:hover em, .home-module-button:focus-visible em { opacity:1; transform:translateY(0); }
    .sidebar-toggle { height:1px; opacity:0; position:fixed; width:1px; }
    .llm-button {
      background:rgba(15,23,42,.76);
      border:1px solid rgba(125,211,252,.48);
      border-radius:999px;
      color:#e0f2fe;
      cursor:pointer;
      font-weight:800;
      padding:10px 14px;
      position:fixed;
      left:20px;
      top:18px;
      z-index:4;
    }
    .worklog-button {
      background:rgba(15,23,42,.76);
      border:1px solid rgba(147,197,253,.45);
      border-radius:999px;
      color:#e0f2fe;
      cursor:pointer;
      font-weight:800;
      padding:10px 14px;
      position:fixed;
      right:20px;
      top:18px;
      z-index:4;
    }
    .llm-panel {
      background:linear-gradient(180deg,rgba(15,23,42,.98),rgba(17,28,49,.98));
      border-right:1px solid rgba(125,211,252,.35);
      bottom:0;
      box-shadow:24px 0 60px rgba(0,0,0,.38);
      color:#e5eefb;
      display:grid;
      gap:14px;
      grid-auto-rows:max-content;
      max-width:420px;
      overflow:auto;
      padding:22px;
      position:fixed;
      left:0;
      top:0;
      transform:translateX(-104%);
      transition:transform .2s ease;
      width:min(88vw,420px);
      z-index:5;
    }
    .worklog-panel {
      background:linear-gradient(180deg,rgba(15,23,42,.98),rgba(17,28,49,.98));
      border-left:1px solid rgba(147,197,253,.35);
      bottom:0;
      box-shadow:-24px 0 60px rgba(0,0,0,.38);
      color:#e5eefb;
      display:grid;
      gap:14px;
      grid-auto-rows:max-content;
      max-width:380px;
      padding:22px;
      position:fixed;
      right:0;
      top:0;
      transform:translateX(104%);
      transition:transform .2s ease;
      width:min(86vw,380px);
      z-index:5;
    }
    #llm-toggle:checked ~ .llm-panel { transform:translateX(0); }
    #worklog-toggle:checked ~ .worklog-panel { transform:translateX(0); }
    .llm-close { color:#67e8f9; cursor:pointer; font-weight:800; justify-self:end; }
    .worklog-close { color:#93c5fd; cursor:pointer; font-weight:800; justify-self:end; }
    .llm-panel .brand-kicker { color:#67e8f9; }
    .llm-panel h2 { color:#f8fafc; font-size:24px; }
    .llm-api-form {
      background:#0b1220;
      border:1px solid rgba(148,163,184,.24);
      border-radius:8px;
      display:grid;
      gap:7px;
      padding:12px;
    }
    .llm-api-form h3 { color:#f8fafc; margin:0; text-transform:none; }
    .llm-check { align-items:center; display:flex; gap:8px; margin:6px 0; }
    .llm-check input { min-height:0; width:auto; }
    .llm-extra-toggle { height:1px; opacity:0; position:absolute; width:1px; }
    .llm-add-button {
      align-items:center;
      background:#172033;
      border:1px solid #334155;
      border-radius:999px;
      color:#e2e8f0;
      cursor:pointer;
      display:inline-flex;
      font-size:24px;
      font-weight:800;
      height:38px;
      justify-content:center;
      line-height:1;
      width:38px;
    }
    .llm-extra-slot { display:none; }
    .llm-extra-toggle:checked ~ .llm-add-button { background:#0e7490; border-color:#22d3ee; color:#ecfeff; }
    .llm-extra-toggle:checked ~ .llm-extra-slot { display:block; }
    .llm-panel .secondary, .llm-panel button { width:100%; }
    .llm-panel code { background:#111827; border-color:#334155; color:#cbd5e1; }
    .worklog-panel .brand-kicker { color:#93c5fd; }
    .worklog-panel h2 { color:#f8fafc; font-size:24px; }
    .worklog-panel ul { display:grid; gap:10px; margin:0; padding:0; }
    .worklog-panel li { border-bottom:1px solid rgba(148,163,184,.24); display:grid; gap:4px; list-style:none; padding-bottom:10px; }
    .worklog-panel li b { color:#f8fafc; }
    .worklog-panel li span { color:#b6c6da; font-size:13px; line-height:1.45; }
    .worklog-panel .secondary { background:#172033; border-color:#334155; color:#e2e8f0; }
    @media (prefers-reduced-motion: reduce) { *, *::before, *::after { scroll-behavior:auto !important; transition:none !important; } }
    @media (max-width:1100px) { .shell, .split, .graph-workspace { grid-template-columns:1fr; } .metrics { grid-template-columns:repeat(3,minmax(0,1fr)); } .workflow, .flow-rail { grid-template-columns:repeat(3,minmax(0,1fr)); } .step, .flow-node { border-bottom:1px solid var(--line); } .home-hero, .hero-strip { align-items:flex-start; flex-direction:column; } .module-grid, .paper-card-grid, .wiki-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } }
    @media (max-width:1100px) { .home-button-grid { grid-template-columns:repeat(3,minmax(150px,1fr)); } .home-module-button { min-height:220px; } }
    @media (max-width:820px) { .home-brand { justify-items:start; text-align:left; } .home-button-grid { grid-template-columns:repeat(2,minmax(150px,1fr)); } .home-module-button { min-height:190px; } .home-module-button::before { width:104px; height:58px; opacity:.22; } }
    @media (max-width:640px) { .topbar { align-items:flex-start; flex-direction:column; padding:14px 12px; } .shell, .home-shell { padding:10px 10px 30px; } .panel { padding:10px; } .metrics, .workflow, .flow-rail, .hero-counters, .module-grid, .empty-grid, .home-button-grid, .paper-card-grid, .wiki-grid { grid-template-columns:1fr; } .action-form { grid-template-columns:1fr; } .llm-button { left:14px; top:14px; } .worklog-button { right:14px; top:14px; } #knowledge-graph { min-height:360px; } .rag-result, .paper-card, .wiki-item { padding:9px; } .home-minimal { gap:20px; padding:34px 14px; } .home-module-button { min-height:126px; padding:14px 96px 14px 14px; } .home-module-button::before { inset:auto 8px 10px auto; transform:none; width:92px; height:48px; } .home-module-button:hover::before, .home-module-button:focus-visible::before { transform:translateY(-3px); } .home-module-button:hover, .home-module-button:focus-visible, .home-module-button.ready:hover, .home-module-button.ready:focus-visible { transform:translateY(-2px) scale(1.02); } }
    """


def render_home(*, message: str | None = None, error: str | None = None) -> str:
    state = load_state()
    latest_goal = state["goals"][0] if state["goals"] else None
    latest_round = state["active_round"] or (state["rounds"][0] if state["rounds"] else None)
    selected_goal_id = latest_goal["id"] if latest_goal else ""
    selected_round_id = latest_round["id"] if latest_round else ""
    metrics = state["metrics"]
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>v3 文献探索工作台</title>
  <style>
    :root {{
      color-scheme: light;
      --bg:#eef3f8;
      --surface:#ffffff;
      --surface-soft:#f8fafc;
      --line:#c9d7e8;
      --line-strong:#8ea3bc;
      --text:#0f172a;
      --muted:#64748b;
      --nav:#0f172a;
      --nav-2:#111c31;
      --nav-line:#243247;
      --primary:#1e40af;
      --primary-soft:#dbeafe;
      --secondary:#3b82f6;
      --accent:#d97706;
      --amber-soft:#fff7e5;
      --ok:#16a34a;
      --ok-soft:#dcfce7;
      --cyan:#0891b2;
      --cyan-soft:#cffafe;
      --red:#dc2626;
      --red-soft:#fff1f1;
      --shadow:0 12px 30px rgba(15,23,42,.08);
    }}
    * {{ box-sizing:border-box; }}
    body {{
      background-color:var(--bg);
      background-image:linear-gradient(rgba(30,64,175,.045) 1px, transparent 1px), linear-gradient(90deg, rgba(30,64,175,.045) 1px, transparent 1px);
      background-size:28px 28px;
      color:var(--text);
      font-family:"Fira Sans", Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size:14px;
      margin:0;
    }}
    header {{ background:var(--nav); border-bottom:1px solid #1d4ed8; box-shadow:0 12px 34px rgba(15,23,42,.22); position:sticky; top:0; z-index:10; }}
    .topbar {{ align-items:center; display:flex; gap:16px; justify-content:space-between; margin:0 auto; max-width:1560px; padding:12px 18px; }}
    .brand-kicker {{ color:#93c5fd; font-family:"Fira Code", ui-monospace, SFMono-Regular, Menlo, monospace; font-size:11px; font-weight:700; letter-spacing:.08em; margin-bottom:4px; text-transform:uppercase; }}
    h1 {{ color:#f8fafc; font-size:22px; letter-spacing:0; line-height:1.2; margin:0; }}
    h2 {{ font-size:15px; letter-spacing:0; margin:0; }}
    h3 {{ color:var(--muted); font-size:12px; font-weight:650; letter-spacing:.02em; margin:0 0 8px; text-transform:uppercase; }}
    a {{ color:var(--primary); }}
    .subtle {{ color:var(--muted); font-size:12px; line-height:1.45; }}
    .header-actions {{ align-items:center; display:flex; flex-wrap:wrap; gap:8px; justify-content:flex-end; }}
    header .subtle {{ color:#94a3b8; }}
    .shell {{ display:grid; gap:10px; grid-template-columns:320px minmax(0,1fr); margin:0 auto; max-width:1560px; padding:12px 18px 42px; }}
    .full {{ grid-column:1 / -1; }}
    .hero-strip {{
      align-items:center;
      background:linear-gradient(135deg,#0f172a,#12284d 62%,#1e40af);
      border:1px solid #274b8d;
      border-radius:8px;
      box-shadow:0 18px 42px rgba(15,23,42,.22);
      color:#f8fafc;
      display:flex;
      gap:20px;
      justify-content:space-between;
      min-height:104px;
      overflow:hidden;
      padding:18px;
      position:relative;
    }}
    .hero-strip::after {{
      background:linear-gradient(90deg, transparent, rgba(255,255,255,.12), transparent);
      content:"";
      height:100%;
      position:absolute;
      right:18%;
      top:0;
      transform:skewX(-18deg);
      width:120px;
    }}
    .hero-strip h2 {{ color:#f8fafc; font-size:22px; line-height:1.25; margin:0 0 6px; }}
    .hero-strip .subtle {{ color:#cbd5e1; max-width:760px; }}
    .hero-counters {{ display:grid; gap:8px; grid-template-columns:repeat(3,minmax(96px,1fr)); position:relative; z-index:1; }}
    .hero-counters span {{ background:rgba(15,23,42,.42); border:1px solid rgba(147,197,253,.28); border-radius:8px; color:#cbd5e1; font-size:11px; padding:10px; text-transform:uppercase; }}
    .hero-counters b {{ color:#f8fafc; display:block; font-family:"Fira Code", ui-monospace, SFMono-Regular, Menlo, monospace; font-size:22px; margin-bottom:2px; }}
    .panel {{ background:rgba(255,255,255,.96); border:1px solid var(--line); border-radius:8px; box-shadow:var(--shadow); min-width:0; padding:12px; }}
    aside .panel {{ background:var(--nav-2); border-color:var(--nav-line); color:#e5eefb; box-shadow:0 14px 32px rgba(15,23,42,.18); }}
    aside .subtle, aside label {{ color:#94a3b8; }}
    aside h2 {{ color:#f8fafc; }}
    .panel-head {{ align-items:center; border-bottom:1px solid rgba(148,163,184,.22); display:flex; gap:10px; justify-content:space-between; margin:-2px 0 10px; padding-bottom:9px; }}
    .stack {{ display:grid; gap:10px; }}
    .metrics {{ display:grid; gap:8px; grid-template-columns:repeat(6,minmax(0,1fr)); }}
    .metric {{ background:linear-gradient(180deg,#ffffff,#f8fbff); border:1px solid var(--line); border-top:4px solid var(--primary); border-radius:8px; min-height:78px; padding:10px 11px; position:relative; }}
    .metric:nth-child(2) {{ border-top-color:var(--secondary); }}
    .metric:nth-child(3) {{ border-top-color:var(--cyan); }}
    .metric:nth-child(4) {{ border-top-color:var(--ok); }}
    .metric:nth-child(5) {{ border-top-color:var(--accent); }}
    .metric:nth-child(6) {{ border-top-color:var(--red); }}
    .metric-value {{ font-family:"Fira Code", ui-monospace, SFMono-Regular, Menlo, monospace; font-size:27px; font-weight:760; line-height:1; margin-bottom:6px; }}
    .metric-label {{ color:var(--muted); font-size:12px; }}
    .metric-rule {{ background:#e2e8f0; border-radius:999px; bottom:9px; height:3px; left:11px; overflow:hidden; position:absolute; right:11px; }}
    .metric-rule span {{ background:var(--primary); display:block; height:100%; width:62%; }}
    .metric:nth-child(4) .metric-rule span {{ background:var(--ok); width:76%; }}
    .metric:nth-child(5) .metric-rule span {{ background:var(--accent); width:42%; }}
    .workflow {{ background:var(--nav); border:1px solid var(--nav-line); border-radius:8px; display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); overflow:hidden; }}
    .step {{ border-right:1px solid var(--nav-line); color:#cbd5e1; display:grid; gap:5px; min-height:58px; padding:9px 11px; position:relative; }}
    .step:last-child {{ border-right:0; }}
    .step.active {{ background:#102a43; color:#f8fafc; }}
    .step.active::before {{ background:var(--ok); bottom:0; content:""; left:0; position:absolute; top:0; width:3px; }}
    .step.done {{ background:#111f35; color:#dbeafe; }}
    .step-num {{ color:#93c5fd; font-family:"Fira Code", ui-monospace, SFMono-Regular, Menlo, monospace; font-size:10px; font-weight:700; }}
    .step-name {{ font-size:13px; font-weight:750; }}
    .notice {{ border-radius:8px; font-size:13px; padding:10px 12px; }}
    .busy {{ background:#eff6ff; border:1px solid #93c5fd; color:#1e3a8a; display:none; }}
    .ok {{ background:var(--ok-soft); border:1px solid #86efac; }}
    .err {{ background:var(--red-soft); border:1px solid #e1a3a0; }}
    label {{ color:var(--muted); display:block; font-size:12px; font-weight:650; margin:8px 0 4px; }}
    input, textarea {{ background:#fff; border:1px solid #c9d3df; border-radius:7px; color:var(--text); font:inherit; min-height:34px; padding:7px 9px; width:100%; }}
    aside input, aside textarea {{ background:#0b1220; border-color:#334155; color:#f8fafc; }}
    textarea {{ min-height:76px; resize:vertical; }}
    input:focus, textarea:focus {{ border-color:var(--secondary); box-shadow:0 0 0 3px rgba(59,130,246,.18); outline:none; }}
    button, a.button {{ align-items:center; background:var(--primary); border:1px solid var(--primary); border-radius:7px; color:#fff; cursor:pointer; display:inline-flex; font:inherit; font-size:13px; font-weight:700; justify-content:center; min-height:34px; padding:7px 10px; text-decoration:none; transition:background .16s ease, border-color .16s ease, box-shadow .16s ease, transform .16s ease; }}
    button:hover, a.button:hover {{ background:#1d4ed8; border-color:#1d4ed8; transform:translateY(-1px); }}
    button:focus-visible, a.button:focus-visible {{ box-shadow:0 0 0 3px rgba(59,130,246,.25); outline:none; }}
    button.danger {{ background:var(--red); border-color:var(--red); min-height:28px; padding:4px 8px; }}
    button.danger:hover {{ background:#b91c1c; border-color:#b91c1c; }}
    button[disabled] {{ cursor:wait; opacity:.72; transform:none; }}
    .secondary {{ background:#f8fafc; border-color:#475569; color:#0f172a; }}
    header .secondary {{ background:#172033; border-color:#334155; color:#e2e8f0; }}
    .secondary:hover {{ background:#eff6ff; border-color:var(--secondary); }}
    .action-row {{ display:grid; gap:8px; grid-template-columns:1fr; }}
    .action-form {{ display:grid; gap:7px; grid-template-columns:minmax(0,1fr) auto; }}
    .action-form input {{ min-width:0; }}
    .status-line {{ align-items:center; display:flex; flex-wrap:wrap; gap:8px; margin-top:6px; }}
    .badge {{ align-items:center; border-radius:999px; display:inline-flex; font-size:12px; font-weight:700; line-height:1; min-height:24px; padding:5px 8px; white-space:nowrap; }}
    .badge-neutral {{ background:#e2e8f0; color:#334155; }}
    header .badge-neutral {{ background:#1e293b; color:#cbd5e1; }}
    .badge-teal {{ background:var(--ok-soft); color:#166534; }}
    .badge-blue {{ background:var(--primary-soft); color:#1e40af; }}
    .badge-amber {{ background:var(--amber-soft); color:var(--amber); }}
    .badge-red {{ background:var(--red-soft); color:var(--red); }}
    .table-wrap {{ border:1px solid var(--line); border-radius:8px; max-height:440px; overflow:auto; }}
    table {{ border-collapse:collapse; font-size:12.5px; width:100%; }}
    th,td {{ border-bottom:1px solid var(--line); padding:7px 8px; text-align:left; vertical-align:top; }}
    th {{ background:#eaf1fb; color:#1e3a8a; font-size:11px; font-weight:800; letter-spacing:.02em; position:sticky; text-transform:uppercase; top:0; z-index:1; }}
    tr:last-child td {{ border-bottom:0; }}
    tr:hover td {{ background:#f0f7ff; }}
    .title-cell {{ min-width:260px; }}
    .num {{ font-family:"Fira Code", ui-monospace, SFMono-Regular, Menlo, monospace; font-variant-numeric:tabular-nums; text-align:right; white-space:nowrap; }}
    code {{ background:#f4f6f8; border:1px solid #e0e5eb; border-radius:6px; color:#314154; display:block; font-size:12px; overflow-x:auto; padding:7px 8px; white-space:nowrap; }}
    .list {{ display:grid; gap:8px; margin:0; padding:0; }}
    .list li {{ border-bottom:1px solid var(--line); list-style:none; padding:8px 0; }}
    .list li:last-child {{ border-bottom:0; }}
    .split {{ display:grid; gap:10px; grid-template-columns:minmax(0,1.25fr) minmax(260px,.75fr); }}
    @media (prefers-reduced-motion: reduce) {{ *, *::before, *::after {{ scroll-behavior:auto !important; transition:none !important; }} }}
    @media (max-width:1100px) {{ .shell, .split {{ grid-template-columns:1fr; }} .metrics {{ grid-template-columns:repeat(3,minmax(0,1fr)); }} .workflow {{ grid-template-columns:repeat(3,minmax(0,1fr)); }} .step {{ border-bottom:1px solid var(--line); }} .hero-strip {{ align-items:flex-start; flex-direction:column; }} }}
    @media (max-width:640px) {{ .topbar {{ align-items:flex-start; flex-direction:column; padding:14px 16px; }} .shell {{ padding:12px 16px 32px; }} .metrics, .workflow, .hero-counters {{ grid-template-columns:1fr 1fr; }} .action-form {{ grid-template-columns:1fr; }} }}
  </style>
  <script>
    window.addEventListener('DOMContentLoaded', () => {{
      const busy = document.querySelector('[data-busy]');
      document.querySelectorAll('form').forEach((form) => {{
        form.addEventListener('submit', () => {{
          if (busy) {{
            const button = form.querySelector('button[type="submit"], button:not([type])');
            const label = button ? button.textContent.trim() : '任务';
            busy.textContent = `正在执行：${{label}}。外部检索、PDF 获取或扫描文件夹可能需要几十秒，请不要重复点击。`;
            busy.style.display = 'block';
          }}
          form.querySelectorAll('button').forEach((button) => {{
            button.disabled = true;
            if (!button.classList.contains('danger')) button.textContent = '执行中...';
          }});
        }});
      }});
    }});
  </script>
</head>
<body>
  <header>
    <div class="topbar">
      <div>
        <div class="brand-kicker">Data-Dense Research Command Center</div>
        <h1>v3 文献探索工作台</h1>
        <div class="status-line">
          {status_badge(latest_round["status"] if latest_round else None)}
          <span class="subtle">127.0.0.1:8765</span>
          <span class="subtle">推荐模式：{html.escape(state["selection_mode"])}</span>
          <span class="subtle">UI system: Data-Dense Dashboard</span>
        </div>
      </div>
      <div class="header-actions">
        <a class="button secondary" href="/">返回首页</a>
        <a class="button secondary" href="/reports/dashboard.html">静态仪表盘</a>
      </div>
    </div>
  </header>
  <main class="shell">
    <div class="notice busy full" data-busy></div>
    {notice(message, "ok") if message else ""}
    {notice(error, "err") if error else ""}
    <section class="full hero-strip">
      <div>
        <div class="brand-kicker">Permanent Magnet Literature Agent</div>
        <h2>{html.escape(str(latest_goal["title"])) if latest_goal else "尚未创建科学问题"}</h2>
        <p class="subtle">小批量候选、人工确认、开放 PDF 获取、手动 DOI 清单、分析综合与下一轮检索建议。</p>
      </div>
      <div class="hero-counters">
        <span><b>{metrics["candidates"]}</b> candidates</span>
        <span><b>{metrics["downloaded"]}</b> pdfs</span>
        <span><b>{metrics["manual_pending"]}</b> manual</span>
      </div>
    </section>
    <section class="full">{workflow_steps(latest_round)}</section>
    <section class="full metrics">
      {metric_card("科学问题", metrics["goals"])}
      {metric_card("轮次", metrics["rounds"])}
      {metric_card("本轮候选", metrics["candidates"])}
      {metric_card("已获 PDF", metrics["downloaded"])}
      {metric_card("手动任务", metrics["manual_pending"])}
      {metric_card("分析记录", metrics["analyses"])}
    </section>
    <aside class="stack">
      <section class="panel">
        <div class="panel-head"><h2>创建科学问题</h2></div>
        <form method="post" action="/goal/create">
          <label>科学问题</label>
          <input name="title" required placeholder="例如：NdFeB grain boundary diffusion for high coercivity">
          <label>补充说明</label>
          <textarea name="description" placeholder="材料体系、性能目标、排除范围等"></textarea>
          <label>每轮目标文献数</label>
          <input name="target_count" type="number" value="20" min="1" max="100">
          <button type="submit">创建科学问题</button>
        </form>
      </section>
      <section class="panel">
        <div class="panel-head"><h2>规划下一轮</h2></div>
        <form method="post" action="/round/plan">
          <label>科学问题 ID</label>
          <input name="goal_id" type="number" value="{html.escape(str(selected_goal_id))}" required>
          <label>目标文献数</label>
          <input name="target_count" type="number" value="20" min="1" max="100">
          <label>Query 数</label>
          <input name="query_limit" type="number" value="4" min="1" max="20">
          <label>每个 query 每源返回数</label>
          <input name="max_results_per_query" type="number" value="8" min="1" max="50">
          <button type="submit">规划本轮候选</button>
        </form>
      </section>
      <section class="panel">
        <div class="panel-head"><h2>轮次操作</h2>{small_id("轮次", selected_round_id)}</div>
        {round_hint(latest_round)}
        <div class="action-row">
          {round_action("/round/approve", selected_round_id, "确认本轮")}
          {round_action("/round/acquire", selected_round_id, "获取 PDF / 生成手动清单")}
          {round_action("/round/intake", selected_round_id, "扫描手动 PDF")}
          {round_action("/round/analyze", selected_round_id, "分析本轮")}
          {round_action("/round/propose-next", selected_round_id, "提出下一轮建议")}
        </div>
      </section>
    </aside>
    <section class="stack">
      <section class="panel">
        <div class="panel-head"><h2>当前状态</h2></div>
        {state_summary(state)}
      </section>
      <section class="panel">
        <div class="panel-head"><h2>本轮候选</h2><span class="subtle">{metrics["candidates"]} 条</span></div>
        {candidate_table(state["candidates"])}
      </section>
      <section class="split">
        <div class="panel">
          <div class="panel-head"><h2>手动下载任务</h2><span class="subtle">{metrics["manual_pending"]} 待处理</span></div>
          {manual_task_table(state["manual_tasks"])}
        </div>
        <div class="panel">
          <div class="panel-head"><h2>下一轮建议</h2></div>
          {synthesis_block(state["synthesis"])}
        </div>
      </section>
      <section class="split">
        <div class="panel">
          <div class="panel-head"><h2>最近轮次</h2></div>
          {round_table(state["rounds"])}
        </div>
        <div class="panel">
          <div class="panel-head"><h2>科学问题</h2></div>
          {goal_table(state["goals"])}
        </div>
      </section>
    </section>
  </main>
</body>
</html>"""


def load_state() -> dict[str, Any]:
    db = LiteratureDB(default_db_path())
    try:
        db.init_schema()
        goals = [dict(row) for row in db.rows("SELECT * FROM scientific_goals ORDER BY id DESC LIMIT 8")]
        rounds = [dict(row) for row in db.rows("SELECT * FROM exploration_rounds ORDER BY id DESC LIMIT 8")]
        active_round = _active_round(rounds)
        round_id = active_round["id"] if active_round else None
        candidates = [dict(row) for row in db.round_candidates(round_id)] if round_id else []
        manual_tasks = [dict(row) for row in db.manual_download_tasks(round_id)] if round_id else []
        metrics = _metrics(db, round_id, candidates, manual_tasks)
        return {
            "goals": goals,
            "rounds": rounds,
            "active_round": active_round,
            "candidates": candidates,
            "manual_tasks": manual_tasks,
            "synthesis": dict(db.round_synthesis(round_id)) if round_id and db.round_synthesis(round_id) else None,
            "metrics": metrics,
            "selection_mode": _selection_mode_label(),
        }
    finally:
        db.close()


def load_paper_analysis_state() -> dict[str, Any]:
    return load_paper_analysis_state_for_query("")


def load_paper_analysis_state_for_query(query: str) -> dict[str, Any]:
    db = LiteratureDB(default_db_path())
    try:
        db.init_schema()
        db_metrics = conversion_metrics(db)
        conversions = [dict(row) for row in db.paper_conversions(limit=80)]
        latest_rounds = [dict(row) for row in db.rows("SELECT * FROM exploration_rounds ORDER BY id DESC LIMIT 1")]
        index_status = _read_json_file(analysis_data_dir() / "index" / "index_status.json", {})
        graph = _read_json_file(analysis_graph_path(), {"nodes": [], "edges": []})
        cards = [_paper_card_view(row) for row in _safe_rows(db, "SELECT * FROM paper_cards ORDER BY updated_at DESC, paper_id LIMIT 24")]
        wiki_pages = [_wiki_page_view(row) for row in _safe_rows(db, "SELECT * FROM wiki_pages ORDER BY updated_at DESC, title LIMIT 24")]
        rag_results = search_with_analysis_agent(query, limit=10) if query.strip() else []
        analysis_counts = _analysis_counts(db, graph)
        metrics = {
            "pdf_total": int(db_metrics.get("pdf_total") or 0),
            "converted": int(db_metrics.get("converted") or 0),
            "failed": int(db_metrics.get("failed") or 0),
            "skipped": int(db_metrics.get("skipped") or 0),
            "sections": sum(int(row.get("section_count") or 0) for row in conversions),
            "references": sum(int(row.get("reference_count") or 0) for row in conversions),
            "chunks": analysis_counts["chunks"],
            "cards": analysis_counts["cards"],
            "nodes": analysis_counts["nodes"],
            "wiki_pages": analysis_counts["wiki_pages"],
        }
        return {
            "metrics": metrics,
            "conversions": conversions,
            "latest_round": latest_rounds[0] if latest_rounds else None,
            "cards": cards,
            "wiki_pages": wiki_pages,
            "graph": graph,
            "query": query,
            "rag_results": rag_results,
            "index_status": index_status,
        }
    finally:
        db.close()


def _analysis_counts(db: LiteratureDB, graph: dict[str, Any]) -> dict[str, int]:
    return {
        "chunks": _safe_count(db, "rag_chunks"),
        "cards": _safe_count(db, "paper_cards"),
        "nodes": len(graph.get("nodes", [])) or _safe_count(db, "knowledge_nodes"),
        "wiki_pages": _safe_count(db, "wiki_pages"),
    }


def _safe_count(db: LiteratureDB, table: str) -> int:
    try:
        return int(db.rows(f"SELECT COUNT(*) AS n FROM {table}")[0]["n"] or 0)
    except Exception:
        return 0


def _safe_rows(db: LiteratureDB, query: str, params: tuple[Any, ...] = ()) -> list[Any]:
    try:
        return db.rows(query, params)
    except Exception:
        return []


def _paper_card_view(row: Any) -> dict[str, Any]:
    card = json.loads(row["card_json"])
    card["_markdown_path"] = row["markdown_path"]
    return card


def _wiki_page_view(row: Any) -> dict[str, Any]:
    page = json.loads(row["page_json"])
    page["_markdown_path"] = row["markdown_path"]
    return page


def _read_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def notice(text: str, klass: str) -> str:
    return f'<div class="notice {klass} full">{html.escape(text)}</div>'


def metric_card(label: str, value: object) -> str:
    return (
        '<div class="metric">'
        f'<div class="metric-value">{html.escape(str(value))}</div>'
        f'<div class="metric-label">{html.escape(label)}</div>'
        "</div>"
    )


def small_id(label: str, value: object) -> str:
    if value in (None, ""):
        return '<span class="badge badge-neutral">未选择</span>'
    return f'<span class="badge badge-neutral">{html.escape(label)} {html.escape(str(value))}</span>'


def round_action(action: str, round_id: object, label: str) -> str:
    return (
        f'<form class="action-form" method="post" action="{html.escape(action)}">'
        f'<input aria-label="轮次 ID" name="round_id" type="number" value="{html.escape(str(round_id))}" required>'
        f'<button type="submit">{html.escape(label)}</button>'
        "</form>"
    )


def status_badge(status: object | None) -> str:
    if not status:
        return '<span class="badge badge-neutral">尚未开始</span>'
    text = str(status)
    klass = {
        "awaiting_user_approval": "badge-amber",
        "approved": "badge-teal",
        "acquiring_pdfs": "badge-blue",
        "awaiting_manual_pdfs": "badge-amber",
        "analyzing": "badge-blue",
        "synthesized": "badge-teal",
        "next_round_proposed": "badge-teal",
        "needs_retry": "badge-red",
    }.get(text, "badge-neutral")
    return f'<span class="badge {klass}">{html.escape(_status_label(text))}</span>'


def round_hint(round_row: dict[str, Any] | None) -> str:
    if not round_row:
        return '<p class="subtle">先创建科学问题并规划一轮候选。</p>'
    status = str(round_row["status"])
    if status == "planned":
        return '<p class="subtle">这一轮还没有进入候选确认状态，通常是检索还没成功完成或候选数为 0。请重新规划，或选择一个“待人工确认”的轮次。</p>'
    if status == "needs_retry":
        return '<p class="subtle">这一轮需要重试。常见原因是外部数据源 429 限流、timeout，或本轮没有选出候选。</p>'
    if status == "awaiting_user_approval":
        return '<p class="subtle">请先确认本轮，确认后才能获取 PDF。</p>'
    if status == "approved":
        return '<p class="subtle">本轮已确认，可以获取开放 PDF，并生成不能自动下载的 DOI 清单。</p>'
    return ""


def workflow_steps(latest_round: dict[str, Any] | None) -> str:
    status = str(latest_round["status"]) if latest_round else ""
    steps = [
        ("创建问题", {"planned", "awaiting_user_approval", "approved", "acquiring_pdfs", "awaiting_manual_pdfs", "analyzing", "synthesized", "next_round_proposed"}),
        ("规划候选", {"awaiting_user_approval", "approved", "acquiring_pdfs", "awaiting_manual_pdfs", "analyzing", "synthesized", "next_round_proposed"}),
        ("人工确认", {"approved", "acquiring_pdfs", "awaiting_manual_pdfs", "analyzing", "synthesized", "next_round_proposed"}),
        ("获取 PDF", {"acquiring_pdfs", "awaiting_manual_pdfs", "analyzing", "synthesized", "next_round_proposed"}),
        ("分析综合", {"analyzing", "synthesized", "next_round_proposed"}),
        ("下一轮建议", {"next_round_proposed"}),
    ]
    active_map = {
        "planned": 1,
        "awaiting_user_approval": 2,
        "approved": 3,
        "acquiring_pdfs": 4,
        "awaiting_manual_pdfs": 4,
        "analyzing": 5,
        "synthesized": 5,
        "next_round_proposed": 6,
        "needs_retry": 2,
    }
    active = active_map.get(status, 0)
    cells = []
    for index, (name, done_statuses) in enumerate(steps, start=1):
        klass = "step active" if index == active else "step done" if status in done_statuses and index < active else "step"
        cells.append(
            f'<div class="{klass}"><div class="step-num">STEP {index}</div><div class="step-name">{html.escape(name)}</div></div>'
        )
    return '<div class="workflow">' + "".join(cells) + "</div>"


def state_summary(state: dict[str, Any]) -> str:
    goals = state["goals"]
    latest_goal = goals[0] if goals else None
    latest_round = state.get("active_round")
    lines = []
    if latest_goal:
        lines.append(f'<div class="status-line"><b>科学问题 {latest_goal["id"]}</b><span>{html.escape(str(latest_goal["title"]))}</span></div>')
        if latest_goal.get("description"):
            lines.append(f'<p class="subtle">{html.escape(str(latest_goal["description"]))}</p>')
    else:
        lines.append('<p class="subtle">尚未创建科学问题。</p>')
    if latest_round:
        lines.append(
            '<div class="status-line">'
            f'<b>轮次 {latest_round["id"]}</b>'
            f'{status_badge(latest_round["status"])}'
            f'<span class="subtle">目标文献数 {latest_round["target_count"]}</span>'
            "</div>"
        )
    return "".join(lines)


def candidate_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<div class="subtle">暂无候选。先规划一轮。</div>'
    body = "".join(
        "<tr>"
        f"<td class=\"num\">{row['rank']}</td>"
        f"<td class=\"title-cell\">{html.escape(str(row['title'] or ''))}<div class=\"subtle\">{html.escape(str(row['venue'] or ''))}</div></td>"
        f"<td class=\"num\">{html.escape(str(row['year'] or ''))}</td>"
        f"<td class=\"num\">{float(row['selection_score']):.2f}</td>"
        f"<td>{status_badge(row.get('evidence_level'))}</td>"
        f"<td>{pdf_path_cell(row)}</td>"
        f"<td>{html.escape(str(row['selection_reason'] or ''))}</td>"
        "</tr>"
        for row in rows[:20]
    )
    return f'<div class="table-wrap"><table><thead><tr><th>序</th><th>标题</th><th>年份</th><th>分数</th><th>证据</th><th>PDF路径</th><th>选择理由</th></tr></thead><tbody>{body}</tbody></table></div>'


def pdf_path_cell(row: dict[str, Any]) -> str:
    path = row.get("local_pdf_path")
    if not path:
        return '<span class="subtle">未获得</span>'
    file_path = Path(str(path))
    if file_path.exists() and file_path.stat().st_size > 0:
        return f'<code>{html.escape(str(file_path))}</code>'
    return f'<code>{html.escape(str(file_path))}</code><div class="subtle">文件缺失，请重新获取或手动放入</div>'


def manual_task_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<div class="subtle">暂无手动下载任务。</div>'
    body = "".join(
        "<tr>"
        f"<td class=\"title-cell\">{html.escape(str(row['title'] or ''))}<div class=\"subtle\">{html.escape(str(row['venue'] or ''))}</div></td>"
        f"<td>{doi_link(row.get('doi'))}</td>"
        f"<td><code>{html.escape(str(Path(row['target_path']).parent))}</code></td>"
        f"<td><code>{html.escape(str(Path(row['target_path']).name))}</code></td>"
        f"<td>{status_badge(row.get('status'))}</td>"
        "</tr>"
        for row in rows[:20]
    )
    return f'<div class="subtle">把下载到的 PDF 放进“文件夹”即可，文件名可以是浏览器默认名；扫描后系统会尽量匹配并复制为推荐文件名。</div><div class="table-wrap"><table><thead><tr><th>标题</th><th>DOI</th><th>文件夹</th><th>推荐文件名</th><th>状态</th></tr></thead><tbody>{body}</tbody></table></div>'


def conversion_summary(state: dict[str, Any]) -> str:
    metrics = state["metrics"]
    pending = max(0, int(metrics["pdf_total"]) - int(metrics["converted"]) - int(metrics["failed"]))
    return (
        '<div class="status-line">'
        f'<span class="badge badge-blue">待处理 {pending}</span>'
        f'<span class="badge badge-teal">已转换 {metrics["converted"]}</span>'
        f'<span class="badge badge-red">失败 {metrics["failed"]}</span>'
        f'<span class="subtle">解析器：PyMuPDF + 规则结构化 v1</span>'
        "</div>"
    )


def embedding_mode_label(state: dict[str, Any]) -> str:
    mode = str(state.get("index_status", {}).get("embedding_mode") or "")
    if mode == "configured":
        return "BM25 + Embedding"
    if mode == "local_hash_fallback":
        return "BM25 + 本地向量"
    return "BM25 待构建"


def rag_results_block(state: dict[str, Any]) -> str:
    query = str(state.get("query") or "").strip()
    rows = state.get("rag_results", [])
    if not query:
        return '<div class="subtle">输入关键词后检索段落、图注、表格和参考文献。构建知识库后可搜索 “coercivity”、“NdFeB”、“micromagnetic”等主题。</div>'
    if not rows:
        return '<div class="subtle">没有检索结果。请先点击“构建知识库”，或换一个关键词。</div>'
    items = []
    for row in rows:
        items.append(
            '<article class="rag-result">'
            '<div class="status-line">'
            f'<span class="badge badge-blue">{html.escape(str(row.get("chunk_type") or ""))}</span>'
            f'<span class="subtle">score {html.escape(str(row.get("score") or 0))}</span>'
            f'<span class="subtle">page {html.escape(str(row.get("page") or ""))}</span>'
            '</div>'
            f'<h3>{html.escape(str(row.get("title") or "Untitled"))}</h3>'
            f'<div class="subtle">{html.escape(str(row.get("section_path") or ""))}</div>'
            f'<p>{html.escape(_truncate(str(row.get("text") or ""), 520))}</p>'
            f'{artifact_link(row.get("source_json_path"), "源 JSON")}'
            '</article>'
        )
    return '<div class="rag-list">' + "".join(items) + "</div>"


def paper_cards_block(cards: list[dict[str, Any]]) -> str:
    if not cards:
        return '<div class="subtle">暂无 Paper Cards。点击“构建知识库”会从 parsed JSON 生成每篇论文的结构化卡片。</div>'
    body = []
    for card in cards[:12]:
        chips = card_chips(card, "materials") + card_chips(card, "methods") + card_chips(card, "properties")
        claims = "".join(
            f'<li>{html.escape(_truncate(str(claim.get("text") or ""), 190))}<div class="subtle">{html.escape(", ".join(claim.get("evidence_ids", [])))}</div></li>'
            for claim in card.get("claims", [])[:3]
        )
        body.append(
            '<article class="paper-card">'
            f'<h3>{html.escape(str(card.get("title") or "Untitled"))}</h3>'
            f'<p>{html.escape(_truncate(str(card.get("summary") or ""), 260))}</p>'
            f'<div class="chip-row">{chips or "<span class=\"subtle\">暂无关键词</span>"}</div>'
            f'<ul class="mini-list">{claims}</ul>'
            '<div class="status-line">'
            f'{artifact_link(card.get("source_json_path"), "源 JSON")}'
            f'{artifact_link(card.get("_markdown_path"), "Card MD")}'
            '</div>'
            '</article>'
        )
    return '<div class="paper-card-grid">' + "".join(body) + "</div>"


def card_chips(card: dict[str, Any], key: str) -> str:
    klass = {"materials": "chip-material", "methods": "chip-method", "properties": "chip-property"}.get(key, "")
    return "".join(f'<span class="topic-chip {klass}">{html.escape(str(item))}</span>' for item in card.get(key, [])[:4])


def wiki_pages_block(pages: list[dict[str, Any]]) -> str:
    if not pages:
        return '<div class="subtle">暂无 Wiki 条目。图谱主题生成后，Wiki 会围绕材料、方法、性能等 topic 生成证据化条目。</div>'
    body = []
    for page in pages[:16]:
        findings = "".join(
            f'<li>{html.escape(_truncate(str(item.get("claim") or ""), 180))}<div class="subtle">{html.escape(", ".join(item.get("evidence_ids", [])))}</div></li>'
            for item in page.get("known_findings", [])[:2]
        )
        body.append(
            '<article class="wiki-item">'
            f'<h3>{html.escape(str(page.get("title") or "Untitled"))}</h3>'
            f'<p>{html.escape(_truncate(str(page.get("summary") or ""), 230))}</p>'
            f'<ul class="mini-list">{findings}</ul>'
            f'{artifact_link(page.get("_markdown_path"), "Wiki MD")}'
            '</article>'
        )
    return '<div class="wiki-grid">' + "".join(body) + "</div>"


def retrieval_context_block(context: dict[str, Any]) -> str:
    goal = context.get("latest_goal") or {}
    questions = list(context.get("retrieval_questions") or [])
    gaps = list(context.get("evidence_gaps") or [])
    candidates = list(context.get("candidate_papers") or [])
    parts: list[str] = []
    if goal:
        description = ""
        if goal.get("description"):
            description = f'<div class="subtle">{html.escape(str(goal.get("description") or ""))}</div>'
        parts.append(
            '<article class="context-card">'
            f'<h3>当前科学问题</h3><p>{html.escape(str(goal.get("title") or ""))}</p>'
            f'{description}'
            '</article>'
        )
    if questions:
        parts.append("<h3>检索问题 / Query</h3>" + compact_list(questions[:8]))
    else:
        parts.append('<div class="subtle">暂无检索问题。请先在 01 模块创建科学问题或完成一轮检索。</div>')
    if gaps:
        parts.append("<h3>证据缺口</h3>" + compact_list(gaps[:8]))
    if candidates:
        items = [
            f"{row.get('rank', '')}. {row.get('title', '')} ({row.get('year') or 'n.d.'})"
            for row in candidates[:8]
        ]
        parts.append("<h3>最近候选文献</h3>" + compact_list(items))
    return "".join(parts)


def analysis_context_block(context: dict[str, Any]) -> str:
    cards = list(context.get("paper_cards") or [])
    pages = list(context.get("wiki_pages") or [])
    parts: list[str] = []
    if cards:
        parts.append("<h3>Paper Cards</h3>")
        for card in cards[:4]:
            chips = "".join(
                f'<span class="topic-chip">{html.escape(str(item))}</span>'
                for item in (list(card.get("materials") or []) + list(card.get("methods") or []) + list(card.get("properties") or []))[:6]
            )
            parts.append(
                '<article class="context-card">'
                f'<h3>{html.escape(str(card.get("title") or "Untitled"))}</h3>'
                f'<p>{html.escape(_truncate(str(card.get("summary") or ""), 220))}</p>'
                f'<div class="chip-row">{chips}</div>'
                '</article>'
            )
    if pages:
        parts.append("<h3>Wiki 开放问题</h3>")
        open_questions = [item for page in pages for item in list(page.get("open_questions") or [])]
        if open_questions:
            parts.append(compact_list(open_questions[:8]))
        else:
            parts.append(
                "".join(
                    '<article class="context-card">'
                    f'<h3>{html.escape(str(page.get("title") or ""))}</h3>'
                    f'<p>{html.escape(_truncate(str(page.get("summary") or ""), 180))}</p>'
                    '</article>'
                    for page in pages[:4]
                )
            )
    if not parts:
        return '<div class="subtle">暂无文献分析结果。请先在 02 模块转换 PDF 并构建知识库。</div>'
    return "".join(parts)


def question_chat_messages(messages: list[dict[str, Any]]) -> str:
    if not messages:
        return '<div class="subtle">正在等待上下文载入。</div>'
    bubbles = []
    for message in messages:
        role = str(message.get("role") or "")
        speaker = str(message.get("speaker") or "")
        content = str(message.get("content") or "")
        model = str(message.get("model") or "")
        if role == "retrieval":
            klass = "bubble bubble-user bubble-retrieval"
        elif role == "user":
            klass = "bubble bubble-user"
        else:
            klass = "bubble bubble-left bubble-llm"
        meta = f'<span>{html.escape(speaker)}</span>'
        if model:
            meta += f'<span>模型：{html.escape(model)}</span>'
        suggestions = ""
        if role == "assistant":
            suggested_questions = list((message.get("metadata") or {}).get("suggested_questions") or [])
            if not suggested_questions:
                suggested_questions = default_frontend_suggestions()
            suggestions = suggested_question_forms(suggested_questions[:3])
        bubbles.append(
            f'<article class="{klass}">'
            f'<div class="bubble-meta">{meta}</div>'
            f'<div class="bubble-content">{format_chat_text(content)}</div>'
            '</article>'
            f'{suggestions}'
        )
    return "".join(bubbles)


def compact_list(items: list[object]) -> str:
    if not items:
        return '<div class="subtle">暂无。</div>'
    return '<ul class="mini-list">' + "".join(f'<li>{html.escape(str(item))}</li>' for item in items if str(item).strip()) + "</ul>"


def format_chat_text(text: str) -> str:
    escaped = html.escape(text.strip())
    if not escaped:
        return ""
    paragraphs = []
    buffer: list[str] = []
    list_items: list[str] = []
    for raw_line in escaped.splitlines():
        line = raw_line.strip()
        if not line:
            if list_items:
                paragraphs.append('<ul class="chat-list">' + "".join(list_items) + "</ul>")
                list_items = []
            if buffer:
                paragraphs.append(render_chat_paragraph(buffer))
                buffer = []
            continue
        heading = chat_heading(line)
        if heading:
            if list_items:
                paragraphs.append('<ul class="chat-list">' + "".join(list_items) + "</ul>")
                list_items = []
            if buffer:
                paragraphs.append(render_chat_paragraph(buffer))
                buffer = []
            paragraphs.append(heading)
            continue
        if line.startswith("- ") or line[:3] in {"1. ", "2. ", "3. ", "4. ", "5. ", "6. ", "7. ", "8. ", "9. "}:
            if buffer:
                paragraphs.append(render_chat_paragraph(buffer))
                buffer = []
            item_text = line[2:] if line.startswith("- ") else line[3:]
            list_items.append(f"<li>{inline_markdown(item_text)}</li>")
        else:
            buffer.append(line)
    if list_items:
        paragraphs.append('<ul class="chat-list">' + "".join(list_items) + "</ul>")
    if buffer:
        paragraphs.append(render_chat_paragraph(buffer))
    return "".join(paragraphs)


def render_chat_paragraph(lines: list[str]) -> str:
    return "<p>" + "<br>".join(inline_markdown(line) for line in lines) + "</p>"


def chat_heading(line: str) -> str:
    stripped = line.strip()
    while stripped.startswith("#"):
        stripped = stripped[1:].strip()
    if len(stripped) > 2 and stripped[0].isdigit() and stripped[1] in {".", "、"}:
        stripped = stripped[2:].strip()
    if stripped.startswith("**") and stripped.endswith("**"):
        stripped = stripped[2:-2].strip()
    strong_heading_terms = ("核心结果", "科学问题", "优先", "建议", "下一步", "证据", "方向", "验证", "机制")
    if line.startswith("#") or (stripped.endswith("：") and len(stripped) < 28) or any(stripped.startswith(term) for term in strong_heading_terms):
        return f"<h3>{inline_markdown(stripped)}</h3>"
    return ""


def inline_markdown(text: str) -> str:
    text = text.replace("**", "<strong>", 1) if text.count("**") >= 2 else text
    while "**" in text:
        text = text.replace("**", "</strong>", 1)
        if "**" in text:
            text = text.replace("**", "<strong>", 1)
    text = text.replace("*   ", "")
    return text


def suggested_question_forms(questions: list[object]) -> str:
    if not questions:
        return ""
    buttons = []
    for question in questions[:3]:
        text = clean_suggested_question(str(question))
        if not text:
            continue
        buttons.append(
            '<form method="post" action="/question-synthesis/chat">'
            f'<input type="hidden" name="message" value="{html.escape(text, quote=True)}">'
            f'<button class="suggestion-button" type="submit" data-busy="正在围绕该问题继续细化。">{html.escape(text)}</button>'
            '</form>'
        )
    if not buttons:
        return ""
    return '<div class="suggestion-strip">' + "".join(buttons) + "</div>"


def clean_suggested_question(text: str) -> str:
    text = text.strip()
    if len(text) > 2 and text[0].isdigit() and text[1] in {".", "、", ")"}:
        text = text[2:].strip()
    if text.startswith("问题") and ":" in text[:6]:
        text = text.split(":", 1)[1].strip()
    if text.startswith("问题") and "：" in text[:6]:
        text = text.split("：", 1)[1].strip()
    return text


def default_frontend_suggestions() -> list[str]:
    return [
        "这个问题最适合聚焦到哪一个材料体系？",
        "哪些局域环境或微结构变量最可能改变目标性能？",
        "用什么计算或实验指标验证这个机制？",
    ]


def question_synthesis_css() -> str:
    return """
    .question-shell { display:grid; gap:10px; margin:0 auto; max-width:1560px; padding:12px 18px 42px; }
    .question-hero { background:linear-gradient(135deg,#102a43,#0f3a54 58%,#1e40af); min-height:78px; padding:12px 14px; }
    .question-hero h2 { font-size:18px; margin-bottom:3px; }
    .question-hero .subtle { max-width:920px; }
    .question-shell > .metrics { grid-template-columns:repeat(6,minmax(0,1fr)); }
    .question-shell > .metrics .metric { min-height:54px; padding:7px 9px; }
    .question-shell > .metrics .metric-value { font-size:18px; margin-bottom:3px; }
    .question-shell > .metrics .metric-rule { display:none; }
    .chat-workspace { align-items:start; display:grid; gap:10px; grid-template-columns:360px minmax(0,1fr); }
    .question-context { min-width:0; }
    .question-context .panel { background:var(--nav-2); border-color:var(--nav-line); color:#e5eefb; box-shadow:0 14px 32px rgba(15,23,42,.18); }
    .question-context h2 { color:#f8fafc; }
    .question-context h3 { color:#93c5fd; margin:10px 0 7px; }
    .question-context .subtle, .question-context li { color:#cbd5e1; }
    .question-context code { background:#0b1220; border-color:#334155; color:#dbeafe; white-space:normal; }
    .context-card { background:rgba(15,23,42,.38); border:1px solid rgba(148,163,184,.24); border-radius:8px; display:grid; gap:7px; margin-bottom:8px; padding:9px; }
    .context-card h3 { color:#f8fafc; font-size:13px; margin:0; text-transform:none; }
    .context-card p { color:#e2e8f0; line-height:1.45; margin:0; }
    .chat-panel { align-self:start; display:grid; grid-template-rows:auto minmax(520px,1fr) auto; height:min(860px, max(640px, calc(100vh - 180px))); min-height:640px; position:sticky; top:86px; }
    .chat-log { align-content:start; display:grid; gap:12px; min-height:520px; overflow:auto; padding:6px; }
    .bubble { border:1px solid var(--line); border-radius:8px; display:grid; gap:7px; max-width:min(760px,92%); padding:11px 12px; }
    .bubble-left { justify-self:start; }
    .bubble-right, .bubble-user { justify-self:end; }
    .bubble-retrieval { background:#f8fafc; border-left:4px solid var(--cyan); }
    .bubble-llm { background:#eff6ff; border-right:4px solid var(--primary); }
    .bubble-user { background:#f0fdf4; border-right:4px solid var(--ok); }
    .bubble-meta { color:var(--muted); display:flex; flex-wrap:wrap; font-size:11px; font-weight:800; gap:8px; text-transform:uppercase; }
    .bubble-content { color:#1f2937; display:grid; gap:8px; line-height:1.55; overflow-wrap:anywhere; }
    .bubble-content p { margin:0; }
    .bubble-content h3 { color:#0f172a; font-size:14px; line-height:1.35; margin:6px 0 0; text-transform:none; }
    .bubble-content strong { font-weight:800; }
    .chat-list { display:grid; gap:5px; margin:0; padding-left:18px; }
    .suggestion-strip { align-items:center; display:grid; gap:7px; justify-self:center; margin:-4px 0 4px; width:min(620px,86%); }
    .suggestion-strip form { display:block; width:100%; }
    .suggestion-button { background:#f8fafc; border:1px solid #dbeafe; border-radius:8px; color:#64748b; font-size:12px; font-weight:650; line-height:1.35; min-height:30px; padding:7px 9px; text-align:left; width:100%; }
    .suggestion-button:hover { background:#eff6ff; border-color:#bfdbfe; color:#334155; transform:none; }
    .chat-input { border-top:1px solid var(--line); display:grid; gap:8px; grid-template-columns:minmax(0,1fr) auto; padding-top:10px; }
    .chat-input textarea { min-height:50px; resize:vertical; }
    .chat-input button { align-self:end; min-height:50px; min-width:88px; }
    .question-context .mini-list li { border-color:rgba(148,163,184,.24); }
    @media (max-width:1100px) { .chat-workspace { grid-template-columns:1fr; } .chat-log { max-height:56vh; } }
    @media (max-width:640px) { .question-shell { padding:10px 10px 30px; } .chat-input { grid-template-columns:1fr; } .chat-input button { min-height:40px; } .bubble { max-width:100%; } .chat-log { max-height:none; } }
    """


def _empty_question_synthesis_state() -> dict[str, Any]:
    return {
        "session": None,
        "messages": [],
        "context": {"metrics": {}, "retrieval_questions": [], "evidence_gaps": [], "paper_cards": [], "wiki_pages": []},
        "model_name": "LLM 未配置",
        "llm_configured": False,
    }


def knowledge_graph_block(state: dict[str, Any]) -> str:
    graph = state.get("graph") or {"nodes": [], "edges": []}
    nodes = graph.get("nodes") or []
    if not nodes:
        return '<div class="graph-empty">暂无图谱。先完成 PDF 转换，再点击“构建知识库”。</div>'
    filters = ["wiki_topic", "paper", "claim", "material", "method", "property", "evidence"]
    controls = "".join(
        f'<label class="filter-check"><input type="checkbox" data-node-filter value="{item}" checked> {node_type_label(item)}</label>'
        for item in filters
    )
    return (
        '<div class="graph-workspace">'
        f'<div class="graph-controls">{controls}</div>'
        '<svg id="knowledge-graph" viewBox="0 0 900 520" role="img" aria-label="文献知识图谱"></svg>'
        '<aside id="graph-detail" class="graph-detail"><h3>选择节点</h3><p class="subtle">点击图中的主题、论文或证据节点，查看 Wiki 摘要、关联论文和 evidence ids。</p></aside>'
        '</div>'
    )


def knowledge_graph_script(state: dict[str, Any]) -> str:
    graph_json = json.dumps(state.get("graph") or {"nodes": [], "edges": []}, ensure_ascii=False).replace("</", "<\\/")
    cards_json = json.dumps(state.get("cards") or [], ensure_ascii=False).replace("</", "<\\/")
    query_json = json.dumps(state.get("query") or "", ensure_ascii=False)
    return f"""<script>
(() => {{
  const graph = {graph_json};
  const cards = {cards_json};
  const initialQuery = {query_json}.toLowerCase();
  const svg = document.getElementById('knowledge-graph');
  const detail = document.getElementById('graph-detail');
  if (!svg || !graph.nodes || graph.nodes.length === 0) return;
  const colors = {{
    wiki_topic: '#1d4ed8',
    paper: '#0891b2',
    claim: '#d97706',
    material: '#16a34a',
    method: '#7c3aed',
    property: '#dc2626',
    evidence: '#64748b'
  }};
  const labels = {{
    wiki_topic: 'Wiki',
    paper: '论文',
    claim: '结论',
    material: '材料',
    method: '方法',
    property: '性能',
    evidence: '证据'
  }};
  const ring = {{
    wiki_topic: 95,
    material: 145,
    method: 165,
    property: 185,
    paper: 245,
    claim: 295,
    evidence: 345
  }};
  const cx = 450;
  const cy = 260;
  const nodes = graph.nodes.map((node, index) => {{
    const radius = ring[node.type] || 250;
    const angle = (index / Math.max(graph.nodes.length, 1)) * Math.PI * 2 + (node.type.length * 0.21);
    return {{...node, x: cx + Math.cos(angle) * radius, y: cy + Math.sin(angle) * radius}};
  }});
  const nodeById = new Map(nodes.map((node) => [node.id, node]));

  function activeTypes() {{
    const checked = Array.from(document.querySelectorAll('[data-node-filter]:checked')).map((item) => item.value);
    return new Set(checked);
  }}

  function clear() {{
    while (svg.firstChild) svg.removeChild(svg.firstChild);
  }}

  function el(name, attrs) {{
    const item = document.createElementNS('http://www.w3.org/2000/svg', name);
    Object.entries(attrs || {{}}).forEach(([key, value]) => item.setAttribute(key, value));
    return item;
  }}

  function matches(node) {{
    if (!initialQuery) return false;
    const wiki = node.wiki ? JSON.stringify(node.wiki).toLowerCase() : '';
    return String(node.label || '').toLowerCase().includes(initialQuery) || String(node.summary || '').toLowerCase().includes(initialQuery) || wiki.includes(initialQuery);
  }}

  function render() {{
    clear();
    const types = activeTypes();
    const visible = new Set(nodes.filter((node) => types.has(node.type)).map((node) => node.id));
    graph.edges.forEach((edge) => {{
      if (!visible.has(edge.source) || !visible.has(edge.target)) return;
      const source = nodeById.get(edge.source);
      const target = nodeById.get(edge.target);
      if (!source || !target) return;
      const line = el('line', {{
        x1: source.x, y1: source.y, x2: target.x, y2: target.y,
        class: 'graph-edge',
        'data-edge-type': edge.type
      }});
      svg.appendChild(line);
    }});
    nodes.forEach((node) => {{
      if (!visible.has(node.id)) return;
      const group = el('g', {{class: 'graph-node' + (matches(node) ? ' graph-highlight' : ''), tabindex: '0'}});
      const size = Math.max(7, Math.min(18, 7 + Number(node.weight || 1)));
      group.appendChild(el('circle', {{cx: node.x, cy: node.y, r: size, fill: colors[node.type] || '#334155'}}));
      const text = el('text', {{x: node.x + size + 4, y: node.y + 4}});
      text.textContent = truncate(node.label || node.id, node.type === 'paper' ? 30 : 22);
      group.appendChild(text);
      group.addEventListener('click', () => showNode(node));
      group.addEventListener('keydown', (event) => {{
        if (event.key === 'Enter' || event.key === ' ') showNode(node);
      }});
      svg.appendChild(group);
    }});
  }}

  function showNode(node) {{
    const wiki = node.wiki || null;
    const evidence = new Set();
    (graph.edges || []).forEach((edge) => {{
      if (edge.source === node.id || edge.target === node.id) (edge.evidence_ids || []).forEach((id) => evidence.add(id));
    }});
    const relatedCards = cards.filter((card) => {{
      const text = [card.title, ...(card.materials || []), ...(card.methods || []), ...(card.properties || [])].join(' ').toLowerCase();
      return text.includes(String(node.label || '').toLowerCase()) || (card.evidence_ids || []).some((id) => evidence.has(id));
    }}).slice(0, 4);
    detail.innerHTML = `
      <h3>${{escapeHtml(node.label || node.id)}}</h3>
      <div class="status-line"><span class="badge badge-blue">${{labels[node.type] || node.type}}</span><span class="subtle">${{escapeHtml(node.id)}}</span></div>
      <p>${{escapeHtml((wiki && wiki.summary) || node.summary || '这个节点还没有摘要。')}}</p>
      ${{wiki ? renderWiki(wiki) : ''}}
      ${{relatedCards.length ? `<h3>关联 Paper Cards</h3><ul class="mini-list">${{relatedCards.map((card) => `<li>${{escapeHtml(card.title || 'Untitled')}}</li>`).join('')}}</ul>` : ''}}
      ${{evidence.size ? `<h3>Evidence IDs</h3><p class="evidence-wrap">${{Array.from(evidence).slice(0, 12).map(escapeHtml).join(', ')}}</p>` : ''}}
    `;
  }}

  function renderWiki(wiki) {{
    const findings = (wiki.known_findings || []).slice(0, 3).map((item) => `<li>${{escapeHtml(item.claim || '')}}<div class="subtle">${{escapeHtml((item.evidence_ids || []).join(', '))}}</div></li>`).join('');
    const questions = (wiki.open_questions || []).slice(0, 2).map((item) => `<li>${{escapeHtml(item)}}</li>`).join('');
    return `<h3>Wiki 摘要</h3><ul class="mini-list">${{findings}}</ul>${{questions ? `<h3>下一步问题</h3><ul class="mini-list">${{questions}}</ul>` : ''}}`;
  }}

  function truncate(value, max) {{
    value = String(value || '');
    return value.length > max ? value.slice(0, max - 1) + '…' : value;
  }}

  function escapeHtml(value) {{
    return String(value || '').replace(/[&<>"']/g, (char) => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[char]));
  }}

  document.querySelectorAll('[data-node-filter]').forEach((item) => item.addEventListener('change', render));
  render();
}})();
</script>"""


def node_type_label(value: str) -> str:
    return {
        "wiki_topic": "主题",
        "paper": "论文",
        "claim": "结论",
        "material": "材料",
        "method": "方法",
        "property": "性能",
        "evidence": "证据",
    }.get(value, value)


def conversion_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<div class="subtle">暂无转换记录。点击“转换全部 PDF”后会显示 Markdown/JSON 输出。</div>'
    body = "".join(
        "<tr>"
        f"<td class=\"title-cell\">{html.escape(str(row.get('title') or Path(str(row['pdf_path'])).name))}<div class=\"subtle\">{html.escape(str(row.get('venue') or ''))}</div></td>"
        f"<td>{conversion_status_badge(row.get('status'))}</td>"
        f"<td class=\"num\">{html.escape(str(row.get('page_count') or 0))}</td>"
        f"<td class=\"num\">{html.escape(str(row.get('section_count') or 0))}</td>"
        f"<td class=\"num\">{html.escape(str(row.get('figure_count') or 0))}</td>"
        f"<td class=\"num\">{html.escape(str(row.get('table_count') or 0))}</td>"
        f"<td class=\"num\">{html.escape(str(row.get('reference_count') or 0))}</td>"
        f"<td>{parsed_file_link(row.get('markdown_path'), 'Markdown')}</td>"
        f"<td>{parsed_file_link(row.get('json_path'), 'JSON')}</td>"
        f"<td><code>{html.escape(str(row.get('pdf_path') or ''))}</code>{conversion_error(row)}</td>"
        "</tr>"
        for row in rows
    )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        '<th>标题</th><th>状态</th><th>页</th><th>章节</th><th>图注</th><th>表格</th><th>参考</th>'
        '<th>MD</th><th>JSON</th><th>PDF / 错误</th>'
        f'</tr></thead><tbody>{body}</tbody></table></div>'
    )


def conversion_status_badge(status: object) -> str:
    text = str(status or "")
    klass = {"converted": "badge-teal", "failed": "badge-red", "skipped": "badge-neutral"}.get(text, "badge-neutral")
    label = {"converted": "已转换", "failed": "失败", "skipped": "跳过"}.get(text, text or "未知")
    return f'<span class="badge {klass}">{html.escape(label)}</span>'


def parsed_file_link(path_value: object, label: str) -> str:
    if not path_value:
        return '<span class="subtle">无</span>'
    path = Path(str(path_value))
    if not path.exists():
        return '<span class="subtle">缺失</span>'
    try:
        rel = path.resolve().relative_to(analysis_parsed_dir().resolve())
    except ValueError:
        return f'<code>{html.escape(str(path))}</code>'
    return f'<a class="button secondary" href="/parsed/{urllib.parse.quote(str(rel))}">{html.escape(label)}</a>'


def artifact_link(path_value: object, label: str) -> str:
    if not path_value:
        return '<span class="subtle">无</span>'
    path = Path(str(path_value))
    if not path.exists():
        return '<span class="subtle">缺失</span>'
    try:
        rel = path.resolve().relative_to(analysis_data_dir().resolve())
    except ValueError:
        return f'<code>{html.escape(str(path))}</code>'
    return f'<a class="button secondary compact" href="/analysis-artifacts/{urllib.parse.quote(str(rel))}">{html.escape(label)}</a>'


def _truncate(value: str, limit: int) -> str:
    value = value.strip()
    return value if len(value) <= limit else value[: max(0, limit - 1)].rstrip() + "..."


def conversion_error(row: dict[str, Any]) -> str:
    message = row.get("error_message")
    if not message:
        return ""
    return f'<div class="subtle">{html.escape(str(message))}</div>'


def synthesis_block(row: dict[str, Any] | None) -> str:
    if not row:
        return '<div class="subtle">暂无综合结论。完成“分析本轮”后会显示。</div>'
    queries = json.loads(row["next_queries_json"])
    gaps = json.loads(row["evidence_gaps_json"])
    query_items = "".join(f"<li>{html.escape(str(query))}</li>" for query in queries)
    gap_items = "".join(f"<li>{html.escape(str(gap))}</li>" for gap in gaps)
    return (
        f"<p>{html.escape(str(row['summary']))}</p>"
        "<h3>证据缺口</h3>"
        f'<ul class="list">{gap_items}</ul>'
        "<h3>建议 query</h3>"
        f'<ul class="list">{query_items}</ul>'
    )


def round_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<div class="subtle">暂无轮次。</div>'
    body = "".join(
        "<tr>"
        f'<td class="num">{row["id"]}</td>'
        f'<td class="num">{row["round_index"]}</td>'
        f'<td>{status_badge(row["status"])}</td>'
        f'<td class="num">{row["target_count"]}</td>'
        "</tr>"
        for row in rows
    )
    return f'<div class="table-wrap"><table><thead><tr><th>ID</th><th>轮次</th><th>状态</th><th>目标</th></tr></thead><tbody>{body}</tbody></table></div>'


def goal_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<div class="subtle">暂无科学问题。</div>'
    body = "".join(
        "<tr>"
        f'<td class="num">{row["id"]}</td>'
        f'<td>{html.escape(str(row["title"]))}<div class="subtle">每轮 {row["default_target_count"]} 篇</div></td>'
        f'<td>{status_badge(row["status"])}</td>'
        f'<td>{delete_goal_form(row["id"])}</td>'
        "</tr>"
        for row in rows
    )
    return f'<div class="table-wrap"><table><thead><tr><th>ID</th><th>问题</th><th>状态</th><th>操作</th></tr></thead><tbody>{body}</tbody></table></div>'


def delete_goal_form(goal_id: object) -> str:
    return (
        '<form method="post" action="/goal/delete" onsubmit="return confirm(\'删除这个科学问题及其轮次记录？全局论文库和已下载 PDF 不会删除。\')">'
        f'<input type="hidden" name="goal_id" value="{html.escape(str(goal_id))}">'
        '<button class="danger" type="submit">删除</button>'
        "</form>"
    )


def doi_link(value: object) -> str:
    if not value:
        return ""
    doi = str(value)
    return f'<a href="https://doi.org/{html.escape(doi)}">{html.escape(doi)}</a>'


def _active_round(rounds: list[dict[str, Any]]) -> dict[str, Any] | None:
    preferred_statuses = {
        "awaiting_user_approval",
        "approved",
        "awaiting_manual_pdfs",
        "analyzing",
        "synthesized",
        "next_round_proposed",
    }
    for row in rounds:
        if row["status"] in preferred_statuses:
            return row
    return rounds[0] if rounds else None


def _metrics(
    db: LiteratureDB,
    round_id: int | None,
    candidates: list[dict[str, Any]],
    manual_tasks: list[dict[str, Any]],
) -> dict[str, int]:
    rows = db.rows(
        """
        SELECT
            (SELECT COUNT(*) FROM scientific_goals) AS goals,
            (SELECT COUNT(*) FROM exploration_rounds) AS rounds,
            (SELECT COUNT(*) FROM pdf_assets WHERE status IN ('downloaded_oa_pdf', 'preprint_pdf') AND file_path IS NOT NULL) AS downloaded_total,
            (SELECT COUNT(*) FROM paper_conversions WHERE status = 'converted') AS converted_total,
            (SELECT COUNT(*) FROM paper_conversions WHERE status = 'failed') AS conversion_failed_total
        """
    )[0]
    analyses = 0
    downloaded = int(rows["downloaded_total"] or 0)
    if round_id:
        analyses = int(db.rows("SELECT COUNT(*) AS n FROM paper_analyses WHERE round_id = ?", (round_id,))[0]["n"] or 0)
        downloaded = sum(1 for row in candidates if _local_pdf_exists(row.get("local_pdf_path")))
    return {
        "goals": int(rows["goals"] or 0),
        "rounds": int(rows["rounds"] or 0),
        "candidates": len(candidates),
        "downloaded": downloaded,
        "manual_pending": sum(1 for row in manual_tasks if row.get("status") != "completed"),
        "analyses": analyses,
        "analysis_pdf_total": int(rows["downloaded_total"] or 0),
        "analysis_converted": _safe_count(db, "paper_cards"),
        "analysis_failed": _safe_count(db, "wiki_pages"),
    }


def _local_pdf_exists(value: object) -> bool:
    if not value:
        return False
    path = Path(str(value))
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _selection_mode_label() -> str:
    mode = os.environ.get("LIT_AGENT_SELECTION_MODE", "rules").strip().lower()
    configured = LLMSettings.from_env() is not None
    if mode in {"llm", "hybrid", "external_llm"} and configured:
        return "外部 LLM 重排 + 规则兜底"
    if mode in {"llm", "hybrid", "external_llm"}:
        return "规则推荐（LLM 未配置）"
    return "规则推荐"


def _status_label(status: str) -> str:
    return {
        "active": "进行中",
        "planned": "已规划",
        "needs_retry": "需要重试",
        "awaiting_user_approval": "待人工确认",
        "approved": "已确认",
        "acquiring_pdfs": "获取 PDF 中",
        "awaiting_manual_pdfs": "待手动 PDF",
        "analyzing": "分析中",
        "synthesized": "已综合",
        "next_round_proposed": "已提出下一轮",
        "high": "高证据",
        "medium": "中证据",
        "low": "低证据",
        "pending": "待处理",
        "completed": "已完成",
    }.get(status, status)


def _friendly_error(message: str) -> str:
    if "is not awaiting approval" in message:
        return "这个轮次还不能确认。请确认它处于“待人工确认”状态；如果它显示“已规划”或“需要重试”，说明本轮候选没有成功生成，需要重新规划。"
    if "must be approved before acquisition" in message:
        return "这个轮次还不能获取 PDF。请先点击“确认本轮”；只有状态为“已确认”的轮次才能进入 PDF 获取。"
    if "Scientific goal not found" in message:
        return "没有找到这个科学问题，可能已经被删除。"
    if "Exploration round not found" in message:
        return "没有找到这个轮次，可能已经被删除或输入了错误 ID。"
    return message


def _required(fields: dict[str, str], name: str) -> str:
    value = fields.get(name, "").strip()
    if not value:
        raise ValueError(f"Missing field: {name}")
    return value


def _int_required(fields: dict[str, str], name: str) -> int:
    return int(_required(fields, name))


def _int(value: str | None, default: int | None) -> int | None:
    if value is None or value == "":
        return default
    return int(value)


def _message_location(message: str) -> str:
    return "/literature?message=" + urllib.parse.quote(message)


def _paper_analysis_message_location(message: str, *, error: bool = False) -> str:
    key = "error" if error else "message"
    return "/paper-analysis?" + key + "=" + urllib.parse.quote(message)


def _question_synthesis_message_location(message: str, *, error: bool = False) -> str:
    key = "error" if error else "message"
    return "/question-synthesis?" + key + "=" + urllib.parse.quote(message)


def _route_candidates_message_location(
    message: str,
    *,
    error: bool = False,
    question_id: object | None = None,
) -> str:
    key = "error" if error else "message"
    params = {key: message}
    if question_id not in (None, ""):
        params["question_id"] = str(question_id)
    return "/route-candidates?" + urllib.parse.urlencode(params)

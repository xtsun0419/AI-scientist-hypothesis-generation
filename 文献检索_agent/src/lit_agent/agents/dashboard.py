from __future__ import annotations

import html
import json
from pathlib import Path

from lit_agent.db import LiteratureDB
from lit_agent.models import utc_now_iso


class HtmlDashboardAgent:
    """Builds a self-contained Chinese HTML dashboard for architecture and run state."""

    def __init__(self, db: LiteratureDB, report_dir: Path):
        self.db = db
        self.report_dir = report_dir
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def build(self) -> Path:
        path = self.report_dir / "dashboard.html"
        self._write_detail_pages()
        path.write_text(self._html(), encoding="utf-8")
        return path

    def _html(self) -> str:
        stats = self._stats()
        method_metrics = self._method_metrics()
        access_rows = self._access_counts()
        source_rows = self._source_counts()
        failure_rows = self._failure_counts()
        llm_rows = self._llm_review_counts()
        latest_runs = self._latest_runs()
        audit_rows = self._audit_counts()
        sample_rows = self._sample_papers()
        download_failed_rows = self._download_failed_rows(limit=8)
        missing_doi_rows = self._missing_doi_rows(limit=8)
        latest_goal = self._latest_goal()
        latest_round = self._latest_round(latest_goal["id"] if latest_goal else None)
        round_candidates = self._round_candidates(latest_round["id"] if latest_round else None)
        manual_tasks = self._manual_tasks(latest_round["id"] if latest_round else None)
        round_synthesis = self._round_synthesis(latest_round["id"] if latest_round else None)
        report_links = [
            ("全部文献", "all_papers.csv"),
            ("开放 PDF 记录", "open_pdf_records.csv"),
            ("非开放但有 DOI", "closed_access_with_doi.csv"),
            ("缺 DOI 清单", "missing_doi.csv"),
            ("缺 DOI HTML", "missing_doi.html"),
            ("下载失败", "download_failed.csv"),
            ("下载失败 HTML", "download_failed.html"),
            ("质量审计", "audit_findings.csv"),
            ("源失败记录", "source_failures.csv"),
            ("LLM 复核记录", "llm_relevance_reviews.csv"),
            ("方法指标", "pipeline_metrics.csv"),
            ("摘要", "summary.md"),
        ]
        return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>文献检索 Agent 工作台</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #5d6978;
      --line: #d8dee7;
      --accent: #1f7a6d;
      --accent-2: #384f9f;
      --warn: #a06000;
      --bad: #a13232;
      --good: #267043;
      --chip: #eef2f6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    header {{
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      padding: 20px 28px;
      position: sticky;
      top: 0;
      z-index: 3;
    }}
    .header-row {{
      align-items: center;
      display: flex;
      gap: 18px;
      justify-content: space-between;
    }}
    h1 {{
      font-size: 24px;
      line-height: 1.2;
      margin: 0;
    }}
    h2 {{
      font-size: 17px;
      margin: 0 0 14px;
    }}
    h3 {{
      font-size: 14px;
      margin: 0 0 10px;
    }}
    .subtle {{
      color: var(--muted);
      font-size: 13px;
    }}
    main {{
      display: grid;
      gap: 18px;
      padding: 18px 28px 40px;
    }}
    .grid {{
      display: grid;
      gap: 16px;
    }}
    .grid.two {{ grid-template-columns: minmax(0, 1.4fr) minmax(320px, .8fr); }}
    .grid.three {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}
    .stats {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(9, minmax(0, 1fr));
    }}
    .stat {{
      background: var(--chip);
      border: 1px solid #e1e7ef;
      border-radius: 8px;
      min-height: 78px;
      padding: 12px;
    }}
    .stat b {{
      display: block;
      font-size: 24px;
      line-height: 1;
      margin-bottom: 8px;
    }}
    .dag {{
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(5, minmax(145px, 1fr));
      overflow-x: auto;
      padding-bottom: 4px;
    }}
    .node {{
      border: 1px solid var(--line);
      border-radius: 8px;
      min-height: 112px;
      padding: 12px;
      position: relative;
    }}
    .node.primary {{ border-color: #82b8ae; background: #eef8f6; }}
    .node.io {{ border-color: #aab5d8; background: #f0f2fb; }}
    .node.check {{ border-color: #d5bd8b; background: #fbf6eb; }}
    .node b {{
      display: block;
      font-size: 13px;
      margin-bottom: 7px;
    }}
    .node p {{
      color: var(--muted);
      font-size: 12px;
      margin: 0;
    }}
    .relationship {{
      display: grid;
      gap: 12px;
      grid-template-columns: 1fr;
    }}
    .relation-band {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }}
    .relation-band.orchestrator {{ background: #eef8f6; border-color: #82b8ae; }}
    .relation-band.agents {{ background: #ffffff; }}
    .relation-band.storage {{ background: #f0f2fb; border-color: #aab5d8; }}
    .relation-band.outputs {{ background: #fbf6eb; border-color: #d5bd8b; }}
    .relation-title {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 10px;
      text-transform: uppercase;
    }}
    .relation-items {{
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(5, minmax(130px, 1fr));
    }}
    .relation-card {{
      background: rgba(255, 255, 255, .72);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-height: 74px;
      padding: 10px;
    }}
    .relation-card b {{
      display: block;
      font-size: 13px;
      margin-bottom: 5px;
    }}
    .relation-card span {{
      color: var(--muted);
      display: block;
      font-size: 12px;
    }}
    .flow-line {{
      align-items: center;
      color: var(--muted);
      display: flex;
      font-size: 13px;
      gap: 8px;
      justify-content: center;
      min-height: 20px;
    }}
    .flow-line::before,
    .flow-line::after {{
      background: var(--line);
      content: "";
      display: block;
      height: 1px;
      max-width: 180px;
      flex: 1;
    }}
    .arrow {{
      align-self: center;
      color: var(--muted);
      font-size: 20px;
      text-align: center;
    }}
    .bar-row {{
      align-items: center;
      display: grid;
      gap: 10px;
      grid-template-columns: minmax(135px, .7fr) minmax(0, 2fr) 70px;
      margin: 9px 0;
    }}
    .bar-track {{
      background: #edf1f5;
      border-radius: 999px;
      height: 10px;
      overflow: hidden;
    }}
    .bar {{
      background: var(--accent);
      height: 100%;
      min-width: 2px;
    }}
    .bar.alt {{ background: var(--accent-2); }}
    table {{
      border-collapse: collapse;
      font-size: 13px;
      width: 100%;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 6px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }}
    code {{
      background: #eef1f4;
      border: 1px solid #dfe4ea;
      border-radius: 6px;
      display: block;
      font-size: 12px;
      margin: 7px 0;
      overflow-x: auto;
      padding: 9px;
      white-space: nowrap;
    }}
    .inline-code {{
      background: #eef1f4;
      border: 1px solid #dfe4ea;
      border-radius: 5px;
      color: var(--ink);
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 12px;
      padding: 1px 5px;
      white-space: nowrap;
    }}
    .links {{
      display: flex;
      flex-wrap: wrap;
      gap: 9px;
    }}
    a.button {{
      background: var(--chip);
      border: 1px solid var(--line);
      border-radius: 8px;
      color: var(--ink);
      font-size: 13px;
      padding: 8px 10px;
      text-decoration: none;
    }}
    .pill {{
      background: var(--chip);
      border-radius: 999px;
      display: inline-block;
      font-size: 12px;
      padding: 4px 8px;
    }}
    .status-closed_access_has_doi {{ color: var(--warn); }}
    .status-closed_access_missing_doi, .status-download_failed {{ color: var(--bad); }}
    .status-downloaded_oa_pdf, .status-preprint_pdf, .status-oa_pdf_available {{ color: var(--good); }}
    @media (max-width: 980px) {{
      header {{ position: static; }}
      .header-row {{ align-items: flex-start; flex-direction: column; }}
      .grid.two, .grid.three {{ grid-template-columns: 1fr; }}
      .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .dag {{ grid-template-columns: 1fr; }}
      .relation-items {{ grid-template-columns: 1fr; }}
      .arrow {{ display: none; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="header-row">
      <div>
        <h1>文献检索 Agent 工作台</h1>
        <div class="subtle">永磁领域文献发现 · 生成时间 {html.escape(utc_now_iso())}</div>
      </div>
      <div class="subtle">用于架构审阅、检索调整和运行状态检查的静态 HTML 控制面板</div>
    </div>
  </header>
  <main>
    <section class="stats">
      {self._stat("主文献记录", stats["papers"])}
      {self._stat("原始来源记录", stats["source_records"])}
      {self._stat("候选文献", stats["paper_candidates"])}
      {self._stat("获取状态记录", stats["access_records"])}
      {self._stat("源失败记录", stats["source_failures"])}
      {self._stat("LLM 复核", stats["llm_reviews"])}
      {self._stat("DOI 覆盖率", self._percent_value(method_metrics.get("doi_coverage")))}
      {self._stat("PDF URL 覆盖率", self._percent_value(method_metrics.get("pdf_url_coverage")))}
      {self._stat("审计问题", stats["audit_findings"])}
    </section>

    <section class="panel">
      <h2>科学问题工作台</h2>
      {self._goal_round_summary(latest_goal, latest_round, round_synthesis)}
      <div class="grid two" style="margin-top:14px">
        <div>
          <h3>本轮候选文献</h3>
          {self._round_candidate_table(round_candidates)}
        </div>
        <div>
          <h3>手动下载任务</h3>
          {self._manual_task_table(manual_tasks)}
        </div>
      </div>
    </section>

    <section class="panel">
      <h2>多 Agent 架构图</h2>
      <div class="dag">
        {self._node("OrchestratorAgent", "总调度器：创建 run，控制阶段顺序、失败标记和子 Agent 调用。", "primary")}
        {self._node("DomainQueryAgent", "领域查询规划：从配置生成永磁领域 QueryPlan。", "")}
        {self._node("SourceDiscoveryAgent", "多源检索：调用数据源 connector，写入原始记录。", "io")}
        {self._node("MetadataNormalizeAgent", "元数据清洗：把不同来源 payload 转成候选文献。", "")}
        {self._node("DeduplicationAgent", "去重合并：优先按 DOI，其次按标题和年份合并。", "")}
        {self._node("RelevanceJudgeAgent", "相关性判断：用可审计规则给永磁相关性打分。", "check")}
        {self._node("OAResolverAgent", "获取状态解析：补 DOI URL、出版商页、OA 状态和 PDF URL。", "check")}
        {self._node("PdfDownloadAgent", "PDF 下载：只下载开放或明确公开的 PDF，并记录哈希。", "io")}
        {self._node("QualityAuditAgent", "质量审计：检查缺 DOI、低相关、PDF 异常等问题。", "check")}
        {self._node("ReportExportAgent", "报表导出：生成 CSV、JSONL、BibTeX、Markdown 和 HTML 工作台。", "io")}
      </div>
    </section>

    <section class="panel">
      <h2>多 Agent 工作关系框架图</h2>
      <div class="relationship">
        <div class="relation-band orchestrator">
          <div class="relation-title">调度层</div>
          <div class="relation-items">
            {self._relation_card("OrchestratorAgent", "读取配置，创建 run，按 DAG 调用各 Agent，记录失败与完成状态。")}
            {self._relation_card("CLI 命令", "search / resolve-oa / download / audit / report / dashboard")}
            {self._relation_card("领域配置", "永磁词表、数据源、年份范围、每个 query 的结果上限。")}
            {self._relation_card("运行日志", "当前以终端输出和 SQLite search_runs 为主。")}
            {self._relation_card("断点续跑", "每个阶段都可单独运行，复用 SQLite 中间状态。")}
          </div>
        </div>
        <div class="flow-line">Orchestrator 按阶段推进，所有 Agent 通过 SQLite 交换状态</div>
        <div class="relation-band agents">
          <div class="relation-title">Agent 执行层</div>
          <div class="relation-items">
            {self._relation_card("Query", "生成永磁领域检索式。")}
            {self._relation_card("Discovery", "连接 OpenAlex、Crossref、Semantic Scholar、arXiv 等来源。")}
            {self._relation_card("Normalize", "统一 DOI、标题、作者、年份、期刊、PDF URL。")}
            {self._relation_card("Dedup", "合并重复文献，保留全部来源证据。")}
            {self._relation_card("Relevance", "判断是否真的属于永磁领域。")}
            {self._relation_card("OA Resolver", "区分开放、非开放、有 DOI、缺 DOI。")}
            {self._relation_card("PDF Download", "只下载开放或明确公开 PDF。")}
            {self._relation_card("Quality Audit", "检查缺 DOI、低相关、PDF 异常。")}
            {self._relation_card("Report / HTML", "生成 CSV、JSONL、BibTeX、Markdown 和本工作台。")}
            {self._relation_card("LLM 边界复核", "只复核规则分数中间区间或证据不足的样本。")}
          </div>
        </div>
        <div class="flow-line">结构化数据进 SQLite，大文件进 PDF 文件库，最终产出报表</div>
        <div class="relation-band storage">
          <div class="relation-title">共享状态与文件层</div>
          <div class="relation-items">
            {self._relation_card("source_records", "保存每个来源返回的原始 payload。")}
            {self._relation_card("paper_candidates", "保存标准化后的候选文献。")}
            {self._relation_card("papers", "去重后的主文献库。")}
            {self._relation_card("access_records", "DOI、DOI URL、出版商页、OA 状态、PDF URL。")}
            {self._relation_card("pdf_assets", "PDF 本地路径、哈希、大小、下载状态。")}
          </div>
        </div>
        <div class="flow-line">给科研使用者交付可追溯的文献清单、PDF 语料和复核列表</div>
        <div class="relation-band outputs">
          <div class="relation-title">交付层</div>
          <div class="relation-items">
            {self._relation_card("开放 PDF 文件库", "data/pdfs，只保存合法可下载 PDF。")}
            {self._relation_card("非开放 DOI 清单", "closed_access_with_doi.csv，后续走机构或手动获取。")}
            {self._relation_card("缺 DOI 清单", "missing_doi.csv，用于人工补全。")}
            {self._relation_card("总文献库导出", "CSV / JSONL / BibTeX。")}
            {self._relation_card("HTML 工作台", "用于可视化审阅和下一轮系统调整。")}
          </div>
        </div>
      </div>
    </section>

    <section class="grid two">
      <div class="panel">
        <h2>文献获取状态</h2>
        {self._bars(access_rows, "access_status")}
      </div>
      <div class="panel">
        <h2>数据源覆盖</h2>
        {self._bars(source_rows, "source", alt=True)}
      </div>
    </section>

    <section class="grid three">
      <div class="panel">
        <h2>v2 方法指标</h2>
        {self._metrics_table(method_metrics)}
      </div>
      <div class="panel">
        <h2>源失败类型</h2>
        {self._bars(failure_rows, "failure_type")}
      </div>
      <div class="panel">
        <h2>LLM 复核结论</h2>
        {self._bars(llm_rows, "decision", alt=True)}
      </div>
    </section>

    <section class="grid three">
      <div class="panel">
        <h2>常用命令</h2>
        <code>python3 run.py search --mode smoke --domain permanent_magnets --from-year 2020 --to-year 2026</code>
        <code>python3 run.py search --mode pilot --domain permanent_magnets --from-year 1900 --to-year 2026</code>
        <code>python3 run.py resolve-oa</code>
        <code>python3 run.py review-relevance</code>
        <code>python3 run.py download</code>
        <code>python3 run.py --pdf-dir /Volumes/YourDisk/permanent_magnet_pdfs download</code>
        <code>python3 run.py audit</code>
        <code>python3 run.py report</code>
        <code>python3 run.py dashboard</code>
      </div>
      <div class="panel">
        <h2>报表文件</h2>
        <div class="links">
          {''.join(self._link(label, filename) for label, filename in report_links)}
        </div>
      </div>
      <div class="panel">
        <h2>质量审计摘要</h2>
        {self._audit_table(audit_rows)}
      </div>
    </section>

    <section class="grid two">
      <div class="panel">
        <h2>最近检索运行</h2>
        {self._runs_table(latest_runs)}
      </div>
      <div class="panel">
        <h2>样本文献</h2>
        {self._sample_table(sample_rows)}
      </div>
    </section>

    <section class="grid two">
      <div class="panel">
        <h2>下载失败 DOI 预览</h2>
        {self._report_preview_table(download_failed_rows, ["id", "title", "year", "doi", "access_status", "download_error"])}
      </div>
      <div class="panel">
        <h2>缺 DOI 预览</h2>
        {self._report_preview_table(missing_doi_rows, ["id", "title", "year", "venue", "access_status"])}
      </div>
    </section>
  </main>
</body>
</html>
"""

    def _write_detail_pages(self) -> None:
        self._write_detail_page(
            "下载失败 DOI 清单",
            "download_failed.html",
            self._download_failed_rows(limit=None),
            ["id", "title", "year", "venue", "doi", "doi_url", "publisher_url", "pdf_url", "access_status", "download_error"],
        )
        self._write_detail_page(
            "缺 DOI 清单",
            "missing_doi.html",
            self._missing_doi_rows(limit=None),
            ["id", "title", "year", "venue", "publisher_url", "source_url", "document_type", "relevance_score", "access_status"],
        )

    def _write_detail_page(
        self,
        title: str,
        filename: str,
        rows: list[dict[str, object]],
        columns: list[str],
    ) -> None:
        path = self.report_dir / filename
        table = self._report_preview_table(rows, columns, empty_text="暂无记录。")
        path.write_text(
            f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{
      background: #f6f7f9;
      color: #17202a;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      padding: 24px;
    }}
    main {{
      background: #ffffff;
      border: 1px solid #d8dee7;
      border-radius: 8px;
      padding: 18px;
    }}
    h1 {{ font-size: 22px; margin: 0 0 8px; }}
    a {{ color: #1f5d9f; }}
    table {{ border-collapse: collapse; font-size: 13px; width: 100%; }}
    th, td {{ border-bottom: 1px solid #d8dee7; padding: 8px 6px; text-align: left; vertical-align: top; }}
    th {{ color: #5d6978; font-size: 12px; }}
    .subtle {{ color: #5d6978; font-size: 13px; margin-bottom: 16px; }}
  </style>
</head>
<body>
  <main>
    <h1>{html.escape(title)}</h1>
    <div class="subtle">共 {len(rows)} 条 · <a href="dashboard.html">返回工作台</a></div>
    {table}
  </main>
</body>
</html>
""",
            encoding="utf-8",
        )

    def _stats(self) -> dict[str, int]:
        tables = [
            "papers",
            "source_records",
            "paper_candidates",
            "access_records",
            "pdf_assets",
            "audit_findings",
            "source_failures",
            "llm_relevance_reviews",
        ]
        stats = {table: self._count(f"SELECT COUNT(*) AS n FROM {table}") for table in tables}
        stats["llm_reviews"] = stats.pop("llm_relevance_reviews")
        return stats

    def _method_metrics(self) -> dict[str, float]:
        rows = self.db.rows(
            """
            SELECT metric_name, metric_value
            FROM pipeline_metrics
            WHERE id IN (
                SELECT MAX(id)
                FROM pipeline_metrics
                GROUP BY metric_name
            )
            """
        )
        return {row["metric_name"]: float(row["metric_value"]) for row in rows}

    def _access_counts(self) -> list[dict[str, object]]:
        return [dict(row) for row in self.db.rows(
            """
            SELECT access_status, COUNT(*) AS count
            FROM access_records
            GROUP BY access_status
            ORDER BY count DESC, access_status
            """
        )]

    def _source_counts(self) -> list[dict[str, object]]:
        return [dict(row) for row in self.db.rows(
            """
            SELECT source, COUNT(*) AS count
            FROM source_records
            GROUP BY source
            ORDER BY count DESC, source
            """
        )]

    def _failure_counts(self) -> list[dict[str, object]]:
        return [dict(row) for row in self.db.rows(
            """
            SELECT failure_type, COUNT(*) AS count
            FROM source_failures
            GROUP BY failure_type
            ORDER BY count DESC, failure_type
            """
        )]

    def _llm_review_counts(self) -> list[dict[str, object]]:
        return [dict(row) for row in self.db.rows(
            """
            SELECT decision, COUNT(*) AS count
            FROM llm_relevance_reviews
            GROUP BY decision
            ORDER BY count DESC, decision
            """
        )]

    def _latest_runs(self) -> list[dict[str, object]]:
        return [dict(row) for row in self.db.rows(
            """
            SELECT id, domain, from_year, to_year, status, started_at, finished_at
            FROM search_runs
            ORDER BY id DESC
            LIMIT 8
            """
        )]

    def _audit_counts(self) -> list[dict[str, object]]:
        return [dict(row) for row in self.db.rows(
            """
            SELECT severity, issue_type, COUNT(*) AS count
            FROM audit_findings
            GROUP BY severity, issue_type
            ORDER BY severity, count DESC
            """
        )]

    def _sample_papers(self) -> list[dict[str, object]]:
        return [dict(row) for row in self.db.rows(
            """
            SELECT p.id, p.title, p.year, p.doi, ar.access_status
            FROM papers p
            LEFT JOIN access_records ar ON ar.paper_id = p.id
            ORDER BY p.id DESC
            LIMIT 8
            """
        )]

    def _latest_goal(self) -> dict[str, object] | None:
        rows = self.db.rows("SELECT * FROM scientific_goals ORDER BY id DESC LIMIT 1")
        return dict(rows[0]) if rows else None

    def _latest_round(self, goal_id: object | None) -> dict[str, object] | None:
        if goal_id is None:
            rows = self.db.rows("SELECT * FROM exploration_rounds ORDER BY id DESC LIMIT 1")
        else:
            rows = self.db.rows(
                "SELECT * FROM exploration_rounds WHERE goal_id = ? ORDER BY id DESC LIMIT 1",
                (goal_id,),
            )
        return dict(rows[0]) if rows else None

    def _round_candidates(self, round_id: object | None) -> list[dict[str, object]]:
        if round_id is None:
            return []
        return [dict(row) for row in self.db.round_candidates(int(round_id))[:8]]

    def _manual_tasks(self, round_id: object | None) -> list[dict[str, object]]:
        if round_id is None:
            return []
        return [dict(row) for row in self.db.manual_download_tasks(int(round_id))[:8]]

    def _round_synthesis(self, round_id: object | None) -> dict[str, object] | None:
        if round_id is None:
            return None
        row = self.db.round_synthesis(int(round_id))
        return dict(row) if row else None

    def _download_failed_rows(self, limit: int | None) -> list[dict[str, object]]:
        sql = """
            SELECT
                p.id, p.title, p.year, p.venue, ar.doi, ar.doi_url,
                ar.publisher_url, ar.source_url, ar.pdf_url, ar.access_status,
                pa.error_message AS download_error
            FROM access_records ar
            JOIN papers p ON p.id = ar.paper_id
            LEFT JOIN pdf_assets pa ON pa.id = (
                SELECT id
                FROM pdf_assets
                WHERE paper_id = p.id
                ORDER BY id DESC
                LIMIT 1
            )
            WHERE ar.access_status = 'download_failed'
            ORDER BY p.year DESC, p.title
        """
        if limit is not None:
            sql += f"\nLIMIT {int(limit)}"
        return [dict(row) for row in self.db.rows(sql)]

    def _missing_doi_rows(self, limit: int | None) -> list[dict[str, object]]:
        sql = """
            SELECT
                p.id, p.title, p.year, p.venue, p.publisher, p.publisher_url,
                p.source_url, p.document_type, p.relevance_score, p.relevance_reason,
                ar.pdf_url, ar.access_status
            FROM papers p
            LEFT JOIN access_records ar ON ar.paper_id = p.id
            WHERE p.doi IS NULL OR p.doi = ''
            ORDER BY p.year DESC, p.title
        """
        if limit is not None:
            sql += f"\nLIMIT {int(limit)}"
        return [dict(row) for row in self.db.rows(sql)]

    def _count(self, sql: str) -> int:
        return int(self.db.rows(sql)[0]["n"])

    @staticmethod
    def _stat(label: str, value: int) -> str:
        return f'<div class="stat"><b>{value}</b><span class="subtle">{html.escape(label)}</span></div>'

    @staticmethod
    def _percent_value(value: float | None) -> str:
        if value is None:
            return "0%"
        return f"{value * 100:.1f}%"

    @staticmethod
    def _node(title: str, body: str, klass: str) -> str:
        class_attr = f"node {klass}".strip()
        return f'<div class="{class_attr}"><b>{html.escape(title)}</b><p>{html.escape(body)}</p></div>'

    @staticmethod
    def _link(label: str, filename: str) -> str:
        return f'<a class="button" href="{html.escape(filename)}">{html.escape(label)}</a>'

    @staticmethod
    def _relation_card(title: str, body: str) -> str:
        return (
            '<div class="relation-card">'
            f'<b>{html.escape(title)}</b>'
            f'<span>{html.escape(body)}</span>'
            '</div>'
        )

    @staticmethod
    def _bars(rows: list[dict[str, object]], label_key: str, alt: bool = False) -> str:
        if not rows:
            return '<div class="subtle">暂无数据。</div>'
        max_count = max(int(row["count"]) for row in rows) or 1
        items = []
        for row in rows:
            label = _display_label(str(row[label_key]))
            count = int(row["count"])
            width = max(2, round(count / max_count * 100))
            bar_class = "bar alt" if alt else "bar"
            items.append(
                '<div class="bar-row">'
                f'<span class="subtle">{html.escape(label)}</span>'
                '<div class="bar-track">'
                f'<div class="{bar_class}" style="width:{width}%"></div>'
                '</div>'
                f'<b>{count}</b>'
                '</div>'
            )
        return "".join(items)

    @staticmethod
    def _metrics_table(metrics: dict[str, float]) -> str:
        if not metrics:
            return '<div class="subtle">暂无方法指标。</div>'
        labels = {
            "dedup_compression_ratio": "去重压缩率",
            "doi_coverage": "DOI 覆盖率",
            "oa_coverage": "OA 覆盖率",
            "pdf_url_coverage": "PDF URL 覆盖率",
            "downloaded_pdf_coverage": "已下载 PDF 覆盖率",
            "llm_reviews": "LLM 复核数",
            "source_failures": "源失败数",
        }
        rows = []
        for key, label in labels.items():
            if key not in metrics:
                continue
            value = metrics[key]
            display = f"{value * 100:.1f}%" if key.endswith("coverage") or key.endswith("ratio") else f"{value:.0f}"
            rows.append(f"<tr><td>{html.escape(label)}</td><td>{html.escape(display)}</td></tr>")
        return "<table><thead><tr><th>指标</th><th>值</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"

    @staticmethod
    def _goal_round_summary(
        goal: dict[str, object] | None,
        round_row: dict[str, object] | None,
        synthesis: dict[str, object] | None,
    ) -> str:
        if not goal:
            return (
                '<div class="subtle">尚未创建科学问题。'
                '可运行 <span class="inline-code">python3 run.py goal create --title "..."</span> 开始 v3 迭代探索。</div>'
            )
        parts = [
            f"<b>{html.escape(str(goal['title']))}</b>",
            f"<span class=\"pill\">目标 {int(goal['default_target_count'])} 篇/轮</span>",
        ]
        if round_row:
            parts.append(
                f"<span class=\"pill\">第 {int(round_row['round_index'])} 轮 · {html.escape(_display_label(str(round_row['status'])))}</span>"
            )
        summary = '<div style="display:flex; gap:8px; flex-wrap:wrap; align-items:center">' + "".join(parts) + "</div>"
        if synthesis:
            summary += f'<div class="subtle" style="margin-top:10px">{html.escape(str(synthesis["summary"]))}</div>'
            next_queries = json.loads(str(synthesis["next_queries_json"]))
            summary += "<code>" + html.escape("；".join(next_queries[:4])) + "</code>"
        return summary

    @staticmethod
    def _round_candidate_table(rows: list[dict[str, object]]) -> str:
        if not rows:
            return '<div class="subtle">暂无本轮候选。</div>'
        body = "".join(
            "<tr>"
            f"<td>{int(row['rank'])}</td>"
            f"<td>{html.escape(str(row['title'] or ''))}</td>"
            f"<td>{html.escape(str(row['year'] or ''))}</td>"
            f"<td>{float(row['selection_score']):.2f}</td>"
            f"<td>{html.escape(str(row['selection_reason'] or ''))}</td>"
            "</tr>"
            for row in rows
        )
        return f"<table><thead><tr><th>序</th><th>标题</th><th>年份</th><th>分数</th><th>选择理由</th></tr></thead><tbody>{body}</tbody></table>"

    @staticmethod
    def _manual_task_table(rows: list[dict[str, object]]) -> str:
        if not rows:
            return '<div class="subtle">暂无手动下载任务。</div>'
        body = "".join(
            "<tr>"
            f"<td>{html.escape(str(row['title'] or ''))}</td>"
            f"<td>{HtmlDashboardAgent._doi_link(row['doi'])}</td>"
            f"<td>{html.escape(_shorten(str(row['target_path'] or ''), 42))}</td>"
            f"<td>{html.escape(_display_label(str(row['status'] or '')))}</td>"
            "</tr>"
            for row in rows
        )
        return f"<table><thead><tr><th>标题</th><th>DOI</th><th>放置路径</th><th>状态</th></tr></thead><tbody>{body}</tbody></table>"

    @staticmethod
    def _audit_table(rows: list[dict[str, object]]) -> str:
        if not rows:
            return '<div class="subtle">暂无审计问题。</div>'
        body = "".join(
            "<tr>"
            f"<td>{html.escape(_display_label(str(row['severity'])))}</td>"
            f"<td>{html.escape(_display_label(str(row['issue_type'])))}</td>"
            f"<td>{int(row['count'])}</td>"
            "</tr>"
            for row in rows
        )
        return f"<table><thead><tr><th>级别</th><th>问题类型</th><th>数量</th></tr></thead><tbody>{body}</tbody></table>"

    @staticmethod
    def _runs_table(rows: list[dict[str, object]]) -> str:
        if not rows:
            return '<div class="subtle">暂无检索运行记录。</div>'
        body = "".join(
            "<tr>"
            f"<td>{row['id']}</td>"
            f"<td>{html.escape(_display_label(str(row['domain'])))}</td>"
            f"<td>{row['from_year']}-{row['to_year']}</td>"
            f"<td><span class=\"pill\">{html.escape(_display_label(str(row['status'])))}</span></td>"
            f"<td>{html.escape(str(row['started_at']))}</td>"
            "</tr>"
            for row in rows
        )
        return f"<table><thead><tr><th>ID</th><th>领域</th><th>年份</th><th>状态</th><th>开始时间</th></tr></thead><tbody>{body}</tbody></table>"

    @staticmethod
    def _sample_table(rows: list[dict[str, object]]) -> str:
        if not rows:
            return '<div class="subtle">暂无文献记录。</div>'
        body = "".join(
            "<tr>"
            f"<td>{row['id']}</td>"
            f"<td>{html.escape(str(row['title'] or ''))}</td>"
            f"<td>{html.escape(str(row['year'] or ''))}</td>"
            f"<td>{HtmlDashboardAgent._doi_link(row['doi'])}</td>"
            f"<td class=\"status-{html.escape(str(row['access_status'] or ''))}\">{html.escape(_display_label(str(row['access_status'] or '')))}</td>"
            "</tr>"
            for row in rows
        )
        return f"<table><thead><tr><th>ID</th><th>标题</th><th>年份</th><th>DOI</th><th>获取状态</th></tr></thead><tbody>{body}</tbody></table>"

    @staticmethod
    def _report_preview_table(
        rows: list[dict[str, object]],
        columns: list[str],
        *,
        empty_text: str = "暂无记录。",
    ) -> str:
        if not rows:
            return f'<div class="subtle">{html.escape(empty_text)}</div>'
        head = "".join(f"<th>{html.escape(_display_label(column))}</th>" for column in columns)
        body_rows = []
        for row in rows:
            cells = []
            for column in columns:
                cells.append(f"<td>{HtmlDashboardAgent._cell_value(column, row.get(column))}</td>")
            body_rows.append("<tr>" + "".join(cells) + "</tr>")
        return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"

    @staticmethod
    def _cell_value(column: str, value: object) -> str:
        if value is None:
            return ""
        text = str(value)
        if column == "doi" and text:
            return HtmlDashboardAgent._doi_link(text)
        if column.endswith("_url") or column in {"pdf_url", "source_url", "publisher_url"}:
            return f'<a href="{html.escape(text)}">{html.escape(_shorten(text, 48))}</a>' if text else ""
        return html.escape(_display_label(text))

    @staticmethod
    def _doi_link(value: object) -> str:
        if not value:
            return ""
        doi = str(value)
        return f'<a href="https://doi.org/{html.escape(doi)}">{html.escape(doi)}</a>'


LABELS = {
    "permanent_magnets": "永磁领域",
    "running": "运行中",
    "finished": "已完成",
    "failed": "失败",
    "downloaded_oa_pdf": "已下载开放 PDF",
    "preprint_pdf": "预印本 PDF",
    "oa_pdf_available": "开放 PDF 可下载",
    "oa_no_pdf_url": "开放但无 PDF 链接",
    "closed_access_has_doi": "非开放，有 DOI",
    "closed_access_missing_doi": "非开放，缺 DOI",
    "download_failed": "下载失败",
    "needs_manual_review": "需要人工复核",
    "planned": "已规划",
    "awaiting_user_approval": "等待人工确认",
    "approved": "已确认",
    "acquiring_pdfs": "正在获取 PDF",
    "awaiting_manual_pdfs": "等待手动 PDF",
    "analyzing": "正在分析",
    "synthesized": "已综合",
    "next_round_proposed": "已提出下一轮",
    "pending": "待处理",
    "completed": "已完成",
    "high": "高",
    "medium": "中",
    "low": "低",
    "title_missing_or_short": "标题缺失或过短",
    "missing_doi": "缺 DOI",
    "low_relevance": "相关性偏低",
    "pdf_missing_or_empty": "PDF 缺失或为空",
    "id": "ID",
    "title": "标题",
    "year": "年份",
    "venue": "期刊/来源",
    "doi": "DOI",
    "doi_url": "DOI URL",
    "publisher": "出版方",
    "publisher_url": "出版商页面",
    "source_url": "来源页面",
    "pdf_url": "PDF URL",
    "access_status": "获取状态",
    "document_type": "文献类型",
    "relevance_score": "相关性分数",
    "download_error": "下载错误",
}


def _display_label(value: str) -> str:
    if not value:
        return ""
    return LABELS.get(value, value)


def _shorten(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"

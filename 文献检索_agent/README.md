# 文献检索 Agent

这是一个面向任意研究主题的单进程、多 Agent 文献检索 CLI 原型。v2 主线是可复现的多源文献语料构建：多源召回、限流与失败记录、元数据去重、DOI/OA/PDF 可得性追踪、边界相关性复核和覆盖率报表。

目标：

- 尽可能全地发现论文和预印本文献。
- 开放获取 PDF 自动下载。
- 非开放文献不绕过权限，但保留 DOI、DOI URL、出版商页面和来源线索。
- 所有结果写入 SQLite，PDF 存入本地文件库。

## 快速开始

```bash
python3 run.py init
python3 run.py goal create --title "研究对象在特定条件下的机制与验证问题"
python3 run.py round plan --goal-id 1 --target-count 20
python3 run.py round approve --round-id 1
python3 run.py round acquire --round-id 1
python3 run.py round intake-manual --round-id 1
python3 run.py round analyze --round-id 1
python3 run.py round propose-next --round-id 1
python3 run.py web
python3 run.py resolve-oa
python3 run.py review-relevance
python3 run.py download
python3 run.py audit
python3 run.py report
python3 run.py dashboard
python3 run.py export --format csv
```

`--mode` 控制检索规模：

- `smoke`：小规模连通性试跑。
- `pilot`：约百篇级别验收，用于 v2 方法检查。
- `full`：使用配置中的完整查询规模。

v3 推荐先使用 `goal` / `round` 命令做小批量迭代探索。系统会从科学问题的标题和说明中提取检索与相关性关键词；每轮默认 20 篇，系统会先给出候选，等待人工确认后才下载；非开放或下载失败论文会进入手动下载任务，推荐放置目录为 `data/manual_pdfs/goal_<id>/round_<id>/`。

也可以启动本地交互网页：

```bash
python3 run.py web
```

然后打开 `http://127.0.0.1:8765`。这个页面用于创建科学问题、规划轮次、人工确认、获取 PDF、扫描手动 PDF、分析本轮和查看下一轮建议。

也可以单独运行每个流水线阶段：

```bash
python3 run.py plan-queries --mode pilot
python3 run.py discover --mode smoke --sources openalex,crossref --from-year 2020 --to-year 2026
python3 run.py normalize
python3 run.py dedup
python3 run.py judge-relevance
python3 run.py review-relevance
```

`reports/dashboard.html` 是项目的可视化工作台：包含多 Agent 工作关系、数据库计数、访问状态分布、来源覆盖、v2 方法指标、源失败统计、LLM 复核结论、最近运行、样本文献和报表链接。后续调整 Agent 架构和检索策略时，优先生成并查看这个 HTML。

如需用 Unpaywall 补全 DOI 的开放获取状态，可设置：

```bash
export UNPAYWALL_EMAIL="your_email@example.com"
python3 run.py resolve-oa
```

边界相关性复核使用 OpenAI 兼容接口。没有 API key 时系统仍会完成规则流程，并把 LLM 跳过原因写入复核记录：

```bash
export OPENAI_BASE_URL="https://your-openai-compatible-endpoint/v1"
export OPENAI_API_KEY="your_api_key"
export OPENAI_MODEL="your_model"
python3 run.py review-relevance
```

v3 的本轮文献推荐默认是确定性规则排序；如果要让外部大语言模型参与推荐，请让模型只重排已经从数据源检索到的真实候选，而不是生成参考文献。启用方式：

```bash
export LIT_AGENT_SELECTION_MODE="llm"
python3 run.py round plan --goal-id 1 --target-count 20
```

也可以复制 `configs/local.env.example` 为 `configs/local.env`，把 `OPENAI_BASE_URL`、`OPENAI_API_KEY`、`OPENAI_MODEL`、`LIT_AGENT_SELECTION_MODE` 和 `UNPAYWALL_EMAIL` 填进去。`configs/local.env` 不应提交到版本控制。

LLM 推荐的约束：

- LLM 只能从真实候选池的 `paper_id` 中选择，不能凭空生成论文、DOI 或 PDF 链接。
- LLM 失败、未配置或返回了不存在的 `paper_id` 时，系统会自动用规则推荐补足。
- Web UI 顶部会显示当前是“规则推荐”还是“外部 LLM 重排 + 规则兜底”。

默认路径：

- 数据库：`data/literature.sqlite`
- PDF 文件库：`data/pdfs`
- 报表：`reports`

v2 报表会额外输出：

- `source_failures.csv`：数据源失败、429、timeout、HTTP 5xx 等分类记录。
- `llm_relevance_reviews.csv`：边界样本 LLM 复核或跳过记录。
- `pipeline_metrics.csv`：DOI 覆盖率、OA/PDF 可得率、去重压缩率等方法指标。

## PDF 存储位置

PDF 下载目录可以用全局参数 `--pdf-dir` 指定，适合把大量 PDF 放到外接硬盘：

```bash
python3 run.py --pdf-dir /Volumes/YourDisk/research_pdfs download
```

推荐做法是：SQLite 数据库留在项目目录，PDF 大文件放外接硬盘。数据库会记录 PDF 本地路径、文件哈希、大小和下载状态；硬盘未连接时不要运行 `download`，否则会把文件写到错误位置或触发路径问题。

## 架构

核心 Agent：

- `OrchestratorAgent`：总调度。
- `ScientificGoalAgent`：创建科学问题。
- `RoundPlanningAgent`：规划一轮小批量文献探索。
- `LiteratureSelectionAgent`：按多样性 + 核心策略选择本轮文献；可选外部 LLM 对真实候选池重排。
- `ManualPdfIntakeAgent`：接收用户手动下载的 PDF。
- `PdfAnalysisAgent`：分析 PDF 或元数据。
- `EvidenceSynthesisAgent`：综合本轮证据并生成缺口。
- `NextQueryProposalAgent`：提出下一轮检索建议。
- `DomainQueryAgent`：领域查询规划。
- `SourceDiscoveryAgent`：多源检索。
- `MetadataNormalizeAgent`：元数据清洗。
- `DeduplicationAgent`：去重合并。
- `RelevanceJudgeAgent`：领域相关性判断。
- `OAResolverAgent`：开放获取与 DOI 解析。
- `PdfDownloadAgent`：PDF 下载。
- `QualityAuditAgent`：质量审计。
- `ReportExportAgent`：报表与导出。
- `HtmlDashboardAgent`：生成可视化 HTML 工作台。
- `LLMRelevanceReviewAgent`：只复核边界相关性样本，并写入结构化结论。

LLM 只参与查询扩展、边界复核和候选重排等辅助判断；核心入库、去重、下载、状态流转和报表统计均由确定性代码完成。

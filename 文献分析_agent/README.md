# 文献分析 Agent

这是 AI Scientist 工作流的第 2 个多 Agent 系统，负责把第 1 步 `文献检索_agent` 收集到的 PDF 转换成 Markdown/JSON，并进一步整理成 RAG 索引、Paper Cards、轻量知识图谱和证据化 Wiki。

默认输入：

- 数据库：`../文献检索_agent/data/literature.sqlite`
- PDF 文件库：`../文献检索_agent/data/pdfs`
- 轮次 PDF：`../文献检索_agent/data/goal_pdfs`

默认输出：

- Markdown/JSON：`data/parsed_papers`
- RAG 索引状态：`data/index`
- Paper Cards：`data/cards`
- 轻量图谱：`data/graph/graph.json`
- Wiki 条目：`data/wiki`
- 转换状态记录：写回检索数据库中的 `paper_conversions` 表，方便 8765 工作流前端统一展示。

## 快速开始

```bash
python3 run.py init
python3 run.py convert-pdfs --limit 2
python3 run.py convert-pdfs
python3 run.py convert-pdfs --round-id 6
python3 run.py convert-pdfs --force
python3 run.py build-all
```

当前版本使用 PyMuPDF 做本地 PDF 文本抽取，并用规则识别标题、摘要、章节、正文段落、图注、表格标题和参考文献。本阶段不做 OCR；扫描版或文本不可抽取 PDF 会记录为失败或低质量结果。

## 知识整理流水线

```bash
python3 run.py index-corpus
python3 run.py search coercivity --limit 5 --json
python3 run.py build-paper-cards
python3 run.py build-graph
python3 run.py build-wiki
python3 run.py build-all --json
```

- RAG 层：从 `parsed_papers/*.json` 建立 SQLite FTS5 索引，并写入本地哈希向量；未配置外部 embedding 时自动使用本地向量兜底。
- Paper Cards 层：每篇论文生成研究对象、材料体系、方法、性能、结论、局限和 evidence ids。
- Graph 层：生成 `wiki_topic`、`paper`、`claim`、`material`、`method`、`property`、`evidence` 节点，并导出 `data/graph/graph.json`。
- Wiki 层：围绕 topic 生成带 evidence ids 的 Markdown/JSON 条目；没有证据的内容进入 `needs_evidence`。

8765 前端仍由同级 `文献检索_agent` 提供，`/paper-analysis` 会调用这里的 CLI 和读取这里的 `data/` 产物。

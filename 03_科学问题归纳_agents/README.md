# 科学问题归纳 Agent

这是 AI Scientist 工作流的第 3 个模块。它读取第 1 步文献检索提出的科学问题、下一轮 query 和证据缺口，再结合第 2 步文献分析生成的 Paper Cards、Wiki 条目和知识图谱，帮助研究者通过对话逐步收敛可研究、可验证的科学问题与细化方向。

默认输入：

- 检索数据库：`../文献检索_agent/data/literature.sqlite`
- 分析产物：`../文献分析_agent/data`
- LLM 配置：优先读取环境变量，也可由 8765 前端沿用 `../文献检索_agent/configs/local.env`

默认输出：

- 对话数据库：`data/question_synthesis.sqlite`

## 快速开始

```bash
python3 run.py init
python3 run.py state
python3 run.py chat "把问题聚焦到可计算验证的方向"
python3 run.py reset
```

8765 前端由同级 `文献检索_agent` 提供，进入 `/question-synthesis` 即可打开第 3 模块对话界面。

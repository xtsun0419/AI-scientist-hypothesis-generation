# AI 假设生成 Agent 改进计划

> 范围：`03_科学问题归纳_agents`（科学问题收敛）与 `04_提出路线`（研究路线生成）两个模块，以及它们与 01 文献检索、02 文献分析之间的数据流。
> 参考资料：
> - AutoResearch AI 综述（arXiv:2605.23204）及导读笔记（`REPORT/自动化科研综述导读.md`）
> - Google Co-Scientist（Nature 2026）、SciAgents / GraphAgents（MIT）、AI Scientist-v2、MOOSE-Chem 系列
> - 评估基准结论：SGI-Bench、IdeaBench、HindSight、FIRE-Bench、ResearchBench、BioDisco、SoundnessBench

---

## 一、现状诊断（基于代码）

1. **知识图谱是"死资产"**：02 模块已生成 `data/graph/graph.json`（含 wiki_topic / paper / claim / material / method / property / evidence 节点），但 03、04 的 `collect_context()` 完全未读取，只用截断的 cards / wiki JSON（前 8–12 条）。
2. **03 → 04 数据流断点**：04 的 `question_options` 只来自 01 的 `scientific_goals` 表与 wiki 的 open_questions；03 对话数据库（question_synthesis.sqlite）中的收敛结果没有结构化地流入 04。03 是"聊天室"，聊完即丢。
3. **生成即终点**：04 是单次 LLM 调用生成 JSON，兜底为 4 个固定模板；无 critic、无假设间比较、无迭代演化。
4. **无新颖性检查**：系统从不问"这条路线是否已被文献做过"。
5. **假设不可审计**：路线 evidence 字段只是文本，未绑定 evidence ids；无假设演化历史；无 provenance 链；研究者反馈无法回流。

## 二、文献核心结论（设计依据）

| 来源 | 结论 |
|---|---|
| 2605.23204 Stage II | 假设形成的技术前沿不是"创意生成量"，而是 **disciplined scientific search**：候选方向在执行前必须保持 **grounded、comparable、feasible、rejectable** |
| 2605.23204 Fig.7 | 执行前应有 **Scientific Selection Gate**：证据支持 / 新颖性 / 可行性 / 资源成本 / 风险安全 五约束过滤 |
| 2605.23204 Fig.7 | Structure-guided 的四种规划操作：**Gap detection**（未探索关系）、**Analogy search**（跨域迁移）、**Constraint matching**（领域规则）、**Feasibility mapping**（工具/数据对齐） |
| 2605.23204 6.2 | Novelty 评估三原则：**literature-aware + temporally grounded + expert-mediated**；LLM-as-Judge 易被表面模式欺骗 |
| Co-Scientist (Nature 2026) | 生成→辩论→排序→演化循环 + Elo 两两辩论（Idea Tournament）+ Proximity 去重 + Meta-review 反馈写回提示词 |
| SciAgents / GraphAgents | 知识图谱启发式遍历发现"未连接的跨域组合"，生成可追溯假设 |
| SGI-Bench / IdeaBench | LLM 想法普遍"新颖性高、可行性低"；评估必须区分二者 |
| Artificial Hivemind | LLM 想法多样性坍缩，需要显式多样性约束 + 相似度去重 |
| Intern-Atlas | 确定性图指标评分与专家相关性 0.81，显著高于 LLM 打分 0.58 |
| 2605.23204 6.1 | 当前系统"well-executed studies around weak hypotheses"——缺反思式迭代（假设↔实验↔异常↔修订回路） |
| 2605.23204 5.4 | 材料领域"计算可行 ≠ 实验室可合成"——路线应区分"计算可验证"与"实验可验证"两级假设 |

## 三、分优先级计划

### P0 — 打通 03→04 数据流 + 结构化科学问题（成本低，收益最高）

1. 03 增加"确认问题"动作：LLM 把收敛结果写成结构化 JSON 存入 `question_synthesis.sqlite` 新表 `confirmed_questions`：
   - 字段：`problem_statement`（问题陈述）、`variables`（关键变量）、`mechanism_hypothesis`（机制假设）、`validation_criteria`（验证判据）、`evidence_ids`（支撑证据 id）、`source_message_id`（来源对话）、`created_at`
   - CLI 增加 `run.py confirm` 子命令 + 8765 前端按钮。
2. 04 优先读 `confirmed_questions` 作为问题来源（其次才回退到现在的拼接逻辑）。
3. 04 每条路线的 `evidence` 字段绑定真实 evidence ids，并在输出中标注"证据支撑 / 推测"两类。
4. 03 对话记录研究者修正次数，作为"人类介入成本"指标（对应 2605.23204 的 evaluative burden 概念）。

### P1 — 激活知识图谱：图引导假设生成 + 图指标新颖性

1. **图上下文注入**：03/04 的 prompt 中注入问题相关的图局部上下文（相关节点 + 边 + evidence ids），替代/补充现在截断的 cards JSON。
2. **图规划操作器**（对应 2605.23204 的四种操作，02 的 graph.json 为输入）：
   - `gap_detection`：在图中找"材料 A 用了方法 X，但方法 X 未用于材料 B"类未连接路径 → 候选假设来源
   - `analogy_search`：找结构同构的跨领域关系（如不同材料体系中相同的"晶界相厚度→矫顽力"模式）
   - `constraint_matching`：用图上的已知限制（limitations 节点）过滤候选方向
   - `feasibility_mapping`：把候选方向映射到 corpus 中已有的 method 节点，标注可执行性
3. **图指标新颖性评分**（确定性，可审计）：
   - 候选组合在图中已有直接路径 → 新颖性低
   - 候选组合在图中有桥接路径（中间节点）→ 新颖性中
   - 候选组合在图中无任何路径 → 新颖性高
   - 作为 04 输出每条路线的 `graph_novelty` 字段，与 LLM 判断并存互证。

### P2 — Critic + 两两比较排序（Co-Scientist 精简版）

1. 新增 `CriticAgent`：对每条路线生成结构化批评（新颖性 / 可行性 / 证据充分性 / 可证伪性四维），输出 JSON。
2. **Hypothesis Arena**：两两比较 + Elo 排序替代绝对打分（避免 LLM-as-Judge 乐观偏差）。
3. 一轮演化：critic 意见反馈给生成器产出 v2 路线（保留父本链接，形成谱系）。
4. 演化历史写入 `route_candidates.json` 的 `lineage` 字段（parent → child），可审计。

### P3 — 假设去重与多样性（低成本）

1. Proximity 检查：生成的多条路线两两计算相似度（文本 + 图路径重叠度），过近的合并或强制重新生成。
2. 显式多样性约束：生成时要求各路线覆盖不同的图区域 / 不同机制层级（机制型 / 体系型 / 工艺型 / 数据型），对抗多样性坍缩。
3. 与 P2 的 Elo 排序共享同一比较器。

### P4 — 可验证性做实 + 回溯评估（论文加分项）

1. 每条路线强制输出 `falsifiable_prediction`（可证伪预测）+ `minimal_experiment`（最小验证实验：方法 / 数据 / 判据）。
2. 区分两级验证假设：`computationally_verifiable`（DFT、微磁模拟、解析估计）与 `experimentally_verifiable`（工艺-微结构-性能闭环），分别给出验证路径（对应材料领域"计算可行 ≠ 可合成"）。
3. 回溯评估（FIRE-Bench rediscovery 思路）：
   - 用 2024 年以前的文献构造证据库（时间分割避免泄漏）
   - 看系统能否"重新发现"2024–2025 年的热点方向（如稀土减量、晶界扩散的新组合）
   - 指标：命中率 + 新颖性/可行性双维专家盲评（对应 novelty 三原则中的 temporally grounded + expert-mediated）

## 四、验证与评估方法

| 层 | 方法 |
|---|---|
| 单元级 | 各 Agent 输出 schema 校验；图操作器与手工标注的图查询结果对比 |
| 系统级 | 回溯测试命中率；多样性指标（路线间相似度分布）；人类介入次数（03 对话修正数） |
| 质量级 | 双盲专家评估（新颖性 / 可行性 / 证据充分性 1–5 分），LLM 评分仅作参考（有乐观偏差） |
| 可靠性 | 同上下文多 seed 运行，看路线集合稳定性（对应 2605.23204 Reliability 维度） |

## 五、风险与注意事项

1. **LLM 自我强化风险**：critic 与 generator 同模型时辩论可能只是"生成更多文本"——需独立证据来源（图指标、RAG 检索）做仲裁。
2. **图质量上限**：02 的图是词表规则构建，关系类型简单——图引导的上限受图质量约束（2605.23204 明确警告），必要时先用 LLM 补一次关系抽取。
3. **新颖性 ≠ 有价值**：图指标新颖性只能排除"已做过"，不能证明"值得做"——最终仍需专家中介。
4. **成本控制**：每轮 LLM 调用量会上升（生成 + critic + arena），需设预算上限与降级路径（本地规则兜底已有基础）。

## 六、实施顺序建议

```
第一步（架构）  P0：确认问题结构化 + 03→04 打通
第二步（资产）  P1：图上下文注入 + gap_detection + 图新颖性指标
第三步（质量）  P2：Critic + Elo Arena + 一轮演化
第四步（稳健）  P3：去重 + 多样性约束
第五步（论证）  P4：可证伪性 + 回溯评估（论文评估章节素材）
```

P0+P1 构成"03 产出结构化问题 → 04 用图谱+证据生成路线"的完整链路，是本项目相对已有系统（多数无图支撑）最有辨识度的贡献点。

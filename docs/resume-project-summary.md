# 多智能体研究助手 — 项目总结（简历用）

> 适用岗位：AI 应用开发 / 大模型应用工程师 / AI Agent 开发

## 1. 一句话概述

基于 **LangGraph 编排 + vendored ReAct Agent 框架 + MCP 工具协议** 的多智能体深度研究系统，6 个专用 Agent 协作完成「主题分解→多源搜索→分析→综合→撰写→评审」全流程，自动生成高质量研究报告。支持**跨轮次增量记忆**（已验证事实/搜索反馈/草稿迭代）、**Human-in-the-Loop 人工审核闭环**、**崩溃恢复**（从任意节点 checkpoint 续跑）。

---

## 2. 技术栈

| 层次 | 技术 | 用途 |
|------|------|------|
| 编排 | LangGraph (StateGraph) | Agent 工作流编排、条件路由、checkpoint 持久化 |
| Agent 框架 | vendored OpenManus (~950 行) | ReAct 循环（Think → Act）、工具绑定、状态管理 |
| LLM 调用 | 原生 SDK (Anthropic/OpenAI/DeepSeek/Ollama) | 4 种 Provider 统一桥接，消息格式双向转换 |
| 工具协议 | MCP (Model Context Protocol) | 双模式工具架构（进程内 / stdio 子进程） |
| 搜索 | Brave, Tavily, DuckDuckGo, arXiv, Wikipedia, Jina | 多源并行搜索（asyncio.gather + Semaphore） |
| 进度显示 | Rich (Live + Table) | 6 Agent 实时状态展示 |
| CLI | Typer + Loguru | 命令行入口 + 结构化日志 |

---

## 3. 架构核心设计

```
用户输入主题
     │
     ▼
┌──────────────────────────────────────────────────────┐
│  Planner ──→ Searcher ──→ Analyst ──→ Synthesizer    │
│     ↑                        │            │          │
│     │                        ▼            ▼          │
│     └────── score < 阈值 ── Critic ←── Writer        │
│                              │                       │
│                         score >= 阈值                 │
│                              ▼                       │
│                         Formatter → 最终报告           │
└──────────────────────────────────────────────────────┘
```

### 关键设计决策

| 决策 | 理由 |
|------|------|
| **LangGraph StateGraph** 而非自建流程引擎 | 内置条件路由、checkpoint 持久化、interrupt() 人工审核 |
| **Vendoring Agent 框架** (~950行) 而非 pip 依赖 | 实际需要代码极少，外部依赖却引入 264+ 传递包 |
| **Supervisor-Worker 拓扑** 而非 Peer-to-Peer | 研究任务天然适合「线性推进 + 质量门禁回退」模式 |
| **每个 Agent 独立 LLM 配置** | Planner 用廉价模型、Analyst 用强推理模型，降成本 |
| **MCP 双模式** | 进程内直接调用用于开发调试，stdio 子进程用于生产隔离 |
| **asyncio.gather 并行搜索** 而非 LangGraph Send API | 同质查询（同一套工具）用 gather 更简洁；异构分发才需要 Send |
| **Operator.add reducer 实现增量记忆** | 跨轮累积 verified_facts/round_history/search_feedback，不覆盖历史 |
| **Writer 迭代修订 vs 每轮重写** | Round ≥2 时传入 previous_draft + critic feedback，仅修改缺陷部分 |

---

## 4. 核心实现细节

### 4.1 Agent 框架（vendored）

从 OpenManus 裁剪内联，继承链：`BaseAgent → ReActAgent → ToolCallAgent → 6 个专用 Agent`

```
ToolCallAgent.think()
  → llm.ask_tool(messages, system_msgs, tools)
    → LLMProvider (多 Provider 统一调用)
      → 原生 SDK (Anthropic/OpenAI/DeepSeek/Ollama)  ← 不依赖 LangChain
  ← ToolCallResult (content + tool_calls)

ToolCallAgent.act()
  → ToolCollection.execute(name, args)
    → MCP 模式: MCPClientTool → stdio JSON-RPC → MCP Server 子进程
    → 进程内模式: tool.execute(**kwargs) 直接调用
  → Terminate 工具设置 state = FINISHED，结束循环
```

### 4.2 LLMProvider — 多 Provider 桥接

封装各 Provider 原生 SDK，重点是 **消息格式转换**：

- **Anthropic**: tool_use/tool_result 通过 content blocks 表达（和 OpenAI 的 tool_calls 数组完全不同）
- **OpenAI / DeepSeek**: tool_calls 数组 + tool role 消息
- 统一返回 `ToolCallResult`（兼容 OpenManus 内部格式）

```python
# 关键点：绕过 Pydantic validator 注入 LLM 实例
agent = CriticAgent(available_tools=tools)
agent.__dict__["llm"] = LLMProvider(config)  # 而非 agent.llm = ...
```

> `BaseAgent.llm` 字段的 model_validator 会将非 `app.llm.LLM` 类型的值替换为默认实例。直接操作 `__dict__` 绕过此检查，保证配置的 Provider 被使用。

### 4.3 状态管理与增量记忆（v0.7.0）

`ResearchState` 是 `TypedDict`（`total=False`），节点间通过 LangGraph 的状态合并机制传递数据：

- `search_results`、`analyses`、`accumulated_knowledge`、`round_history`、`search_feedback` 使用 `Annotated[list, operator.add]` reducer — 跨节点/跨轮次追加自动合并
- `previous_draft` 使用普通 `str` 字段 — 只需要最新版草稿，不保留历史
- `messages` 使用 `add_messages` — LangGraph 内置的消息合并 reducer

**增量记忆机制**：多轮迭代时，通过 4 个 state 字段传递跨轮上下文：

| 字段 | 来源 | 消费方 | 作用 |
|------|------|--------|------|
| `accumulated_knowledge` | Analyst（verified_facts + confidence） | Planner, Analyst, Synthesizer | 避免重复搜索已高置信度事实 |
| `round_history` | Critic（每轮评分/优缺点/gaps） | Planner | 了解历史覆盖面和薄弱点 |
| `search_feedback` | Critic（query 有效性评估） | Planner | 避开无效搜索方向 |
| `previous_draft` | Critic（记录上轮草稿） | Writer | 迭代修订而非从零重写 |

所有增量行为通过 **prompt 上下文传递**（不修改 Agent 框架代码），当字段为空时自动降级为原始行为。

### 4.4 并行搜索实现

Searcher 节点是唯一需要并行化的节点：

```python
semaphore = asyncio.Semaphore(max_parallel)  # 控制并发

async def _search_one(query, index):
    async with semaphore:
        agent = searcher_factory()  # 每个查询独立 agent 实例
        await agent.run(query_request)
        return extract_result(agent)

tasks = [_search_one(q, i) for q in queries]
results = await asyncio.gather(*tasks, return_exceptions=True)
```

设计要点：
- **工厂函数** 每次创建新的 Agent 实例，状态隔离
- **Semaphore** 限流防止 API rate limit
- **`return_exceptions=True`** 单个查询失败不崩溃
- 结果通过 `operator.add` reducer 自动合并

### 4.5 Human-in-the-Loop (HITL)

基于 LangGraph `interrupt()` 实现执行暂停 + 人工决策：

```
critic → human_review_node (interrupt) → [暂停]
                                            │
                            ┌───────────────┼───────────────┐
                            ▼               ▼               ▼
                        approve          revise          abort
                            │               │               │
                            ▼               ▼               ▼
                       formatter      planner (再一轮)   formatter
```

关键实现：
```python
# human_review_node 中
decision = interrupt({"report_file": ..., "quality_score": ..., ...})

# main.py 中
tracker.pause()  # 暂停 Rich Live，避免覆盖输入行
decision = console.input("👉 决策: ").strip()
tracker.resume()
final_state = await workflow.ainvoke(Command(resume=decision), config)
```

### 4.6 Critic 评分与 JSON 提取

Critic 评审报告质量（6 维度 0-100 评分），`parse_review()` 用多层降级策略提取 JSON：

1. Markdown code fence（```json ... ```）
2. **花括号深度计数**：从 `"overall_score"` 位置向前找 `{`，通过计数花括号深度找到匹配的 `}` — 解决了简单正则 `\{[^{}]*\}` 无法匹配嵌套 `scores` 对象的问题
3. 简单正则兜底

实践中还遇到 LLM 直接调 Terminate 不产内容的问题，通过两个手段解决：
- 请求中加入强指令（"MUST output JSON BEFORE Terminate"）
- 消息选择从「取最后一条」改为从后向前搜索实质性内容（>50 字符）

### 4.7 崩溃恢复（v0.7.0）

基于 LangGraph SQLite checkpoint 实现从任意节点恢复执行：

```python
# main.py — 启动时记录运行状态
thread_id = str(uuid.uuid4())
console.print(f"Run ID: {thread_id}")
console.print(f"If interrupted, resume with: uv run python -m src.main resume --last")
save_run_record(thread_id, topic, status="running")

# 崩溃后恢复入口
uv run python -m src.main resume --last    # 最近中断的 run
uv run python -m src.main resume <run-id>  # 指定 run
```

恢复流程：
1. 从 `./data/runs.json` 查找中断记录（按 `updated_at` 倒序，`--last` 取第一条）
2. 连接 `./data/checkpoints.sqlite`，用 `thread_id` 加载 checkpoint state
3. **HITL 中断**：state 有 pending interrupt → 重新显示评审面板 → 等待用户输入 → `Command(resume=decision)`
4. **节点崩溃**：state 无 interrupt → `workflow.ainvoke(None, config)` 从上一 super-step 重新执行
5. 异常时保存 `crashed` 状态 + 错误信息，打印恢复提示

设计要点：
- 运行记录 `runs.json` 与 checkpoint 文件解耦，前者用于快速查找和列表，后者用于 LangGraph 恢复
- `--last` 消除了用户记 thread_id 的负担

### 4.8 节点输出调试（_NodeDebugSaver）

将每个节点的输出 dict 序列化保存到 `debug/<时间戳>_<主题>/<序号>_<节点名>.json`，用于：
- 质量评估（对比各节点的输入输出）
- 问题诊断（如发现 Critic 的 `quality_score` 恒为 0）
- 运行时可配置开关（`config/research.yaml` 中 `debug_dir`）

---

## 5. 难点与解决方案

| 难点 | 原因 | 解决方案 |
|------|------|---------|
| **quality_score 恒为 0** | Critic agent `max_steps=1`，LLM 一步直接 Terminate，不产评审内容 | 增大 max_steps；请求加强指令；消息选择改为后向搜索 |
| **嵌套 JSON 解析失败** | Critic 的 JSON 含嵌套 `scores` 对象，简单正则 `\{[^{}]*\}` 在第一个内层 `}` 处截断 | 改用花括号深度计数，正确匹配嵌套结构 |
| **LangGraph conditional edge 中的 state mutation 被丢弃** | Edge 函数不是 Node，返回值不合并到 state | `research_round` 递增移到 `critic_node` 返回值 |
| **HITL 输入不可见** | Rich Live 刷新覆盖 `console.input()` 的输入行 | `ProgressTracker.pause()/resume()` 在输入期间暂停 Live |
| **Event loop 关闭噪声** | Windows asyncio 子进程在 GC 时尝试操作已关闭的 event loop | `try/except RuntimeError` 静默清理阶段错误 |
| **Pydantic validator 替换 LLM 实例** | `BaseAgent` 的 model_validator 将非 `app.llm.LLM` 对象替换为默认值 | `agent.__dict__["llm"] = provider` 绕过 validator |
| **MCP 子进程 stderr 泄露** | anyio.open_process 默认继承父进程 stderr | 传递 `errlog=open(os.devnull, "w")` |
| **多轮迭代信息丢失** | Critic→Planner 回环只传 gaps 文字，搜索结果、已验证事实、搜索有效性全被丢弃 | 新增 4 个 operator.add 增量记忆字段，通过 prompt 上下文传递给后续轮次 |
| **崩溃后无法续跑** | 每次 run 生成随机 UUID，崩溃后无入口获取 thread_id | 运行记录持久化到 runs.json，CLI 增加 `resume --last` 一键恢复 |

---

## 6. 面试可能考察点

### Q1: 为什么选择 LangGraph 而不是自己写编排逻辑？

**答**：LangGraph 提供了三个不可或缺的能力：① **条件路由**（Supervisor → Planner 或 Formatter）；② **checkpoint 持久化**（崩溃后可恢复）；③ **`interrupt()`**（HITL 人工审核的基础）。自建编排也可以，但会重复造轮子。选择 LangGraph 是因为它的抽象层次刚好——比裸 asyncio 高（有状态管理），比 LangChain Agent 低（不做 Agent 运行时，只做编排）。

### Q2: 为什么不直接用 LangChain 的 Agent？

**答**：LangChain Agent 绑定其工具定义和 LLM 调用方式，而我们选择了 vendored OpenManus 框架作为 Agent 运行时——它更轻量（~950 行）、行为可控（ReAct 循环完全透明）、且支持我们自己的 LLMProvider 多 Provider 桥接层。LangGraph 作为「纯编排框架」可以和任意 Agent 运行时组合，正好满足需求。

### Q3: MCP 协议解决了什么问题？和直接调工具有什么区别？

**答**：MCP 将工具从主进程中解耦出来，运行在独立子进程中。好处：① **进程隔离**（工具崩溃不影响主流程）；② **语言无关**（工具可以用任意语言实现）；③ **标准化**（外部 MCP Server 可以直接接入，如 GitHub MCP Server）。双模式设计（配置项切换）保留了开发时的便利性。

### Q4: Anthropic 和 OpenAI 的消息格式有什么区别？怎么处理的？

**答**：核心差异在工具调用。OpenAI 用 `tool_calls` 数组 + `role=tool` 消息；Anthropic 用 `content` blocks，`tool_use` block 表示调用工具，`tool_result` block（嵌在 user 消息中）表示工具结果。`LLMProvider._to_anthropic_messages()` 和 `_to_openai_messages()` 各自实现转换。响应也统一转回 `ToolCallResult` 格式。

### Q5: 并行搜索怎么实现的？为什么不用 LangGraph 的 Send API？

**答**：用 `asyncio.gather` + `asyncio.Semaphore`。每个查询创建独立 Agent 实例（工厂函数），Semaphore 控制最大并发。选择 gather 而非 Send API 是因为 Searcher 的查询是**同质的**（同一套工具、同一种 Agent），用 gather 更简洁，图拓扑不变。Send API 更适合异构并行（不同节点不同逻辑）。

### Q6: LangGraph 的 conditional edge 函数中能修改 state 吗？

**答**：**不能**。Edge 函数只是纯路由函数（返回下一个节点名），它的返回值不会被 LangGraph 合并到 state 中——LangGraph 会在不报错的情况下**静默丢弃**。这就是为什么 `research_round` 的递增放在 `critic_node` 的返回值中，而不是 `_supervisor_router` 中。这是一个实践中容易踩的坑。

### Q7: 怎么绕过 Pydantic BaseModel 的 validator？

**答**：直接操作 `__dict__`：`agent.__dict__["llm"] = provider`。不优雅但是务实——Pydantic model_validator 的类型检查在这里是阻碍而非帮助。如果追求更干净的方式，可以在 model_config 中设置 `validate_assignment=False` 或用 `object.__setattr__()`。

### Q8: 这个系统最大的限制是什么？

**答**：① Agent `max_steps` 是硬限制——步数用完直接截断，输出可能不完整（如之前的 quality_score=0 bug）；② 引用提取依赖 LLM 自行解析，未做结构化处理，引用可能不准确；③ `PythonExecuteTool` 无 Docker 隔离，不可信代码不能执行；④ 没有 Token 计数和成本统计；⑤ 崩溃恢复要求 config 文件不变更（workflow 结构需与中断时一致）；⑥ Searcher 的并行仅限同质查询，异构分发待支持。

---

## 7. 项目规模

| 指标 | 数值 |
|------|------|
| Agent 框架（vendored） | ~950 行 Python（8 个模块） |
| 专用 Agent | 6 个 |
| 工具（含搜索、分析、导出） | 10 个 |
| 支持的 LLM Provider | 4 个（Anthropic / OpenAI / DeepSeek / Ollama） |
| 总代码量 | ~3500 行 |
| 开发周期 | 约 2 周 |

---

## 8. 可以延伸的方向（面试加分）

- Token 计数 + 成本统计 → 体现对 LLM 应用的成本意识
- Docker 隔离 `PythonExecuteTool` → 体现代码执行安全意识
- LangGraph Send API 实现异构并行 → 体现对框架深入理解
- 结构化引用提取（用 LLM 输出 JSON Schema 约束）→ 体现代码质量意识
- Streaming 输出报告 → 体现用户体验意识
- `runs.json` 自动清理 + 多 run 管理 → 体现运维意识

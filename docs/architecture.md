# Multi-Agent Research Framework — 架构与实现文档

> 基于 LangGraph 编排 + vendored Agent 框架 + MCP 工具服务器的多智能体研究助手。

## 1. 项目概述

用户输入一个研究主题，6 个专用 Agent 在 LangGraph StateGraph 编排下协作完成：

**主题分解 → 多源搜索 → 深度分析 → 综合归纳 → 报告撰写 → 质量审查 → 最终报告**

### 核心设计原则

- **关注点分离**：LangGraph 负责编排（何时执行谁），Agent 框架负责 ReAct 循环（如何思考与行动），工具层负责外部数据获取
- **模型可配置**：每个 Agent 通过 `config/agents.yaml` 独立配置 provider / model / temperature
- **质量闭环**：Critic 评分 → 低于阈值自动重试（最多 N 轮）
- **轻量 vendoring**：Agent 基座代码（~950 行）内联在 `src/_framework/`，不依赖外部 OpenManus 仓库
- **双模式工具**：支持进程内直接调用或 MCP Server 子进程远程调用，通过配置切换

---

## 2. 系统架构

### 2.1 拓扑结构

```
用户输入研究主题
     │
     ▼
┌──────────────────────────────────────────────────────┐
│                  LangGraph StateGraph                │
│                                                      │
│   Planner → Searcher → Analyst → Synthesizer → Writer → Critic
│      ↑                                                      │
│      └── score < threshold && rounds remain ────────────────┘
│                                          │                   │
│                                     score ok                │
│                                          ▼                   │
│                                     Formatter → END         │
└──────────────────────────────────────────────────────┘
     │
     ▼
  ./reports/report-*.md
```

### 2.2 Agent 角色

| Agent | 继承自 | 职责 | 工具 |
|-------|--------|------|------|
| **PlannerAgent** | ToolCallAgent | 分解主题、生成搜索查询、制定大纲 | Terminate |
| **SearcherAgent** | ToolCallAgent | 多源搜索（Web/arXiv/Wikipedia） | Brave, Tavily, DuckDuckGo, arXiv, Wikipedia, Jina, WebScraper, Terminate |
| **AnalystAgent** | ToolCallAgent | 交叉验证、可信度评估、矛盾识别 | PythonExecute, Terminate |
| **SynthesizerAgent** | ToolCallAgent | 整合分析结果、构建逻辑框架 | Terminate |
| **WriterAgent** | ToolCallAgent | 撰写结构化报告、引用管理 | CitationFormatter, Terminate |
| **CriticAgent** | ToolCallAgent | 6 维度评分、缺口识别、accept/revise | Terminate |

### 2.3 研究流程（多轮迭代）

```
轮次 1（广度）：Planner → Searcher → Analyst → Synthesizer → Writer → Critic
    │
    ├── score >= threshold → Formatter → 最终报告
    │
    └── score < threshold → Planner(缺口) → Searcher → ... → Critic（轮次 2）
         │
         └── ...最多 max_rounds 轮
```

参数通过 `config/research.yaml` 配置：`max_rounds`、`quality_threshold`、`max_agent_turns`、`agent_timeouts`。

---

## 3. 技术栈分层

```
┌──────────────────────────────────────────────────┐
│  src/main.py              CLI (typer)            │  入口层
├──────────────────────────────────────────────────┤
│  src/graph/               LangGraph              │  编排层
│  workflow.py              StateGraph 构建         │  Supervisor 路由
│  nodes.py                 6 个 node 实现          │  agent.run() 调用
│  state.py                 ResearchState          │  共享状态定义
├──────────────────────────────────────────────────┤
│  src/agents/              Agent 层               │
│  specialized.py           6 个专用 Agent         │  继承 ToolCallAgent
│  llm_adapter.py           LLMProvider            │  桥接 OpenManus → 原生 SDK
├──────────────────────────────────────────────────┤
│  src/llm/                 LLM 配置               │
│  config.py                AgentLLMConfig         │  LLM 配置 dataclass
├──────────────────────────────────────────────────┤
│  src/tools/               工具层                 │
│  search.py                7 个搜索工具           │  继承 BaseTool
│  analysis.py              PythonExecute          │  子进程沙箱执行
│  export.py                CitationFormatter      │  引用格式化
│  mcp_server.py            统一 MCP Server        │  FastMCP 包装（stdio）
│  mcp/manager.py           MCPManager             │  MCP Server 生命周期管理
├──────────────────────────────────────────────────┤
│  src/_framework/          Agent 框架（vendored） │  8 个模块, ~950 行
│  agent_base.py            BaseAgent              │
│  react.py                 ReActAgent             │  Think → Act 循环
│  toolcall.py              ToolCallAgent          │  工具绑定调用
│  tool_base.py             BaseTool               │
│  tool_collection.py       ToolCollection         │
│  mcp.py                   MCPClients             │  MCP 客户端
│  schema.py                Message / Memory 等    │  数据结构
│  terminate.py             Terminate              │  终止工具
├──────────────────────────────────────────────────┤
│  src/memory/              记忆系统               │
│  long_term.py             ChromaDB 长期记忆       │
├──────────────────────────────────────────────────┤
│  src/utils/               工具                   │
│  progress.py              ProgressTracker        │  Rich 实时进度显示
└──────────────────────────────────────────────────┘
```

---

## 4. Agent 框架（vendored 自 OpenManus）

`src/_framework/` 包含 8 个模块，~950 行代码，提供了完整的 ReAct Agent 运行时。

### 4.1 继承体系

```
BaseAgent (Pydantic BaseModel)
  ├── name, description, system_prompt
  ├── llm, memory, state
  ├── max_steps, current_step
  ├── run(request) → str        主循环: step() 迭代直到 FINISHED
  ├── step() → str              抽象方法
  ├── update_memory(role, content)
  └── is_stuck() / handle_stuck_state()
       │
       └── ReActAgent
             ├── think() → bool    LLM 推理 → 决定是否行动
             ├── act() → str       执行行动
             └── step() → str      think() + act()
                  │
                  └── ToolCallAgent
                        ├── available_tools: ToolCollection
                        ├── tool_choices: AUTO / REQUIRED / NONE
                        ├── think() → bool     调用 llm.ask_tool() 获取 tool_calls
                        ├── act() → str         执行 tool_calls，结果写入 memory
                        ├── execute_tool(cmd)    单个工具执行
                        ├── _handle_special_tool()  Terminate → FINISHED
                        └── cleanup()            工具资源回收
                             │
                             ├── PlannerAgent    (system_prompt + parse_plan())
                             ├── SearcherAgent
                             ├── AnalystAgent
                             ├── SynthesizerAgent
                             ├── WriterAgent
                             └── CriticAgent     (parse_review())
```

### 4.2 关键交互流程

```
ToolCallAgent.think()
  → self.llm.ask_tool(messages, system_msgs, tools, tool_choice)
    → LLMProvider.ask_tool()                       ← 多 provider 统一调用层
      → 原生 SDK client.messages.create() / client.chat.completions.create()
        → Anthropic / OpenAI / DeepSeek / Ollama   ← 各 provider 原生 API
  ← ToolCallResult (content + tool_calls)
  → self.memory.add_message(assistant_msg)
  → return bool(tool_calls)

ToolCallAgent.act()
  → for cmd in self.tool_calls:
      self.execute_tool(cmd)
        → self.available_tools.execute(name, args)
          → MCP 模式: MCPClientTool.call_tool()  ← stdio JSON-RPC → MCP Server
          → 进程内模式: tool.execute(**kwargs)    ← BaseTool
        → self.memory.add_message(tool_msg)
```

### 4.3 模块清单

| 模块 | 文件 | 内容 |
|------|------|------|
| Schema | `schema.py` | Message, Memory, AgentState, ToolCall, ToolChoice, Function, Role |
| BaseTool | `tool_base.py` | BaseTool (Pydantic ABC), ToolResult, ToolFailure |
| ToolCollection | `tool_collection.py` | 工具集合管理, to_params() → OpenAI function calling 格式, execute() 路由 |
| Terminate | `terminate.py` | 终止工具，触发 agent.state = FINISHED |
| BaseAgent | `agent_base.py` | 状态管理、stuck detection、生命周期控制 |
| ReActAgent | `react.py` | think() + act() 抽象循环 |
| ToolCallAgent | `toolcall.py` | 完整 think/act 实现、工具执行、cleanup |
| MCP | `mcp.py` | MCPClientTool、MCPClients（继承 ToolCollection，stdio JSON-RPC 通信） |

---

## 5. Agent 实现

### 5.1 所有 Agent 的共同特征

`src/agents/specialized.py` — 6 个 Agent 均继承 `ToolCallAgent`，各自拥有：

- `name` — 唯一标识（planner / searcher / analyst / synthesizer / writer / critic）
- `system_prompt` — 定义角色、行为和输出格式
- `max_steps` — ReAct 循环上限（硬限制）
- `available_tools` — 注入的 ToolCollection

### 5.2 Agent 详情

| Agent | max_steps | 输出 | 特殊方法 |
|-------|-----------|------|---------|
| **PlannerAgent** | 5 | JSON 计划 | `parse_plan()` — 从 LLM 回复提取 JSON |
| **SearcherAgent** | 20 | 搜索结果文本 | — |
| **AnalystAgent** | 10 | 分析文本 | — |
| **SynthesizerAgent** | 10 | 综合文本 | — |
| **WriterAgent** | 15 | Markdown 报告 | — |
| **CriticAgent** | 5 | JSON 评分 | `parse_review()` — 3 层 JSON 提取 |

### 5.3 System Prompt 设计

每个 Agent 的 system prompt 明确规定了其角色和输出格式：

- **Planner**: 要求输出结构化 JSON（sub_topics、outline、information_needs、approach），通过 `parse_plan()` 解析
- **Searcher**: 要求使用搜索工具，记录来源和可信度
- **Analyst**: 要求交叉验证、识别矛盾、评估可信度
- **Synthesizer**: 要求构建统一叙述、解决矛盾、识别最佳结论
- **Writer**: 要求 Markdown 格式、执行摘要、章节内容、文中引用 [1] [2]、参考文献
- **Critic**: 要求输出 6 维度 JSON 评分（0-100），通过 `parse_review()` 解析

### 5.4 CriticAgent.parse_review() 的三层 JSON 提取

由于 LLM 输出的 JSON 格式多变，`parse_review()` 使用三层降级策略：

1. **Tier 1**: 尝试从 markdown code fence（```json ... ```）中提取
2. **Tier 2**: 搜索包含 `"overall_score"` 键的裸 JSON 对象
3. **Tier 3**: 用简单正则匹配 `overall_score: <数字>` 或 `"overall_score": <数字>`
4. **兜底**: 返回 `overall_score: 0, recommendation: "revise"`

---

## 6. LLMProvider — 多 Provider 统一调用层

`src/agents/llm_adapter.py` — 封装各 Provider 的原生 SDK，对外暴露统一的 `ask_tool()` 接口。

核心职责：
- 根据 provider 创建对应的原生客户端（`AsyncAnthropic` / `AsyncOpenAI` / httpx）
- 将 OpenManus Message 转换为各 Provider 的消息格式（主要复杂度在 Anthropic：tool_use/tool_result 用 content blocks 表达，和 OpenAI 的 tool_calls 数组完全不同）
- 调用 SDK 并将响应统一转换回 OpenAI 格式的 `ToolCallResult`

### 6.1 数据流

```
OpenManus Message (role + content + tool_calls)
  → Provider 原生消息格式
    → 原生 SDK / HTTP API 调用
      → Provider 原生响应
        → ToolCallResult (content + OM_ToolCall[])
```

### 6.2 消息格式转换

| OpenManus | Anthropic | OpenAI / DeepSeek / Ollama |
|-----------|-----------|---------------------------|
| `Message(role="user")` | `{"role": "user", "content": str}` | `{"role": "user", "content": str}` |
| `Message(role="assistant")` + tool_calls | `{"role": "assistant", "content": [{"type": "tool_use", ...}]}` | `{"role": "assistant", "tool_calls": [...]}` |
| `Message(role="tool")` | `{"role": "user", "content": [{"type": "tool_result", ...}]}` | `{"role": "tool", "tool_call_id": ...}` |
| `Message(role="system")` | 独立 `system` 参数 | `{"role": "system", "content": str}` |

### 6.3 异常处理

LLM 调用失败时记录详细日志（provider、model、消息数量）后重新抛出，不再静默吞异常。

---

## 7. LLM 配置

`src/llm/config.py`

### 7.1 AgentLLMConfig

```python
@dataclass
class AgentLLMConfig:
    provider: Literal["anthropic", "openai", "deepseek", "ollama"]
    model: str
    api_key: str = ""
    base_url: str | None = None
    temperature: float = 0.5
    max_tokens: int = 4096
    top_p: float = 1.0
```

### 7.2 客户端创建

LLMProvider 在 `__init__` 中根据 provider 创建对应的原生客户端：

| Provider | 客户端 | 备注 |
|----------|--------|------|
| `anthropic` | `anthropic.AsyncAnthropic` | 原生 Anthropic SDK |
| `openai` | `openai.AsyncOpenAI` | 原生 OpenAI SDK |
| `deepseek` | `openai.AsyncOpenAI` | OpenAI 兼容，默认 base_url: `https://api.deepseek.com` |
| `ollama` | `httpx.AsyncClient` | 直接调 REST API (`POST /api/chat`) |

### 7.3 配置注入

```python
# config/agents.yaml → AgentLLMConfig → LLMProvider
# → agent.__dict__["llm"] = provider  ← 绕过 Pydantic validator
```

为何需要 `__dict__` 赋值？`BaseAgent` 的 `llm` 字段的 model_validator 会将非 `app.llm.LLM` 类型的值替换为默认 LLM 实例，直接操作 `__dict__` 绕过此检查。

---

## 8. 工具层

### 8.1 基类

所有工具继承 `src/_framework/tool_base.py` 中的 `BaseTool`（Pydantic ABC）：

```python
class BaseTool(BaseModel, ABC):
    name: str                # 唯一标识
    description: str         # 发送给 LLM
    parameters: dict | None  # JSON Schema

    async def execute(self, **kwargs) -> Any: ...
    def to_param(self) -> dict: ...  # → OpenAI function calling 格式
```

`ToolCollection` 管理工具集合，提供 `to_params()`（生成 LLM 可用的工具列表）和 `execute(name, args)`（按名路由执行）。

### 8.2 搜索工具（`src/tools/search.py`）

| 工具 | 名称 | API | 免费额度 |
|------|------|-----|---------|
| **BraveSearchTool** | `brave_search` | Brave Search API | 2,000 次/月 |
| **TavilySearchTool** | `tavily_search` | Tavily API (AI 优化) | 1,000 次/月 |
| **DuckDuckGoTool** | `duckduckgo_search` | DuckDuckGo Instant Answer | 无限额 |
| **ArxivSearchTool** | `arxiv_search` | arXiv API (学术论文) | 无限额 |
| **WikipediaSearchTool** | `wikipedia_search` | Wikipedia API | 无限额 |
| **JinaReaderTool** | `jina_reader` | Jina Reader (URL → Markdown) | 免费 |
| **WebScraperTool** | `web_scraper` | HTTP GET + HTML 清洗 | — |

### 8.3 分析工具（`src/tools/analysis.py`）

**PythonExecuteTool** (`python_execute`)

- 在子进程中执行 Python 代码，捕获 stdout/stderr/exit_code
- 工作目录：`./data/workspace/`，超时：60s（可配置）
- 代码写入临时文件 → `subprocess.run()` → 读取输出 → 清理临时文件
- 沙箱程度有限：进程隔离但无 Docker 隔离、无网络限制

### 8.4 导出工具（`src/tools/export.py`）

**CitationFormatterTool** (`citation_formatter`)

- 将 JSON 引用列表格式化为 numbered / apa / mla 格式
- 每个引用对象：`title`, `authors`, `year`, `url`, `source`

### 8.5 Terminate 工具

`src/_framework/terminate.py` — 特殊控制工具，不产生有意义的输出。当 `ToolCallAgent._handle_special_tool()` 检测到 Terminate 调用时，将 `agent.state` 设为 `FINISHED`，结束 ReAct 循环。**始终在进程内运行**，不走 MCP。

### 8.6 双模式工具架构

由 `config/research.yaml` 中的 `mcp.enabled` 控制。

**In-Process 模式**（`mcp.enabled: false`）：

```
ToolCallAgent.act()
  → ToolCollection.execute(name, args)
    → tool.execute(**kwargs)         ← 直接 Python 异步方法调用
```

**MCP 模式**（`mcp.enabled: true`）：

```
Agent 主进程                                MCP Server 子进程
══════════════════                         ══════════════════
ToolCallAgent.act()                        FastMCP (stdio)
  → MCPClients.execute(name, args)          → @mcp.tool() handler
    → MCPClientTool.call_tool()               → tool.execute(**kwargs)
      → session.call_tool(name, args)           → 返回 ToolResult
        → stdio JSON-RPC ────────────────────→
        ← stdio JSON-RPC ←────────────────────
```

> **透明性**：`MCPClients` 继承 `ToolCollection`，Agent 层无需感知工具是本地还是远程。

### 8.7 统一 MCP Server

`src/tools/mcp_server.py` — 使用 FastMCP 在 stdio 上暴露所有工具（一个 Server 包含所有 9 个工具）：

```python
mcp = FastMCP("research-tools")
# 7 个搜索工具 + python_execute + citation_formatter
```

启动方式：
```bash
uv run python -m src.tools.mcp_server
```

关键实现细节：顶部有 `load_dotenv()`，确保子进程能加载 `.env` 中的 API key。

### 8.8 MCPManager

`src/tools/mcp/manager.py` — 管理 MCP Server 子进程生命周期：

| 方法 | 说明 |
|------|------|
| `enabled` (property) | 返回配置中 MCP 是否启用 |
| `create_tool_collection(server_names, local_tools)` | 连接指定 MCP Server，收集工具，与本地工具合并为 ToolCollection |
| `disconnect_all()` | 断开所有 MCP 连接 |

### 8.9 Agent 与工具的绑定关系

#### In-Process 模式

| Agent | Brave | Tavily | DuckDuckGo | arXiv | Wikipedia | Jina | WebScraper | PythonExec | CitationFmt | Terminate |
|-------|:-----:|:------:|:----------:|:-----:|:---------:|:----:|:----------:|:----------:|:-----------:|:---------:|
| Planner | | | | | | | | | | ✓ |
| Searcher | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | | | ✓ |
| Analyst | | | | | | | | ✓ | | ✓ |
| Synthesizer | | | | | | | | | | ✓ |
| Writer | | | | | | | | | ✓ | ✓ |
| Critic | | | | | | | | | | ✓ |

#### MCP 模式

| Agent | Search MCP | 本地 PythonExec | 本地 CitationFmt | 本地 Terminate |
|-------|:----------:|:---------------:|:----------------:|:-------------:|
| Planner | | | | ✓ |
| Searcher | ✓ | | | ✓ |
| Analyst | | ✓ | | ✓ |
| Synthesizer | | | | ✓ |
| Writer | | | ✓ | ✓ |
| Critic | | | | ✓ |

> Searcher 是唯一需要外部数据源的工具密集 Agent。Analyst 有 Python 执行工具用于数据计算。Writer 有引用格式化工具。其余 Agent 为纯 LLM 推理。

---

## 9. LangGraph 工作流

### 9.1 ResearchState（`src/graph/state.py`）

`TypedDict`（`total=False`）表示节点间共享状态：

| 类别 | 字段 | 类型 |
|------|------|------|
| 输入 | `topic`, `language`, `search_sources` | str / str / list[str] |
| Planner 输出 | `outline`, `search_queries`, `information_needs` | list[dict] / list[str] / list[str] |
| Searcher 输出 | `search_results` | Annotated[list[dict], operator.add] |
| Analyst 输出 | `analyses` | Annotated[list[dict], operator.add] |
| Synthesizer 输出 | `synthesized_findings` | str |
| Writer 输出 | `draft_report`, `final_report`, `citations` | str / str / list[dict] |
| Critic 输出 | `quality_score`, `critique`, `gaps` | float / dict / list[str] |
| 流程控制 | `current_phase`, `research_round`, `max_rounds`, `quality_threshold`, `overall_score` | — |
| 消息 | `messages` | Annotated[list, add_messages] |

`operator.add` reducer 用于 `search_results` 和 `analyses`，支持并行节点结果自动追加合并。

### 9.2 节点实现（`src/graph/nodes.py`）

每个 node 函数签名：`async def xxx_node(state: ResearchState, agent: XxxAgent, progress=None, timeout=600) -> dict`

通过 `_run_agent_with_progress()` 统一管理 Agent 执行：

1. `_reset_agent(agent)` — state → IDLE, memory 清空, current_step → 0
2. `progress.agent_started()` — 显示进度
3. 后台 asyncio 轮询 — 每 500ms 采样 `agent.current_step` 和 `agent.state`
4. `asyncio.timeout(timeout)` — 超时保护
5. `agent.run(request)` — OpenManus think→act 循环，直到 FINISHED
6. `progress.agent_finished()` / `agent_timeout()` / `agent_error()`

Agent 执行结果从 `agent.messages` 中最后一条 assistant 回复提取。

#### 各节点特点

| 节点 | 请求内容 | 超时 | 特殊行为 |
|------|---------|------|---------|
| `planner_node` | 轮次 1: topic + "create plan"。后续轮次: topic + gaps + "focus on gaps" | 300s | 调用 `planner.parse_plan()` 提取 JSON |
| `searcher_node` | 逐 query 执行（最多 10 个），每个 query 独立 agent run | 600s | 串行搜索，每次重置 agent 上下文 |
| `analyst_node` | 所有搜索结果拼接（截断 30k chars）+ analysis 指令 | 600s | — |
| `synthesizer_node` | outline (JSON) + 所有分析（截断 30k chars）+ synthesize 指令 | 600s | — |
| `writer_node` | outline + synthesized findings（截断 20k chars）+ writing 指令 | 900s | — |
| `critic_node` | draft report（截断 20k chars）+ review 指令 | 300s | 调用 `critic.parse_review()` 提取评分 |

**Searcher 特殊处理**：与其它 Agent 不同，Searcher 对每个 query 串行执行一次完整的 agent run（含重置），确保每个搜索有干净的上下文。整体超时 600s 覆盖全部搜索。

### 9.3 工作流构建（`src/graph/workflow.py`）

`build_workflow()` 组装完整流水线：

1. **Agent 实例化**：对每个 Agent 调用 `_create_agent()`：
   - 构造 agent 并注入 `ToolCollection`
   - 创建 `LLMProvider` 并注入 `agent.__dict__["llm"]`
   - 从配置覆写 `max_steps`

2. **图构建**：
   ```
   planner → searcher → analyst → synthesizer → writer → critic
                                                              │
        ┌─────────────────────────────────────────────────────┘
        │ quality < threshold && round <= max_rounds → planner
        │ quality >= threshold || round > max_rounds → formatter → END
        └──────────────────────────────────────────────────────→ END
   ```

3. **条件路由** `_supervisor_router()`（纯函数，不修改 state）：
   ```python
   def _supervisor_router(state) -> Literal["planner", "formatter", "end"]:
       if score >= threshold or current_round > max_rounds:
           return "formatter"
       return "planner"
   ```

   > **关键设计决策**：`research_round` 的递增在 `critic_node` 的返回 dict 中完成（Node 的返回值会合并到 state），而不在 conditional edge 函数中进行。这是因为 LangGraph 的 conditional edge 函数不是 Node，在其中修改 state 会被**静默丢弃**。

4. **Formatter** `_formatter_node()`：为 draft report 添加标题头（时间戳、轮次、质量分数），输出 `final_report`。

5. **Checkpointer**（可选）：`use_checkpointer=True` 时使用 SQLite 持久化状态，支持崩溃恢复。

---

## 10. 记忆系统

### 10.1 短期记忆

由 `src/_framework/schema.py` 中的 `Memory` 类管理：
- 存储 `list[Message]`，容量上限 `max_messages`（默认 50）
- FIFO 淘汰：超限时淘汰最旧的非 system 消息（system 消息保留）
- Token 估算：`len(content) // 4`
- 每个 Agent 独立记忆，每轮 run 前通过 `_reset_agent()` 清空

### 10.2 长期记忆

`src/memory/long_term.py` — ChromaDB 持久化向量存储：

| 特性 | 实现 |
|------|------|
| 存储后端 | ChromaDB PersistentClient，余弦相似度 |
| 分块策略 | 按句子边界分割（`. `），每块 ~2000 字符 |
| Entry ID | `MD5(topic + timestamp)` |
| 写入 | `add(topic, content, metadata)` — 自动分块 + 元数据 |
| 检索 | `query(topic, n_results)` — 语义搜索，返回 content + metadata + distance |
| 删除 | `delete_topic(topic)` — 按 topic 过滤删除 |

通过 `config/research.yaml` 中的 `use_long_term_memory: true` 启用。

---

## 11. 进度追踪

`src/utils/progress.py` — `ProgressTracker` 使用 Rich `Live` + `Table` 实时展示 6 个 Agent 的执行状态：

- **状态图标**: ○ pending, ◎ running (黄), ● done (绿), ✕ error (红), ⏱ timeout (红)
- **显示列**: Agent, Status, Steps (current/max), Elapsed, Detail
- **刷新率**: 4 Hz
- **使用模式**: Context manager (`with tracker:`)，进入时启动 Live，退出时渲染最终汇总表
- **API**: `agent_started()`, `agent_step_update()`, `agent_finished()`, `agent_timeout()`, `agent_error()`

---

## 12. CLI 入口

`src/main.py` — 基于 Typer 的命令行入口：

```bash
uv run python -m src.main "Research topic"
uv run python -m src.main "研究主题" --config config/research.yaml --agents config/agents.yaml -v
```

### 执行流程

1. 加载 `config/agents.yaml` 和 `config/research.yaml`
2. `build_agent_configs()` — 解析为 `AgentLLMConfig` 字典，读取环境变量中的 API key
3. `build_tool_collections()` — In-Process 模式直接创建工具实例；MCP 模式通过 MCPManager 连接子进程
4. 创建 `ProgressTracker`
5. `build_workflow()` — 实例化 6 个 Agent（含 LLMProvider + ToolCollection），编译 StateGraph
6. 构建 `initial_state`（从配置读取 topic, language, max_rounds 等）
7. `workflow.ainvoke(initial_state)` — 执行流水线
8. 保存报告到 `output_dir`，文件名自动生成（topic + 时间戳）
9. 终端显示报告预览（前 1000 字符）

### 配置文件

**`config/agents.yaml`** — 每个 Agent 独立 LLM 配置（provider, model, temperature, max_tokens），有 `default` 段和 per-agent 覆盖。

**`config/research.yaml`** — 流水线配置：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `max_rounds` | 最大研究轮次 | 2 |
| `quality_threshold` | 质量通过阈值 (0-100) | 50 |
| `search_sources` | 搜索源列表 | brave, tavily, duckduckgo, arxiv, wikipedia |
| `max_agent_turns` | 每个 Agent 最大 ReAct 步数 | 视 Agent 而定 |
| `agent_timeouts` | 每个 Agent 墙钟超时（秒） | 300-900 |
| `language` | 报告语言 (zh/en) | zh |
| `output_dir` | 输出目录 | ./reports |
| `mcp.enabled` | MCP 模式开关 | true |
| `mcp.servers` | MCP Server 配置 | — |
| `use_checkpointer` | SQLite 状态持久化 | false |
| `use_long_term_memory` | ChromaDB 长期记忆 | false |

---

## 13. 支持的 LLM Provider

| Provider | 环境变量 | 原生 SDK | 备注 |
|----------|---------|---------|------|
| anthropic | `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL` | `AsyncAnthropic` | Claude 系列 |
| openai | `OPENAI_API_KEY`, `OPENAI_BASE_URL` | `AsyncOpenAI` | GPT 及兼容 API |
| deepseek | `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL` | `AsyncOpenAI` | OpenAI 兼容，默认 base_url |
| ollama | `OLLAMA_BASE_URL` | `httpx` → REST API | 本地模型 |

---

## 14. 扩展指南

### 14.1 添加新工具（In-Process）

1. 在对应 `src/tools/*.py` 中创建类，继承 `BaseTool`
2. 定义 `name`、`description`、`parameters`（JSON Schema）
3. 实现 `async def execute(self, **kwargs) -> ToolResult`
4. 在 `src/tools/__init__.py` 中导出
5. 在 `src/main.py` 的 `build_tool_collections()` 中注册到对应 Agent

### 14.2 添加新工具（含 MCP 支持）

1. 完成 In-Process 模式的 5 步
2. 在 `src/tools/mcp_server.py` 中添加 FastMCP 工具函数：
```python
@mcp.tool()
async def my_tool(param1: str) -> str:
    result = await _my_tool_instance.execute(param1=param1)
    return result.output if result.output is not None else f"Error: {result.error}"
```

### 14.3 添加外部 MCP Server

在 `config/research.yaml` 中声明：

```yaml
mcp:
  enabled: true
  servers:
    search:
      command: uv
      args: ["run", "python", "-m", "src.tools.mcp_server"]
    github:
      command: npx
      args: ["-y", "@anthropic/mcp-server-github"]
```

然后在 `src/main.py` 的 `build_tool_collections()` 中绑定到 Agent：

```python
"searcher": await mcp_manager.create_tool_collection(
    ["search", "github"], [terminate]
),
```

### 14.4 添加新的 LLM Provider

在 `src/agents/llm_adapter.py` 的 `LLMProvider._create_client()` 中添加一个分支：

```python
if config.provider == "new_provider":
    import new_provider_sdk
    return new_provider_sdk.AsyncClient(...)
```

`AgentLLMConfig.provider` 的类型声明也需要扩展。

---

## 15. 已知限制

- Searcher 当前串行执行多 query（LangGraph Send API 并行调用预留但未实现）
- 报告仅支持 Markdown 格式（PDF/Word/HTML 待扩展）
- 无 Token 计数和成本统计
- 长期记忆未深度集成到工作流中
- 引用提取依赖 LLM 自行解析，未做结构化处理
- `PythonExecuteTool` 无 Docker 隔离，不适用于执行不可信代码
- `WebScraperTool` 不执行 JavaScript，对动态渲染页面无效
- MCP 模式默认启用的前提下，需确保 `uv` 和 `.env` 中 API key 可用
- Agent 超时采用 `asyncio.timeout` 合作式取消，极端情况下 LLM HTTP 请求可能需等 TCP 层超时

---

## 16. 项目结构

```
muti_agent/
├── src/
│   ├── _framework/          # Vendored OpenManus 核心（8 模块, ~950 行）
│   │   ├── __init__.py      # 统一导出
│   │   ├── schema.py        # Message, Memory, AgentState
│   │   ├── tool_base.py     # BaseTool, ToolResult, ToolFailure
│   │   ├── tool_collection.py # ToolCollection
│   │   ├── terminate.py     # Terminate 工具
│   │   ├── agent_base.py    # BaseAgent
│   │   ├── react.py         # ReActAgent
│   │   ├── toolcall.py      # ToolCallAgent
│   │   └── mcp.py           # MCPClients, MCPClientTool
│   ├── agents/
│   │   ├── specialized.py   # 6 个研究 Agent
│   │   └── llm_adapter.py   # LLMProvider 桥接层
│   ├── graph/
│   │   ├── state.py         # ResearchState TypedDict
│   │   ├── nodes.py         # 6 个节点实现 + _run_agent_with_progress
│   │   └── workflow.py      # build_workflow + _supervisor_router
│   ├── tools/
│   │   ├── __init__.py      # 工具统一导出
│   │   ├── base.py          # BaseTool, ToolCollection, ToolResult 重导出
│   │   ├── search.py        # 7 个搜索工具
│   │   ├── analysis.py      # PythonExecuteTool
│   │   ├── export.py        # CitationFormatterTool
│   │   ├── mcp_server.py    # 统一 MCP Server（FastMCP, stdio）
│   │   └── mcp/
│   │       ├── __init__.py
│   │       └── manager.py   # MCPManager
│   ├── llm/
│   │   └── config.py        # AgentLLMConfig
│   ├── memory/
│   │   └── long_term.py     # ChromaDB 长期记忆
│   ├── utils/
│   │   └── progress.py      # Rich 实时进度显示
│   └── main.py              # CLI 入口 (typer)
├── config/
│   ├── agents.yaml          # Agent LLM 配置
│   └── research.yaml        # 研究流程配置
├── docs/
│   └── architecture.md      # 本文档
├── reports/                 # 输出报告
├── data/                    # 运行时数据（checkpoints, chroma, workspace）
├── pyproject.toml
├── .env.example
└── README.md
```

---

## 附录 A: 变更记录

### [0.5.0] — 2026-06-08

**Changed**
- 工具层从 subpackage 重构为扁平文件：`src/tools/search/tools.py` → `src/tools/search.py`，analysis 和 export 同理
- MCP Server 从 3 个独立文件合并为统一的 `src/tools/mcp_server.py`
- `src/tools/base.py` 简化为纯重导出（移除 `ToolRegistry`）

**Removed**
- `ToolRegistry` 类（`src/tools/base.py`）— 从未被使用
- `ReportSaverTool`（`src/tools/export.py`）— main.py 自行保存报告，无需 LLM 调用工具

**Added**
- `BraveSearchTool`、`TavilySearchTool`、`JinaReaderTool` 整合到 Searcher 工具集
- `PythonExecuteTool` 分配给 Analyst Agent（In-Process 和 MCP 模式）
- `load_dotenv()` 到 `mcp_server.py` — 修复子进程无法加载 .env 的问题

**Fixed**
- `LLMProvider.ask_tool()` 异常不再静默吞（改为记录日志后 re-raise）
- Searcher 节点重复的超时/异常处理合并到 `_run_agent_with_progress()`
- MCP 模式 Writer 获得错误的 search 工具 → 改为只获得 CitationFormatterTool
- `_supervisor_router` 中 state mutation 被 LangGraph 静默丢弃的 bug → `research_round` 递增移到 `critic_node` 返回值
- `CriticAgent.parse_review()` JSON 解析从单一脆弱正则改为 3 层降级策略

### [0.4.0] — 2026-06-05

**Changed**
- OpenManus 代码 Vendoring：不再依赖外部子模块，核心框架（~950 行）内联到 `src/_framework/`
- 全项目导入路径从 `from app.*` 更新为 `from src._framework`
- 依赖精简：移除 121 个传递依赖包（browser-use, playwright, boto3 等）

**Removed**
- `src/_bootstrap.py`（MockFinder, _MockModule）
- `.gitmodules`, `OpenManus_origin/` 子模块
- 所有 `sys.path` 操作

### [0.3.0] — 2026-06-02

**Added**
- MCP Server 工具架构（双模式：In-Process + MCP）
- `MCPManager` 管理 MCP Server 子进程生命周期
- `ProgressTracker` 实时进度显示（Rich Live + Table）
- Agent 超时保护（`asyncio.timeout` + `agent_timeouts` 配置）
- `max_agent_turns` 从无效配置改为实际生效

### [0.2.0] — 2026-06-01

**Changed**
- 引入 OpenManus 作为 Agent 基座（替换自实现）
- 新增 `LLMProvider` 桥接层，支持 Anthropic / OpenAI / DeepSeek / Ollama
- 6 个 Agent 改为继承 `ToolCallAgent`

### [0.1.0] — 2026-06-01

项目初始化：Agent 基础架构、6 个专用 Agent、工具层、LangGraph 编排、配置系统、CLI 入口、记忆系统。

---

## 附录 B: 设计决策

| 决策 | 原因 |
|------|------|
| LangGraph StateGraph 而非自建流程引擎 | 提供 checkpointing、条件路由、并行 Send API |
| Supervisor-Worker 拓扑而非 Peer-to-Peer | 研究任务适合线性+回退流程 |
| 每个 Agent 独立 LLM 配置 | 不同任务适合不同模型，降低成本 |
| Vedoring OpenManus 而非外部依赖 | 实际使用 ~950 行代码却引入 264 个包 |
| `agent.__dict__["llm"]` 赋值 | 绕过 Pydantic model_validator 的类型替换 |
| Critic 评分 3 层 JSON 解析 | LLM 输出格式多变，需要鲁棒的提取策略 |
| MCP 双模式架构 | 进程内用于开发调试，MCP 模式用于生产隔离 |
| `research_round` 在 node 中递增 | LangGraph conditional edge 不修改 state（会被丢弃） |

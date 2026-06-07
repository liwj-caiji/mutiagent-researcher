# 架构设计文档

> 多智能体研究助手 — LangGraph 编排 + vendored Agent 框架 + MCP 工具服务器

## 1. 项目概述

多 Agent 协作研究助手。用户输入研究主题，6 个专用 Agent 在 LangGraph 编排下协作完成：

**主题分解 → 多源搜索 → 深度分析 → 综合归纳 → 报告撰写 → 质量审查**，最终输出结构化 Markdown 研究报告。

### 核心设计原则

- **关注点分离**：LangGraph 负责编排（何时执行谁），Agent 框架负责 Agent 内部循环（如何思考与行动），MCP Server 负责工具执行（隔离进程）
- **模型可配置**：每个 Agent 通过 `config/agents.yaml` 独立配置 provider / model / temperature
- **质量闭环**：Critic 评分 → 低于阈值自动重试（最多 3 轮）
- **轻量 vendoring**：Agent 基座代码（~950 行）内联在 `src/_framework/`，不依赖外部 OpenManus 仓库
- **工具隔离**：支持进程内直接调用或 MCP Server 子进程远程调用，通过配置切换

## 2. 系统架构

### 2.1 拓扑结构

```
用户输入研究主题
     │
     ▼
┌──────────────────────────────────────────────────────┐
│                  LangGraph StateGraph                │
│                                                      │
│   ┌──────────┐                                       │
│   │ Supervisor│ ← 条件路由（quality vs threshold）    │
│   └────┬─────┘                                       │
│        │                                              │
│   ┌────┴────────────────────────────┐                │
│   │     ResearchState (TypedDict)    │ 共享状态       │
│   └────┬────────────────────────────┘                │
│        │                                              │
│   Planner → Searcher → Analyst → Synthesizer → Writer → Critic
│      ↑                                                      │
│      └────── score < threshold && rounds remain ────────────┘
│                                          │                   │
│                                     score ok                │
│                                          ▼                   │
│                                     Formatter → END         │
└──────────────────────────────────────────────────────┘
     │
     ▼
  ./reports/report-*.md
```

> **渲染提示**：在 HTML 视图中，流程图应使用深色背景（`#1a1a2e`）配浅色文字（`#e0e0e0`），或浅色背景（`#f8f9fa`）配深色文字（`#212529`），确保对比度 ≥ 4.5:1。

### 2.2 Agent 角色

| Agent | 继承自 | 职责 | 工具 |
|-------|--------|------|------|
| **PlannerAgent** | ToolCallAgent | 分解主题 → 生成搜索查询 → 制定大纲 | Terminate |
| **SearcherAgent** | ToolCallAgent | 多源搜索（Web/arXiv/Wikipedia）→ 内容提取 | WebSearch, ArxivSearch, WikipediaSearch, WebScraper, Terminate |
| **AnalystAgent** | ToolCallAgent | 交叉验证 → 可信度评估 → 矛盾识别 | Terminate |
| **SynthesizerAgent** | ToolCallAgent | 整合分析结果 → 构建逻辑框架 → 统一结论 | Terminate |
| **WriterAgent** | ToolCallAgent | 撰写结构化报告 → 引用管理 | CitationFormatter, Terminate |
| **CriticAgent** | ToolCallAgent | 6 维度评分 → 缺口识别 → accept/revise | Terminate |

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

参数通过 `config/research.yaml` 配置：`max_rounds`，`quality_threshold`，`max_agent_turns`，`agent_timeouts`。

执行过程中，`ProgressTracker` 通过 Rich `Live` 实时展示每个 Agent 的状态（Pending → Running → Done/Error/Timeout）、步数进度和耗时。每个 Agent 的 `run()` 调用由 `asyncio.timeout` 保护，超时后自动终止。

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
│  specialized.py           6 个专用 Agent         │  继承 OpenManus ToolCallAgent
│  llm_adapter.py           LLMAdapter             │  桥接 OpenManus → LangChain
├──────────────────────────────────────────────────┤
│  src/llm/                 LLM 配置               │
│  config.py                AgentLLMConfig         │  create_llm() 工厂
├──────────────────────────────────────────────────┤
│  src/tools/               工具层                 │
│  search/tools.py          WebSearch 等           │  继承 src._framework BaseTool
│  search/mcp_server.py     Search MCP Server      │  FastMCP 包装（stdio）
│  analysis/tools.py        PythonExecute          │  子进程沙箱执行
│  analysis/mcp_server.py   Analysis MCP Server    │  FastMCP 包装（stdio）
│  export/tools.py          CitationFormatter 等    │  引用格式化 + 报告保存
│  export/mcp_server.py     Export MCP Server      │  FastMCP 包装（stdio）
│  mcp/manager.py           MCPManager             │  MCP Server 生命周期管理
├──────────────────────────────────────────────────┤
│  src/_framework/          Agent 框架（vendored） │
│  agent_base.py            BaseAgent              │
│  react.py                 ReActAgent             │  Think → Act 循环
│  toolcall.py              ToolCallAgent          │  工具绑定调用
│  tool_base.py             BaseTool               │
│  tool_collection.py       ToolCollection         │
│  mcp.py                   MCPClients             │  MCP 客户端（继承 ToolCollection）
│  schema.py                Message / Memory 等    │  数据结构
│  terminate.py             Terminate              │  终止工具
├──────────────────────────────────────────────────┤
│  src/memory/              记忆系统               │
│  memory.py                短期记忆               │
│  long_term.py             ChromaDB 长期          │
└──────────────────────────────────────────────────┘
```

## 4. Agent 继承体系（vendored 自 OpenManus）

```
src._framework.agent_base.BaseAgent        (Pydantic BaseModel)
  ├── name, description, system_prompt
  ├── llm, memory, state
  ├── max_steps, current_step
  ├── run(request) → str        主循环: step() 迭代直到 FINISHED 或 max_steps
  ├── step() → str              抽象方法
  ├── update_memory(role, content)
  └── is_stuck() / handle_stuck_state()
       │
       └── src._framework.react.ReActAgent
             ├── think() → bool    LLM 推理 → 决定是否行动
             ├── act() → str       执行行动
             └── step() → str      think() + act()
                  │
                  └── src._framework.toolcall.ToolCallAgent
                        ├── available_tools: ToolCollection
                        ├── tool_choices: AUTO / REQUIRED / NONE
                        ├── think() → bool     调用 llm.ask_tool() 获取 tool_calls
                        ├── act() → str         执行 tool_calls，结果写入 memory
                        ├── execute_tool(cmd)    单个工具执行
                        ├── _handle_special_tool()  Terminate → FINISHED
                        └── cleanup()            工具资源回收
                             │
                             ├── PlannerAgent    (system_prompt + parse_plan())
                             ├── SearcherAgent   (搜索工具集)
                             ├── AnalystAgent
                             ├── SynthesizerAgent
                             ├── WriterAgent
                             └── CriticAgent     (parse_review())
```

### 关键交互

```
ToolCallAgent.think()
  → self.llm.ask_tool(messages, system_msgs, tools, tool_choice)
    → LLMAdapter.ask_tool()                       ← 我们的桥接
      → LangChain ChatModel.bind_tools().ainvoke()
        → Anthropic / OpenAI / DeepSeek / Ollama   ← 多 provider
  ← ChatCompletionMessage (content + tool_calls)
  → self.memory.add_message(assistant_msg)
  → return bool(tool_calls)

ToolCallAgent.act()
  → for cmd in self.tool_calls:
      self.execute_tool(cmd)
        → self.available_tools.execute(name, args)
          → MCP 模式: MCPClientTool.call_tool()  ← stdio JSON-RPC → MCP Server
          → 进程内模式: tool.execute(**kwargs)    ← OpenManus BaseTool
        → self.memory.add_message(tool_msg)
```

## 5. LLMAdapter — 多 Provider 桥接

### 设计动机

OpenManus 的 `ToolCallAgent.think()` 调用 `self.llm.ask_tool()`，该接口返回 OpenAI 格式的 `ChatCompletionMessage`。

`LLMAdapter` 包装 LangChain ChatModel，对外提供相同的 `ask_tool()` 签名，内部使用 LangChain 原生支持 Anthropic / OpenAI / DeepSeek / Ollama。

### 接口

```python
class LLMAdapter:
    async def ask_tool(
        self,
        messages: list[OM_Message],       # OpenManus Message 格式
        system_msgs: list[OM_Message] | None,
        timeout: int = 300,
        tools: list[dict] | None = None,  # OpenAI function calling 格式
        tool_choice: Any = None,           # ToolChoice.AUTO / REQUIRED / NONE
        temperature: float | None = None,
    ) -> ToolCallResult | None:            # 模拟 ChatCompletionMessage
        ...
```

### 数据流

```
OpenManus Message (role + content + tool_calls)
  → LangChain HumanMessage / AIMessage / SystemMessage / ToolMessage
    → ChatModel.bind_tools(tools).ainvoke()
      → LangChain AIMessage (content + tool_calls)
        → ToolCallResult (content + OM_ToolCall[])
```

### AgentConfig → LLMAdapter 流程

```
config/agents.yaml
  → AgentLLMConfig(provider="anthropic", model="claude-sonnet-4-6", ...)
    → create_llm(config) → ChatAnthropic / ChatOpenAI / ChatOllama
      → LLMAdapter(llm_config)
        → agent.__dict__["llm"] = adapter   ← 注入（绕过 Pydantic validator）
```

为何需要 `agent.__dict__["llm"]` 赋值？`BaseAgent` 的 `llm` 字段声明为 `Any = Field(default=None)`，通过 `__dict__` 直接赋值以绕过 Pydantic 的类型检查和 validator 逻辑。

## 6. 工具层 — 双模式架构

### 6.1 双模式支持

工具层支持两种运行模式，由 `config/research.yaml` 中的 `mcp.enabled` 控制。

**模式 1 — 进程内调用**（默认，`mcp.enabled: false`）：

```
Agent 进程
  ToolCallAgent.act()
    → ToolCollection.execute(name, args)
      → tool.execute(**kwargs)         ← 直接 Python 异步方法调用
```

**模式 2 — MCP Server 远程调用**（`mcp.enabled: true`）：

```
Agent 进程                                    MCP Server 子进程
══════════════════                         ══════════════════
ToolCallAgent.act()                        FastMCP (stdio transport)
  → MCPClients.execute(name, args)          → @mcp.tool() handler
    → MCPClientTool.call_tool()               → tool.execute(**kwargs)
      → session.call_tool(name, args)           → 返回结果
        → stdio JSON-RPC ────────────────────→
        ← stdio JSON-RPC ←────────────────────
```

> **关键设计**：`MCPClients` 继承 `ToolCollection`，对 Agent 层完全透明。Agent 不需要知道工具是本地还是远程的。

### 6.2 MCP Server 架构

```
                      ┌──────────────────────┐
                      │     MCPManager        │
                      │  (src/tools/mcp/)     │
                      │                      │
                      │  create_tool_         │
                      │  collection()         │
                      │  disconnect_all()     │
                      └──────┬───────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────┐ ┌──────────────┐ ┌─────────────────┐
│ Search MCP       │ │ Analysis MCP │ │ Export MCP      │
│ Server (stdio)   │ │ Server(stdio)│ │ Server (stdio)  │
│                 │ │              │ │                 │
│ FastMCP("search")│ │FastMCP("ana")│ │FastMCP("export")│
│                 │ │              │ │                 │
│ web_search      │ │ python_exec  │ │ citation_fmt    │
│ arxiv_search    │ │              │ │ report_saver    │
│ wikipedia_search│ │              │ │                 │
│ web_scraper     │ │              │ │                 │
└─────────────────┘ └──────────────┘ └─────────────────┘
```

### 6.3 MCP 配置

`config/research.yaml` 中的 MCP 配置段：

```yaml
mcp:
  enabled: false          # 设为 true 启用 MCP 模式
  transport: stdio        # 传输方式：stdio / sse
  servers:
    search:               # 搜索工具服务器
      command: uv
      args: ["run", "python", "-m", "src.tools.search.mcp_server"]
    analysis:             # 分析工具服务器
      command: uv
      args: ["run", "python", "-m", "src.tools.analysis.mcp_server"]
    export:               # 导出工具服务器
      command: uv
      args: ["run", "python", "-m", "src.tools.export.mcp_server"]
```

### 6.4 Agent 与 MCP Server 绑定

| Agent | Search MCP | Analysis MCP | Export MCP | 本地 Terminate |
|-------|:----------:|:------------:|:----------:|:-------------:|
| Planner | | | | ✓ |
| Searcher | ✓ | | | ✓ |
| Analyst | | | | ✓ |
| Synthesizer | | | | ✓ |
| Writer | | | ✓ | ✓ |
| Critic | | | | ✓ |

> `Terminate` 保持本地运行，因为它是控制信号工具而非外部数据工具。

### 6.5 BaseTool 接口

```python
class BaseTool(BaseModel, ABC):
    name: str
    description: str
    parameters: Optional[dict]   # JSON Schema

    async def execute(self, **kwargs) -> Any: ...   # 子类实现
    def to_param(self) -> dict: ...                 # → OpenAI function calling 格式
```

### 6.6 已实现工具

| 工具 | 文件 | MCP Server | 用途 |
|------|------|:----------:|------|
| `WebSearchTool` | `src/tools/search/tools.py` | search | DuckDuckGo 网络搜索 |
| `ArxivSearchTool` | `src/tools/search/tools.py` | search | arXiv 学术论文搜索 |
| `WikipediaSearchTool` | `src/tools/search/tools.py` | search | Wikipedia 百科查询 |
| `WebScraperTool` | `src/tools/search/tools.py` | search | URL 内容提取 |
| `PythonExecuteTool` | `src/tools/analysis/tools.py` | analysis | Python 代码执行（子进程沙箱） |
| `CitationFormatterTool` | `src/tools/export/tools.py` | export | 引用格式化（numbered/APA/MLA） |
| `ReportSaverTool` | `src/tools/export/tools.py` | export | 报告保存到文件 |
| `Terminate` | `src/_framework/terminate.py` | — | 标记 Agent 任务完成，触发 FINISHED 状态 |

### 6.7 扩展新工具（同时支持双模式）

1. 在对应 `src/tools/*/tools.py` 中创建类，继承 `src._framework.BaseTool`
2. 在对应的 MCP Server 文件（`mcp_server.py`）中添加 FastMCP 工具函数
3. 在 `src/main.py` 的 `build_tool_collections()` 中注册到对应 Agent 的工具集

## 7. LangGraph 工作流

### 7.1 ResearchState（`src/graph/state.py`）

```python
class ResearchState(TypedDict, total=False):
    topic: str
    language: str                              # zh / en
    outline: list[dict]
    search_queries: list[str]
    search_results: Annotated[list[dict], operator.add]  # 并行追加
    analyses: Annotated[list[dict], operator.add]
    synthesized_findings: str
    draft_report: str
    final_report: str
    citations: list[dict]
    quality_score: float
    critique: dict
    gaps: list[str]
    current_phase: str
    research_round: int
    max_rounds: int
    quality_threshold: float
```

### 7.2 节点实现（`src/graph/nodes.py`）

每个 node 函数签名：`async def xxx_node(state: ResearchState, agent: XxxAgent, progress=None, timeout=600) -> dict`

通过 `_run_agent_with_progress()` 辅助函数统一管理 Agent 执行：

```python
# _run_agent_with_progress 封装了：
#   1. _reset_agent(agent)          — state → IDLE, memory 清空
#   2. progress.agent_started()     — 开始显示该 Agent 进度
#   3. 后台 asyncio 轮询任务         — 每 500ms 采样 agent.current_step / agent.state
#   4. asyncio.timeout(timeout)     — 超时保护，防止 LLM 调用挂起
#   5. agent.run(request)           — OpenManus think→act 循环，直到 FINISHED
#   6. progress.agent_finished() / agent_timeout() / agent_error()
```

Agent 执行结果从 `agent.messages` 提取 assistant 最后一条回复。

### 7.3 条件路由（`src/graph/workflow.py`）

```python
def _supervisor_router(state) -> Literal["planner", "formatter", "end"]:
    if score >= threshold or current_round >= max_rounds:
        return "formatter"   # 质量合格或已达最大轮次
    return "planner"         # 重试下一轮
```

### 7.4 Checkpointer（可选）

```python
# config/research.yaml 中启用
use_checkpointer: true
checkpointer_path: ./data/checkpoints.sqlite
```

## 8. 记忆系统

### 短期记忆

- 由 `src/_framework/schema.py` 中的 `Memory` 类管理
- 每个 Agent 独立记忆，存储 Message 列表
- 每轮 run 前通过 `_reset_agent()` 清空

### 长期记忆（可选）

`src/memory/long_term.py` — ChromaDB 持久化：
- `add(topic, content)` — 存储研究发现
- `query(topic, n_results)` — 相似度检索历史发现
- `delete_topic(topic)` — 按主题删除

在 `config/research.yaml` 中通过 `use_long_term_memory: true` 启用。

## 9. 配置系统

### agents.yaml

```yaml
default:
  provider: deepseek
  model: deepseek-v4-flash
  temperature: 0.3
  max_tokens: 4096

planner:
  provider: deepseek
  model: deepseek-v4-flash
  temperature: 0.2

searcher:
  provider: deepseek
  model: deepseek-v4-flash
  temperature: 0.1
  max_tokens: 2048

analyst:
  provider: deepseek
  model: deepseek-v4-flash
  temperature: 0.4
  max_tokens: 8192
# synthesizer, writer, critic 均使用 deepseek-v4-flash
```

### research.yaml

```yaml
max_rounds: 3
quality_threshold: 75
search_sources: [web, arxiv, wikipedia]
language: zh
output_dir: ./reports

# 每个 Agent 的最大 ReAct 步数（直接设置 agent.max_steps）
max_agent_turns:
  planner: 5
  searcher: 15
  analyst: 10
  synthesizer: 10
  writer: 15
  critic: 5

# 每个 Agent 的墙钟超时（秒）— 防止 LLM 调用挂起导致流水线卡死
agent_timeouts:
  planner: 300
  searcher: 600
  analyst: 600
  synthesizer: 600
  writer: 900
  critic: 300

# MCP 工具服务器配置
mcp:
  enabled: false
  transport: stdio
  servers:
    search:
      command: uv
      args: ["run", "python", "-m", "src.tools.search.mcp_server"]
    analysis:
      command: uv
      args: ["run", "python", "-m", "src.tools.analysis.mcp_server"]
    export:
      command: uv
      args: ["run", "python", "-m", "src.tools.export.mcp_server"]

use_checkpointer: false
use_long_term_memory: false
```

## 10. 支持的 LLM Provider

| Provider | 底层 LangChain 类 | 环境变量 |
|----------|-------------------|---------|
| anthropic | `ChatAnthropic` | `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL` |
| openai | `ChatOpenAI` | `OPENAI_API_KEY`, `OPENAI_BASE_URL` |
| deepseek | `ChatOpenAI(base_url=...)` | `DEEPSEEK_API_KEY` |
| ollama | `ChatOllama` | `OLLAMA_BASE_URL` |

通过 `src/llm/config.py` 的 `create_llm(config: AgentLLMConfig) → BaseChatModel` 创建。

## 11. 模块映射

| 模块 | 文件 | 依赖 |
|------|------|------|
| ResearchState | `src/graph/state.py` | langgraph |
| LangGraph Workflow | `src/graph/workflow.py` | langgraph, src._framework |
| Agent Nodes | `src/graph/nodes.py` | src._framework |
| PlannerAgent 等 6 个 Agent | `src/agents/specialized.py` | src._framework `ToolCallAgent` |
| LLMAdapter | `src/agents/llm_adapter.py` | LangChain, src._framework |
| AgentLLMConfig / create_llm | `src/llm/config.py` | LangChain |
| ProgressTracker | `src/utils/progress.py` | rich |
| Memory（短期） | `src/_framework/schema.py` | - |
| LongTermMemory | `src/memory/long_term.py` | chromadb |
| BaseTool | `src/_framework/tool_base.py` | - |
| ToolCollection | `src/_framework/tool_collection.py` | - |
| MCPClients / MCPClientTool | `src/_framework/mcp.py` | mcp |
| WebSearch 等搜索工具 | `src/tools/search/tools.py` | src._framework `BaseTool` |
| PythonExecute | `src/tools/analysis/tools.py` | src._framework `BaseTool` |
| CitationFormatter / ReportSaver | `src/tools/export/tools.py` | src._framework `BaseTool` |
| Search MCP Server | `src/tools/search/mcp_server.py` | FastMCP, BaseTool |
| Analysis MCP Server | `src/tools/analysis/mcp_server.py` | FastMCP, BaseTool |
| Export MCP Server | `src/tools/export/mcp_server.py` | FastMCP, BaseTool |
| MCPManager | `src/tools/mcp/manager.py` | src._framework `MCPClients` |
| agents.yaml | `config/agents.yaml` | - |
| research.yaml | `config/research.yaml` | - |
| CLI 入口 | `src/main.py` | typer, rich, LangGraph |

## 12. 已知限制

- Searcher 当前串行执行多 query（LangGraph Send API 并行预留未实现）
- 报告仅支持 Markdown 格式（PDF/Word/HTML 待扩展）
- 无 Token 计数和成本统计
- 长期记忆未深度集成到工作流中
- 引用提取依赖 LLM 自行解析，未做结构化处理
- MCP 模式默认关闭，需手动启用并确保 `uv` 可用
- Agent 超时采用 `asyncio.timeout` 合作式取消，若 LLM HTTP 请求不响应可能需等 TCP 层超时

## 13. 变更记录

| 日期 | 变更 | 原因 |
|------|------|------|
| 2026-06-05 | OpenManus 代码 vendoring → `src/_framework/`，移除子模块及 121 个传递依赖 | 实际使用量仅 ~950 行，依赖 264 个包过于沉重 |
| 2026-06-02 | 新增 ProgressTracker 实时进度显示（Rich Live + Table） | 用户需要直观看到每个 Agent 的执行状态、步数和耗时 |
| 2026-06-02 | Agent 超时保护（`asyncio.timeout` + `agent_timeouts` 配置） | 防止 LLM 调用挂起导致流水线无限等待 |
| 2026-06-02 | `max_agent_turns` 从无效配置改为实际生效（传递到 agent.max_steps） | 之前配置存在但从未赋值给 Agent |
| 2026-06-02 | 节点函数签名更新（新增 progress / timeout 参数） | 支持进度上报和超时控制 |
| 2026-06-02 | 工具层支持 MCP Server 模式（双模式架构） | 工具进程隔离，支持独立部署和扩展 |
| 2026-06-02 | 修复 OpenManus 导入链（懒加载 BrowserUseTool 等） | 移除不必要的重依赖（browser-use, docker, boto3） |
| 2026-06-02 | 新增 `src/tools/mcp/manager.py` MCPManager | MCP Server 生命周期管理 |
| 2026-06-01 | 初始架构设计与自实现 Agent 基座 | 项目启动 |
| 2026-06-01 | Formatter 从独立 Agent 改为普通 node | 格式化不需要 LLM |
| 2026-06-01 | 搜索工具 DuckDuckGo fallback | 降低零配置门槛 |
| 2026-06-01 | 引入 FoundationAgents/OpenManus，删除自实现 Agent 基座 | 复用上游成熟实现 |
| 2026-06-01 | 新增 LLMAdapter 桥接层 | 支持非 OpenAI 的 provider |

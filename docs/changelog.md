# 变更日志

记录项目开发过程中的所有变更。

## [0.4.0] — 2026-06-05

### Changed

- **OpenManus 代码 Vendoring**：不再依赖 OpenManus_origin 子模块，改为将核心框架代码（~950 行）内联到 `src/_framework/`
  - `src/_framework/schema.py` — Message、Memory、AgentState、ToolCall、ToolChoice 等数据结构
  - `src/_framework/tool_base.py` — BaseTool、ToolResult、ToolFailure 基类
  - `src/_framework/terminate.py` — Terminate 工具
  - `src/_framework/tool_collection.py` — ToolCollection 工具集合
  - `src/_framework/agent_base.py` — BaseAgent（状态管理、stuck detection、生命周期）
  - `src/_framework/react.py` — ReActAgent（Think → Act 抽象循环）
  - `src/_framework/toolcall.py` — ToolCallAgent（完整 think/act 实现、工具执行、cleanup）
  - `src/_framework/mcp.py` — MCPClientTool、MCPClients（MCP 客户端）
  - `src/_framework/exceptions.py` — ToolError、TokenLimitExceeded

- **依赖大幅精简**：移除 `OpenManus_origin/` 子模块及其 121 个传递依赖包
  - 删除：browser-use、browsergym、playwright、gymnasium、boto3、crawl4ai、baidusearch、duckduckgo_search、fastapi、uvicorn、datasets、huggingface-hub 等
  - 保留：pydantic、openai、loguru、tenacity、tiktoken、structlog（框架核心依赖）

- **全项目导入路径更新**（13 个文件）：`from app.*` → `from src._framework`

### Removed

- `src/_bootstrap.py` — 不再需要 mock finder（MockFinder、_MockModule、_Dummy）
- `.gitmodules` — 不再使用 git 子模块
- `OpenManus_origin/` — 子模块目录
- 所有 `sys.path` 操作及 `import src._bootstrap` 调用（`main.py`、3 个 `mcp_server.py`）
- `src/_framework/__init__.py` 改用相对导入（`from .schema import ...`）

### Rationale

本项目的 OpenManus 实际使用量仅 ~950 行（8 个模块），却引入了 50,000+ 行代码和 160+ 个传递依赖包，包括浏览器自动化（playwright）、AWS SDK（boto3）、机器学习（torch、transformers）等重型包。Vendoring 后：
- 依赖包数量从 264 降至 143
- 不再需要 `sys.path` hack、mock finder、config.toml 适配等变通方案
- 代码所有权完全归属本项目，可自由裁剪和修改框架层

## [0.3.1] — 2026-06-02

### Added

- **实时进度显示**：新增 `src/utils/progress.py` — `ProgressTracker` 类，使用 Rich `Live` + `Table` 实时展示 6 个 Agent 的执行状态（Pending → Running → Done/Error/Timeout）、步数进度、耗时和当前操作
- **Agent 超时保护**：每个 `agent.run()` 调用由 `asyncio.timeout` 包装，防止 LLM 调用挂起导致整个流水线卡死
- **`agent_timeouts` 配置**：`config/research.yaml` 新增 `agent_timeouts` 段，每个 Agent 独立配置超时时间（300s–900s）
- **`_run_agent_with_progress()` 辅助函数**：在 `src/graph/nodes.py` 中封装 Agent 执行逻辑，统一管理进度上报、超时保护和状态重置

### Changed

- **max_agent_turns 生效**：`config/research.yaml` 中的 `max_agent_turns` 不再是无用配置——现已通过 `build_workflow()` → `_create_agent()` 传递到 Agent 的 `max_steps`，真正限制每个 Agent 的最大执行步数
- **节点函数签名更新**：`src/graph/nodes.py` 中 6 个节点函数新增 `progress` 和 `timeout` 参数
- **`build_workflow()` 签名扩展**：新增 `max_agent_turns`、`progress`、`agent_timeouts` 三个可选参数
- **Rich 进度显示替换**：`src/main.py` 中原来的 `Progress` + `SpinnerColumn` 替换为 `ProgressTracker` + `Live` 上下文管理器

### Documentation

- **`docs/architecture.md`** — 更新节点实现、配置示例、模块映射，新增进度显示和超时保护说明
- **`docs/changelog.md`** — 本文件

## [0.3.0] — 2026-06-02

### Added

- **MCP Server 工具架构**
  - 新增三种 MCP Server：Search、Analysis、Export（`src/tools/*/mcp_server.py`）
  - 使用 `mcp.server.fastmcp.FastMCP` 包装现有 BaseTool 实例，通过 stdio transport 暴露工具
  - 新增 `src/tools/mcp/manager.py` — `MCPManager` 管理 MCP Server 子进程生命周期
  - `config/research.yaml` 新增 `mcp` 配置段，支持按 Agent 绑定不同的 MCP Server
  - `src/main.py` 中 `build_tool_collections()` 支持双模式：MCP 远程调用 / 进程内直接调用
- **MCP 依赖**：`pyproject.toml` 新增 `mcp`、`fastmcp`、`structlog` 依赖
- **工具层懒加载**：`src/tools/__init__.py` 改为懒加载，避免子模块导入时触发 OpenManus 配置初始化

### Fixed

- **OpenManus 导入链优化**（懒加载重依赖模块）：
  - `app/tool/__init__.py` — 懒加载 `BrowserUseTool`、`Crawl4aiTool`、`Bash` 等重依赖工具
  - `app/agent/__init__.py` — 懒加载 `BrowserAgent`、`SWEAgent`、`MCPAgent`
  - `app/llm.py` — 懒加载 `BedrockClient`（移除 `boto3` 强制依赖）
  - `app/agent/base.py` — 懒加载 `SANDBOX_CLIENT`（移除 `docker` 强制依赖）
  - `app/config.py` — `DaytonaSettings` 在无配置时设为 `None` 而非空实例
  - `src/graph/workflow.py` — 懒加载 `SqliteSaver`（移除 `langgraph.checkpoint.sqlite` 强制依赖）
- **构建配置**：`pyproject.toml` 新增 `[tool.hatch.build.targets.wheel]` 配置，修复 `uv sync` 打包错误

### Documentation

- **`docs/architecture.md`** — 全文中文化，新增 MCP 架构章节（双模式流程、MCP Server 拓扑、配置示例、模块映射）
- **`docs/tools.md`** — 全文中文，新增 MCP 调用机制、MCP Server 架构图、工具绑定对比表、MCP 扩展指南
- **`docs/changelog.md`** — 本文件，补充 v0.3.0 变更记录

## [0.2.0] — 2026-06-01

### Changed

- **引入原始 OpenManus（FoundationAgents/OpenManus）作为 Agent 基座**
  - 新增 `OpenManus_origin/` 目录（从 GitHub 克隆）
  - 删除自实现的 `src/agents/base.py`、`react_agent.py`、`tool_call_agent.py`
  - `specialized.py` 中 6 个 Agent 改为继承 OpenManus 的 `ToolCallAgent`
  - 新增 `src/agents/llm_adapter.py` — 桥接 OpenManus LLM 接口到 LangChain 多 provider
  - 所有工具改为继承 OpenManus 的 `BaseTool`
  - `pyproject.toml` 新增 OpenManus 核心依赖（openai, tenacity, loguru）
- **LLMAdapter**：包装 LangChain ChatModel，对外暴露 OpenManus 兼容的 `ask_tool()` 接口，原生支持 Anthropic / OpenAI / DeepSeek / Ollama
- **工作流集成**：`build_workflow` 中使用 `_create_agent()` 工厂函数，绕过 OpenManus 的 Pydantic model_validator 注入 LLMAdapter

### Rationale

原始 OpenManus 提供了成熟的 ReAct + ToolCall 循环实现，包括：
- 基于 Pydantic 的 Agent 配置和状态管理
- 完善的 think()/act() 循环和工具执行
- 重复检测（stuck detection）
- Terminate 工具集成

引入后我们只需关注 Agent 的 system prompt 和工具配置，不维护 Agent 执行循环。

## [0.1.0] — 2026-06-01

### Added

- **项目初始化**：使用 uv 管理环境，创建项目骨架结构
- **LLM 配置工厂**（`src/llm/`）：支持 Anthropic、OpenAI、DeepSeek、Ollama 四种 provider，每个 Agent 可独立配置
- **Agent 基础架构**（`src/agents/`）：
  - `BaseAgent` — 状态管理、生命周期控制
  - `ReActAgent` — Think → Act → Observe 循环
  - `ToolCallAgent` — 工具调用机制、JSON Schema 解析
- **6 个专用研究 Agent**（`src/agents/specialized.py`）：
  - `PlannerAgent` — 主题分解、大纲生成、缺口分析
  - `SearcherAgent` — 多源信息检索（Web、arXiv、Wikipedia）
  - `AnalystAgent` — 深度分析、交叉验证、矛盾识别
  - `SynthesizerAgent` — 综合归纳、逻辑框架构建
  - `WriterAgent` — 结构化报告撰写
  - `CriticAgent` — 质量评分、事实核查、缺口识别
- **工具层**（`src/tools/`）：
  - 搜索工具：`WebSearchTool`（Tavily/DuckDuckGo）、`ArxivSearchTool`、`WikipediaSearchTool`、`WebScraperTool`
  - 分析工具：`PythonExecuteTool`
  - 导出工具：`CitationFormatterTool`、`ReportSaverTool`
  - `BaseTool` + `ToolRegistry` 基础框架
- **记忆系统**（`src/memory/`）：
  - `Memory` — 短期对话历史管理（max_messages 限制）
  - `LongTermMemory` — ChromaDB 持久化研究知识
- **LangGraph 编排**（`src/graph/`）：
  - `ResearchState` — TypedDict 共享状态（支持 Annotated reducers 并行）
  - 6 个节点实现：planner → searcher → analyst → synthesizer → writer → critic → formatter
  - Supervisor 条件路由：质量不通过自动重试（最多 N 轮）
  - SQLite checkpointer 支持（可选）
- **配置系统**（`config/`）：
  - `agents.yaml` — 每个 Agent 独立模型配置
  - `research.yaml` — 研究流程参数（轮次、阈值、搜索源等）
- **CLI 入口**（`src/main.py`）：
  - `uv run python -m src.main "研究主题"` 一键启动
  - `--config` / `--agents` 参数指定配置文件
  - `--verbose` 调试模式
- **文档**（`docs/`）：
  - `architecture.md` — 完整架构设计文档
  - `changelog.md` — 本文件

### 设计决策

| 决策 | 原因 |
|------|------|
| 使用 LangGraph StateGraph 而非自建流程引擎 | LangGraph 提供 checkpointing、条件路由、并行 Send API，减少造轮子 |
| Supervisor-Worker 拓扑而非 Peer-to-Peer | 研究任务适合线性+回退流程，Supervisor 模式简单可靠 |
| 每个 Agent 独立 LLM 配置 | 不同任务适合不同模型（搜索用便宜模型，分析用强模型），降低成本 |
| BaseAgent + ReActAgent + ToolCallAgent 继承体系 | 参考 OpenManus 的成熟设计，职责分层清晰 |
| 工具层不直接使用 LangChain 工具类 | 自己的 BaseTool 提供更好的可控性，通过 to_langchain_tool() 桥接 |
| ChromaDB 用作长期记忆 | 轻量嵌入方案，支持相似度检索，本地部署 |
| MCP Server 双模式架构 | 进程内用于开发调试，MCP 模式用于生产部署和工具隔离 |

### 已知限制

- 并行 Searcher 使用 Send API 的方案尚未实现（当前为串行搜索多个 query）
- 报告目前仅支持 Markdown 格式导出（PDF/Word 待扩展）
- 缺乏完善的 Token 计数和成本统计
- 长期记忆的相似度检索尚未充分集成到工作流中
- 搜索结果的引用格式提取较简单（依赖 LLM 自行解析）
- MCP 模式默认关闭，需手动启用

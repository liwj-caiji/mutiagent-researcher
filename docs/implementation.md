# Implementation Details

## Overview

The project is a **multi-agent research assistant** that orchestrates 6 specialized LLM agents through a LangGraph StateGraph pipeline. Each agent inherits from OpenManus's `ToolCallAgent`, which provides a ReAct (think → act) loop with tool execution. The pipeline follows a Supervisor pattern: Planner → Searcher → Analyst → Synthesizer → Writer → Critic, with conditional looping back to Planner when quality falls below a threshold.

---

## 1. Tools Module (`src/tools/`)

### 1.1 ToolRegistry (`base.py`)

A thin wrapper around OpenManus's `BaseTool` and `ToolCollection`. It provides a registry pattern for managing tool instances by name.

| Method | Description |
|---|---|
| `register(tool)` | Store a tool instance keyed by `tool.name` |
| `get(name)` | Retrieve a single tool by name |
| `get_collection(names)` | Build a `ToolCollection` from a list of tool names |
| `get_all()` | Return a `ToolCollection` containing every registered tool |
| `list_names()` | Return all registered tool names |

The registry pattern lets you pre-register all tools and then compose agent-specific tool sets declaratively (e.g., `registry.get_collection(["web_search", "arxiv_search"])` for the Searcher).

### 1.2 Search Tools (`search/tools.py`)

Four tools extending `BaseTool`, each with an async `execute()` method returning a `ToolResult`:

**WebSearchTool** (`web_search`)
- Invokes DuckDuckGo Instant Answer API via HTTP GET
- Parameters: `query` (str, required), `max_results` (int, default 5)
- Returns the Abstract text + up to `max_results` RelatedTopics with URLs
- No API key required; suitable as the default general-purpose search backend

**ArxivSearchTool** (`arxiv_search`)
- Uses the `arxiv` Python library to query arXiv's API
- Parameters: `query` (str, required), `max_results` (int, default 5)
- Returns paper title, authors, published date, URL (`entry_id`), and abstract for each result
- Sorted by relevance via `arxiv.SortCriterion.Relevance`

**WikipediaSearchTool** (`wikipedia_search`)
- Uses the `wikipedia` Python library
- Parameters: `query` (str, required), `language` (str, default "en")
- Searches for matching pages, retrieves up to 2 page summaries (3 sentences each) with title and URL

**WebScraperTool** (`web_scraper`)
- Fetches a URL via `httpx` with a browser User-Agent header
- Parameters: `url` (str, required)
- Strips `<script>` and `<style>` tags, removes all HTML tags, collapses whitespace, truncates to 8000 chars
- Follows redirects; 20s timeout

**MCP Server** (`search/mcp_server.py`): Exposes the same 4 tools via FastMCP on stdio transport, enabling remote tool execution. Each tool is a thin async wrapper around the corresponding tool instance's `execute()`.

### 1.3 Analysis Tools (`analysis/tools.py`)

**PythonExecuteTool** (`python_execute`)
- Executes arbitrary Python code in a sandboxed subprocess
- Parameters: `code` (str, required)
- Writes code to a temp file in `./data/workspace/`, runs via `subprocess.run()` with `capture_output=True`
- Collects stdout, stderr, and exit code; cleans up the temp file afterward
- Configurable timeout (default 60s); catches `TimeoutExpired` separately
- Used by the Analyst agent for data computation, statistical analysis, or text processing

**MCP Server** (`analysis/mcp_server.py`): Exposes `python_execute` via FastMCP.

### 1.4 Export Tools (`export/tools.py`)

**CitationFormatterTool** (`citation_formatter`)
- Formats a JSON list of citation objects into a specific style
- Parameters: `citations` (JSON string), `style` (str: "numbered", "apa", "mla"; default "numbered")
- Each citation object expected to have: `title`, `authors`, `year`, `url`, `source`
- Outputs one line per citation in the requested format

**ReportSaverTool** (`report_saver`)
- Saves markdown content to a file in `./reports/` (configurable `output_dir`)
- Parameters: `content` (str, required), `filename` (str, optional; auto-generated with timestamp if omitted)
- Returns the saved file path on success

**MCP Server** (`export/mcp_server.py`): Exposes both tools via FastMCP.

### 1.5 MCP Manager (`mcp/manager.py`)

`MCPManager` manages the lifecycle of MCP (Model Context Protocol) server subprocesses. It provides a dual-mode tool architecture:

- **In-process mode** (default): Tools run directly in the Python process. Fast, no subprocess overhead. Used when MCP is disabled in config.
- **MCP mode**: Tools run in separate subprocesses via `stdio` transport. Enables tool isolation and remote execution.

Key methods:

| Method | Description |
|---|---|
| `enabled` (property) | Returns whether MCP mode is enabled in config |
| `create_tool_collection(server_names, local_tools)` | Connects to specified MCP servers, collects their tools, merges with local tools into a `ToolCollection` |
| `disconnect_all()` | Cleanly disconnects all MCP client connections |

In MCP mode, each server (search, analysis, export) runs as an independent subprocess. The manager connects via `MCPClients.connect_stdio()`, then aggregates all tools from all connected servers. The `Terminate` tool (which agents use to signal completion) is always added locally.

---

## 2. Agents Module (`src/agents/`)

### 2.1 Specialized Agents (`specialized.py`)

All 6 agents inherit from OpenManus's `ToolCallAgent`, which provides the ReAct reasoning loop. Each agent has:

- A `name` and `description` (Pydantic `Field`)
- A `system_prompt` that defines its role and behavior
- A `max_steps` limit (hard cap on ReAct loop iterations)
- Access to a `ToolCollection` via `available_tools`

The ReAct loop in `ToolCallAgent.step()`:
1. **Think**: Send conversation + system prompt to the LLM. The LLM returns either a text response or a tool call.
2. **Act**: If tool call — execute the tool, append the result to conversation, loop back to think. If text only — agent is done (transition to FINISHED state).
3. **Terminate**: The agent must explicitly call the `Terminate` tool to signal completion.

Agent details:

| Agent | Name | max_steps | Role |
|---|---|---|---|
| `PlannerAgent` | `planner` | 5 | Decomposes research topic into sub-topics, search queries, and structured outline. Outputs JSON plan. Has `parse_plan()` to extract JSON from LLM response. |
| `SearcherAgent` | `searcher` | 20 | Searches multiple information sources. Higher step limit because each search is a separate tool call. |
| `AnalystAgent` | `analyst` | 10 | Critically analyzes search results: cross-validation, credibility assessment, contradiction detection. |
| `SynthesizerAgent` | `synthesizer` | 10 | Integrates all analysis findings into a coherent framework following the outline. |
| `WriterAgent` | `writer` | 15 | Produces the final research report with citations and references. |
| `CriticAgent` | `critic` | 5 | Quality evaluation. Outputs JSON scores (0-100) for completeness, accuracy, structure, depth, credibility, clarity. Has `parse_review()` to extract JSON. |

**Design rationale for step limits**: Searcher needs the most steps (each search query is 1+ tool calls). Critic needs the fewest (just review and output scores). The limits act as safety bounds — in practice agents typically finish via `Terminate` before hitting the limit.

### 2.2 LLM Adapter (`llm_adapter.py`)

`LLMAdapter` bridges two different ecosystems:

- **OpenManus** expects an `LLM` object with `ask_tool(messages, system_msgs, tools, tool_choice) → ChatCompletionMessage`
- **LangChain** provides `BaseChatModel` with `ainvoke(messages) → BaseMessage`

The adapter performs bidirectional format conversion:

**Input conversion** (OpenManus → LangChain):
- `Message(role="user")` → `HumanMessage`
- `Message(role="assistant")` → `AIMessage` (including tool_calls mapping)
- `Message(role="tool")` → `ToolMessage` (with `tool_call_id`)
- `Message(role="system")` → `SystemMessage`
- OpenManus tool definitions (`{"type": "function", "function": {...}}`) → LangChain tool format (`{"name": ..., "description": ..., "parameters": ...}`)
- Tool choice mapping: `"required"` → `"any"`, `"auto"` → `"auto"`, `"none"` → `"none"` (LangChain-compatible strings)

**Output conversion** (LangChain → OpenManus):
- `AIMessage.content` → `ToolCallResult.content`
- `AIMessage.tool_calls` → list of `ToolCall` objects with `Function(name, arguments)`

The adapter also handles edge cases: treating non-string content as string, safely parsing JSON from tool call arguments, and returning `None` on exception rather than crashing.

---

## 3. LLM Configuration (`src/llm/config.py`)

### 3.1 AgentLLMConfig

A `@dataclass` holding per-agent LLM settings:

| Field | Type | Default | Description |
|---|---|---|---|
| `provider` | `Literal["anthropic", "openai", "deepseek", "ollama"]` | required | Which LLM provider to use |
| `model` | `str` | required | Model name (e.g., `"claude-sonnet-4-6"`) |
| `api_key` | `str` | `""` | API key; empty means use env var |
| `base_url` | `str \| None` | `None` | Custom API endpoint (e.g., proxy) |
| `temperature` | `float` | `0.5` | Sampling temperature |
| `max_tokens` | `int` | `4096` | Max output tokens |
| `top_p` | `float` | `1.0` | Nucleus sampling parameter |

Each of the 6 agents gets its own config, read from `config/agents.yaml`. The config file has a `default` section and per-agent overrides. In `build_agent_configs()` (`main.py`), agent-specific config falls back to `default` if not specified, and an empty agent config also falls back to `default`.

### 3.2 create_llm() Factory

Creates a LangChain `BaseChatModel` from an `AgentLLMConfig`:

| Provider | LangChain Class | Notes |
|---|---|---|
| `anthropic` | `ChatAnthropic` | Supports `base_url` for proxies |
| `openai` | `ChatOpenAI` | Standard OpenAI-compatible |
| `deepseek` | `ChatOpenAI` | Uses OpenAI-compatible endpoint, defaults to `https://api.deepseek.com` |
| `ollama` | `ChatOllama` | Local models; only temperature (no max_tokens/top_p) |

The factory pattern decouples agent creation from specific model providers — adding a new provider only requires adding one branch here.

---

## 4. Graph Module (`src/graph/`)

### 4.1 State (`state.py`)

`ResearchState` is a `TypedDict` (with `total=False` — all fields optional) representing the shared state passed between nodes in the LangGraph workflow.

**Annotated reducers**: Some fields use `Annotated[type, reducer]` to control how parallel node outputs merge:
- `search_results: Annotated[list[dict], operator.add]` — parallel searchers append results
- `analyses: Annotated[list[dict], operator.add]` — parallel analysts append analyses
- `messages: Annotated[list, add_messages]` — LangGraph's built-in message merge

**State fields by category**:

| Category | Fields |
|---|---|
| Input | `topic`, `language`, `search_sources` |
| Planner output | `outline`, `search_queries`, `information_needs` |
| Searcher output | `search_results` (additive) |
| Analyst output | `analyses` (additive) |
| Synthesizer output | `synthesized_findings` |
| Writer output | `draft_report`, `final_report`, `citations` |
| Critic output | `quality_score`, `critique`, `gaps` |
| Flow control | `current_phase`, `research_round`, `max_rounds`, `quality_threshold`, `overall_score` |
| Messages | `messages` (LangGraph checkpointing) |

### 4.2 Nodes (`nodes.py`)

Each node is an async function taking `(state: ResearchState, agent, ...) → dict` that updates selected state fields. The node pattern:

1. **Build a request string** from relevant state fields
2. **Call `_run_agent_with_progress()`** — resets the agent to IDLE, runs `agent.run(request)` with timeout, polls progress every 500ms
3. **Extract the result** from the agent's last assistant message
4. **Return a dict** of state updates

**_run_agent_with_progress()** internals:
- Calls `_reset_agent(agent)` which sets state to IDLE, resets step counter, clears memory
- Creates an async polling task that reads `agent.current_step` and `agent.state` every 500ms
- Runs `agent.run(request)` with `asyncio.timeout(timeout)` wrapping
- On timeout: sets agent state to FINISHED, raises `RuntimeError`
- On error: calls `progress.agent_error()` and re-raises

**Node-specific behavior**:

| Node | Request Content | Timeout |
|---|---|---|
| `planner_node` | On round 1: topic + "create comprehensive plan". On later rounds: topic + gaps + "focus on filling gaps" | 300s |
| `searcher_node` | Iterates queries one at a time (up to 10), each in a fresh agent run. Sends query + "use search tools, call Terminate when done" | 600s |
| `analyst_node` | Concatenates all search result contents (truncated to 30000 chars) + "critically analyze, provide: key findings, credibility, contradictions, gaps, conclusions" | 600s |
| `synthesizer_node` | Outline (as JSON) + all analysis contents (30000 chars) + "synthesize into cohesive framework" | 600s |
| `writer_node` | Outline + synthesized findings (20000 chars) + "write comprehensive report with executive summary, citations, references" | 900s |
| `critic_node` | Draft report (20000 chars) + "evaluate, output scores in JSON" | 300s |

**Searcher is different**: Unlike other agents that run once, the searcher runs sequentially for each query (up to 10), with a full agent reset between queries. This ensures each search gets a clean context. The overall searcher timeout (600s) covers all individual searches.

### 4.3 Workflow (`workflow.py`)

**build_workflow()** assembles the complete pipeline:

1. **Agent instantiation**: For each of the 6 agents, calls `_create_agent()` which:
   - Constructs the agent with its `ToolCollection`
   - Creates an `LLMAdapter` from the agent's config
   - Injects the adapter into `agent.__dict__["llm"]` (bypassing Pydantic validation that would otherwise replace it with an `app.llm.LLM` instance)
   - Overrides `max_steps` from config if specified

2. **Graph construction**:
   ```
   planner → searcher → analyst → synthesizer → writer → critic
                                                              │
                                     ┌────────────────────────┘
                                     │ quality < threshold & rounds remain → planner
                                     │ quality >= threshold or max rounds → formatter → END
                                     │ (also accepts "end" as a valid routing target)
                                     └──────────────────────────────────────────────→ END
   ```

3. **`_supervisor_router()`**: The conditional edge after critic:
   - Increments `research_round` by 1
   - Returns `"formatter"` if score >= threshold or max rounds reached
   - Returns `"planner"` otherwise (re-plan with gaps from critic)

4. **`_formatter_node()`**: Adds a header (title, timestamp, rounds, quality score) to the draft report, producing `final_report`.

5. **Checkpointing** (optional): If `use_checkpointer=True`, uses `SqliteSaver` for fault tolerance — state is persisted to SQLite after each node, allowing resume on failure.

**Routing edge case**: `_supervisor_router` returns `Literal["planner", "formatter", "end"]` — the `"end"` branch maps to `END` in `add_conditional_edges` but in practice the router always returns `"planner"` or `"formatter"`. The `"end"` branch exists as a safety valve.

---

## 5. Memory Module (`src/memory/`)

### 5.1 Short-term Memory (`memory.py`)

`Memory` manages in-memory conversation history for a single agent session:

| Feature | Implementation |
|---|---|
| Storage | `list[BaseMessage]` with configurable `max_messages` (default 50) |
| Eviction | FIFO: when limit exceeded, evicts the oldest non-system message (system messages preserved) |
| Token estimation | Rough heuristic: `len(content) // 4` characters per token |
| Serialization | `to_dict()` converts messages to `{"type": str, "content": str}` dicts for checkpointing |
| Reset | `clear()` empties all messages |

Used primarily by OpenManus's `BaseAgent.memory` field internally. Each agent reset (in `_reset_agent()`) clears this memory.

### 5.2 Long-term Memory (`long_term.py`)

`LongTermMemory` provides persistent vector-backed storage using ChromaDB:

| Feature | Implementation |
|---|---|
| Storage backend | ChromaDB `PersistentClient` with cosine similarity (`hnsw:space: cosine`) |
| Entry ID | MD5 hash of `topic + timestamp` |
| Chunking | Splits content at sentence boundaries into ~2000-char chunks |
| Add | `add(topic, content, metadata)` — chunks the content, stores with metadata including topic, timestamp, chunk_index |
| Query | `query(topic, n_results)` — semantic search returning content, metadata, and distance score |
| Delete | `delete_topic(topic)` — removes all entries matching a topic filter |

The chunking strategy splits on `. ` (period-space) to preserve sentence boundaries, avoiding mid-sentence cuts that would reduce retrieval quality. Chunks are stored as separate ChromaDB documents but share the same entry ID prefix (e.g., `abc123_chunk_0`, `abc123_chunk_1`).

---

## 6. Utils Module (`src/utils/`)

### 6.1 Progress Tracker (`progress.py`)

`ProgressTracker` provides a real-time Rich Live display showing agent status during pipeline execution.

**Architecture**: Uses a `@dataclass` `_AgentSlot` per agent tracking: name, status (pending/running/finished/error/timeout), current step, max steps, start time, and detail text.

**Display**: A Rich `Table` with columns: Agent (with status icon), Status label, Steps (current/max), Elapsed time, and Detail text. Refreshes at 4 Hz via `Live`.

**Status icons**: ○ pending, ◎ running (yellow), ● done (green), ✕ error (red), ⏱ timeout (red)

**Usage pattern**: Context manager (`with tracker:`) that starts the Live display on enter and renders a final summary table on exit. The pipeline calls:
- `agent_started(name, max_steps)` — when a node begins
- `agent_step_update(name, step, max_steps, detail)` — at each step (polled at 500ms intervals)
- `agent_finished(name)` — on successful completion
- `agent_timeout(name, timeout_s)` — on timeout
- `agent_error(name, msg)` — on failure

The final render shows a summary line with completion count, error count, and total elapsed time.

---

## 7. Main Entry Point (`src/main.py`)

### CLI

Uses Typer with a single `research` command:

```
uv run python -m src.main "Research topic here"
uv run python -m src.main "研究主题" --config config/research.yaml --agents config/agents.yaml -v
```

### Configuration Loading

Two YAML config files:

**`config/agents.yaml`**: Per-agent LLM settings with a `default` section and optional per-agent overrides for provider, model, temperature, max_tokens.

**`config/research.yaml`**: Pipeline settings:
- `mcp.enabled` — whether to use MCP subprocess tools
- `mcp.servers` — per-server command/args config
- `language` — "zh" or "en"
- `search_sources` — list of source names
- `max_rounds` — max research iterations (default 3)
- `quality_threshold` — score to pass critic (default 75)
- `max_agent_turns` — per-agent step overrides
- `agent_timeouts` — per-agent timeout in seconds
- `output_dir` — where to save reports
- `use_checkpointer` — whether to persist state
- `checkpointer_path` — SQLite database path

### Execution Flow

1. Load both YAML configs
2. `build_agent_configs()` — parse into `AgentLLMConfig` dict, resolving env vars for API keys and base URLs
3. `build_tool_collections()` — either create in-process tools or connect MCP servers
4. Create `ProgressTracker`
5. `build_workflow()` — instantiate all 6 agents with their tools and LLM configs, compile the StateGraph
6. Build `initial_state` from config
7. `workflow.ainvoke(initial_state)` — run the pipeline
8. Save report to `output_dir` with auto-generated filename
9. Display report preview (first 1000 chars) in the terminal

### 导入路径

项目将 OpenManus 核心 Agent 框架内联到 `src/_framework/`（~950 行），无需安装 OpenManus 或操作 `sys.path`。所有模块通过 `from src._framework import ...` 获取 BaseTool、ToolCollection、Terminate、ToolCallAgent 等。

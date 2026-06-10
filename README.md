# Multi-Agent Research Assistant

基于 LangGraph + vendored Agent 框架的多 Agent 研究助手系统。

多个专用 Agent 协作完成研究主题的分解、搜索、分析、综合、撰写，最终生成详细的研究报告。每个 Agent 可独立配置不同的大语言模型。

## 架构

```
用户输入 → Planner → Searcher → Analyst → Synthesizer → Writer → Critic
                ↑                                                    │
                └────────── 质量不通过则重试 ─────────────────────────┘
                                            │
                                      质量通过 ↓
                                      最终报告
```

- **LangGraph** 负责编排：哪个 Agent 何时执行、条件路由
- **Vendored Agent 框架**（`src/_framework/`）提供 Agent 基座：`BaseAgent` → `ReActAgent` → `ToolCallAgent`
- **LLMProvider** 桥接 OpenManus LLM 接口到原生 SDK，支持 Anthropic / OpenAI / DeepSeek / Ollama
- 每个 Agent 通过 `config/agents.yaml` 独立配置模型

详细架构与实现文档：[docs/architecture.md](docs/architecture.md)

## 项目结构

```
muti_agent/
├── src/
│   ├── _framework/          # Vendored Agent 框架
│   ├── agents/              # 6 个研究 Agent + LLMProvider
│   ├── graph/               # LangGraph 工作流（State, Nodes, Workflow）
│   ├── tools/               # 工具层（search, analysis, export, MCP）
│   ├── llm/                 # LLM 配置工厂
│   ├── memory/              # ChromaDB 长期记忆
│   ├── utils/               # 进度追踪
│   └── main.py              # CLI 入口
├── config/                  # YAML 配置文件
├── docs/                    # 架构文档
└── reports/                 # 输出报告
```

## 快速开始

### 1. 安装

```bash
# 使用 uv 管理环境
uv sync
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入所需 API key
```

至少需要配置一个 LLM 提供商的 API key（如 `DEEPSEEK_API_KEY`）。

### 3. 配置 Agent 模型

编辑 `config/agents.yaml` 为每个 Agent 指定模型：

```yaml
default:
  provider: deepseek
  model: deepseek-v4-flash
  temperature: 0.3

planner:
  provider: deepseek
  model: deepseek-v4-flash
  temperature: 0.2
```

### 4. 运行

```bash
uv run python -m src.main "人工智能对教育的影响"
uv run python -m src.main "Research topic" --config config/research.yaml -v
```

## Agent 角色

| Agent | 职责 | 推荐模型 |
|-------|------|---------|
| **Planner** | 分解主题、生成大纲 | DeepSeek Flash |
| **Searcher** | 多源并行搜索（Web/学术/百科） | DeepSeek Flash |
| **Analyst** | 深度分析、交叉验证 | DeepSeek Pro |
| **Synthesizer** | 综合归纳、构建逻辑框架 | DeepSeek Pro |
| **Writer** | 撰写结构化报告 | DeepSeek Flash |
| **Critic** | 质量评分、缺口识别 | DeepSeek Flash |

## 搜索工具

Searcher Agent 可使用以下工具进行多源信息检索：

| 工具 | 来源 | API Key | 免费额度 |
|------|------|---------|---------|
| **brave_search** | Brave Search API — 真实网页搜索 | `BRAVE_SEARCH_API_KEY` | 2,000 次/月 |
| **tavily_search** | Tavily — AI 优化的搜索 | `TAVILY_API_KEY` | 1,000 次/月 |
| **duckduckgo_search** | DuckDuckGo Instant Answers | 无需 | 无限制 |
| **arxiv_search** | arXiv — 学术论文 | 无需 | 无限制 |
| **wikipedia_search** | Wikipedia — 百科知识 | 无需 | 无限制 |
| **jina_reader** | Jina Reader — URL 转 Markdown | 无需 | 免费 |
| **web_scraper** | 正则 HTML 抓取（降级备选） | 无需 | — |

> API key 通过 `.env` 文件配置。未配置 key 的工具会返回错误提示，Agent 会自动选择可用的工具。

## 配置

### `config/research.yaml` — 研究流程配置

```yaml
max_rounds: 2                    # 最大研究轮次
quality_threshold: 50            # 质量阈值（0-100）
search_sources:                  # 搜索源
  - brave
  - tavily
  - duckduckgo
  - arxiv
  - wikipedia
language: zh                     # 报告语言
output_dir: ./reports            # 输出目录
```

### `config/agents.yaml` — Agent 模型配置

每个 Agent 独立配置：provider, model, temperature, max_tokens

## MCP 工具扩展

项目支持通过 **Model Context Protocol (MCP)** 接入外部工具服务器。

通过 `config/research.yaml` 中的 `mcp.enabled` 切换：

```yaml
mcp:
  enabled: true   # true = MCP 模式, false = In-Process 模式
  servers:
    search:
      command: uv
      args: ["run", "python", "-m", "src.tools.mcp_server"]
```

### 添加外部 MCP Server

```yaml
mcp:
  enabled: true
  servers:
    search:
      command: uv
      args: ["run", "python", "-m", "src.tools.mcp_server"]
    github:                                     # GitHub MCP Server
      command: npx
      args: ["-y", "@anthropic/mcp-server-github"]
```

然后在 `src/main.py` 的 `build_tool_collections()` 中绑定到 Agent。

常用外部 MCP Server：

| Server | 用途 | command |
|--------|------|---------|
| `@anthropic/mcp-server-github` | 搜索仓库、Issue、PR | `npx -y @anthropic/mcp-server-github` |
| `@anthropic/mcp-server-filesystem` | 读写本地文件 | `npx -y @anthropic/mcp-server-filesystem <dir>` |
| `@anthropic/mcp-server-puppeteer` | 浏览器自动化 | `npx -y @anthropic/mcp-server-puppeteer` |
| `@anthropic/mcp-server-postgres` | PostgreSQL 查询 | `npx -y @anthropic/mcp-server-postgres <url>` |

## 支持的 LLM 提供商

- **Anthropic** — Claude 系列
- **OpenAI** — GPT 系列 + 任何 OpenAI 兼容 API
- **DeepSeek** — DeepSeek 系列
- **Ollama** — 本地模型

## License

MIT

## Attribution

The agent framework in `src/_framework/` is vendored from [OpenManus](https://github.com/FoundationAgents/OpenManus) (MIT License) by the OpenManus contributors. See `src/_framework/ATTRIBUTION.md` for details.

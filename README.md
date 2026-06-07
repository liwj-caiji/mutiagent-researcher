# Multi-Agent Research Assistant

基于 LangGraph + OpenManus 的多 Agent 研究助手系统。

多个专用 Agent 协作完成研究主题的分解、搜索、分析、综合、撰写，最终生成详细的研究报告。每个 Agent 可独立配置不同的大语言模型。

## 架构

```
用户输入 → Planner → Searcher → Analyst → Synthesizer → Writer → Critic
                ↑         (多源并行)                             │
                └────────── 质量不通过则重试 ─────────────────────┘
                                    │
                              质量通过 ↓
                              最终报告
```

- **LangGraph** 负责编排：哪个 Agent 何时执行、条件路由、并行分发
- **OpenManus**（FoundationAgents/OpenManus）提供 Agent 基座：`BaseAgent` → `ReActAgent` → `ToolCallAgent`
- **LLMAdapter** 桥接 OpenManus LLM 接口到 LangChain，支持 Anthropic / OpenAI / DeepSeek / Ollama
- 每个 Agent 通过 `config/agents.yaml` 独立配置模型

## 项目结构

```
muti_agent/
├── src/
│   ├── agents/
│   │   ├── specialized.py    # 6 个研究 Agent（继承 OpenManus ToolCallAgent）
│   │   └── llm_adapter.py   # 桥接 OpenManus LLM → LangChain 多 provider
│   ├── graph/               # LangGraph 工作流（State, Nodes, Workflow）
│   ├── tools/               # 工具层（继承 OpenManus BaseTool）
│   ├── memory/              # 记忆系统
│   ├── llm/                 # LLM 配置工厂
│   └── main.py              # CLI 入口
├── OpenManus_origin/        # 原始 OpenManus（FoundationAgents/OpenManus）
├── config/                  # YAML 配置文件
├── docs/                    # 架构与变更文档
└── reports/                 # 输出报告
```

## 快速开始

### 1. 安装

```bash
# 使用 uv 管理环境
uv sync

# 或 pip
pip install -e .
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入所需 API key
```

至少需要配置一个 LLM 提供商的 API key（如 `ANTHROPIC_API_KEY`）。

### 3. 配置 Agent 模型

编辑 `config/agents.yaml` 为每个 Agent 指定模型：

```yaml
planner:
  provider: anthropic
  model: claude-sonnet-4-6
  temperature: 0.3

searcher:
  provider: deepseek
  model: deepseek-chat
  temperature: 0.1
```

### 4. 运行

```bash
uv run python -m src.main "人工智能对教育的影响"
```

## Agent 角色

| Agent | 职责 | 推荐模型 |
|-------|------|---------|
| **Planner** | 分解主题、生成大纲 | Claude Sonnet |
| **Searcher** | 多源并行搜索（Web/学术/百科） | DeepSeek / Haiku |
| **Analyst** | 深度分析、交叉验证 | Claude Opus |
| **Synthesizer** | 综合归纳、构建逻辑框架 | Claude Opus |
| **Writer** | 撰写结构化报告 | Claude Sonnet |
| **Critic** | 质量评分、缺口识别 | Claude Sonnet |

## 配置

### `config/research.yaml` — 研究流程配置

```yaml
max_rounds: 3                    # 最大研究轮次
quality_threshold: 75            # 质量阈值（0-100）
search_sources:                  # 搜索源
  - web
  - arxiv
  - wikipedia
language: zh                     # 报告语言
output_dir: ./reports            # 输出目录
```

### `config/agents.yaml` — Agent 模型配置

每个 Agent 独立配置：provider, model, temperature, max_tokens

## 项目结构

```
muti_agent/
├── src/
│   ├── agents/          # Agent 实现（Base, ReAct, ToolCall, 专用 Agent）
│   ├── graph/           # LangGraph 工作流（State, Nodes, Workflow）
│   ├── tools/           # 工具层（搜索、分析、导出）
│   ├── memory/          # 记忆系统（短期 + ChromaDB 长期）
│   ├── llm/             # LLM 配置工厂（多 provider 支持）
│   └── main.py          # CLI 入口
├── config/              # YAML 配置文件
├── docs/                # 架构文档
└── reports/             # 输出报告
```

## 支持的 LLM 提供商

- **Anthropic** — Claude 系列
- **OpenAI** — GPT 系列 + 任何 OpenAI 兼容 API
- **DeepSeek** — DeepSeek 系列
- **Ollama** — 本地模型

## License

MIT

## Attribution

The agent framework in `src/_framework/` is vendored from [OpenManus](https://github.com/FoundationAgents/OpenManus) (MIT License) by the OpenManus contributors. See `src/_framework/ATTRIBUTION.md` for details.

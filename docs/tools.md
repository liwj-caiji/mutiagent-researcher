# 工具实现文档

> 本文档描述所有 Agent 可调用工具的调用方式、实现细节、参数和返回值，以及 MCP Server 部署方案。

## 调用机制

### 双模式：进程内 + MCP 服务器

工具支持两种调用模式，通过 `config/research.yaml` 中的 `mcp.enabled` 切换。

**模式 1 — 进程内调用（默认，`mcp.enabled: false`）**

```
Agent 进程
  ToolCallAgent.act()
    → ToolCollection.execute(name, args)
      → tool.execute(**kwargs)         ← 直接 Python 异步方法调用
```

**模式 2 — MCP 服务器远程调用（`mcp.enabled: true`）**

```
Agent 进程（主进程）                        MCP Server 子进程
═══════════════════════                   ═══════════════════
ToolCallAgent.act()                       FastMCP
  → MCPClients.execute(name, args)         → @mcp.tool() handler
    → MCPClientTool.call_tool()              → tool.execute(**kwargs)
      → session.call_tool(name, args)          → 返回 ToolResult
        → stdio JSON-RPC ────────────────────→
        ← stdio JSON-RPC ←────────────────────
```

> **透明性**：`MCPClients` 继承 `ToolCollection`，Agent 层无需感知工具是本地还是远程。

### MCP Server 架构

```
                        ┌───────────────────────┐
                        │      MCPManager        │
                        │  (src/tools/mcp/)      │
                        │                        │
                        │  • 启动/停止子进程      │
                        │  • 管理 MCPClients 连接 │
                        │  • 构建 ToolCollection  │
                        └───────────┬───────────┘
                                    │
             ┌──────────────────────┼──────────────────────┐
             │                      │                      │
             ▼                      ▼                      ▼
┌─────────────────────┐ ┌──────────────────┐ ┌─────────────────────┐
│  Search MCP Server   │ │ Analysis MCP     │ │  Export MCP Server   │
│  (stdio transport)   │ │ Server (stdio)   │ │  (stdio transport)   │
│                     │ │                  │ │                     │
│  4 个工具:           │ │  1 个工具:        │ │  2 个工具:            │
│  • web_search       │ │  • python_execute │ │  • citation_formatter│
│  • arxiv_search     │ │                  │ │  • report_saver      │
│  • wikipedia_search │ │                  │ │                     │
│  • web_scraper      │ │                  │ │                     │
└─────────────────────┘ └──────────────────┘ └─────────────────────┘
```

### MCP Server 启动方式

每个 MCP Server 可独立启动：

```bash
# 搜索工具服务器
uv run python -m src.tools.search.mcp_server

# 分析工具服务器
uv run python -m src.tools.analysis.mcp_server

# 导出工具服务器
uv run python -m src.tools.export.mcp_server
```

启动后，服务器在 stdio 上监听 MCP 协议（JSON-RPC）。`MCPManager` 在 Agent 初始化时自动启动并连接这些服务器。

### MCP 配置

```yaml
# config/research.yaml
mcp:
  enabled: false          # 设为 true 启用 MCP 模式
  transport: stdio        # 传输方式（当前仅支持 stdio）
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
```

### 添加新的 MCP Server

1. 创建 `src/tools/<category>/mcp_server.py`
2. 使用 FastMCP 包装已有的 BaseTool 实例：

```python
from mcp.server.fastmcp import FastMCP
from src.tools.xxx.tools import MyTool

mcp = FastMCP("server_name")
_tool = MyTool()

@mcp.tool()
async def my_tool(param1: str, param2: int = 5) -> str:
    result = await _tool.execute(param1=param1, param2=param2)
    return result.output or f"Error: {result.error}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

3. 在 `config/research.yaml` 的 `mcp.servers` 中添加配置
4. 在 `src/main.py` 的 `build_tool_collections()` 中绑定到对应 Agent

### MCP vs 进程内模式对比

| 维度 | 进程内模式 | MCP 模式 |
|------|-----------|---------|
| 部署 | Agent 进程内 | 独立子进程 |
| 延迟 | 零（直接调用） | 低（stdio IPC） |
| 隔离 | 无（共享进程） | 强（独立进程） |
| 资源管理 | 自动 | MCPManager 管理 |
| 扩展性 | 需修改 Agent 代码 | 独立扩展 MCP Server |
| 适用场景 | 开发调试 | 生产部署、分布式 |

---

### 基类：OpenManus BaseTool

所有工具继承 `app.tool.base.BaseTool`（Pydantic 模型）：

```python
class BaseTool(BaseModel, ABC):
    name: str                # 唯一标识，LLM 通过 name 选择工具
    description: str         # 工具用途描述，发送给 LLM
    parameters: dict | None  # JSON Schema，定义参数

    async def execute(self, **kwargs) -> Any: ...  # 子类实现
    def to_param(self) -> dict: ...                # 转为 OpenAI function calling 格式
```

### ToolCollection 管理

`app.tool.ToolCollection` 管理工具集合，负责：
- `to_params()` — 将所有工具转为 OpenAI function calling 格式发给 LLM
- `execute(name, tool_input)` — 按名称路由到对应工具执行

`app.tool.mcp.MCPClients` 继承 `ToolCollection`，使用相同接口但通过 MCP 协议远程调用工具。

### 在 main.py 中组装工具集

```python
# 进程内模式 — SearcherAgent 工具集（4 个搜索工具 + Terminate）
ToolCollection(
    WebSearchTool(),
    ArxivSearchTool(),
    WikipediaSearchTool(),
    WebScraperTool(),
    Terminate(),
)

# MCP 模式 — SearcherAgent 工具集（连接 search MCP Server + Terminate）
await mcp_manager.create_tool_collection(
    server_names=["search"],
    local_tools=[Terminate()],
)

# 其他 Agent（进程内和 MCP 模式相同）— 仅 Terminate
ToolCollection(Terminate())
```

---

## 工具清单

### 1. WebSearchTool — 网络搜索

| 项目 | 说明 |
|------|------|
| 文件 | `src/tools/search/tools.py` |
| MCP Server | `src/tools/search/mcp_server.py`（工具名：`web_search`） |
| 调用方式 | DuckDuckGo Instant Answer API（HTTP GET） |
| 需要认证 | 否（免费，无限额） |
| 超时 | 15 秒 |

**参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query` | string | 是 | 搜索关键词 |
| `max_results` | integer | 否 | 最大结果数，默认 5 |

**实现细节：**

```python
async def execute(self, query, max_results=5):
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1},
        )
        data = resp.json()
        # 1. 取 Abstract（百科摘要）+ AbstractURL
        # 2. 取 RelatedTopics[]（关联主题）前 N 条，每条的 Text + FirstURL
        # 拼接为 Markdown 文本返回
```

**返回示例：**

```markdown
**Abstract**
DuckDuckGo is a search engine...
Source: https://en.wikipedia.org/wiki/DuckDuckGo

- Related topic text
  https://example.com/page
```

**错误处理：** 网络异常返回 `ToolResult(error="Search failed: ...")`

---

### 2. ArxivSearchTool — 学术论文搜索

| 项目 | 说明 |
|------|------|
| 文件 | `src/tools/search/tools.py` |
| MCP Server | `src/tools/search/mcp_server.py`（工具名：`arxiv_search`） |
| 调用方式 | `arxiv` Python 库（arXiv API 封装） |
| 需要认证 | 否 |
| 依赖 | `pip install arxiv` |

**参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query` | string | 是 | 搜索词（支持关键词/作者/标题） |
| `max_results` | integer | 否 | 最大结果数，默认 5 |

**实现细节：**

```python
async def execute(self, query, max_results=5):
    import arxiv
    search = arxiv.Search(query=query, max_results=max_results,
                          sort_by=arxiv.SortCriterion.Relevance)
    for paper in search.results():
        # 提取: paper.title / authors / published / entry_id（URL）/ summary
        # 每条结果用 "---" 分隔
```

**返回示例：**

```markdown
**Attention Is All You Need**
Authors: Ashish Vaswani, Noam Shazeer, ...
Published: 2017-06-12
URL: http://arxiv.org/abs/1706.03762v5
Abstract: The dominant sequence transduction models...

---
**BERT: Pre-training of Deep Bidirectional Transformers**
...
```

**错误处理：** 无结果返回 `"No arXiv results for 'query'."`，异常返回 `ToolResult(error=...)`

---

### 3. WikipediaSearchTool — 维基百科搜索

| 项目 | 说明 |
|------|------|
| 文件 | `src/tools/search/tools.py` |
| MCP Server | `src/tools/search/mcp_server.py`（工具名：`wikipedia_search`） |
| 调用方式 | `wikipedia` Python 库 |
| 需要认证 | 否 |
| 依赖 | `pip install wikipedia` |

**参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query` | string | 是 | 搜索词或页面标题 |
| `language` | string | 否 | 语言代码，默认 `"en"` |

**实现细节：**

```python
async def execute(self, query, language="en"):
    import wikipedia
    wikipedia.set_lang(language)
    # 1. wikipedia.search(query, results=3) — 搜索匹配页面标题
    # 2. 取前 2 个结果
    # 3. wikipedia.page(title) — 获取页面对象（含 URL）
    # 4. wikipedia.summary(title, sentences=3) — 获取前 3 句摘要
    # 5. 拼接为 Markdown，页面间用 "---" 分隔
```

**返回示例：**

```markdown
**Artificial intelligence**
URL: https://en.wikipedia.org/wiki/Artificial_intelligence
Summary: Artificial intelligence (AI) is intelligence demonstrated by machines...

---
**Machine learning**
URL: https://en.wikipedia.org/wiki/Machine_learning
Summary: Machine learning (ML) is a field of inquiry devoted to...
```

**错误处理：** 无匹配页面返回 `"No Wikipedia articles found for 'query'."`；单页面获取失败跳过继续。

---

### 4. WebScraperTool — 网页内容提取

| 项目 | 说明 |
|------|------|
| 文件 | `src/tools/search/tools.py` |
| MCP Server | `src/tools/search/mcp_server.py`（工具名：`web_scraper`） |
| 调用方式 | HTTP GET + 正则去 HTML 标签 |
| 需要认证 | 否 |
| 超时 | 20 秒 |

**参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `url` | string | 是 | 目标网页 URL |

**实现细节：**

```python
async def execute(self, url):
    async with httpx.AsyncClient(follow_redirects=True) as client:
        resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0 (ResearchBot/1.0)"})
        resp.raise_for_status()  # 非 2xx 抛异常

        # 三步清洗：
        # 1. 去除 <script>...</script> 和 <style>...</style> 标签及其内容
        # 2. 去除所有 HTML 标签（<...> → 换行）
        # 3. 合并连续换行、合并连续空格
        # 截断到 8000 字符
```

**注意：** 这是一个极简的文本提取。不执行 JavaScript，不处理动态渲染页面。对于 JS 渲染的页面会得到空内容。

**错误处理：** HTTP 错误或网络异常返回 `ToolResult(error="Web scraping failed: ...")`

---

### 5. PythonExecuteTool — Python 代码执行

| 项目 | 说明 |
|------|------|
| 文件 | `src/tools/analysis/tools.py` |
| MCP Server | `src/tools/analysis/mcp_server.py`（工具名：`python_execute`） |
| 调用方式 | `subprocess.run()` 子进程执行 |
| 超时 | 60 秒（可配置） |
| 工作目录 | `./data/workspace` |

**参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `code` | string | 是 | 要执行的 Python 源代码 |

**实现细节：**

```python
async def execute(self, code):
    # 1. 将代码写入临时 .py 文件（data/workspace/ 目录下）
    # 2. subprocess.run([sys.executable, script_path],
    #       capture_output=True, text=True, timeout=60, cwd=workspace)
    # 3. 收集 stdout + stderr + exit_code
    # 4. 删除临时文件
    # 5. 返回 ToolResult(output=stdout + stderr)
```

**沙箱程度：** 有限。代码在**子进程**中运行（与主进程隔离），但无 Docker 隔离、无网络限制、无文件系统限制（仅限工作目录写文件）。适用于数据分析计算，不适用于执行不可信代码。

**错误处理：** 超时返回 `"Execution timed out after 60s"`，其他异常返回错误消息。

---

### 6. CitationFormatterTool — 引用格式化

| 项目 | 说明 |
|------|------|
| 文件 | `src/tools/export/tools.py` |
| MCP Server | `src/tools/export/mcp_server.py`（工具名：`citation_formatter`） |
| 调用方式 | 纯 Python 字符串拼接 |
| 需要认证 | 否 |

**参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `citations` | string | 是 | JSON 字符串，引用对象数组 |
| `style` | string | 否 | 格式风格：`numbered`（默认）/ `apa` / `mla` |

**引用对象结构：**

```json
[
  {
    "title": "Paper Title",
    "authors": "Author A, Author B",
    "year": "2024",
    "url": "https://...",
    "source": "arXiv / Wikipedia / Website"
  }
]
```

**实现细节：**

```python
async def execute(self, citations, style="numbered"):
    citations_list = json.loads(citations)
    for i, cite in enumerate(citations_list, 1):
        if style == "numbered":
            line = f"[{i}] {authors}. **{title}**. {source}. {year}. {url}"
        elif style == "apa":
            line = f"{authors} ({year}). *{title}*. {source}. {url}"
        elif style == "mla":
            line = f'{authors}. "{title}." *{source}*, {year}. {url}'
```

**输出示例（numbered）：**

```
[1] Vaswani et al. **Attention Is All You Need**. arXiv. 2017. http://arxiv.org/abs/1706.03762
[2] Devlin et al. **BERT: Pre-training**. arXiv. 2018. http://arxiv.org/abs/1810.04805
```

**错误处理：** JSON 解析失败返回 `"Invalid JSON for citations"`

---

### 7. ReportSaverTool — 报告保存

| 项目 | 说明 |
|------|------|
| 文件 | `src/tools/export/tools.py` |
| MCP Server | `src/tools/export/mcp_server.py`（工具名：`report_saver`） |
| 调用方式 | Python `pathlib.Path.write_text()` |
| 需要认证 | 否 |

**参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `content` | string | 是 | 报告全文（Markdown） |
| `filename` | string | 否 | 文件名，默认 `report-<时间戳>.md` |

**实现细节：**

```python
async def execute(self, content, filename=None):
    filepath = output_dir / (filename or f"report-{timestamp}.md")
    filepath.write_text(content, encoding="utf-8")
```

**输出目录：** `./reports/`（由构造函数 `output_dir` 参数控制）

**错误处理：** 写入失败返回 `ToolResult(error="Failed to save report: ...")`

---

### 8. Terminate — 任务终止

| 项目 | 说明 |
|------|------|
| 文件 | `src/_framework/terminate.py` |
| 调用方式 | 不执行实际操作，仅标记 Agent 状态 |
| MCP Server | 无（始终本地运行） |

**参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `status` | string | 是 | `"success"` 或 `"failure"` |

**实现细节：**

```python
class Terminate(BaseTool):
    async def execute(self, status: str) -> str:
        return f"The interaction has been completed with status: {status}"
```

`Terminate` 是一个**特殊工具**。它本身不产生有意义的输出，但 `ToolCallAgent._handle_special_tool()` 检测到 `Terminate` 时会将 `Agent.state` 设为 `FINISHED`，从而退出 ReAct 循环。每个 Agent 都必须拥有此工具，否则无法正常结束。**Terminate 始终在进程内运行**，不走 MCP Server。

---

## 工具与 Agent 的绑定关系

### 进程内模式

| Agent | WebSearch | Arxiv | Wikipedia | WebScraper | PythonExecute | CitationFormatter | ReportSaver | Terminate |
|-------|:---------:|:-----:|:---------:|:----------:|:------------:|:----------------:|:-----------:|:---------:|
| Planner | | | | | | | | ✓ |
| Searcher | ✓ | ✓ | ✓ | ✓ | | | | ✓ |
| Analyst | | | | | | | | ✓ |
| Synthesizer | | | | | | | | ✓ |
| Writer | | | | | | ✓ | | ✓ |
| Critic | | | | | | | | ✓ |

### MCP 模式

| Agent | Search MCP | Analysis MCP | Export MCP | 本地 Terminate |
|-------|:----------:|:------------:|:----------:|:-------------:|
| Planner | | | | ✓ |
| Searcher | ✓ | | | ✓ |
| Analyst | | | | ✓ |
| Synthesizer | | | | ✓ |
| Writer | | | ✓ | ✓ |
| Critic | | | | ✓ |

> Searcher 是唯一需要外部数据源的工具密集 Agent。Writer 需要引用格式化。其余 Agent 为纯 LLM 推理。

---

## 扩展工具指南

### 添加一个新工具（进程内模式）

1. 在对应 `src/tools/*/tools.py` 中创建类，继承 `app.tool.base.BaseTool`
2. 定义 `name`、`description`、`parameters`（JSON Schema）
3. 实现 `async def execute(self, **kwargs) -> Any`，返回 `ToolResult` 或字符串
4. 在 `src/main.py` 的 `build_tool_collections()` 中注册到对应 Agent 的 `ToolCollection`

### 添加一个新工具（含 MCP 支持）

1. 完成上述进程内模式的 4 步
2. 在对应类别的 `mcp_server.py` 中添加 FastMCP 工具函数：

```python
@mcp.tool()
async def my_tool_name(param1: str, param2: int = 5) -> str:
    """工具描述"""
    result = await _my_tool.execute(param1=param1, param2=param2)
    return result.output or f"Error: {result.error}"
```

3. 如需独立的 MCP Server，创建新的 `src/tools/<new_category>/mcp_server.py`
4. 在 `config/research.yaml` 的 `mcp.servers` 中添加配置
5. 在 `src/main.py` 的 `build_tool_collections()` MCP 分支中绑定到对应 Agent

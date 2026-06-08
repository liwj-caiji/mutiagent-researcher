"""MCP Server — exposes all research tools via Model Context Protocol (stdio).

Usage:
    uv run python -m src.tools.mcp_server
"""

from dotenv import load_dotenv
load_dotenv()

from mcp.server.fastmcp import FastMCP

from src.tools.search import (
    ArxivSearchTool,
    BraveSearchTool,
    DuckDuckGoTool,
    JinaReaderTool,
    TavilySearchTool,
    WebScraperTool,
    WikipediaSearchTool,
)
from src.tools.analysis import PythonExecuteTool
from src.tools.export import CitationFormatterTool

mcp = FastMCP("research-tools")

_duckduckgo = DuckDuckGoTool()
_brave_search = BraveSearchTool()
_tavily_search = TavilySearchTool()
_jina_reader = JinaReaderTool()
_arxiv_search = ArxivSearchTool()
_wikipedia = WikipediaSearchTool()
_web_scraper = WebScraperTool()
_python_exec = PythonExecuteTool()
_citation_fmt = CitationFormatterTool()


# ── Search tools ──────────────────────────────────────────────────────────

@mcp.tool()
async def brave_search(query: str, max_results: int = 5) -> str:
    """Search the web with Brave Search (real web results). Returns title, URL, snippet. Requires BRAVE_SEARCH_API_KEY."""
    result = await _brave_search.execute(query=query, max_results=max_results)
    return result.output if result.output is not None else f"Error: {result.error}"


@mcp.tool()
async def tavily_search(query: str, max_results: int = 5, include_answer: bool = True) -> str:
    """Search the web with Tavily (AI-optimized). Returns clean structured results with an AI summary. Requires TAVILY_API_KEY."""
    result = await _tavily_search.execute(query=query, max_results=max_results, include_answer=include_answer)
    return result.output if result.output is not None else f"Error: {result.error}"


@mcp.tool()
async def duckduckgo_search(query: str, max_results: int = 5) -> str:
    """Search DuckDuckGo Instant Answers for definitions and curated snippets (not full web search). No API key needed."""
    result = await _duckduckgo.execute(query=query, max_results=max_results)
    return result.output if result.output is not None else f"Error: {result.error}"


@mcp.tool()
async def jina_reader(url: str) -> str:
    """Extract clean Markdown content from a URL using Jina Reader. No API key required."""
    result = await _jina_reader.execute(url=url)
    return result.output if result.output is not None else f"Error: {result.error}"


@mcp.tool()
async def arxiv_search(query: str, max_results: int = 5) -> str:
    """Search arXiv for academic papers. Returns paper title, authors, abstract, and URL."""
    result = await _arxiv_search.execute(query=query, max_results=max_results)
    return result.output if result.output is not None else f"Error: {result.error}"


@mcp.tool()
async def wikipedia_search(query: str, language: str = "en") -> str:
    """Search Wikipedia for encyclopedia articles. Returns title, summary, and URL."""
    result = await _wikipedia.execute(query=query, language=language)
    return result.output if result.output is not None else f"Error: {result.error}"


@mcp.tool()
async def web_scraper(url: str) -> str:
    """Extract text content from a web page URL. Use this to read full articles."""
    result = await _web_scraper.execute(url=url)
    return result.output if result.output is not None else f"Error: {result.error}"


# ── Analysis tools ────────────────────────────────────────────────────────

@mcp.tool()
async def python_execute(code: str) -> str:
    """Execute Python code and return stdout/stderr. Use for computation, data analysis, text processing."""
    result = await _python_exec.execute(code=code)
    return result.output if result.output is not None else f"Error: {result.error}"


# ── Export tools ──────────────────────────────────────────────────────────

@mcp.tool()
async def citation_formatter(citations: str, style: str = "numbered") -> str:
    """Format a list of citations/references into a specific style (numbered, apa, or mla)."""
    result = await _citation_fmt.execute(citations=citations, style=style)
    return result.output if result.output is not None else f"Error: {result.error}"


if __name__ == "__main__":
    mcp.run(transport="stdio")

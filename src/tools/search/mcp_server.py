"""Search MCP Server — exposes search tools via Model Context Protocol (stdio)."""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from src.tools.search.tools import ArxivSearchTool, WebScraperTool, WebSearchTool, WikipediaSearchTool

mcp = FastMCP("search")

_web_search = WebSearchTool()
_arxiv_search = ArxivSearchTool()
_wikipedia = WikipediaSearchTool()
_web_scraper = WebScraperTool()


@mcp.tool()
async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for information. Returns title, URL, and snippet for each result."""
    result = await _web_search.execute(query=query, max_results=max_results)
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


if __name__ == "__main__":
    mcp.run(transport="stdio")

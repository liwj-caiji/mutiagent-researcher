"""Search tools — extend OpenManus BaseTool.

Tools: WebSearchTool, ArxivSearchTool, WikipediaSearchTool, WebScraperTool.
"""

from __future__ import annotations

import re
from typing import Optional

import httpx
from src._framework import BaseTool, ToolResult


class WebSearchTool(BaseTool):
    """Search the web using DuckDuckGo (no API key needed)."""

    name: str = "web_search"
    description: str = (
        "Search the web for information. Returns title, URL, and snippet for each result. "
        "Use this for general information queries."
    )
    parameters: Optional[dict] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
            "max_results": {"type": "integer", "description": "Max results (default 5)", "default": 5},
        },
        "required": ["query"],
    }

    async def execute(self, **kwargs) -> ToolResult:
        query = kwargs.get("query", "")
        max_results = kwargs.get("max_results", 5)

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://api.duckduckgo.com/",
                    params={"q": query, "format": "json", "no_html": 1},
                    timeout=15,
                )
                data = resp.json()
                results = []

                if data.get("Abstract"):
                    results.append(
                        f"**Abstract**\n{data['Abstract']}\nSource: {data.get('AbstractURL', '')}\n"
                    )

                for topic in data.get("RelatedTopics", [])[:max_results]:
                    if isinstance(topic, dict) and topic.get("Text"):
                        results.append(f"- {topic['Text']}\n  {topic.get('FirstURL', '')}")

                content = "\n".join(results) if results else f"No results found for '{query}'."
                return ToolResult(output=content)
        except Exception as e:
            return ToolResult(error=f"Search failed: {e}")


class ArxivSearchTool(BaseTool):
    """Search arXiv for academic papers."""

    name: str = "arxiv_search"
    description: str = (
        "Search arXiv for academic papers. Returns paper title, authors, abstract, and URL. "
        "Use this for scientific and technical research."
    )
    parameters: Optional[dict] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query (keywords, author, title)"},
            "max_results": {"type": "integer", "description": "Max results (default 5)", "default": 5},
        },
        "required": ["query"],
    }

    async def execute(self, **kwargs) -> ToolResult:
        query = kwargs.get("query", "")
        max_results = kwargs.get("max_results", 5)

        try:
            import arxiv

            search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
            results = []
            for paper in search.results():
                results.append(
                    f"**{paper.title}**\n"
                    f"Authors: {', '.join(a.name for a in paper.authors)}\n"
                    f"Published: {paper.published.strftime('%Y-%m-%d') if paper.published else 'N/A'}\n"
                    f"URL: {paper.entry_id}\n"
                    f"Abstract: {paper.summary}\n"
                )
            content = "\n---\n".join(results) if results else f"No arXiv results for '{query}'."
            return ToolResult(output=content)
        except Exception as e:
            return ToolResult(error=f"arXiv search failed: {e}")


class WikipediaSearchTool(BaseTool):
    """Search Wikipedia for encyclopedia entries."""

    name: str = "wikipedia_search"
    description: str = (
        "Search Wikipedia for encyclopedia articles. Returns title, summary, and URL. "
        "Use this for background information and general knowledge."
    )
    parameters: Optional[dict] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search topic or page title"},
            "language": {"type": "string", "description": "Language code (default: en)", "default": "en"},
        },
        "required": ["query"],
    }

    async def execute(self, **kwargs) -> ToolResult:
        query = kwargs.get("query", "")
        language = kwargs.get("language", "en")

        try:
            import wikipedia

            wikipedia.set_lang(language)
            search_results = wikipedia.search(query, results=3)
            if not search_results:
                return ToolResult(output=f"No Wikipedia articles found for '{query}'.")

            summaries = []
            for title in search_results[:2]:
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    summary = wikipedia.summary(title, sentences=3, auto_suggest=False)
                    summaries.append(f"**{page.title}**\nURL: {page.url}\nSummary: {summary}\n")
                except Exception:
                    continue

            content = "\n---\n".join(summaries) if summaries else f"Could not retrieve articles for '{query}'."
            return ToolResult(output=content)
        except Exception as e:
            return ToolResult(error=f"Wikipedia search failed: {e}")


class WebScraperTool(BaseTool):
    """Extract text content from a URL."""

    name: str = "web_scraper"
    description: str = "Extract text content from a web page URL. Use this to read full articles."
    parameters: Optional[dict] = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to extract content from"},
        },
        "required": ["url"],
    }

    async def execute(self, **kwargs) -> ToolResult:
        url = kwargs.get("url", "")

        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=20) as client:
                resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0 (ResearchBot/1.0)"})
                resp.raise_for_status()

                html = resp.text
                text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'<[^>]+>', '\n', text)
                text = re.sub(r'\n{3,}', '\n\n', text)
                text = re.sub(r'[ \t]{2,}', ' ', text)
                text = text.strip()[:8000]

                return ToolResult(output=text)
        except Exception as e:
            return ToolResult(error=f"Web scraping failed: {e}")

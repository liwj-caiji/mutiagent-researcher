"""Search tool implementations — Web, arXiv, Wikipedia, WebScraper."""

from __future__ import annotations

import os
import re
from typing import Optional

import httpx
from src._framework import BaseTool, ToolResult


class DuckDuckGoTool(BaseTool):
    """Search DuckDuckGo Instant Answers (no API key needed).

    NOTE: This is NOT a full web search — it returns curated encyclopedia-style
    snippets and related topics. For real web search results, prefer brave_search
    or tavily_search.
    """

    name: str = "duckduckgo_search"
    description: str = (
        "Search DuckDuckGo Instant Answers for definitions, abstracts, and related topics. "
        "NOTE: limited to curated snippets — not a full web search. "
        "For real web results, use brave_search or tavily_search instead."
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


class BraveSearchTool(BaseTool):
    """Search the web using Brave Search API.

    Requires BRAVE_SEARCH_API_KEY environment variable.
    Free tier: 2,000 queries/month (https://brave.com/search/api/).
    """

    name: str = "brave_search"
    description: str = (
        "Search the web using Brave Search. Returns real web results with title, "
        "URL, and snippet. Best for general-purpose web search. Requires API key."
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

        api_key = os.getenv("BRAVE_SEARCH_API_KEY", "")
        if not api_key:
            return ToolResult(error="BRAVE_SEARCH_API_KEY not set. Get a free key at https://brave.com/search/api/")

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": max_results},
                    headers={
                        "X-Subscription-Token": api_key,
                        "Accept": "application/json",
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()

                results = []
                for item in data.get("web", {}).get("results", []):
                    results.append(
                        f"**{item.get('title', 'Untitled')}**\n"
                        f"{item.get('description', 'No description')}\n"
                        f"URL: {item.get('url', '')}\n"
                    )

                content = "\n".join(results) if results else f"No Brave Search results for '{query}'."
                return ToolResult(output=content)
        except Exception as e:
            return ToolResult(error=f"Brave Search failed: {e}")


class TavilySearchTool(BaseTool):
    """Search the web using Tavily Search API (optimized for AI agents).

    Requires TAVILY_API_KEY environment variable.
    Free tier: 1,000 queries/month (https://tavily.com/).
    """

    name: str = "tavily_search"
    description: str = (
        "Search the web using Tavily — an AI-optimized search engine. "
        "Returns clean, structured results with an AI-generated answer summary. "
        "Best for research and fact-finding. Requires API key."
    )
    parameters: Optional[dict] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
            "max_results": {"type": "integer", "description": "Max results (default 5)", "default": 5},
            "include_answer": {
                "type": "boolean",
                "description": "Include an AI-generated answer summary (default true)",
                "default": True,
            },
        },
        "required": ["query"],
    }

    async def execute(self, **kwargs) -> ToolResult:
        query = kwargs.get("query", "")
        max_results = kwargs.get("max_results", 5)
        include_answer = kwargs.get("include_answer", True)

        api_key = os.getenv("TAVILY_API_KEY", "")
        if not api_key:
            return ToolResult(error="TAVILY_API_KEY not set. Get a free key at https://tavily.com/")

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": api_key,
                        "query": query,
                        "max_results": max_results,
                        "include_answer": include_answer,
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()

                lines = []
                if include_answer and data.get("answer"):
                    lines.append(f"**AI Answer:** {data['answer']}\n")

                for item in data.get("results", []):
                    lines.append(
                        f"**{item.get('title', 'Untitled')}**\n"
                        f"{item.get('content', 'No content')}\n"
                        f"URL: {item.get('url', '')}\n"
                    )

                content = "\n".join(lines) if lines else f"No Tavily results for '{query}'."
                return ToolResult(output=content)
        except Exception as e:
            return ToolResult(error=f"Tavily Search failed: {e}")


class JinaReaderTool(BaseTool):
    """Extract clean Markdown content from a URL using Jina Reader API.

    No API key required. Free to use.
    Returns well-formatted Markdown suitable for LLM consumption.
    """

    name: str = "jina_reader"
    description: str = (
        "Read and extract clean Markdown content from any web page URL. "
        "Returns well-formatted text suitable for analysis. "
        "Use this to read full article content from URLs found by other search tools. "
        "No API key required."
    )
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
            async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
                resp = await client.get(
                    f"https://r.jina.ai/{url}",
                    headers={"Accept": "text/markdown"},
                )
                resp.raise_for_status()
                text = resp.text.strip()[:16000]
                return ToolResult(output=text or "(empty page)")
        except Exception as e:
            return ToolResult(error=f"Jina Reader failed: {e}")

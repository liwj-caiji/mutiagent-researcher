"""Export tool implementations — citation formatting."""

from __future__ import annotations

import json
from typing import Optional

from src._framework import BaseTool, ToolResult


class CitationFormatterTool(BaseTool):
    """Format citations in various styles."""

    name: str = "citation_formatter"
    description: str = "Format a list of citations/references into a specific style."
    parameters: Optional[dict] = {
        "type": "object",
        "properties": {
            "citations": {"type": "string", "description": "JSON list of citation objects"},
            "style": {"type": "string", "description": "Citation style: numbered, apa, mla", "default": "numbered"},
        },
        "required": ["citations"],
    }

    async def execute(self, **kwargs) -> ToolResult:
        citations_str = kwargs.get("citations", "[]")
        style = kwargs.get("style", "numbered")

        try:
            citations = json.loads(citations_str)
        except json.JSONDecodeError:
            return ToolResult(error="Invalid JSON for citations")

        formatted = []
        for i, cite in enumerate(citations, 1):
            title = cite.get("title", "Untitled")
            authors = cite.get("authors", "")
            year = cite.get("year", "n.d.")
            url = cite.get("url", "")
            source = cite.get("source", "")

            if style == "numbered":
                line = f"[{i}] {authors}. **{title}**. {source}. {year}. {url}"
            elif style == "apa":
                line = f"{authors} ({year}). *{title}*. {source}. {url}"
            elif style == "mla":
                line = f'{authors}. "{title}." *{source}*, {year}. {url}'
            else:
                line = f"[{i}] {title} — {authors}, {source}, {year}"

            formatted.append(line)

        return ToolResult(output="\n".join(formatted))



"""Export MCP Server — exposes export tools via Model Context Protocol (stdio)."""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from src.tools.export.tools import CitationFormatterTool, ReportSaverTool

mcp = FastMCP("export")

_citation_fmt = CitationFormatterTool()
_report_saver = ReportSaverTool()


@mcp.tool()
async def citation_formatter(citations: str, style: str = "numbered") -> str:
    """Format a list of citations/references into a specific style (numbered, apa, or mla)."""
    result = await _citation_fmt.execute(citations=citations, style=style)
    return result.output if result.output is not None else f"Error: {result.error}"


@mcp.tool()
async def report_saver(content: str, filename: str = "") -> str:
    """Save the research report content to a markdown file."""
    result = await _report_saver.execute(content=content, filename=filename)
    return result.output if result.output is not None else f"Error: {result.error}"


if __name__ == "__main__":
    mcp.run(transport="stdio")

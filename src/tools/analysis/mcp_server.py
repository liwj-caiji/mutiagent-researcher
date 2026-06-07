"""Analysis MCP Server — exposes analysis tools via Model Context Protocol (stdio)."""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from src.tools.analysis.tools import PythonExecuteTool

mcp = FastMCP("analysis")

_python_exec = PythonExecuteTool()


@mcp.tool()
async def python_execute(code: str) -> str:
    """Execute Python code and return stdout/stderr. Use for computation, data analysis, text processing."""
    result = await _python_exec.execute(code=code)
    return result.output if result.output is not None else f"Error: {result.error}"


if __name__ == "__main__":
    mcp.run(transport="stdio")

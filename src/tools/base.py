"""Tool registry — wraps OpenManus BaseTool and ToolCollection.

We use OpenManus's BaseTool/ToolCollection directly. This module provides
a registry for managing tool instances and building ToolCollection objects.
"""

from __future__ import annotations

from src._framework import BaseTool, ToolCollection, ToolResult


class ToolRegistry:
    """Registry for research tools.

    Wraps OpenManus ToolCollection, providing a convenient way to
    build tool sets for different agents.

    Usage:
        registry = ToolRegistry()
        registry.register(WebSearchTool())
        searcher_tools = registry.get_collection(["web_search", "arxiv_search"])
    """

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def get_collection(self, names: list[str]) -> ToolCollection:
        """Build a ToolCollection from tool names."""
        tools = []
        for name in names:
            t = self._tools.get(name)
            if t:
                tools.append(t)
        return ToolCollection(*tools)

    def get_all(self) -> ToolCollection:
        return ToolCollection(*self._tools.values())

    def list_names(self) -> list[str]:
        return list(self._tools.keys())


__all__ = ["BaseTool", "ToolResult", "ToolCollection", "ToolRegistry"]

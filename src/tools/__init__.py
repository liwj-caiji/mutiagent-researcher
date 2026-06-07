"""Tools package — research tools based on OpenManus BaseTool.

Imports are lazy to avoid triggering OpenManus config loading when
only a submodule is imported (e.g. by MCP server entry points).
"""

__all__ = [
    "ToolRegistry",
    "WebSearchTool",
    "ArxivSearchTool",
    "WikipediaSearchTool",
    "WebScraperTool",
]


def __getattr__(name: str):
    if name == "ToolRegistry":
        from src.tools.base import ToolRegistry as _tr
        globals()["ToolRegistry"] = _tr
        return _tr
    if name in ("WebSearchTool", "ArxivSearchTool", "WikipediaSearchTool", "WebScraperTool"):
        import importlib
        mod = importlib.import_module("src.tools.search.tools")
        attr = getattr(mod, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Vendored from OpenManus (https://github.com/FoundationAgents/OpenManus) under MIT License.

Contains the core agent framework used by this project:
- Tool abstractions (BaseTool, ToolResult, ToolCollection, Terminate)
- Agent runtime (BaseAgent, ReActAgent, ToolCallAgent)
- Message/State schemas (Message, Memory, AgentState)
- MCP client integration (MCPClients)
"""

from .schema import AgentState, Function, Memory, Message, Role, ToolCall, ToolChoice
from .tool_base import BaseTool, ToolFailure, ToolResult
from .terminate import Terminate
from .tool_collection import ToolCollection
from .agent_base import BaseAgent
from .react import ReActAgent
from .toolcall import ToolCallAgent
from .mcp import MCPClientTool, MCPClients

__all__ = [
    "AgentState",
    "BaseAgent",
    "BaseTool",
    "Function",
    "MCPClients",
    "MCPClientTool",
    "Memory",
    "Message",
    "ReActAgent",
    "Role",
    "Terminate",
    "ToolCall",
    "ToolCallAgent",
    "ToolChoice",
    "ToolCollection",
    "ToolFailure",
    "ToolResult",
]

"""MCPManager — manages MCP server subprocess lifecycle and tool collection creation."""
from __future__ import annotations

from typing import Any

from loguru import logger

from src._framework import MCPClients, Terminate, ToolCollection


class MCPManager:
    """Manages MCP server process lifecycle.

    Creates MCPClients connections for agents and provides
    tool collections that combine MCP tools with local Terminate.
    """

    def __init__(self, config: dict[str, Any]):
        self._config = config
        self._clients: list[MCPClients] = []

    @property
    def enabled(self) -> bool:
        return bool(self._config.get("enabled", False))

    async def create_tool_collection(
        self,
        server_names: list[str],
        local_tools: list | None = None,
    ) -> ToolCollection:
        """Create a ToolCollection with MCP tools + local tools.

        If MCP is disabled, returns a collection with only local tools.
        Each server is connected via its own MCPClients (stdlib transport).
        """
        tc = ToolCollection(*(local_tools or []))

        if not self.enabled or not server_names:
            return tc

        servers_cfg = self._config.get("servers", {})
        for name in server_names:
            cfg = servers_cfg.get(name)
            if not cfg:
                logger.warning(f"MCP server '{name}' not found in config, skipping")
                continue

            command = cfg.get("command", "uv")
            args = cfg.get("args", [])
            logger.info(f"Connecting to MCP server '{name}': {command} {' '.join(args)}")

            try:
                clients = MCPClients()
                await clients.connect_stdio(command=command, args=args, server_id=name)
                self._clients.append(clients)

                for tool in clients.tools:
                    tc.add_tool(tool)

                logger.info(f"Connected to '{name}' — tools: {[t.name for t in clients.tools]}")
            except Exception as e:
                logger.error(f"Failed to connect to MCP server '{name}': {e}")

        return tc

    async def disconnect_all(self) -> None:
        """Disconnect all MCP server connections."""
        for clients in self._clients:
            try:
                await clients.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting MCP client: {e}")
        self._clients.clear()
        logger.info("All MCP connections closed")

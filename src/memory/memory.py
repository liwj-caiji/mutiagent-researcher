"""Memory system — short-term conversation history and long-term persistent storage."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from langchain_core.messages import BaseMessage


class Memory:
    """Short-term memory: maintains conversation history for a single agent.

    Features:
        - Message list with configurable max_messages
        - FIFO eviction when limit exceeded
        - Token count estimation
        - Serialization for checkpoints
    """

    def __init__(self, max_messages: int = 50):
        self.max_messages = max_messages
        self._messages: list[BaseMessage] = []

    def add_message(self, message: BaseMessage) -> None:
        """Add a message, evicting oldest if over limit."""
        self._messages.append(message)
        if len(self._messages) > self.max_messages:
            # Keep system message if present, evict oldest non-system
            for i, msg in enumerate(self._messages):
                if msg.type != "system":
                    self._messages.pop(i)
                    break

    def get_messages(self) -> list[BaseMessage]:
        """Return all messages in order."""
        return list(self._messages)

    def estimate_tokens(self) -> int:
        """Rough token count estimate (4 chars ≈ 1 token)."""
        total = 0
        for msg in self._messages:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            total += len(content) // 4
        return total

    def clear(self) -> None:
        """Reset memory."""
        self._messages = []

    def to_dict(self) -> list[dict]:
        """Serialize messages to dicts for checkpointing."""
        return [
            {"type": m.type, "content": m.content if isinstance(m.content, str) else str(m.content)}
            for m in self._messages
        ]

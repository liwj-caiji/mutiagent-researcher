"""LLM configuration for multi-provider support.

Each agent can have independent model configuration including provider, model name,
API key, base URL, temperature, etc.

Client creation is handled by LLMProvider in src/agents/llm_adapter.py.
"""

from dataclasses import dataclass
from typing import Literal

Provider = Literal["anthropic", "openai", "deepseek", "ollama"]


@dataclass
class AgentLLMConfig:
    """Per-agent LLM configuration."""

    provider: Provider
    model: str
    api_key: str = ""
    base_url: str | None = None
    temperature: float = 0.5
    max_tokens: int = 4096
    top_p: float = 1.0

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }

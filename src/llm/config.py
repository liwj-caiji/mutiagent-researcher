"""LLM configuration and factory for multi-provider support.

Each agent can have independent model configuration including provider, model name,
API key, base URL, temperature, etc.
"""

from dataclasses import dataclass, field
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


def create_llm(config: AgentLLMConfig):
    """Create a LangChain ChatModel from AgentLLMConfig.

    Supports: anthropic (Claude), openai (GPT), deepseek, ollama (local).
    """
    if config.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs = dict(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
        )
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatAnthropic(**kwargs)

    if config.provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs = dict(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
        )
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOpenAI(**kwargs)

    if config.provider == "deepseek":
        from langchain_openai import ChatOpenAI

        kwargs = dict(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            base_url=config.base_url or "https://api.deepseek.com",
        )
        if config.api_key:
            kwargs["api_key"] = config.api_key
        return ChatOpenAI(**kwargs)

    if config.provider == "ollama":
        from langchain_community.chat_models import ChatOllama

        kwargs = dict(
            model=config.model,
            temperature=config.temperature,
        )
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOllama(**kwargs)

    raise ValueError(f"Unsupported provider: {config.provider}")

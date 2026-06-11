"""LLM adapter — bridges OpenManus LLM interface to native provider SDKs.

OpenManus's ToolCallAgent expects `llm.ask_tool()` returning OpenAI-format response.
This adapter calls each provider's native SDK directly (no LangChain),
converting OpenManus message/tool formats to the provider's format and back.

Supports: Anthropic (Claude), OpenAI (GPT), DeepSeek, Ollama (local).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import httpx

from src._framework import Function as OM_Function
from src._framework import Message as OM_Message
from src._framework import ToolCall as OM_ToolCall
from src.llm.config import AgentLLMConfig

from loguru import logger


@dataclass
class ToolCallResult:
    """Mimics openai.types.chat.ChatCompletionMessage for tool responses."""
    content: str | None = None
    tool_calls: list[Any] | None = None


class LLMProvider:
    """Wraps native provider SDKs to provide OpenManus-compatible LLM interface.

    Creates the appropriate async client based on provider at init time.
    The key method is `ask_tool()` which matches OpenManus's LLM.ask_tool() signature.
    """

    def __init__(self, llm_config: AgentLLMConfig):
        self._config = llm_config
        self.model = llm_config.model
        self.provider = llm_config.provider
        self._client = self._create_client(llm_config)

    # ── Client creation ──────────────────────────────────────────────────

    def _create_client(self, config: AgentLLMConfig):
        """Create the appropriate native SDK client based on provider."""
        if config.provider == "anthropic":
            import anthropic
            kwargs = {}
            api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY", "")
            if api_key:
                kwargs["api_key"] = api_key
            if config.base_url:
                kwargs["base_url"] = config.base_url
            return anthropic.AsyncAnthropic(**kwargs)

        if config.provider in ("openai", "deepseek"):
            import openai
            base_url = config.base_url
            if config.provider == "deepseek" and not base_url:
                base_url = "https://api.deepseek.com"

            api_key = config.api_key
            if not api_key:
                key_env = f"{config.provider.upper()}_API_KEY"
                api_key = os.getenv(key_env) or os.getenv("OPENAI_API_KEY", "")

            return openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

        if config.provider == "ollama":
            # Ollama has a simple REST API, use httpx directly
            return httpx.AsyncClient(
                base_url=config.base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                timeout=httpx.Timeout(config.max_tokens or 300),
            )

        raise ValueError(f"Unsupported provider: {config.provider}")

    # ── Main entry point ─────────────────────────────────────────────────

    async def ask_tool(
        self,
        messages: list[OM_Message],
        system_msgs: list[OM_Message] | None = None,
        timeout: int = 300,
        tools: list[dict] | None = None,
        tool_choice: Any = None,
        temperature: float | None = None,
        **kwargs,
    ) -> ToolCallResult | None:
        """Send a tool-enabled request to the LLM.

        Dispatches to provider-specific implementation based on self.provider.

        Returns:
            ToolCallResult with .content and .tool_calls, or None on error.
        """
        if not self._should_use_tools(tool_choice):
            tools = None

        temp = temperature if temperature is not None else self._config.temperature

        try:
            if self.provider == "anthropic":
                return await self._ask_anthropic(messages, system_msgs, tools, tool_choice, temp, timeout)
            elif self.provider in ("openai", "deepseek"):
                return await self._ask_openai(messages, system_msgs, tools, tool_choice, temp, timeout)
            elif self.provider == "ollama":
                return await self._ask_ollama(messages, system_msgs, tools, tool_choice, temp, timeout)
        except Exception:
            logger.exception(
                "LLM call failed (provider=%s, model=%s, messages_count=%d)",
                self.provider,
                self.model,
                len(messages),
            )
            raise

    # ── Anthropic ────────────────────────────────────────────────────────

    async def _ask_anthropic(
        self,
        messages: list[OM_Message],
        system_msgs: list[OM_Message] | None,
        tools: list[dict] | None,
        tool_choice: Any,
        temperature: float,
        timeout: int,
    ) -> ToolCallResult:
        anthropic_messages = self._to_anthropic_messages(messages)

        system_text = None
        if system_msgs:
            parts = [sm.content for sm in system_msgs if sm.content]
            if parts:
                system_text = "\n".join(parts)

        anthropic_tools = self._to_anthropic_tools(tools) if tools else None
        tc = self._to_anthropic_tool_choice(tool_choice) if tools else None

        request_kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=anthropic_messages,
            max_tokens=self._config.max_tokens,
        )
        if system_text:
            request_kwargs["system"] = system_text
        if anthropic_tools:
            request_kwargs["tools"] = anthropic_tools
            if tc is not None:
                request_kwargs["tool_choice"] = tc
        if temperature >= 0:
            request_kwargs["temperature"] = temperature

        response = await self._client.messages.create(timeout=timeout, **request_kwargs)

        content_text = ""
        tool_calls: list[OM_ToolCall] = []
        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tool_calls.append(OM_ToolCall(
                    id=block.id,
                    type="function",
                    function=OM_Function(
                        name=block.name,
                        arguments=json.dumps(block.input, ensure_ascii=False),
                    ),
                ))

        return ToolCallResult(
            content=content_text.strip() or None,
            tool_calls=tool_calls if tool_calls else None,
        )

    # ── OpenAI / DeepSeek ─────────────────────────────────────────────────

    async def _ask_openai(
        self,
        messages: list[OM_Message],
        system_msgs: list[OM_Message] | None,
        tools: list[dict] | None,
        tool_choice: Any,
        temperature: float,
        timeout: int,
    ) -> ToolCallResult:
        openai_messages = self._to_openai_messages(messages, system_msgs)

        request_kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=openai_messages,
            max_tokens=self._config.max_tokens,
            temperature=temperature,
        )
        if tools:
            request_kwargs["tools"] = tools
            tc = self._convert_tool_choice(tool_choice)
            if tc:
                request_kwargs["tool_choice"] = tc

        response = await self._client.chat.completions.create(timeout=timeout, **request_kwargs)
        choice = response.choices[0]
        msg = choice.message

        tool_calls: list[OM_ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(OM_ToolCall(
                    id=tc.id,
                    type="function",
                    function=OM_Function(
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    ),
                ))

        return ToolCallResult(
            content=msg.content or None,
            tool_calls=tool_calls if tool_calls else None,
        )

    # ── Ollama ────────────────────────────────────────────────────────────

    async def _ask_ollama(
        self,
        messages: list[OM_Message],
        system_msgs: list[OM_Message] | None,
        tools: list[dict] | None,
        tool_choice: Any,
        temperature: float,
        timeout: int,
    ) -> ToolCallResult:
        openai_messages = self._to_openai_messages(messages, system_msgs)

        body: dict[str, Any] = dict(
            model=self.model,
            messages=openai_messages,
            stream=False,
            options={"temperature": temperature},
        )
        if tools:
            body["tools"] = tools

        resp = await self._client.post("/api/chat", json=body)
        resp.raise_for_status()
        data = resp.json()
        msg = data.get("message", {})

        content = msg.get("content", "")
        tool_calls: list[OM_ToolCall] = []
        for tc in msg.get("tool_calls", []) or []:
            fn = tc.get("function", {})
            tool_calls.append(OM_ToolCall(
                id=tc.get("id", ""),
                type="function",
                function=OM_Function(
                    name=fn.get("name", ""),
                    arguments=json.dumps(fn.get("arguments", {})) if isinstance(fn.get("arguments"), dict) else str(fn.get("arguments", "")),
                ),
            ))

        return ToolCallResult(
            content=content or None,
            tool_calls=tool_calls if tool_calls else None,
        )

    # ── Message conversion ────────────────────────────────────────────────

    @staticmethod
    def _to_openai_messages(
        messages: list[OM_Message],
        system_msgs: list[OM_Message] | None,
    ) -> list[dict]:
        """Convert OpenManus messages to OpenAI-compatible message dicts."""
        result: list[dict] = []

        if system_msgs:
            for sm in system_msgs:
                if sm.content:
                    result.append({"role": "system", "content": sm.content})

        for om in messages:
            role = om.role
            content = om.content or ""

            if role == "user":
                result.append({"role": "user", "content": content})
            elif role == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": content or None}
                if om.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in om.tool_calls
                    ]
                result.append(msg)
            elif role == "tool":
                result.append({
                    "role": "tool",
                    "tool_call_id": om.tool_call_id or "",
                    "content": content,
                })
            elif role == "system":
                result.append({"role": "system", "content": content})

        return result

    @staticmethod
    def _to_anthropic_messages(messages: list[OM_Message]) -> list[dict]:
        """Convert OpenManus messages to Anthropic message format.

        Key differences from OpenAI:
        - No "tool" role — tool results go in user messages with tool_result blocks.
        - Assistant tool calls are content blocks (tool_use), not a separate array.
        """
        result: list[dict] = []

        for om in messages:
            role = om.role
            content = om.content or ""

            if role == "user":
                result.append({"role": "user", "content": content})
            elif role == "assistant":
                if om.tool_calls:
                    blocks: list[dict] = []
                    if content:
                        blocks.append({"type": "text", "text": content})
                    for tc in om.tool_calls:
                        blocks.append({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.function.name,
                            "input": LLMProvider._parse_json_safe(tc.function.arguments),
                        })
                    result.append({"role": "assistant", "content": blocks})
                else:
                    result.append({"role": "assistant", "content": content})
            elif role == "tool":
                # Anthropic: tool results go in user messages
                result.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": om.tool_call_id or "",
                        "content": content,
                    }],
                })
            elif role == "system":
                # system in conversation → user message (shouldn't normally happen)
                result.append({"role": "user", "content": f"[System]: {content}"})

        return result

    # ── Tool format conversion ────────────────────────────────────────────

    @staticmethod
    def _to_anthropic_tools(tools: list[dict]) -> list[dict]:
        """Convert OpenAI function-calling tools to Anthropic tool format.

        OpenAI:  {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
        Anthropic: {"name": ..., "description": ..., "input_schema": ...}
        """
        result = []
        for tool in tools:
            func = tool.get("function", {})
            params = func.get("parameters", {})
            if not params:
                params = {"type": "object", "properties": {}}
            result.append({
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": params,
            })
        return result

    @staticmethod
    def _to_anthropic_tool_choice(tool_choice: Any) -> dict | None:
        """Convert OpenAI tool_choice to Anthropic tool_choice."""
        if tool_choice is None:
            return None
        choice_str = str(tool_choice)
        if hasattr(tool_choice, "value"):
            choice_str = tool_choice.value
        if choice_str in ("required", "any"):
            return {"type": "any"}
        # "auto" or anything else → Anthropic default (auto when tools present)
        return {"type": "auto"}

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _should_use_tools(tool_choice: Any) -> bool:
        if tool_choice is None:
            return True
        choice_str = str(tool_choice)
        if hasattr(tool_choice, "value"):
            choice_str = tool_choice.value
        return choice_str not in ("none",)

    @staticmethod
    def _convert_tool_choice(tool_choice: Any) -> str:
        """Convert OpenManus ToolChoice to OpenAI/LangChain format string."""
        choice_str = str(tool_choice)
        if hasattr(tool_choice, "value"):
            choice_str = tool_choice.value
        mapping = {"auto": "auto", "required": "required", "none": "none", "any": "required"}
        return mapping.get(choice_str, "auto")

    @staticmethod
    def _parse_json_safe(text: str) -> dict:
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return {}

    @staticmethod
    def _to_json(obj: dict) -> str:
        return json.dumps(obj, ensure_ascii=False)

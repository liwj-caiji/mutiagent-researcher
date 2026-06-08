"""LLM adapter — bridges OpenManus LLM interface to LangChain multi-provider ChatModels.

OpenManus's ToolCallAgent expects `llm.ask_tool()` returning OpenAI-format response.
This adapter wraps LangChain ChatModels to provide the same interface, supporting
Anthropic, OpenAI, DeepSeek, and Ollama providers natively.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src._framework import Function as OM_Function
from src._framework import Message as OM_Message
from src._framework import ToolCall as OM_ToolCall
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.language_models import BaseChatModel

from src.llm.config import AgentLLMConfig, create_llm

logger = logging.getLogger(__name__)


@dataclass
class ToolCallResult:
    """Mimics openai.types.chat.ChatCompletionMessage for tool responses."""
    content: str | None = None
    tool_calls: list[Any] | None = None


class LLMAdapter:
    """Wraps LangChain ChatModel to provide OpenManus-compatible LLM interface.

    The key method is `ask_tool()` which matches OpenManus's LLM.ask_tool() signature.
    """

    def __init__(self, llm_config: AgentLLMConfig):
        self._chat_model: BaseChatModel = create_llm(llm_config)
        self.model = llm_config.model

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

        Converts OpenManus Message format to LangChain format, invokes the model,
        and returns a ToolCallResult mimicking OpenAI's ChatCompletionMessage.

        Args:
            messages: Conversation messages in OpenManus format.
            system_msgs: Optional system messages.
            timeout: Request timeout (not used by LangChain, kept for compatibility).
            tools: Tool definitions in OpenAI function calling format.
            tool_choice: Tool choice strategy (auto/none/required).
            temperature: Override default temperature.
            **kwargs: Additional arguments.

        Returns:
            ToolCallResult with .content and .tool_calls, or None on error.
        """
        lc_messages: list[BaseMessage] = []

        # Add system messages
        if system_msgs:
            for sm in system_msgs:
                if sm.content:
                    lc_messages.append(SystemMessage(content=sm.content))

        # Convert conversation messages
        for om_msg in messages:
            role = om_msg.role
            content = om_msg.content or ""

            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                ai_msg = AIMessage(content=content)
                # Convert tool_calls if present
                if om_msg.tool_calls:
                    lc_tool_calls = []
                    for tc in om_msg.tool_calls:
                        lc_tool_calls.append({
                            "id": tc.id,
                            "name": tc.function.name,
                            "args": self._parse_json_safe(tc.function.arguments),
                        })
                    ai_msg.tool_calls = lc_tool_calls
                    # Also set additional_kwargs for Anthropic compatibility
                    ai_msg.additional_kwargs = {"tool_calls": lc_tool_calls}
                lc_messages.append(ai_msg)
            elif role == "tool":
                lc_messages.append(ToolMessage(
                    content=content,
                    tool_call_id=om_msg.tool_call_id or "",
                ))
            # system role in middle of conversation (unusual but handle)
            elif role == "system":
                lc_messages.append(SystemMessage(content=content))

        try:
            if tools and self._should_use_tools(tool_choice):
                # Bind tools and invoke
                llm_with_tools = self._chat_model.bind_tools(
                    self._convert_tools(tools),
                    tool_choice=self._convert_tool_choice(tool_choice),
                )
                response: AIMessage = await llm_with_tools.ainvoke(lc_messages)
            else:
                response: AIMessage = await self._chat_model.ainvoke(lc_messages)

            if response is None:
                return None

            # Convert LangChain AIMessage back to ToolCallResult
            tool_calls = []
            if response.tool_calls:
                for tc in response.tool_calls:
                    tool_calls.append(OM_ToolCall(
                        id=tc.get("id", ""),
                        type="function",
                        function=OM_Function(
                            name=tc["name"],
                            arguments=self._to_json(tc.get("args", {})),
                        ),
                    ))

            return ToolCallResult(
                content=response.content if isinstance(response.content, str) else str(response.content or ""),
                tool_calls=tool_calls if tool_calls else None,
            )
        except Exception:
            logger.exception(
                "LLM call failed (provider=%s, model=%s, messages_count=%d)",
                getattr(self._chat_model, 'provider', 'unknown'),
                self.model,
                len(lc_messages),
            )
            raise

    def _should_use_tools(self, tool_choice: Any) -> bool:
        if tool_choice is None:
            return True
        choice_str = str(tool_choice)
        if hasattr(tool_choice, 'value'):
            choice_str = tool_choice.value
        return choice_str not in ("none",)

    @staticmethod
    def _convert_tool_choice(tool_choice: Any) -> str:
        """Convert OpenManus ToolChoice to LangChain format."""
        choice_str = str(tool_choice)
        if hasattr(tool_choice, 'value'):
            choice_str = tool_choice.value
        mapping = {"auto": "auto", "required": "any", "none": "none"}
        return mapping.get(choice_str, "auto")

    @staticmethod
    def _convert_tools(openmanus_tools: list[dict]) -> list[dict]:
        """Convert OpenManus tool format to LangChain tool format."""
        lc_tools = []
        for tool in openmanus_tools:
            func = tool.get("function", {})
            lc_tools.append({
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
            })
        return lc_tools

    @staticmethod
    def _parse_json_safe(text: str) -> dict:
        import json
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return {}

    @staticmethod
    def _to_json(obj: dict) -> str:
        import json
        return json.dumps(obj, ensure_ascii=False)

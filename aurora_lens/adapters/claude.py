"""Claude adapter — Anthropic API integration (optional extra).

Install: pip install aurora-lens[claude]
"""

from __future__ import annotations

import logging

from .base import LLMAdapter, AdapterResponse

_log = logging.getLogger(__name__)


class ClaudeAdapter(LLMAdapter):
    """Adapter for Anthropic's Claude API. Transport-only.

    Requires: pip install aurora-lens[claude]
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1024,
    ):
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError(
                "ClaudeAdapter requires the anthropic package. "
                "Install it with: pip install aurora-lens[claude]"
            ) from None

        self._client = _anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    async def generate(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> AdapterResponse:
        """Send messages upstream. System messages concatenated into Anthropic system=; rest as messages."""
        system_parts: list[str] = []
        chat_messages: list[dict[str, str]] = []

        for msg in messages:
            role = (msg.get("role") or "").lower()
            content = (msg.get("content") or "").strip()
            if role == "system":
                if content:
                    system_parts.append(content)
            else:
                chat_messages.append({"role": role, "content": content})

        max_tokens = kwargs.get("max_tokens", self._max_tokens)

        create_kwargs: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": chat_messages,
        }
        if system_parts:
            create_kwargs["system"] = "\n\n".join(system_parts)

        _log.info(
            "claude_adapter_generate_start",
            extra={"model": self._model, "msg_count": len(chat_messages)},
        )
        response = await self._client.messages.create(**create_kwargs)

        text = response.content[0].text if response.content else ""
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        _log.info(
            "claude_adapter_generate_done",
            extra={
                "model": response.model,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )

        return AdapterResponse(
            text=text,
            model=response.model,
            usage=usage,
        )

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ):
        """Stream completion from Claude. Yields (chunk_dict, content_delta) tuples."""
        system_parts: list[str] = []
        chat_messages: list[dict[str, str]] = []

        for msg in messages:
            role = (msg.get("role") or "").lower()
            content = (msg.get("content") or "").strip()
            if role == "system":
                if content:
                    system_parts.append(content)
            else:
                chat_messages.append({"role": role, "content": content})

        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        stream_kwargs: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": chat_messages,
        }
        if system_parts:
            stream_kwargs["system"] = "\n\n".join(system_parts)

        _log.info(
            "claude_adapter_stream_start",
            extra={"model": self._model, "msg_count": len(chat_messages)},
        )
        async with self._client.messages.stream(**stream_kwargs) as stream:
            async for text in stream.text_stream:
                chunk_dict = {"choices": [{"delta": {"content": text}, "index": 0}]}
                yield (chunk_dict, text)
        _log.info("claude_adapter_stream_done", extra={"model": self._model})

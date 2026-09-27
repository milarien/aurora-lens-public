"""Abstract LLM adapter interface.

Transport-only: adapters send already-constructed messages upstream and return raw text.
They do not know about PEF, system prompts, or injection. Lens builds the messages.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass
class AdapterResponse:
    """Response from an LLM adapter."""
    text: str
    model: str
    usage: dict | None = None


class LLMAdapter(ABC):
    """Abstract interface for LLM adapters."""

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> AdapterResponse:
        """
        Transport-only: send already-constructed chat messages upstream and return raw text.
        `messages` uses OpenAI-style shape: [{"role": "...", "content": "..."}]
        kwargs is passthrough for provider-specific options (temperature, max_tokens, etc).
        """
        ...

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> AsyncIterator[tuple[dict[str, object], str]]:
        """
        Stream completion from upstream. Yields (chunk_dict, content_delta) tuples.
        chunk_dict is OpenAI SSE format for forwarding; content_delta is text to accumulate.

        Default raises ``NotImplementedError`` — Lens/proxy fall back to buffered
        ``generate()``. Override in adapters that support true streaming; implementers
        who omit this override silently downgrade to non-streaming mode.
        """
        raise NotImplementedError(
            "Streaming not supported by this adapter; callers fall back to generate()"
        )
        yield  # make this an async generator so callers can async-for

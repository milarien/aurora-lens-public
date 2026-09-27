"""LangChain integration — use aurora-lens proxy as the LLM backend.

The proxy is OpenAI-compatible. Point LangChain's ChatOpenAI at it.

Usage:
    from aurora_lens.integrations.langchain import get_governed_chat_openai

    llm = get_governed_chat_openai(
        base_url="http://localhost:8081/v1",
        api_key="your-openai-key",  # or set OPENAI_API_KEY
        model="gpt-4",
    )
    response = llm.invoke("What is the capital of France?")

Requires: pip install aurora-lens[langchain]
"""

from __future__ import annotations


def get_governed_chat_openai(
    base_url: str = "http://localhost:8081/v1",
    api_key: str | None = None,
    model: str = "gpt-4",
    **kwargs,
):
    """Return a LangChain ChatOpenAI instance configured to use aurora-lens proxy.

    All requests go through the proxy; governance is applied transparently.
    Aurora-* headers in the response carry governance metadata.

    Args:
        base_url: Proxy URL (default: http://localhost:8081/v1)
        api_key: OpenAI API key (or set OPENAI_API_KEY). Proxy forwards to upstream.
        model: Model name (must match proxy config)
        **kwargs: Passed to ChatOpenAI (temperature, max_tokens, etc.)
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as err:
        raise ImportError(
            "LangChain integration requires langchain-openai. "
            "Install with: pip install aurora-lens[langchain]"
        ) from err

    import os
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    return ChatOpenAI(
        base_url=base_url.rstrip("/"),
        api_key=key,
        model=model,
        **kwargs,
    )

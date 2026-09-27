"""Direct upstream corpus review (document plane — no Lens proxy)."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Sequence

from aurora_lens.corpus.models import CorpusChunk

REVIEW_PROMPT_PREFIX = """\
Read the corpus below as the only authority source.

Produce four lists grounded in exact verbatim quotes from the corpus only:

1. Supported — claims explicitly stated and supported.
2. Undefined — terms used without definition.
3. Self-contradictory — mutually opposing passages.
4. Does not follow — implications not entailed by stated claims.

Rules:
- Each item: one verbatim quote from the corpus.
- Empty category: None found in text.
- Do not invent items. Do not critique style unless meaning is affected.

CORPUS:
"""

SYSTEM_PROMPT = (
    "Documentation review only. Quote the corpus verbatim. "
    "No outside knowledge. No actionable advice to third parties."
)


def select_chunks(
    chunks: Sequence[CorpusChunk],
    *,
    chunk_ordinals: set[int] | None = None,
    max_chars: int | None = None,
    preserve_order: bool = False,
) -> list[CorpusChunk]:
    """Filter by ordinal list and/or cumulative character budget."""
    ordered = list(chunks) if preserve_order else sorted(chunks, key=lambda c: c.ordinal)
    if chunk_ordinals is not None:
        ordered = [c for c in ordered if c.ordinal in chunk_ordinals]
    if not ordered:
        raise ValueError("no chunks selected")
    if max_chars is None:
        return ordered

    selected: list[CorpusChunk] = []
    total = 0
    for ch in ordered:
        add = len(ch.text) + (2 if selected else 0)
        if selected and total + add > max_chars:
            break
        if not selected and len(ch.text) > max_chars:
            selected.append(ch)
            break
        selected.append(ch)
        total += add
    return selected


def format_corpus_for_review(chunks: Sequence[CorpusChunk]) -> str:
    parts: list[str] = []
    for ch in chunks:
        header = ch.locator or f"chunk {ch.ordinal}"
        parts.append(f"### {header}\n{ch.text}")
    return "\n\n".join(parts)


def upstream_config_status() -> dict[str, str | bool | None]:
    """Non-throwing upstream readiness check for operator console preflight."""
    provider = os.environ.get("AURORA_LENS_UPSTREAM_PROVIDER", "").strip().lower()
    if not provider:
        if os.environ.get("ANTHROPIC_API_KEY", "").strip():
            provider = "anthropic"
        elif os.environ.get("OPENAI_API_KEY", "").strip():
            provider = "openai"
        else:
            return {
                "ok": False,
                "reason": "no upstream configured",
                "detail": "Set OPENAI_API_KEY or ANTHROPIC_API_KEY (same env vars as the proxy).",
                "provider": None,
                "model": None,
            }

    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        default_model = "claude-haiku-4-5-20251001"
    else:
        provider = "openai"
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        default_model = "gpt-4o-mini"

    if not api_key:
        return {
            "ok": False,
            "reason": "no API key",
            "detail": f"Missing API key for provider {provider!r}.",
            "provider": provider,
            "model": None,
        }

    model = os.environ.get("AURORA_LENS_UPSTREAM_MODEL", "").strip() or default_model
    return {
        "ok": True,
        "reason": None,
        "detail": None,
        "provider": provider,
        "model": model,
    }


def resolve_upstream() -> tuple[str, str, str]:
    status = upstream_config_status()
    if not status["ok"]:
        raise SystemExit(str(status["detail"] or status["reason"]))
    provider = str(status["provider"])
    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    return provider, api_key, str(status["model"])


async def run_corpus_review(
    corpus_text: str,
    *,
    provider: str,
    api_key: str,
    model: str,
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": REVIEW_PROMPT_PREFIX + corpus_text},
    ]
    if provider == "anthropic":
        from aurora_lens.adapters.claude import ClaudeAdapter

        adapter = ClaudeAdapter(api_key=api_key, model=model, max_tokens=8192)
    else:
        from aurora_lens.adapters.openai import OpenAIUpstreamAdapter

        adapter = OpenAIUpstreamAdapter(
            api_key=api_key, model=model, max_tokens=8192, timeout_s=600.0
        )
    resp = await adapter.generate(messages)
    return resp.text


logger = logging.getLogger(__name__)


def safe_print(text: str) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            logger.debug("stdout reconfigure failed", exc_info=True)
    print(text)


def run_corpus_review_sync(corpus_text: str) -> str:
    provider, api_key, model = resolve_upstream()
    return asyncio.run(
        run_corpus_review(
            corpus_text,
            provider=provider,
            api_key=api_key,
            model=model,
        )
    )

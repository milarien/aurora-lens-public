"""Shared sentence / clause splitting for verification (finance and causal paths).

Uses a single regex so behavior stays consistent across Layer 1 finance scans,
speculative causal enumeration, and similar call sites.
"""

from __future__ import annotations

import re

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences_at_punctuation(text: str) -> list[str]:
    """Split *text* on sentence-ending punctuation followed by whitespace.

    Empty segments are dropped; each segment is stripped.
    """
    if not text or not text.strip():
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]

"""Narrow C1 seam: calendar month + day co-occurring with "last week" in one blob.

Used by verify (committed PEF literals) and RAG admission (context claims).
This is intentionally not a general temporal reasoner — extend with care.
"""

from __future__ import annotations

import re

_CALENDAR_MONTH_DAY_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\b",
    re.IGNORECASE,
)


def incompatible_arrival_time_joined_text(joined_lower: str) -> bool:
    """True when the same lowercased string contains both a calendar date and *last week*."""
    if "last week" not in joined_lower:
        return False
    return bool(_CALENDAR_MONTH_DAY_RE.search(joined_lower))


def incompatible_arrival_time_literal_strings(literal_strings: list[str]) -> bool:
    """Join literal fragments (e.g. per-relationship or per-claim) and apply the same test."""
    joined = " ".join(literal_strings).lower()
    return incompatible_arrival_time_joined_text(joined)


def text_contains_calendar_month_day(text: str) -> bool:
    """Whether *text* contains a MonthName + day pattern (same regex as the C1 seam)."""
    return bool(_CALENDAR_MONTH_DAY_RE.search(text.lower()))


def rag_context_body_from_user_input(user_input: str | None) -> str | None:
    """Return the body after ``Context:`` … ``Question:`` (eval harness). Plain string ops only.

    Duplicates :func:`aurora_lens.lens.split_rag_context_question` logic to avoid
    ``checker`` importing ``lens`` (circular import). Kept in sync manually.
    """
    if user_input is None:
        return None
    s = user_input.strip()
    if len(s) < 20:
        return None
    if s[:8].lower() != "context:":
        return None
    rest = s[8:].lstrip()
    if not rest:
        return None
    low = rest.lower()
    sep = "\n\nquestion:"
    idx = low.find(sep)
    if idx < 0:
        return None
    context_body = rest[:idx].strip()
    tail = rest[idx:]
    cpos = tail.find(":")
    if cpos < 0 or not context_body:
        return None
    question_line = tail[cpos + 1 :].lstrip()
    if not question_line:
        return None
    return context_body


def entity_name_evident_in_text(text: str, entity_name: str) -> bool:
    """Whether *entity_name* (or its first word) appears in *text* (case-insensitive)."""
    en = entity_name.strip().lower()
    if not en:
        return False
    low = text.lower()
    if en in low:
        return True
    first = en.split()[0]
    return len(first) >= 3 and first in low

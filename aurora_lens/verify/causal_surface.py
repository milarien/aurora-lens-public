"""Shared causal-enumeration surface patterns for :class:`~aurora_lens.verify.checker.Checker`.

A future extraction-time INCLUDE patch would import these patterns; they are not
duplicated in ``checker.py`` so semantics stay in one place.
"""

from __future__ import annotations

import re

# Causal-scaffold subjects: generic, anonymous subject phrases (INCLUDE/INCLUDES only).
CAUSAL_SCAFFOLD_RE = re.compile(
    r"\b(?:"
    r"common\s+(?:factor|cause|reason|driver|issue)s?|"
    r"(?:possible|potential|typical|frequent|general)\s+(?:factor|cause|reason|driver)s?|"
    r"(?:factor|cause|reason|driver)s?\s+(?:that\s+)?(?:can|could|may|might)"
    r"\s+(?:\w+\s+){0,2}(?:lead|contribute|result|cause)"
    r")\b",
    re.IGNORECASE,
)

# Speculative causal enumeration — "factors such as / like / including", etc.
CAUSAL_FACTORS_TEXT_RE = re.compile(
    r"(?:"
    r"\b(?:factor|cause|reason|driver)s?,?\s+(?:such\s+as|including|like)\b"
    r"|"
    r"\b(?:can\s+be\s+)?attributed\s+to\s+(?:the\s+)?(?:factor|cause|reason|driver)s?\s+"
    r"(?:such\s+as|including|like)\b"
    r"|"
    r"\bdue\s+to\s+(?:factor|cause|reason|driver)s?\s+(?:such\s+as|including|like)\b"
    r")",
    re.IGNORECASE,
)

# Open-domain hedging after a "why" turn: modal + "due to" + ungrounded quantifier
# (e.g. "could be due to various personal or professional reasons") — not a concrete
# admitted cause list from context.
CAUSAL_HEDGED_OPEN_DOMAIN_RE = re.compile(
    r"\b(?:could|may|might)\s+be\s+due\s+to\s+"
    r"(?:various|several|many|multiple|different|a\s+number\s+of)\b",
    re.IGNORECASE,
)

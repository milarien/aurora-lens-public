"""epistemic_normalisation — deterministic surface pass for clean PASS outcomes.

Runs **after** checker + governance have admitted the draft (no flags, PASS).
Removes rhetorical certainty, narrow advisory openers, and stock conclusion
prefixes. Does **not** add probabilistic hedging, new facts, or binding changes.

``apply_epistemic_normalisation`` is intentionally conservative; any expansion
must stay lexical and test-locked (see ``tests/test_epistemic_normalisation.py``).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aurora_lens.pef.state import PEFState

# Unsupported certainty / emphasis markers (word-boundary; removed, not replaced).
_CERTAINTY_MARKERS = re.compile(
    r"\b(?:"
    r"definitely|certainly|clearly|obviously|absolutely|undeniably|"
    r"unquestionably|invariably|undoubtedly"
    r")\b",
    re.IGNORECASE,
)

# Fixed phrases (no substitution with hedging language).
_FIXED_PHRASES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bwithout\s+a\s+doubt\b", re.IGNORECASE), ""),
    (re.compile(r"\bno\s+doubt\b", re.IGNORECASE), ""),
]

# Stock conclusion / pivot prefixes at paragraph or document start only.
_CONCLUSION_PREFIX = re.compile(
    r"(?:(?<=^)|(?<=\n))\s*"
    r"(?:Therefore|Thus|Hence|In conclusion|To summarize|In summary)\s*,?\s+",
    re.IGNORECASE | re.MULTILINE,
)

# Stock meta-advisory / process framing (narrow). Deliberately excludes bare
# ``You should …`` / ``You must …`` — those strings appear in governed benign
# healthcare continuations and must not be stripped as "filler".
_ADVISORY_LINE_START = re.compile(
    r"^(?:I recommend that you|We recommend that you|"
    r"It is important to note that|Please note that|Please ensure that|Make sure to)\s+",
    re.IGNORECASE | re.MULTILINE,
)
_ADVISORY_AFTER_SENTENCE = re.compile(
    r"(?<=[.!?])\s+"
    r"(?:I recommend that you|We recommend that you|"
    r"It is important to note that|Please note that|Please ensure that|Make sure to)\s+",
    re.IGNORECASE,
)

_WS_COLLAPSE = re.compile(r" {2,}")


def apply_epistemic_normalisation(text: str, _pef: PEFState | None = None) -> str:
    """Return surface-normalised text. ``_pef`` is reserved for future structural gates."""
    s = text or ""
    if not s.strip():
        return s
    s = _ADVISORY_LINE_START.sub("", s)
    s = _ADVISORY_AFTER_SENTENCE.sub(" ", s)
    s = _CONCLUSION_PREFIX.sub("", s)
    for pat, repl in _FIXED_PHRASES:
        s = pat.sub(repl, s)
    s = _CERTAINTY_MARKERS.sub("", s)
    s = _WS_COLLAPSE.sub(" ", s)
    s = re.sub(r" *\n *", "\n", s)
    return s.strip()

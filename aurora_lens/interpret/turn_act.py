"""Typed classification of user-turn intent (thin layer over candidate detectors).

Governance and PEF code should prefer ``TurnAct`` / ``classify_turn_act`` over ad-hoc
regex checks where possible. Regex remains an implementation detail inside
classifiers and in modules like ``revision_gate`` — it is not the final authority;
this enum is the vocabulary the rest of the stack should query.
"""

from __future__ import annotations

from enum import Enum

from aurora_lens.interpret.revision_gate import explicit_correction_intent


class TurnAct(str, Enum):
    """High-level act for the current user message (user plane)."""

    ASSERT = "assert"
    REVISE = "revise"
    CLARIFY = "clarify"
    QUERY = "query"


_QUERY_START_TOKENS: frozenset[str] = frozenset(
    {
        "what",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "why",
        "how",
        "which",
        "is",
        "are",
        "was",
        "were",
        "am",
        "do",
        "does",
        "did",
        "can",
        "could",
        "would",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "will",
    }
)
_QUERY_START_BIGRAMS: frozenset[tuple[str, str]] = frozenset(
    {
        ("have", "you"),
        ("has", "he"),
        ("has", "she"),
        ("has", "it"),
        ("have", "they"),
    }
)


# Meta-questions about what information is still needed (used when pending
# clarification is unresolved; see ``lens`` binding resolution). Moved from
# ``lens.py`` for typed routing — kept as candidate detectors only.
_CLARIFICATION_INQUIRY_STEMS: frozenset[str] = frozenset({
    "what do you need",
    "what do you still need",
    "what information do you need",
    "what information do you still need",
    "what should i clarify",
    "what needs to be clarified",
    "what is ambiguous",
    "what is unclear",
    "what are you asking",
    "what did you need",
    "what more do you need",
    "what else do you need",
    "what information is needed",
    "what's unclear",
    "what's ambiguous",
    "what info do you need",
    "what details do you need",
    "what do i need to clarify",
    "what was ambiguous",
})

def _scan_word_tokens(text: str) -> tuple[str, ...]:
    out: list[str] = []
    cur: list[str] = []
    for ch in (text or "").lower():
        if ch.isalnum():
            cur.append(ch)
            continue
        if cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return tuple(out)


def _contains_phrase(tokens: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    if not phrase or len(tokens) < len(phrase):
        return False
    n = len(phrase)
    for i in range(0, len(tokens) - n + 1):
        if tokens[i : i + n] == phrase:
            return True
    return False


def _looks_like_clarification_inquiry_meta(user_text: str) -> bool:
    """Candidate: user asks what to clarify / what is still needed (ambiguous slot)."""
    normalized = user_text.strip().rstrip("?!.").strip().lower()
    if normalized in _CLARIFICATION_INQUIRY_STEMS:
        return True
    toks = _scan_word_tokens(normalized)
    if not toks or toks[0] != "what":
        return False
    window = toks[1:9]
    stems = ("need", "clarif", "ambigu", "unclear", "ask", "miss")
    return any(any(tok.startswith(stem) for stem in stems) for tok in window)


def _looks_like_clarify_signal(user_text: str) -> bool:
    """Candidate signal for meta-clarification phrasing (not a full REVISE act)."""
    toks = _scan_word_tokens(user_text)
    if not toks:
        return False
    if _contains_phrase(toks, ("to", "clarify")):
        return True
    if _contains_phrase(toks, ("for", "clarification")):
        return True
    if _contains_phrase(toks, ("just", "to", "be", "clear")):
        return True
    if _contains_phrase(toks, ("what", "i", "mean", "is")):
        return True
    for phrase in (
        ("i", "mean", "that"),
        ("i", "mean", "this"),
        ("i", "mean", "when"),
        ("i", "mean", "the"),
    ):
        if _contains_phrase(toks, phrase):
            return True
    return False


def _candidate_clarify_act(user_text: str) -> bool:
    """True when text looks like a clarification act (inquiry or meta phrasing)."""
    return _looks_like_clarification_inquiry_meta(user_text) or _looks_like_clarify_signal(
        user_text
    )


def _looks_like_query(user_text: str) -> bool:
    """Candidate signal only — not authoritative without session context."""
    t = user_text.strip()
    if not t:
        return False
    if t.endswith("?"):
        return True
    toks = _scan_word_tokens(t)
    if not toks:
        return False
    if toks[0] in _QUERY_START_TOKENS:
        return True
    if len(toks) >= 2 and (toks[0], toks[1]) in _QUERY_START_BIGRAMS:
        return True
    return False


def classify_turn_act(user_text: str) -> TurnAct:
    """Classify *user_text* into a :class:`TurnAct`.

    First version: delegates **REVISE** to :func:`user_explicit_revision_act`
    (regex lives in ``revision_gate``). **CLARIFY** is checked **before** **QUERY**
    so phrases like "What should I clarify?" classify as clarification acts, not
    generic questions. **QUERY** / remaining **CLARIFY** use lightweight
    candidate patterns; everything else defaults to **ASSERT**.

    Callers should treat the result as the typed handle for routing; extend with
    session state (e.g. pending clarification) in a later cut.
    """
    if not user_text or not user_text.strip():
        return TurnAct.ASSERT

    if explicit_correction_intent(user_text):
        return TurnAct.REVISE

    if _candidate_clarify_act(user_text):
        return TurnAct.CLARIFY

    if _looks_like_query(user_text):
        return TurnAct.QUERY

    return TurnAct.ASSERT

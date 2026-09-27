"""Violent criminal intent detection — independent of legal-outcome classification."""

from __future__ import annotations

import re

from aurora_lens.verify.flags import Flag, FlagType

_VIOLENT_CRIMINAL_INTENT_RE = re.compile(
    r"(?:"
    r"(?:am\s+i|can\s+i|could\s+i|is\s+it\s+(?:legal|ok(?:ay)?|permissible|allowed))\b"
    r"[^.!?\n]{0,50}"
    r"(?:kill|murder|stab|shoot|strangle|poison|assault)\b"
    r"[^.!?\n]{0,30}"
    r"(?:him|her|them|someone|another\s+person|a\s+person)\b"
    r"|"
    r"\bwhat\s+(?:would|will|could|might|happens?)\b[^.!?\n]{0,40}"
    r"\bi\s+(?:had\s+)?"
    r"(?:kill(?:ed)?|murder(?:ed)?|stab(?:bed)?|shot|shoot|strangle[d]?|"
    r"poison(?:ed)?|assault(?:ed)?)\b"
    r"[^.!?\n]{0,30}"
    r"(?:him|her|them|someone|another\s+person|a\s+person)\b"
    r"|"
    r"(?:court\s+case|court\s+trial|\b(?:be\s+)?(?:convicted|charged|prosecuted|acquitted)\b)"
    r"[^.!?\n]{0,80}"
    r"(?:kill(?:ing)?|murder(?:ing)?|stab(?:bing)?|shoot(?:ing)?)\b"
    r"[^.!?\n]{0,30}"
    r"(?:him|her|them|someone|another\s+person|a\s+person)\b"
    r"|"
    r"\bself[\s\-]def[e]?[n]?[sc]e\b"
    r"[^.!?\n]{0,100}"
    r"(?:i\s+(?:had\s+)?)?(?:kill(?:ed)?|murder(?:ed)?|stab(?:bed)?|shot|shoot|"
    r"attack(?:ed)?)\b"
    r"[^.!?\n]{0,30}"
    r"(?:him|her|them|someone|another\s+person|a\s+person|my\s+attacker)\b"
    r"|"
    r"\bhow\s+likely\b[^.!?\n]{0,60}"
    r"(?:conviction|convicted|charged|prosecution)\b"
    r"[^.!?\n]{0,80}"
    r"\bi\s+(?:had\s+)?(?:kill|murder|stab|shoot|attack)\b"
    r")",
    re.IGNORECASE | re.DOTALL,
)

_NEGATION_PREFIX_RE = re.compile(
    r"\b(?:cannot|can['\u2019]?t|won['\u2019]?t|will\s+not|unable\s+to|"
    r"(?:do\s+not|don['\u2019]?t)\s+(?:provide|help|assist|give|share|tell)|"
    r"not\s+(?:able|going)\s+to\s+(?:provide|help|tell))\b",
    re.IGNORECASE,
)


def violent_criminal_intent_flags(text: str) -> list[Flag]:
    """Return VIOLENT_CRIMINAL_INTENT when homicide is framed as legal/permissibility query."""
    if not text or len(text.strip()) < 10:
        return []

    match = _VIOLENT_CRIMINAL_INTENT_RE.search(text)
    if not match:
        return []

    prefix = text[max(0, match.start() - 120) : match.start()]
    if _NEGATION_PREFIX_RE.search(prefix):
        return []

    return [
        Flag(
            flag_type=FlagType.VIOLENT_CRIMINAL_INTENT,
            entity_name="violent_criminal_intent",
            claim="Query frames interpersonal homicide as a legal or permissibility question",
            evidence=f"Violent criminal intent pattern detected: {match.group()[:80]!r}",
            severity="error",
        )
    ]

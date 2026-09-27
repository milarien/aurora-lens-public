"""Deterministic lexical helpers for possession item phrases (matching vs quantity display).

``item_key`` and ``analyze_item_phrase`` consolidate noun-phrase normalization that
formerly duplicated naive ``endswith("s")`` logic across inventory, spaCy projection,
Lens replay, and possession mutations.

This module MUST NOT import from ``eval.*`` or ``interpret.*`` (cycle guard).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Number = Literal["singular", "plural", "unknown"]

_TRAILING_ITEM_PUNCT_RE = re.compile(r"[.?!,;:]+$")

# First-token determiners aligned with ``_strip_leading_determiners_np`` in
# ``aurora_lens/interpret/spacy_backend.py`` so ``item_key("the apples")`` matches
# ``item_key("apples")`` and spaCy projection stays stable.
_LEADING_NP_DETERMINERS: frozenset[str] = frozenset(
    {"the", "a", "an", "this", "that", "these", "those"}
)

# Plural surface (lowercase, last token) -> singular lemma (lowercase).
_IRREGULAR_PLURAL_TO_LEMMA: dict[str, str] = {
    "children": "child",
    "people": "person",
    "mice": "mouse",
    "men": "man",
    "women": "woman",
    "teeth": "tooth",
    "feet": "foot",
    "geese": "goose",
    "oxen": "ox",
}

# Singular lemma (lowercase) -> default plural surface (lowercase) for displays.
_IRREGULAR_LEMMA_TO_PLURAL: dict[str, str] = {
    lemma: plural for plural, lemma in _IRREGULAR_PLURAL_TO_LEMMA.items()
}


def normalize_possession_item_surface(item: str) -> str:
    """Strip outer whitespace and trailing sentence punctuation from item phrases.

    Shared with HAS object literals, GIVE/CONSUME tails, and inventory queries so
    ``apples``, ``apples.``, and ``apples...`` normalize the same way.
    """
    s = (item or "").strip()
    s = _TRAILING_ITEM_PUNCT_RE.sub("", s).strip()
    return s


def strip_leading_np_determiners(text: str) -> str:
    """Remove one leading NP determiner token (spaCy-aligned), if present."""
    s = text.strip()
    parts = s.split(None, 1)
    if parts and parts[0].lower() in _LEADING_NP_DETERMINERS:
        return parts[1].strip() if len(parts) > 1 else ""
    return s


def _cased_like(model_token: str, lower_target: str) -> str:
    """Shape ``lower_target`` to match ``model_token`` capitalization pattern."""
    if not model_token:
        return lower_target
    if model_token.isupper():
        return lower_target.upper()
    if model_token[0].isupper():
        # Title-case first grapheme only (handles "Alice" vs "APPLE").
        return lower_target[0].upper() + lower_target[1:] if lower_target else lower_target
    return lower_target


def _looks_plural_y_ies(low: str) -> bool:
    if not low.endswith("ies") or len(low) <= 4:
        return False
    return low[-4] not in "aeiou"


def _plural_surface_from_lemma(lower_lemma: str, casing_token: str) -> str:
    """English plural stem for displays when input was singular."""
    if lower_lemma in _IRREGULAR_LEMMA_TO_PLURAL:
        return _cased_like(casing_token, _IRREGULAR_LEMMA_TO_PLURAL[lower_lemma])

    # Consonant + y -> ...ies (party -> parties).
    if len(lower_lemma) >= 2 and lower_lemma.endswith("y"):
        prev = lower_lemma[-2]
        if prev not in "aeiou":
            plural_low = lower_lemma[:-1] + "ies"
            return _cased_like(casing_token, plural_low)

    # Sibilant / consonant-cluster stems that normally take ``-es``.
    nounish = lower_lemma
    if nounish.endswith(("s", "x", "z", "ch", "sh", "o")):
        plural_low = nounish + "es"
        return _cased_like(casing_token, plural_low)

    plural_low = nounish + "s"
    return _cased_like(casing_token, plural_low)


def _analyze_last_token(orig_tok: str) -> tuple[str, Number, str, str]:
    """Return lemma_last (lower), number, display_singular, display_plural."""

    low = orig_tok.lower()

    # ss / very short: avoid "glass" -> "gla", "bus" edge cases with unknown number.
    if low.endswith("ss") or len(low) <= 2:
        return low, "unknown", orig_tok, orig_tok

    # Irregular plural input (e.g. children -> child).
    if low in _IRREGULAR_PLURAL_TO_LEMMA:
        lem = _IRREGULAR_PLURAL_TO_LEMMA[low]
        ds = _cased_like(orig_tok, lem)
        return lem, "plural", ds, orig_tok

    # Singular lemmas with irregular plurals (e.g. mouse -> mice).
    if low in _IRREGULAR_LEMMA_TO_PLURAL:
        plural_low = _IRREGULAR_LEMMA_TO_PLURAL[low]
        return low, "singular", orig_tok, _cased_like(orig_tok, plural_low)

    # ``…vie`` + plural ``s`` (e.g. ``movies``, not consonant+y ``…ies`` like ``berries``).
    if low.endswith("vies") and len(low) >= 6:
        lemma = low[:-1]
        dsp = _cased_like(orig_tok, lemma)
        return lemma, "plural", dsp, orig_tok

    if _looks_plural_y_ies(low):
        lemma = low[:-3] + "y"
        dsp = _cased_like(orig_tok, lemma)
        return lemma, "plural", dsp, orig_tok

    # Sibilant / ``-o`` plurals ending in ``…es`` (exclude ``apples``-style ``…es`` tails).
    if len(low) > 4 and low.endswith("es"):
        es_ok = (
            low.endswith(("xes", "ches", "shes", "zes", "oes"))
            or (low.endswith("ses") and not low.endswith("ases"))
        )
        if es_ok:
            lemma = low[:-2]
            if lemma:
                dsp = _cased_like(orig_tok, lemma)
                return lemma, "plural", dsp, orig_tok

    if low.endswith("s") and not low.endswith("ss") and len(low) > 3:
        lemma = low[:-1]
        if lemma:
            dsp = _cased_like(orig_tok, lemma)
            return lemma, "plural", dsp, orig_tok

    # Singular/other.
    dsp = orig_tok
    dpl = _plural_surface_from_lemma(low, orig_tok)
    return low, "singular", dsp, dpl


@dataclass(frozen=True)
class ItemLexeme:
    """Lexical breakdown of one possession-item phrase."""

    #: Surface after punctuation cleanup and NP determiner stripping.
    surface: str
    #: Lowercased matching key head (singularised lemma path when confident).
    lemma: str
    number: Number
    display_singular: str
    display_plural: str


def analyze_item_phrase(text: str) -> ItemLexeme:
    """Deterministic morphology for item phrases (last-token head; multi-word prefixes preserved)."""

    surf0 = normalize_possession_item_surface(text)
    work = strip_leading_np_determiners(surf0)
    tokens = work.split()
    if not tokens:
        return ItemLexeme(
            surface="",
            lemma="",
            number="unknown",
            display_singular="",
            display_plural="",
        )

    prefix = tokens[:-1]
    last = tokens[-1]
    lemma_last, num_last, dsp_last, dpl_last = _analyze_last_token(last)

    prefix_lower = [t.lower() for t in prefix]
    lemma_parts = [*prefix_lower, lemma_last]
    lemma = " ".join(lemma_parts)

    dsp = " ".join([*prefix, dsp_last]).strip()
    dpl = " ".join([*prefix, dpl_last]).strip()

    if num_last == "unknown":
        num: Number = "unknown"
        dsp_final = work
        dpl_final = work
        lemma_final = work.lower()
        return ItemLexeme(
            surface=work,
            lemma=lemma_final,
            number=num,
            display_singular=dsp_final,
            display_plural=dpl_final,
        )

    phrase_num: Number = num_last
    return ItemLexeme(
        surface=work,
        lemma=lemma,
        number=phrase_num,
        display_singular=dsp,
        display_plural=dpl,
    )


def item_key(item_phrase: str) -> str:
    """Return the canonical lowercase lemma used for HAS / inventory matching."""
    return analyze_item_phrase(item_phrase).lemma


def display_quantity_item(qty: int, item_phrase: str) -> str:
    """Format ``qty`` + item wording (``1`` uses singular display; plural otherwise)."""

    lx = analyze_item_phrase(item_phrase)
    if not lx.surface:
        return ""

    wording = lx.display_singular if qty == 1 else lx.display_plural
    # Match prior UX: integer quantities print without thousands separators.
    return f"{qty} {wording}".strip()

"""Deterministic surface realization for committed action relations (display only).

Maps canonical relation names and object literals to short English clauses for
state-native answers. Does not alter PEF semantics or extraction.
"""

from __future__ import annotations

# Canonical relation → past-tense verb for human-readable responses.
_PAST_TENSE: dict[str, str] = {
    "ORDER": "ordered",
    "ORDERED": "ordered",
    "GIVE": "gave",
    "TAKE": "took",
    "SEND": "sent",
    "TELL": "told",
    "SHOW": "showed",
    "RETURN": "returned",
    "LIKES": "liked",
    "LOVES": "loved",
    "WANTS": "wanted",
    "KNOWS": "knew",
}


def verb_past_tense_for_display(relation: str) -> str:
    """Map a canonical relation label to a past-tense verb for display."""
    upper = relation.upper()
    if upper in _PAST_TENSE:
        return _PAST_TENSE[upper]
    base = relation.lower()
    if base.endswith("e"):
        return base + "d"
    return base + "ed"


def normalize_event_object_literal(obj: str) -> str:
    """Normalize stored object text for display (underscores → spaces)."""
    if not obj:
        return ""
    s = str(obj).strip().replace("_", " ")
    return " ".join(s.split())


def object_phrase_with_determiner(item: str) -> str:
    """Prepend *the* to a bare noun phrase; leave proper names and articled NPs alone."""
    if not item:
        return item
    low = item.lower()
    if low.startswith(("the ", "a ", "an ")):
        return item
    if item[0].isupper():
        return item
    return f"the {item}"


def format_action_realization(relation: str, obj: str) -> str:
    """Return *verb [object]* clause, e.g. ``ordered the medication change``."""
    verb = verb_past_tense_for_display(relation)
    obj_norm = normalize_event_object_literal(obj)
    if not obj_norm:
        return verb
    return f"{verb} {object_phrase_with_determiner(obj_norm)}"

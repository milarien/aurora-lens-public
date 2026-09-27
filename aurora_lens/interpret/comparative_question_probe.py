"""Narrow comparative-question surface + PEF-structural underdetermination (v10-aligned).

Closed-class question shapes (literal normalization only) may *suggest* a comparative
clarification is needed; **governance** still requires structural evidence from PEF:

- at least two distinct owners sharing the same HAS/MANAGE object literal (eligible
  comparands for that head), and
- no explicit ``than <Name>`` comparand in the user text that names a PEF entity.

This is **not** open-ended regex-as-governor: only a fixed set of normalized strings
is recognized; the admit/no-admit decision is entirely from committed relationships.

See ``docs/reference/aurora_cli_v10.py`` (multi-candidate comparative → CLARIFY,
no admit) and ``SpacyBackend._detect_comparative_claims`` for copula-linked JJR/RBR
paths that may *also* populate ``comparative_ambiguities``.
"""

from __future__ import annotations

import re
from collections import defaultdict

from aurora_lens.interpret.schema import ComparativeAmbiguity, ExtractionResult
from aurora_lens.pef.state import PEFState

_WS_RE = re.compile(r"\s+")


def _normalize_comparative_question_probe_text(text: str) -> str:
    """Lowercase, trim, strip trailing ``?.!``, collapse internal whitespace."""
    s = text.strip().lower()
    s = s.rstrip("?.!")
    s = _WS_RE.sub(" ", s).strip()
    return s


# Normalized surface → comparative adjective token for UNRESOLVED_COMPARAND payload.
# Extend only with new **whole-utterance** keys (v10-style closed class).
_NARROW_COMPARATIVE_QUESTION_FORMS: dict[str, str] = {
    "bigger than what": "bigger",
    "which one is bigger": "bigger",
    "the bigger one": "bigger",
    "is it bigger": "bigger",
}


def _strip_possessive_suffix_casefold(token: str) -> str:
    """Remove trailing ``'s`` / ``'`` possessive markers only (not ``rstrip("'s")`` on names)."""
    t = token.casefold()
    if t.endswith("'s"):
        return t[:-2]
    if t.endswith("'"):
        return t[:-1]
    return t


def _explicit_named_comparand_after_than(user_text: str, pef: PEFState) -> bool:
    """True when ``than`` is followed by a token that names a committed PEF entity."""
    low = user_text.lower()
    idx = low.find(" than ")
    if idx < 0:
        return False
    tail = user_text[idx + len(" than ") :].strip()
    if not tail:
        return False
    first = tail.split()[0].strip().strip(".,!?")
    if not first or first.casefold() in ("what", "which", "who", "whom"):
        return False
    first_stem = _strip_possessive_suffix_casefold(first)
    for ent in pef.entities.values():
        head = ent.name.split()[0].casefold()
        if head == first_stem:
            return True
    return False


def _shared_has_literals_with_multiple_owners(
    pef: PEFState,
) -> list[tuple[str, str, list[str]]]:
    """Return ``(literal_key, literal_display, sorted_entity_names)`` for shared heads.

    ``literal_key`` is lowercased for matching; ``literal_display`` is the first
    non-empty surface ``object_literal`` seen for that key (audit-friendly).
    """
    owners: dict[str, set[str]] = defaultdict(set)
    display: dict[str, str] = {}
    for rel in pef.relationships:
        if rel.relation not in ("HAS", "MANAGE"):
            continue
        lit = (rel.object_literal or "").strip()
        if not lit:
            continue
        ent = pef.entities.get(rel.subject_id)
        if ent is None:
            continue
        key = lit.lower()
        owners[key].add(ent.name)
        display.setdefault(key, lit)
    out: list[tuple[str, str, list[str]]] = []
    for key, names in sorted(owners.items()):
        if len(names) >= 2:
            out.append((key, display[key], sorted(names)))
    return out


def structural_comparative_question_ambiguities(
    user_text: str,
    pef: PEFState,
) -> list[ComparativeAmbiguity]:
    """Return extra ``ComparativeAmbiguity`` rows from narrow surface + PEF structure."""
    norm = _normalize_comparative_question_probe_text(user_text)
    if norm not in _NARROW_COMPARATIVE_QUESTION_FORMS:
        return []
    if _explicit_named_comparand_after_than(user_text, pef):
        return []
    adj = _NARROW_COMPARATIVE_QUESTION_FORMS[norm]
    buckets = _shared_has_literals_with_multiple_owners(pef)
    if not buckets:
        return []
    # One ComparativeAmbiguity per structural conflict; Lens gate consumes the first.
    _key, lit_display, names = buckets[0]
    return [
        ComparativeAmbiguity(
            adjective=adj,
            noun=lit_display,
            candidates=list(names),
        )
    ]


def merge_structural_comparative_question_probe(
    extraction: ExtractionResult,
    user_text: str,
    pef: PEFState,
) -> None:
    """Append probe-derived ambiguities without duplicating spaCy-derived rows."""
    extra = structural_comparative_question_ambiguities(user_text, pef)
    if not extra:
        return
    seen = {
        (ca.noun.casefold(), ca.adjective.casefold(), tuple(sorted(ca.candidates)))
        for ca in extraction.comparative_ambiguities
    }
    for ca in extra:
        key = (ca.noun.casefold(), ca.adjective.casefold(), tuple(sorted(ca.candidates)))
        if key in seen:
            continue
        extraction.comparative_ambiguities.append(ca)
        seen.add(key)

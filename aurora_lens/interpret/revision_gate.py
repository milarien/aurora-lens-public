"""Gate PEF updates when user claims contradict grounded state without a revision act.

Ordinary assertions must not silently replace durable facts.  Explicit correction
markers (``Correction:``, ``I misspoke``, …) license lawful replacement; hostile
``Actually …`` phrasing does not.
"""

from __future__ import annotations

import unicodedata

from aurora_lens.interpret.pef_updater import (
    _relation_for_pef_write,
    _resolve_single_pronoun,
)
from aurora_lens.interpret.schema import ExtractedClaim, ExtractionResult
from aurora_lens.pef.state import PEFState, canonicalize_relation

_PRONOUNS = frozenset({"he", "she", "it", "they", "him", "her", "them", "his", "its", "their"})

_COLOR_LEX: frozenset[str] = frozenset({
    "red", "blue", "green", "yellow", "orange", "purple", "pink",
    "black", "white", "gray", "grey", "brown", "violet", "gold", "silver",
})
_HEAD_STOPWORDS: frozenset[str] = frozenset(
    {"a", "an", "the", "of", "to", "for", "with", "and", "or"}
)


def explicit_correction_intent(user_text: str) -> bool:
    """Canonical explicit correction/revision intent detector (no regex)."""
    t = (user_text or "").strip().lower()
    if not t:
        return False
    # Clarification framing ("Just to be clear, I meant ...") is meta-disambiguation,
    # not an authoritative state revision.
    if t.startswith("just to be clear"):
        return False
    if t.startswith("correction:"):
        return True
    if t.startswith("i misspoke") or " i misspoke " in f" {t} ":
        return True
    if t.startswith("i meant"):
        return True
    if t.startswith("i was wrong") or " i was wrong " in f" {t} ":
        return True
    if "to correct that" in t:
        return True
    if "let me correct" in t:
        return True
    if "i need to correct" in t:
        return True
    return False


def user_explicit_revision_act(user_text: str) -> bool:
    """Backward-compatible alias for explicit revision/correction intent."""
    return explicit_correction_intent(user_text)


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


def _scan_tokens_keep_apostrophe(text: str) -> tuple[str, ...]:
    out: list[str] = []
    cur: list[str] = []
    for ch in text or "":
        if ch.isalnum() or ch == "'":
            cur.append(ch)
            continue
        if cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return tuple(out)


def _object_head_literal(literal: str) -> str:
    toks = _scan_word_tokens(literal)
    for tok in reversed(toks):
        if tok not in _HEAD_STOPWORDS:
            return tok
    return toks[-1] if toks else ""


def _color_tokens(s: str) -> frozenset[str]:
    return frozenset(w for w in _scan_word_tokens(s) if w in _COLOR_LEX)


def _parse_possessive_subject(subject_text: str) -> tuple[str, str] | None:
    """Parse owner/object-head from possessive subject forms.

    Supported deterministic shapes:
    - "Emma's book"
    - "Emma 's book"
    - "Emma's red box"
    """
    subj = unicodedata.normalize("NFKC", subject_text or "")
    subj = subj.replace("\u2019", "'").replace("\u2018", "'").strip()
    toks = _scan_tokens_keep_apostrophe(subj)
    if len(toks) < 2:
        return None

    owner = ""
    obj_tokens: tuple[str, ...] = ()
    if toks[0].endswith("'s") and len(toks[0]) > 2:
        owner = toks[0][:-2]
        obj_tokens = toks[1:]
    elif len(toks) >= 3 and toks[1] == "'s":
        owner = toks[0]
        obj_tokens = toks[2:]
    else:
        return None

    head = _object_head_literal(" ".join(obj_tokens))
    if not owner or not head:
        return None
    return owner, head


def _parse_possessive_is_clause(text: str) -> tuple[str, str, str] | None:
    """Parse "<owner>'s <object> is/are/was/were <predicate>" clauses."""
    raw = unicodedata.normalize("NFKC", text or "")
    raw = raw.replace("\u2019", "'").replace("\u2018", "'")
    toks = tuple(t.lower() for t in _scan_tokens_keep_apostrophe(raw))
    if len(toks) < 4:
        return None

    for start in range(0, len(toks) - 3):
        owner = ""
        rest: tuple[str, ...] = ()
        if toks[start].endswith("'s") and len(toks[start]) > 2:
            owner = toks[start][:-2]
            rest = toks[start + 1 :]
        elif start + 2 < len(toks) and toks[start + 1] == "'s":
            owner = toks[start]
            rest = toks[start + 2 :]
        else:
            continue

        copula_idx = -1
        for i, tok in enumerate(rest):
            if tok in {"is", "are", "was", "were"}:
                copula_idx = i
                break
        if copula_idx <= 0:
            continue
        obj_tokens = rest[:copula_idx]
        pred_tokens = rest[copula_idx + 1 :]
        while pred_tokens and pred_tokens[0] in {"a", "an", "the"}:
            pred_tokens = pred_tokens[1:]
        if not obj_tokens or not pred_tokens:
            continue
        obj_head = _object_head_literal(" ".join(obj_tokens))
        if not obj_head:
            continue
        return owner, obj_head, " ".join(pred_tokens)
    return None


def _split_clauses(text: str) -> tuple[str, ...]:
    out: list[str] = []
    cur: list[str] = []
    for ch in text or "":
        if ch in ".!?;\n":
            part = "".join(cur).strip()
            if part:
                out.append(part)
            cur = []
            continue
        cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        out.append(tail)
    return tuple(out)


def _same_relation_slot(a: str, b: str) -> bool:
    ca, cb = canonicalize_relation(a), canonicalize_relation(b)
    if ca == cb:
        return True
    if {ca, cb} <= {"HAS", "MANAGE"}:
        return True
    return False


def _literals_conflict(old_lit: str, new_lit: str, relation: str) -> bool:
    """True when two object literals assert incompatible facts for the same slot."""
    o, n = old_lit.lower(), new_lit.lower()
    co, cn = _color_tokens(o), _color_tokens(n)
    rel_u = canonicalize_relation(relation)
    if rel_u == "IS":
        if co and cn and co.isdisjoint(cn):
            return True
        return False
    if rel_u in ("HAS", "MANAGE"):
        if not co or not cn:
            return False
        if co & cn:
            return False
        if _object_head_literal(o) and _object_head_literal(o) == _object_head_literal(n):
            return True
        return False
    return False


def _possessive_subject_is_conflicts_prior_has(
    pef: PEFState,
    claim: ExtractedClaim,
) -> tuple[bool, str]:
    """Detect ``owner's object`` IS ``value`` vs prior ``owner`` HAS same-object value."""
    if canonicalize_relation(claim.relation) != "IS":
        return False, ""
    parsed = _parse_possessive_subject(claim.subject)
    if parsed is None:
        return False, ""
    owner_name, object_head = parsed
    ent = pef.find_entity_by_name(owner_name)
    if ent is None:
        return False, ""
    co_new = _color_tokens(claim.obj)
    if not co_new:
        return False, ""
    for rel in pef.get_relationships_for_subject(ent.id):
        if rel.relation not in ("HAS", "MANAGE"):
            continue
        if rel.object_literal and _object_head_literal(rel.object_literal) == object_head:
            co_old = _color_tokens(rel.object_literal)
            if co_old and co_new and co_old.isdisjoint(co_new):
                return (
                    True,
                    f"{owner_name} HAS {rel.object_literal!r} vs possessive subject IS {claim.obj!r}",
                )
    return False, ""


def user_text_conflicts_grounded_pef(pef: PEFState, user_text: str) -> tuple[bool, str]:
    """True when *raw* user text declaratively asserts owned-object value contradicting PEF.

    Does not depend on structured extraction. Use when ``extraction_conflicts_grounded_pef``
    is false because the backend produced no/empty claims for the turn.
    """
    if not user_text.strip():
        return False, ""
    t = unicodedata.normalize("NFKC", user_text)
    t = t.replace("\u2019", "'").replace("\u2018", "'")
    for clause in _split_clauses(t):
        parsed = _parse_possessive_is_clause(clause)
        if parsed is None:
            continue
        owner_name, object_head, predicate = parsed
        ent = pef.find_entity_by_name(owner_name)
        if ent is None:
            continue
        co_new = _color_tokens(predicate)
        if not co_new:
            continue
        for rel in pef.get_relationships_for_subject(ent.id):
            if rel.relation not in ("HAS", "MANAGE"):
                continue
            if rel.object_literal and _object_head_literal(rel.object_literal) == object_head:
                co_old = _color_tokens(rel.object_literal)
                if co_old and co_new and co_old.isdisjoint(co_new):
                    return (
                        True,
                        f"user text: {owner_name}'s {object_head} is {predicate} vs grounded {rel.object_literal!r}",
                    )
    return False, ""


def extraction_conflicts_grounded_pef(
    pef: PEFState,
    extraction: ExtractionResult,
) -> tuple[bool, str]:
    """Return (True, detail) if any extracted claim would contradict existing PEF literals.

    Mirrors subject resolution in ``pef_updater.update_pef`` (pronouns, ambiguous block).
    """
    blocked = frozenset(p.lower() for p in extraction.ambiguous_referents)

    for claim in extraction.claims:
        _pb, _pd = _possessive_subject_is_conflicts_prior_has(pef, claim)
        if _pb:
            return True, _pd

        subject_name = claim.subject
        if subject_name.lower() in _PRONOUNS:
            if subject_name.lower() in blocked:
                continue
            resolved = _resolve_single_pronoun(subject_name, pef)
            if resolved:
                subject_name = resolved
            else:
                continue

        subj_entity = pef.find_entity_by_name(subject_name)
        if subj_entity is None:
            continue
        stored_rel = _relation_for_pef_write(claim)

        for rel in pef.get_relationships_for_subject(subj_entity.id):
            if rel.object_literal is None:
                continue
            if not _same_relation_slot(rel.relation, stored_rel):
                continue
            if _literals_conflict(rel.object_literal, claim.obj, stored_rel):
                return (
                    True,
                    f"{subj_entity.name} {stored_rel}: grounded {rel.object_literal!r} vs new {claim.obj!r}",
                )
    return False, ""


def revision_clarification_message(detail: str) -> str:
    """User-facing line when a hostile contradiction is blocked pre-LLM."""
    return (
        "You stated something that conflicts with an earlier fact in this session. "
        "If you want to change the record, say so explicitly (for example, start with "
        "\"Correction:\" or \"I misspoke earlier\"). "
        f"({detail})"
    )

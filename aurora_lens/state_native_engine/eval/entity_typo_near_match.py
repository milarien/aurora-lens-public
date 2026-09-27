"""Deterministic near-name analysis for state-native entity ambiguity (no LLM)."""

from __future__ import annotations

from collections.abc import Sequence

from aurora_lens.pef.entity import Entity
from aurora_lens.pef.state import PEFState


def levenshtein(a: str, b: str) -> int:
    """Classic edit distance for short ASCII-ish entity names."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def _norm_match_key(s: str) -> str:
    return (s or "").strip().lower()


def plausible_typo_surface(typ_surface: str, candidate_surface: str) -> bool:
    """Conservative typo acceptance (normalised lowercase).

    Rules:
    - Surfaces must differ after normalisation (distinct spellings).
    - Length delta at most 1.
    - Edit distance at most 1 (strict for typical entity-name lengths).
    """
    a = _norm_match_key(typ_surface)
    b = _norm_match_key(candidate_surface)
    if not a or not b or a == b:
        return False
    if abs(len(a) - len(b)) > 1:
        return False
    return levenshtein(a, b) <= 1


def pick_singleton_typo_resolution(
    pef: PEFState,
    ambiguous_entities: Sequence[Entity],
) -> tuple[Entity, Entity] | None:
    """If exactly one ambiguous entity is a plausible typo of exactly one other entity, return pair.

    Returns (typo_ambiguous_entity, committed_target_entity) or None.
    """
    amb_list = list(ambiguous_entities)
    if len(amb_list) < 2:
        return None
    typo_singletons: list[tuple[Entity, Entity]] = []
    for amb in amb_list:
        matches = [
            e
            for e in pef.entities.values()
            if e.id != amb.id
            and plausible_typo_surface(amb.name, e.name)
            and _norm_match_key(amb.name) != _norm_match_key(e.name)
        ]
        if len(matches) == 1:
            typo_singletons.append((amb, matches[0]))
        # len(matches) > 1: this ambiguous entity has multiple typo targets — skip it.

    if len(typo_singletons) != 1:
        return None
    typo_ent, canon_ent = typo_singletons[0]
    return typo_ent, canon_ent


def augment_state_native_ambiguity_with_typo_recovery(
    pef: PEFState,
    ambiguous_entities: Sequence[Entity],
    *,
    ambiguity_kind: str,
    generic_user_visible_text: str,
    clarify_context: dict,
) -> tuple[str, dict]:
    """Attach typo_recovery and typo clarification text when structurally safe."""
    pair = pick_singleton_typo_resolution(pef, ambiguous_entities)
    ctx = dict(clarify_context)
    ctx["state_native_ambiguity_kind"] = ambiguity_kind
    if pair is None:
        return generic_user_visible_text, ctx
    typo_ent, canon_ent = pair
    ctx["typo_recovery"] = {
        "typo_surface": typo_ent.name,
        "typo_entity_id": typo_ent.id,
        "suggested_canonical_name": canon_ent.name,
        "canonical_entity_id": canon_ent.id,
    }
    msg = f"Did you mean {canon_ent.name} instead of {typo_ent.name}?"
    ctx["clarification_prompt"] = msg
    return msg, ctx

"""Read-only helpers for AT (locative) and HAS (possession) relationships.

Mirrors the old kernel pattern in the smallest surface: **store -> filter ->
deterministic sort -> read head** (and optional second row for prior).

Current / prior resolution (no schema changes):

- Consider positive AT/HAS rows for the given ``subject_id``.
- A positive fact is active only if no later negation exists for the same object.
- Order by ``source_turn`` descending, then by append index in
  ``PEFState.relationships`` descending (later list position = admitted
  later within the same turn - stable tie-break).
"""

from __future__ import annotations

from typing import Any

from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState, Relationship

AT_BASIS_SCHEMA_VERSION = 1


def _non_negated_at_or_has_pairs(
    pef: PEFState, subject_id: str
) -> list[tuple[int, Relationship]]:
    """DEPRECATED: use _active_positive_at_or_has_pairs."""
    pairs: list[tuple[int, Relationship]] = []
    for i, rel in enumerate(pef.relationships):
        if rel.subject_id != subject_id:
            continue
        if rel.relation not in ("AT", "HAS"):
            continue
        if rel.negated:
            continue
        pairs.append((i, rel))
    return pairs


def _effective_at_or_has_chain_indices_newest_first(
    pef: PEFState, subject_id: str
) -> list[tuple[int, Relationship]]:
    """All AT/HAS relationships for subject, newest first, including negated ones."""
    pairs: list[tuple[int, Relationship]] = []
    for i, rel in enumerate(pef.relationships):
        if rel.subject_id != subject_id:
            continue
        if rel.relation not in ("AT", "HAS"):
            continue
        pairs.append((i, rel))
    # Sort by turn desc, then index desc
    pairs.sort(key=lambda p: (-p[1].source_turn, -p[0]))
    return pairs


def _get_object_key(rel: Relationship) -> str:
    """Stable key for comparing objects across relationships."""
    if rel.object_entity_id:
        return f"ent:{rel.object_entity_id}"
    return f"lit:{str(rel.object_literal).strip().lower()}"


def _active_positive_at_or_has_pairs(
    pef: PEFState, subject_id: str
) -> list[tuple[int, Relationship]]:
    """Positive AT/HAS facts that have not been negated by a later relationship."""
    chain = _effective_at_or_has_chain_indices_newest_first(pef, subject_id)
    if not chain:
        return []
        
    negated_objects: set[str] = set()
    active_positive_pairs: list[tuple[int, Relationship]] = []
    
    for idx, rel in chain:
        obj_key = _get_object_key(rel)
        if rel.negated:
            negated_objects.add(obj_key)
        else:
            if obj_key not in negated_objects:
                active_positive_pairs.append((idx, rel))
                
    return active_positive_pairs


def _at_or_has_chain_indices_newest_first(
    pef: PEFState, subject_id: str
) -> list[tuple[int, Relationship]]:
    """DEPRECATED: use _active_positive_at_or_has_pairs."""
    return _active_positive_at_or_has_pairs(pef, subject_id)


def _non_negated_at_pairs(
    pef: PEFState, subject_id: str
) -> list[tuple[int, Relationship]]:
    """DEPRECATED: use _active_positive_at_pairs."""
    pairs: list[tuple[int, Relationship]] = []
    for i, rel in enumerate(pef.relationships):
        if rel.subject_id != subject_id:
            continue
        if rel.relation != "AT":
            continue
        if rel.negated:
            continue
        pairs.append((i, rel))
    return pairs


def _active_positive_at_pairs(
    pef: PEFState, subject_id: str
) -> list[tuple[int, Relationship]]:
    """Positive AT facts that have not been negated by a later relationship."""
    # Filter active positive pairs to only those with relation "AT"
    all_active = _active_positive_at_or_has_pairs(pef, subject_id)
    return [(idx, rel) for idx, rel in all_active if rel.relation == "AT"]


def _at_chain_newest_first(pef: PEFState, subject_id: str) -> list[Relationship]:
    """Latest active positive AT facts for subject."""
    pairs = _active_positive_at_pairs(pef, subject_id)
    return [p[1] for p in pairs]


def current_at(pef: PEFState, subject_id: str) -> Relationship | None:
    """Latest positive active AT fact for ``subject_id``, or ``None`` if none."""
    chain = _at_chain_newest_first(pef, subject_id)
    return chain[0] if chain else None


def prior_at(pef: PEFState, subject_id: str) -> Relationship | None:
    """Second-most-recent positive active AT fact, or ``None`` if fewer than two."""
    chain = _at_chain_newest_first(pef, subject_id)
    return chain[1] if len(chain) >= 2 else None


def _at_chain_indices_newest_first(
    pef: PEFState, subject_id: str
) -> list[tuple[int, Relationship]]:
    """DEPRECATED: use _active_positive_at_pairs."""
    return _active_positive_at_pairs(pef, subject_id)


def _relationship_basis_slot(index: int, rel: Relationship) -> dict[str, Any]:
    span_val = rel.span.value if isinstance(rel.span, Span) else str(rel.span)
    return {
        "relationship_index": index,
        "relation": rel.relation,
        "object_entity_id": rel.object_entity_id,
        "object_literal": rel.object_literal,
        "source_turn": rel.source_turn,
        "span": span_val,
        "negated": rel.negated,
    }


def at_basis_snapshot(pef: PEFState) -> dict[str, Any]:
    """Serializable snapshot of effective current/prior AT/HAS per subject (verification-time basis).

    One row per entity that has at least one positive ``AT`` or ``HAS`` relationship
    that has not been superseded by a later negation for the same object.
    """
    # Identify all subjects who have ever had an AT or HAS relationship
    all_subject_ids = {
        rel.subject_id
        for rel in pef.relationships
        if rel.relation in ("AT", "HAS")
    }
    
    subjects: list[dict[str, Any]] = []
    for sid in sorted(all_subject_ids):
        active_positive_pairs = _active_positive_at_or_has_pairs(pef, sid)
        
        if not active_positive_pairs:
            continue
            
        # The first one in the active list is the 'current' state.
        cur_idx, cur_rel = active_positive_pairs[0]
        cur = _relationship_basis_slot(cur_idx, cur_rel)
        
        prior_dict: dict[str, Any] | None = None
        if len(active_positive_pairs) >= 2:
            pr_idx, pr_rel = active_positive_pairs[1]
            prior_dict = _relationship_basis_slot(pr_idx, pr_rel)
            
        ent = pef.entities.get(sid)
        subjects.append(
            {
                "subject_id": sid,
                "subject_name": ent.name if ent else sid,
                "current": cur,
                "prior": prior_dict,
            }
        )

    return {
        "schema_version": AT_BASIS_SCHEMA_VERSION,
        "pef_current_turn": pef.current_turn,
        "subjects": subjects,
    }

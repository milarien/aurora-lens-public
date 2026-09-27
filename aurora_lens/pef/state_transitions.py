"""Generic state-transition helpers over PEF relationships.

Domain adapters may emit typed relations with structured ``object_literal`` payloads.
These helpers project latest/current state and provenance from ordinary PEF
relationships without introducing domain-specific side stores.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from aurora_lens.pef.state import PEFState, Relationship


LiteralFilter = Callable[[Relationship, dict[str, Any]], bool]


def iter_literal_relationships(
    pef: PEFState,
    *,
    relation_allowlist: set[str] | None = None,
    literal_filter: LiteralFilter | None = None,
) -> Iterable[tuple[int, Relationship, dict[str, Any]]]:
    """Yield ``(index, relationship, literal_dict)`` for matching PEF relationships."""
    for idx, rel in enumerate(pef.relationships):
        if relation_allowlist is not None and rel.relation not in relation_allowlist:
            continue
        lit = rel.object_literal
        if not isinstance(lit, dict):
            continue
        if literal_filter is not None and not literal_filter(rel, lit):
            continue
        yield idx, rel, lit


def project_latest_by_literal_key(
    pef: PEFState,
    *,
    key_fields: tuple[str, ...],
    order_fields: tuple[str, ...] = (),
    relation_allowlist: set[str] | None = None,
    literal_filter: LiteralFilter | None = None,
) -> dict[tuple[str, ...], tuple[int, Relationship, dict[str, Any]]]:
    """Project latest relationship per literal key tuple.

    Key tuple values are lowercased strings, preserving deterministic matching for
    domain-specific typed transitions that share a common state key shape.
    """
    latest: dict[tuple[str, ...], tuple[int, Relationship, dict[str, Any]]] = {}
    latest_rank: dict[tuple[str, ...], tuple[Any, ...]] = {}
    for idx, rel, lit in iter_literal_relationships(
        pef,
        relation_allowlist=relation_allowlist,
        literal_filter=literal_filter,
    ):
        key_vals: list[str] = []
        for field in key_fields:
            raw = lit.get(field)
            if raw is None:
                key_vals = []
                break
            key_vals.append(str(raw).strip().lower())
        if not key_vals:
            continue
        key = tuple(key_vals)
        rank: list[Any] = []
        for f in order_fields:
            rank.append(lit.get(f))
        rank.extend([rel.source_turn, idx])
        rank_t = tuple(rank)
        if key not in latest or rank_t >= latest_rank[key]:
            latest[key] = (idx, rel, lit)
            latest_rank[key] = rank_t
    return latest


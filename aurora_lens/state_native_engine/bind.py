"""Shared entity-binding primitive for state-native evaluators."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from aurora_lens.pef.entity import Entity
from aurora_lens.pef.state import PEFState


class BindingKind(str, Enum):
    """Determinacy classification for entity binding."""

    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    MISSING = "missing"


@dataclass(frozen=True)
class BindingResult:
    """Result of resolving a phrase against committed PEF entities."""

    kind: BindingKind
    entities: tuple[Entity, ...]


def _resolve_entities(phrase: str, pef: PEFState) -> list[Entity]:
    """Return zero/one/many entity matches using existing deterministic rules."""
    raw = (phrase or "").strip()
    if not raw:
        return []

    ent = pef.find_entity_by_name(raw)
    if ent is not None:
        return [ent]

    low = raw.lower()
    if low.startswith("the "):
        stripped = raw[4:].strip()
        ent = pef.find_entity_by_name(stripped)
        if ent is not None:
            return [ent]
        low = stripped.lower()

    exact_name_matches = [
        e for e in pef.entities.values() if e.name.lower() == low
    ]
    if len(exact_name_matches) == 1:
        return exact_name_matches
    if len(exact_name_matches) > 1:
        return exact_name_matches

    if not low:
        return []

    return [e for e in pef.entities.values() if low in e.name.lower()]


def resolve_entity(phrase: str, pef: PEFState) -> BindingResult:
    """Resolve phrase to RESOLVED / AMBIGUOUS / MISSING with matched entities."""
    entities = tuple(_resolve_entities(phrase, pef))
    if len(entities) == 1:
        return BindingResult(kind=BindingKind.RESOLVED, entities=entities)
    if len(entities) > 1:
        return BindingResult(kind=BindingKind.AMBIGUOUS, entities=entities)
    return BindingResult(kind=BindingKind.MISSING, entities=())

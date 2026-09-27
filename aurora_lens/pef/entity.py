"""Entity, EchoTrace, and TurnBindings — persistent conceptual entities."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from .span import Span

# Deterministic entity IDs for state_hash stability (replay verification).
_ENTITY_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "aurora-lens.local")


def _deterministic_entity_id(
    session_id: str,
    canon_name: str,
    *,
    ordinal: int | None = None,
) -> str:
    """Session-scoped content-addressed ID. Siblings use ordinal for same-name collisions."""
    salt = f"{session_id}:{canon_name}"
    if ordinal is not None:
        salt = f"{salt}#{ordinal}"
    return str(uuid.uuid5(_ENTITY_NAMESPACE, salt))


@dataclass
class EchoTrace:
    """Record of an entity's activation in a specific turn and span."""
    turn: int
    context: str       # Surface text that activated this entity
    span: Span


@dataclass
class Entity:
    """A persistent conceptual entity in PEF state.

    No span on Entity — entities don't live in past/present;
    claims about them do. Span belongs on EchoTrace and Relationship.
    """
    id: str                              # Stable UUID
    name: str                            # Canonical name ("Emma")
    aliases: set[str] = field(default_factory=set)
        # STABLE surface forms only ("Emma", "Ms Smith", "the teacher")
        # NO pronouns — pronouns are situational bindings, not aliases
    attributes: dict[str, Any] = field(default_factory=dict)
    echo_traces: list[EchoTrace] = field(default_factory=list)
    turn_introduced: int = 0
    turn_last_active: int = 0
    resolved: bool = True
        # False = UNRESOLVED placeholder (mentioned but not asserted about)

    @staticmethod
    def create(
        name: str,
        turn: int,
        resolved: bool = True,
        *,
        session_id: str = "",
        ordinal: int | None = None,
    ) -> Entity:
        """Create a new entity with session-scoped deterministic ID.
        ordinal=None for base; ordinal=k for sibling (k>=2).
        """
        canon_name = name.strip().lower()
        eid = _deterministic_entity_id(session_id, canon_name, ordinal=ordinal)
        return Entity(
            id=eid,
            name=name,
            aliases={name},
            turn_introduced=turn,
            turn_last_active=turn,
            resolved=resolved,
        )


@dataclass
class TurnBindings:
    """Per-turn pronoun-to-entity resolution.

    Pronouns are situational, not global aliases.
    Key is occurrence-based (e.g. "she@token_14"), NOT surface form,
    to prevent collisions when multiple pronouns resolve differently.
    """
    turn: int
    bindings: dict[str, str] = field(default_factory=dict)
        # pronoun occurrence key (e.g. "she@token_14") -> entity_id

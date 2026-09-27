"""Action-agent query: who performed the relevant committed action?

Invariant: resolved binding beats role/title heuristic.
If PEF records Emma ORDERED medication_change, follow-up target selection
must return Emma regardless of co-present entities with professional titles.
"""

from __future__ import annotations

from aurora_lens.pef.entity import Entity
from aurora_lens.pef.state import PEFState
from aurora_lens.state_native_engine.contracts import StateNativeOutcome
from aurora_lens.state_native_engine.eval.action_realization import format_action_realization

# Relations that are structural/state, not agent actions requiring follow-up.
_NON_ACTION_RELATIONS: frozenset[str] = frozenset({
    "HAS", "AT", "IS", "BECAUSE", "LIKES", "LOVES", "WANTS", "KNOWS",
})


def evaluate_action_agent(
    pef: PEFState,
) -> tuple[str | None, str, dict | None, str | None]:
    """Return the subject of the most recently committed action relation.

    Returns ``(None, "", None, None)`` when PEF has no action relations —
    caller should pass through to LLM.
    """
    # Collect (source_turn, entity, relation, obj_text) for non-negated action relations.
    candidates: list[tuple[int, Entity, str, str]] = []
    for rel in pef.relationships:
        if rel.relation in _NON_ACTION_RELATIONS:
            continue
        if rel.negated:
            continue
        ent = pef.entities.get(rel.subject_id)
        if ent is None:
            continue
        obj: str = ""
        if rel.object_literal is not None:
            obj = str(rel.object_literal).strip()
        elif rel.object_entity_id is not None:
            obj_ent = pef.entities.get(rel.object_entity_id)
            if obj_ent:
                obj = obj_ent.name
        candidates.append((rel.source_turn, ent, rel.relation, obj))

    if not candidates:
        return (None, "", None, None)

    # Latest source_turn wins; collect all candidates at that turn.
    max_turn = max(c[0] for c in candidates)
    latest = [c for c in candidates if c[0] == max_turn]

    # Deduplicate by agent name, preserving the first occurrence.
    seen_names: dict[str, tuple[int, Entity, str, str]] = {}
    for c in latest:
        seen_names.setdefault(c[1].name, c)

    if len(seen_names) == 1:
        ent_name, (_, _, relation, obj) = next(iter(seen_names.items()))
        detail = f"{ent_name} {format_action_realization(relation, obj)}."
        return (
            StateNativeOutcome.ANSWER.value,
            f"Follow up with {ent_name}. {detail}",
            None,
            None,
        )

    names = sorted(seen_names.keys())
    return (
        StateNativeOutcome.ANSWER.value,
        f"Multiple agents are involved: {', '.join(names)}. Which would you like to follow up with?",
        None,
        None,
    )

"""Existence and actor-value query evaluation over committed PEF.

Handles:
  ``Did X [event]?``  → TRUE / FALSE / UNKNOWN / AMBIGUOUS
  ``Who [event]?``    → VALUE / UNKNOWN

Design
------
1. Extract keywords from the event phrase (stopwords removed).
2. Scan PEF for *action* relations whose relation name, evidence text, or
   object_literal contains at least one event keyword.
3. Apply epistemic rules:

   Existence (Did X …?):
     a. Claimed actor found as committed subject of a matching relation → TRUE
     b. Claimed actor NOT the subject but another entity IS → FALSE
     c. Matching action exists but claimed actor not in PEF → FALSE
        (competing committed subject beats unknown actor)
     d. No matching action relation exists at all → UNKNOWN
     e. Claimed actor phrase resolves to 2+ PEF entities → AMBIGUOUS

   Actor-value (Who …?):
     a. Matching action exists → VALUE (most recent subject)
     b. No matching action → UNKNOWN

Relations treated as non-action (structural/state) are excluded from matching.
"""

from __future__ import annotations

from aurora_lens.pef.entity import Entity
from aurora_lens.pef.state import PEFState, Relationship
from aurora_lens.state_native_engine.bind import resolve_entity
from aurora_lens.state_native_engine.contracts import (
    StateNativeDelegationResult,
    StateNativeOutcome,
    StateNativeSolverFamily,
)
from aurora_lens.state_native_engine.epistemic import EpistemicResult
from aurora_lens.state_native_engine.eval.action_realization import format_action_realization

# Relations that describe static state, not agent-performed actions.
_NON_ACTION_RELATIONS: frozenset[str] = frozenset({
    "HAS", "AT", "IS", "BECAUSE", "LIKES", "LOVES", "WANTS", "KNOWS",
    "WITH", "WEARING", "CARRYING",
    "NEAR", "FROM",
    "REMAIN", "OUTSIDE",
})

_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "of", "to", "in", "for", "on", "at", "by",
    "with", "and", "or", "not", "is", "are", "was", "were", "be",
    "been", "have", "has", "had", "do", "does", "did", "this", "that",
    "these", "those",
})

_SOLVER = StateNativeSolverFamily.COMMITTED_EXISTENCE_READ


def _event_keywords(text: str) -> frozenset[str]:
    """Extract meaningful (non-stopword) keywords from an event phrase."""
    words: set[str] = set()
    for w in text.lower().split():
        w = w.strip(".,!?;:'\"")
        if w and len(w) > 2 and w not in _STOPWORDS:
            words.add(w)
    return frozenset(words)


def _relation_matches_event(rel: Relationship, keywords: frozenset[str]) -> bool:
    """True when this relation's name, evidence, or object_literal contains a keyword."""
    if not keywords:
        return True

    # Relation name (e.g. "CHANGE" matches keyword "change")
    rel_words = frozenset(rel.relation.lower().split("_"))
    if keywords & rel_words:
        return True

    # Evidence text
    if rel.evidence:
        ev_words = frozenset(w.strip(".,!?;:'\"") for w in rel.evidence.lower().split())
        if keywords & ev_words:
            return True

    # Object literal
    obj = rel.object_literal
    if isinstance(obj, str):
        obj_words = frozenset(w.strip(".,!?;:'\"") for w in obj.lower().split())
        if keywords & obj_words:
            return True
    elif isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, str):
                v_words = frozenset(w.strip(".,!?;:'\"") for w in v.lower().split())
                if keywords & v_words:
                    return True

    return False


def _collect_action_matches(
    pef: PEFState,
    keywords: frozenset[str],
) -> list[tuple[int, Entity, str, str]]:
    """Return (source_turn, entity, relation, obj_text) for all matching action relations."""
    out: list[tuple[int, Entity, str, str]] = []
    for rel in pef.relationships:
        if rel.relation in _NON_ACTION_RELATIONS:
            continue
        if rel.negated:
            continue
        if not _relation_matches_event(rel, keywords):
            continue
        ent = pef.entities.get(rel.subject_id)
        if ent is None:
            continue
        obj: str = ""
        if rel.object_literal is not None:
            if isinstance(rel.object_literal, str):
                obj = rel.object_literal.strip()
            elif isinstance(rel.object_literal, dict):
                med = rel.object_literal.get("medication") or rel.object_literal.get("item")
                obj = str(med).strip() if med else ""
        elif rel.object_entity_id is not None:
            obj_ent = pef.entities.get(rel.object_entity_id)
            if obj_ent:
                obj = obj_ent.name
        out.append((rel.source_turn, ent, rel.relation, obj))
    return out


def evaluate_existence_query(
    pef: PEFState,
    actor_phrase: str,
    event_tail: str,
) -> StateNativeDelegationResult | None:
    """Evaluate ``Did X [event]?`` against committed PEF action relations.

    Returns None when the event keywords match nothing at all (caller should
    fall through to next evaluator or LLM).
    """
    keywords = _event_keywords(event_tail)
    matches = _collect_action_matches(pef, keywords)

    if not matches:
        # No matching action committed → UNKNOWN → fall through to LLM
        return None

    # Resolve the claimed actor to PEF entities.
    actor_entities = list(resolve_entity(actor_phrase, pef).entities)

    if len(actor_entities) > 1:
        names = sorted(e.name for e in actor_entities)
        return StateNativeDelegationResult(
            handled=True,
            outcome=StateNativeOutcome.CLARIFY,
            user_visible_text=f"Which did you mean: {', '.join(names)}?",
            clarify_context={
                "failed_constraint": "STATE_NATIVE_ENTITY_AMBIGUITY",
                "candidate_entities": names,
                "subject_phrase": actor_phrase.strip(),
            },
            solver_family=_SOLVER,
            epistemic_result=EpistemicResult.AMBIGUOUS,
        )

    # Most recent match (for FALSE text generation).
    best_turn = max(m[0] for m in matches)
    best = next(m for m in matches if m[0] == best_turn)
    competing_name = best[1].name
    # Build a concise action description from the best match.
    action_desc = format_action_realization(best[2], best[3])

    if len(actor_entities) == 0:
        # Actor not in PEF; a different entity committed the action → FALSE.
        return StateNativeDelegationResult(
            handled=True,
            outcome=StateNativeOutcome.ANSWER,
            user_visible_text=(
                f"No. {actor_phrase.strip()} was not the actor. "
                f"{competing_name} {action_desc}."
            ),
            solver_family=_SOLVER,
            epistemic_result=EpistemicResult.FALSE,
        )

    actor_entity = actor_entities[0]
    actor_is_subject = any(m[1].id == actor_entity.id for m in matches)

    if actor_is_subject:
        # Claimed actor IS the committed subject → TRUE.
        return StateNativeDelegationResult(
            handled=True,
            outcome=StateNativeOutcome.ANSWER,
            user_visible_text=f"Yes. {actor_entity.name} {action_desc}.",
            solver_family=_SOLVER,
            epistemic_result=EpistemicResult.TRUE,
        )

    # Actor exists in PEF but did NOT commit this action; someone else did → FALSE.
    return StateNativeDelegationResult(
        handled=True,
        outcome=StateNativeOutcome.ANSWER,
        user_visible_text=(
            f"No. {actor_entity.name} did not {event_tail.strip()}. "
            f"{competing_name} {action_desc}."
        ),
        solver_family=_SOLVER,
        epistemic_result=EpistemicResult.FALSE,
    )


def evaluate_actor_value_query(
    pef: PEFState,
    event_tail: str,
) -> StateNativeDelegationResult | None:
    """Evaluate ``Who [event]?`` — return the committed actor as a VALUE.

    Returns None when no matching action exists (caller falls through to LLM).
    """
    keywords = _event_keywords(event_tail)
    matches = _collect_action_matches(pef, keywords)

    if not matches:
        return None

    # Most recent committed actor wins.
    best_turn = max(m[0] for m in matches)
    latest = [m for m in matches if m[0] == best_turn]

    # Deduplicate by name.
    seen: dict[str, tuple[int, Entity, str, str]] = {}
    for m in latest:
        seen.setdefault(m[1].name, m)

    if len(seen) == 1:
        name, (_, _, relation, obj) = next(iter(seen.items()))
        realized = format_action_realization(relation, obj)
        return StateNativeDelegationResult(
            handled=True,
            outcome=StateNativeOutcome.ANSWER,
            user_visible_text=f"{name} {realized}.",
            solver_family=_SOLVER,
            epistemic_result=EpistemicResult.VALUE,
        )

    names = sorted(seen.keys())
    return StateNativeDelegationResult(
        handled=True,
        outcome=StateNativeOutcome.ANSWER,
        user_visible_text=(
            f"Multiple actors are on record: {', '.join(names)}. "
            "Which action did you mean?"
        ),
        solver_family=_SOLVER,
        epistemic_result=EpistemicResult.VALUE,
    )

"""State-native COMPARE query evaluator — Law C1/C3 enforcement."""

from __future__ import annotations

from aurora_lens.pef.state import PEFState
from aurora_lens.state_native_engine.contracts import (
    StateNativeDelegationResult,
    StateNativeOutcome,
    StateNativeSolverFamily,
)
from aurora_lens.state_native_engine.epistemic import EpistemicResult
from aurora_lens.state_native_engine.lexical import item_key

_COMPARE_SOLVER = StateNativeSolverFamily.COMMITTED_COMPARE_READ


def _compare_delegation(
    outcome: StateNativeOutcome,
    text: str,
    *,
    epistemic: EpistemicResult,
    stop_reason_code: str | None = None,
) -> StateNativeDelegationResult:
    return StateNativeDelegationResult(
        handled=True,
        outcome=outcome,
        user_visible_text=text,
        clarify_context=None,
        stop_reason_code=stop_reason_code,
        solver_family=_COMPARE_SOLVER,
        epistemic_result=epistemic,
    )


def evaluate_comparative_query(
    pef: PEFState,
    noun: str,
    adjective: str,
) -> StateNativeDelegationResult:
    """Answer 'Whose NOUN was ADJECTIVE?' from committed COMPARE relations.

    Matches by adjective and noun (both normalized via item_key). Latest turn wins
    when multiple matches exist for the same adjective+noun pair.
    """
    target_adj = adjective.strip().lower()
    target_noun = item_key(noun.strip())

    matches = [
        rel
        for rel in pef.get_relationships_by_relation("COMPARE")
        if (rel.relation_metadata or {}).get("adjective", "").lower() == target_adj
        and item_key((rel.relation_metadata or {}).get("noun", "")) == target_noun
    ]

    if not matches:
        return _compare_delegation(
            StateNativeOutcome.STOP,
            f"I cannot answer that from committed state: "
            f"no comparison for {noun} has been recorded.",
            epistemic=EpistemicResult.UNKNOWN,
            stop_reason_code="state_native_no_compare",
        )

    rel = max(matches, key=lambda r: r.source_turn)

    subj_ent = pef.entities.get(rel.subject_id)
    subj_name = subj_ent.name if subj_ent else "unknown"

    if rel.object_entity_id:
        comp_ent = pef.entities.get(rel.object_entity_id)
        comp_name = comp_ent.name if comp_ent else str(rel.object_literal or "")
    else:
        comp_name = str(rel.object_literal or "")

    adj_text = adjective.strip()
    response = f"{subj_name}'s {noun} was {adj_text} than {comp_name}'s."
    return _compare_delegation(
        StateNativeOutcome.ANSWER,
        response,
        epistemic=EpistemicResult.VALUE,
    )

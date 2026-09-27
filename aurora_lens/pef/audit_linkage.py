"""Audit–PEF linkage: governing posture at decision boundaries for ledger rows.

Each field records **which durable epistemic posture is in force** at a replay-stable
boundary (turn start / effect of this decision), not a narrative of prior events or
policy action names.

Used by :meth:`GovernanceBridge.log_decision` together with ``pef_snapshot``. See
roadmap ``audit-pef-linkage``.

**Semantic refinement for response-level refusal:**
When a refusal blocks a generated claim (e.g., an unsupported advisory), subsequent
state-native answers that query admitted PEF relationships are not globally contaminated.
The `state_native_handled` parameter allows `classify_pef_turn_start()` to distinguish:
- Response-level refusal: Blocks future generated claims; state-native queries remain valid
- State-level hold: Would contaminate required PEF relationships for downstream inference

This prevents false audit classification where state-native answers incorrectly inherit
a prior refusal's `HELD_REFUSAL` posture when they query uncontaminated relationships.
"""

from __future__ import annotations

from enum import StrEnum

from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
from aurora_lens.pef.state import (
    EPISTEMIC_MODE_AMBIGUITY,
    EPISTEMIC_MODE_REFUSAL,
    EPISTEMIC_MODE_STOP,
    PEFState,
)


class PefTurnClassification(StrEnum):
    """Durable epistemic posture governing the session at **turn entry** (before this turn mutates PEF)."""

    FRESH = "fresh"
    HELD_STATE = "held_state"
    HELD_AMBIGUITY = "held_ambiguity"
    HELD_REFUSAL = "held_refusal"
    STOPPED = "stopped"


class PefHoldTransition(StrEnum):
    """How durable epistemic **posture** is asserted or updated at this decision (not InterventionAction names).

    **Retired from live code** (may still appear in historical JSONL; we do not rewrite files
    or parse these strings): ``enter_ambiguity``, ``enter_refusal``, ``enter_stop``,
    ``replace_hold``, ``clear_ambiguity``. Nothing in this package maps or branches on them.
    """

    NONE = "none"
    AMBIGUITY = "ambiguity"
    REFUSAL = "refusal"
    STOP = "stop"
    SUPERSEDE = "supersede"
    RELEASE = "release"


def classify_pef_turn_start(
    pef: PEFState,
    *,
    state_native_handled: bool | None = None,
    state_native_source_status: str | None = None,
) -> PefTurnClassification:
    """Posture in force at turn entry from ``PEFState`` (before Lens applies this request).

    **LIMITATION: Global Hold Scoping (Not Branch-Scoped)**
    
    The current implementation treats all epistemic holds as globally contaminating.
    A response-level refusal (blocking a generated advisory claim) will prevent
    verification of ANY PEF relationships, even unrelated admitted ones.

    **Why this limitation exists:**
    Without branch/source-scoped contamination tracking, we cannot distinguish:
    - relationships sourced from the refused/generated claim's provenance chain
    - relationships that are admitted and completely independent

    **Semantic inaccuracy this creates:**
    If Dr Chen's advisory is refused (UNSUPPORTED_EVENT), and Dr Chen's PRESCRIBE
    relationship is admitted, a query about the prescription will still fail
    verification because the hold is global, not scoped to the advisory claim.

    **Conservative behavior (correct until infrastructure exists):**
    Only explicit ``state_native_source_status="admitted_uncontaminated"`` downgrades
    from HELD_REFUSAL. Without this, refusal holds remain globally governing to avoid
    false classification.

    **Future improvement requires:**
    1. Tracking provenance chains (dependency graphs, not just immediate provenance)
    2. Marking which relationships depend on which refusals (not available in current PEF)
    3. Scoped hold checking (is THIS relationship blocked by THIS hold, per-relationship)

    Until then, this approach trades precision for safety: false HELD_REFUSAL is better
    than false HELD_STATE.
    """
    hold = pef.epistemic_hold
    if hold:
        mode = hold.get("mode")
        if mode == EPISTEMIC_MODE_REFUSAL:
            # Only downgrade from HELD_REFUSAL to HELD_STATE if explicitly marked as
            # "admitted_uncontaminated" (proof of source integrity), not just "handled".
            if state_native_source_status == "admitted_uncontaminated":
                return PefTurnClassification.HELD_STATE
            return PefTurnClassification.HELD_REFUSAL
        if mode == EPISTEMIC_MODE_STOP:
            return PefTurnClassification.STOPPED
        if mode == EPISTEMIC_MODE_AMBIGUITY:
            return PefTurnClassification.HELD_AMBIGUITY
    if pef.pending_clarification:
        return PefTurnClassification.HELD_AMBIGUITY
    if pef.current_turn > 1 or pef.relationships:
        return PefTurnClassification.HELD_STATE
    return PefTurnClassification.FRESH


def _posture_class(
    pef: PEFState,
    *,
    state_native_handled: bool | None = None,
    state_native_source_status: str | None = None,
) -> PefTurnClassification:
    return classify_pef_turn_start(
        pef,
        state_native_handled=state_native_handled,
        state_native_source_status=state_native_source_status,
    )


def classify_hold_transition(
    decision: GovernanceDecision,
    pef_before: PEFState,
    pef_after: PEFState,
    *,
    state_native_handled: bool | None = None,
    state_native_source_status: str | None = None,
) -> PefHoldTransition:
    """Posture delta implied by PEF before/after this decision (same rules as Lens hold updates)."""
    act = decision.action
    before = _posture_class(
        pef_before,
        state_native_handled=state_native_handled,
        state_native_source_status=state_native_source_status,
    )
    after = _posture_class(
        pef_after,
        state_native_handled=state_native_handled,
        state_native_source_status=state_native_source_status,
    )

    if act in (InterventionAction.PASS, InterventionAction.SOFT_CORRECT):
        if before == PefTurnClassification.HELD_AMBIGUITY and after == PefTurnClassification.FRESH:
            return PefHoldTransition.RELEASE
        return PefHoldTransition.NONE

    if act == InterventionAction.CONTAIN:
        return PefHoldTransition.AMBIGUITY

    if act == InterventionAction.FORCE_REVISE:
        if before == PefTurnClassification.HELD_AMBIGUITY:
            return PefHoldTransition.SUPERSEDE
        return PefHoldTransition.REFUSAL

    if act == InterventionAction.HARD_STOP:
        if before == PefTurnClassification.HELD_AMBIGUITY:
            return PefHoldTransition.SUPERSEDE
        return PefHoldTransition.STOP

    return PefHoldTransition.NONE


def returns_to_holding(transition: PefHoldTransition) -> bool:
    """True when this decision leaves the session under a durable refusal/stop/ambiguity posture (or supersedes one)."""
    return transition in (
        PefHoldTransition.AMBIGUITY,
        PefHoldTransition.REFUSAL,
        PefHoldTransition.STOP,
        PefHoldTransition.SUPERSEDE,
    )

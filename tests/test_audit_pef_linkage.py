"""Unit tests for audit–PEF linkage (governing posture at boundaries, not event chronology)."""

from __future__ import annotations

from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
from aurora_lens.pef.audit_linkage import (
    PefHoldTransition,
    PefTurnClassification,
    classify_hold_transition,
    classify_pef_turn_start,
    returns_to_holding,
)
from aurora_lens.pef.state import (
    EPISTEMIC_HOLD_SCHEMA_VERSION,
    EPISTEMIC_MODE_AMBIGUITY,
    EPISTEMIC_MODE_REFUSAL,
    EPISTEMIC_MODE_STOP,
    PEFState,
)
from aurora_lens.verify.flags import Flag, FlagType


def _flag() -> Flag:
    return Flag(
        flag_type=FlagType.UNRESOLVED_REFERENT,
        entity_name="e",
        claim="c",
        evidence="ev",
        severity="warning",
    )


def test_turn_start_fresh_when_no_hold_no_pending():
    """No governing hold or pending slot → posture is fresh (no spurious held_* link)."""
    pef = PEFState()
    assert classify_pef_turn_start(pef) == PefTurnClassification.FRESH


def test_turn_start_ambiguous_when_pending_only():
    """Pending clarification without epistemic_hold still counts as ambiguity posture."""
    pef = PEFState()
    pef.pending_clarification = {"original_question": "who?"}
    assert classify_pef_turn_start(pef) == PefTurnClassification.HELD_AMBIGUITY


def test_turn_start_refusal_when_refusal_hold_governs():
    pef = PEFState()
    pef.epistemic_hold = {
        "schema_version": EPISTEMIC_HOLD_SCHEMA_VERSION,
        "mode": EPISTEMIC_MODE_REFUSAL,
        "since_turn": 2,
        "pathway_id": "P_REFUSE",
        "interaction_open": True,
        "commitment_closed": True,
        "last_audit_id": "x",
    }
    assert classify_pef_turn_start(pef) == PefTurnClassification.HELD_REFUSAL


def test_turn_start_stopped_when_stop_hold_governs():
    pef = PEFState()
    pef.epistemic_hold = {
        "schema_version": EPISTEMIC_HOLD_SCHEMA_VERSION,
        "mode": EPISTEMIC_MODE_STOP,
        "since_turn": 3,
        "pathway_id": "P_STOP",
        "interaction_open": False,
        "commitment_closed": True,
        "last_audit_id": "y",
    }
    assert classify_pef_turn_start(pef) == PefTurnClassification.STOPPED


def test_hold_mode_supersedes_pending_for_turn_classification():
    """When both hold and pending exist, serialized hold mode defines governing posture."""
    pef = PEFState()
    pef.pending_clarification = {"original_question": "x"}
    pef.epistemic_hold = {
        "schema_version": EPISTEMIC_HOLD_SCHEMA_VERSION,
        "mode": EPISTEMIC_MODE_REFUSAL,
        "since_turn": 1,
        "pathway_id": "P",
        "interaction_open": True,
        "commitment_closed": True,
        "last_audit_id": "",
    }
    assert classify_pef_turn_start(pef) == PefTurnClassification.HELD_REFUSAL


def test_no_posture_transition_when_pass_and_nothing_governed_before_or_after():
    """PASS with no durable hold before or after → transition none (no false posture link)."""
    before = PEFState()
    after = PEFState()
    d = GovernanceDecision(action=InterventionAction.PASS, flags=[], rationale="ok")
    assert classify_hold_transition(d, before, after) == PefHoldTransition.NONE


def test_no_posture_transition_when_soft_correct_and_already_fresh():
    before = PEFState()
    after = PEFState()
    d = GovernanceDecision(action=InterventionAction.SOFT_CORRECT, flags=[], rationale="fix")
    assert classify_hold_transition(d, before, after) == PefHoldTransition.NONE


def test_ambiguity_posture_governs_after_contain():
    before = PEFState()
    after = PEFState()
    after.epistemic_hold = {
        "schema_version": EPISTEMIC_HOLD_SCHEMA_VERSION,
        "mode": EPISTEMIC_MODE_AMBIGUITY,
        "since_turn": 0,
        "pathway_id": "P_ASK",
        "interaction_open": True,
        "commitment_closed": False,
        "last_audit_id": "",
    }
    after.pending_clarification = {"original_question": "q"}
    d = GovernanceDecision(
        action=InterventionAction.CONTAIN,
        flags=[_flag()],
        rationale="ask",
    )
    assert classify_hold_transition(d, before, after) == PefHoldTransition.AMBIGUITY


def test_stop_posture_governs_after_hard_stop_from_fresh():
    before = PEFState()
    after = PEFState()
    after.epistemic_hold = {
        "schema_version": EPISTEMIC_HOLD_SCHEMA_VERSION,
        "mode": EPISTEMIC_MODE_STOP,
        "since_turn": 0,
        "pathway_id": "P_STOP",
        "interaction_open": True,
        "commitment_closed": True,
        "last_audit_id": "",
    }
    d = GovernanceDecision(
        action=InterventionAction.HARD_STOP,
        flags=[_flag()],
        rationale="stop",
    )
    assert classify_hold_transition(d, before, after) == PefHoldTransition.STOP


def test_refusal_posture_governs_after_force_revise_from_fresh():
    before = PEFState()
    after = PEFState()
    after.epistemic_hold = {
        "schema_version": EPISTEMIC_HOLD_SCHEMA_VERSION,
        "mode": EPISTEMIC_MODE_REFUSAL,
        "since_turn": 0,
        "pathway_id": "P_REFUSE",
        "interaction_open": True,
        "commitment_closed": True,
        "last_audit_id": "",
    }
    d = GovernanceDecision(
        action=InterventionAction.FORCE_REVISE,
        flags=[_flag()],
        rationale="revise",
    )
    assert classify_hold_transition(d, before, after) == PefHoldTransition.REFUSAL


def test_supersede_when_refusal_replaces_ambiguity_posture():
    b = PEFState()
    b.epistemic_hold = {
        "schema_version": EPISTEMIC_HOLD_SCHEMA_VERSION,
        "mode": EPISTEMIC_MODE_AMBIGUITY,
        "since_turn": 0,
        "pathway_id": "P_ASK",
        "interaction_open": True,
        "commitment_closed": False,
        "last_audit_id": "",
    }
    a = PEFState()
    a.epistemic_hold = {
        "schema_version": EPISTEMIC_HOLD_SCHEMA_VERSION,
        "mode": EPISTEMIC_MODE_REFUSAL,
        "since_turn": 1,
        "pathway_id": "P_REFUSE",
        "interaction_open": True,
        "commitment_closed": True,
        "last_audit_id": "",
    }
    d = GovernanceDecision(
        action=InterventionAction.FORCE_REVISE,
        flags=[_flag()],
        rationale="revise",
    )
    assert classify_hold_transition(d, b, a) == PefHoldTransition.SUPERSEDE


def test_release_when_ambiguity_posture_cleared_to_fresh():
    b = PEFState()
    b.epistemic_hold = {
        "schema_version": EPISTEMIC_HOLD_SCHEMA_VERSION,
        "mode": EPISTEMIC_MODE_AMBIGUITY,
        "since_turn": 0,
        "pathway_id": "P_ASK",
        "interaction_open": True,
        "commitment_closed": False,
        "last_audit_id": "",
    }
    a = PEFState()
    d = GovernanceDecision(action=InterventionAction.PASS, flags=[], rationale="ok")
    assert classify_hold_transition(d, b, a) == PefHoldTransition.RELEASE


def test_pef_hold_transition_emits_only_canonical_posture_tokens():
    """Guards against reintroducing legacy strings (enter_*, replace_hold, clear_*) in the enum."""
    assert {e.value for e in PefHoldTransition} == frozenset(
        ("none", "ambiguity", "refusal", "stop", "supersede", "release")
    )


def test_returns_to_holding_matches_posture_tokens():
    assert returns_to_holding(PefHoldTransition.AMBIGUITY) is True
    assert returns_to_holding(PefHoldTransition.REFUSAL) is True
    assert returns_to_holding(PefHoldTransition.STOP) is True
    assert returns_to_holding(PefHoldTransition.SUPERSEDE) is True
    assert returns_to_holding(PefHoldTransition.NONE) is False
    assert returns_to_holding(PefHoldTransition.RELEASE) is False


def test_refusal_hold_requires_explicit_source_status_to_downgrade():
    """Response-level refusal downgrade requires explicit uncontaminated source proof.

    When a refusal blocks a generated advisory claim, later state-native queries
    should only classify as HELD_STATE if explicitly marked as sourcing from
    admitted_uncontaminated relationships. Without explicit proof, refusal hold
    remains globally governing (HELD_REFUSAL) to avoid false downgrades.
    """
    pef_with_refusal = PEFState()
    pef_with_refusal.epistemic_hold = {
        "schema_version": EPISTEMIC_HOLD_SCHEMA_VERSION,
        "mode": EPISTEMIC_MODE_REFUSAL,
        "since_turn": 2,
        "pathway_id": "P_REFUSE",
        "interaction_open": True,
        "commitment_closed": True,
        "last_audit_id": "cid-refuse",
    }
    # Without explicit source status, refusal hold governs globally
    assert classify_pef_turn_start(pef_with_refusal) == PefTurnClassification.HELD_REFUSAL
    
    # state_native_handled=True alone is NOT sufficient (engine may have answered
    # without checking PEF epistemic status)
    assert (
        classify_pef_turn_start(pef_with_refusal, state_native_handled=True)
        == PefTurnClassification.HELD_REFUSAL
    )
    
    # Only explicit "admitted_uncontaminated" downgrades to HELD_STATE
    assert (
        classify_pef_turn_start(pef_with_refusal, state_native_source_status="admitted_uncontaminated")
        == PefTurnClassification.HELD_STATE
    )


def test_refusal_hold_transition_source_status_propagated():
    """Hold transition respects source_status for downgrading refusal scope."""
    before = PEFState()
    before.epistemic_hold = {
        "schema_version": EPISTEMIC_HOLD_SCHEMA_VERSION,
        "mode": EPISTEMIC_MODE_REFUSAL,
        "since_turn": 1,
        "pathway_id": "P_REFUSE",
        "interaction_open": True,
        "commitment_closed": True,
        "last_audit_id": "cid-refuse",
    }
    after = PEFState()
    after.epistemic_hold = {
        "schema_version": EPISTEMIC_HOLD_SCHEMA_VERSION,
        "mode": EPISTEMIC_MODE_REFUSAL,
        "since_turn": 1,
        "pathway_id": "P_REFUSE",
        "interaction_open": True,
        "commitment_closed": True,
        "last_audit_id": "cid-refuse",
    }
    # PASS with refusal state before and after = NONE transition
    d = GovernanceDecision(action=InterventionAction.PASS, flags=[], rationale="ok")
    assert (
        classify_hold_transition(d, before, after)
        == PefHoldTransition.NONE
    )
    # Transition doesn't change with source_status (only classification does)
    assert (
        classify_hold_transition(d, before, after, state_native_source_status="admitted_uncontaminated")
        == PefHoldTransition.NONE
    )

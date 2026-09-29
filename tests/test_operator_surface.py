"""Tests for operator-plane PEF surface (session mode, hold summaries)."""

from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
from aurora_lens.pef.operator_surface import (
    binding_governance_decision_is_live,
    build_operator_pef_surface,
    session_mode_label,
)
from aurora_lens.pef.state import (
    EPISTEMIC_HOLD_SCHEMA_VERSION,
    EPISTEMIC_MODE_REFUSAL,
    EPISTEMIC_MODE_STOP,
    PEFState,
)


def test_session_mode_normal():
    pef = PEFState()
    assert session_mode_label(pef) == "normal"


def test_session_mode_held_ambiguity_pending():
    pef = PEFState()
    pef.pending_clarification = {"original_question": "q?", "failed_constraint": "UNRESOLVED_REFERENT"}
    assert session_mode_label(pef) == "held_ambiguity"


def test_session_mode_terminal_stop_closed():
    pef = PEFState()
    pef.epistemic_hold = {
        "schema_version": EPISTEMIC_HOLD_SCHEMA_VERSION,
        "mode": EPISTEMIC_MODE_STOP,
        "since_turn": 1,
        "pathway_id": "P_STOP_TERMINAL",
        "interaction_open": False,
        "commitment_closed": True,
        "last_audit_id": "cid-1",
    }
    assert session_mode_label(pef) == "terminal_stop_closed"


def test_build_operator_pef_surface_binding_from_live_decision():
    pef = PEFState()
    pef.current_turn = 3
    pef.epistemic_hold = {
        "schema_version": EPISTEMIC_HOLD_SCHEMA_VERSION,
        "mode": EPISTEMIC_MODE_REFUSAL,
        "since_turn": 2,
        "pathway_id": "P_REFUSE",
        "interaction_open": True,
        "commitment_closed": True,
        "last_audit_id": "cid-x",
    }
    d = GovernanceDecision(
        action=InterventionAction.FORCE_REVISE,
        flags=[],
        rationale="r",
        policy="strict",
        pathway_id="P_REFUSE",
        interaction_open=True,
        commitment_closed=True,
    )
    surf = build_operator_pef_surface(pef, d)
    assert surf["session_mode"] == "held_refusal"
    assert surf["epistemic_hold"]["active"] is True
    assert surf["epistemic_hold"]["interaction_open"] is True
    assert surf["binding_governance_state"]["action"] == "FORCE_REVISE"


def test_binding_governance_state_prefers_durable_when_decision_pathway_stale():
    """Stale Lens decision must not override the durable refusal hold on the session."""
    pef = PEFState()
    pef.current_turn = 3
    pef.epistemic_hold = {
        "schema_version": EPISTEMIC_HOLD_SCHEMA_VERSION,
        "mode": EPISTEMIC_MODE_REFUSAL,
        "since_turn": 2,
        "pathway_id": "P_REFUSE",
        "interaction_open": True,
        "commitment_closed": True,
        "last_audit_id": "cid-x",
    }
    stale = GovernanceDecision(
        action=InterventionAction.FORCE_REVISE,
        flags=[],
        rationale="r",
        policy="strict",
        pathway_id="OTHER_PATHWAY",
        interaction_open=True,
        commitment_closed=True,
    )
    assert binding_governance_decision_is_live(stale, pef) is False
    surf = build_operator_pef_surface(pef, stale)
    assert surf["binding_governance_state"]["action"] == "FORCE_REVISE"
    assert surf["binding_governance_state"]["pathway_id"] == "P_REFUSE"


def test_binding_governance_state_from_durable_when_no_lens_decision():
    pef = PEFState()
    pef.epistemic_hold = {
        "schema_version": EPISTEMIC_HOLD_SCHEMA_VERSION,
        "mode": EPISTEMIC_MODE_STOP,
        "since_turn": 1,
        "pathway_id": "P_STOP_TERMINAL",
        "interaction_open": False,
        "commitment_closed": True,
        "last_audit_id": "cid-z",
    }
    surf = build_operator_pef_surface(pef, None)
    assert surf["binding_governance_state"]["action"] == "HARD_STOP"
    assert surf["binding_governance_state"]["pathway_id"] == "P_STOP_TERMINAL"


def test_soft_correct_binding_governance_state_for_current_turn():
    pef = PEFState()
    d = GovernanceDecision(
        action=InterventionAction.SOFT_CORRECT,
        flags=[],
        rationale="r",
        policy="strict",
    )
    assert binding_governance_decision_is_live(d, pef) is True
    surf = build_operator_pef_surface(pef, d)
    assert surf["binding_governance_state"]["action"] == "SOFT_CORRECT"


def test_pass_decision_does_not_block_durable_binding_state():
    pef = PEFState()
    pef.epistemic_hold = {
        "schema_version": EPISTEMIC_HOLD_SCHEMA_VERSION,
        "mode": EPISTEMIC_MODE_REFUSAL,
        "since_turn": 1,
        "pathway_id": "P_REFUSE",
        "interaction_open": True,
        "commitment_closed": True,
        "last_audit_id": "",
    }
    d = GovernanceDecision(
        action=InterventionAction.PASS,
        flags=[],
        rationale="",
        policy="strict",
    )
    surf = build_operator_pef_surface(pef, d)
    assert surf["binding_governance_state"]["action"] == "FORCE_REVISE"

"""State-native epistemic insufficiency STOP must remain interaction-open (not terminal lockout)."""

from __future__ import annotations

import pytest

from aurora_lens.govern.decision import InterventionAction
from aurora_lens.govern.state_native_mapping import (
    governance_decision_from_state_native,
    is_state_native_epistemic_insufficiency_stop,
    reconcile_state_native_stop_after_policy_projection,
)
from aurora_lens.state_native_engine.contracts import (
    StateNativeDelegationResult,
    StateNativeOutcome,
    StateNativeSolverFamily,
)
from aurora_lens.state_native_engine.epistemic import EpistemicResult


def test_insufficiency_stop_is_reopenable_in_mapping() -> None:
    sn = StateNativeDelegationResult(
        handled=True,
        outcome=StateNativeOutcome.STOP,
        user_visible_text="I cannot answer that from committed state: no location (AT) recorded for Anna.",
        solver_family=StateNativeSolverFamily.COMMITTED_LOCATION_READ,
        epistemic_result=EpistemicResult.UNKNOWN,
        stop_reason_code="state_native_no_at",
    )
    assert is_state_native_epistemic_insufficiency_stop(sn)
    decision = governance_decision_from_state_native(sn)
    assert decision.action == InterventionAction.HARD_STOP
    assert decision.interaction_open is True
    assert decision.pathway_id == "P_STOP_REFUSE_CLEAN"


def test_policy_projection_reconcile_restores_reopenable_posture() -> None:
    sn = StateNativeDelegationResult(
        handled=True,
        outcome=StateNativeOutcome.STOP,
        user_visible_text="I cannot answer that from committed state.",
        solver_family=StateNativeSolverFamily.COMMITTED_INVENTORY_READ,
        epistemic_result=EpistemicResult.UNKNOWN,
        stop_reason_code="state_native_no_inventory",
    )
    decision = governance_decision_from_state_native(sn)
    # Simulate canonical policy projection (general:GP:STOP terminal overwrite).
    decision.pathway_id = "P_STOP_TERMINAL"
    decision.interaction_open = False
    reconcile_state_native_stop_after_policy_projection(decision, sn)
    assert decision.pathway_id == "P_STOP_REFUSE_CLEAN"
    assert decision.interaction_open is True


def test_non_insufficiency_stop_remains_terminal_in_mapping() -> None:
    sn = StateNativeDelegationResult(
        handled=True,
        outcome=StateNativeOutcome.STOP,
        user_visible_text="State-native stop: needs formalisation.",
        solver_family=StateNativeSolverFamily.CLOSED_WORLD,
        epistemic_result=None,
        stop_reason_code="state_native_needs_formalisation",
    )
    assert not is_state_native_epistemic_insufficiency_stop(sn)
    decision = governance_decision_from_state_native(sn)
    assert decision.action == InterventionAction.HARD_STOP
    assert decision.interaction_open is False
    assert decision.pathway_id == "P_STOP_TERMINAL"

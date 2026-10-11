"""A new governance outcome cannot silently dispose of an open matter."""
from copy import deepcopy

import pytest

from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
from aurora_lens.lens import (
    _apply_epistemic_hold_after_non_admit,
    _apply_state_native_pending_clarify,
    _assign_pending_without_replacing_other_matter,
)
from aurora_lens.pef.state import PEFState


@pytest.mark.parametrize("action", [InterventionAction.FORCE_REVISE, InterventionAction.HARD_STOP])
@pytest.mark.parametrize("constraint", ["UNRESOLVED_REFERENT", "INTERPRETATION_LIMIT"])
def test_refusal_does_not_discharge_pending_matter(action, constraint):
    pef = PEFState()
    pending = {"original_question": "original", "failed_constraint": constraint,
               "blocked_claims": [{"relation": "APPROVE", "obj": "this"}]}
    pef.pending_clarification = deepcopy(pending)
    decision = GovernanceDecision(action=action, flags=[], rationale="new refusal", policy="strict")
    _apply_epistemic_hold_after_non_admit(pef, decision, turn=2)
    assert pef.pending_clarification == pending
    assert pef.epistemic_hold["mode"] in {"refusal", "stop"}


@pytest.mark.parametrize("constraint", ["UNRESOLVED_REFERENT", "INTERPRETATION_LIMIT"])
def test_new_matter_with_same_constraint_does_not_replace_old_matter(constraint):
    pef = PEFState()
    pending = {"original_question": "original", "failed_constraint": constraint}
    pef.pending_clarification = deepcopy(pending)
    _assign_pending_without_replacing_other_matter(pef, {
        "original_question": "different", "failed_constraint": constraint})
    assert pef.pending_clarification == pending


def test_same_matter_can_refresh_its_clarification():
    pef = PEFState()
    pef.pending_clarification = {"original_question": "original", "failed_constraint": "UNRESOLVED_REFERENT"}
    updated = {**pef.pending_clarification, "candidate_entities": ["Emma"]}
    _assign_pending_without_replacing_other_matter(pef, updated)
    assert pef.pending_clarification == updated


def test_state_native_clarification_cannot_dispose_of_prior_matter():
    from aurora_lens.pef.span import Span
    pef = PEFState()
    pending = {"original_question": "original", "failed_constraint": "INTERPRETATION_LIMIT"}
    pef.pending_clarification = deepcopy(pending)
    _apply_state_native_pending_clarify(pef, original_question="new inventory question",
        detected_span=Span.PRESENT, clarify_context={"candidate_entities": ["Emma"]})
    assert pef.pending_clarification == pending

"""Unit tests for session-level unresolved-referent gate."""

from __future__ import annotations

from aurora_lens.pef.state import PEFState
from aurora_lens.pef.unresolved_referents import register_unresolved_referents
from aurora_lens.pef.unresolved_session_gate import (
    act_depends_on_open_registry,
    build_open_registry_context,
    dismiss_open_unresolved_referents,
    evaluate_unresolved_session_gate,
    is_meta_safe_turn,
    is_session_reset_request,
    is_unrelated_safe_turn,
)


def _seed_operator_contractor_registry(pef: PEFState) -> None:
    register_unresolved_referents(
        pef,
        turn=1,
        utterance=(
            "The operator informed the contractor that their certification had expired "
            "before the work commenced. Both parties hold certifications."
        ),
        tokens=["their"],
        candidate_entities=["Operator", "Contractor"],
        blocked_proposition=(
            "their certification had expired before the work commenced"
        ),
    )


class TestUnresolvedSessionGateModule:
    def test_act_depends_without_pronoun_suspension_admissible(self):
        pef = PEFState(session_id="t")
        _seed_operator_contractor_registry(pef)
        ctx = build_open_registry_context(pef)
        assert ctx is not None
        depends, kind = act_depends_on_open_registry("Is suspension admissible?", ctx)
        assert depends is True
        assert kind == "decision_seeking"

    def test_act_depends_responsible_party(self):
        pef = PEFState(session_id="t")
        _seed_operator_contractor_registry(pef)
        ctx = build_open_registry_context(pef)
        assert ctx is not None
        depends, kind = act_depends_on_open_registry("Determine the responsible party.", ctx)
        assert depends is True
        assert kind == "attribution_seeking"

    def test_meta_safe_not_blocked(self):
        assert is_meta_safe_turn("What information do you need?")

    def test_unrelated_joke_not_dependent(self):
        pef = PEFState(session_id="t")
        _seed_operator_contractor_registry(pef)
        gate = evaluate_unresolved_session_gate(pef, "Tell me a joke about penguins.")
        assert gate is not None
        assert gate.allow_through is True

    def test_session_reset_dismisses_registry(self):
        pef = PEFState(session_id="t")
        _seed_operator_contractor_registry(pef)
        assert is_session_reset_request("Ignore the previous scenario and reset.")
        gate = evaluate_unresolved_session_gate(pef, "Ignore the previous scenario and reset.")
        assert gate is not None
        assert gate.allow_through is True
        dismiss_open_unresolved_referents(pef)
        assert build_open_registry_context(pef) is None

    def test_candidate_specific_suspension_depends(self):
        pef = PEFState(session_id="t")
        _seed_operator_contractor_registry(pef)
        ctx = build_open_registry_context(pef)
        assert ctx is not None
        depends, _ = act_depends_on_open_registry(
            "Should the contractor be suspended?", ctx,
        )
        assert depends is True

    def test_unrelated_safe_phrase(self):
        assert is_unrelated_safe_turn("Tell me a joke about penguins.")

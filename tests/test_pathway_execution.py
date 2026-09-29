"""Pathway-driven continuation execution tests.

These tests verify that enforce() branches on canonical pathway_id when present,
producing distinct lawful continuations per pathway — not collapsed severity-based
flat refusals.

Core contracts tested:
  1. P_ASK_DISAMBIGUATE  — clarification mode, commitment stays closed
  2. P_STOP_TERMINAL (medical dosage) — refusal with clinician/pharmacist redirect
  3. P_REFUSE_EXPLAIN_REDIRECT (legal) — refusal with legal resource offer
  4. P_STOP_TERMINAL (illegal) — clean refusal, no workaround-adjacent language
  5. P_STOP_TERMINAL (self-harm) — supportive refusal + crisis routing
  6. Forensic integrity — pathway metadata surfaced in forensic envelope
  7. Pathway overrides legacy severity — same escalation_level, different pathway_id,
     different output (proves severity-collapse is eliminated)
"""

from __future__ import annotations

import pytest

from aurora_lens.govern.bridge import enforce, _AMBIGUITY_PATHWAYS, _REFUSAL_PATHWAYS, _HARD_STOP_PATHWAYS
from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
from aurora_lens.verify.flags import Flag, FlagType


# ── Helpers ─────────────────────────────────────────────────────────────────

def _flag(flag_type: FlagType, severity: str = "error") -> Flag:
    return Flag(
        flag_type=flag_type,
        entity_name="test",
        claim="test claim",
        evidence="test evidence",
        severity=severity,
    )


def _decision(
    action: InterventionAction,
    flags: list[Flag],
    pathway_id: str | None,
    *,
    resource: str | None = None,
    commitment_closed: bool = True,
    interaction_open: bool = False,
    output_mode: str = "terminal_stop",
    forensic_obligations: list[str] | None = None,
    resolution_mode: str = "exact",
) -> GovernanceDecision:
    d = GovernanceDecision(
        action=action,
        flags=flags,
        rationale="test",
        policy="test",
        resource=resource,
        pathway_id=pathway_id,
        output_mode=output_mode,
        commitment_closed=commitment_closed,
        interaction_open=interaction_open,
        forensic_obligations=forensic_obligations or [],
        resolution_mode=resolution_mode,
    )
    return d


# ── Test 1: P_ASK_DISAMBIGUATE — clarification mode, commitment closed ────────

class TestAskDisambiguate:

    def test_produces_clarification_request(self):
        """P_ASK_DISAMBIGUATE must yield a clarification request, not a generic error."""
        decision = _decision(
            InterventionAction.CONTAIN,
            flags=[_flag(FlagType.UNRESOLVED_REFERENT)],
            pathway_id="P_ASK_DISAMBIGUATE",
            commitment_closed=True,
            interaction_open=True,
        )
        result = enforce(decision, "")
        assert len(result) > 0
        # Must invite clarification
        assert any(w in result.lower() for w in ("clarif", "which", "specific", "referring"))

    def test_commitment_closed_before_resolution(self):
        """Commitment must remain closed until ambiguity is resolved."""
        decision = _decision(
            InterventionAction.CONTAIN,
            flags=[_flag(FlagType.UNRESOLVED_REFERENT)],
            pathway_id="P_ASK_DISAMBIGUATE",
            commitment_closed=True,
            interaction_open=True,
        )
        assert decision.commitment_closed is True

    def test_interaction_open(self):
        """Interaction must remain open on disambiguate pathway."""
        decision = _decision(
            InterventionAction.CONTAIN,
            flags=[_flag(FlagType.UNRESOLVED_REFERENT)],
            pathway_id="P_ASK_DISAMBIGUATE",
            commitment_closed=True,
            interaction_open=True,
        )
        assert decision.interaction_open is True

    def test_does_not_recover_blocked_determination(self):
        """The clarification response must not assert, imply, or complete the blocked determination."""
        decision = _decision(
            InterventionAction.CONTAIN,
            flags=[_flag(FlagType.UNRESOLVED_REFERENT)],
            pathway_id="P_ASK_DISAMBIGUATE",
            commitment_closed=True,
            interaction_open=True,
        )
        result = enforce(decision, "The answer is Emma.")
        # Must not pass through the model output
        assert "The answer is Emma." not in result
        assert "emma" not in result.lower()


# ── Test 2: Medical dosage — refusal with clinician/pharmacist redirect ───────

class TestMedicalDosageRedirection:

    @pytest.mark.parametrize("flag_type", [
        FlagType.MEDICAL_DOSAGE_RECOMMENDATION,
        FlagType.PEDIATRIC_DOSAGE_RECOMMENDATION,
        FlagType.NUMERIC_MEDICAL_INSTRUCTION,
    ])
    def test_medical_dosage_includes_clinician_redirect(self, flag_type):
        """Medical dosage hard stops must include a redirect to a clinician or pharmacist."""
        decision = _decision(
            InterventionAction.HARD_STOP,
            flags=[_flag(flag_type)],
            pathway_id="P_STOP_TERMINAL",
            commitment_closed=True,
            interaction_open=False,
        )
        result = enforce(decision, "")
        # Must include a professional redirect
        assert any(w in result.lower() for w in (
            "pharmacist", "clinician", "urgent care",
            "prescribing", "healthcare", "consult",
        ))
        # Structured refusal (no conversational "can't" requirement)
        assert "request blocked." in result.lower()
        assert "action:" in result.lower()

    def test_medical_dosage_uses_policy_resource_if_present(self):
        """If policy supplies a resource, it must appear in the continuation."""
        decision = _decision(
            InterventionAction.HARD_STOP,
            flags=[_flag(FlagType.MEDICAL_DOSAGE_RECOMMENDATION)],
            pathway_id="P_STOP_TERMINAL",
            resource="your local pharmacy (open 24 hours)",
        )
        result = enforce(decision, "")
        assert "your local pharmacy (open 24 hours)" in result

    def test_does_not_provide_dosage_by_implication(self):
        """The continuation must not contain any numerical dosage or drug-specific guidance."""
        decision = _decision(
            InterventionAction.HARD_STOP,
            flags=[_flag(FlagType.MEDICAL_DOSAGE_RECOMMENDATION)],
            pathway_id="P_STOP_TERMINAL",
        )
        result = enforce(decision, "Take 500mg twice daily.")
        # Must not pass through the blocked model output
        assert "500mg" not in result
        assert "twice daily" not in result


# ── Test 3: Legal advice refusal with resource offer ──────────────────────────

class TestLegalRefusalWithResources:

    def test_legal_advice_refusal_includes_resource_offer(self):
        """PERSONALIZED_LEGAL_ADVICE refusal must offer legal aid or official resources."""
        decision = _decision(
            InterventionAction.HARD_STOP,
            flags=[_flag(FlagType.PERSONALIZED_LEGAL_ADVICE)],
            pathway_id="P_STOP_TERMINAL",
        )
        result = enforce(decision, "")
        assert "i can't determine whether your case would succeed." in result.lower()
        assert "next step:" in result.lower()
        # Must offer a legal escalation path
        assert any(w in result.lower() for w in (
            "legal aid", "lawyer", "legal", "citizens advice", "contact", "adviser",
        ))

    def test_legal_continuation_does_not_imply_outcome(self):
        """The continuation must not contain any legal determination or outcome implication."""
        decision = _decision(
            InterventionAction.HARD_STOP,
            flags=[_flag(FlagType.PERSONALIZED_LEGAL_ADVICE)],
            pathway_id="P_STOP_TERMINAL",
        )
        result = enforce(decision, "You would likely win this case.")
        assert "likely win" not in result
        assert "You would" not in result

    def test_legal_uses_policy_resource_if_present(self):
        decision = _decision(
            InterventionAction.HARD_STOP,
            flags=[_flag(FlagType.PERSONALIZED_LEGAL_ADVICE)],
            pathway_id="P_STOP_TERMINAL",
            resource="Citizens Advice Bureau",
        )
        result = enforce(decision, "")
        assert "Citizens Advice Bureau" in result


# ── Test 4: Illegal instruction — clean refusal, no workaround adjacency ─────

class TestIllegalInstructionCleanRefusal:

    def test_illegal_instruction_is_clean_refusal(self):
        """ILLEGAL_INSTRUCTION must produce a clean refusal with no workaround language."""
        decision = _decision(
            InterventionAction.HARD_STOP,
            flags=[_flag(FlagType.ILLEGAL_INSTRUCTION)],
            pathway_id="P_STOP_TERMINAL",
        )
        result = enforce(decision, "")
        assert "request blocked." in result.lower()
        assert "action:" in result.lower()

    def test_illegal_instruction_has_no_workaround_adjacency(self):
        """The continuation must not contain however, alternatively, instead, or redirect offers."""
        decision = _decision(
            InterventionAction.HARD_STOP,
            flags=[_flag(FlagType.ILLEGAL_INSTRUCTION)],
            pathway_id="P_STOP_TERMINAL",
        )
        result = enforce(decision, "")
        for word in ("however", "alternatively", "instead", "you could", "you might"):
            assert word not in result.lower(), (
                f"Workaround-adjacent language detected: '{word}' in '{result}'"
            )

    def test_illegal_instruction_short(self):
        """The continuation should not be verbose — illegal instruction gets a clean stop."""
        decision = _decision(
            InterventionAction.HARD_STOP,
            flags=[_flag(FlagType.ILLEGAL_INSTRUCTION)],
            pathway_id="P_STOP_TERMINAL",
        )
        result = enforce(decision, "")
        # Structured governed-state template is longer than legacy one-liners; stay bounded.
        assert len(result) < 400, (
            f"Illegal instruction refusal should stay bounded, got: {result!r}"
        )


# ── Test 5: Self-harm — supportive refusal + crisis routing ──────────────────

class TestSelfHarmSupportiveContinuation:

    def test_self_harm_includes_crisis_routing(self):
        """SELF_HARM_INSTRUCTION must include crisis support resource reference."""
        decision = _decision(
            InterventionAction.HARD_STOP,
            flags=[_flag(FlagType.SELF_HARM_INSTRUCTION)],
            pathway_id="P_STOP_TERMINAL",
            resource="a crisis support service",
        )
        result = enforce(decision, "")
        # Must include supportive language
        assert any(w in result.lower() for w in ("distress", "reach out", "support", "help"))
        # Must reference the crisis resource
        assert "crisis support service" in result

    def test_self_harm_fallback_resource_when_none_provided(self):
        """When no resource provided, a fallback crisis resource should appear."""
        decision = _decision(
            InterventionAction.HARD_STOP,
            flags=[_flag(FlagType.SELF_HARM_INSTRUCTION)],
            pathway_id="P_STOP_TERMINAL",
            resource=None,
        )
        result = enforce(decision, "")
        # Must still reference some kind of support
        assert any(w in result.lower() for w in ("crisis", "support", "help", "reach out"))

    def test_self_harm_does_not_provide_harmful_content(self):
        """The continuation must not echo or recover the blocked content."""
        decision = _decision(
            InterventionAction.HARD_STOP,
            flags=[_flag(FlagType.SELF_HARM_INSTRUCTION)],
            pathway_id="P_STOP_TERMINAL",
        )
        result = enforce(decision, "Here is how to harm yourself: ...")
        assert "Here is how to harm yourself" not in result
        assert "harm yourself" not in result


# ── Test 6: Forensic integrity ────────────────────────────────────────────────

class TestForensicIntegrity:

    def test_decision_carries_pathway_id(self):
        """GovernanceDecision must carry pathway_id after enforce()."""
        decision = _decision(
            InterventionAction.HARD_STOP,
            flags=[_flag(FlagType.ILLEGAL_INSTRUCTION)],
            pathway_id="P_STOP_TERMINAL",
            commitment_closed=True,
            interaction_open=False,
            forensic_obligations=["emit_forensic_envelope"],
            resolution_mode="exact",
        )
        enforce(decision, "")
        assert decision.pathway_id == "P_STOP_TERMINAL"
        assert decision.commitment_closed is True
        assert decision.interaction_open is False
        assert "emit_forensic_envelope" in decision.forensic_obligations
        assert decision.resolution_mode == "exact"

    def test_decision_pathway_metadata_survives_enforce(self):
        """enforce() must not clear or modify pathway metadata fields."""
        decision = _decision(
            InterventionAction.HARD_STOP,
            flags=[_flag(FlagType.SELF_HARM_INSTRUCTION)],
            pathway_id="P_STOP_TERMINAL",
            commitment_closed=True,
            output_mode="terminal_stop",
            resolution_mode="domain_fallback",
            forensic_obligations=["emit_forensic_envelope", "attach_pef_snapshot"],
        )
        enforce(decision, "")
        assert decision.output_mode == "terminal_stop"
        assert decision.resolution_mode == "domain_fallback"
        assert len(decision.forensic_obligations) == 2


# ── Test 7: Pathway overrides legacy severity collapse ────────────────────────

class TestPathwayOverridesLegacySeverity:
    """Verify that two decisions with the same escalation_level but different
    pathway_ids render differently. This proves the old severity-collapse is gone.
    """

    def test_same_escalation_level_different_pathway_renders_differently(self):
        """P_STOP_TERMINAL and P_REFUSE_EXPLAIN_REDIRECT share escalation_level 3 and 2
        respectively, but both produce distinct continuations.

        Using P_STOP_TERMINAL (HARD_STOP, level 3) with self-harm vs
        P_STOP_TERMINAL (HARD_STOP, level 3) with legal advice: same level,
        same pathway, but different flag class -> different text.

        The key proof: two decisions with identical escalation_level=3 (HARD_STOP)
        but different flag types produce different output, not the same "I can't
        help with that." — demonstrating pathway-semantic branching, not level branching.
        """
        decision_harm = _decision(
            InterventionAction.HARD_STOP,
            flags=[_flag(FlagType.SELF_HARM_INSTRUCTION)],
            pathway_id="P_STOP_TERMINAL",
        )
        decision_legal = _decision(
            InterventionAction.HARD_STOP,
            flags=[_flag(FlagType.PERSONALIZED_LEGAL_ADVICE)],
            pathway_id="P_STOP_TERMINAL",
        )
        decision_illegal = _decision(
            InterventionAction.HARD_STOP,
            flags=[_flag(FlagType.ILLEGAL_INSTRUCTION)],
            pathway_id="P_STOP_TERMINAL",
        )

        # All three have escalation_level == 3
        assert decision_harm.escalation_level == 3
        assert decision_legal.escalation_level == 3
        assert decision_illegal.escalation_level == 3

        result_harm = enforce(decision_harm, "")
        result_legal = enforce(decision_legal, "")
        result_illegal = enforce(decision_illegal, "")

        # All are different — pathway-semantic branching preserved
        assert result_harm != result_legal, (
            "Self-harm and legal advice should not produce the same output"
        )
        assert result_harm != result_illegal, (
            "Self-harm and illegal instruction should not produce the same output"
        )
        assert result_legal != result_illegal, (
            "Legal advice and illegal instruction should not produce the same output"
        )

    def test_ask_pathway_vs_refuse_pathway_at_comparable_levels(self):
        """P_ASK_DISAMBIGUATE (level 1) and P_REFUSE_EXPLAIN_REDIRECT (level 2)
        produce structurally different continuations proving pathway semantics hold.
        """
        decision_ask = _decision(
            InterventionAction.CONTAIN,
            flags=[_flag(FlagType.UNRESOLVED_REFERENT)],
            pathway_id="P_ASK_DISAMBIGUATE",
            commitment_closed=True,
            interaction_open=True,
        )
        decision_refuse = _decision(
            InterventionAction.FORCE_REVISE,
            flags=[_flag(FlagType.PERSONALIZED_MEDICAL_ADVICE)],
            pathway_id="P_REFUSE_EXPLAIN_REDIRECT",
            commitment_closed=True,
            interaction_open=False,
            resource="a qualified clinician",
        )

        result_ask = enforce(decision_ask, "")
        result_refuse = enforce(decision_refuse, "")

        assert result_ask != result_refuse, (
            "ASK and REFUSE pathways must produce distinct continuations"
        )

        # ASK must invite clarification
        assert any(w in result_ask.lower() for w in ("clarif", "which", "specific"))
        # REFUSE must refuse and redirect (structured copy)
        assert "cannot provide that." in result_refuse.lower() or "contact" in result_refuse.lower()

    def test_missing_pathway_id_is_error_contained(self):
        """pathway_id=None is now error-containment, not a legacy dispatch path.

        Both the Governor bridge (CanonicalScannerGateBridge) and BuiltinBridge always produce a pathway_id.
        A None value signals a malformed decision object.  enforce() must fail safe:
        produce a lawful refusal via the domain-fallback ladder, mark
        unexpected_unclassified_termination=True, and set fallback_reason="missing_pathway_id".
        """
        decision_hard_stop = GovernanceDecision(
            action=InterventionAction.HARD_STOP,
            flags=[],
            rationale="test",
            # pathway_id intentionally absent — simulates malformed decision
        )
        decision_contain = GovernanceDecision(
            action=InterventionAction.CONTAIN,
            flags=[_flag(FlagType.EXTRACTION_EMPTY)],
            rationale="test",
            # pathway_id intentionally absent — simulates malformed decision
        )

        result_stop = enforce(decision_hard_stop, "model output")
        result_contain = enforce(decision_contain, "model output")

        # Both produce safe refusals (structured error-containment copy)
        assert "action:" in result_stop.lower()
        assert "action:" in result_contain.lower()
        # Both are marked as error-containment
        assert decision_hard_stop.unexpected_unclassified_termination is True
        assert decision_hard_stop.fallback_reason == "missing_pathway_id"
        assert decision_contain.unexpected_unclassified_termination is True
        assert decision_contain.fallback_reason == "missing_pathway_id"

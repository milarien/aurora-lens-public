"""Renderer leakage tests.

INVARIANT UNDER TEST
--------------------
No renderer may reflect blocked content to the user.

Specifically:
  - flag.evidence text must never appear in renderer output
  - flag.claim text must never appear in renderer output
  - decision.original_response must never appear in renderer output
  - decision.governance_note must never appear in renderer output

The governing principle:
  Clarification may expose admissible alternatives (PEF candidate labels).
  Refusal may not expose blocked substance.

These tests apply the hostile-lawyer standard:
  "Did the system formally refuse while effectively giving the answer anyway?"

Pathways under highest scrutiny:
  P_REFUSE_EXPLAIN_REDIRECT, P_REFUSE_ESCALATE_PRO, P_HANDOFF_SUMMARY
  P_STOP_TERMINAL (self-harm, medical, legal)
  Legacy clarify template
"""

from __future__ import annotations

import pytest

from aurora_lens.govern.bridge import (
    enforce,
    AmbiguityRendererContext,
    RefusalRendererContext,
    HardStopRendererContext,
    _render_ask_disambiguate,
    _render_ask_missing_fact,
    _render_refuse_explain_redirect,
    _render_refuse_escalate_pro,
    _render_handoff_summary,
    _render_stop_terminal,
    _render_stop_forensic,
    _domain_fallback_continuation,
    _domain_for_refusal,
)
from aurora_lens.govern.decision import (
    GovernanceDecision,
    GovernanceRuleResult,
    InterventionAction,
)
from aurora_lens.verify.flags import DefamationRole, Flag, FlagType


# ── Fixtures ─────────────────────────────────────────────────────────────────

BLOCKED_EVIDENCE = "Take 500mg metoprolol twice daily with food"
BLOCKED_CLAIM = "The patient should take 500mg metoprolol"
BLOCKED_RESPONSE = "Based on your weight and blood pressure, I recommend 500mg metoprolol twice daily."
GOVERNANCE_NOTE = "MEDICAL_DOSAGE_RECOMMENDATION: patient dosing — Take 500mg metoprolol twice daily"

SELF_HARM_EVIDENCE = "cut the wrist along the vein using a sharp blade"
LEGAL_EVIDENCE = "this is clearly unfair dismissal and you will win"
LEGAL_CLAIM = "the claimant has a strong unfair dismissal case"


def _decision(
    flag_type: FlagType,
    evidence: str = BLOCKED_EVIDENCE,
    claim: str = BLOCKED_CLAIM,
    pathway_id: str = "P_STOP_TERMINAL",
    resource: str | None = None,
    candidates: tuple[str, ...] = (),
    interaction_open: bool = False,
    allowed_continuations: tuple[str, ...] = (),
) -> GovernanceDecision:
    flag = Flag(
        flag_type=flag_type,
        entity_name="test_entity",
        claim=claim,
        evidence=evidence,
        severity="error",
        candidates=candidates,
    )
    d = GovernanceDecision(
        action=InterventionAction.HARD_STOP,
        flags=[flag],
        rationale=f"HARD_STOP: [{flag_type.name}]",
        original_response=BLOCKED_RESPONSE,
        governance_note=GOVERNANCE_NOTE,
        pathway_id=pathway_id,
        resource=resource,
        interaction_open=interaction_open,
        allowed_continuations=list(allowed_continuations),
    )
    return d


# ── Evidence must never reach the user ───────────────────────────────────────

class TestEvidenceNotReflected:

    def test_medical_stop_terminal_no_evidence(self):
        """P_STOP_TERMINAL medical: evidence text must not appear in output."""
        d = _decision(FlagType.MEDICAL_DOSAGE_RECOMMENDATION)
        out = enforce(d, BLOCKED_RESPONSE)
        assert BLOCKED_EVIDENCE not in out
        assert "500mg" not in out
        assert "metoprolol" not in out

    def test_self_harm_stop_terminal_no_evidence(self):
        """P_STOP_TERMINAL self-harm: evidence text must not appear in output."""
        d = _decision(FlagType.SELF_HARM_INSTRUCTION, evidence=SELF_HARM_EVIDENCE)
        out = enforce(d, "")
        assert SELF_HARM_EVIDENCE not in out
        assert "wrist" not in out
        assert "blade" not in out

    def test_legal_refuse_explain_no_evidence(self):
        """P_REFUSE_EXPLAIN_REDIRECT legal: evidence text must not appear in output."""
        d = _decision(
            FlagType.PERSONALIZED_LEGAL_ADVICE,
            evidence=LEGAL_EVIDENCE,
            claim=LEGAL_CLAIM,
            pathway_id="P_REFUSE_EXPLAIN_REDIRECT",
        )
        out = enforce(d, "")
        assert LEGAL_EVIDENCE not in out
        assert "unfair dismissal" not in out
        assert LEGAL_CLAIM not in out

    def test_legal_refuse_escalate_no_evidence(self):
        """P_REFUSE_ESCALATE_PRO legal: evidence must not appear in output."""
        d = _decision(
            FlagType.PERSONALIZED_LEGAL_ADVICE,
            evidence=LEGAL_EVIDENCE,
            pathway_id="P_REFUSE_ESCALATE_PRO",
        )
        out = enforce(d, "")
        assert "unfair dismissal" not in out
        assert LEGAL_EVIDENCE not in out

    def test_handoff_summary_no_evidence(self):
        """P_HANDOFF_SUMMARY: evidence must not appear in output."""
        d = _decision(
            FlagType.PERSONALIZED_MEDICAL_ADVICE,
            evidence=BLOCKED_EVIDENCE,
            pathway_id="P_HANDOFF_SUMMARY",
            interaction_open=True,
        )
        out = enforce(d, "")
        assert BLOCKED_EVIDENCE not in out
        assert "metoprolol" not in out


# ── Claim text must never reach the user ─────────────────────────────────────

class TestClaimNotReflected:

    def test_medical_stop_no_claim(self):
        out = enforce(_decision(FlagType.MEDICAL_DOSAGE_RECOMMENDATION), "")
        assert BLOCKED_CLAIM not in out

    def test_legal_refuse_no_claim(self):
        d = _decision(
            FlagType.PERSONALIZED_LEGAL_ADVICE,
            claim=LEGAL_CLAIM,
            pathway_id="P_REFUSE_EXPLAIN_REDIRECT",
        )
        out = enforce(d, "")
        assert LEGAL_CLAIM not in out
        assert "strong" not in out  # no implication of strength of case

    def test_financial_stop_no_claim(self):
        claim = "You should sell your Tesla shares immediately"
        d = _decision(
            FlagType.PERSONALIZED_FINANCIAL_ADVICE,
            claim=claim,
            evidence="sell Tesla shares immediately, the stock will drop",
        )
        out = enforce(d, "")
        assert "Tesla" not in out
        assert "sell" not in out
        assert claim not in out


# ── Original LLM response must never reach the user ──────────────────────────

class TestOriginalResponseNotReflected:

    def test_blocked_response_not_in_stop_terminal(self):
        d = _decision(FlagType.MEDICAL_DOSAGE_RECOMMENDATION)
        out = enforce(d, BLOCKED_RESPONSE)
        assert BLOCKED_RESPONSE not in out

    def test_blocked_response_not_in_refuse_redirect(self):
        d = _decision(
            FlagType.PERSONALIZED_LEGAL_ADVICE,
            pathway_id="P_REFUSE_EXPLAIN_REDIRECT",
        )
        out = enforce(d, BLOCKED_RESPONSE)
        assert BLOCKED_RESPONSE not in out

    def test_blocked_response_not_in_handoff(self):
        d = _decision(
            FlagType.PERSONALIZED_MEDICAL_ADVICE,
            pathway_id="P_HANDOFF_SUMMARY",
            interaction_open=True,
        )
        out = enforce(d, BLOCKED_RESPONSE)
        assert BLOCKED_RESPONSE not in out


# ── Hostile-lawyer standard: no implied conclusions ───────────────────────────

class TestNoImpliedConclusion:

    def test_legal_renderer_does_not_imply_case_strength(self):
        """'You would likely win' must never appear even implicitly."""
        d = _decision(
            FlagType.PERSONALIZED_LEGAL_ADVICE,
            evidence="this is clearly unfair dismissal and you will win",
            pathway_id="P_REFUSE_EXPLAIN_REDIRECT",
        )
        out = enforce(d, "You would likely win this case.")
        assert "win" not in out
        assert "strong" not in out
        assert "likely" not in out
        assert "unfair dismissal" not in out

    def test_medical_renderer_does_not_imply_urgency_beyond_policy(self):
        """Renderer must not invent a triage determination from evidence."""
        d = _decision(
            FlagType.PERSONALIZED_MEDICAL_ADVICE,
            evidence="chest pain, shortness of breath, left arm pain",
            pathway_id="P_REFUSE_EXPLAIN_REDIRECT",
        )
        out = enforce(d, "")
        assert "chest" not in out
        assert "arm" not in out
        # The renderer may mention urgency only via policy-sourced redirect,
        # not by reading the clinical evidence and rendering a triage judgment.
        assert "immediate" not in out or "emergency" in out  # if 'immediate' appears it must be the emergency template

    def test_self_harm_renderer_does_not_describe_method(self):
        """Self-harm renderer must not echo or paraphrase the described method."""
        d = _decision(
            FlagType.SELF_HARM_INSTRUCTION,
            evidence=SELF_HARM_EVIDENCE,
        )
        out = enforce(d, "")
        assert "cut" not in out
        assert "wrist" not in out
        assert "blade" not in out
        assert "vein" not in out

    def test_illegal_instruction_renderer_no_workaround_adjacency(self):
        """Illegal instruction: clean stop, no 'however', no alternatives."""
        d = _decision(
            FlagType.ILLEGAL_INSTRUCTION,
            evidence="synthesize TATP using acetone and hydrogen peroxide",
        )
        out = enforce(d, "")
        assert "however" not in out.lower()
        assert "alternative" not in out.lower()
        assert "acetone" not in out
        assert "TATP" not in out
        assert "Request blocked." in out
        assert "Action:" in out
        assert "Status: Blocked after model generation." in out


# ── Clarification: candidates permitted, evidence forbidden ──────────────────

class TestAmbiguityRendererSurface:

    def test_disambiguate_surfaces_candidates(self):
        """P_ASK_DISAMBIGUATE may surface PEF candidate labels — that is its job."""
        ctx = AmbiguityRendererContext(
            candidates=("Anna's sister", "Emma's sister"),
            missing_fields=(),
        )
        out = _render_ask_disambiguate(ctx)
        assert "Anna's sister" in out
        assert "Emma's sister" in out

    def test_disambiguate_no_candidates_generic(self):
        """Without candidates, ask generically — do not invent specifics."""
        ctx = AmbiguityRendererContext(candidates=(), missing_fields=())
        out = _render_ask_disambiguate(ctx)
        assert "More information required." in out
        assert "Action: Choose one option to continue" in out
        # Must not contain any content-derived text.
        assert BLOCKED_EVIDENCE not in out

    def test_missing_fact_surfaces_field_names(self):
        """P_ASK_MISSING_FACT may surface field names — not evidence text."""
        ctx = AmbiguityRendererContext(
            candidates=(),
            missing_fields=("dosage unit", "patient weight"),
        )
        out = _render_ask_missing_fact(ctx)
        assert "dosage unit" in out
        assert "patient weight" in out

    def test_missing_fact_no_evidence_text(self):
        """Field names are admissible. Raw evidence text is not."""
        ctx = AmbiguityRendererContext(
            candidates=(),
            missing_fields=("dosage unit",),
        )
        out = _render_ask_missing_fact(ctx)
        assert BLOCKED_EVIDENCE not in out
        assert "500mg" not in out

    def test_missing_fact_hides_internal_extraction_label(self):
        """Internal *extraction* sentinel must not appear as a raw system term."""
        ctx = AmbiguityRendererContext(
            candidates=(),
            missing_fields=("extraction",),
        )
        out = _render_ask_missing_fact(ctx)
        assert "extraction" not in out.lower()
        assert "clearer wording" in out.lower()

    def test_error_containment_no_evidence_reflection(self):
        """Error-containment path (pathway_id=None) must not reflect evidence to user."""
        d = GovernanceDecision(
            action=InterventionAction.CONTAIN,
            flags=[Flag(
                flag_type=FlagType.UNRESOLVED_REFERENT,
                entity_name="she",
                claim="referent unresolved",
                evidence="she went to the store",  # must not appear in output
                severity="warning",
            )],
            rationale="",
        )
        out = enforce(d, "original")
        assert "she went to the store" not in out
        assert "store" not in out


class TestInterpretationHardStopCopy:
    """EXTRACTION_* hard stops use governed copy (no extraction / verification jargon)."""

    def test_extraction_failed_matches_release_constant(self):
        ctx = HardStopRendererContext(
            flag_type=FlagType.EXTRACTION_FAILED,
            resource=None,
            interaction_open=False,
            domain=None,
        )
        out = _render_stop_terminal(ctx)
        assert "More information required." in out
        assert "Action: Choose one option to continue" in out
        assert "extraction" not in out.lower()

    def test_extraction_empty_open_vs_closed(self):
        open_ctx = HardStopRendererContext(
            flag_type=FlagType.EXTRACTION_EMPTY,
            resource=None,
            interaction_open=True,
            domain=None,
        )
        closed_ctx = HardStopRendererContext(
            flag_type=FlagType.EXTRACTION_EMPTY,
            resource=None,
            interaction_open=False,
            domain=None,
        )
        assert "Status: Waiting for clarification." in _render_stop_terminal(open_ctx)
        assert "Status: Blocked after model generation." in _render_stop_terminal(closed_ctx)


# ── Typed context surfaces only permitted fields ──────────────────────────────

class TestContextTypeSafety:

    def test_hard_stop_context_has_no_content_fields(self):
        """HardStopRendererContext must not carry evidence, claim, or response."""
        ctx = HardStopRendererContext(
            flag_type=FlagType.MEDICAL_DOSAGE_RECOMMENDATION,
            resource=None,
            interaction_open=False,
        )
        assert not hasattr(ctx, "evidence")
        assert not hasattr(ctx, "claim")
        assert not hasattr(ctx, "original_response")
        assert not hasattr(ctx, "governance_note")

    def test_refusal_context_has_no_content_fields(self):
        """RefusalRendererContext must not carry evidence, claim, or response."""
        ctx = RefusalRendererContext(
            flag_type=FlagType.PERSONALIZED_LEGAL_ADVICE,
            resource=None,
            interaction_open=True,
        )
        assert not hasattr(ctx, "evidence")
        assert not hasattr(ctx, "claim")
        assert not hasattr(ctx, "original_response")

    def test_ambiguity_context_has_no_evidence_field(self):
        """AmbiguityRendererContext carries candidates and field names, not evidence."""
        ctx = AmbiguityRendererContext(
            candidates=("Anna's sister",),
            missing_fields=(),
        )
        assert not hasattr(ctx, "evidence")
        assert not hasattr(ctx, "claim")
        assert not hasattr(ctx, "original_response")
        assert hasattr(ctx, "candidates")
        assert hasattr(ctx, "missing_fields")


class TestRefusalDomainDispatch:

    def test_domain_for_refusal_resolution_order(self):
        ctx = RefusalRendererContext(
            flag_type=FlagType.PERSONALIZED_FINANCIAL_ADVICE,
            resource=None,
            interaction_open=False,
            domain="legal",
            request_domain="finance",
        )
        assert _domain_for_refusal(ctx) == "legal"

        ctx2 = RefusalRendererContext(
            flag_type=FlagType.PERSONALIZED_FINANCIAL_ADVICE,
            resource=None,
            interaction_open=False,
            domain=None,
            request_domain="medical",
        )
        assert _domain_for_refusal(ctx2) == "medical"

        ctx3 = RefusalRendererContext(
            flag_type=FlagType.PERSONALIZED_LEGAL_ADVICE,
            resource=None,
            interaction_open=False,
            domain=None,
            request_domain=None,
        )
        assert _domain_for_refusal(ctx3) == "legal"

    def test_legal_refusal_never_uses_finance_wording(self):
        ctx = RefusalRendererContext(
            flag_type=FlagType.PERSONALIZED_FINANCIAL_ADVICE,
            resource="Citizens Advice Bureau",
            interaction_open=False,
            domain="legal",
            request_domain=None,
        )
        out = _render_refuse_explain_redirect(ctx)
        low = out.lower()
        assert "financial action" not in low
        assert "your case would succeed" in low
        assert "Citizens Advice Bureau" in out
        assert "?" not in out

    def test_enforce_refusal_uses_rule_result_domain_and_template_key(self):
        d = _decision(
            FlagType.PERSONALIZED_FINANCIAL_ADVICE,
            pathway_id="P_REFUSE_EXPLAIN_REDIRECT",
            interaction_open=False,
            resource="Citizens Advice Bureau",
        )
        d.rule_result = GovernanceRuleResult(
            rule_id="blocked.request.personalized_legal_outcome",
            domain="legal",
            outcome="HARD_STOP",
            flag_type=FlagType.PERSONALIZED_LEGAL_ADVICE,
            reason_code="personalized_legal_case_outcome",
            continuation_type="neutral_legal_timeline",
            interaction_open=False,
            escalation_target="Citizens Advice Bureau",
            user_facing_template_key="legal.blocked.personalized_case_outcome",
        )
        out = enforce(d, "")
        low = out.lower()
        assert "financial action" not in low
        assert "your case would succeed" in low
        assert "Citizens Advice Bureau" in out

    def test_medical_refusal_never_uses_finance_wording(self):
        ctx = RefusalRendererContext(
            flag_type=FlagType.PERSONALIZED_FINANCIAL_ADVICE,
            resource="your local urgent care centre",
            interaction_open=False,
            domain="medical",
            request_domain=None,
        )
        out = _render_refuse_explain_redirect(ctx)
        low = out.lower()
        assert "financial action" not in low
        assert "medical decision" in low
        assert "your local urgent care centre" in out
        assert "?" not in out

    def test_finance_refusal_uses_finance_wording(self):
        ctx = RefusalRendererContext(
            flag_type=FlagType.PERSONALIZED_FINANCIAL_ADVICE,
            resource="licensed financial adviser",
            interaction_open=False,
            domain="finance",
            request_domain=None,
        )
        out = _render_refuse_explain_redirect(ctx)
        low = out.lower()
        assert "financial action" in low
        assert "your case would succeed" not in low
        assert "medical decision" not in low
        assert "licensed financial adviser" in low
        assert "?" not in out


# ── Governed continuation: interaction_open produces follow-up question ────────

class TestGovernedContinuation:
    """Governor closes the prohibited determination, not necessarily the conversation.

    When interaction_open=True, the renderer must append a bounded follow-up
    question inside the lawful procedural corridor.
    When interaction_open=False, no follow-up question should appear.
    ILLEGAL_INSTRUCTION and TARGETED_DEFAMATION are always clean stops.
    """

    def _stop_decision(
        self,
        flag_type: FlagType,
        interaction_open: bool,
        resource: str | None = None,
        defamation_role: DefamationRole | None = None,
    ) -> GovernanceDecision:
        flag = Flag(
            flag_type=flag_type,
            entity_name="test_entity",
            claim="test claim",
            evidence="test evidence",
            severity="error",
            defamation_role=defamation_role,
        )
        return GovernanceDecision(
            action=InterventionAction.HARD_STOP,
            flags=[flag],
            rationale=f"HARD_STOP: [{flag_type.name}]",
            pathway_id="P_STOP_TERMINAL",
            resource=resource,
            interaction_open=interaction_open,
        )

    def test_medical_dosage_interaction_open_has_no_continuation_offer(self):
        """Dosage stop copy must not promise continuation help."""
        d = self._stop_decision(FlagType.MEDICAL_DOSAGE_RECOMMENDATION, interaction_open=True)
        out = enforce(d, "")
        # Must still refuse
        assert "Request blocked." in out
        assert "Action: Contact a prescribing clinician or appropriate urgent care service." in out
        assert "neutral timeline" not in out.lower()
        assert "i can help" not in out.lower()

    def test_medical_dosage_interaction_closed_no_follow_up(self):
        """Medical dosage with interaction_open=False must not append a follow-up question."""
        d = self._stop_decision(FlagType.MEDICAL_DOSAGE_RECOMMENDATION, interaction_open=False)
        out = enforce(d, "")
        assert "Status: Blocked after model generation." in out
        assert "?" not in out

    def test_personalized_medical_hard_stop_shows_symptom_corridor(self):
        """GP medical personalized gate uses symptom-summary safe corridor."""
        d = self._stop_decision(FlagType.PERSONALIZED_MEDICAL_ADVICE, interaction_open=True)
        out = enforce(d, "")
        assert "Reason: I can't make this medical decision" in out
        assert "To continue, send:" in out
        assert "symptoms" in out
        assert "Safe continuation available: symptom summary" in out

    def test_legal_advice_interaction_open_uses_only_allowed_continuations(self):
        """GP legal HARD_STOP: boundary + concrete neutral-timeline corridor copy."""
        d = self._stop_decision(FlagType.PERSONALIZED_LEGAL_ADVICE, interaction_open=True)
        d.allowed_continuations = ["neutral_timeline"]
        out = enforce(d, "")
        assert "Blocked — not admissible" not in out
        assert "I can't determine whether your case would succeed." in out
        assert "Next step:" in out
        assert "Provide:" in out
        assert "key dates" in out
        assert "relevant documents or messages" in out
        assert "Ask for a neutral summary or timeline instead" not in out
        assert "Action:" not in out
        assert "clearer summary" not in out.lower()
        assert "list documents" not in out.lower()
        assert "questions for" not in out.lower()
        # Must not imply outcome or case strength
        assert "win" not in out.lower()
        assert "strong" not in out.lower()

    def test_legal_advice_interaction_closed_no_follow_up(self):
        """Legal advice with interaction_open=False must not append a follow-up question."""
        d = self._stop_decision(FlagType.PERSONALIZED_LEGAL_ADVICE, interaction_open=False)
        out = enforce(d, "")
        assert "I can't determine whether your case would succeed." in out
        assert "Next step:" in out
        assert "Provide:" in out
        assert "?" not in out

    def test_self_harm_interaction_open_stays_with_person(self):
        """Self-harm with interaction_open=True must stay present — not abandon."""
        d = self._stop_decision(
            FlagType.SELF_HARM_INSTRUCTION,
            interaction_open=True,
            resource="a crisis support line",
        )
        out = enforce(d, "")
        # Must include crisis resource
        assert "crisis support line" in out
        # Must stay with the conversation
        assert any(w in out.lower() for w in ("here", "talk", "going on"))
        # Must not describe harm or method
        assert "cut" not in out
        assert "blade" not in out

    def test_self_harm_interaction_closed_no_follow_up(self):
        """Self-harm with interaction_open=False does not append follow-up."""
        d = self._stop_decision(FlagType.SELF_HARM_INSTRUCTION, interaction_open=False)
        out = enforce(d, "")
        assert any(w in out.lower() for w in ("distress", "reach out", "support", "help"))
        assert "here if you want to talk" not in out

    def test_financial_advice_interaction_open_without_continuation_has_no_offer(self):
        """Financial GP HARD_STOP must surface concrete safe corridor, not vague reformulation."""
        d = self._stop_decision(FlagType.PERSONALIZED_FINANCIAL_ADVICE, interaction_open=True)
        out = enforce(d, "")
        assert "Blocked — not admissible" not in out
        assert "I can't determine whether you should take this financial action." in out
        assert "Next step:" in out
        assert "Provide:" in out
        assert "amounts involved" in out
        assert "accounts or assets" in out
        assert "neutral financial facts summary" in out
        assert "stated as a question rather than an instruction" not in out
        assert "Ask for a neutral summary or timeline instead" not in out
        assert "questions for an adviser" not in out.lower()
        assert "work out what to ask" not in out.lower()

    def test_illegal_instruction_terminates_act_and_offers_transparency(self):
        """ILLEGAL_INSTRUCTION with interaction_open=True: act terminated, transparency offered.

        Bounded transparency ("I can explain the policy boundary") is not
        workaround adjacency — it makes the refusal legible without narrowing
        toward the blocked output.
        """
        d = self._stop_decision(FlagType.ILLEGAL_INSTRUCTION, interaction_open=True)
        out = enforce(d, "")
        assert "Request blocked." in out
        assert "Action: Reframe as a general educational question." in out
        # Must not narrow toward the blocked act
        assert "however" not in out.lower()
        assert "alternatively" not in out.lower()
        assert "?" not in out

    def test_illegal_instruction_closed_is_clean(self):
        """ILLEGAL_INSTRUCTION with interaction_open=False: clean act termination only."""
        d = self._stop_decision(FlagType.ILLEGAL_INSTRUCTION, interaction_open=False)
        out = enforce(d, "")
        assert "Request blocked." in out
        assert "Status: Blocked after model generation." in out

    def test_targeted_defamation_terminates_act_and_diverts_to_reporting(self):
        """TARGETED_DEFAMATION with interaction_open=True: act terminated, reporting diversion offered.

        Reporting-channel diversion does not launder defamation back in —
        it moves from accusation-production to lawful process.
        """
        d = self._stop_decision(
            FlagType.TARGETED_DEFAMATION,
            interaction_open=True,
            defamation_role=DefamationRole.REPORTER,
        )
        out = enforce(d, "")
        assert "Request blocked." in out
        assert any(w in out.lower() for w in ("reporting", "conduct", "channel"))
        assert "however" not in out.lower()
        assert "alternatively" not in out.lower()
        assert "?" not in out

    def test_follow_up_does_not_reflect_blocked_evidence(self):
        """The follow-up question must not contain any blocked content."""
        flag = Flag(
            flag_type=FlagType.PERSONALIZED_LEGAL_ADVICE,
            entity_name="test_entity",
            claim="the claimant has a strong unfair dismissal case",
            evidence="this is clearly unfair dismissal and you will win",
            severity="error",
        )
        d = GovernanceDecision(
            action=InterventionAction.HARD_STOP,
            flags=[flag],
            rationale="HARD_STOP: [PERSONALIZED_LEGAL_ADVICE]",
            pathway_id="P_STOP_TERMINAL",
            interaction_open=True,
        )
        out = enforce(d, "")
        assert "unfair dismissal" not in out
        assert "you will win" not in out
        assert "strong" not in out
        assert "claimant" not in out


# ── Domain fallback ladder: unknown / future flag types fail lawfully ──────────

class TestDomainFallbackLadder:
    """Tier-2 fallback: unknown flag types must fail into a lawful, auditable
    continuation, not a dead terminal wall.

    The Governor's role is to constrain, explain, summarise, route, or escalate
    without widening commitment. Only intrinsically hostile domains get a true
    terminal stop.

    Fallback ladder:
      medical  → REFUSE_ESCALATE_PRO shape (redirect to professional)
      legal    → REFUSE_EXPLAIN_REDIRECT shape (boundary + redirect)
      finance  → REFUSE_EXPLAIN_REDIRECT shape (boundary + redirect)
      harmful  → STOP_TERMINAL shape (clean stop)
      None     → final clean stop
    """

    def test_medical_fallback_is_refuse_escalate_pro_shape(self):
        """Unknown medical flag → GP medical personalized safe-corridor template."""
        out = _domain_fallback_continuation("medical", None, interaction_open=False)
        assert "Outside permitted scope." in out
        assert "I can't make this medical decision" in out
        assert "To continue, send:" in out
        assert "symptoms" in out
        assert "Safe continuation available: symptom summary" in out

    def test_medical_fallback_interaction_open_without_capability_has_no_offer(self):
        """Fallback uses same GP template (concrete symptom-summary corridor)."""
        out = _domain_fallback_continuation("medical", None, interaction_open=True)
        assert "?" not in out
        assert "neutral symptom and event summary" in out.lower()

    def test_medical_fallback_uses_policy_resource(self):
        """Policy-supplied resource must appear in medical fallback output."""
        out = _domain_fallback_continuation("medical", "your local urgent care centre", interaction_open=False)
        assert "Next step: Contact your local urgent care centre." in out

    def test_legal_fallback_is_refuse_explain_redirect_shape(self):
        """Unknown legal flag → GP legal safe-corridor template."""
        out = _domain_fallback_continuation("legal", None, interaction_open=False)
        assert "Blocked — not admissible" not in out
        assert "I can't determine whether your case would succeed." in out
        assert "Next step:" in out
        assert "Provide:" in out
        assert "neutral timeline" in out.lower()

    def test_legal_fallback_interaction_open_timeline_only(self):
        """Legal fallback renders timeline helper only when explicitly authorized."""
        out = _domain_fallback_continuation(
            "legal",
            None,
            interaction_open=True,
            allowed_continuations=("neutral_timeline",),
        )
        assert "neutral timeline" in out.lower()
        assert "Provide:" in out
        assert "Ask for a neutral summary or timeline instead" not in out
        assert "questions for" not in out.lower()
        assert "clearer summary" not in out.lower()
        # Must not imply outcome
        assert "win" not in out.lower()
        assert "strong" not in out.lower()
        assert "likely" not in out.lower()

    def test_finance_fallback_is_refuse_explain_redirect_shape(self):
        """Unknown finance flag → GP finance safe-corridor template."""
        out = _domain_fallback_continuation("finance", None, interaction_open=False)
        assert "Blocked — not admissible" not in out
        assert "I can't determine whether you should take this financial action." in out
        assert "Provide:" in out
        assert "neutral financial facts summary" in out
        assert "stated as a question rather than an instruction" not in out

    def test_finance_fallback_interaction_open_without_capability_has_no_offer(self):
        """Finance fallback must use GP finance corridor (no vague buy/sell coaching)."""
        out = _domain_fallback_continuation("finance", None, interaction_open=True)
        assert "neutral financial facts summary" in out.lower()
        # Must not provide specific financial guidance
        assert "invest" not in out.lower()
        assert "sell" not in out.lower()

    def test_domain_fallback_timeline_only_renders_single_mechanical_offer(self):
        out = _domain_fallback_continuation(
            "legal",
            None,
            interaction_open=True,
            allowed_continuations=("neutral_timeline",),
        )
        assert "I can't determine whether your case would succeed." in out
        assert "clearer summary" not in out.lower()
        assert "list documents" not in out.lower()

    def test_harmful_domain_is_clean_stop(self):
        """Unknown harmful-domain flag → STOP_TERMINAL shape: clean stop, no redirect."""
        out = _domain_fallback_continuation("harmful", None, interaction_open=True)
        # Clean stop — no redirect, no follow-up, no workaround adjacency
        assert "?" not in out
        assert "however" not in out.lower()
        assert "alternatively" not in out.lower()

    def test_none_domain_is_clean_stop(self):
        """Unknown domain (None) → final clean stop."""
        out = _domain_fallback_continuation(None, None, interaction_open=True)
        assert "?" not in out

    def test_fallback_never_carries_blocked_content(self):
        """Domain fallback must not carry any content from the blocked determination."""
        for domain in ("medical", "legal", "finance", "harmful", None):
            out = _domain_fallback_continuation(domain, None, interaction_open=True)
            # Sentinel strings from blocked content must never appear
            assert "500mg" not in out
            assert "metoprolol" not in out
            assert "unfair dismissal" not in out
            assert "Tesla" not in out
            assert "TATP" not in out
            assert "wrist" not in out

    def test_fallback_with_resource_uses_it(self):
        """All non-harmful domains must surface the policy-supplied resource."""
        resource = "Citizens Advice Bureau"
        for domain in ("medical", "legal", "finance"):
            out = _domain_fallback_continuation(domain, resource, interaction_open=False)
            assert resource in out, f"Resource not in {domain} fallback: {out!r}"


# ── Act-termination permanence invariant ──────────────────────────────────────
#
# Doctrine (2026-03-14):
#   Hostile-class handling must terminate the prohibited act without collapsing
#   the interaction into silence. Bounded transparency, routing, and support
#   remain lawful continuations where they do not create workaround adjacency.
#
# Four things separated cleanly:
#   1. Content stop       — prohibited output is not produced
#   2. Determination stop — system will not assess/decide the blocked question
#   3. Corridor continuation — interaction may continue in bounded mode
#   4. Human escalation   — external intervention may be triggered/prepared
#
# What these tests protect:
#   - Content stop: no prohibited content in output
#   - Determination stop: no assessment/determination of the blocked question
#   - No narrowing corridor: no "however", "alternatively", or path back to the act
#
# What these tests do NOT protect:
#   - Silence: bounded transparency and reporting-channel diversion are lawful
#   - Fixed text: exact phrasing may evolve; the invariants above must hold

class TestActTerminationPermanence:
    """Permanence guard for act-termination invariants on hostile-class flags.

    Covers: ILLEGAL_INSTRUCTION, TARGETED_DEFAMATION, SENSITIVE_PII_EXPOSURE,
    and the harmful-domain fallback (unknown hostile-class flags).

    Protects: content stop + determination stop + no narrowing corridor.
    Does NOT protect silence — bounded transparency is lawful continuation.
    """

    # Words that suggest a path back toward the blocked act.
    # "consider" and "however" are the key workaround markers.
    _NARROWING_WORDS = (
        "however", "alternatively", "instead, you could",
        "another option", "you could try", "you might try",
    )

    def _assert_act_terminated(self, out: str, label: str) -> None:
        """Assert that the act is terminated without producing a narrowing corridor."""
        # Determination stop: no assessment or outcome language
        assert "would win" not in out.lower(), f"{label}: determination must not appear"
        assert "strong case" not in out.lower(), f"{label}: determination must not appear"
        # No narrowing corridor back toward the blocked act
        for phrase in self._NARROWING_WORDS:
            assert phrase not in out.lower(), (
                f"{label}: narrowing phrase '{phrase}' must not appear"
            )

    def _assert_no_blocked_content(self, out: str, blocked_tokens: tuple[str, ...], label: str) -> None:
        """Assert that none of the blocked content tokens appear in output."""
        for token in blocked_tokens:
            assert token not in out, f"{label}: blocked token {token!r} must not appear"

    def test_illegal_instruction_terminates_act(self):
        """ILLEGAL_INSTRUCTION: act is terminated. No narrowing corridor back to the act."""
        from aurora_lens.govern.bridge import _hard_stop_text, FlagType
        for interaction_open in (True, False):
            out = _hard_stop_text(FlagType.ILLEGAL_INSTRUCTION, None, interaction_open)
            self._assert_act_terminated(out, f"ILLEGAL_INSTRUCTION(interaction_open={interaction_open})")
            assert "Request blocked." in out
            assert "Status: Blocked after model generation." in out

    def test_illegal_instruction_resource_does_not_appear(self):
        """ILLEGAL_INSTRUCTION: policy-supplied resource must not appear — no routing corridor."""
        from aurora_lens.govern.bridge import _hard_stop_text, FlagType
        for interaction_open in (True, False):
            out = _hard_stop_text(FlagType.ILLEGAL_INSTRUCTION, "a specialist resource", interaction_open)
            assert "specialist resource" not in out

    def test_illegal_instruction_bounded_transparency_when_open(self):
        """ILLEGAL_INSTRUCTION with interaction_open=True: policy-boundary explanation is lawful."""
        from aurora_lens.govern.bridge import _hard_stop_text, FlagType
        out = _hard_stop_text(FlagType.ILLEGAL_INSTRUCTION, None, interaction_open=True)
        assert "Action: Reframe as a general educational question." in out
        # Must not narrow toward the blocked act
        self._assert_act_terminated(out, "ILLEGAL_INSTRUCTION(open)")

    def test_illegal_instruction_closed_is_clean(self):
        """ILLEGAL_INSTRUCTION with interaction_open=False: clean act termination."""
        from aurora_lens.govern.bridge import _hard_stop_text, FlagType
        out = _hard_stop_text(FlagType.ILLEGAL_INSTRUCTION, None, interaction_open=False)
        assert "Request blocked." in out
        assert "Action: Reframe as a general educational question." in out
        assert "Status: Blocked after model generation." in out

    def test_targeted_defamation_terminates_act(self):
        """TARGETED_DEFAMATION: act is terminated. No accusation-production path."""
        from aurora_lens.govern.bridge import _hard_stop_text, FlagType
        for interaction_open in (True, False):
            out = _hard_stop_text(
                FlagType.TARGETED_DEFAMATION,
                None,
                interaction_open,
                None,
                (),
                DefamationRole.REPORTER,
            )
            self._assert_act_terminated(out, f"TARGETED_DEFAMATION(interaction_open={interaction_open})")
            assert "Request blocked." in out
            assert "Action:" in out
            assert "Status: Blocked after model generation." in out

    def test_targeted_defamation_reporting_channel_when_open(self):
        """TARGETED_DEFAMATION with interaction_open=True: reporting-channel diversion is lawful."""
        from aurora_lens.govern.bridge import _hard_stop_text, FlagType
        out = _hard_stop_text(
            FlagType.TARGETED_DEFAMATION,
            None,
            True,
            None,
            (),
            DefamationRole.REPORTER,
        )
        # Reporting-channel diversion is not laundering — it moves to lawful process
        assert any(w in out.lower() for w in ("reporting", "conduct", "channel"))
        # Must not produce the defamatory content or narrow toward it
        self._assert_act_terminated(out, "TARGETED_DEFAMATION(open)")

    def test_sensitive_pii_terminates_act(self):
        """SENSITIVE_PII_EXPOSURE: act is terminated. No data surfaced."""
        from aurora_lens.govern.bridge import _hard_stop_text, FlagType
        for interaction_open in (True, False):
            out = _hard_stop_text(FlagType.SENSITIVE_PII_EXPOSURE, None, interaction_open)
            self._assert_act_terminated(out, f"SENSITIVE_PII(interaction_open={interaction_open})")
            assert "Request blocked." in out
            assert "Action:" in out
            assert "Status: Blocked after model generation." in out

    def test_sensitive_pii_data_governance_transparency_when_open(self):
        """SENSITIVE_PII with interaction_open=True: data-governance transparency is lawful."""
        from aurora_lens.govern.bridge import _hard_stop_text, FlagType
        out = _hard_stop_text(FlagType.SENSITIVE_PII_EXPOSURE, None, interaction_open=True)
        assert any(w in out.lower() for w in ("support", "data", "governance", "boundary"))
        self._assert_act_terminated(out, "SENSITIVE_PII(open)")

    def test_harmful_domain_fallback_terminates_cleanly(self):
        """Unknown hostile-class fallback: clean act termination, no corridor.

        The 'harmful' domain fallback covers unknown future flag types that
        can't be safely routed. These remain a clean stop — no bounded transparency
        because the system doesn't know what the flag class is.
        """
        out = _domain_fallback_continuation("harmful", None, interaction_open=True)
        self._assert_act_terminated(out, "harmful-domain fallback")
        assert "?" not in out
        assert "Request blocked." in out
        assert "Action: Reframe as a general educational question." in out
        assert "Status: Blocked after model generation." in out

    def test_no_blocked_content_in_any_hostile_class_output(self):
        """No blocked content tokens appear in any hostile-class output."""
        from aurora_lens.govern.bridge import _hard_stop_text, FlagType
        hostile_flags = (
            FlagType.ILLEGAL_INSTRUCTION,
            FlagType.TARGETED_DEFAMATION,
            FlagType.SENSITIVE_PII_EXPOSURE,
        )
        blocked = ("TATP", "acetone", "500mg", "metoprolol", "John Smith", "wrist", "blade")
        for flag_type in hostile_flags:
            for interaction_open in (True, False):
                dr = (
                    DefamationRole.REPORTER
                    if flag_type == FlagType.TARGETED_DEFAMATION
                    else None
                )
                out = _hard_stop_text(flag_type, None, interaction_open, None, (), dr)
                self._assert_no_blocked_content(
                    out, blocked, f"{flag_type.name}(interaction_open={interaction_open})"
                )


# ── Error-containment audit marking ─────────────────────────────────────────

class TestErrorContainmentAuditMarking:
    """When Tier 3/4 of the domain-fallback ladder fires, enforce() must mark
    the GovernanceDecision with unexpected_unclassified_termination=True and
    a fallback_reason. These are error signals, not product features.

    Tier 3: domain="harmful" — unmapped hostile flag type reaching the system
    Tier 4: domain=None      — malformed decision (no flags, no domain signal)
    Unknown pathway_id       — routing gap in the canonical Governor path
    """

    def _decision(self, pathway_id: str | None = None, flags: list | None = None) -> GovernanceDecision:
        return GovernanceDecision(
            action=InterventionAction.HARD_STOP,
            flags=flags or [],
            rationale="test",
            pathway_id=pathway_id,
        )

    def test_harmful_domain_fallback_sets_unexpected_flag(self):
        """_domain_fallback_continuation('harmful') Tier 3 sets sentinel; enforce marks decision."""
        from aurora_lens.govern.bridge import _fallback_sentinel
        _fallback_sentinel.reason = None
        _domain_fallback_continuation("harmful", None, False)
        assert _fallback_sentinel.reason == "unmapped_hostile_flag"

    def test_none_domain_fallback_sets_malformed_decision_sentinel(self):
        """_domain_fallback_continuation(None) Tier 4 sets sentinel with 'malformed_decision'."""
        from aurora_lens.govern.bridge import _fallback_sentinel
        _fallback_sentinel.reason = None
        _domain_fallback_continuation(None, None, False)
        assert _fallback_sentinel.reason == "malformed_decision"

    def test_medical_domain_does_not_set_sentinel(self):
        """Tier 2 (medical) is lawful continuation — must not set the error sentinel."""
        from aurora_lens.govern.bridge import _fallback_sentinel
        _fallback_sentinel.reason = None
        _domain_fallback_continuation("medical", None, False)
        assert getattr(_fallback_sentinel, "reason", None) is None

    def test_legal_domain_does_not_set_sentinel(self):
        """Tier 2 (legal) is lawful continuation — must not set the error sentinel."""
        from aurora_lens.govern.bridge import _fallback_sentinel
        _fallback_sentinel.reason = None
        _domain_fallback_continuation("legal", None, False)
        assert getattr(_fallback_sentinel, "reason", None) is None

    def _make_flag(self, flag_type: FlagType) -> Flag:
        return Flag(
            flag_type=flag_type,
            entity_name="test",
            claim="x",
            evidence="x",
            severity="error",
        )

    def test_unknown_pathway_marks_decision_unmapped_pathway(self):
        """Unknown pathway_id: enforce() marks decision with unexpected_unclassified_termination=True."""
        decision = self._decision(
            pathway_id="P_UNKNOWN_FUTURE_PATHWAY",
            flags=[self._make_flag(FlagType.PERSONALIZED_LEGAL_ADVICE)],
        )
        enforce(decision, "original")
        assert decision.unexpected_unclassified_termination is True
        assert decision.fallback_reason is not None

    def test_unknown_pathway_fallback_reason_is_unmapped_pathway_id(self):
        """Unknown pathway_id with non-harmful domain: fallback_reason = 'unmapped_pathway_id'."""
        decision = self._decision(
            pathway_id="P_FUTURE_PATHWAY",
            flags=[self._make_flag(FlagType.PERSONALIZED_MEDICAL_ADVICE)],
        )
        enforce(decision, "original")
        # Medical domain → Tier 2 fires (lawful), sentinel not set → fallback_reason = "unmapped_pathway_id"
        assert decision.fallback_reason == "unmapped_pathway_id"

    def test_normal_hard_stop_does_not_mark_unexpected(self):
        """A normal P_STOP_TERMINAL with a specific handler must not set unexpected_unclassified_termination."""
        decision = self._decision(
            pathway_id="P_STOP_TERMINAL",
            flags=[self._make_flag(FlagType.ILLEGAL_INSTRUCTION)],
        )
        enforce(decision, "original")
        assert decision.unexpected_unclassified_termination is False
        assert decision.fallback_reason is None

    def test_normal_medical_stop_does_not_mark_unexpected(self):
        """A normal P_STOP_TERMINAL for MEDICAL_DOSAGE must not set unexpected_unclassified_termination."""
        decision = self._decision(
            pathway_id="P_STOP_TERMINAL",
            flags=[self._make_flag(FlagType.MEDICAL_DOSAGE_RECOMMENDATION)],
        )
        enforce(decision, "original")
        assert decision.unexpected_unclassified_termination is False
        assert decision.fallback_reason is None

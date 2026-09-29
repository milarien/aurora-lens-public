"""Continuation execution containment tests.

CONTRACT
--------
Matrix row chooses the lawful corridor.
Executor carries it out.
Renderer expresses it.
None of them may widen it.

Concretely, for each pathway_id:

  1. Context isolation     — typed renderer contexts expose only their declared
                             fields; no blocked content (original_response,
                             evidence, rationale, corrected_response) can reach
                             a renderer through any context path.

  2. Field immutability    — enforce() must not mutate commitment_closed,
                             interaction_open, or pathway_id on the decision
                             object it receives.

  3. Model output excluded — for every blocking pathway, the model_output
                             argument to enforce() does not appear in the
                             returned string.

  4. Blocked content       — decision fields that carry blocked content
     does not leak           (original_response, corrected_response,
                             rationale, governance_note) do not appear in
                             any renderer's output.

  5. Interaction openness  — interaction_open=False from the matrix row
                             suppresses all continuation invitations
                             ("Would you like", "I'm here if", etc.) across
                             every pathway.

  6. Resource provenance   — a renderer may only surface the string that
                             arrives in decision.resource; it cannot invent
                             a different external resource.

  7. Clean-stop isolation  — P_STOP_REFUSE_CLEAN for ILLEGAL_INSTRUCTION
                             explicitly suppresses decision.resource even
                             when that field is populated.  No professional
                             resource string appears in the output.

  8. Matrix-to-output      — for every non-ADMIT row in _TABLE, enforce()
     pipeline                with the row's parameters produces a non-empty,
                             lawful string without mutating the 6 lawful
                             continuation parameters.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from aurora_lens.govern.bridge import (
    AmbiguityRendererContext,
    HardStopRendererContext,
    RefusalRendererContext,
    _build_ambiguity_context,
    _build_hard_stop_context,
    _build_refusal_context,
    enforce,
)
from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
from aurora_lens.verify.flags import Flag, FlagType
from governor.continuation_matrix import CONTINUATION_MATRIX, ContinuationRow, _TABLE
from governor.models import AuthorityClass, ContinuationPathway, Domain, LensStatus


# ── Sentinels ─────────────────────────────────────────────────────────────────

_SENTINEL_RESOURCE = "SENTINEL_RESOURCE_FOR_CONTAINMENT_TEST_2025XYZ"
_POISON_BLOCKED    = "POISON_BLOCKED_CONTENT_MUST_NOT_SURFACE_XYZ"
_POISON_OUTPUT     = "POISON_MODEL_OUTPUT_MUST_NOT_APPEAR_IN_RENDERER_XYZ"

# Phrases that indicate an open continuation invitation.
# No blocking pathway with interaction_open=False may emit these.
_OPEN_INVITATIONS = (
    "Would you like",
    "I'm here if you want",
    "Could you clarify",
    "Could you provide",
    "I can help you prepare",
    "I can help identify",
    "I can help with general",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _flag(flag_type: FlagType, *, entity: str = "test", severity: str = "error") -> Flag:
    return Flag(
        flag_type=flag_type,
        entity_name=entity,
        claim="test claim",
        evidence="test evidence",
        severity=severity,
    )


def _flag_with_candidates(flag_type: FlagType, candidates: list[str]) -> Flag:
    return Flag(
        flag_type=flag_type,
        entity_name="test",
        claim="test claim",
        evidence="test evidence",
        severity="error",
        candidates=candidates,
    )


def _decision(
    action: InterventionAction,
    flags: list[Flag],
    pathway_id: str,
    *,
    resource: str | None = None,
    commitment_closed: bool = True,
    interaction_open: bool = False,
    output_mode: str = "terminal_stop",
    forensic_obligations: list[str] | None = None,
    resolution_mode: str = "exact",
) -> GovernanceDecision:
    return GovernanceDecision(
        action=action,
        flags=flags,
        rationale="test rationale",
        policy="test",
        resource=resource,
        pathway_id=pathway_id,
        output_mode=output_mode,
        commitment_closed=commitment_closed,
        interaction_open=interaction_open,
        forensic_obligations=forensic_obligations or [],
        resolution_mode=resolution_mode,
    )


# Pathway → (InterventionAction, representative FlagType)
_PATHWAY_EXEC: dict[str, tuple[InterventionAction, FlagType]] = {
    "P_ASK_DISAMBIGUATE":        (InterventionAction.CONTAIN,      FlagType.UNRESOLVED_REFERENT),
    "P_ASK_MISSING_FACT":        (InterventionAction.CONTAIN,      FlagType.UNRESOLVED_REFERENT),
    "P_REFUSE_EXPLAIN_REDIRECT": (InterventionAction.FORCE_REVISE, FlagType.PERSONALIZED_FINANCIAL_ADVICE),
    "P_REFUSE_ESCALATE_PRO":     (InterventionAction.FORCE_REVISE, FlagType.PERSONALIZED_MEDICAL_ADVICE),
    "P_HANDOFF_SUMMARY":         (InterventionAction.FORCE_REVISE, FlagType.PERSONALIZED_MEDICAL_ADVICE),
    "P_STOP_TERMINAL":           (InterventionAction.HARD_STOP,    FlagType.PERSONALIZED_MEDICAL_ADVICE),
    "P_STOP_FORENSIC":           (InterventionAction.HARD_STOP,    FlagType.SENSITIVE_PII_EXPOSURE),
    "P_STOP_ESCALATE":           (InterventionAction.HARD_STOP,    FlagType.EMERGENCY_TRIAGE_GUIDANCE),
    "P_STOP_ESCALATE_EMERGENCY": (InterventionAction.HARD_STOP,    FlagType.EMERGENCY_TRIAGE_GUIDANCE),
    "P_STOP_SUPPORTIVE_DEESCALATE": (InterventionAction.HARD_STOP, FlagType.SELF_HARM_INSTRUCTION),
    "P_STOP_REDIRECT_QUALIFIED": (InterventionAction.HARD_STOP,    FlagType.MEDICAL_DOSAGE_RECOMMENDATION),
    "P_STOP_REFUSE_CLEAN":       (InterventionAction.HARD_STOP,    FlagType.ILLEGAL_INSTRUCTION),
}


def _exec_params(pathway_id: str) -> tuple[InterventionAction, FlagType]:
    return _PATHWAY_EXEC.get(pathway_id, (InterventionAction.HARD_STOP, FlagType.PERSONALIZED_MEDICAL_ADVICE))


def _has_open_invitation(text: str) -> bool:
    return any(phrase in text for phrase in _OPEN_INVITATIONS)


def _has_refusal_language(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in (
        "not able", "can't", "cannot", "unable", "i'm not", "i can't",
        "not permitted", "request blocked",
    ))


# ── Non-ADMIT matrix rows for parametric tests ────────────────────────────────

_NON_ADMIT_ROWS: list[tuple[str, ContinuationRow]] = [
    (key, row)
    for key, row in _TABLE.items()
    if row.pathway_id != ContinuationPathway.P_ADMIT_STANDARD
]

_CLOSED_INTERACTION_ROWS: list[tuple[str, ContinuationRow]] = [
    (key, row)
    for key, row in _NON_ADMIT_ROWS
    if row.interaction_open is False
]

_CLEAN_STOP_ROWS: list[tuple[str, ContinuationRow]] = [
    (key, row)
    for key, row in _NON_ADMIT_ROWS
    if row.pathway_id == ContinuationPathway.P_STOP_REFUSE_CLEAN
]

_ESCALATION_TARGET_RENDER_ROWS: list[tuple[str, ContinuationRow]] = [
    (key, row)
    for key, row in _NON_ADMIT_ROWS
    if row.escalation_target is not None
    and row.pathway_id not in {
        ContinuationPathway.P_STOP_REFUSE_CLEAN,
        ContinuationPathway.P_HANDOFF_SUMMARY,
        ContinuationPathway.P_ASK_DISAMBIGUATE,
        ContinuationPathway.P_ASK_MISSING_FACT,
    }
]

_REFUSAL_VERBATIM_ROWS: list[tuple[str, ContinuationRow]] = [
    (key, row)
    for key, row in _NON_ADMIT_ROWS
    if row.pathway_id
    not in {
        ContinuationPathway.P_ASK_DISAMBIGUATE,
        ContinuationPathway.P_ASK_MISSING_FACT,
        ContinuationPathway.P_HANDOFF_SUMMARY,
    }
]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Context type isolation
# ═══════════════════════════════════════════════════════════════════════════════

class TestContextTypeIsolation:
    """Typed renderer contexts are information barriers: exactly the right fields,
    nothing more.  If a field that carries blocked content appeared here, the
    renderer would have a path to surface it.
    """

    def test_ambiguity_context_declared_fields(self):
        """AmbiguityRendererContext carries candidates/missing_fields/interaction_open plus
        optional routing fields for the UNRESOLVED_REFERENT path."""
        names = {f.name for f in dataclasses.fields(AmbiguityRendererContext)}
        required = {"candidates", "missing_fields", "interaction_open"}
        optional = {"flag_type", "ambiguous_token", "flag_claim", "flag_evidence", "original_question"}
        assert required <= names, f"Missing required fields: {required - names}"
        assert names <= required | optional, f"Unexpected fields in AmbiguityRendererContext: {names - (required | optional)}"

    def test_refusal_context_declared_fields(self):
        """RefusalRendererContext carries permitted routing identity fields for dispatch."""
        names = {f.name for f in dataclasses.fields(RefusalRendererContext)}
        assert names == {
            "flag_type",
            "resource",
            "interaction_open",
            "rule_id",
            "domain",
            "request_domain",
            "reason_code",
            "continuation_type",
            "user_facing_template_key",
        }, (
            f"Unexpected fields in RefusalRendererContext: {names}"
        )

    def test_hard_stop_context_declared_fields(self):
        """HardStopRendererContext carries only policy/taxonomy renderer inputs."""
        names = {f.name for f in dataclasses.fields(HardStopRendererContext)}
        assert names == {
            "flag_type", "resource", "interaction_open", "domain", "defamation_role", "pre_llm",
        }, (
            f"Unexpected fields in HardStopRendererContext: {names}"
        )

    def test_context_types_are_frozen(self):
        """All renderer context types are frozen — no mutation after construction."""
        assert AmbiguityRendererContext.__dataclass_params__.frozen is True  # type: ignore[attr-defined]
        assert RefusalRendererContext.__dataclass_params__.frozen is True    # type: ignore[attr-defined]
        assert HardStopRendererContext.__dataclass_params__.frozen is True   # type: ignore[attr-defined]

    def test_build_ambiguity_context_does_not_carry_blocked_fields(self):
        """_build_ambiguity_context must not copy original_response, evidence, or rationale."""
        d = _decision(
            InterventionAction.CONTAIN,
            [_flag_with_candidates(FlagType.UNRESOLVED_REFERENT, ["Anna", "Emma"])],
            "P_ASK_DISAMBIGUATE",
            interaction_open=True,
        )
        d.original_response = _POISON_BLOCKED
        d.corrected_response = _POISON_BLOCKED
        d.rationale = _POISON_BLOCKED
        d.governance_note = _POISON_BLOCKED

        ctx = _build_ambiguity_context(d)

        # Context carries only what's permitted
        assert isinstance(ctx, AmbiguityRendererContext)
        assert _POISON_BLOCKED not in str(ctx.candidates)
        assert _POISON_BLOCKED not in str(ctx.missing_fields)

    def test_build_refusal_context_does_not_carry_blocked_fields(self):
        """_build_refusal_context must not copy original_response or rationale."""
        d = _decision(
            InterventionAction.FORCE_REVISE,
            [_flag(FlagType.PERSONALIZED_MEDICAL_ADVICE)],
            "P_REFUSE_ESCALATE_PRO",
            resource="a licensed clinician",
            interaction_open=True,
        )
        d.original_response = _POISON_BLOCKED
        d.corrected_response = _POISON_BLOCKED
        d.rationale = _POISON_BLOCKED

        ctx = _build_refusal_context(d)

        assert isinstance(ctx, RefusalRendererContext)
        # resource comes from decision.resource, not from blocked fields
        assert ctx.resource == "a licensed clinician"
        assert _POISON_BLOCKED != ctx.resource

    def test_build_hard_stop_context_does_not_carry_blocked_fields(self):
        """_build_hard_stop_context must not copy original_response or rationale."""
        d = _decision(
            InterventionAction.HARD_STOP,
            [_flag(FlagType.ILLEGAL_INSTRUCTION)],
            "P_STOP_REFUSE_CLEAN",
            resource=None,
        )
        d.original_response = _POISON_BLOCKED
        d.rationale = _POISON_BLOCKED
        d.governance_note = _POISON_BLOCKED

        ctx = _build_hard_stop_context(d)

        assert isinstance(ctx, HardStopRendererContext)
        # No poison leaks into the context
        assert ctx.resource is None
        assert _POISON_BLOCKED != ctx.resource


# ═══════════════════════════════════════════════════════════════════════════════
# 2. enforce() field immutability
# ═══════════════════════════════════════════════════════════════════════════════

class TestEnforceFieldImmutability:
    """enforce() must not mutate the six lawful continuation parameters."""

    @pytest.mark.parametrize("pathway_id", list(_PATHWAY_EXEC))
    def test_commitment_closed_unchanged(self, pathway_id: str):
        action, flag_type = _exec_params(pathway_id)
        d = _decision(action, [_flag(flag_type)], pathway_id, commitment_closed=True)
        enforce(d, "model output")
        assert d.commitment_closed is True, (
            f"{pathway_id}: enforce() mutated commitment_closed to False"
        )

    @pytest.mark.parametrize("pathway_id,io", [
        (pid, io) for pid in _PATHWAY_EXEC for io in (True, False)
    ])
    def test_interaction_open_unchanged(self, pathway_id: str, io: bool):
        action, flag_type = _exec_params(pathway_id)
        d = _decision(action, [_flag(flag_type)], pathway_id, interaction_open=io)
        enforce(d, "model output")
        assert d.interaction_open is io, (
            f"{pathway_id}: enforce() mutated interaction_open from {io} to {d.interaction_open}"
        )

    @pytest.mark.parametrize("pathway_id", list(_PATHWAY_EXEC))
    def test_pathway_id_unchanged(self, pathway_id: str):
        action, flag_type = _exec_params(pathway_id)
        d = _decision(action, [_flag(flag_type)], pathway_id)
        enforce(d, "model output")
        assert d.pathway_id == pathway_id, (
            f"enforce() mutated pathway_id from {pathway_id!r} to {d.pathway_id!r}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Model output excluded from blocking pathways
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelOutputExclusion:
    """For blocking pathways, enforce(decision, model_output) must not include
    model_output in the returned string — the renderer is the sole author.
    """

    @pytest.mark.parametrize("pathway_id", list(_PATHWAY_EXEC))
    def test_model_output_not_in_result(self, pathway_id: str):
        action, flag_type = _exec_params(pathway_id)
        d = _decision(action, [_flag(flag_type)], pathway_id)
        result = enforce(d, _POISON_OUTPUT)
        assert _POISON_OUTPUT not in result, (
            f"{pathway_id}: model_output leaked into renderer output"
        )

    def test_pass_action_returns_model_output(self):
        """Sanity: PASS must return model_output unchanged (not a regression)."""
        d = _decision(InterventionAction.PASS, [], "P_ADMIT_STANDARD")
        result = enforce(d, "expected output")
        assert result == "expected output"

    def test_soft_correct_returns_model_output(self):
        """Sanity: SOFT_CORRECT must return model_output unchanged."""
        d = _decision(InterventionAction.SOFT_CORRECT, [], "P_ADMIT_STANDARD")
        result = enforce(d, "expected output")
        assert result == "expected output"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Blocked decision content does not leak
# ═══════════════════════════════════════════════════════════════════════════════

class TestBlockedContentDoesNotLeak:
    """Fields that carry blocked content (original_response, corrected_response,
    rationale, governance_note) must not appear in any renderer's output.
    """

    @pytest.mark.parametrize("pathway_id", list(_PATHWAY_EXEC))
    def test_original_response_not_in_output(self, pathway_id: str):
        action, flag_type = _exec_params(pathway_id)
        d = _decision(action, [_flag(flag_type)], pathway_id)
        d.original_response = _POISON_BLOCKED
        result = enforce(d, "")
        assert _POISON_BLOCKED not in result, (
            f"{pathway_id}: original_response leaked into renderer output"
        )

    @pytest.mark.parametrize("pathway_id", list(_PATHWAY_EXEC))
    def test_rationale_not_in_output(self, pathway_id: str):
        action, flag_type = _exec_params(pathway_id)
        d = _decision(action, [_flag(flag_type)], pathway_id)
        d.rationale = _POISON_BLOCKED
        result = enforce(d, "")
        assert _POISON_BLOCKED not in result, (
            f"{pathway_id}: rationale leaked into renderer output"
        )

    @pytest.mark.parametrize("pathway_id", list(_PATHWAY_EXEC))
    def test_governance_note_not_in_output(self, pathway_id: str):
        action, flag_type = _exec_params(pathway_id)
        d = _decision(action, [_flag(flag_type)], pathway_id)
        d.governance_note = _POISON_BLOCKED
        result = enforce(d, "")
        assert _POISON_BLOCKED not in result, (
            f"{pathway_id}: governance_note leaked into renderer output"
        )

    @pytest.mark.parametrize("pathway_id", list(_PATHWAY_EXEC))
    def test_corrected_response_not_in_output(self, pathway_id: str):
        action, flag_type = _exec_params(pathway_id)
        d = _decision(action, [_flag(flag_type)], pathway_id)
        d.corrected_response = _POISON_BLOCKED
        result = enforce(d, "")
        assert _POISON_BLOCKED not in result, (
            f"{pathway_id}: corrected_response leaked into renderer output"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Interaction openness respected
# ═══════════════════════════════════════════════════════════════════════════════

class TestInteractionOpenness:
    """interaction_open=False must suppress continuation invitations.
    interaction_open=True must not suppress them for pathways that add them.
    """

    # Pathways that explicitly append a follow-up when interaction_open=True
    _FOLLOWUP_PATHWAYS_CLOSED = {
        "P_STOP_TERMINAL":            FlagType.PERSONALIZED_MEDICAL_ADVICE,
        "P_STOP_FORENSIC":            FlagType.SENSITIVE_PII_EXPOSURE,
        "P_STOP_REDIRECT_QUALIFIED":  FlagType.MEDICAL_DOSAGE_RECOMMENDATION,
        "P_STOP_REFUSE_CLEAN":        FlagType.ILLEGAL_INSTRUCTION,
    }

    @pytest.mark.parametrize("pathway_id,flag_type", [
        ("P_STOP_TERMINAL",           FlagType.PERSONALIZED_MEDICAL_ADVICE),
        ("P_STOP_FORENSIC",           FlagType.SENSITIVE_PII_EXPOSURE),
        ("P_STOP_REDIRECT_QUALIFIED", FlagType.MEDICAL_DOSAGE_RECOMMENDATION),
        ("P_STOP_REFUSE_CLEAN",       FlagType.ILLEGAL_INSTRUCTION),
        ("P_REFUSE_EXPLAIN_REDIRECT", FlagType.PERSONALIZED_FINANCIAL_ADVICE),
        ("P_REFUSE_ESCALATE_PRO",     FlagType.PERSONALIZED_MEDICAL_ADVICE),
        ("P_STOP_ESCALATE",           FlagType.EMERGENCY_TRIAGE_GUIDANCE),
        ("P_STOP_ESCALATE_EMERGENCY", FlagType.EMERGENCY_TRIAGE_GUIDANCE),
    ])
    def test_interaction_open_false_suppresses_followup(
        self, pathway_id: str, flag_type: FlagType
    ):
        """When interaction_open=False, no continuation invitation in output."""
        action, _ = _exec_params(pathway_id)
        d = _decision(action, [_flag(flag_type)], pathway_id, interaction_open=False)
        result = enforce(d, "")
        assert not _has_open_invitation(result), (
            f"{pathway_id} with interaction_open=False emitted a continuation "
            f"invitation: {result!r}"
        )

    def test_supportive_pathway_keeps_conversation_open(self):
        """P_STOP_SUPPORTIVE_DEESCALATE has interaction_open=True in the matrix;
        the renderer must append 'I'm here if you want to talk'.
        """
        d = _decision(
            InterventionAction.HARD_STOP,
            [_flag(FlagType.SELF_HARM_INSTRUCTION)],
            "P_STOP_SUPPORTIVE_DEESCALATE",
            interaction_open=True,
        )
        result = enforce(d, "")
        assert "I'm here if you want to talk" in result, (
            f"P_STOP_SUPPORTIVE_DEESCALATE with interaction_open=True must keep "
            f"conversation open, but output was: {result!r}"
        )

    def test_supportive_pathway_always_open(self):
        """_render_stop_supportive_deescalate unconditionally keeps the
        conversation open regardless of interaction_open.  The renderer's
        docstring makes this explicit: 'interaction_open=True from policy' is a
        design premise — the matrix never assigns interaction_open=False to
        P_STOP_SUPPORTIVE_DEESCALATE.  Testing with interaction_open=False is
        a non-real scenario; this test documents the invariant instead.
        """
        row = CONTINUATION_MATRIX.lookup(
            Domain.CRISIS, AuthorityClass.GP, LensStatus.STOP
        )
        assert row.interaction_open is True, (
            "Matrix must always assign interaction_open=True to "
            "P_STOP_SUPPORTIVE_DEESCALATE"
        )

    def test_medical_dosage_open_no_longer_adds_preparation_offer(self):
        """Medical dosage hard-stop copy stays strict: no continuation promise."""
        d = _decision(
            InterventionAction.HARD_STOP,
            [_flag(FlagType.MEDICAL_DOSAGE_RECOMMENDATION)],
            "P_STOP_REDIRECT_QUALIFIED",
            interaction_open=True,
        )
        result = enforce(d, "")
        low = result.lower()
        assert "prepare" not in low
        assert "discuss" not in low
        assert "i can help" not in low

    def test_medical_dosage_closed_omits_preparation_offer(self):
        d = _decision(
            InterventionAction.HARD_STOP,
            [_flag(FlagType.MEDICAL_DOSAGE_RECOMMENDATION)],
            "P_STOP_REDIRECT_QUALIFIED",
            interaction_open=False,
        )
        result = enforce(d, "")
        assert not _has_open_invitation(result), (
            f"interaction_open=False on dosage pathway should omit invitation, "
            f"got: {result!r}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Escalation resource provenance
# ═══════════════════════════════════════════════════════════════════════════════

class TestEscalationResourceProvenance:
    """A renderer may only surface the resource supplied in decision.resource.
    It must use that string verbatim; it may not invent an alternative resource.
    """

    @pytest.mark.parametrize("pathway_id,flag_type", [
        ("P_REFUSE_ESCALATE_PRO",     FlagType.PERSONALIZED_MEDICAL_ADVICE),
        ("P_REFUSE_EXPLAIN_REDIRECT",  FlagType.PERSONALIZED_FINANCIAL_ADVICE),
        ("P_STOP_SUPPORTIVE_DEESCALATE", FlagType.SELF_HARM_INSTRUCTION),
        ("P_STOP_ESCALATE",            FlagType.EMERGENCY_TRIAGE_GUIDANCE),
        ("P_STOP_ESCALATE_EMERGENCY",  FlagType.EMERGENCY_TRIAGE_GUIDANCE),
        ("P_STOP_REDIRECT_QUALIFIED",  FlagType.MEDICAL_DOSAGE_RECOMMENDATION),
    ])
    def test_sentinel_resource_surfaced_verbatim(self, pathway_id: str, flag_type: FlagType):
        """When decision.resource is set to a unique sentinel, that sentinel and
        only that sentinel should appear as the resource in the rendered output.
        """
        action, _ = _exec_params(pathway_id)
        d = _decision(
            action,
            [_flag(flag_type)],
            pathway_id,
            resource=_SENTINEL_RESOURCE,
            interaction_open=True,
        )
        result = enforce(d, "")
        assert _SENTINEL_RESOURCE in result, (
            f"{pathway_id}: expected sentinel resource in output, got: {result!r}"
        )

    @pytest.mark.parametrize("pathway_id,flag_type", [
        ("P_REFUSE_ESCALATE_PRO",     FlagType.PERSONALIZED_MEDICAL_ADVICE),
        ("P_STOP_REDIRECT_QUALIFIED",  FlagType.MEDICAL_DOSAGE_RECOMMENDATION),
        ("P_STOP_SUPPORTIVE_DEESCALATE", FlagType.SELF_HARM_INSTRUCTION),
        ("P_STOP_ESCALATE_EMERGENCY",  FlagType.EMERGENCY_TRIAGE_GUIDANCE),
    ])
    def test_default_resource_not_invented_when_sentinel_set(
        self, pathway_id: str, flag_type: FlagType
    ):
        """When a specific resource is supplied, the renderer must not also emit
        its hardcoded default resource (that would double-route the user).
        The output should contain the sentinel but not both.
        """
        action, _ = _exec_params(pathway_id)
        d = _decision(
            action,
            [_flag(flag_type)],
            pathway_id,
            resource=_SENTINEL_RESOURCE,
            interaction_open=False,
        )
        result = enforce(d, "")
        # Output must contain the sentinel
        assert _SENTINEL_RESOURCE in result
        # Output must NOT also contain a separate well-known resource string
        # (e.g. the hardcoded default "a crisis support service")
        # We test this by checking that removing the sentinel leaves nothing
        # resource-like — simplified: result should contain exactly one
        # provider reference, not multiple.
        # Proxy check: common default strings not present alongside sentinel.
        assert "a crisis support service" not in result or _SENTINEL_RESOURCE == "a crisis support service"
        assert "a pharmacist" not in result
        assert "a qualified clinician" not in result


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Clean-stop isolation
# ═══════════════════════════════════════════════════════════════════════════════

class TestCleanStopIsolation:
    """P_STOP_REFUSE_CLEAN for ILLEGAL_INSTRUCTION must not surface the
    escalation target even when decision.resource is set.

    This is the strongest form of the resource containment contract: the
    renderer explicitly passes None to _hard_stop_text rather than ctx.resource,
    ensuring that even a policy-sourced resource cannot be used to imply the
    prohibited act has a lawful channel.
    """

    def test_illegal_instruction_no_resource_when_resource_none(self):
        """Clean stop with resource=None must not mention any professional resource."""
        d = _decision(
            InterventionAction.HARD_STOP,
            [_flag(FlagType.ILLEGAL_INSTRUCTION)],
            "P_STOP_REFUSE_CLEAN",
            resource=None,
            interaction_open=False,
        )
        result = enforce(d, "")
        assert _has_refusal_language(result)
        for professional_term in ("lawyer", "clinician", "adviser", "financial", "pharmacist"):
            assert professional_term.lower() not in result.lower(), (
                f"Clean stop (ILLEGAL_INSTRUCTION, resource=None) must not mention "
                f"a professional resource, but found {professional_term!r} in: {result!r}"
            )

    def test_illegal_instruction_resource_suppressed_even_when_set(self):
        """When decision.resource is set, ILLEGAL_INSTRUCTION clean stop must still
        suppress it — the renderer calls _hard_stop_text(flag_type, None, ...).
        """
        d = _decision(
            InterventionAction.HARD_STOP,
            [_flag(FlagType.ILLEGAL_INSTRUCTION)],
            "P_STOP_REFUSE_CLEAN",
            resource=_SENTINEL_RESOURCE,
            interaction_open=False,
        )
        result = enforce(d, "")
        assert _SENTINEL_RESOURCE not in result, (
            "ILLEGAL_INSTRUCTION clean stop must suppress decision.resource; "
            f"sentinel appeared in: {result!r}"
        )
        assert _has_refusal_language(result)

    def test_targeted_defamation_no_resource_when_resource_none(self):
        """Defamation clean stop with resource=None must not mention a resource."""
        d = _decision(
            InterventionAction.HARD_STOP,
            [_flag(FlagType.TARGETED_DEFAMATION)],
            "P_STOP_REFUSE_CLEAN",
            resource=None,
            interaction_open=False,
        )
        result = enforce(d, "")
        assert _has_refusal_language(result)
        assert _SENTINEL_RESOURCE not in result

    def test_clean_stop_no_workaround_adjacency(self):
        """Clean-stop output must not contain 'however', 'alternatively', or
        'instead' — language that implies a workaround path exists.
        """
        for flag_type in (FlagType.ILLEGAL_INSTRUCTION, FlagType.TARGETED_DEFAMATION):
            d = _decision(
                InterventionAction.HARD_STOP,
                [_flag(flag_type)],
                "P_STOP_REFUSE_CLEAN",
                resource=None,
                interaction_open=False,
            )
            result = enforce(d, "")
            for workaround_word in ("however", "alternatively", "instead", "but you could"):
                assert workaround_word.lower() not in result.lower(), (
                    f"Clean stop ({flag_type.name}) contains workaround-adjacent "
                    f"language {workaround_word!r}: {result!r}"
                )

    def test_clean_stop_no_escalation_target_from_matrix_row(self):
        """The matrix rows for legal:GP:STOP:ILLEGAL_INSTRUCTION and TARGETED_DEFAMATION
        have escalation_target=None.  Verify no resource appears in the output
        when decision.resource matches the matrix (i.e. None).
        """
        from governor.continuation_matrix import _TABLE
        for key in ("legal:GP:STOP:ILLEGAL_INSTRUCTION", "legal:GP:STOP:TARGETED_DEFAMATION"):
            row = _TABLE[key]
            assert row.escalation_target is None, f"{key} should have no escalation_target"
            flag_name = key.split(":")[-1]
            flag_type = FlagType[flag_name]
            d = _decision(
                InterventionAction.HARD_STOP,
                [_flag(flag_type)],
                "P_STOP_REFUSE_CLEAN",
                resource=None,           # matches matrix row
                interaction_open=True,   # open but still no resource
            )
            result = enforce(d, "")
            assert _has_refusal_language(result)
            assert "lawyer" not in result.lower(), (
                f"{key}: output mentions 'lawyer' when escalation_target=None"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Matrix-to-output pipeline (parametric)
# ═══════════════════════════════════════════════════════════════════════════════

class TestMatrixRowToOutputPipeline:
    """For every non-ADMIT row in the continuation matrix, execute enforce() with
    the row's lawful parameters and verify the full set of containment invariants.

    The decision is seeded from the row's 6 continuation parameters:
      - pathway_id         → row.pathway_id.value
      - commitment_closed  → row.commitment_closed
      - interaction_open   → row.interaction_open
      - resource           → row.escalation_target
      - output_mode        → row.output_mode.value
      - forensic_obligations → [fo.value for fo in row.forensic_obligations]

    No other parameters are used to construct the renderer output.
    """

    def _make(self, key: str, row: ContinuationRow) -> tuple[GovernanceDecision, str]:
        pathway_id = row.pathway_id.value
        action, flag_type = _exec_params(pathway_id)
        d = _decision(
            action,
            [_flag(flag_type)],
            pathway_id,
            resource=row.escalation_target,
            commitment_closed=row.commitment_closed,
            interaction_open=row.interaction_open,
            output_mode=row.output_mode.value,
            forensic_obligations=[fo.value for fo in row.forensic_obligations],
        )
        return d, enforce(d, _POISON_OUTPUT)

    # ── invariant A: output is non-empty ──────────────────────────────────────

    @pytest.mark.parametrize("key,row", _NON_ADMIT_ROWS, ids=[k for k, _ in _NON_ADMIT_ROWS])
    def test_output_is_non_empty(self, key: str, row: ContinuationRow):
        _, result = self._make(key, row)
        assert len(result.strip()) > 0, f"{key}: enforce() returned empty output"

    # ── invariant B: model output excluded ────────────────────────────────────

    @pytest.mark.parametrize("key,row", _NON_ADMIT_ROWS, ids=[k for k, _ in _NON_ADMIT_ROWS])
    def test_model_output_not_in_result(self, key: str, row: ContinuationRow):
        _, result = self._make(key, row)
        assert _POISON_OUTPUT not in result, (
            f"{key}: model_output leaked into renderer output"
        )

    # ── invariant C: commitment_closed unchanged ──────────────────────────────

    @pytest.mark.parametrize("key,row", _NON_ADMIT_ROWS, ids=[k for k, _ in _NON_ADMIT_ROWS])
    def test_commitment_closed_unchanged(self, key: str, row: ContinuationRow):
        d, _ = self._make(key, row)
        assert d.commitment_closed == row.commitment_closed, (
            f"{key}: enforce() mutated commitment_closed from "
            f"{row.commitment_closed} to {d.commitment_closed}"
        )

    # ── invariant D: interaction_open unchanged ───────────────────────────────

    @pytest.mark.parametrize("key,row", _NON_ADMIT_ROWS, ids=[k for k, _ in _NON_ADMIT_ROWS])
    def test_interaction_open_unchanged(self, key: str, row: ContinuationRow):
        d, _ = self._make(key, row)
        assert d.interaction_open == row.interaction_open, (
            f"{key}: enforce() mutated interaction_open from "
            f"{row.interaction_open} to {d.interaction_open}"
        )

    # ── invariant E: interaction_open=False suppresses follow-ups ─────────────

    @pytest.mark.parametrize(
        "key,row", _CLOSED_INTERACTION_ROWS, ids=[k for k, _ in _CLOSED_INTERACTION_ROWS]
    )
    def test_no_invitation_when_interaction_closed(self, key: str, row: ContinuationRow):
        _, result = self._make(key, row)
        assert not _has_open_invitation(result), (
            f"{key} (interaction_open=False) emitted a continuation "
            f"invitation: {result!r}"
        )

    # ── invariant F: clean-stop rows never surface a resource ────────────────

    @pytest.mark.parametrize("key,row", _CLEAN_STOP_ROWS, ids=[k for k, _ in _CLEAN_STOP_ROWS])
    def test_clean_stop_suppresses_resource(self, key: str, row: ContinuationRow):
        # Even if row.escalation_target were set, ILLEGAL_INSTRUCTION renderer
        # suppresses it.  The matrix currently sets escalation_target=None for
        # clean-stop rows, so this also verifies the matrix is consistent.
        _, result = self._make(key, row)
        for professional_term in ("lawyer", "clinician", "adviser", "financial", "pharmacist"):
            assert professional_term.lower() not in result.lower(), (
                f"{key}: clean-stop row mentions {professional_term!r}: {result!r}"
            )

    # ── invariant G: escalation_target from row is the only resource ─────────

    @pytest.mark.parametrize(
        "key,row", _ESCALATION_TARGET_RENDER_ROWS, ids=[k for k, _ in _ESCALATION_TARGET_RENDER_ROWS]
    )
    def test_resource_matches_row_escalation_target(self, key: str, row: ContinuationRow):
        """When the row's escalation_target is non-None AND the pathway surfaces
        the resource, the output must contain that exact escalation_target string.
        """
        _, result = self._make(key, row)
        assert row.escalation_target in result, (
            f"{key}: expected escalation_target {row.escalation_target!r} in output, "
            f"but got: {result!r}"
        )

    # ── invariant H: refusal language present for all blocking pathways ───────

    @pytest.mark.parametrize(
        "key,row", _REFUSAL_VERBATIM_ROWS, ids=[k for k, _ in _REFUSAL_VERBATIM_ROWS]
    )
    def test_refusal_language_present(self, key: str, row: ContinuationRow):
        _, result = self._make(key, row)
        assert _has_refusal_language(result), (
            f"{key}: blocking pathway must contain refusal language, got: {result!r}"
        )

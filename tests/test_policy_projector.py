"""Unit tests for PolicyProjector.

KEY INVARIANT UNDER TEST
------------------------
PROJECTION INVARIANT: projection is information-losing in only one direction.

  ALLOWED:   many Governor pathways → same InterventionAction
  FORBIDDEN: projection alters commitment_closed, forensic_obligations,
             escalation_target, or output_mode

These fields pass through unchanged from GovernorPolicy. If any test finds
the projector modifying them, it is a bug — the projector has become a
second policy brain.
"""

import pytest

from aurora_lens.govern.adapters.policy_projector import PolicyProjector
from aurora_lens.govern.adapters.runtime_types import (
    ContextResolutionProvenance,
    RuntimeDecisionProjection,
)
from aurora_lens.govern.decision import InterventionAction
from aurora_lens.governor.models import (
    AuthorityClass,
    ContinuationPathway,
    DisclosureType,
    Domain,
    ExposureLevel,
    ForensicObligation,
    GovernorPolicy,
    LensStatus,
    OutputMode,
    ResolutionMode,
    SpeechAct,
    UserClass,
)


def _policy(
    *,
    pathway_id: ContinuationPathway = ContinuationPathway.P_ADMIT_STANDARD,
    commitment_closed: bool = False,
    interaction_open: bool = True,
    lens_status: LensStatus = LensStatus.ADMIT,
    forensic_obligations: list[ForensicObligation] | None = None,
    escalation_target: str | None = None,
    output_mode: OutputMode = OutputMode.FULL_RESPONSE,
    exposure_level: ExposureLevel = ExposureLevel.MINIMAL,
    allowed_speech_acts: list[SpeechAct] | None = None,
    required_disclosures: list[DisclosureType] | None = None,
    resolution_mode: ResolutionMode = ResolutionMode.EXACT,
) -> GovernorPolicy:
    """Build a minimal GovernorPolicy for projection tests."""
    return GovernorPolicy(
        domain=Domain.GENERAL,
        authority_class=AuthorityClass.GP,
        lens_status=lens_status,
        user_class=UserClass.GENERAL,
        pathway_id=pathway_id,
        commitment_closed=commitment_closed,
        interaction_open=interaction_open,
        forensic_obligations=forensic_obligations or [],
        escalation_target=escalation_target,
        output_mode=output_mode,
        exposure_level=exposure_level,
        allowed_speech_acts=allowed_speech_acts or [],
        required_disclosures=required_disclosures or [],
        resolution_mode=resolution_mode,
    )


@pytest.fixture
def projector() -> PolicyProjector:
    return PolicyProjector()


# ── Pathway → InterventionAction table ───────────────────────────────────────

def test_admit_standard_projects_to_pass(projector):
    policy = _policy(pathway_id=ContinuationPathway.P_ADMIT_STANDARD)
    projection = projector.project(policy)
    assert projection.intervention_action == InterventionAction.PASS


def test_ask_disambiguate_projects_to_contain(projector):
    policy = _policy(
        pathway_id=ContinuationPathway.P_ASK_DISAMBIGUATE,
        lens_status=LensStatus.ASK,
        commitment_closed=True,
        output_mode=OutputMode.CLARIFICATION_REQUEST,
    )
    projection = projector.project(policy)
    assert projection.intervention_action == InterventionAction.CONTAIN


def test_ask_missing_fact_projects_to_contain(projector):
    policy = _policy(
        pathway_id=ContinuationPathway.P_ASK_MISSING_FACT,
        lens_status=LensStatus.ASK,
        commitment_closed=True,
        output_mode=OutputMode.CLARIFICATION_REQUEST,
    )
    projection = projector.project(policy)
    assert projection.intervention_action == InterventionAction.CONTAIN


def test_refuse_explain_redirect_projects_to_force_revise(projector):
    policy = _policy(
        pathway_id=ContinuationPathway.P_REFUSE_EXPLAIN_REDIRECT,
        lens_status=LensStatus.REFUSE,
        commitment_closed=True,
        output_mode=OutputMode.REFUSAL_WITH_EXPLANATION,
    )
    projection = projector.project(policy)
    assert projection.intervention_action == InterventionAction.FORCE_REVISE


def test_refuse_escalate_pro_projects_to_force_revise(projector):
    policy = _policy(
        pathway_id=ContinuationPathway.P_REFUSE_ESCALATE_PRO,
        lens_status=LensStatus.REFUSE,
        commitment_closed=True,
        output_mode=OutputMode.PRO_REFUSAL,
    )
    projection = projector.project(policy)
    assert projection.intervention_action == InterventionAction.FORCE_REVISE


def test_handoff_summary_projects_to_force_revise(projector):
    policy = _policy(
        pathway_id=ContinuationPathway.P_HANDOFF_SUMMARY,
        lens_status=LensStatus.REFUSE,
        commitment_closed=True,
    )
    projection = projector.project(policy)
    assert projection.intervention_action == InterventionAction.FORCE_REVISE


def test_stop_terminal_projects_to_hard_stop(projector):
    policy = _policy(
        pathway_id=ContinuationPathway.P_STOP_TERMINAL,
        lens_status=LensStatus.STOP,
        commitment_closed=True,
        interaction_open=False,
        output_mode=OutputMode.TERMINAL_STOP,
    )
    projection = projector.project(policy)
    assert projection.intervention_action == InterventionAction.HARD_STOP


def test_stop_forensic_projects_to_hard_stop(projector):
    policy = _policy(
        pathway_id=ContinuationPathway.P_STOP_FORENSIC,
        lens_status=LensStatus.STOP,
        commitment_closed=True,
        interaction_open=False,
        output_mode=OutputMode.FORENSIC_STOP,
    )
    projection = projector.project(policy)
    assert projection.intervention_action == InterventionAction.HARD_STOP


# ── PROJECTION INVARIANT: pass-through fields are never altered ───────────────

def test_commitment_closed_passes_through_unchanged(projector):
    """commitment_closed is NOT altered by the projector. PROJECTION INVARIANT."""
    for closed in (True, False):
        # Only use closed=False with ADMIT status (invariant enforcement)
        ls = LensStatus.ADMIT if not closed else LensStatus.REFUSE
        pathway = ContinuationPathway.P_ADMIT_STANDARD if not closed else ContinuationPathway.P_REFUSE_EXPLAIN_REDIRECT
        policy = _policy(
            pathway_id=pathway,
            commitment_closed=closed,
            lens_status=ls,
        )
        projection = projector.project(policy)
        assert projection.commitment_closed == closed, (
            f"commitment_closed={closed} was altered by projector"
        )


def test_forensic_obligations_pass_through_unchanged(projector):
    """forensic_obligations are NOT altered by the projector. PROJECTION INVARIANT."""
    obligations = [ForensicObligation.EMIT_FORENSIC_ENVELOPE]
    policy = _policy(
        pathway_id=ContinuationPathway.P_STOP_TERMINAL,
        lens_status=LensStatus.STOP,
        commitment_closed=True,
        forensic_obligations=obligations,
    )
    projection = projector.project(policy)
    assert projection.forensic_obligations == obligations


def test_escalation_target_passes_through_unchanged(projector):
    """escalation_target is NOT altered by the projector. PROJECTION INVARIANT."""
    policy = _policy(
        pathway_id=ContinuationPathway.P_REFUSE_ESCALATE_PRO,
        lens_status=LensStatus.REFUSE,
        commitment_closed=True,
        escalation_target="licensed_clinician",
    )
    projection = projector.project(policy)
    assert projection.escalation_target == "licensed_clinician"


def test_output_mode_passes_through_unchanged(projector):
    """output_mode is NOT altered by the projector. PROJECTION INVARIANT."""
    policy = _policy(
        pathway_id=ContinuationPathway.P_STOP_FORENSIC,
        lens_status=LensStatus.STOP,
        commitment_closed=True,
        output_mode=OutputMode.FORENSIC_STOP,
    )
    projection = projector.project(policy)
    assert projection.output_mode == OutputMode.FORENSIC_STOP


def test_exposure_level_passes_through_unchanged(projector):
    """exposure_level is NOT altered by the projector."""
    policy = _policy(
        pathway_id=ContinuationPathway.P_STOP_TERMINAL,
        lens_status=LensStatus.STOP,
        commitment_closed=True,
        exposure_level=ExposureLevel.FULL,
    )
    projection = projector.project(policy)
    assert projection.exposure_level == ExposureLevel.FULL


def test_resolution_mode_passes_through_unchanged(projector):
    """resolution_mode is NOT altered by the projector."""
    policy = _policy(
        pathway_id=ContinuationPathway.P_ADMIT_STANDARD,
        resolution_mode=ResolutionMode.DOMAIN_FALLBACK,
    )
    projection = projector.project(policy)
    assert projection.resolution_mode == ResolutionMode.DOMAIN_FALLBACK


# ── Projector is a pure function ──────────────────────────────────────────────

def test_projector_is_pure_same_policy_same_result(projector):
    """Calling project() twice with the same policy returns equal projections."""
    policy = _policy(
        pathway_id=ContinuationPathway.P_STOP_TERMINAL,
        lens_status=LensStatus.STOP,
        commitment_closed=True,
    )
    projection_a = projector.project(policy)
    projection_b = projector.project(policy)
    assert projection_a == projection_b


def test_projector_is_pure_no_state_accumulation(projector):
    """Multiple calls do not cause state accumulation on the projector instance."""
    policies = [
        _policy(pathway_id=ContinuationPathway.P_ADMIT_STANDARD),
        _policy(
            pathway_id=ContinuationPathway.P_STOP_TERMINAL,
            lens_status=LensStatus.STOP,
            commitment_closed=True,
        ),
        _policy(pathway_id=ContinuationPathway.P_ADMIT_STANDARD),
    ]
    actions = [projector.project(p).intervention_action for p in policies]
    assert actions[0] == InterventionAction.PASS
    assert actions[1] == InterventionAction.HARD_STOP
    assert actions[2] == InterventionAction.PASS  # not contaminated by STOP call


# ── Provenance pass-through ───────────────────────────────────────────────────

def test_provenance_carried_through_to_projection(projector):
    """Provenance supplied to project() appears unchanged in the projection."""
    prov = ContextResolutionProvenance(
        domain_source="operator_channel",
        authority_source="config",
        user_class_source="header",
    )
    policy = _policy()
    projection = projector.project(policy, provenance=prov)
    assert projection.provenance == prov


def test_provenance_none_when_not_supplied(projector):
    """When no provenance supplied, projection.provenance is None."""
    projection = projector.project(_policy())
    assert projection.provenance is None


# ── ADMIT pathway with no commitment — distinct from REFUSE/STOP ──────────────

def test_admit_pathway_commitment_open(projector):
    """P_ADMIT_STANDARD with commitment_closed=False → PASS + commitment remains open."""
    policy = _policy(
        pathway_id=ContinuationPathway.P_ADMIT_STANDARD,
        commitment_closed=False,
        interaction_open=True,
    )
    projection = projector.project(policy)
    assert projection.intervention_action == InterventionAction.PASS
    assert projection.commitment_closed is False
    assert projection.interaction_open is True

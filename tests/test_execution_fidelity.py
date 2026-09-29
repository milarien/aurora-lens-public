"""Execution-fidelity contract tests.

These tests prove that resolved continuation rows are carried through the
full pipeline — ContinuationMatrix → PolicyResolver → PolicyProjector →
enforce() — without widening or mutation, and make explicit which
ContinuationRow fields are user-facing versus audit-only.

CONTRACTS
---------
1. Resolver fidelity   — PolicyResolver.resolve() maps every _TABLE key to a
                         GovernorPolicy whose pathway_id, commitment_closed,
                         interaction_open, output_mode, escalation_target, and
                         forensic_obligations all match the row exactly.

2. Projector fidelity  — PolicyProjector.project() passes those six fields
                         through unchanged.  intervention_action is the only
                         translation product; all other fields are pass-through.

3. Non-widening at     — commitment_closed is never relaxed by the projector.
   projector stage       interaction_open is never widened.
                         escalation_target is never altered.

4. Audit-only field    — pathway_id.value, output_mode.value, and every
   suppression           ForensicObligation.value string never appear verbatim
                         in renderer output.  These are governance routing
                         metadata with no user-visible meaning.

5. ASK field taxonomy  — escalation_target in ASK rows is audit metadata
                         identifying the supervising authority.  It is present
                         (non-None) in domain-specific corridors but MUST NOT
                         appear in renderer output.  ASK renderers work from
                         PEF-derived candidates/missing_fields, not from the
                         escalation_target.

6. STOP/REFUSE field   — escalation_target in STOP/REFUSE rows IS user-facing.
   taxonomy              It MUST appear in renderer output (except for
                         P_STOP_REFUSE_CLEAN, which suppresses all resource
                         strings by design).
"""
from __future__ import annotations

from typing import Optional

import pytest

from aurora_lens.govern.adapters.policy_projector import (
    PolicyProjector,
    _PATHWAY_FALLBACK,
    _PATHWAY_TO_ACTION,
)
from aurora_lens.govern.bridge import enforce
from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
from aurora_lens.verify.flags import Flag, FlagType
from governor.continuation_matrix import CONTINUATION_MATRIX, ContinuationRow, _TABLE
from governor.models import (
    AuthorityClass,
    ContinuationPathway,
    Domain,
    LensStatus,
    UserClass,
)
from governor.resolver import PolicyResolver


# ── Helpers ───────────────────────────────────────────────────────────────────

_USER_CLASS_BY_VALUE: dict[str, UserClass] = {uc.value: uc for uc in UserClass}

# Canonical mapping from pathway_id to (InterventionAction, FlagType).
# FlagType is chosen to activate the correct renderer branch in enforce().
_PATHWAY_EXEC: dict[ContinuationPathway, tuple[InterventionAction, FlagType]] = {
    ContinuationPathway.P_ADMIT_STANDARD:             (InterventionAction.PASS,        FlagType.UNSUPPORTED_ATTRIBUTE),
    ContinuationPathway.P_ASK_DISAMBIGUATE:           (InterventionAction.CONTAIN,     FlagType.UNRESOLVED_REFERENT),
    ContinuationPathway.P_ASK_MISSING_FACT:           (InterventionAction.CONTAIN,     FlagType.UNRESOLVED_REFERENT),
    ContinuationPathway.P_REFUSE_EXPLAIN_REDIRECT:    (InterventionAction.FORCE_REVISE, FlagType.PERSONALIZED_FINANCIAL_ADVICE),
    ContinuationPathway.P_REFUSE_ESCALATE_PRO:        (InterventionAction.FORCE_REVISE, FlagType.PERSONALIZED_MEDICAL_ADVICE),
    ContinuationPathway.P_HANDOFF_SUMMARY:            (InterventionAction.FORCE_REVISE, FlagType.PERSONALIZED_MEDICAL_ADVICE),
    ContinuationPathway.P_STOP_TERMINAL:              (InterventionAction.HARD_STOP,   FlagType.PERSONALIZED_MEDICAL_ADVICE),
    ContinuationPathway.P_STOP_FORENSIC:              (InterventionAction.HARD_STOP,   FlagType.SENSITIVE_PII_EXPOSURE),
    ContinuationPathway.P_STOP_ESCALATE:              (InterventionAction.HARD_STOP,   FlagType.EMERGENCY_TRIAGE_GUIDANCE),
    ContinuationPathway.P_STOP_ESCALATE_EMERGENCY:    (InterventionAction.HARD_STOP,   FlagType.EMERGENCY_TRIAGE_GUIDANCE),
    ContinuationPathway.P_STOP_SUPPORTIVE_DEESCALATE: (InterventionAction.HARD_STOP,   FlagType.SELF_HARM_INSTRUCTION),
    ContinuationPathway.P_STOP_REDIRECT_QUALIFIED:    (InterventionAction.HARD_STOP,   FlagType.MEDICAL_DOSAGE_RECOMMENDATION),
    ContinuationPathway.P_STOP_REFUSE_CLEAN:          (InterventionAction.HARD_STOP,   FlagType.ILLEGAL_INSTRUCTION),
}


def _parse_key(
    key: str,
) -> tuple[Domain, AuthorityClass, LensStatus, UserClass, Optional[str]]:
    """Parse a _TABLE key to (domain, authority, status, user_class, reason_code).

    4-part user-class keys ("general:GP:STOP:auditor") yield the corresponding
    UserClass; reason_code=None.
    4-part flag-type keys ("medical:GP:STOP:SELF_HARM_INSTRUCTION") yield
    UserClass.GENERAL; reason_code=<flag_name>.
    3-part keys yield UserClass.GENERAL; reason_code=None.
    """
    parts = key.split(":")
    domain    = Domain(parts[0])
    authority = AuthorityClass(parts[1])
    status    = LensStatus(parts[2])
    user_class  = UserClass.GENERAL
    reason_code = None
    if len(parts) > 3:
        disc = parts[3]
        if disc in _USER_CLASS_BY_VALUE:
            user_class = _USER_CLASS_BY_VALUE[disc]
        else:
            reason_code = disc
    return domain, authority, status, user_class, reason_code


def _resolve(key: str):
    """Return (GovernorPolicy, RuntimeDecisionProjection) for a _TABLE key.

    Uses PolicyResolver to guarantee the same code path as production.
    The resolver and projector are instantiated fresh per call so tests
    cannot share mutable state.
    """
    domain, authority, status, user_class, reason_code = _parse_key(key)
    policy     = PolicyResolver().resolve(
        domain, authority, status, user_class=user_class, reason_code=reason_code
    )
    projection = PolicyProjector().project(policy)
    return policy, projection


def _flag(flag_type: FlagType, entity: str = "test") -> Flag:
    return Flag(
        flag_type=flag_type,
        entity_name=entity,
        claim="test claim",
        evidence="test evidence",
        severity="error",
    )


def _decision(row: ContinuationRow, *, resource: Optional[str] = "USE_ROW") -> GovernanceDecision:
    """Build a GovernanceDecision whose continuation fields mirror a matrix row.

    resource="USE_ROW" (the default sentinel) copies row.escalation_target;
    pass an explicit string (including None) to override.
    """
    action, flag_type = _PATHWAY_EXEC[row.pathway_id]
    actual_resource = row.escalation_target if resource == "USE_ROW" else resource
    d = GovernanceDecision(
        action=action,
        flags=[_flag(flag_type)],
        rationale="test",
        policy="strict",
        attempt=0,
        resource=actual_resource,
    )
    d.pathway_id           = row.pathway_id.value
    d.commitment_closed    = row.commitment_closed
    d.interaction_open     = row.interaction_open
    d.output_mode          = row.output_mode.value
    d.forensic_obligations = [fo.value for fo in row.forensic_obligations]
    return d


# ── Parametric fixtures ───────────────────────────────────────────────────────

_ALL_ROWS = list(_TABLE.items())

_NON_ADMIT_ROWS = [
    (k, r) for k, r in _ALL_ROWS
    if r.pathway_id != ContinuationPathway.P_ADMIT_STANDARD
]

# ASK rows that carry a non-None escalation_target (audit metadata).
_ASK_ROWS_TARGETED = [
    (k, r) for k, r in _ALL_ROWS
    if k.split(":")[2] == "ASK" and r.escalation_target is not None
]

# All ASK rows (for renderer PEF-source verification).
_ASK_ROWS_ALL = [
    (k, r) for k, r in _ALL_ROWS
    if k.split(":")[2] == "ASK"
]

# STOP/REFUSE rows where escalation_target is user-facing (should surface in output).
# P_STOP_REFUSE_CLEAN suppresses all resource strings by design; excluded here.
# P_HANDOFF_SUMMARY uses fixed text and does not surface a specific resource; excluded.
_STOP_REFUSE_USER_FACING = [
    (k, r) for k, r in _ALL_ROWS
    if k.split(":")[2] in ("STOP", "REFUSE")
    and r.escalation_target is not None
    and r.pathway_id not in (
        ContinuationPathway.P_STOP_REFUSE_CLEAN,
        ContinuationPathway.P_HANDOFF_SUMMARY,
    )
]

# Rows whose commitment_closed is True (projector must not relax these).
_COMMITMENT_CLOSED_ROWS = [
    (k, r) for k, r in _ALL_ROWS if r.commitment_closed
]

# Rows whose interaction_open is False (projector must not widen these).
_INTERACTION_CLOSED_ROWS = [
    (k, r) for k, r in _ALL_ROWS if not r.interaction_open
]

# Non-ADMIT rows that carry at least one forensic obligation.
_ROWS_WITH_FORENSIC = [
    (k, r) for k, r in _NON_ADMIT_ROWS if r.forensic_obligations
]

_STOP_ROWS = [
    (k, r) for k, r in _ALL_ROWS if k.split(":")[2] == "STOP"
]

_REFUSE_ROWS = [
    (k, r) for k, r in _ALL_ROWS if k.split(":")[2] == "REFUSE"
]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Resolver fidelity
# ═══════════════════════════════════════════════════════════════════════════════

class TestResolverFidelity:
    """PolicyResolver.resolve() maps every _TABLE key to a GovernorPolicy
    that carries all six continuation row fields without mutation.

    This verifies that the JSON policy matrix and CONTINUATION_MATRIX agree
    end-to-end through the full resolver code path, not just at table-lookup.
    """

    @pytest.mark.parametrize("key,row", _ALL_ROWS)
    def test_pathway_id_preserved(self, key: str, row: ContinuationRow):
        policy, _ = _resolve(key)
        assert policy.pathway_id == row.pathway_id, (
            f"{key}: resolver returned pathway_id={policy.pathway_id.value!r}, "
            f"expected {row.pathway_id.value!r}"
        )

    @pytest.mark.parametrize("key,row", _ALL_ROWS)
    def test_commitment_closed_preserved(self, key: str, row: ContinuationRow):
        policy, _ = _resolve(key)
        assert policy.commitment_closed == row.commitment_closed, (
            f"{key}: resolver returned commitment_closed={policy.commitment_closed}, "
            f"expected {row.commitment_closed}"
        )

    @pytest.mark.parametrize("key,row", _ALL_ROWS)
    def test_interaction_open_preserved(self, key: str, row: ContinuationRow):
        policy, _ = _resolve(key)
        assert policy.interaction_open == row.interaction_open, (
            f"{key}: resolver returned interaction_open={policy.interaction_open}, "
            f"expected {row.interaction_open}"
        )

    @pytest.mark.parametrize("key,row", _ALL_ROWS)
    def test_output_mode_preserved(self, key: str, row: ContinuationRow):
        policy, _ = _resolve(key)
        assert policy.output_mode == row.output_mode, (
            f"{key}: resolver returned output_mode={policy.output_mode.value!r}, "
            f"expected {row.output_mode.value!r}"
        )

    @pytest.mark.parametrize("key,row", _ALL_ROWS)
    def test_escalation_target_preserved(self, key: str, row: ContinuationRow):
        policy, _ = _resolve(key)
        assert policy.escalation_target == row.escalation_target, (
            f"{key}: resolver returned escalation_target={policy.escalation_target!r}, "
            f"expected {row.escalation_target!r}"
        )

    @pytest.mark.parametrize("key,row", _ALL_ROWS)
    def test_forensic_obligations_preserved(self, key: str, row: ContinuationRow):
        policy, _ = _resolve(key)
        assert set(policy.forensic_obligations) == set(row.forensic_obligations), (
            f"{key}: resolver returned forensic_obligations={policy.forensic_obligations}, "
            f"expected {list(row.forensic_obligations)}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Projector fidelity
# ═══════════════════════════════════════════════════════════════════════════════

class TestProjectorFidelity:
    """PolicyProjector.project() passes six row fields through unchanged.
    intervention_action is the only translation product.
    """

    @pytest.mark.parametrize("key,row", _ALL_ROWS)
    def test_invariant_fields_pass_through(self, key: str, row: ContinuationRow):
        """All six continuation fields survive the projector step unchanged."""
        _, proj = _resolve(key)
        assert proj.pathway_id == row.pathway_id, \
            f"{key}: pathway_id changed at projector"
        assert proj.commitment_closed == row.commitment_closed, \
            f"{key}: commitment_closed changed at projector"
        assert proj.interaction_open == row.interaction_open, \
            f"{key}: interaction_open changed at projector"
        assert proj.output_mode == row.output_mode, \
            f"{key}: output_mode changed at projector"
        assert proj.escalation_target == row.escalation_target, \
            f"{key}: escalation_target changed at projector"
        assert set(proj.forensic_obligations) == set(row.forensic_obligations), \
            f"{key}: forensic_obligations changed at projector"

    @pytest.mark.parametrize("key,row", _ALL_ROWS)
    def test_intervention_action_is_deterministic_from_pathway_id(
        self, key: str, row: ContinuationRow
    ):
        """intervention_action is solely a function of pathway_id.
        It carries no new policy information: many-to-one collapse is allowed,
        but the mapping must be the same stable table every time.
        """
        _, proj = _resolve(key)
        expected = _PATHWAY_TO_ACTION.get(row.pathway_id, _PATHWAY_FALLBACK)
        assert proj.intervention_action == expected, (
            f"{key}: intervention_action={proj.intervention_action.value!r} does not "
            f"match _PATHWAY_TO_ACTION[{row.pathway_id.value!r}]={expected.value!r}"
        )

    def test_intervention_action_is_the_only_translation_product(self):
        """The projector's contract is that exactly one field is a translation
        product (intervention_action) and all others pass through.  This test
        names the contract explicitly so it cannot be quietly violated by adding
        new fields that transform in the projector.
        """
        from dataclasses import fields as dc_fields
        from aurora_lens.govern.adapters.runtime_types import RuntimeDecisionProjection

        # Fields that are pass-throughs from GovernorPolicy (not translations).
        passthrough_fields = {
            "pathway_id", "commitment_closed", "interaction_open",
            "output_mode", "escalation_target", "forensic_obligations",
            "required_disclosures", "exposure_level", "resolution_mode",
            "projection_source",
        }
        # Fields that are translation products or projector-only additions.
        translation_fields = {"intervention_action", "provenance"}

        all_rdf = {f.name for f in dc_fields(RuntimeDecisionProjection)}
        assert passthrough_fields <= all_rdf, (
            "Pass-through field set is stale — update this test to reflect "
            "the current RuntimeDecisionProjection definition."
        )
        assert translation_fields <= all_rdf, (
            "Translation field set is stale."
        )
        uncategorised = all_rdf - passthrough_fields - translation_fields
        assert not uncategorised, (
            f"New fields in RuntimeDecisionProjection are not classified: "
            f"{uncategorised!r}. Add them to passthrough_fields or "
            f"translation_fields in this test and update the projector docstring."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Non-widening invariant at projector
# ═══════════════════════════════════════════════════════════════════════════════

class TestProjectorNonWidening:
    """The projector may never widen the governance corridor set by the matrix.

    'Widen' means:
      - relaxing commitment_closed from True to False
      - opening interaction_open from False to True
      - altering escalation_target to a different (possibly broader) target
    """

    @pytest.mark.parametrize("key,row", _COMMITMENT_CLOSED_ROWS)
    def test_commitment_closed_not_relaxed(self, key: str, row: ContinuationRow):
        _, proj = _resolve(key)
        assert proj.commitment_closed is True, (
            f"{key}: row.commitment_closed=True but projector returned False — "
            f"epistemic commitment was widened"
        )

    @pytest.mark.parametrize("key,row", _INTERACTION_CLOSED_ROWS)
    def test_interaction_open_not_widened(self, key: str, row: ContinuationRow):
        _, proj = _resolve(key)
        assert proj.interaction_open is False, (
            f"{key}: row.interaction_open=False but projector returned True — "
            f"interaction was illegitimately opened"
        )

    @pytest.mark.parametrize("key,row", _ALL_ROWS)
    def test_escalation_target_not_substituted(self, key: str, row: ContinuationRow):
        """The projector must not substitute a different escalation target.
        Even replacing None with a non-None string would be a new policy decision
        that belongs in the matrix, not the projector.
        """
        _, proj = _resolve(key)
        assert proj.escalation_target == row.escalation_target, (
            f"{key}: escalation_target was altered by projector — "
            f"row={row.escalation_target!r}, projection={proj.escalation_target!r}"
        )

    @pytest.mark.parametrize("key,row", _ALL_ROWS)
    def test_forensic_obligations_not_expanded(self, key: str, row: ContinuationRow):
        """The projector must not add forensic obligations that are not in the row.
        Adding an obligation at the projector stage would be an unauditable side effect.
        """
        _, proj = _resolve(key)
        row_set  = set(row.forensic_obligations)
        proj_set = set(proj.forensic_obligations)
        assert not (proj_set - row_set), (
            f"{key}: projector added forensic obligations not in row: "
            f"{proj_set - row_set!r}"
        )


class TestGovernorNonWideningByStatusClass:
    """Status-class corridor must not be widened by resolver/projector."""

    @pytest.mark.parametrize("key,row", _STOP_ROWS)
    def test_stop_rows_never_project_to_admit_or_refuse(self, key: str, row: ContinuationRow):
        _, proj = _resolve(key)
        assert proj.intervention_action == InterventionAction.HARD_STOP, (
            f"{key}: STOP row widened to {proj.intervention_action.value!r}"
        )

    @pytest.mark.parametrize("key,row", _REFUSE_ROWS)
    def test_refuse_rows_never_project_to_pass(self, key: str, row: ContinuationRow):
        _, proj = _resolve(key)
        assert proj.intervention_action == InterventionAction.FORCE_REVISE, (
            f"{key}: REFUSE row widened to {proj.intervention_action.value!r}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Audit-only field suppression
# ═══════════════════════════════════════════════════════════════════════════════

class TestAuditFieldSuppression:
    """Governance routing metadata never appears verbatim in renderer output.

    pathway_id.value, output_mode.value, and ForensicObligation.value are
    internal identifiers used by the pipeline for dispatch and audit.  They
    carry no semantics for end users and must not leak into the output string.
    """

    @pytest.mark.parametrize("key,row", _NON_ADMIT_ROWS)
    def test_pathway_id_not_in_output(self, key: str, row: ContinuationRow):
        result = enforce(_decision(row), "")
        assert row.pathway_id.value not in result, (
            f"{key}: pathway_id {row.pathway_id.value!r} leaked into renderer output: "
            f"{result!r}"
        )

    @pytest.mark.parametrize("key,row", _NON_ADMIT_ROWS)
    def test_output_mode_not_in_output(self, key: str, row: ContinuationRow):
        result = enforce(_decision(row), "")
        assert row.output_mode.value not in result, (
            f"{key}: output_mode {row.output_mode.value!r} leaked into renderer output: "
            f"{result!r}"
        )

    @pytest.mark.parametrize("key,row", _ROWS_WITH_FORENSIC)
    def test_forensic_obligation_values_not_in_output(
        self, key: str, row: ContinuationRow
    ):
        result = enforce(_decision(row), "")
        for fo in row.forensic_obligations:
            assert fo.value not in result, (
                f"{key}: ForensicObligation {fo.value!r} leaked into renderer output: "
                f"{result!r}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ASK field taxonomy — escalation_target is audit-only
# ═══════════════════════════════════════════════════════════════════════════════

class TestAskFieldTaxonomy:
    """ASK pathway escalation_target is an audit field, not a user-facing field.

    The field is non-None for domain-specific corridors (e.g. medical:GP:ASK
    carries "a licensed clinician") because the matrix needs to record which
    authority should supervise resolution.  But the ASK renderer uses
    PEF-derived candidates and missing_fields — it has no access to
    decision.resource and must not surface it.

    Contrast with STOP/REFUSE pathways (TestStopRefuseFieldTaxonomy) where
    the same escalation_target field IS user-facing.
    """

    @pytest.mark.parametrize("key,row", _ASK_ROWS_TARGETED)
    def test_escalation_target_present_in_row(self, key: str, row: ContinuationRow):
        """Precondition: escalation_target IS present in the matrix row
        as audit metadata.  This distinguishes 'intentionally absent' from
        'accidentally stripped'.
        """
        assert row.escalation_target is not None, (
            f"{key}: escalation_target should be non-None for domain-specific ASK rows"
        )

    @pytest.mark.parametrize("key,row", _ASK_ROWS_TARGETED)
    def test_escalation_target_not_in_renderer_output(self, key: str, row: ContinuationRow):
        """The escalation_target value is audit metadata and must not appear
        in the string returned by enforce().
        """
        result = enforce(_decision(row), "")
        assert row.escalation_target not in result, (
            f"{key}: ASK escalation_target {row.escalation_target!r} is audit-only "
            f"and must not appear in renderer output — got: {result!r}"
        )

    @pytest.mark.parametrize("key,row", _ASK_ROWS_ALL)
    def test_ask_renderer_ignores_resource_entirely(self, key: str, row: ContinuationRow):
        """ASK renderers derive their output from PEF-sourced candidates and
        missing_fields, not from decision.resource.  Injecting an arbitrary
        sentinel as the resource must not alter output.

        This confirms the information barrier between the ASK renderer path
        and the escalation_target field.
        """
        _SENTINEL = "AUDIT_ONLY_SENTINEL_MUST_NOT_APPEAR_IN_ASK_OUTPUT_XYZ"
        result_without = enforce(_decision(row, resource=None), "")
        result_with    = enforce(_decision(row, resource=_SENTINEL), "")
        assert _SENTINEL not in result_with, (
            f"{key}: ASK renderer surfaced decision.resource ({_SENTINEL!r}). "
            f"ASK renderers must use PEF-derived fields only."
        )
        assert result_without == result_with, (
            f"{key}: ASK renderer output changed when resource was injected. "
            f"Without: {result_without!r}\nWith: {result_with!r}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. STOP/REFUSE field taxonomy — escalation_target is user-facing
# ═══════════════════════════════════════════════════════════════════════════════

class TestStopRefuseFieldTaxonomy:
    """STOP/REFUSE pathway escalation_target is the user-facing redirect resource.

    When non-None, it MUST appear verbatim in renderer output so the user
    receives the correct professional or crisis contact.

    Exceptions (excluded from this class):
      - P_STOP_REFUSE_CLEAN: suppresses all resource strings by design.
      - P_HANDOFF_SUMMARY: uses fixed text, does not surface a specific resource.
    """

    @pytest.mark.parametrize("key,row", _STOP_REFUSE_USER_FACING)
    def test_escalation_target_surfaced_in_output(self, key: str, row: ContinuationRow):
        result = enforce(_decision(row), "")
        assert row.escalation_target in result, (
            f"{key}: expected escalation_target {row.escalation_target!r} in output — "
            f"got: {result!r}"
        )

    @pytest.mark.parametrize("key,row", _STOP_REFUSE_USER_FACING)
    def test_resource_is_the_only_external_string(self, key: str, row: ContinuationRow):
        """No external string other than escalation_target appears in output.
        Replace the escalation_target with a sentinel and verify the sentinel
        appears; inject a second sentinel as a different resource and verify
        the second one does NOT appear.
        """
        sentinel_a = "SENTINEL_A_PRIMARY_RESOURCE_XYZ_2025"
        sentinel_b = "SENTINEL_B_INJECTED_MUST_NOT_APPEAR_XYZ_2025"

        # Build decision with sentinel_a as the resource.
        d = _decision(row, resource=sentinel_a)
        result = enforce(d, sentinel_b)

        assert sentinel_a in result, (
            f"{key}: primary resource sentinel not found in output: {result!r}"
        )
        assert sentinel_b not in result, (
            f"{key}: model_output sentinel appeared in blocking pathway output: {result!r}"
        )

    def test_clean_stop_suppresses_resource(self):
        """P_STOP_REFUSE_CLEAN is the explicit exception to the taxonomy.
        Even when decision.resource is populated, the renderer passes None
        to _hard_stop_text so no resource string reaches the user.
        """
        row = CONTINUATION_MATRIX.lookup(
            Domain.LEGAL, AuthorityClass.GP, LensStatus.STOP, "ILLEGAL_INSTRUCTION"
        )
        assert row.pathway_id == ContinuationPathway.P_STOP_REFUSE_CLEAN
        sentinel = "SENTINEL_RESOURCE_CLEAN_STOP_MUST_SUPPRESS_XYZ"
        result = enforce(_decision(row, resource=sentinel), "")
        assert sentinel not in result, (
            f"P_STOP_REFUSE_CLEAN must suppress resource but sentinel appeared: {result!r}"
        )

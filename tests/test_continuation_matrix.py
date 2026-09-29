"""Contract tests for the canonical Governor continuation matrix.

Test surface
------------
1. Table invariant  — every non-ADMIT row has commitment_closed=True.
2. Row pathway      — each explicit table row produces the expected pathway_id.
3. No widening      — no row with status ∈ {ASK, REFUSE, STOP} may have
                      commitment_closed=False (epistemic non-widening invariant).
4. Fallback ladder  — lookup() returns the correct row for each level of the
                      explicit four-level fallback.
5. Safe fallback    — an unmapped (domain, authority, status) triple resolves
                      to 'general:GP:STOP' without raising.
6. Matrix integrity — construction with a violating row raises ValueError.
7. Policy resolver  — PolicyResolver._build_policy() uses the matrix for
                      continuation parameters, verified by spot-checking the
                      resolved GovernorPolicy against the matrix row.
"""

from __future__ import annotations

import pytest

from governor.continuation_matrix import (
    CONTINUATION_MATRIX,
    ContinuationMatrix,
    ContinuationRow,
    _TABLE,
    _CLOSED_STATUSES,
)
from governor.models import (
    AuthorityClass,
    ContinuationPathway,
    Domain,
    ForensicObligation,
    LensStatus,
    OutputMode,
    UserClass,
)
from governor.resolver import PolicyResolver


# ── Helpers ───────────────────────────────────────────────────────────────────


def _status_of(key: str) -> str:
    """Extract the status component from a table key."""
    return key.split(":")[2]


# ── 1. Table invariant ────────────────────────────────────────────────────────


class TestTableInvariant:
    """Every non-ADMIT row must have commitment_closed=True."""

    def test_no_widening_in_any_row(self):
        for key, row in _TABLE.items():
            status_str = _status_of(key)
            if status_str in _CLOSED_STATUSES:
                assert row.commitment_closed is True, (
                    f"Epistemic widening violation: key={key!r} has "
                    f"status={status_str!r} but commitment_closed=False"
                )

    def test_admit_rows_may_be_open(self):
        """ADMIT rows with commitment_closed=False are legal; verify at least one exists."""
        open_admit = [
            key for key, row in _TABLE.items()
            if _status_of(key) == "ADMIT" and not row.commitment_closed
        ]
        assert len(open_admit) > 0, (
            "Expected at least one ADMIT row with commitment_closed=False "
            "(e.g. general:GP:ADMIT)"
        )

    def test_safe_fallback_present(self):
        assert "general:GP:STOP" in _TABLE


# ── 2. Row pathway correctness ────────────────────────────────────────────────


@pytest.mark.parametrize("key,expected_pathway", [
    # general
    ("general:GP:ADMIT",   ContinuationPathway.P_ADMIT_STANDARD),
    ("general:GP:ASK",     ContinuationPathway.P_ASK_MISSING_FACT),
    ("general:GP:REFUSE",  ContinuationPathway.P_REFUSE_EXPLAIN_REDIRECT),
    ("general:GP:STOP",    ContinuationPathway.P_STOP_TERMINAL),
    ("general:GP:STOP:auditor", ContinuationPathway.P_STOP_TERMINAL),
    # ambiguity
    ("ambiguity:GP:ADMIT",  ContinuationPathway.P_ADMIT_STANDARD),
    ("ambiguity:GP:ASK",    ContinuationPathway.P_ASK_DISAMBIGUATE),
    ("ambiguity:GP:REFUSE", ContinuationPathway.P_REFUSE_EXPLAIN_REDIRECT),
    ("ambiguity:GP:STOP",   ContinuationPathway.P_STOP_TERMINAL),
    # medical:GP
    ("medical:GP:ADMIT",   ContinuationPathway.P_HANDOFF_SUMMARY),
    ("medical:GP:ASK",     ContinuationPathway.P_ASK_MISSING_FACT),
    ("medical:GP:REFUSE",  ContinuationPathway.P_REFUSE_EXPLAIN_REDIRECT),
    ("medical:GP:REFUSE:clinician", ContinuationPathway.P_REFUSE_ESCALATE_PRO),
    ("medical:GP:STOP",    ContinuationPathway.P_STOP_ESCALATE),
    ("medical:GP:STOP:EMERGENCY_TRIAGE_GUIDANCE",      ContinuationPathway.P_STOP_ESCALATE_EMERGENCY),
    ("medical:GP:STOP:SELF_HARM_INSTRUCTION",          ContinuationPathway.P_STOP_SUPPORTIVE_DEESCALATE),
    ("medical:GP:STOP:MEDICAL_DOSAGE_RECOMMENDATION",  ContinuationPathway.P_STOP_REDIRECT_QUALIFIED),
    ("medical:GP:STOP:PEDIATRIC_DOSAGE_RECOMMENDATION", ContinuationPathway.P_STOP_REDIRECT_QUALIFIED),
    ("medical:GP:STOP:NUMERIC_MEDICAL_INSTRUCTION",    ContinuationPathway.P_STOP_REDIRECT_QUALIFIED),
    ("medical:GP:STOP:PERSONALIZED_MEDICAL_ADVICE",    ContinuationPathway.P_STOP_REDIRECT_QUALIFIED),
    # medical:DA
    ("medical:DA:ADMIT",   ContinuationPathway.P_ADMIT_STANDARD),
    ("medical:DA:REFUSE",  ContinuationPathway.P_REFUSE_ESCALATE_PRO),
    ("medical:DA:STOP",    ContinuationPathway.P_STOP_TERMINAL),
    # medical:HS
    ("medical:HS:ADMIT",   ContinuationPathway.P_ADMIT_STANDARD),
    ("medical:HS:REFUSE",  ContinuationPathway.P_REFUSE_ESCALATE_PRO),
    ("medical:HS:STOP",    ContinuationPathway.P_STOP_FORENSIC),
    # legal:GP
    ("legal:GP:ADMIT",   ContinuationPathway.P_HANDOFF_SUMMARY),
    ("legal:GP:ASK",     ContinuationPathway.P_ASK_MISSING_FACT),
    ("legal:GP:REFUSE",  ContinuationPathway.P_REFUSE_ESCALATE_PRO),
    ("legal:GP:STOP",    ContinuationPathway.P_STOP_TERMINAL),
    ("legal:GP:STOP:PERSONALIZED_LEGAL_ADVICE",  ContinuationPathway.P_STOP_REDIRECT_QUALIFIED),
    ("legal:GP:STOP:ILLEGAL_INSTRUCTION",        ContinuationPathway.P_STOP_REFUSE_CLEAN),
    ("legal:GP:STOP:TARGETED_DEFAMATION",        ContinuationPathway.P_STOP_REFUSE_CLEAN),
    # legal:DA
    ("legal:DA:ADMIT",   ContinuationPathway.P_ADMIT_STANDARD),
    ("legal:DA:STOP",    ContinuationPathway.P_STOP_TERMINAL),
    # legal:HS
    ("legal:HS:ADMIT",   ContinuationPathway.P_ADMIT_STANDARD),
    ("legal:HS:STOP",    ContinuationPathway.P_STOP_FORENSIC),
    # finance:GP
    ("finance:GP:ADMIT",   ContinuationPathway.P_HANDOFF_SUMMARY),
    ("finance:GP:REFUSE",  ContinuationPathway.P_REFUSE_ESCALATE_PRO),
    ("finance:GP:STOP",    ContinuationPathway.P_STOP_TERMINAL),
    ("finance:GP:STOP:PERSONALIZED_FINANCIAL_ADVICE", ContinuationPathway.P_STOP_REDIRECT_QUALIFIED),
    # finance:DA / finance:HS
    ("finance:DA:STOP",    ContinuationPathway.P_STOP_TERMINAL),
    ("finance:HS:STOP",    ContinuationPathway.P_STOP_FORENSIC),
    # research:GP
    ("research:GP:ADMIT",  ContinuationPathway.P_ADMIT_STANDARD),
    ("research:GP:ASK",    ContinuationPathway.P_ASK_MISSING_FACT),
    ("research:GP:REFUSE", ContinuationPathway.P_REFUSE_EXPLAIN_REDIRECT),
    ("research:GP:STOP",   ContinuationPathway.P_STOP_TERMINAL),
    # crisis:GP
    ("crisis:GP:ADMIT",   ContinuationPathway.P_ADMIT_STANDARD),
    ("crisis:GP:ASK",     ContinuationPathway.P_ASK_DISAMBIGUATE),
    ("crisis:GP:REFUSE",  ContinuationPathway.P_REFUSE_EXPLAIN_REDIRECT),
    ("crisis:GP:STOP",    ContinuationPathway.P_STOP_TERMINAL),
])
def test_row_pathway(key: str, expected_pathway: ContinuationPathway):
    """Each explicit table entry produces the expected pathway_id."""
    assert _TABLE[key].pathway_id == expected_pathway, (
        f"key={key!r}: expected pathway {expected_pathway.value!r}, "
        f"got {_TABLE[key].pathway_id.value!r}"
    )


# ── 3. Epistemic non-widening — parametric ────────────────────────────────────


@pytest.mark.parametrize("key,row", list(_TABLE.items()))
def test_no_widening_parametric(key: str, row: ContinuationRow):
    """Parametric version of the invariant — one test per table row."""
    status_str = _status_of(key)
    if status_str in _CLOSED_STATUSES:
        assert row.commitment_closed is True, (
            f"key={key!r}: status={status_str!r} requires commitment_closed=True"
        )


# ── 4. Lookup fallback ladder ─────────────────────────────────────────────────


class TestFallbackLadder:

    def test_level1_discriminator_hit(self):
        """A 4-component key resolves to the specific row, not the 3-component row."""
        row = CONTINUATION_MATRIX.lookup(
            Domain.MEDICAL, AuthorityClass.GP, LensStatus.STOP,
            discriminator="SELF_HARM_INSTRUCTION",
        )
        assert row.pathway_id == ContinuationPathway.P_STOP_SUPPORTIVE_DEESCALATE

    def test_level1_user_class_discriminator(self):
        """auditor discriminator returns the auditor-specific forensic_stop row."""
        row = CONTINUATION_MATRIX.lookup(
            Domain.GENERAL, AuthorityClass.GP, LensStatus.STOP,
            discriminator="auditor",
        )
        assert row.output_mode == OutputMode.FORENSIC_STOP
        assert ForensicObligation.ATTACH_PEF_SNAPSHOT in row.forensic_obligations

    def test_level2_authority_key_used_when_no_discriminator(self):
        """Without discriminator, the 3-component authority key is used."""
        row = CONTINUATION_MATRIX.lookup(
            Domain.MEDICAL, AuthorityClass.GP, LensStatus.STOP,
        )
        assert row.pathway_id == ContinuationPathway.P_STOP_ESCALATE

    def test_level3_gp_fallback_for_unknown_authority(self):
        """An authority not in the table falls back to the GP row for that domain."""
        # RESEARCH has only GP entries; HS should fall back to GP
        row = CONTINUATION_MATRIX.lookup(
            Domain.RESEARCH, AuthorityClass.HS, LensStatus.STOP,
        )
        # Should get research:GP:STOP
        assert row.pathway_id == ContinuationPathway.P_STOP_TERMINAL
        assert row.commitment_closed is True

    def test_level4_global_status_fallback(self):
        """An unmapped domain falls back to general:GP:<status>."""
        # AMBIGUITY:DA is not in the table; should fall back via GP then general
        row = CONTINUATION_MATRIX.lookup(
            Domain.AMBIGUITY, AuthorityClass.DA, LensStatus.REFUSE,
        )
        # ambiguity:DA:REFUSE not in table → ambiguity:GP:REFUSE is in table
        assert row.pathway_id == ContinuationPathway.P_REFUSE_EXPLAIN_REDIRECT

    def test_level5_safe_fallback_for_completely_unknown(self):
        """A completely unmapped combo resolves to the safe fallback without raising."""
        # No 'general:GP:ADMIT' test here since that exists; use a domain that
        # has no entries at all for GP+REFUSE — but all domains have GP entries,
        # so we test that a missing discriminator falls through correctly.
        row = CONTINUATION_MATRIX.lookup(
            Domain.GENERAL, AuthorityClass.GP, LensStatus.STOP,
            discriminator="NONEXISTENT_FLAG_TYPE_XYZ",
        )
        # Discriminator not found → falls back to general:GP:STOP
        assert row.pathway_id == ContinuationPathway.P_STOP_TERMINAL
        assert row.commitment_closed is True

    def test_discriminator_miss_falls_through_to_base_key(self):
        """A discriminator that does not match any 4-component key falls to base."""
        row = CONTINUATION_MATRIX.lookup(
            Domain.MEDICAL, AuthorityClass.GP, LensStatus.REFUSE,
            discriminator="UNKNOWN_FLAG_TYPE",
        )
        # medical:GP:REFUSE:UNKNOWN_FLAG_TYPE not in table
        # → falls to medical:GP:REFUSE
        assert row.pathway_id == ContinuationPathway.P_REFUSE_EXPLAIN_REDIRECT


# ── 5. Safe fallback always present ──────────────────────────────────────────


def test_safe_fallback_is_closed():
    """The global safe fallback must never widen epistemic commitment."""
    fallback = CONTINUATION_MATRIX.lookup(
        Domain.GENERAL, AuthorityClass.GP, LensStatus.STOP,
    )
    assert fallback.commitment_closed is True
    assert fallback.pathway_id == ContinuationPathway.P_STOP_TERMINAL


# ── 6. Construction with a violating row raises ValueError ────────────────────


class TestConstructionValidation:

    def test_stop_row_commitment_open_raises(self):
        bad_table = dict(_TABLE)
        bad_table["test:GP:STOP"] = ContinuationRow(
            pathway_id=ContinuationPathway.P_STOP_TERMINAL,
            commitment_closed=False,  # VIOLATION
            interaction_open=False,
            output_mode=OutputMode.TERMINAL_STOP,
            forensic_obligations=(),
        )
        with pytest.raises(ValueError, match="epistemic widening"):
            ContinuationMatrix(bad_table)

    def test_refuse_row_commitment_open_raises(self):
        bad_table = dict(_TABLE)
        bad_table["test:GP:REFUSE"] = ContinuationRow(
            pathway_id=ContinuationPathway.P_REFUSE_EXPLAIN_REDIRECT,
            commitment_closed=False,  # VIOLATION
            interaction_open=True,
            output_mode=OutputMode.REFUSAL_WITH_EXPLANATION,
            forensic_obligations=(),
        )
        with pytest.raises(ValueError, match="epistemic widening"):
            ContinuationMatrix(bad_table)

    def test_ask_row_commitment_open_raises(self):
        bad_table = dict(_TABLE)
        bad_table["test:GP:ASK"] = ContinuationRow(
            pathway_id=ContinuationPathway.P_ASK_MISSING_FACT,
            commitment_closed=False,  # VIOLATION
            interaction_open=True,
            output_mode=OutputMode.CLARIFICATION_REQUEST,
            forensic_obligations=(),
        )
        with pytest.raises(ValueError, match="epistemic widening"):
            ContinuationMatrix(bad_table)

    def test_missing_safe_fallback_raises(self):
        bad_table = {k: v for k, v in _TABLE.items() if k != "general:GP:STOP"}
        with pytest.raises(ValueError, match="safe fallback"):
            ContinuationMatrix(bad_table)

    def test_valid_table_constructs_cleanly(self):
        """A table that passes all invariants should construct without error."""
        m = ContinuationMatrix(_TABLE)
        assert m is not None

    def test_admit_row_with_open_commitment_is_valid(self):
        """ADMIT rows may have commitment_closed=False — this must NOT raise."""
        valid_table = dict(_TABLE)
        valid_table["test:GP:ADMIT"] = ContinuationRow(
            pathway_id=ContinuationPathway.P_ADMIT_STANDARD,
            commitment_closed=False,
            interaction_open=True,
            output_mode=OutputMode.FULL_RESPONSE,
            forensic_obligations=(),
        )
        m = ContinuationMatrix(valid_table)
        row = m.lookup(Domain.GENERAL, AuthorityClass.GP, LensStatus.ADMIT)
        assert row.commitment_closed is False  # general:GP:ADMIT is open


# ── 7. PolicyResolver integration ────────────────────────────────────────────


class TestPolicyResolverIntegration:
    """Verify that PolicyResolver._build_policy() uses the matrix for continuation fields."""

    @pytest.fixture(scope="class")
    def resolver(self):
        return PolicyResolver()

    def _row(self, key: str) -> ContinuationRow:
        return _TABLE[key]

    def test_general_gp_stop_uses_matrix(self, resolver: PolicyResolver):
        policy = resolver.resolve(Domain.GENERAL, AuthorityClass.GP, LensStatus.STOP)
        row = self._row("general:GP:STOP")
        assert policy.pathway_id == row.pathway_id
        assert policy.commitment_closed == row.commitment_closed
        assert policy.interaction_open == row.interaction_open
        assert policy.output_mode == row.output_mode

    def test_medical_gp_stop_self_harm_uses_matrix(self, resolver: PolicyResolver):
        policy = resolver.resolve(
            Domain.MEDICAL, AuthorityClass.GP, LensStatus.STOP,
            reason_code="SELF_HARM_INSTRUCTION",
        )
        row = self._row("medical:GP:STOP:SELF_HARM_INSTRUCTION")
        assert policy.pathway_id == row.pathway_id
        assert policy.commitment_closed is True
        assert policy.interaction_open == row.interaction_open
        assert policy.escalation_target == row.escalation_target

    def test_medical_gp_stop_emergency_uses_matrix(self, resolver: PolicyResolver):
        policy = resolver.resolve(
            Domain.MEDICAL, AuthorityClass.GP, LensStatus.STOP,
            reason_code="EMERGENCY_TRIAGE_GUIDANCE",
        )
        row = self._row("medical:GP:STOP:EMERGENCY_TRIAGE_GUIDANCE")
        assert policy.pathway_id == ContinuationPathway.P_STOP_ESCALATE_EMERGENCY
        assert policy.commitment_closed is True
        assert policy.interaction_open is False

    def test_legal_gp_stop_illegal_instruction_uses_matrix(self, resolver: PolicyResolver):
        policy = resolver.resolve(
            Domain.LEGAL, AuthorityClass.GP, LensStatus.STOP,
            reason_code="ILLEGAL_INSTRUCTION",
        )
        row = self._row("legal:GP:STOP:ILLEGAL_INSTRUCTION")
        assert policy.pathway_id == ContinuationPathway.P_STOP_REFUSE_CLEAN
        assert policy.commitment_closed is True
        assert policy.escalation_target is None

    def test_ambiguity_gp_ask_uses_matrix(self, resolver: PolicyResolver):
        policy = resolver.resolve(Domain.AMBIGUITY, AuthorityClass.GP, LensStatus.ASK)
        row = self._row("ambiguity:GP:ASK")
        assert policy.pathway_id == ContinuationPathway.P_ASK_DISAMBIGUATE
        assert policy.commitment_closed is True

    def test_medical_hs_stop_uses_matrix(self, resolver: PolicyResolver):
        policy = resolver.resolve(Domain.MEDICAL, AuthorityClass.HS, LensStatus.STOP)
        row = self._row("medical:HS:STOP")
        assert policy.pathway_id == ContinuationPathway.P_STOP_FORENSIC
        assert policy.commitment_closed is True
        assert policy.interaction_open is False
        assert policy.output_mode == OutputMode.FORENSIC_STOP

    def test_general_gp_admit_open(self, resolver: PolicyResolver):
        policy = resolver.resolve(Domain.GENERAL, AuthorityClass.GP, LensStatus.ADMIT)
        assert policy.commitment_closed is False
        assert policy.pathway_id == ContinuationPathway.P_ADMIT_STANDARD

    def test_finance_gp_stop_personalized_uses_matrix(self, resolver: PolicyResolver):
        policy = resolver.resolve(
            Domain.FINANCE, AuthorityClass.GP, LensStatus.STOP,
            reason_code="PERSONALIZED_FINANCIAL_ADVICE",
        )
        assert policy.pathway_id == ContinuationPathway.P_STOP_REDIRECT_QUALIFIED
        assert policy.commitment_closed is True

    def test_resolver_non_admit_always_commitment_closed(self, resolver: PolicyResolver):
        """No PolicyResolver call with non-ADMIT status may return commitment_closed=False."""
        non_admit_statuses = [LensStatus.ASK, LensStatus.REFUSE, LensStatus.STOP]
        domains = [Domain.GENERAL, Domain.MEDICAL, Domain.LEGAL, Domain.FINANCE]
        authorities = [AuthorityClass.GP, AuthorityClass.DA, AuthorityClass.HS]

        for domain in domains:
            for authority in authorities:
                for status in non_admit_statuses:
                    policy = resolver.resolve(domain, authority, status)
                    assert policy.commitment_closed is True, (
                        f"commitment_closed=False for "
                        f"{domain.value}:{authority.value}:{status.value}"
                    )

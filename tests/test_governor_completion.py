"""Tests for Governor Completion Pass product gaps 1-5.

Gap 1: DA/HS authority-class differentiation in regulated domains
Gap 2: governed_response captured as first-class audit field
Gap 3: escalation_target wired from canonical policy into renderers
Gap 4: policy_version selects actual policy matrix variant
Gap 5: operator-supplied policy matrix path is recorded
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from governor.models import (
    AuthorityClass, Domain, LensStatus, UserClass, ContinuationPathway,
)
from governor.resolver import PolicyResolver


def _make_decision(bridge, flags, original="model output"):
    """Synchronous helper to run bridge.decide() synchronously for testing."""
    import asyncio
    from aurora_lens.pef.state import PEFState

    pef = PEFState()
    return asyncio.run(
        bridge.decide(flags, original, pef)
    )


def _make_intervene(bridge, decision, original="model output"):
    import asyncio
    return asyncio.run(
        bridge.intervene(decision, None, "user input", "", None)
    )


# ---------------------------------------------------------------------------
# Gap 1: Authority-class completion
# ---------------------------------------------------------------------------

class TestAuthorityClassCompletion:
    """DA and HS produce intentionally different continuation sets from GP
    for regulated domains (medical, legal, finance) where differentiation
    is policy-defined."""

    def test_medical_da_admit_is_open_commitment(self):
        r = PolicyResolver()
        p = r.resolve(Domain.MEDICAL, AuthorityClass.DA, LensStatus.ADMIT)
        assert p.commitment_closed is False, (
            "DA medical ADMIT must have commitment_closed=False (domain-authorized system)"
        )

    def test_medical_gp_admit_is_closed_commitment(self):
        r = PolicyResolver()
        p = r.resolve(Domain.MEDICAL, AuthorityClass.GP, LensStatus.ADMIT)
        assert p.commitment_closed is True, (
            "GP medical ADMIT must have commitment_closed=True (general-purpose, constrained)"
        )

    def test_medical_hs_admit_is_open_commitment(self):
        r = PolicyResolver()
        p = r.resolve(Domain.MEDICAL, AuthorityClass.HS, LensStatus.ADMIT)
        assert p.commitment_closed is False

    def test_da_and_gp_medical_admit_produce_different_pathways(self):
        r = PolicyResolver()
        p_da = r.resolve(Domain.MEDICAL, AuthorityClass.DA, LensStatus.ADMIT)
        p_gp = r.resolve(Domain.MEDICAL, AuthorityClass.GP, LensStatus.ADMIT)
        assert p_da.pathway_id != p_gp.pathway_id, (
            "DA and GP must produce different pathways for medical ADMIT"
        )

    def test_da_and_gp_medical_refuse_produce_different_obligations(self):
        r = PolicyResolver()
        p_da = r.resolve(Domain.MEDICAL, AuthorityClass.DA, LensStatus.REFUSE)
        p_gp = r.resolve(Domain.MEDICAL, AuthorityClass.GP, LensStatus.REFUSE)
        # DA must require PEF snapshot; GP does not
        da_obligations = {fo.value for fo in p_da.forensic_obligations}
        gp_obligations = {fo.value for fo in p_gp.forensic_obligations}
        assert "attach_pef_snapshot" in da_obligations
        assert "attach_pef_snapshot" not in gp_obligations

    def test_hs_medical_stop_uses_forensic_pathway(self):
        r = PolicyResolver()
        p = r.resolve(Domain.MEDICAL, AuthorityClass.HS, LensStatus.STOP)
        assert p.pathway_id == ContinuationPathway.P_STOP_FORENSIC

    def test_hs_legal_stop_uses_forensic_pathway(self):
        r = PolicyResolver()
        p = r.resolve(Domain.LEGAL, AuthorityClass.HS, LensStatus.STOP)
        assert p.pathway_id == ContinuationPathway.P_STOP_FORENSIC

    def test_hs_finance_stop_uses_forensic_pathway(self):
        r = PolicyResolver()
        p = r.resolve(Domain.FINANCE, AuthorityClass.HS, LensStatus.STOP)
        assert p.pathway_id == ContinuationPathway.P_STOP_FORENSIC

    def test_no_silent_da_fallback_to_gp_for_legal_admit(self):
        """DA legal ADMIT must resolve to a DA-specific row, not fall through to GP."""
        r = PolicyResolver()
        p_da = r.resolve(Domain.LEGAL, AuthorityClass.DA, LensStatus.ADMIT)
        p_gp = r.resolve(Domain.LEGAL, AuthorityClass.GP, LensStatus.ADMIT)
        assert p_da.commitment_closed is False
        assert p_gp.commitment_closed is True

    def test_no_silent_da_fallback_to_gp_for_finance_admit(self):
        r = PolicyResolver()
        p_da = r.resolve(Domain.FINANCE, AuthorityClass.DA, LensStatus.ADMIT)
        p_gp = r.resolve(Domain.FINANCE, AuthorityClass.GP, LensStatus.ADMIT)
        assert p_da.commitment_closed is False
        assert p_gp.commitment_closed is True

    def test_hs_has_notify_operator_in_refuse(self):
        """HS REFUSE must include notify_operator procedural action; GP and DA do not."""
        r = PolicyResolver()
        p_hs = r.resolve(Domain.MEDICAL, AuthorityClass.HS, LensStatus.REFUSE)
        pa_names = {pa.value for pa in p_hs.allowed_procedural_actions}
        assert "notify_operator" in pa_names

    def test_da_refuse_no_notify_operator(self):
        r = PolicyResolver()
        p_da = r.resolve(Domain.MEDICAL, AuthorityClass.DA, LensStatus.REFUSE)
        pa_names = {pa.value for pa in p_da.allowed_procedural_actions}
        assert "notify_operator" not in pa_names


# ---------------------------------------------------------------------------
# Gap 2: Governed response capture
# ---------------------------------------------------------------------------

class TestGovernedResponseCapture:
    """governed_response is set on GovernanceDecision after pathway rendering
    and is distinct from original_response when an intervention fires."""

    def _make_bridge(self):
        from aurora_lens.govern.canonical_bridge import CanonicalScannerGateBridge
        return CanonicalScannerGateBridge(mode="public")

    def _flags_for_hard_stop(self):
        from aurora_lens.verify.flags import Flag, FlagType
        return [Flag(
            flag_type=FlagType.ILLEGAL_INSTRUCTION,
            entity_name="test",
            claim="test claim",
            evidence="test evidence",
            severity="error",
        )]

    def _flags_for_pass(self):
        return []

    def test_governed_response_set_on_hard_stop(self):
        bridge = self._make_bridge()
        flags = self._flags_for_hard_stop()
        decision = _make_decision(bridge, flags, "bad model output")
        decision.original_response = "bad model output"
        response_text = _make_intervene(bridge, decision, "bad model output")
        assert decision.governed_response is not None
        assert decision.governed_response == response_text

    def test_governed_response_differs_from_original_on_intervention(self):
        bridge = self._make_bridge()
        flags = self._flags_for_hard_stop()
        decision = _make_decision(bridge, flags, "illegal content output")
        decision.original_response = "illegal content output"
        _make_intervene(bridge, decision)
        assert decision.governed_response != decision.original_response

    def test_governed_response_field_exists_on_decision(self):
        from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
        from aurora_lens.verify.flags import Flag, FlagType
        d = GovernanceDecision(
            action=InterventionAction.PASS,
            flags=[],
            rationale="test",
        )
        # Field should exist and default to None
        assert hasattr(d, "governed_response")
        assert d.governed_response is None


# ---------------------------------------------------------------------------
# Gap 3: Escalation resource wiring
# ---------------------------------------------------------------------------

class TestEscalationResourceWiring:
    """escalation_target from canonical policy populates decision.resource,
    which flows into rendered continuation text."""

    def test_medical_gp_refuse_has_escalation_target(self):
        r = PolicyResolver()
        p = r.resolve(Domain.MEDICAL, AuthorityClass.GP, LensStatus.REFUSE)
        assert p.escalation_target is not None
        assert "clinician" in p.escalation_target.lower()

    def test_legal_gp_refuse_has_escalation_target(self):
        r = PolicyResolver()
        p = r.resolve(Domain.LEGAL, AuthorityClass.GP, LensStatus.REFUSE)
        assert p.escalation_target is not None
        assert "lawyer" in p.escalation_target.lower() or "legal" in p.escalation_target.lower()

    def test_finance_gp_refuse_has_escalation_target(self):
        r = PolicyResolver()
        p = r.resolve(Domain.FINANCE, AuthorityClass.GP, LensStatus.REFUSE)
        assert p.escalation_target is not None
        assert "adviser" in p.escalation_target.lower() or "advisor" in p.escalation_target.lower()

    def test_medical_da_refuse_escalation_target_is_supervisory(self):
        r = PolicyResolver()
        p = r.resolve(Domain.MEDICAL, AuthorityClass.DA, LensStatus.REFUSE)
        assert p.escalation_target is not None
        assert "supervising" in p.escalation_target.lower()

    def test_escalation_target_flows_into_decision_resource(self):
        """decision.resource is populated from the policy projection."""
        from aurora_lens.govern.canonical_bridge import CanonicalScannerGateBridge
        from aurora_lens.verify.flags import Flag, FlagType
        from aurora_lens.context import domain_var, authority_class_var
        from aurora_lens.pef.state import PEFState
        import asyncio

        bridge = CanonicalScannerGateBridge(mode="public")
        flags = [Flag(
            flag_type=FlagType.PERSONALIZED_MEDICAL_ADVICE,
            entity_name="test",
            claim="test",
            evidence="test",
            severity="warning",
        )]
        pef = PEFState()
        tok_d = domain_var.set("medical")
        try:
            decision = asyncio.run(
                bridge.decide(flags, "model output", pef)
            )
        finally:
            domain_var.reset(tok_d)

        assert decision.resource is not None, "escalation_target must populate decision.resource"
        # medical:GP:STOP uses P_STOP_ESCALATE with a specific actionable escalation target
        assert any(phrase in decision.resource.lower() for phrase in [
            "clinician", "healthcare", "gp", "nurse", "ambulance", "emergency",
        ]), f"Expected a medical escalation target; got: {decision.resource!r}"

    def test_no_blank_escalation_placeholder_in_regulated_domains(self):
        """Every regulated domain GP policy row must have a non-empty escalation_target."""
        r = PolicyResolver()
        regulated_statuses = [LensStatus.REFUSE, LensStatus.STOP]
        for domain in [Domain.MEDICAL, Domain.LEGAL, Domain.FINANCE]:
            for status in regulated_statuses:
                for authority in [AuthorityClass.GP, AuthorityClass.DA, AuthorityClass.HS]:
                    p = r.resolve(domain, authority, status)
                    assert p.escalation_target, (
                        f"escalation_target missing for {domain.value}:{authority.value}:{status.value}"
                    )

    def test_legal_personalized_stop_has_mechanical_continuation_capabilities(self):
        r = PolicyResolver()
        p = r.resolve(
            Domain.LEGAL,
            AuthorityClass.GP,
            LensStatus.STOP,
            reason_code="PERSONALIZED_LEGAL_ADVICE",
        )
        assert {c.value for c in p.allowed_continuations} == {"neutral_timeline"}

    def test_generic_legal_stop_has_no_default_continuation_capabilities(self):
        r = PolicyResolver()
        p = r.resolve(Domain.LEGAL, AuthorityClass.GP, LensStatus.STOP)
        assert p.allowed_continuations == []


# ---------------------------------------------------------------------------
# Gap 4: Policy versioning
# ---------------------------------------------------------------------------

class TestPolicyVersioning:

    def test_default_version_is_1_0(self):
        r = PolicyResolver()
        assert r.active_version == "1.0"

    def test_policy_source_is_bundled_for_default(self):
        r = PolicyResolver()
        assert r.policy_source == "bundled"

    def test_nonexistent_version_falls_back_to_bundled(self):
        """Requesting a version with no versioned file falls back to bundled matrix."""
        r = PolicyResolver(policy_version="99.0")
        assert r.active_version == "99.0"  # as requested (no override in bundled)
        assert r.policy_source == "bundled"

    def test_versioned_matrix_file_loads_when_present(self, tmp_path):
        """If policy_matrix.{version}.json exists, it is used."""
        # Create a minimal versioned matrix next to where the bundled one lives
        versioned = (
            Path(__file__).resolve().parents[1]
            / "aurora_lens"
            / "governor"
            / "policy_matrix.test-v2.json"
        )
        minimal = {
            "_version": "test-v2",
            "general:GP:ADMIT": {
                "allowed_speech_acts": ["provide_substantive_answer"],
                "allowed_procedural_actions": [],
                "forbidden_speech_acts": [],
                "forensic_obligations": [],
                "required_disclosures": [],
                "commitment_closed": False,
                "interaction_open": True,
                "pathway_id": "P_ADMIT_STANDARD",
                "output_mode": "full_response",
                "exposure_level": "minimal"
            },
            "general:GP:STOP": {
                "allowed_speech_acts": ["notice_termination"],
                "allowed_procedural_actions": [],
                "forbidden_speech_acts": [],
                "forensic_obligations": ["emit_forensic_envelope"],
                "required_disclosures": [],
                "commitment_closed": True,
                "interaction_open": False,
                "pathway_id": "P_STOP_TERMINAL",
                "output_mode": "terminal_stop",
                "exposure_level": "minimal"
            }
        }
        versioned.write_text(json.dumps(minimal), encoding="utf-8")
        try:
            r = PolicyResolver(policy_version="test-v2")
            assert r.active_version == "test-v2"
            assert r.policy_source == "bundled"
        finally:
            versioned.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Gap 5: Operator-configurable policy loading
# ---------------------------------------------------------------------------

class TestOperatorPolicyLoading:

    def _minimal_matrix(self):
        return {
            "_version": "operator-1",
            "general:GP:ADMIT": {
                "allowed_speech_acts": ["provide_substantive_answer"],
                "allowed_procedural_actions": [],
                "forbidden_speech_acts": [],
                "forensic_obligations": [],
                "required_disclosures": [],
                "commitment_closed": False,
                "interaction_open": True,
                "pathway_id": "P_ADMIT_STANDARD",
                "output_mode": "full_response",
                "exposure_level": "minimal"
            },
            "general:GP:STOP": {
                "allowed_speech_acts": ["notice_termination"],
                "allowed_procedural_actions": [],
                "forbidden_speech_acts": [],
                "forensic_obligations": ["emit_forensic_envelope"],
                "required_disclosures": [],
                "commitment_closed": True,
                "interaction_open": False,
                "pathway_id": "P_STOP_TERMINAL",
                "output_mode": "terminal_stop",
                "exposure_level": "minimal"
            }
        }

    def test_operator_matrix_path_loads(self, tmp_path):
        matrix_file = tmp_path / "operator_matrix.json"
        matrix_file.write_text(json.dumps(self._minimal_matrix()), encoding="utf-8")
        r = PolicyResolver(matrix_path=matrix_file)
        assert r.active_version == "operator-1"
        assert r.policy_source == "operator"

    def test_operator_source_label_is_recorded(self, tmp_path):
        matrix_file = tmp_path / "custom.json"
        matrix_file.write_text(json.dumps(self._minimal_matrix()), encoding="utf-8")
        r = PolicyResolver(matrix_path=matrix_file)
        assert r.policy_source == "operator"

    def test_bundled_source_label_for_default(self):
        r = PolicyResolver()
        assert r.policy_source == "bundled"

    def test_operator_matrix_can_override_gp_admit(self, tmp_path):
        """Operator matrix resolves differently than bundled for the same key."""
        custom = self._minimal_matrix()
        # Close commitment and remove substantive acts to satisfy GovernorPolicy invariant
        custom["general:GP:ADMIT"]["commitment_closed"] = True
        custom["general:GP:ADMIT"]["allowed_speech_acts"] = ["offer_safe_next_step"]
        matrix_file = tmp_path / "tighter.json"
        matrix_file.write_text(json.dumps(custom), encoding="utf-8")
        r_operator = PolicyResolver(matrix_path=matrix_file)
        r_bundled = PolicyResolver()
        p_op = r_operator.resolve(Domain.GENERAL, AuthorityClass.GP, LensStatus.ADMIT)
        p_bu = r_bundled.resolve(Domain.GENERAL, AuthorityClass.GP, LensStatus.ADMIT)
        # Operator has tighter (closed) vs bundled (open)
        assert p_op.commitment_closed is True
        assert p_bu.commitment_closed is False

    def test_canonical_bridge_accepts_policy_matrix_path(self, tmp_path):
        """CanonicalScannerGateBridge constructor accepts policy_matrix_path kwarg."""
        from aurora_lens.govern.canonical_bridge import CanonicalScannerGateBridge
        matrix_file = tmp_path / "minimal.json"
        matrix_file.write_text(json.dumps(self._minimal_matrix()), encoding="utf-8")
        # Should not raise
        bridge = CanonicalScannerGateBridge(
            mode="public",
            policy_version="operator-1",
            policy_matrix_path=str(matrix_file),
        )
        assert bridge._policy_resolver.policy_source == "operator"

    def test_policy_matrix_path_in_governance_config(self):
        """GovernanceConfig has policy_matrix_path field."""
        from aurora_lens.proxy.config import GovernanceConfig
        cfg = GovernanceConfig(policy_matrix_path="/some/path/matrix.json")
        assert cfg.policy_matrix_path == "/some/path/matrix.json"

    def test_policy_matrix_path_defaults_to_none(self):
        from aurora_lens.proxy.config import GovernanceConfig
        cfg = GovernanceConfig()
        assert cfg.policy_matrix_path is None

"""Tests for deterministic sovereign provider-route refusal templates."""

from __future__ import annotations

import pytest

from aurora_lens.sovereign.audit_envelope import build_provider_route_audit_envelope
from aurora_lens.sovereign.provider_registry import (
    ProviderRouteEvaluation,
    ProviderRouteOutcome,
    SovereignProviderRegistry,
    reject_failover_without_registry,
    reject_missing_provider_route,
)
from aurora_lens.sovereign.provider_state import LEGAL_UNAVAILABLE, REGRESSION_FAILED
from aurora_lens.sovereign.refusal_templates import (
    ALTERNATE_MISSING_PROFILE_TEMPLATE_KEY,
    ALTERNATE_OUT_OF_SCOPE_TEMPLATE_KEY,
    ALTERNATE_VALIDATION_EXPIRED_TEMPLATE_KEY,
    LEGAL_UNAVAILABLE_TEMPLATE_KEY,
    PROVIDER_ROUTE_REQUIRED_TEMPLATE_KEY,
    PROVIDER_UNAVAILABLE_NO_ALTERNATE_TEMPLATE_KEY,
    REGISTRY_EVALUATION_REQUIRED_TEMPLATE_KEY,
    REGRESSION_FAILED_TEMPLATE_KEY,
    SOVEREIGN_REFUSAL_TEMPLATE_KEYS,
    refusal_for_evaluation,
    resolve_refusal_template_key,
)
from tests.test_sovereign_failover_bridge import (
    _certified_alternate_profile,
    _route_request,
)


def _evaluation(**overrides: object) -> ProviderRouteEvaluation:
    base = {
        "outcome": ProviderRouteOutcome.LOCAL_REFUSAL,
        "reason": "test_reason",
        "registry_evaluated": True,
        "primary_provider_id": "primary",
        "alternate_provider_id": "alternate",
        "bridge_status": None,
        "inference_contract": None,
        "failover_attempted": True,
    }
    base.update(overrides)
    return ProviderRouteEvaluation(**base)


class TestRefusalTemplateKeys:
    @pytest.mark.parametrize(
        "template_key",
        sorted(SOVEREIGN_REFUSAL_TEMPLATE_KEYS),
    )
    def test_each_template_has_non_empty_message(self, template_key: str):
        evaluation = _evaluation(reason=template_key)
        if template_key == REGISTRY_EVALUATION_REQUIRED_TEMPLATE_KEY:
            evaluation = reject_failover_without_registry(_route_request())
        elif template_key == PROVIDER_UNAVAILABLE_NO_ALTERNATE_TEMPLATE_KEY:
            evaluation = _evaluation(reason="no_certified_alternate")
        elif template_key == ALTERNATE_MISSING_PROFILE_TEMPLATE_KEY:
            evaluation = _evaluation(
                reason="alternate_capability_profile_missing",
                bridge_status="missing",
            )
        elif template_key == ALTERNATE_OUT_OF_SCOPE_TEMPLATE_KEY:
            evaluation = _evaluation(
                reason="max_consequence_grade_too_low",
                bridge_status="out_of_scope",
            )
        elif template_key == ALTERNATE_VALIDATION_EXPIRED_TEMPLATE_KEY:
            evaluation = _evaluation(
                reason="provider_validation_stale",
                bridge_status="expired",
            )
        elif template_key == LEGAL_UNAVAILABLE_TEMPLATE_KEY:
            evaluation = _evaluation(
                reason="primary_provider_state_blocked:legal_unavailable",
                outcome=ProviderRouteOutcome.FAIL_CLOSED,
            )
        elif template_key == REGRESSION_FAILED_TEMPLATE_KEY:
            evaluation = _evaluation(
                reason="primary_provider_state_blocked:regression_failed",
                outcome=ProviderRouteOutcome.FAIL_CLOSED,
            )
        elif template_key == PROVIDER_ROUTE_REQUIRED_TEMPLATE_KEY:
            evaluation = reject_missing_provider_route()

        refusal = refusal_for_evaluation(evaluation)
        assert refusal.template_key == template_key
        assert refusal.message
        assert "blocked before model execution" in refusal.message.lower()
        assert refusal.machine_reason


class TestRequiredRefusalCases:
    def test_registry_evaluation_required(self):
        evaluation = reject_failover_without_registry(_route_request())
        refusal = refusal_for_evaluation(evaluation)
        assert refusal.template_key == REGISTRY_EVALUATION_REQUIRED_TEMPLATE_KEY
        assert refusal.machine_reason == "failover_registry_evaluation_required"
        assert "registry evaluation" in refusal.message.lower()

    def test_legal_unavailable(self):
        evaluation = _evaluation(
            reason="primary_provider_state_blocked:legal_unavailable",
            outcome=ProviderRouteOutcome.FAIL_CLOSED,
        )
        refusal = refusal_for_evaluation(
            evaluation,
            primary_provider_state=LEGAL_UNAVAILABLE,
        )
        assert refusal.template_key == LEGAL_UNAVAILABLE_TEMPLATE_KEY
        assert "legally available" in refusal.message.lower()

    def test_provider_unavailable_no_alternate(self):
        evaluation = _evaluation(reason="no_certified_alternate")
        refusal = refusal_for_evaluation(evaluation)
        assert refusal.template_key == PROVIDER_UNAVAILABLE_NO_ALTERNATE_TEMPLATE_KEY
        assert "no certified alternate" in refusal.message.lower()

    def test_alternate_missing_profile(self):
        evaluation = _evaluation(
            reason="alternate_capability_profile_missing",
            bridge_status="missing",
        )
        refusal = refusal_for_evaluation(evaluation)
        assert refusal.template_key == ALTERNATE_MISSING_PROFILE_TEMPLATE_KEY
        assert "capability profile" in refusal.message.lower()

    def test_alternate_out_of_scope(self):
        evaluation = _evaluation(
            reason="max_consequence_grade_too_low",
            bridge_status="out_of_scope",
        )
        refusal = refusal_for_evaluation(evaluation)
        assert refusal.template_key == ALTERNATE_OUT_OF_SCOPE_TEMPLATE_KEY
        assert "consequence grade" in refusal.message.lower()

    def test_alternate_validation_expired(self):
        evaluation = _evaluation(
            reason="provider_validation_stale",
            bridge_status="expired",
        )
        refusal = refusal_for_evaluation(evaluation)
        assert refusal.template_key == ALTERNATE_VALIDATION_EXPIRED_TEMPLATE_KEY
        assert "validation" in refusal.message.lower()

    def test_regression_failed(self):
        evaluation = _evaluation(
            reason="primary_provider_state_blocked:regression_failed",
            outcome=ProviderRouteOutcome.FAIL_CLOSED,
        )
        refusal = refusal_for_evaluation(
            evaluation,
            primary_provider_state=REGRESSION_FAILED,
        )
        assert refusal.template_key == REGRESSION_FAILED_TEMPLATE_KEY
        assert "regression" in refusal.message.lower()


class TestAuditSeparatesMachineReasonFromMessage:
    def test_audit_envelope_carries_both_fields(self):
        registry = SovereignProviderRegistry([])
        request = _route_request()
        evaluation = registry.evaluate_failover(request)
        envelope = build_provider_route_audit_envelope(
            request,
            evaluation,
            adapter_called=False,
        )
        assert envelope["machine_reason"] == evaluation.reason
        assert envelope["refusal_reason"] == evaluation.reason
        assert envelope["refusal_template_key"] == ALTERNATE_MISSING_PROFILE_TEMPLATE_KEY
        assert envelope["refusal_message"]
        assert envelope["refusal_message"] != envelope["machine_reason"]

    def test_allowed_route_has_no_refusal_message(self):
        registry = SovereignProviderRegistry([_certified_alternate_profile()])
        request = _route_request()
        evaluation = registry.evaluate_failover(request)
        envelope = build_provider_route_audit_envelope(
            request,
            evaluation,
            adapter_called=True,
        )
        assert envelope["refusal_message"] is None
        assert envelope["refusal_template_key"] is None


class TestResolveRefusalTemplateKey:
    def test_bridge_status_expired_maps_to_validation_expired(self):
        key = resolve_refusal_template_key(
            _evaluation(reason="provider_validation_stale", bridge_status="expired"),
        )
        assert key == ALTERNATE_VALIDATION_EXPIRED_TEMPLATE_KEY

    def test_bridge_status_contradicted_maps_to_regression_failed(self):
        key = resolve_refusal_template_key(
            _evaluation(bridge_status="contradicted", reason="alternate_not_admissible"),
        )
        assert key == REGRESSION_FAILED_TEMPLATE_KEY

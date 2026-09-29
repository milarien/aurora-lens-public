"""Tests for Track B Phase 3C — validation freshness and regression suite."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from aurora_lens.sovereign.provider_profile import ProviderProfile
from aurora_lens.sovereign.provider_registry import (
    ProviderRouteOutcome,
    SovereignProviderRegistry,
)
from aurora_lens.sovereign.provider_state import (
    PROVIDER_IDENTITY_CHANGED,
    PROVIDER_UNAVAILABLE,
    REGRESSION_FAILED,
    VALIDATED_CURRENT,
    VALIDATION_STALE,
)
from aurora_lens.sovereign.regression_suite import (
    assess_provider_validation,
    compute_regression_report_hash,
    provider_identity_changed,
    resolve_route_primary_state,
)
from aurora_lens.sovereign.validation_freshness import (
    ValidationFreshnessPolicy,
    validation_timestamp_is_stale,
)
from tests.test_sovereign_failover_bridge import (
    PRIMARY_ID,
    _certified_alternate_profile,
    _route_request,
)


def _fresh_profile(**overrides: object) -> ProviderProfile:
    base = _certified_alternate_profile(
        provider_id=PRIMARY_ID,
        last_validated_at=datetime.now(timezone.utc).isoformat(),
        regression_result="passed",
    )
    if not overrides:
        return base
    data = {
        "provider_id": base.provider_id,
        "provider_name": base.provider_name,
        "model_id": base.model_id,
        "endpoint_url": base.endpoint_url,
        "hosting_jurisdiction": base.hosting_jurisdiction,
        "data_boundary": base.data_boundary,
        "allowed_data_classes": base.allowed_data_classes,
        "permitted_domains": base.permitted_domains,
        "max_consequence_grade": base.max_consequence_grade,
        "supports_tools": base.supports_tools,
        "supports_structured_output": base.supports_structured_output,
        "context_window_tokens": base.context_window_tokens,
        "retention_policy": base.retention_policy,
        "validation_suite_id": base.validation_suite_id,
        "last_validated_at": base.last_validated_at,
        "status": base.status,
        "regression_result": base.regression_result,
        "report_hash": base.report_hash,
        "failed_checks": base.failed_checks,
        "observed_model_id": base.observed_model_id,
    }
    data.update(overrides)
    return ProviderProfile(**data)


class TestValidationFreshness:
    def test_missing_timestamp_is_stale(self):
        assert validation_timestamp_is_stale("", policy=ValidationFreshnessPolicy(stale_threshold_days=30))

    def test_old_timestamp_is_stale(self):
        old = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat()
        assert validation_timestamp_is_stale(old, policy=ValidationFreshnessPolicy(stale_threshold_days=30))

    def test_recent_timestamp_is_fresh(self):
        recent = datetime.now(timezone.utc).isoformat()
        assert not validation_timestamp_is_stale(recent, policy=ValidationFreshnessPolicy(stale_threshold_days=30))


class TestRegressionSuite:
    def test_report_hash_is_stable(self):
        body = {
            "provider_id": "p1",
            "validation_suite_id": "provider_regression_v1",
            "last_validated_at": "2026-06-25T10:00:00+10:00",
            "result": "passed",
            "failed_checks": [],
            "observed_model_id": None,
            "model_id": "m1",
        }
        assert compute_regression_report_hash(body) == compute_regression_report_hash(body)

    def test_provider_identity_changed_detection(self):
        profile = _fresh_profile(model_id="claude-haiku", observed_model_id="claude-opus")
        assert provider_identity_changed(profile)

    def test_failed_checks_map_to_regression_failed(self):
        profile = _fresh_profile(failed_checks=("schema_conformance",))
        assessment = assess_provider_validation(profile, declared_state=VALIDATED_CURRENT)
        assert assessment.effective_state == REGRESSION_FAILED
        assert assessment.failed_checks == ("schema_conformance",)
        assert assessment.report_hash.startswith("sha256:")


class TestRegistryValidationResolution:
    def test_declared_validated_current_downgrades_when_stale(self):
        old = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        registry = SovereignProviderRegistry([
            _fresh_profile(
                provider_id=PRIMARY_ID,
                last_validated_at=old,
                regression_result="passed",
                status=VALIDATED_CURRENT,
            ),
        ], validation_policy=ValidationFreshnessPolicy(stale_threshold_days=30))
        evaluation = registry.evaluate_failover(
            _route_request(primary_state=VALIDATED_CURRENT, primary_provider_id=PRIMARY_ID),
        )
        assert evaluation.outcome == ProviderRouteOutcome.FAIL_CLOSED
        assert evaluation.reason == f"primary_provider_state_blocked:{VALIDATION_STALE}"
        assert evaluation.validation_summary is not None
        assert evaluation.validation_summary["primary_effective_state"] == VALIDATION_STALE

    def test_provider_identity_changed_fail_closed(self):
        registry = SovereignProviderRegistry([
            _fresh_profile(
                provider_id=PRIMARY_ID,
                observed_model_id="different-model",
            ),
        ])
        evaluation = registry.evaluate_failover(
            _route_request(primary_state=VALIDATED_CURRENT, primary_provider_id=PRIMARY_ID),
        )
        assert evaluation.outcome == ProviderRouteOutcome.FAIL_CLOSED
        assert evaluation.reason == f"primary_provider_state_blocked:{PROVIDER_IDENTITY_CHANGED}"
        assert evaluation.validation_summary["primary_provider_identity_changed"] is True

    def test_operational_unavailable_preserved_for_failover(self):
        registry = SovereignProviderRegistry([_fresh_profile(provider_id=PRIMARY_ID)])
        state, assessment = resolve_route_primary_state(
            registry.get_profile(PRIMARY_ID),
            declared_state=PROVIDER_UNAVAILABLE,
        )
        assert state == PROVIDER_UNAVAILABLE
        assert assessment is not None

    def test_alternate_validation_summary_on_refused_failover(self):
        registry = SovereignProviderRegistry([
            _certified_alternate_profile(status=VALIDATION_STALE, max_consequence_grade="high"),
        ], validation_policy=ValidationFreshnessPolicy(stale_threshold_days=30))
        evaluation = registry.evaluate_failover(_route_request())
        assert evaluation.validation_summary is not None
        assert evaluation.validation_summary.get("alternate_effective_state") == VALIDATION_STALE
        assert evaluation.validation_summary.get("alternate_report_hash", "").startswith("sha256:")

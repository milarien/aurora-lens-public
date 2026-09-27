"""Regression-suite evaluation and placeholder runner for provider validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from aurora_lens.sovereign.provider_profile import ProviderProfile
from aurora_lens.sovereign.provider_state import (
    ACTIVE_PRIMARY_STATES,
    PROVIDER_IDENTITY_CHANGED,
    PROVIDER_UNAVAILABLE,
    REGRESSION_FAILED,
    REGRESSION_UNCERTAIN,
    VALIDATED_CURRENT,
    VALIDATION_STALE,
)
from aurora_lens.sovereign.validation_freshness import (
    ValidationFreshnessPolicy,
    validation_timestamp_is_stale,
)

REGRESSION_PASSED = "passed"
REGRESSION_FAILED_RESULT = "failed"
REGRESSION_UNCERTAIN_RESULT = "uncertain"


@dataclass(frozen=True)
class RegressionReport:
    """Stored regression-suite outcome for one provider profile."""

    validation_suite_id: str
    last_validated_at: str
    result: str
    report_hash: str | None = None
    failed_checks: tuple[str, ...] = ()
    observed_model_id: str | None = None


@dataclass(frozen=True)
class ProviderValidationAssessment:
    """Resolved validation posture for route admissibility."""

    effective_state: str
    declared_state: str
    report_hash: str | None
    failed_checks: tuple[str, ...]
    provider_identity_changed: bool
    validation_stale: bool
    regression_result: str | None


def _normalize_model_id(value: str | None) -> str:
    return str(value or "").strip().lower()


def provider_identity_changed(profile: ProviderProfile) -> bool:
    """True when observed runtime model id differs from declared profile model id."""
    observed = _normalize_model_id(profile.observed_model_id)
    declared = _normalize_model_id(profile.model_id)
    if not observed or not declared:
        return False
    return observed != declared


def compute_regression_report_hash(body: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(body, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def regression_report_from_profile(profile: ProviderProfile) -> RegressionReport:
    """Build a regression report view from declared profile fields."""
    failed_checks = tuple(str(c).strip() for c in profile.failed_checks if str(c).strip())
    report_hash = profile.report_hash
    if not report_hash:
        report_hash = compute_regression_report_hash({
            "provider_id": profile.provider_id,
            "validation_suite_id": profile.validation_suite_id,
            "last_validated_at": profile.last_validated_at,
            "result": profile.regression_result,
            "failed_checks": list(failed_checks),
            "observed_model_id": profile.observed_model_id,
            "model_id": profile.model_id,
        })
    return RegressionReport(
        validation_suite_id=profile.validation_suite_id,
        last_validated_at=profile.last_validated_at,
        result=str(profile.regression_result or "").strip().lower(),
        report_hash=report_hash,
        failed_checks=failed_checks,
        observed_model_id=profile.observed_model_id,
    )


def apply_regression_report_to_profile(
    profile: ProviderProfile,
    report: RegressionReport,
) -> ProviderProfile:
    """Return a profile copy with regression validation fields refreshed from a report."""
    return ProviderProfile(
        provider_id=profile.provider_id,
        provider_name=profile.provider_name,
        model_id=profile.model_id,
        endpoint_url=profile.endpoint_url,
        hosting_jurisdiction=profile.hosting_jurisdiction,
        data_boundary=profile.data_boundary,
        allowed_regions=profile.allowed_regions,
        allowed_data_classes=profile.allowed_data_classes,
        permitted_domains=profile.permitted_domains,
        max_consequence_grade=profile.max_consequence_grade,
        supports_tools=profile.supports_tools,
        supports_structured_output=profile.supports_structured_output,
        context_window_tokens=profile.context_window_tokens,
        retention_policy=profile.retention_policy,
        validation_suite_id=report.validation_suite_id or profile.validation_suite_id,
        last_validated_at=report.last_validated_at,
        status=profile.status,
        regression_result=report.result,
        report_hash=report.report_hash,
        failed_checks=report.failed_checks,
        observed_model_id=report.observed_model_id,
    )


def run_regression_suite_placeholder(profile: ProviderProfile) -> RegressionReport:
    """Placeholder runner: evaluates declared/stored report — no live model calls."""
    return regression_report_from_profile(profile)


def assess_provider_validation(
    profile: ProviderProfile,
    *,
    declared_state: str,
    policy: ValidationFreshnessPolicy | None = None,
) -> ProviderValidationAssessment:
    """Derive effective provider state from freshness, regression report, and identity."""
    freshness = policy or ValidationFreshnessPolicy()
    declared = str(declared_state or profile.status or "").strip().lower()
    report = run_regression_suite_placeholder(profile)
    identity_changed = provider_identity_changed(profile)
    stale = validation_timestamp_is_stale(
        profile.last_validated_at,
        policy=freshness,
    )
    failed_checks = report.failed_checks
    regression_result = report.result or None

    if identity_changed:
        effective = PROVIDER_IDENTITY_CHANGED
    elif regression_result == REGRESSION_FAILED_RESULT or failed_checks:
        effective = REGRESSION_FAILED
    elif regression_result == REGRESSION_UNCERTAIN_RESULT:
        effective = REGRESSION_UNCERTAIN
    elif stale or profile.status == VALIDATION_STALE:
        effective = VALIDATION_STALE
    elif regression_result == REGRESSION_PASSED and not stale:
        effective = VALIDATED_CURRENT
    elif declared:
        effective = declared
    else:
        effective = str(profile.status or VALIDATED_CURRENT).strip().lower()

    return ProviderValidationAssessment(
        effective_state=effective,
        declared_state=declared,
        report_hash=report.report_hash,
        failed_checks=failed_checks,
        provider_identity_changed=identity_changed,
        validation_stale=stale,
        regression_result=regression_result,
    )


def assessment_to_summary(assessment: ProviderValidationAssessment) -> dict[str, Any]:
    return {
        "effective_state": assessment.effective_state,
        "declared_state": assessment.declared_state,
        "report_hash": assessment.report_hash,
        "failed_checks": list(assessment.failed_checks),
        "provider_identity_changed": assessment.provider_identity_changed,
        "validation_stale": assessment.validation_stale,
        "regression_result": assessment.regression_result,
    }


def resolve_route_primary_state(
    profile: ProviderProfile | None,
    *,
    declared_state: str,
    policy: ValidationFreshnessPolicy | None = None,
) -> tuple[str, ProviderValidationAssessment | None]:
    """Resolve primary route state; preserve operational unavailable when declared."""
    declared = str(declared_state or "").strip().lower()
    if profile is None:
        return declared, None
    assessment = assess_provider_validation(
        profile,
        declared_state=declared_state,
        policy=policy,
    )
    if declared == PROVIDER_UNAVAILABLE:
        return PROVIDER_UNAVAILABLE, assessment
    if declared not in ACTIVE_PRIMARY_STATES:
        if assessment.effective_state in {PROVIDER_IDENTITY_CHANGED, REGRESSION_FAILED}:
            return assessment.effective_state, assessment
        return declared or assessment.effective_state, assessment
    return assessment.effective_state, assessment

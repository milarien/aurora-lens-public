"""Deterministic governed refusal copy for sovereign provider-route failures.

Templates are selected by machine-readable failure reason — never improvised by
the model. Audit rows carry ``refusal_reason`` / ``machine_reason`` separately
from ``refusal_message`` (human-facing text).
"""

from __future__ import annotations

from dataclasses import dataclass

from aurora_lens.sovereign.provider_registry import ProviderRouteEvaluation, ProviderRouteOutcome
from aurora_lens.sovereign.provider_state import LEGAL_UNAVAILABLE, REGRESSION_FAILED

PROVIDER_ROUTE_REQUIRED_TEMPLATE_KEY = "provider_route_required"
LEGAL_UNAVAILABLE_TEMPLATE_KEY = "legal_unavailable"
PROVIDER_UNAVAILABLE_NO_ALTERNATE_TEMPLATE_KEY = "provider_unavailable_no_alternate"
ALTERNATE_MISSING_PROFILE_TEMPLATE_KEY = "alternate_missing_profile"
ALTERNATE_OUT_OF_SCOPE_TEMPLATE_KEY = "alternate_out_of_scope"
ALTERNATE_VALIDATION_EXPIRED_TEMPLATE_KEY = "alternate_validation_expired"
REGISTRY_EVALUATION_REQUIRED_TEMPLATE_KEY = "registry_evaluation_required"
REGRESSION_FAILED_TEMPLATE_KEY = "regression_failed"
ROUTE_NOT_ADMISSIBLE_TEMPLATE_KEY = "route_not_admissible"

SOVEREIGN_REFUSAL_TEMPLATE_KEYS: frozenset[str] = frozenset({
    PROVIDER_ROUTE_REQUIRED_TEMPLATE_KEY,
    LEGAL_UNAVAILABLE_TEMPLATE_KEY,
    PROVIDER_UNAVAILABLE_NO_ALTERNATE_TEMPLATE_KEY,
    ALTERNATE_MISSING_PROFILE_TEMPLATE_KEY,
    ALTERNATE_OUT_OF_SCOPE_TEMPLATE_KEY,
    ALTERNATE_VALIDATION_EXPIRED_TEMPLATE_KEY,
    REGISTRY_EVALUATION_REQUIRED_TEMPLATE_KEY,
    REGRESSION_FAILED_TEMPLATE_KEY,
    ROUTE_NOT_ADMISSIBLE_TEMPLATE_KEY,
})

_REFUSAL_MESSAGES: dict[str, str] = {
    PROVIDER_ROUTE_REQUIRED_TEMPLATE_KEY: (
        "This request cannot proceed because sovereign provider route metadata was not "
        "attached for registry evaluation. Provider routes must be declared or auto-constructed "
        "before model execution. The request has been blocked before model execution and "
        "recorded for authorized review."
    ),
    LEGAL_UNAVAILABLE_TEMPLATE_KEY: (
        "This request cannot proceed because the approved provider route is not legally "
        "available for this jurisdiction or consequence class. The request has been blocked "
        "before model execution and recorded for authorized review."
    ),
    PROVIDER_UNAVAILABLE_NO_ALTERNATE_TEMPLATE_KEY: (
        "This request cannot proceed because the approved provider route is unavailable and "
        "no certified alternate is available for this consequence class. The request has been "
        "blocked before model execution and recorded for authorized review."
    ),
    ALTERNATE_MISSING_PROFILE_TEMPLATE_KEY: (
        "This request cannot proceed because the nominated alternate provider has no declared "
        "capability profile. Failover has been refused to prevent an uncontrolled provider "
        "substitution. The request has been blocked before model execution and recorded for "
        "authorized review."
    ),
    ALTERNATE_OUT_OF_SCOPE_TEMPLATE_KEY: (
        "This request cannot proceed because the available replacement provider is not certified "
        "for this task domain or consequence grade. Failover has been refused to prevent an "
        "uncontrolled operational decision. The request has been blocked before model execution "
        "and recorded for authorized review."
    ),
    ALTERNATE_VALIDATION_EXPIRED_TEMPLATE_KEY: (
        "This request cannot proceed because the available replacement provider's validation "
        "state is not current for this consequence grade. Failover has been refused. The "
        "request has been blocked before model execution and recorded for authorized review."
    ),
    REGISTRY_EVALUATION_REQUIRED_TEMPLATE_KEY: (
        "This request cannot proceed because provider failover was attempted without Sovereign "
        "Provider Registry evaluation. Failover must never proceed silently. The request has "
        "been blocked before model execution and recorded for authorized review."
    ),
    REGRESSION_FAILED_TEMPLATE_KEY: (
        "This request cannot proceed because the provider route failed declared regression "
        "validation and is not admissible for this consequence class. The request has been "
        "blocked before model execution and recorded for authorized review."
    ),
    ROUTE_NOT_ADMISSIBLE_TEMPLATE_KEY: (
        "This request cannot proceed because the provider route is not admissible under "
        "declared registry policy. The request has been blocked before model execution and "
        "recorded for authorized review."
    ),
}


@dataclass(frozen=True)
class SovereignRefusal:
    """Structured refusal: template key, machine reason, and human-facing message."""

    template_key: str
    machine_reason: str
    message: str


def resolve_refusal_template_key(
    evaluation: ProviderRouteEvaluation,
    *,
    primary_provider_state: str | None = None,
) -> str:
    """Map evaluation to a deterministic template key."""
    reason = str(evaluation.reason or "").strip()
    primary_state = str(primary_provider_state or "").strip().lower()

    if not evaluation.registry_evaluated:
        if reason == "provider_route_metadata_required":
            return PROVIDER_ROUTE_REQUIRED_TEMPLATE_KEY
        return REGISTRY_EVALUATION_REQUIRED_TEMPLATE_KEY

    if reason == "provider_route_metadata_required":
        return PROVIDER_ROUTE_REQUIRED_TEMPLATE_KEY
    if reason == "failover_registry_evaluation_required":
        return REGISTRY_EVALUATION_REQUIRED_TEMPLATE_KEY

    if primary_state == LEGAL_UNAVAILABLE or reason == "legal_unavailable":
        return LEGAL_UNAVAILABLE_TEMPLATE_KEY
    if reason.startswith("primary_provider_state_blocked:legal_unavailable"):
        return LEGAL_UNAVAILABLE_TEMPLATE_KEY

    if primary_state == REGRESSION_FAILED or reason == "regression_failed":
        return REGRESSION_FAILED_TEMPLATE_KEY
    if reason.startswith("primary_provider_state_blocked:regression_failed"):
        return REGRESSION_FAILED_TEMPLATE_KEY

    if reason == "no_certified_alternate":
        return PROVIDER_UNAVAILABLE_NO_ALTERNATE_TEMPLATE_KEY

    if reason == "alternate_capability_profile_missing":
        return ALTERNATE_MISSING_PROFILE_TEMPLATE_KEY
    if evaluation.bridge_status == "missing":
        return ALTERNATE_MISSING_PROFILE_TEMPLATE_KEY

    if reason == "provider_validation_stale" or evaluation.bridge_status == "expired":
        return ALTERNATE_VALIDATION_EXPIRED_TEMPLATE_KEY

    if evaluation.bridge_status == "contradicted":
        return REGRESSION_FAILED_TEMPLATE_KEY

    if reason in {
        "max_consequence_grade_too_low",
        "domain_not_permitted",
        "data_class_not_permitted",
        "capability_profile_insufficient",
        "alternate_not_admissible",
    }:
        return ALTERNATE_OUT_OF_SCOPE_TEMPLATE_KEY
    if evaluation.bridge_status == "out_of_scope":
        return ALTERNATE_OUT_OF_SCOPE_TEMPLATE_KEY

    if evaluation.outcome == ProviderRouteOutcome.FAIL_CLOSED:
        if "legal" in reason:
            return LEGAL_UNAVAILABLE_TEMPLATE_KEY
        if "regression" in reason:
            return REGRESSION_FAILED_TEMPLATE_KEY
        return ROUTE_NOT_ADMISSIBLE_TEMPLATE_KEY

    if evaluation.bridge_status in {"missing", "contradicted", "expired", "out_of_scope"}:
        return ROUTE_NOT_ADMISSIBLE_TEMPLATE_KEY

    return ROUTE_NOT_ADMISSIBLE_TEMPLATE_KEY


def refusal_for_evaluation(
    evaluation: ProviderRouteEvaluation,
    *,
    primary_provider_state: str | None = None,
) -> SovereignRefusal:
    """Return structured refusal for a blocked sovereign route evaluation."""
    template_key = resolve_refusal_template_key(
        evaluation,
        primary_provider_state=primary_provider_state,
    )
    machine_reason = str(evaluation.reason or template_key)
    message = _REFUSAL_MESSAGES.get(template_key, _REFUSAL_MESSAGES[ROUTE_NOT_ADMISSIBLE_TEMPLATE_KEY])
    return SovereignRefusal(
        template_key=template_key,
        machine_reason=machine_reason,
        message=message,
    )


def refusal_message_for_evaluation(
    evaluation: ProviderRouteEvaluation,
    *,
    primary_provider_state: str | None = None,
) -> str:
    """Human-facing refusal text (backward-compatible helper)."""
    return refusal_for_evaluation(
        evaluation,
        primary_provider_state=primary_provider_state,
    ).message

"""Declared provider lifecycle states for the Sovereign Provider Registry."""

from __future__ import annotations

from typing import Final

ProviderState = str

VALIDATED_CURRENT: Final[ProviderState] = "validated_current"
VALIDATION_STALE: Final[ProviderState] = "validation_stale"
PROVIDER_UNAVAILABLE: Final[ProviderState] = "provider_unavailable"
PROVIDER_ACCESS_UNCERTAINTY: Final[ProviderState] = "provider_access_uncertainty"
GEOFENCED_CONFIRMED: Final[ProviderState] = "geofenced_confirmed"
LEGAL_UNAVAILABLE: Final[ProviderState] = "legal_unavailable"
NATIONALITY_SCREENING_REQUIRED: Final[ProviderState] = "nationality_screening_required"
CREDENTIAL_SCOPE_CHANGED: Final[ProviderState] = "credential_scope_changed"
PROVIDER_POLICY_CHANGED: Final[ProviderState] = "provider_policy_changed"
REGRESSION_UNCERTAIN: Final[ProviderState] = "regression_uncertain"
REGRESSION_FAILED: Final[ProviderState] = "regression_failed"
PROVIDER_IDENTITY_CHANGED: Final[ProviderState] = "provider_identity_changed"

ALL_PROVIDER_STATES: frozenset[str] = frozenset({
    VALIDATED_CURRENT,
    VALIDATION_STALE,
    PROVIDER_UNAVAILABLE,
    PROVIDER_ACCESS_UNCERTAINTY,
    GEOFENCED_CONFIRMED,
    LEGAL_UNAVAILABLE,
    NATIONALITY_SCREENING_REQUIRED,
    CREDENTIAL_SCOPE_CHANGED,
    PROVIDER_POLICY_CHANGED,
    REGRESSION_UNCERTAIN,
    REGRESSION_FAILED,
    PROVIDER_IDENTITY_CHANGED,
})

ACTIVE_PRIMARY_STATES: frozenset[str] = frozenset({VALIDATED_CURRENT})

"""Capability alignment rules for sovereign provider failover."""

from __future__ import annotations

from aurora_lens.sovereign.provider_profile import ProviderProfile
from aurora_lens.sovereign.provider_state import (
    REGRESSION_FAILED,
    VALIDATED_CURRENT,
    VALIDATION_STALE,
    PROVIDER_IDENTITY_CHANGED,
)

CONSEQUENCE_GRADE_RANK: dict[str, int] = {
    "minimal": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
    "critical": 4,
}


def normalize_consequence_grade(raw: str) -> str | None:
    grade = str(raw or "").strip().lower()
    if grade in CONSEQUENCE_GRADE_RANK:
        return grade
    return None


def grade_rank(grade: str) -> int | None:
    normalized = normalize_consequence_grade(grade)
    if normalized is None:
        return None
    return CONSEQUENCE_GRADE_RANK[normalized]


def alternate_supports_task(
    profile: ProviderProfile,
    *,
    task_domain: str,
    consequence_grade: str,
    data_class: str,
) -> tuple[bool, str | None]:
    """Return (aligned, refusal_reason). refusal_reason set when not aligned."""
    domain = str(task_domain or "").strip().lower()
    if domain and domain not in {d.lower() for d in profile.permitted_domains}:
        return False, "domain_not_permitted"

    task_grade = normalize_consequence_grade(consequence_grade)
    max_grade = normalize_consequence_grade(profile.max_consequence_grade)
    if task_grade is None or max_grade is None:
        return False, "capability_profile_insufficient"

    task_rank = CONSEQUENCE_GRADE_RANK[task_grade]
    max_rank = CONSEQUENCE_GRADE_RANK[max_grade]
    if task_rank > max_rank:
        return False, "max_consequence_grade_too_low"

    dc = str(data_class or "").strip().lower()
    if dc and dc not in {c.lower() for c in profile.allowed_data_classes}:
        return False, "data_class_not_permitted"

    return True, None


def bridge_status_for_provider_state(
    provider_state: str,
    *,
    consequence_grade: str,
) -> str:
    """Map resolved provider state + task grade to bridge_status."""
    state = str(provider_state or "").strip().lower()
    task_grade = normalize_consequence_grade(consequence_grade)
    if state == VALIDATED_CURRENT:
        return "valid"
    if state == VALIDATION_STALE and task_grade in {"high", "critical"}:
        return "expired"
    if state in {REGRESSION_FAILED, PROVIDER_IDENTITY_CHANGED}:
        return "contradicted"
    if state == VALIDATION_STALE:
        return "expired"
    return "out_of_scope"


def bridge_status_for_alternate_profile(
    profile: ProviderProfile,
    *,
    consequence_grade: str,
    provider_state: str | None = None,
) -> str:
    """Map profile validation state + task grade to bridge_status."""
    state = provider_state or profile.status
    return bridge_status_for_provider_state(state, consequence_grade=consequence_grade)

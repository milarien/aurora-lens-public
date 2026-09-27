"""Domain and consequence-grade policy evaluation for trust registry."""

from __future__ import annotations

from dataclasses import dataclass

from aurora_lens.trust.trust_profile import (
    TRUST_STATUS_RESTRICTED,
    TRUST_STATUS_TRUSTED,
    TRUST_STATUS_UNTRUSTED,
    TrustSourceProfile,
)
from aurora_lens.trust.source_id import normalize_source_id

CONSEQUENCE_GRADE_RANK: dict[str, int] = {
    "minimal": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
    "critical": 4,
}

TRUST_CONSEQUENCE_GRADES: frozenset[str] = frozenset(CONSEQUENCE_GRADE_RANK)


def normalize_consequence_grade(raw: object) -> str | None:
    grade = str(raw or "").strip().lower()
    if grade in TRUST_CONSEQUENCE_GRADES:
        return grade
    return None


@dataclass(frozen=True)
class TrustEvaluationResult:
    """Outcome of registry-backed trust policy evaluation."""

    source_id: str
    admissible: bool
    registry_evaluated: bool
    reason: str | None = None
    trust_decision: str | None = None


def evaluate_trust_admissibility(
    profile: TrustSourceProfile | None,
    *,
    source_id: str,
    task_domain: str,
    consequence_grade: str,
) -> TrustEvaluationResult:
    """Evaluate whether a declared source is admissible for task domain + grade.

    Unknown sources (not in registry) are not treated as untrusted — callers
    must not emit ``source_untrusted`` when ``registry_evaluated`` is False.
    """
    normalized_id = normalize_source_id(source_id)
    if normalized_id is None:
        return TrustEvaluationResult(
            source_id=str(source_id or ""),
            admissible=True,
            registry_evaluated=False,
            reason="invalid_source_id",
        )

    if profile is None:
        return TrustEvaluationResult(
            source_id=normalized_id,
            admissible=True,
            registry_evaluated=False,
            reason="source_not_in_registry",
        )

    status = str(profile.trust_status or "").strip().lower()
    if status == TRUST_STATUS_UNTRUSTED:
        return TrustEvaluationResult(
            source_id=normalized_id,
            admissible=False,
            registry_evaluated=True,
            reason="source_marked_untrusted",
            trust_decision="deny",
        )

    domain = str(task_domain or "").strip().lower()
    permitted = {d.lower() for d in profile.permitted_domains}
    if domain and permitted and domain not in permitted:
        return TrustEvaluationResult(
            source_id=normalized_id,
            admissible=False,
            registry_evaluated=True,
            reason="domain_not_permitted",
            trust_decision="deny",
        )

    task_grade = normalize_consequence_grade(consequence_grade)
    max_grade = normalize_consequence_grade(profile.max_consequence_grade)
    if task_grade is None or max_grade is None:
        return TrustEvaluationResult(
            source_id=normalized_id,
            admissible=False,
            registry_evaluated=True,
            reason="capability_profile_insufficient",
            trust_decision="deny",
        )

    if CONSEQUENCE_GRADE_RANK[task_grade] > CONSEQUENCE_GRADE_RANK[max_grade]:
        return TrustEvaluationResult(
            source_id=normalized_id,
            admissible=False,
            registry_evaluated=True,
            reason="max_consequence_grade_too_low",
            trust_decision="deny",
        )

    if status not in {TRUST_STATUS_TRUSTED, TRUST_STATUS_RESTRICTED}:
        return TrustEvaluationResult(
            source_id=normalized_id,
            admissible=False,
            registry_evaluated=True,
            reason="unknown_trust_status",
            trust_decision="deny",
        )

    return TrustEvaluationResult(
        source_id=normalized_id,
        admissible=True,
        registry_evaluated=True,
        reason=None,
        trust_decision="allow",
    )

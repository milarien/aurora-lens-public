"""Deterministic freshness/revalidation permission policy.

This module is intentionally pure and does not inspect user prose.
It consumes canonical freshness failure facts and returns a governed
permission outcome for commitment control wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FreshnessFailureKind(str, Enum):
    """Canonical freshness/revalidation failures consumed by policy."""

    STALE_EVIDENCE = "stale_evidence"
    MISSING_FRESHNESS = "missing_freshness"
    UNREVALIDATED_ORIENTATION = "unrevalidated_orientation"


class ConsequenceGrade(str, Enum):
    """Ordered consequence grades used by threshold and freshness policy."""

    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class FreshnessPermissionOutcome(str, Enum):
    """Permission outcomes produced by freshness policy."""

    PASS_WITH_WARNING = "PASS_WITH_WARNING"
    CONTAIN = "CONTAIN"
    RE_ATTEST = "RE_ATTEST"
    QUEUE_FOR_REVALIDATION = "QUEUE_FOR_REVALIDATION"
    SUPERVISED_CONTINUE = "SUPERVISED_CONTINUE"
    ESCALATE = "ESCALATE"
    SUSPEND = "SUSPEND"
    HARD_STOP = "HARD_STOP"


class AuthorityState(str, Enum):
    """Authority capability envelope available for this attempted action."""

    NONE = "none"
    OPERATOR = "operator"
    SUPERVISED = "supervised"
    FULL = "full"


_BASE_OUTCOME_MATRIX: dict[FreshnessFailureKind, dict[ConsequenceGrade, FreshnessPermissionOutcome]] = {
    FreshnessFailureKind.STALE_EVIDENCE: {
        ConsequenceGrade.MINIMAL: FreshnessPermissionOutcome.PASS_WITH_WARNING,
        ConsequenceGrade.LOW: FreshnessPermissionOutcome.CONTAIN,
        ConsequenceGrade.MODERATE: FreshnessPermissionOutcome.RE_ATTEST,
        ConsequenceGrade.HIGH: FreshnessPermissionOutcome.ESCALATE,
        ConsequenceGrade.CRITICAL: FreshnessPermissionOutcome.HARD_STOP,
    },
    FreshnessFailureKind.MISSING_FRESHNESS: {
        ConsequenceGrade.MINIMAL: FreshnessPermissionOutcome.CONTAIN,
        ConsequenceGrade.LOW: FreshnessPermissionOutcome.RE_ATTEST,
        ConsequenceGrade.MODERATE: FreshnessPermissionOutcome.QUEUE_FOR_REVALIDATION,
        ConsequenceGrade.HIGH: FreshnessPermissionOutcome.SUSPEND,
        ConsequenceGrade.CRITICAL: FreshnessPermissionOutcome.HARD_STOP,
    },
    FreshnessFailureKind.UNREVALIDATED_ORIENTATION: {
        ConsequenceGrade.MINIMAL: FreshnessPermissionOutcome.QUEUE_FOR_REVALIDATION,
        ConsequenceGrade.LOW: FreshnessPermissionOutcome.SUPERVISED_CONTINUE,
        ConsequenceGrade.MODERATE: FreshnessPermissionOutcome.ESCALATE,
        ConsequenceGrade.HIGH: FreshnessPermissionOutcome.SUSPEND,
        ConsequenceGrade.CRITICAL: FreshnessPermissionOutcome.HARD_STOP,
    },
}

_ALLOWED_OUTCOMES_BY_AUTHORITY: dict[AuthorityState, frozenset[FreshnessPermissionOutcome]] = {
    AuthorityState.NONE: frozenset({
        FreshnessPermissionOutcome.CONTAIN,
        FreshnessPermissionOutcome.QUEUE_FOR_REVALIDATION,
        FreshnessPermissionOutcome.SUSPEND,
        FreshnessPermissionOutcome.HARD_STOP,
    }),
    AuthorityState.OPERATOR: frozenset({
        FreshnessPermissionOutcome.PASS_WITH_WARNING,
        FreshnessPermissionOutcome.CONTAIN,
        FreshnessPermissionOutcome.RE_ATTEST,
        FreshnessPermissionOutcome.QUEUE_FOR_REVALIDATION,
        FreshnessPermissionOutcome.ESCALATE,
        FreshnessPermissionOutcome.SUSPEND,
        FreshnessPermissionOutcome.HARD_STOP,
    }),
    AuthorityState.SUPERVISED: frozenset({
        FreshnessPermissionOutcome.PASS_WITH_WARNING,
        FreshnessPermissionOutcome.CONTAIN,
        FreshnessPermissionOutcome.RE_ATTEST,
        FreshnessPermissionOutcome.QUEUE_FOR_REVALIDATION,
        FreshnessPermissionOutcome.SUPERVISED_CONTINUE,
        FreshnessPermissionOutcome.ESCALATE,
        FreshnessPermissionOutcome.SUSPEND,
        FreshnessPermissionOutcome.HARD_STOP,
    }),
    AuthorityState.FULL: frozenset(outcome for outcome in FreshnessPermissionOutcome),
}

_ESCALATE_FALLBACK_BY_GRADE: dict[ConsequenceGrade, FreshnessPermissionOutcome] = {
    ConsequenceGrade.MINIMAL: FreshnessPermissionOutcome.CONTAIN,
    ConsequenceGrade.LOW: FreshnessPermissionOutcome.CONTAIN,
    ConsequenceGrade.MODERATE: FreshnessPermissionOutcome.QUEUE_FOR_REVALIDATION,
    ConsequenceGrade.HIGH: FreshnessPermissionOutcome.SUSPEND,
    ConsequenceGrade.CRITICAL: FreshnessPermissionOutcome.HARD_STOP,
}

_AUTHORITY_FALLBACK_BY_GRADE: dict[ConsequenceGrade, FreshnessPermissionOutcome] = {
    ConsequenceGrade.MINIMAL: FreshnessPermissionOutcome.CONTAIN,
    ConsequenceGrade.LOW: FreshnessPermissionOutcome.CONTAIN,
    ConsequenceGrade.MODERATE: FreshnessPermissionOutcome.QUEUE_FOR_REVALIDATION,
    ConsequenceGrade.HIGH: FreshnessPermissionOutcome.SUSPEND,
    ConsequenceGrade.CRITICAL: FreshnessPermissionOutcome.HARD_STOP,
}


@dataclass(frozen=True)
class FreshnessPermissionInput:
    failure_kind: FreshnessFailureKind
    consequence_grade: ConsequenceGrade
    policy_ref: str
    authority_state: AuthorityState
    reversibility: bool
    escalation_available: bool


@dataclass(frozen=True)
class FreshnessPermissionDecision:
    outcome: FreshnessPermissionOutcome
    reason_code: str
    failure_kind: FreshnessFailureKind
    consequence_grade: ConsequenceGrade
    policy_ref: str
    authority_state: AuthorityState
    reversibility: bool
    escalation_available: bool


def evaluate_freshness_permission(
    policy_input: FreshnessPermissionInput,
) -> FreshnessPermissionDecision:
    """Resolve deterministic permission outcome for freshness/revalidation failures."""
    outcome = _BASE_OUTCOME_MATRIX[policy_input.failure_kind][policy_input.consequence_grade]
    reason_code = "matrix_base"

    # Irreversible actions cannot retain pass-with-warning or supervised continuation.
    if not policy_input.reversibility:
        if outcome == FreshnessPermissionOutcome.PASS_WITH_WARNING:
            outcome = FreshnessPermissionOutcome.CONTAIN
            reason_code = "irreversible_downgrade_from_warning"
        elif outcome == FreshnessPermissionOutcome.SUPERVISED_CONTINUE:
            outcome = FreshnessPermissionOutcome.ESCALATE
            reason_code = "irreversible_downgrade_from_supervised_continue"

    # Warning corridors require explicit policy reference.
    if outcome == FreshnessPermissionOutcome.PASS_WITH_WARNING and not policy_input.policy_ref.strip():
        outcome = FreshnessPermissionOutcome.CONTAIN
        reason_code = "missing_policy_ref_for_warning"

    # Escalation cannot be emitted when no escalation path is available.
    if outcome == FreshnessPermissionOutcome.ESCALATE and not policy_input.escalation_available:
        outcome = _ESCALATE_FALLBACK_BY_GRADE[policy_input.consequence_grade]
        reason_code = "escalation_unavailable_fallback"

    allowed = _ALLOWED_OUTCOMES_BY_AUTHORITY[policy_input.authority_state]
    if outcome not in allowed:
        outcome = _AUTHORITY_FALLBACK_BY_GRADE[policy_input.consequence_grade]
        reason_code = "authority_restriction_fallback"
        if outcome not in allowed:
            if FreshnessPermissionOutcome.SUSPEND in allowed:
                outcome = FreshnessPermissionOutcome.SUSPEND
                reason_code = "authority_restriction_suspend_fallback"
            else:
                outcome = FreshnessPermissionOutcome.HARD_STOP
                reason_code = "authority_restriction_hard_stop_fallback"

    return FreshnessPermissionDecision(
        outcome=outcome,
        reason_code=reason_code,
        failure_kind=policy_input.failure_kind,
        consequence_grade=policy_input.consequence_grade,
        policy_ref=policy_input.policy_ref,
        authority_state=policy_input.authority_state,
        reversibility=policy_input.reversibility,
        escalation_available=policy_input.escalation_available,
    )


__all__ = [
    "AuthorityState",
    "ConsequenceGrade",
    "FreshnessFailureKind",
    "FreshnessPermissionDecision",
    "FreshnessPermissionInput",
    "FreshnessPermissionOutcome",
    "evaluate_freshness_permission",
]

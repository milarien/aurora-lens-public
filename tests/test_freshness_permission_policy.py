"""Deterministic tests for freshness/revalidation permission policy."""

from __future__ import annotations

import pytest

from aurora_lens.govern.freshness_permission_policy import (
    AuthorityState,
    ConsequenceGrade,
    FreshnessFailureKind,
    FreshnessPermissionInput,
    FreshnessPermissionOutcome,
    evaluate_freshness_permission,
)


@pytest.mark.parametrize("failure_kind", list(FreshnessFailureKind))
@pytest.mark.parametrize("consequence_grade", list(ConsequenceGrade))
def test_every_kind_and_grade_pair_resolves_to_explicit_outcome(
    failure_kind: FreshnessFailureKind,
    consequence_grade: ConsequenceGrade,
) -> None:
    decision = evaluate_freshness_permission(
        FreshnessPermissionInput(
            failure_kind=failure_kind,
            consequence_grade=consequence_grade,
            policy_ref="policy.freshness.v1",
            authority_state=AuthorityState.FULL,
            reversibility=True,
            escalation_available=True,
        )
    )
    assert isinstance(decision.outcome, FreshnessPermissionOutcome)


def test_no_freshness_failure_can_return_ordinary_pass() -> None:
    for failure_kind in FreshnessFailureKind:
        for grade in ConsequenceGrade:
            decision = evaluate_freshness_permission(
                FreshnessPermissionInput(
                    failure_kind=failure_kind,
                    consequence_grade=grade,
                    policy_ref="policy.freshness.v1",
                    authority_state=AuthorityState.FULL,
                    reversibility=True,
                    escalation_available=True,
                )
            )
            assert decision.outcome.value != "PASS"


def test_escalation_unavailable_never_returns_escalate() -> None:
    decision = evaluate_freshness_permission(
        FreshnessPermissionInput(
            failure_kind=FreshnessFailureKind.STALE_EVIDENCE,
            consequence_grade=ConsequenceGrade.HIGH,
            policy_ref="policy.freshness.v1",
            authority_state=AuthorityState.OPERATOR,
            reversibility=True,
            escalation_available=False,
        )
    )
    assert decision.outcome != FreshnessPermissionOutcome.ESCALATE
    assert decision.reason_code == "escalation_unavailable_fallback"


def test_authority_restrictions_are_enforced_for_supervised_continue() -> None:
    decision = evaluate_freshness_permission(
        FreshnessPermissionInput(
            failure_kind=FreshnessFailureKind.UNREVALIDATED_ORIENTATION,
            consequence_grade=ConsequenceGrade.LOW,
            policy_ref="policy.freshness.v1",
            authority_state=AuthorityState.OPERATOR,
            reversibility=True,
            escalation_available=True,
        )
    )
    assert decision.outcome == FreshnessPermissionOutcome.CONTAIN
    assert decision.reason_code == "authority_restriction_fallback"


def test_irreversible_action_downgrades_warning_corridor() -> None:
    decision = evaluate_freshness_permission(
        FreshnessPermissionInput(
            failure_kind=FreshnessFailureKind.STALE_EVIDENCE,
            consequence_grade=ConsequenceGrade.MINIMAL,
            policy_ref="policy.freshness.v1",
            authority_state=AuthorityState.FULL,
            reversibility=False,
            escalation_available=True,
        )
    )
    assert decision.outcome == FreshnessPermissionOutcome.CONTAIN
    assert decision.reason_code == "irreversible_downgrade_from_warning"


def test_warning_requires_explicit_policy_ref() -> None:
    decision = evaluate_freshness_permission(
        FreshnessPermissionInput(
            failure_kind=FreshnessFailureKind.STALE_EVIDENCE,
            consequence_grade=ConsequenceGrade.MINIMAL,
            policy_ref="",
            authority_state=AuthorityState.FULL,
            reversibility=True,
            escalation_available=True,
        )
    )
    assert decision.outcome == FreshnessPermissionOutcome.CONTAIN
    assert decision.reason_code == "missing_policy_ref_for_warning"

"""Governor continuation alignment for freshness permission outcomes."""

from __future__ import annotations

from aurora_lens.govern.freshness_permission_policy import AuthorityState, FreshnessPermissionOutcome
from aurora_lens.governor.freshness_pathways import project_freshness_pathway


def test_pass_with_warning_and_contain_have_distinct_corridors() -> None:
    warning = project_freshness_pathway(
        outcome=FreshnessPermissionOutcome.PASS_WITH_WARNING,
        authority_state=AuthorityState.OPERATOR,
    )
    contain = project_freshness_pathway(
        outcome=FreshnessPermissionOutcome.CONTAIN,
        authority_state=AuthorityState.OPERATOR,
    )
    assert warning.pathway_id != contain.pathway_id
    assert warning.output_mode != contain.output_mode


def test_suspend_and_hard_stop_are_behaviorally_distinct() -> None:
    suspend = project_freshness_pathway(
        outcome=FreshnessPermissionOutcome.SUSPEND,
        authority_state=AuthorityState.OPERATOR,
    )
    hard_stop = project_freshness_pathway(
        outcome=FreshnessPermissionOutcome.HARD_STOP,
        authority_state=AuthorityState.OPERATOR,
    )
    assert suspend.interaction_open is True
    assert hard_stop.interaction_open is False
    assert suspend.pathway_id != hard_stop.pathway_id


def test_supervised_continue_requires_supervised_authority() -> None:
    projected = project_freshness_pathway(
        outcome=FreshnessPermissionOutcome.SUPERVISED_CONTINUE,
        authority_state=AuthorityState.OPERATOR,
    )
    assert projected.normalized_outcome == FreshnessPermissionOutcome.SUSPEND


def test_re_attest_requires_operator_authority() -> None:
    projected = project_freshness_pathway(
        outcome=FreshnessPermissionOutcome.RE_ATTEST,
        authority_state=AuthorityState.NONE,
    )
    assert projected.normalized_outcome == FreshnessPermissionOutcome.HARD_STOP

"""Governor-owned continuation mapping for freshness permission outcomes."""

from __future__ import annotations

from dataclasses import dataclass

from aurora_lens.govern.freshness_permission_policy import (
    AuthorityState,
    FreshnessPermissionOutcome,
)
from aurora_lens.governor.models import ContinuationPathway, OutputMode


@dataclass(frozen=True)
class FreshnessPathwayDecision:
    pathway_id: str
    output_mode: str
    commitment_closed: bool
    interaction_open: bool
    action_is_hard_stop: bool
    normalized_outcome: FreshnessPermissionOutcome


_MIN_AUTHORITY_RANK: dict[AuthorityState, int] = {
    AuthorityState.NONE: 0,
    AuthorityState.OPERATOR: 1,
    AuthorityState.SUPERVISED: 2,
    AuthorityState.FULL: 3,
}

_REQUIRED_OUTCOME_AUTHORITY: dict[FreshnessPermissionOutcome, int] = {
    FreshnessPermissionOutcome.RE_ATTEST: _MIN_AUTHORITY_RANK[AuthorityState.OPERATOR],
    FreshnessPermissionOutcome.QUEUE_FOR_REVALIDATION: _MIN_AUTHORITY_RANK[AuthorityState.OPERATOR],
    FreshnessPermissionOutcome.SUPERVISED_CONTINUE: _MIN_AUTHORITY_RANK[AuthorityState.SUPERVISED],
}

_OUTCOME_TO_PATHWAY: dict[FreshnessPermissionOutcome, FreshnessPathwayDecision] = {
    FreshnessPermissionOutcome.PASS_WITH_WARNING: FreshnessPathwayDecision(
        pathway_id=ContinuationPathway.P_HANDOFF_SUMMARY.value,
        output_mode=OutputMode.CONSTRAINED_RESPONSE.value,
        commitment_closed=True,
        interaction_open=True,
        action_is_hard_stop=False,
        normalized_outcome=FreshnessPermissionOutcome.PASS_WITH_WARNING,
    ),
    FreshnessPermissionOutcome.CONTAIN: FreshnessPathwayDecision(
        pathway_id=ContinuationPathway.P_ASK_DISAMBIGUATE.value,
        output_mode=OutputMode.CLARIFICATION_REQUEST.value,
        commitment_closed=True,
        interaction_open=True,
        action_is_hard_stop=False,
        normalized_outcome=FreshnessPermissionOutcome.CONTAIN,
    ),
    FreshnessPermissionOutcome.RE_ATTEST: FreshnessPathwayDecision(
        pathway_id=ContinuationPathway.P_ASK_MISSING_FACT.value,
        output_mode=OutputMode.CLARIFICATION_REQUEST.value,
        commitment_closed=True,
        interaction_open=True,
        action_is_hard_stop=False,
        normalized_outcome=FreshnessPermissionOutcome.RE_ATTEST,
    ),
    FreshnessPermissionOutcome.QUEUE_FOR_REVALIDATION: FreshnessPathwayDecision(
        pathway_id=ContinuationPathway.P_REFUSE_EXPLAIN_REDIRECT.value,
        output_mode=OutputMode.REFUSAL_WITH_EXPLANATION.value,
        commitment_closed=True,
        interaction_open=True,
        action_is_hard_stop=False,
        normalized_outcome=FreshnessPermissionOutcome.QUEUE_FOR_REVALIDATION,
    ),
    FreshnessPermissionOutcome.SUPERVISED_CONTINUE: FreshnessPathwayDecision(
        pathway_id=ContinuationPathway.P_HANDOFF_SUMMARY.value,
        output_mode=OutputMode.CONSTRAINED_RESPONSE.value,
        commitment_closed=True,
        interaction_open=True,
        action_is_hard_stop=False,
        normalized_outcome=FreshnessPermissionOutcome.SUPERVISED_CONTINUE,
    ),
    FreshnessPermissionOutcome.ESCALATE: FreshnessPathwayDecision(
        pathway_id=ContinuationPathway.P_STOP_ESCALATE.value,
        output_mode=OutputMode.TERMINAL_STOP.value,
        commitment_closed=True,
        interaction_open=False,
        action_is_hard_stop=True,
        normalized_outcome=FreshnessPermissionOutcome.ESCALATE,
    ),
    FreshnessPermissionOutcome.SUSPEND: FreshnessPathwayDecision(
        pathway_id=ContinuationPathway.P_STOP_ESCALATE.value,
        output_mode=OutputMode.TERMINAL_STOP.value,
        commitment_closed=True,
        interaction_open=True,
        action_is_hard_stop=True,
        normalized_outcome=FreshnessPermissionOutcome.SUSPEND,
    ),
    FreshnessPermissionOutcome.HARD_STOP: FreshnessPathwayDecision(
        pathway_id=ContinuationPathway.P_STOP_TERMINAL.value,
        output_mode=OutputMode.TERMINAL_STOP.value,
        commitment_closed=True,
        interaction_open=False,
        action_is_hard_stop=True,
        normalized_outcome=FreshnessPermissionOutcome.HARD_STOP,
    ),
}


def project_freshness_pathway(
    *,
    outcome: FreshnessPermissionOutcome,
    authority_state: AuthorityState,
) -> FreshnessPathwayDecision:
    """Map freshness permission outcomes to explicit lawful continuation pathways."""
    required = _REQUIRED_OUTCOME_AUTHORITY.get(outcome)
    if required is not None and _MIN_AUTHORITY_RANK[authority_state] < required:
        outcome = (
            FreshnessPermissionOutcome.SUSPEND
            if authority_state != AuthorityState.NONE
            else FreshnessPermissionOutcome.HARD_STOP
        )
    return _OUTCOME_TO_PATHWAY[outcome]


__all__ = ["FreshnessPathwayDecision", "project_freshness_pathway"]

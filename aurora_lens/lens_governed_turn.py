"""Shared governed-turn envelope for Lens sync/stream parity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
from aurora_lens.pef.span import Span
from aurora_lens.verify.flags import Flag

if TYPE_CHECKING:
    from aurora_lens.lens import LensResult


@dataclass
class GovernedTurnEnvelope:
    """Audit-critical governance state for a single Lens turn."""

    turn: int
    span: Span
    action: InterventionAction
    pre_llm_blocked: bool
    llm_called: bool
    flags: list[Flag]
    decision: GovernanceDecision | None = None
    response: str = ""
    original_response: str | None = None
    upstream_model_draft: str | None = None
    epistemic_normalisation_applied: bool = False
    self_refused: bool = False
    model: str = ""
    route_reason_code: str | None = None

    @property
    def governance_outcome(self) -> str:
        return self.action.name

    @property
    def pathway_id(self) -> str | None:
        if self.decision is None:
            return None
        return getattr(self.decision, "pathway_id", None)

    @property
    def commitment_closed(self) -> bool | None:
        if self.decision is None:
            return None
        return getattr(self.decision, "commitment_closed", None)

    @property
    def interaction_open(self) -> bool | None:
        if self.decision is None:
            return None
        return getattr(self.decision, "interaction_open", None)

    @property
    def rationale(self) -> str | None:
        if self.decision is None:
            return None
        return getattr(self.decision, "rationale", None)


@dataclass
class PostGenerationGovernanceOutcome:
    """Shared post-LLM governance result consumed by sync and stream adapters."""

    flags: list[Flag]
    decision: GovernanceDecision
    final_response: str
    original_response: str | None
    epistemic_normalisation_applied: bool
    self_refused: bool
    admit_path: bool
    ambiguous_snapshot_merged_tokens: list[str] | None


def envelope_from_lens_result(
    result: LensResult,
    *,
    pre_llm_blocked: bool,
    llm_called: bool,
    route_reason_code: str | None = None,
) -> GovernedTurnEnvelope:
    return GovernedTurnEnvelope(
        turn=result.turn,
        span=result.span,
        action=result.action,
        pre_llm_blocked=pre_llm_blocked,
        llm_called=llm_called,
        flags=list(result.flags),
        decision=result.decision,
        response=result.response,
        original_response=result.original_response,
        upstream_model_draft=result.upstream_model_draft,
        epistemic_normalisation_applied=result.epistemic_normalisation_applied,
        self_refused=result.self_refused,
        model=result.model,
        route_reason_code=route_reason_code,
    )


def envelope_from_post_generation(
    outcome: PostGenerationGovernanceOutcome,
    *,
    turn: int,
    span: Span,
    upstream_model: str,
    upstream_model_draft: str,
    route_reason_code: str | None = None,
) -> GovernedTurnEnvelope:
    return GovernedTurnEnvelope(
        turn=turn,
        span=span,
        action=outcome.decision.action,
        pre_llm_blocked=False,
        llm_called=True,
        flags=list(outcome.flags),
        decision=outcome.decision,
        response=outcome.final_response,
        original_response=outcome.original_response,
        upstream_model_draft=upstream_model_draft,
        epistemic_normalisation_applied=outcome.epistemic_normalisation_applied,
        self_refused=outcome.self_refused,
        model=upstream_model,
        route_reason_code=route_reason_code,
    )


def governance_parity_material(envelope: GovernedTurnEnvelope) -> dict[str, Any]:
    """Governance-critical fields compared in sync/stream parity tests."""
    decision = envelope.decision
    return {
        "governance_outcome": envelope.governance_outcome,
        "pre_llm_blocked": envelope.pre_llm_blocked,
        "llm_called": envelope.llm_called,
        "action": envelope.action.name,
        "pathway_id": envelope.pathway_id,
        "commitment_closed": envelope.commitment_closed,
        "interaction_open": envelope.interaction_open,
        "rationale": envelope.rationale,
        "flag_types": [f.flag_type.name for f in envelope.flags],
        "response": envelope.response,
        "self_refused": envelope.self_refused,
        "epistemic_normalisation_applied": envelope.epistemic_normalisation_applied,
        "policy": getattr(decision, "policy", None) if decision else None,
        "output_mode": getattr(decision, "output_mode", None) if decision else None,
    }

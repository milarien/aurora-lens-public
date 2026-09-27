"""Sovereign Provider Registry — route evaluation and PEF bridge declaration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from aurora_lens.pef.entity import Entity
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import Relationship
from aurora_lens.sovereign.failover_bridge import (
    SOVEREIGN_FAILOVER_SOURCE,
    build_failover_inference_contract,
    build_failover_premises,
)
from aurora_lens.sovereign.failover_policy import (
    alternate_supports_task,
    bridge_status_for_alternate_profile,
)
from aurora_lens.sovereign.provider_profile import ProviderProfile, profiles_by_id
from aurora_lens.sovereign.regression_suite import (
    assessment_to_summary,
    assess_provider_validation,
    resolve_route_primary_state,
)
from aurora_lens.sovereign.provider_state import (
    ACTIVE_PRIMARY_STATES,
    PROVIDER_UNAVAILABLE,
)
from aurora_lens.sovereign.validation_freshness import ValidationFreshnessPolicy

if TYPE_CHECKING:
    from aurora_lens.pef.state import PEFState
    from aurora_lens.request_metadata import ProviderRouteRequest


class ProviderRouteOutcome(str, Enum):
    USE_ACTIVE_PROVIDER = "use_active_provider"
    USE_CERTIFIED_ALTERNATE = "use_certified_alternate"
    LOCAL_REFUSAL = "local_refusal"
    FAIL_CLOSED = "fail_closed"


@dataclass(frozen=True)
class ProviderRouteEvaluation:
    """Result of sovereign registry evaluation for one provider route request."""

    outcome: ProviderRouteOutcome
    reason: str
    registry_evaluated: bool
    primary_provider_id: str
    alternate_provider_id: str | None
    bridge_status: str | None
    inference_contract: dict[str, Any] | None
    failover_attempted: bool
    validation_summary: dict[str, Any] | None = None

    @property
    def bridge_contract(self) -> dict[str, Any] | None:
        return self.inference_contract


def evaluation_blocks_adapter(evaluation: ProviderRouteEvaluation) -> bool:
    """Return True when adapter execution must not proceed for this route."""
    if not evaluation.registry_evaluated:
        return evaluation.failover_attempted
    if evaluation.outcome in {
        ProviderRouteOutcome.LOCAL_REFUSAL,
        ProviderRouteOutcome.FAIL_CLOSED,
    }:
        return True
    if evaluation.failover_attempted and evaluation.bridge_status != "valid":
        return True
    return False


class SovereignProviderRegistry:
    """In-memory registry of declared provider profiles."""

    def __init__(
        self,
        profiles: list[ProviderProfile] | None = None,
        *,
        validation_policy: ValidationFreshnessPolicy | None = None,
    ) -> None:
        self._profiles = profiles_by_id(profiles or [])
        self._validation_policy = validation_policy or ValidationFreshnessPolicy()

    def register(self, profile: ProviderProfile) -> None:
        self._profiles[profile.provider_id] = profile

    def get_profile(self, provider_id: str) -> ProviderProfile | None:
        return self._profiles.get(str(provider_id or "").strip())

    def _merge_validation_summary(
        self,
        base: dict[str, Any] | None,
        extra: dict[str, Any] | None,
        *,
        prefix: str,
    ) -> dict[str, Any] | None:
        if extra is None:
            return base
        merged = dict(base or {})
        for key, value in extra.items():
            merged[f"{prefix}_{key}"] = value
        return merged

    def evaluate_failover(self, request: ProviderRouteRequest) -> ProviderRouteEvaluation:
        primary_id = request.primary_provider_id
        declared_primary_state = str(request.primary_state or "").strip().lower()
        primary_profile = self.get_profile(primary_id)
        primary_state, primary_assessment = resolve_route_primary_state(
            primary_profile,
            declared_state=declared_primary_state,
            policy=self._validation_policy,
        )
        validation_summary = None
        if primary_assessment is not None:
            validation_summary = self._merge_validation_summary(
                None,
                assessment_to_summary(primary_assessment),
                prefix="primary",
            )
        alternate_id = request.alternate_provider_id
        task_domain = request.task_domain
        consequence_grade = request.consequence_grade
        data_class = request.data_class

        if primary_state in ACTIVE_PRIMARY_STATES:
            return ProviderRouteEvaluation(
                outcome=ProviderRouteOutcome.USE_ACTIVE_PROVIDER,
                reason="primary_provider_active",
                registry_evaluated=True,
                primary_provider_id=primary_id,
                alternate_provider_id=alternate_id,
                bridge_status=None,
                inference_contract=None,
                failover_attempted=False,
                validation_summary=validation_summary,
            )

        if primary_state != PROVIDER_UNAVAILABLE:
            return ProviderRouteEvaluation(
                outcome=ProviderRouteOutcome.FAIL_CLOSED,
                reason=f"primary_provider_state_blocked:{primary_state}",
                registry_evaluated=True,
                primary_provider_id=primary_id,
                alternate_provider_id=alternate_id,
                bridge_status=None,
                inference_contract=None,
                failover_attempted=True,
                validation_summary=validation_summary,
            )

        if not alternate_id:
            return ProviderRouteEvaluation(
                outcome=ProviderRouteOutcome.LOCAL_REFUSAL,
                reason="no_certified_alternate",
                registry_evaluated=True,
                primary_provider_id=primary_id,
                alternate_provider_id=None,
                bridge_status=None,
                inference_contract=None,
                failover_attempted=True,
                validation_summary=validation_summary,
            )

        alternate_profile = self.get_profile(alternate_id)
        profile_present = alternate_profile is not None
        premises = build_failover_premises(
            primary_provider_id=primary_id,
            alternate_provider_id=alternate_id,
            task_domain=task_domain,
            consequence_grade=consequence_grade,
            alternate_profile_present=profile_present,
        )

        if alternate_profile is None:
            bridge_status = "missing"
            return ProviderRouteEvaluation(
                outcome=ProviderRouteOutcome.LOCAL_REFUSAL,
                reason="alternate_capability_profile_missing",
                registry_evaluated=True,
                primary_provider_id=primary_id,
                alternate_provider_id=alternate_id,
                bridge_status=bridge_status,
                inference_contract=build_failover_inference_contract(
                    premises=premises,
                    bridge_status=bridge_status,
                ),
                failover_attempted=True,
                validation_summary=validation_summary,
            )

        aligned, refusal_reason = alternate_supports_task(
            alternate_profile,
            task_domain=task_domain,
            consequence_grade=consequence_grade,
            data_class=data_class,
        )
        if not aligned:
            if refusal_reason == "max_consequence_grade_too_low":
                bridge_status = "out_of_scope"
            else:
                bridge_status = "out_of_scope"
            return ProviderRouteEvaluation(
                outcome=ProviderRouteOutcome.LOCAL_REFUSAL,
                reason=refusal_reason or "capability_profile_insufficient",
                registry_evaluated=True,
                primary_provider_id=primary_id,
                alternate_provider_id=alternate_id,
                bridge_status=bridge_status,
                inference_contract=build_failover_inference_contract(
                    premises=premises,
                    bridge_status=bridge_status,
                ),
                failover_attempted=True,
                validation_summary=validation_summary,
            )

        alternate_assessment = assess_provider_validation(
            alternate_profile,
            declared_state=alternate_profile.status,
            policy=self._validation_policy,
        )
        bridge_status = bridge_status_for_alternate_profile(
            alternate_profile,
            consequence_grade=consequence_grade,
            provider_state=alternate_assessment.effective_state,
        )
        alternate_summary = assessment_to_summary(alternate_assessment)
        validation_summary = self._merge_validation_summary(
            validation_summary,
            alternate_summary,
            prefix="alternate",
        )
        if bridge_status == "valid":
            outcome = ProviderRouteOutcome.USE_CERTIFIED_ALTERNATE
            reason = "alternate_certified_for_task"
        else:
            outcome = ProviderRouteOutcome.LOCAL_REFUSAL
            reason = (
                "provider_validation_stale"
                if bridge_status == "expired"
                else "alternate_not_admissible"
            )

        return ProviderRouteEvaluation(
            outcome=outcome,
            reason=reason,
            registry_evaluated=True,
            primary_provider_id=primary_id,
            alternate_provider_id=alternate_id,
            bridge_status=bridge_status,
            inference_contract=build_failover_inference_contract(
                premises=premises,
                bridge_status=bridge_status,
            ),
            failover_attempted=True,
            validation_summary=validation_summary,
        )


def apply_route_evaluation_to_pef(
    pef: PEFState,
    evaluation: ProviderRouteEvaluation,
    *,
    turn: int,
) -> None:
    """Commit or replace sovereign-declared failover bridge metadata on PEF."""
    pef.relationships = [
        rel
        for rel in pef.relationships
        if not _is_sovereign_failover_relationship(rel)
    ]
    if evaluation.inference_contract is None:
        return

    entity = _ensure_entity(pef, "alternate_provider", turn=turn)
    pef.relationships.append(
        Relationship(
            subject_id=entity.id,
            relation="MAY",
            object_entity_id=None,
            object_literal="route_admissible",
            span=Span.PRESENT,
            source_turn=turn,
            evidence="sovereign provider registry failover evaluation",
            relation_metadata={
                "sovereign": {
                    "source": SOVEREIGN_FAILOVER_SOURCE,
                    "registry_evaluated": evaluation.registry_evaluated,
                    "outcome": evaluation.outcome.value,
                    "reason": evaluation.reason,
                    "primary_provider_id": evaluation.primary_provider_id,
                    "alternate_provider_id": evaluation.alternate_provider_id,
                },
                "inference": dict(evaluation.inference_contract),
            },
        )
    )


def reject_missing_provider_route() -> ProviderRouteEvaluation:
    """Fail closed when sovereign enforcement requires provider route metadata."""
    return ProviderRouteEvaluation(
        outcome=ProviderRouteOutcome.LOCAL_REFUSAL,
        reason="provider_route_metadata_required",
        registry_evaluated=False,
        primary_provider_id="",
        alternate_provider_id=None,
        bridge_status=None,
        inference_contract=None,
        failover_attempted=True,
    )


def reject_failover_without_registry(
    request: ProviderRouteRequest,
) -> ProviderRouteEvaluation:
    """Fail closed when failover is attempted but registry evaluation did not run."""
    return ProviderRouteEvaluation(
        outcome=ProviderRouteOutcome.LOCAL_REFUSAL,
        reason="failover_registry_evaluation_required",
        registry_evaluated=False,
        primary_provider_id=request.primary_provider_id,
        alternate_provider_id=request.alternate_provider_id,
        bridge_status=None,
        inference_contract=None,
        failover_attempted=True,
    )


def _is_sovereign_failover_relationship(rel: Relationship) -> bool:
    rm = rel.relation_metadata if isinstance(rel.relation_metadata, dict) else {}
    sovereign = rm.get("sovereign")
    if not isinstance(sovereign, dict):
        return False
    return str(sovereign.get("source") or "") == SOVEREIGN_FAILOVER_SOURCE


def _ensure_entity(pef: PEFState, name: str, *, turn: int) -> Entity:
    for entity in pef.entities.values():
        if entity.name == name:
            return entity
    entity = Entity.create(name, turn=turn)
    pef.entities[entity.id] = entity
    return entity

"""Provider-route audit envelope for Sovereign Provider Registry decisions."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from aurora_lens.govern.bridge import refresh_forensic_event_hash
from aurora_lens.sovereign.failover_bridge import FAILOVER_BRIDGE_REF
from aurora_lens.sovereign.provider_registry import (
    ProviderRouteEvaluation,
    ProviderRouteOutcome,
    evaluation_blocks_adapter,
)
from aurora_lens.sovereign.refusal_templates import refusal_for_evaluation

if TYPE_CHECKING:
    from aurora_lens.govern.decision import GovernanceDecision
    from aurora_lens.request_metadata import ProviderRouteRequest
    from aurora_lens.sovereign.provider_profile import ProviderProfile


def _profile_fingerprint(profile: ProviderProfile | None) -> dict[str, Any] | None:
    if profile is None:
        return None
    return {
        "provider_id": profile.provider_id,
        "status": profile.status,
        "max_consequence_grade": profile.max_consequence_grade,
        "permitted_domains": list(profile.permitted_domains),
        "allowed_data_classes": list(profile.allowed_data_classes),
        "last_validated_at": profile.last_validated_at,
        "validation_suite_id": profile.validation_suite_id,
        "regression_result": profile.regression_result,
        "report_hash": profile.report_hash,
        "observed_model_id": profile.observed_model_id,
    }


def compute_registry_state_hash(
    request: ProviderRouteRequest,
    evaluation: ProviderRouteEvaluation,
    *,
    alternate_profile: ProviderProfile | None = None,
) -> str:
    """Deterministic hash over declared route request, evaluation, and profile snapshot."""
    contract = evaluation.inference_contract if isinstance(evaluation.inference_contract, dict) else {}
    body = {
        "primary_provider_id": request.primary_provider_id,
        "primary_provider_state": request.primary_state,
        "alternate_provider_id": request.alternate_provider_id,
        "task_domain": request.task_domain,
        "consequence_grade": request.consequence_grade,
        "data_class": request.data_class,
        "outcome": evaluation.outcome.value,
        "reason": evaluation.reason,
        "registry_evaluated": evaluation.registry_evaluated,
        "bridge_status": evaluation.bridge_status,
        "bridge_ref": contract.get("bridge_ref"),
        "alternate_profile": _profile_fingerprint(alternate_profile),
    }
    digest = hashlib.sha256(
        json.dumps(body, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def _failover_result(
    evaluation: ProviderRouteEvaluation,
    *,
    adapter_called: bool,
) -> str:
    if not evaluation.failover_attempted:
        return "not_attempted"
    if adapter_called and not evaluation_blocks_adapter(evaluation):
        return "allowed"
    return "refused"


def _selected_provider_id(
    request: ProviderRouteRequest,
    evaluation: ProviderRouteEvaluation,
) -> str | None:
    if evaluation.outcome == ProviderRouteOutcome.USE_ACTIVE_PROVIDER:
        return request.primary_provider_id
    if evaluation.outcome == ProviderRouteOutcome.USE_CERTIFIED_ALTERNATE:
        return request.alternate_provider_id
    return None


def build_provider_route_audit_envelope(
    request: ProviderRouteRequest,
    evaluation: ProviderRouteEvaluation,
    *,
    adapter_called: bool,
    alternate_profile: ProviderProfile | None = None,
) -> dict[str, Any]:
    """Build replayable provider_route block for audit / forensic rows."""
    contract = evaluation.inference_contract if isinstance(evaluation.inference_contract, dict) else {}
    bridge_ref = str(contract.get("bridge_ref") or "").strip() or None
    if evaluation.failover_attempted and bridge_ref is None:
        bridge_ref = FAILOVER_BRIDGE_REF

    refusal = None
    if evaluation_blocks_adapter(evaluation):
        refusal = refusal_for_evaluation(
            evaluation,
            primary_provider_state=request.primary_state,
        )

    envelope: dict[str, Any] = {
        "requested_provider_id": request.primary_provider_id,
        "selected_provider_id": _selected_provider_id(request, evaluation),
        "selected_route": evaluation.outcome.value,
        "primary_provider_state": request.primary_state,
        "alternate_provider_id": request.alternate_provider_id,
        "bridge_status": evaluation.bridge_status,
        "bridge_ref": bridge_ref,
        "registry_policy_result": evaluation.outcome.value,
        "failover_attempted": evaluation.failover_attempted,
        "failover_result": _failover_result(evaluation, adapter_called=adapter_called),
        "refusal_reason": refusal.machine_reason if refusal else None,
        "machine_reason": refusal.machine_reason if refusal else None,
        "refusal_template_key": refusal.template_key if refusal else None,
        "refusal_message": refusal.message if refusal else None,
        "registry_state_hash": compute_registry_state_hash(
            request,
            evaluation,
            alternate_profile=alternate_profile,
        ),
        "adapter_called": adapter_called,
        "registry_evaluated": evaluation.registry_evaluated,
        "task_domain": request.task_domain,
        "consequence_grade": request.consequence_grade,
    }
    summary = evaluation.validation_summary if isinstance(evaluation.validation_summary, dict) else None
    if summary:
        envelope["validation_summary"] = summary
        envelope["effective_primary_state"] = summary.get("primary_effective_state")
        envelope["report_hash"] = summary.get("alternate_report_hash") or summary.get("primary_report_hash")
        envelope["failed_checks"] = summary.get("alternate_failed_checks") or summary.get("primary_failed_checks") or []
        envelope["provider_identity_changed"] = bool(
            summary.get("alternate_provider_identity_changed")
            or summary.get("primary_provider_identity_changed")
        )
    return envelope


def apply_provider_route_to_audit_entry(
    entry: dict[str, Any],
    decision: GovernanceDecision,
) -> None:
    """Attach ``provider_route`` to an audit row and nested ``forensic_event`` when present."""
    provider_route = decision.provider_route
    if not isinstance(provider_route, dict):
        return
    entry["provider_route"] = provider_route
    forensic_event = entry.get("forensic_event")
    if isinstance(forensic_event, dict):
        forensic_event["provider_route"] = provider_route
        refresh_forensic_event_hash(forensic_event)

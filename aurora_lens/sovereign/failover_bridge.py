"""Failover bridge contract constants and builders for sovereign route admissibility."""

from __future__ import annotations

from typing import Any

FAILOVER_BRIDGE_REF = "sovereign.failover.certification.v1"
FAILOVER_CONCLUSION = "alternate_provider.route_admissible"
FAILOVER_POLICY_REF = "sovereign.provider_registry.v1"
SOVEREIGN_FAILOVER_SOURCE = "failover_bridge"


def build_failover_inference_contract(
    *,
    premises: list[str],
    bridge_status: str,
    bridge_ref: str = FAILOVER_BRIDGE_REF,
    conclusion: str = FAILOVER_CONCLUSION,
    policy_ref: str = FAILOVER_POLICY_REF,
) -> dict[str, Any]:
    """Build relation_metadata.inference for a sovereign-declared failover bridge."""
    return {
        "premises": premises,
        "conclusion": conclusion,
        "bridge_ref": bridge_ref,
        "bridge_status": bridge_status,
        "bridge_mode": "requires_declared_bridge",
        "policy_ref": policy_ref,
    }


def build_failover_premises(
    *,
    primary_provider_id: str,
    alternate_provider_id: str | None,
    task_domain: str,
    consequence_grade: str,
    alternate_profile_present: bool,
) -> list[str]:
    premises = [
        f"primary_provider.unavailable",
        f"task.consequence_grade={consequence_grade}",
        f"task.domain={task_domain}",
    ]
    if alternate_provider_id:
        if alternate_profile_present:
            premises.append("alternate_provider.capability_profile")
        else:
            premises.append(f"alternate_provider.missing_profile={alternate_provider_id}")
    return premises

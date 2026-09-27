from typing import Any, Dict, List
from datetime import datetime, timezone

from .models import GovernorPolicy, SpeechAct, ProceduralAction, ForensicObligation

def enrich_forensic_envelope(
    base_envelope: Dict[str, Any],
    policy: GovernorPolicy,
    reason_code: str | None = None,
    parent_timestamp: str | None = None,
) -> Dict[str, Any]:
    """
    Appends continuation record to the canonical forensic event envelope.
    Ensures replayability and post-incident reconstruction.

    parent_timestamp: ISO-8601 UTC timestamp from the parent audit entry.
    When provided, governor_update_at is set to the same value to avoid temporal
    skew between the outer entry and the governor enrichment record.
    """
    continuation_data = {
        "governor_policy_id": f"{policy.domain.value}:{policy.authority_class.value}:{policy.lens_status.value}",
        "authority_class": policy.authority_class.value,
        "user_class": policy.user_class.value,
        "lens_status": policy.lens_status.value,
        "resolution_mode": policy.resolution_mode.value,
        "reason_code": reason_code,
        "domain": policy.domain.value,
        "commitment_closed": policy.commitment_closed,
        "interaction_open": policy.interaction_open,
        "pathway_id": policy.pathway_id.value,
        "allowed_speech_acts": [sa.value for sa in policy.allowed_speech_acts],
        "allowed_procedural_actions": [pa.value for pa in policy.allowed_procedural_actions],
        "forbidden_speech_acts": [sa.value for sa in policy.forbidden_speech_acts],
        "forensic_obligations": [fo.value for fo in policy.forensic_obligations],
        "escalation_target": policy.escalation_target,
        "exposure_level": policy.exposure_level.value,
        "governor_update_at": parent_timestamp or datetime.now(timezone.utc).isoformat(),
    }

    # Merge into the existing envelope or return a new one
    enriched = base_envelope.copy()
    enriched.update(continuation_data)
    return enriched

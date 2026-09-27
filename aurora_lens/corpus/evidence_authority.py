"""Evidence authority demotion for retrieved corpus evidence.

Maps admissibility posture to Lens intervention actions. Stale evidence remains
available as contextual signal (``signal_only``) rather than being treated as absent.

Dual-signature attestation is unrelated; this module governs evidence authority only.
"""

from __future__ import annotations

from enum import Enum

from aurora_lens.govern.decision import InterventionAction

_CANONICAL_EVIDENCE_AUTHORITY_STATES = frozenset({
    "authoritative",
    "signal_only",
    "requires_revalidation",
    "inadmissible",
})

_FRESHNESS_REVALIDATION_KINDS = frozenset({
    "missing_freshness",
    "unrevalidated_orientation",
})

_STALE_SIGNAL_KINDS = frozenset({
    "stale_evidence",
    "superseded_evidence",
})

_UNRESOLVED_KINDS = frozenset({
    "authority_unknown",
    "scope_unknown",
    "scope_mismatch",
})


class EvidenceAuthorityState(str, Enum):
    AUTHORITATIVE = "authoritative"
    SIGNAL_ONLY = "signal_only"
    REQUIRES_REVALIDATION = "requires_revalidation"
    INADMISSIBLE = "inadmissible"


EVIDENCE_AUTHORITY_TO_LENS_ACTION: dict[EvidenceAuthorityState, InterventionAction] = {
    EvidenceAuthorityState.AUTHORITATIVE: InterventionAction.PASS,
    EvidenceAuthorityState.SIGNAL_ONLY: InterventionAction.FORCE_REVISE,
    EvidenceAuthorityState.REQUIRES_REVALIDATION: InterventionAction.CONTAIN,
    EvidenceAuthorityState.INADMISSIBLE: InterventionAction.HARD_STOP,
}


def is_canonical_evidence_authority_state(value: str | None) -> bool:
    return str(value or "").strip().lower() in _CANONICAL_EVIDENCE_AUTHORITY_STATES


def lens_action_for_evidence_authority(
    authority_state: str | EvidenceAuthorityState | None,
) -> InterventionAction | None:
    if authority_state is None:
        return None
    if isinstance(authority_state, EvidenceAuthorityState):
        return EVIDENCE_AUTHORITY_TO_LENS_ACTION.get(authority_state)
    key = str(authority_state).strip().lower()
    try:
        state = EvidenceAuthorityState(key)
    except ValueError:
        return None
    return EVIDENCE_AUTHORITY_TO_LENS_ACTION.get(state)


def derive_evidence_authority(
    *,
    status: str,
    failure_kind: str | None,
    failure_meta: dict[str, str] | None = None,
    record_status: str | None = None,
) -> tuple[EvidenceAuthorityState, str | None, str | None, str | None]:
    """Return (authority_state, freshness_status, authority_status, demotion_reason)."""
    meta = failure_meta or {}
    demotion_reason = failure_kind
    freshness_status = str(meta.get("stale_source") or meta.get("freshness_status") or "").strip() or None
    authority_status = str(meta.get("record_status") or record_status or "").strip() or None

    if status == "admissible":
        return EvidenceAuthorityState.AUTHORITATIVE, freshness_status, authority_status, None

    kind = str(failure_kind or "").strip()
    if kind in _STALE_SIGNAL_KINDS:
        if not freshness_status and kind == "stale_evidence":
            freshness_status = str(meta.get("stale_source") or "stale").strip() or "stale"
        return EvidenceAuthorityState.SIGNAL_ONLY, freshness_status, authority_status, demotion_reason

    if kind in _FRESHNESS_REVALIDATION_KINDS or kind in _UNRESOLVED_KINDS or status == "unresolved":
        return (
            EvidenceAuthorityState.REQUIRES_REVALIDATION,
            freshness_status,
            authority_status,
            demotion_reason,
        )

    return EvidenceAuthorityState.INADMISSIBLE, freshness_status, authority_status, demotion_reason

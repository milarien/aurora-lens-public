"""Shared ``aurora`` response block for HTTP and streaming (user vs operator plane)."""

from __future__ import annotations

from typing import Any

from aurora_lens.context import authority_class_var, domain_var, trace_id_var
from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
from aurora_lens.pef.operator_surface import build_operator_pef_surface
from aurora_lens.pef.state import PEFState
from aurora_lens.verify.flags import Flag

# Operator metadata when upstream was blocked/refused — never echo raw model text in wire.
SUPPRESSED_BLOCKED_UPSTREAM_REASON = "suppressed_blocked_upstream_output"

_TERMINAL_ACTIONS = frozenset({
    InterventionAction.HARD_STOP,
    InterventionAction.CONTAIN,
    InterventionAction.FORCE_REVISE,
})


def _collect_failed_constraints(
    flags: list[Flag],
    decision: GovernanceDecision | None,
) -> list[str]:
    """Stable stop-reason list for user/operator planes (flag type + policy rule_id)."""
    out: list[str] = []
    for flag in flags:
        name = flag.flag_type.name
        if name not in out:
            out.append(name)
        rule_id = str(flag.rule_id or "").strip()
        if rule_id and rule_id not in out:
            out.append(rule_id)
    if decision is not None and decision.rule_result is not None:
        rr = decision.rule_result
        for candidate in (
            rr.flag_type.name if rr.flag_type is not None else None,
            rr.reason_code,
            rr.rule_id,
        ):
            text = str(candidate or "").strip()
            if text and text.lower() != "none" and text not in out:
                out.append(text)
    return out


_PUBLIC_GOVERNANCE_HINT: dict[InterventionAction, str | None] = {
    InterventionAction.PASS: None,
    InterventionAction.SOFT_CORRECT: "response_adjusted",
    InterventionAction.CONTAIN: "clarification_requested",
    InterventionAction.FORCE_REVISE: "response_revised",
    InterventionAction.HARD_STOP: "blocked",
}


def build_audit_receipt_operator_plane(
    *,
    trace_id: str | None,
    session_id: str | None,
    decision: GovernanceDecision | None,
) -> dict[str, Any] | None:
    """Safe per-turn receipt for operator-plane HTTP metadata only.

    Populated from ``GovernanceDecision.audit_receipt_snapshot`` after audit append.
    Never includes signing keys, raw env/config, or internal URLs.
    """
    if decision is None:
        return None
    snap = decision.audit_receipt_snapshot
    evidence_snap = decision.evidence_receipt_snapshot
    merged_snap: dict[str, Any] = {}
    if isinstance(evidence_snap, dict):
        merged_snap.update(evidence_snap)
    if isinstance(snap, dict):
        merged_snap.update(snap)
    if not merged_snap:
        return None
    out: dict[str, Any] = {}
    tid = merged_snap.get("trace_id") or trace_id
    if tid:
        out["trace_id"] = str(tid)
    if session_id:
        out["session_id"] = str(session_id)
    if merged_snap.get("entry_index") is not None:
        out["entry_index"] = merged_snap["entry_index"]
    if merged_snap.get("prev_hash") is not None:
        out["prev_hash"] = merged_snap["prev_hash"]
    if merged_snap.get("hash") is not None:
        out["hash"] = merged_snap["hash"]
    if merged_snap.get("state_hash") is not None:
        out["state_hash"] = merged_snap["state_hash"]
    hv = merged_snap.get("hmac_verified")
    if isinstance(hv, bool):
        out["hmac_verified"] = hv
    cv = merged_snap.get("chain_verified_to_entry")
    if isinstance(cv, bool):
        out["chain_verified_to_entry"] = cv
    for field in (
        "request_capture_status",
        "request_evidence_ref",
        "request_raw_sha256",
        "request_canonical_sha256",
        "evidence_capture_mode",
        "evidence_access_policy",
        "summary",
    ):
        value = merged_snap.get(field)
        if value is not None and str(value).strip() != "":
            out[field] = value
    if len(out) < 2:
        return None
    return out


def build_aurora_block(
    *,
    action: InterventionAction,
    flags: list[Flag],
    turn: int,
    decision: GovernanceDecision | None,
    pef: PEFState | None,
    include_operator_detail: bool,
    session_id: str | None = None,
    original_response: str | None = None,
    governed_response_body: str | None = None,
    stream_governed: bool | None = None,
    stream_truncated: bool = False,
    stream_dropped_chars: int = 0,
    operator_request_domain: str | None = None,
    pef_admission_result_wire: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the ``aurora`` object for chat completions (user plane + optional operator plane).

    User plane (always): governance, turn, optional session_id, governance_hint,
    audit_id, unverified (SOFT_CORRECT). Never includes ``forensic_event`` or raw hold blobs.

    Operator plane (include_operator_detail): forensic_event (non-admit), flags, rationale,
    policy, governance_note, operator_pef (present-state summary including
    ``binding_governance_state`` when causally live, not audit history).
    Optional ``pef_admission_result`` carries the serialized last ``update_pef`` envelope
    for this turn when the host supplies ``pef_admission_result_wire`` (operator payloads only).
    ``original_response`` (upstream model text) is included when provided for operator
    plane on ADMIT-like paths where raw echo is permitted. For HARD_STOP /
    FORCE_REVISE / CONTAIN, ``original_response`` is omitted; operator payloads must not
    echo the raw withheld candidate—use ``intercepted_upstream_redacted_reason`` with
    :data:`SUPPRESSED_BLOCKED_UPSTREAM_REASON` and audit logs for the buffer.
    """
    aurora: dict[str, Any] = {
        "governance": action.name,
        "turn": turn,
    }
    if session_id:
        aurora["session_id"] = session_id
        # Canonical name for the durable PEF continuity handle. Today this is
        # the same value as session_id (one store key backs both); the
        # separate field name lets clients depend on "the world handle"
        # without coupling to session/routing terminology that may evolve
        # independently (see docs/FRAME_LIFECYCLE_INVARIANT.md).
        aurora["pef_context_id"] = session_id

    _req_dom = str(
        operator_request_domain or domain_var.get(None) or ""
    ).strip().lower()
    if _req_dom:
        aurora["request_domain"] = _req_dom

    hint = _PUBLIC_GOVERNANCE_HINT.get(action)
    if hint:
        aurora["governance_hint"] = hint

    if action == InterventionAction.SOFT_CORRECT:
        aurora["unverified"] = True

    if decision and decision.cid:
        aurora["audit_id"] = decision.cid
    if decision and decision.rule_result is not None:
        aurora["rule_result"] = decision.rule_result.to_dict()
        dom_rr = str(decision.rule_result.domain or "").strip()
        if dom_rr:
            aurora["domain"] = dom_rr

    ac_fe: str | None = None
    if decision and isinstance(decision.forensic_event, dict):
        raw_ac = decision.forensic_event.get("authority_class")
        if raw_ac is not None and str(raw_ac).strip():
            ac_fe = str(raw_ac).strip()
    if ac_fe:
        aurora["authority_class"] = ac_fe
    elif authority_class_var.get(None):
        aurora["authority_class"] = str(authority_class_var.get(None))

    if action in _TERMINAL_ACTIONS:
        _failed_user = _collect_failed_constraints(flags, decision)
        if _failed_user:
            aurora["failed_constraints"] = _failed_user
            aurora["failed_constraint"] = _failed_user[0]

    # Operator plane only: full forensic envelope and diagnostics.
    if include_operator_detail:
        if pef is not None:
            aurora["operator_pef"] = build_operator_pef_surface(pef, decision)
        tid_eff = trace_id_var.get(None)
        receipt = build_audit_receipt_operator_plane(
            trace_id=tid_eff,
            session_id=session_id,
            decision=decision,
        )
        if receipt:
            aurora["audit_receipt"] = receipt
        if pef_admission_result_wire:
            aurora["pef_admission_result"] = pef_admission_result_wire
        if decision is not None and action != InterventionAction.PASS:
            aurora["flags"] = list(dict.fromkeys(f.flag_type.name for f in flags))
            _failed = _collect_failed_constraints(flags, decision)
            if _failed:
                aurora["failed_constraints"] = _failed
                aurora["failed_constraint"] = _failed[0]
            aurora["rationale"] = decision.rationale
            aurora["policy"] = decision.policy
            if decision.governance_note:
                aurora["governance_note"] = decision.governance_note
            if action not in (
                InterventionAction.PASS,
                InterventionAction.SOFT_CORRECT,
            ) and decision.forensic_event is not None:
                aurora["forensic_event"] = decision.forensic_event
        if original_response is not None:
            # Never echo blocked/revised raw model text on HTTP/SSE. PASS/SOFT_CORRECT
            # may surface ``original_response`` on the operator plane; HARD_STOP /
            # FORCE_REVISE / CONTAIN emit ``intercepted_upstream_redacted_reason`` only.
            suppresses_raw = action in (
                InterventionAction.HARD_STOP,
                InterventionAction.FORCE_REVISE,
                InterventionAction.CONTAIN,
            )
            if not suppresses_raw:
                aurora["original_response"] = original_response
            orig_s = str(original_response).strip()
            if suppresses_raw and orig_s:
                aurora["intercepted_upstream_redacted_reason"] = (
                    SUPPRESSED_BLOCKED_UPSTREAM_REASON
                )

    if stream_governed is not None:
        aurora["stream_governed"] = stream_governed
    if stream_truncated:
        aurora["stream_truncated"] = True
        aurora["stream_dropped_chars"] = stream_dropped_chars

    return aurora

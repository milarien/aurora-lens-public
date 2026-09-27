"""First-class audit records for user clarification / disambiguation resolution."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable

from aurora_lens.govern.audit_io import _recompute_state_hash

if TYPE_CHECKING:
    from aurora_lens.pef.state import PEFState

_SKIP_CLARIFICATION_AUDIT_CONSTRAINTS = frozenset(
    {"AGENCY_RISK_CONTEXT_UNRESOLVED"},
)

CLARIFICATION_RESOLUTION_OUTCOME = "USER_DISAMBIGUATION"
CLARIFICATION_RESOLUTION_EVENT_TYPE = "CLARIFICATION_RESOLVED"
RESOLUTION_SOURCE_USER_SELECTION = "user_selection"
REPLAY_LINE_HELD = "HELD: clarification pending"


def build_clarification_resolution_payload(
    *,
    pending: dict[str, Any],
    user_selection_text: str,
    selected_option: str,
    bound_entity_id: str | None,
    resolution_source: str = RESOLUTION_SOURCE_USER_SELECTION,
    pef_snapshot_before: dict[str, Any] | None,
    pef_snapshot_after: dict[str, Any] | None,
    prior_ask_cid: str | None,
    prior_ask_trace_id: str | None = None,
) -> dict[str, Any]:
    """Structured clarification-resolution provenance for audit append."""
    unresolved = [
        str(x).strip()
        for x in (pending.get("ambiguous_referents") or [])
        if str(x).strip()
    ]
    candidates = [
        str(x).strip()
        for x in (pending.get("candidate_entities") or [])
        if str(x).strip()
    ]
    unresolved_ids = [
        str(x).strip()
        for x in (pending.get("unresolved_entity_ids") or [])
        if str(x).strip()
    ]
    payload: dict[str, Any] = {
        "event_type": CLARIFICATION_RESOLUTION_EVENT_TYPE,
        "failed_constraint": str(pending.get("failed_constraint") or ""),
        "unresolved_referents": unresolved,
        "unresolved_entity_ids": unresolved_ids,
        "candidate_options": candidates,
        "selected_option": selected_option,
        "user_selection_text": user_selection_text,
        "resolution_source": resolution_source,
        "bound_entity_name": selected_option,
        "bound_entity_id": bound_entity_id,
        "prior_ask_cid": prior_ask_cid,
    }
    if prior_ask_trace_id:
        payload["prior_ask_trace_id"] = prior_ask_trace_id
    comp_adj = pending.get("comparand_adjective")
    comp_noun = pending.get("comparand_noun")
    if isinstance(comp_adj, str) and comp_adj.strip():
        payload["comparand_adjective"] = comp_adj.strip()
    if isinstance(comp_noun, str) and comp_noun.strip():
        payload["comparand_noun"] = comp_noun.strip()
    if pef_snapshot_before is not None:
        payload["state_hash_before"] = _recompute_state_hash(pef_snapshot_before)
    if pef_snapshot_after is not None:
        payload["state_hash_after"] = _recompute_state_hash(pef_snapshot_after)
    return payload


def build_clarification_resolution_audit_entry(
    *,
    schema_version: int,
    run_id: str,
    trace_id: str,
    timestamp: str,
    session_id: str | None,
    turn: int,
    tenant_label: str | None,
    mode: str,
    policy_version: str,
    clarification_resolution: dict[str, Any],
    request_hash: str | None = None,
    request_domain: str | None = None,
) -> dict[str, Any]:
    """Flat JSONL audit row for a user disambiguation event."""
    entry: dict[str, Any] = {
        "schema_version": schema_version,
        "run_id": run_id,
        "trace_id": trace_id,
        "timestamp": timestamp,
        "session_id": session_id,
        "turn": turn,
        "tenant_label": tenant_label,
        "mode": mode,
        "policy_version": policy_version,
        "outcome": CLARIFICATION_RESOLUTION_OUTCOME,
        "event_type": CLARIFICATION_RESOLUTION_EVENT_TYPE,
        "clarification_resolution": clarification_resolution,
        "failed_constraints": [],
        "log_slice_present": False,
    }
    fc = clarification_resolution.get("failed_constraint")
    if isinstance(fc, str) and fc.strip():
        entry["failed_constraints"] = [fc.strip()]
    if request_hash:
        entry["request_hash"] = request_hash
    if request_domain:
        entry["request_domain"] = request_domain
    return entry


def label_for_resolved_unresolved_entity_ids(pef: "PEFState", pending: dict[str, Any]) -> str | None:
    """Surface label when placeholder entities in pending are all resolved."""
    ids = list(pending.get("unresolved_entity_ids") or [])
    if not ids:
        return None
    tracked = [pef.entities.get(entity_id) for entity_id in ids]
    if any(entity is None for entity in tracked):
        return None
    if not all(bool(entity.resolved) for entity in tracked):
        return None
    names = [entity.name for entity in tracked if entity is not None]
    if not names:
        return None
    return names[0] if len(names) == 1 else ", ".join(names)


def label_for_typo_recovery_merge(pef: "PEFState", pending: dict[str, Any]) -> str | None:
    """Selected canonical entity after state-native typo-recovery merge."""
    tr = pending.get("typo_recovery")
    if not isinstance(tr, dict):
        return None
    cid = tr.get("canonical_entity_id")
    if isinstance(cid, str) and cid in pef.entities:
        return pef.entities[cid].name
    suggested = tr.get("suggested_canonical_name")
    if isinstance(suggested, str) and suggested.strip():
        return suggested.strip()
    return None


def resolve_selected_option_for_audit(
    *,
    pending: dict[str, Any],
    pef: "PEFState",
    user_selection_text: str,
    clar_ext: Any,
    explicit_selected: str | None,
    resolve_binding_entity: Callable[..., str | None],
    typo_merged: bool,
) -> str | None:
    """Best-effort selected option across candidate, pronoun, entity-id, and typo paths."""
    if explicit_selected and str(explicit_selected).strip():
        return str(explicit_selected).strip()
    resolved = resolve_binding_entity(user_selection_text, clar_ext, pending, pef)
    if resolved:
        return resolved
    entity_label = label_for_resolved_unresolved_entity_ids(pef, pending)
    if entity_label:
        return entity_label
    if typo_merged:
        return label_for_typo_recovery_merge(pef, pending)
    return None


def bound_entity_id_for_selected(pef: "PEFState", selected_option: str) -> str | None:
    """Resolve PEF entity id for audit (first name when compound label)."""
    primary = selected_option.split(",")[0].strip()
    if not primary:
        return None
    ent = pef.find_entity_by_name(primary)
    return ent.id if ent is not None else None


def should_emit_clarification_resolution_audit(
    *,
    pending: dict[str, Any],
    binding_found: bool,
    selected_option: str | None,
) -> bool:
    """True when this turn consumed pending clarification via user resolution."""
    if not binding_found:
        return False
    fc = str(pending.get("failed_constraint") or "")
    if fc in _SKIP_CLARIFICATION_AUDIT_CONSTRAINTS:
        return False
    return bool(selected_option and str(selected_option).strip())


def count_user_disambiguation_rows(entries: list[dict[str, Any]]) -> int:
    """Count USER_DISAMBIGUATION rows in a mixed JSONL / AFL export."""
    n = 0
    for raw in entries:
        row = _normalize_audit_entry(raw)
        if row.get("outcome") == CLARIFICATION_RESOLUTION_OUTCOME:
            n += 1
        elif raw.get("op") == CLARIFICATION_RESOLUTION_OUTCOME:
            n += 1
    return n


def _normalize_audit_entry(entry: dict[str, Any]) -> dict[str, Any]:
    if entry.get("kind") == "aurora.event":
        data = (entry.get("payload") or {}).get("data")
        if isinstance(data, dict):
            return data
    return entry


def format_clarification_replay_lines(entries: list[dict[str, Any]]) -> list[str]:
    """Human-readable replay lines for clarification acceptance tests."""
    lines: list[str] = []
    for raw in entries:
        row = _normalize_audit_entry(raw)
        outcome = row.get("outcome") or raw.get("op")
        cr = row.get("clarification_resolution")
        if not isinstance(cr, dict) and isinstance(raw.get("payload"), dict):
            pdata = (raw.get("payload") or {}).get("data")
            if isinstance(pdata, dict):
                cr = pdata.get("clarification_resolution") or pdata

        if outcome in ("CONTAIN", "CLARIFY", "ASK", "FORCE_REVISE"):
            rationale = str(row.get("rationale") or "")
            if (
                "Non-selection turn during P_ASK_DISAMBIGUATE hold" in rationale
                or "Clarification continuation from pending state" in rationale
            ):
                lines.append(REPLAY_LINE_HELD)
                continue
            constraints = row.get("failed_constraints") or []
            if isinstance(constraints, list):
                for c in constraints:
                    cs = str(c).strip()
                    if cs:
                        lines.append(f"ASK: {cs}")
                        break
            fe = row.get("forensic_event")
            if isinstance(fe, dict):
                for c in fe.get("failed_constraints") or []:
                    cs = str(c).strip()
                    if cs and not any(
                        ln == f"ASK: {cs}" for ln in lines
                    ):
                        lines.append(f"ASK: {cs}")
                        break

        if outcome == CLARIFICATION_RESOLUTION_OUTCOME and isinstance(cr, dict):
            opts = cr.get("candidate_options") or []
            if isinstance(opts, list) and opts:
                lines.append("OPTIONS: " + ", ".join(str(x) for x in opts))
            selected = cr.get("selected_option")
            if selected:
                lines.append(f"USER_DISAMBIGUATION: selected {selected}")

        if outcome == "PASS":
            response = row.get("governed_response") or row.get("final_response") or ""
            if str(response).strip():
                lines.append(f"PASS: {str(response).strip()}")

    return lines


def assert_replay_subsequence(expected: list[str], actual: list[str]) -> None:
    """Assert *expected* lines appear in order within *actual* (PASS lines: prefix match)."""
    idx = 0
    for want in expected:
        while idx < len(actual):
            line = actual[idx]
            idx += 1
            if want.startswith("PASS:"):
                if line.startswith("PASS:"):
                    break
            elif line == want:
                break
        else:
            raise AssertionError(
                f"replay missing expected line {want!r} after {actual[:idx]!r}"
            )


def read_jsonl_audit_entries(path: str) -> list[dict[str, Any]]:
    """Load all JSON objects from a flat or mixed JSONL audit file."""
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out

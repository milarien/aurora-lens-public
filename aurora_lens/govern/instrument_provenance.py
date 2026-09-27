"""Instrument provenance and co-attestation for governance audit rows.

Aurora-Lens supports co-attestation of governed decision records and the
governance instrument provenance payload. Decision attestation covers the
audit row content. Instrument attestation covers policy / instrument / ruleset
provenance used to authorize admissibility.

Where instrument provenance is incomplete, rows are surfaced as
``provenance_status=degraded`` and ``attestation_mode=decision_only``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

from aurora_lens.govern.decision import GovernanceDecision

PROVENANCE_STATUS_COMPLETE = "complete"
PROVENANCE_STATUS_DEGRADED = "degraded"

_SIGNATURE_STATUS_SIGNED = "signed"
_SIGNATURE_STATUS_UNSIGNED = "unsigned"
_SIGNATURE_STATUS_CO_ATTESTED = "co_attested"

ATTESTATION_MODE_NONE = "none"
ATTESTATION_MODE_DECISION_ONLY = "decision_only"
ATTESTATION_MODE_CO_ATTESTED = "co_attested"

_INSTRUMENT_SIGNATURE_STATUS_UNSIGNED = "unsigned"
_INSTRUMENT_SIGNATURE_STATUS_SIGNED = "signed"
_INSTRUMENT_SIGNATURE_STATUS_DEGRADED = "degraded"


def _present(value: object | None) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return bool(text) and text.lower() != "unknown"


def compute_provenance_status(
    *,
    instrument_id: str | None,
    instrument_version: str | None,
    ruleset_hash: str | None,
) -> str:
    """Degraded when instrument_id, instrument_version, and ruleset_hash are all missing."""
    if _present(instrument_id) or _present(instrument_version) or _present(ruleset_hash):
        return PROVENANCE_STATUS_COMPLETE
    return PROVENANCE_STATUS_DEGRADED


def resolve_instrument_provenance_sources(
    *,
    row: dict[str, Any],
    chain_of_custody: dict[str, Any] | None,
    policy_profile: str | None,
    outcome: str | None,
    governor_policy_id: str | None = None,
    policy_version: str | None = None,
) -> dict[str, str]:
    """Collect instrument provenance from an audit row and chain-of-custody bundle."""
    policy_block = (
        chain_of_custody.get("policy")
        if isinstance(chain_of_custody, dict)
        else None
    )
    coc_policy_version = (
        str(policy_block.get("policy_version")).strip()
        if isinstance(policy_block, dict) and policy_block.get("policy_version") is not None
        else ""
    )
    coc_ruleset_hash = (
        str(policy_block.get("governance_config_fingerprint")).strip()
        if isinstance(policy_block, dict) and policy_block.get("governance_config_fingerprint") is not None
        else ""
    )

    gid = str(governor_policy_id).strip() if governor_policy_id is not None else ""
    if not gid:
        gid = str(row.get("governor_policy_id") or "").strip()

    pv = str(policy_version).strip() if policy_version is not None else ""
    if not pv:
        pv = str(row.get("policy_version") or "").strip()
    if not pv or pv.lower() == "unknown":
        pv = ""
    if not pv:
        pv = coc_policy_version if _present(coc_policy_version) else ""

    row_ruleset = str(row.get("ruleset_hash") or "").strip()
    ruleset_hash = row_ruleset if _present(row_ruleset) else ""
    if not ruleset_hash:
        ruleset_hash = coc_ruleset_hash if _present(coc_ruleset_hash) else ""

    policy_ref = gid
    if not policy_ref:
        profile = (
            str(policy_profile).strip()
            if policy_profile is not None and str(policy_profile).strip()
            else str(row.get("policy_profile") or "").strip()
        )
        out = (
            str(outcome).strip()
            if outcome is not None and str(outcome).strip()
            else str(row.get("outcome") or "").strip() or "UNKNOWN"
        )
        policy_ref = f"{profile or 'unknown'}:{out}"

    instrument_id = gid
    instrument_version = pv

    return {
        "policy_ref": policy_ref,
        "instrument_id": instrument_id,
        "instrument_version": instrument_version,
        "ruleset_hash": ruleset_hash,
    }


def signature_status_from_audit_context(
    *,
    signing_key_configured: bool,
    attestation_signed: bool,
) -> str:
    if attestation_signed or signing_key_configured:
        return _SIGNATURE_STATUS_SIGNED
    return _SIGNATURE_STATUS_UNSIGNED


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8")


def _hmac_hex(payload: dict[str, Any], key: bytes) -> str:
    return hmac.new(key, _canonical_json_bytes(payload), hashlib.sha256).hexdigest()


def _meaningful_provenance_fields(payload: dict[str, str]) -> bool:
    for key in (
        "instrument_id",
        "instrument_version",
        "ruleset_hash",
        "governance_config_fingerprint",
        "policy_version",
    ):
        if _present(payload.get(key)):
            return True
    return False


def build_instrument_attestation_payload(
    *,
    row: dict[str, Any],
    chain_of_custody: dict[str, Any] | None,
) -> dict[str, str]:
    """Deterministic instrument provenance payload for co-attestation."""
    policy_block = (
        chain_of_custody.get("policy")
        if isinstance(chain_of_custody, dict)
        else None
    )
    payload: dict[str, str] = {}
    candidates: dict[str, object] = {
        "policy_ref": row.get("policy_ref"),
        "instrument_id": row.get("instrument_id"),
        "instrument_version": row.get("instrument_version"),
        "ruleset_hash": row.get("ruleset_hash"),
        "governance_config_fingerprint": (
            policy_block.get("governance_config_fingerprint")
            if isinstance(policy_block, dict)
            else None
        ),
        "policy_version": row.get("policy_version")
        or (
            policy_block.get("policy_version")
            if isinstance(policy_block, dict)
            else None
        ),
        "chain_of_custody_version": (
            chain_of_custody.get("chain_of_custody_version")
            if isinstance(chain_of_custody, dict)
            else None
        ),
    }
    for key, value in candidates.items():
        text = str(value).strip() if value is not None else ""
        if text and text.lower() != "unknown":
            payload[key] = text
    return payload


def build_decision_attestation_payload(row: dict[str, Any]) -> dict[str, Any]:
    """Deterministic decision payload for audit-row attestation."""
    excluded = {
        "hmac",
        "cid",
        "prev_cid",
        "decision_signature",
        "instrument_signature",
        "decision_signature_status",
        "instrument_signature_status",
        "instrument_attestation_hash",
        "attestation_mode",
        "signature_status",
        "decision_record_hash",
    }
    return {k: v for k, v in row.items() if k not in excluded}


def apply_attestation_fields_to_row(
    row: dict[str, Any],
    *,
    chain_of_custody: dict[str, Any] | None,
    signing_key: bytes | None,
) -> dict[str, Any]:
    """Apply dual/co-attestation fields to one governance audit row."""
    instrument_payload = build_instrument_attestation_payload(
        row=row,
        chain_of_custody=chain_of_custody,
    )
    if _meaningful_provenance_fields(instrument_payload):
        instrument_hash = "sha256:" + hashlib.sha256(
            _canonical_json_bytes(instrument_payload)
        ).hexdigest()
    else:
        instrument_payload = {}
        instrument_hash = None

    if signing_key is None:
        row["decision_signature"] = None
        row["decision_signature_status"] = _SIGNATURE_STATUS_UNSIGNED
        row["instrument_signature"] = None
        row["instrument_signature_status"] = _INSTRUMENT_SIGNATURE_STATUS_UNSIGNED
        row["instrument_attestation_hash"] = instrument_hash
        row["attestation_mode"] = ATTESTATION_MODE_NONE
        row["signature_status"] = _SIGNATURE_STATUS_UNSIGNED
        return {
            "attestation_mode": ATTESTATION_MODE_NONE,
            "decision_signature_status": _SIGNATURE_STATUS_UNSIGNED,
            "instrument_signature_status": _INSTRUMENT_SIGNATURE_STATUS_UNSIGNED,
            "instrument_attestation_hash": instrument_hash,
            "instrument_payload": instrument_payload,
        }

    decision_payload = build_decision_attestation_payload(row)
    row["decision_signature"] = _hmac_hex(decision_payload, signing_key)
    row["decision_signature_status"] = _SIGNATURE_STATUS_SIGNED
    if instrument_hash is None:
        row["instrument_signature"] = None
        row["instrument_signature_status"] = _INSTRUMENT_SIGNATURE_STATUS_DEGRADED
        row["instrument_attestation_hash"] = None
        row["attestation_mode"] = ATTESTATION_MODE_DECISION_ONLY
        row["signature_status"] = _SIGNATURE_STATUS_SIGNED
    else:
        row["instrument_signature"] = _hmac_hex(instrument_payload, signing_key)
        row["instrument_signature_status"] = _INSTRUMENT_SIGNATURE_STATUS_SIGNED
        row["instrument_attestation_hash"] = instrument_hash
        row["attestation_mode"] = ATTESTATION_MODE_CO_ATTESTED
        row["signature_status"] = _SIGNATURE_STATUS_CO_ATTESTED
    return {
        "attestation_mode": row.get("attestation_mode"),
        "decision_signature_status": row.get("decision_signature_status"),
        "instrument_signature_status": row.get("instrument_signature_status"),
        "instrument_attestation_hash": row.get("instrument_attestation_hash"),
        "instrument_payload": instrument_payload,
    }


def verify_attestation_fields(
    row: dict[str, Any],
    *,
    signing_keys: list[bytes] | tuple[bytes, ...] | None,
) -> tuple[bool, str | None]:
    """Verify co-attestation fields where present; legacy rows pass gracefully."""
    keys = [k for k in (signing_keys or []) if k]
    mode = str(row.get("attestation_mode") or "").strip().lower()
    has_decision_sig = bool(str(row.get("decision_signature") or "").strip())
    has_instrument_sig = bool(str(row.get("instrument_signature") or "").strip())
    if not mode and not has_decision_sig and not has_instrument_sig:
        return True, "legacy"
    if not keys and (has_decision_sig or has_instrument_sig):
        return False, "signed_without_keys"
    if mode in {"", ATTESTATION_MODE_NONE}:
        if has_decision_sig or has_instrument_sig:
            return False, "mode_none_with_signatures"
        return True, "none"

    if has_decision_sig:
        expected = str(row.get("decision_signature"))
        decision_payload = build_decision_attestation_payload(row)
        if not any(hmac.compare_digest(expected, _hmac_hex(decision_payload, k)) for k in keys):
            return False, "decision_signature_mismatch"
    elif mode in {ATTESTATION_MODE_DECISION_ONLY, ATTESTATION_MODE_CO_ATTESTED}:
        return False, "missing_decision_signature"

    if has_instrument_sig:
        expected_hash = str(row.get("instrument_attestation_hash") or "").strip()
        if not expected_hash:
            return False, "missing_instrument_attestation_hash"
        instrument_payload = build_instrument_attestation_payload(
            row=row,
            chain_of_custody=row.get("chain_of_custody")
            if isinstance(row.get("chain_of_custody"), dict)
            else None,
        )
        computed_hash = "sha256:" + hashlib.sha256(
            _canonical_json_bytes(instrument_payload)
        ).hexdigest()
        if not hmac.compare_digest(expected_hash, computed_hash):
            return False, "instrument_attestation_hash_mismatch"
        expected_sig = str(row.get("instrument_signature"))
        if not any(hmac.compare_digest(expected_sig, _hmac_hex(instrument_payload, k)) for k in keys):
            return False, "instrument_signature_mismatch"
    elif mode == ATTESTATION_MODE_CO_ATTESTED:
        return False, "missing_instrument_signature"

    if mode == ATTESTATION_MODE_CO_ATTESTED and not (has_decision_sig and has_instrument_sig):
        return False, "co_attested_incomplete"
    if mode == ATTESTATION_MODE_DECISION_ONLY and not has_decision_sig:
        return False, "decision_only_missing_decision_signature"
    return True, None


def apply_instrument_provenance_to_row(
    row: dict[str, Any],
    *,
    chain_of_custody: dict[str, Any] | None,
    policy_profile: str | None,
    outcome: str | None,
    governor_policy_id: str | None = None,
    policy_version: str | None = None,
    signing_key_configured: bool = False,
    attestation_signed: bool = False,
) -> dict[str, str]:
    """Write instrument provenance fields onto an audit row before append."""
    fields = resolve_instrument_provenance_sources(
        row=row,
        chain_of_custody=chain_of_custody,
        policy_profile=policy_profile,
        outcome=outcome,
        governor_policy_id=governor_policy_id,
        policy_version=policy_version,
    )
    for key, value in fields.items():
        if value:
            row[key] = value
    row["provenance_status"] = compute_provenance_status(
        instrument_id=fields.get("instrument_id"),
        instrument_version=fields.get("instrument_version"),
        ruleset_hash=fields.get("ruleset_hash"),
    )
    row["signature_status"] = signature_status_from_audit_context(
        signing_key_configured=signing_key_configured,
        attestation_signed=attestation_signed,
    )
    return fields


def sync_instrument_provenance_to_decision(
    decision: GovernanceDecision,
    fields: dict[str, str],
    *,
    signature_status: str | None = None,
    decision_record_hash: str | None = None,
) -> None:
    """Mirror row provenance onto the governance decision record."""
    decision.policy_ref = fields.get("policy_ref") or None
    decision.instrument_id = fields.get("instrument_id") or None
    decision.instrument_version = fields.get("instrument_version") or None
    decision.ruleset_hash = fields.get("ruleset_hash") or None
    decision.provenance_status = compute_provenance_status(
        instrument_id=decision.instrument_id,
        instrument_version=decision.instrument_version,
        ruleset_hash=decision.ruleset_hash,
    )
    if signature_status is not None:
        decision.signature_status = signature_status
    if decision_record_hash is not None:
        decision.decision_record_hash = decision_record_hash
    decision.decision_signature = row_or_none(fields, "decision_signature")
    decision.decision_signature_status = row_or_none(fields, "decision_signature_status")
    decision.instrument_signature = row_or_none(fields, "instrument_signature")
    decision.instrument_signature_status = row_or_none(fields, "instrument_signature_status")
    decision.instrument_attestation_hash = row_or_none(fields, "instrument_attestation_hash")
    decision.attestation_mode = row_or_none(fields, "attestation_mode")


def row_or_none(fields: dict[str, Any], key: str) -> str | None:
    value = fields.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def finalize_decision_record_hash(
    decision: GovernanceDecision,
    row: dict[str, Any],
    *,
    cid: str | None,
    ledger_hash: str | None = None,
) -> None:
    """Attach decision_record_hash after the canonical audit row hash exists."""
    record_hash: str | None = None
    if cid:
        record_hash = cid if str(cid).startswith("sha256:") else f"sha256:{cid}"
    elif ledger_hash:
        record_hash = (
            ledger_hash
            if str(ledger_hash).startswith("sha256:")
            else f"sha256:{ledger_hash}"
        )
    if record_hash is None:
        return
    decision.decision_record_hash = record_hash
    row["decision_record_hash"] = record_hash

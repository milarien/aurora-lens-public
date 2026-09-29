"""Tests for instrument provenance and co-attestation modes."""

from __future__ import annotations

import json
from pathlib import Path

from aurora_lens.govern.bridge import BuiltinBridge
from aurora_lens.govern.audit_io import verify_co_attestation_window
from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
from aurora_lens.govern.instrument_provenance import (
    ATTESTATION_MODE_CO_ATTESTED,
    ATTESTATION_MODE_DECISION_ONLY,
    ATTESTATION_MODE_NONE,
    PROVENANCE_STATUS_COMPLETE,
    PROVENANCE_STATUS_DEGRADED,
    apply_attestation_fields_to_row,
    apply_instrument_provenance_to_row,
    build_instrument_attestation_payload,
    compute_provenance_status,
    resolve_instrument_provenance_sources,
    verify_attestation_fields,
    sync_instrument_provenance_to_decision,
)
from aurora_lens.verify.flags import Flag, FlagType


def test_provenance_complete_when_instrument_fields_present():
    status = compute_provenance_status(
        instrument_id="strict:CONTAIN",
        instrument_version="1.0",
        ruleset_hash="sha256:abc",
    )
    assert status == PROVENANCE_STATUS_COMPLETE


def test_provenance_degraded_when_all_core_fields_missing():
    status = compute_provenance_status(
        instrument_id=None,
        instrument_version=None,
        ruleset_hash=None,
    )
    assert status == PROVENANCE_STATUS_DEGRADED


def test_governance_decision_sync_not_degraded_with_ruleset_sources():
    row: dict[str, object] = {"policy_profile": "strict", "outcome": "CONTAIN"}
    coc = {
        "policy": {
            "policy_version": "1.0",
            "governance_config_fingerprint": "sha256:deadbeef",
        }
    }
    fields = resolve_instrument_provenance_sources(
        row=row,
        chain_of_custody=coc,
        policy_profile="strict",
        outcome="CONTAIN",
        governor_policy_id="strict:CONTAIN",
    )
    decision = GovernanceDecision(
        action=InterventionAction.CONTAIN,
        flags=[],
        rationale="test",
    )
    sync_instrument_provenance_to_decision(
        decision,
        {
            **fields,
            "attestation_mode": ATTESTATION_MODE_NONE,
            "decision_signature_status": "unsigned",
            "instrument_signature_status": "unsigned",
        },
        signature_status="unsigned",
    )
    assert decision.provenance_status == PROVENANCE_STATUS_COMPLETE
    assert decision.instrument_id == "strict:CONTAIN"
    assert decision.instrument_version == "1.0"
    assert decision.ruleset_hash == "sha256:deadbeef"


def test_governance_decision_sync_degraded_without_sources():
    decision = GovernanceDecision(
        action=InterventionAction.PASS,
        flags=[],
        rationale="test",
    )
    sync_instrument_provenance_to_decision(
        decision,
        {
            "attestation_mode": ATTESTATION_MODE_NONE,
            "decision_signature_status": "unsigned",
            "instrument_signature_status": "unsigned",
        },
        signature_status="unsigned",
    )
    assert decision.provenance_status == PROVENANCE_STATUS_DEGRADED


def test_audit_row_includes_provenance_status(tmp_path: Path):
    audit_path = tmp_path / "audit.jsonl"
    bridge = BuiltinBridge(audit_path=str(audit_path))
    decision = GovernanceDecision(
        action=InterventionAction.HARD_STOP,
        flags=[
            Flag(
                flag_type=FlagType.ILLEGAL_INSTRUCTION,
                entity_name="x",
                claim="blocked",
                evidence="test",
                severity="error",
            )
        ],
        rationale="stop",
        policy="strict",
    )
    decision.original_response = "bad"
    bridge.log_decision(decision, turn=1, pef_context="ctx")

    entry = json.loads(audit_path.read_text(encoding="utf-8").strip())
    assert entry.get("provenance_status") in {PROVENANCE_STATUS_COMPLETE, PROVENANCE_STATUS_DEGRADED}
    assert entry["provenance_status"] == PROVENANCE_STATUS_COMPLETE
    assert entry.get("instrument_id")
    assert entry.get("instrument_version")
    assert entry.get("ruleset_hash", "").startswith("sha256:")
    assert entry.get("policy_ref")
    assert entry.get("signature_status") in {"signed", "unsigned", "co_attested"}
    assert decision.provenance_status == PROVENANCE_STATUS_COMPLETE


def test_apply_instrument_provenance_row_degraded_without_chain():
    row: dict[str, object] = {}
    apply_instrument_provenance_to_row(
        row,
        chain_of_custody=None,
        policy_profile=None,
        outcome=None,
    )
    assert row["provenance_status"] == PROVENANCE_STATUS_DEGRADED


def test_unsigned_mode_attestation_fields():
    row: dict[str, object] = {}
    apply_instrument_provenance_to_row(
        row,
        chain_of_custody=None,
        policy_profile=None,
        outcome=None,
        signing_key_configured=False,
    )
    att = apply_attestation_fields_to_row(
        row,
        chain_of_custody=None,
        signing_key=None,
    )
    assert att["attestation_mode"] == ATTESTATION_MODE_NONE
    assert row["decision_signature_status"] == "unsigned"
    assert row["instrument_signature_status"] == "unsigned"
    assert row["attestation_mode"] == "none"
    assert row["signature_status"] == "unsigned"


def test_decision_only_mode_when_instrument_provenance_missing():
    row: dict[str, object] = {}
    apply_instrument_provenance_to_row(
        row,
        chain_of_custody=None,
        policy_profile=None,
        outcome=None,
        signing_key_configured=True,
    )
    att = apply_attestation_fields_to_row(
        row,
        chain_of_custody=None,
        signing_key=b"k-test",
    )
    assert att["attestation_mode"] == ATTESTATION_MODE_DECISION_ONLY
    assert row["decision_signature_status"] == "signed"
    assert row["instrument_signature_status"] == "degraded"
    assert row["attestation_mode"] == "decision_only"
    assert row["signature_status"] == "signed"
    assert row["provenance_status"] == PROVENANCE_STATUS_DEGRADED


def test_co_attested_mode_with_ruleset_hash_present():
    row: dict[str, object] = {
        "ruleset_hash": "sha256:deadbeef",
        "policy_version": "1.0",
    }
    apply_instrument_provenance_to_row(
        row,
        chain_of_custody={"policy": {"governance_config_fingerprint": "sha256:deadbeef"}},
        policy_profile="strict",
        outcome="CONTAIN",
        signing_key_configured=True,
    )
    payload = build_instrument_attestation_payload(
        row=row,
        chain_of_custody={"policy": {"governance_config_fingerprint": "sha256:deadbeef"}},
    )
    assert payload
    att = apply_attestation_fields_to_row(
        row,
        chain_of_custody={"policy": {"governance_config_fingerprint": "sha256:deadbeef"}},
        signing_key=b"k-test",
    )
    assert att["attestation_mode"] == ATTESTATION_MODE_CO_ATTESTED
    assert row["decision_signature_status"] == "signed"
    assert row["instrument_signature_status"] == "signed"
    assert row["instrument_attestation_hash"]
    assert row["attestation_mode"] == "co_attested"
    assert row["signature_status"] == "co_attested"


def test_tamper_detection_for_instrument_and_decision_signatures():
    key = b"k-test"
    coc = {"policy": {"governance_config_fingerprint": "sha256:abc"}}
    row: dict[str, object] = {
        "ruleset_hash": "sha256:abc",
        "policy_version": "1.0",
        "instrument_id": "strict:CONTAIN",
        "chain_of_custody": coc,
    }
    apply_attestation_fields_to_row(
        row,
        chain_of_custody=coc,
        signing_key=key,
    )
    ok, reason = verify_attestation_fields(row, signing_keys=[key])
    assert ok is True
    assert reason is None

    tampered_instrument = dict(row)
    tampered_instrument["ruleset_hash"] = "sha256:tampered"
    ok, reason = verify_attestation_fields(tampered_instrument, signing_keys=[key])
    assert ok is False
    assert reason in {
        "instrument_attestation_hash_mismatch",
        "instrument_signature_mismatch",
        "decision_signature_mismatch",
    }

    tampered_decision = dict(row)
    tampered_decision["outcome"] = "HARD_STOP"
    ok, reason = verify_attestation_fields(tampered_decision, signing_keys=[key])
    assert ok is False
    assert reason == "decision_signature_mismatch"


def test_missing_decision_signature_fails_verification():
    key = b"k-test"
    row: dict[str, object] = {
        "ruleset_hash": "sha256:abc",
        "policy_version": "1.0",
        "instrument_id": "strict:CONTAIN",
        "chain_of_custody": {"policy": {"governance_config_fingerprint": "sha256:abc"}},
    }
    apply_attestation_fields_to_row(
        row,
        chain_of_custody=row["chain_of_custody"],
        signing_key=key,
    )
    row["decision_signature"] = None
    ok, reason = verify_attestation_fields(row, signing_keys=[key])
    assert ok is False
    assert reason == "missing_decision_signature"


def test_missing_instrument_signature_fails_co_attested_verification():
    key = b"k-test"
    coc = {"policy": {"governance_config_fingerprint": "sha256:abc"}}
    row: dict[str, object] = {
        "ruleset_hash": "sha256:abc",
        "policy_version": "1.0",
        "instrument_id": "strict:CONTAIN",
        "chain_of_custody": coc,
    }
    apply_attestation_fields_to_row(
        row,
        chain_of_custody=coc,
        signing_key=key,
    )
    assert row.get("attestation_mode") == "co_attested"
    row["instrument_signature"] = None
    ok, reason = verify_attestation_fields(row, signing_keys=[key])
    assert ok is False
    assert reason == "missing_instrument_signature"


def test_verify_co_attestation_window_reports_signature_mismatch(tmp_path: Path):
    audit_path = tmp_path / "audit.jsonl"
    key = b"k-test"
    coc = {"policy": {"governance_config_fingerprint": "sha256:abc"}}
    row: dict[str, object] = {
        "trace_id": "trace-1",
        "outcome": "CONTAIN",
        "ruleset_hash": "sha256:abc",
        "policy_version": "1.0",
        "instrument_id": "strict:CONTAIN",
        "chain_of_custody": coc,
    }
    apply_attestation_fields_to_row(
        row,
        chain_of_custody=coc,
        signing_key=key,
    )
    row["instrument_signature"] = "tampered-signature"
    audit_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    ok, detail = verify_co_attestation_window(audit_path, n=10, signing_keys=[key])
    assert ok is False
    assert detail["reason"] == "instrument_signature_mismatch"
    assert detail["first_failed_line_index"] == 0


def test_backward_compat_legacy_row_verifies_without_crash(tmp_path: Path):
    audit_path = tmp_path / "legacy.jsonl"
    legacy_row = {
        "trace_id": "legacy-1",
        "outcome": "PASS",
        "signature_status": "signed",
        "hmac": "abc123",
    }
    audit_path.write_text(json.dumps(legacy_row) + "\n", encoding="utf-8")
    ok, detail = verify_co_attestation_window(audit_path, n=10, signing_keys=[b"k-test"])
    assert ok is True
    assert detail["legacy_rows"] == 1

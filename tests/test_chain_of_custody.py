"""Chain-of-custody evidentiary packaging on audit rows."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import aurora_lens.govern.chain_of_custody as coc_module
from aurora_lens.govern.audit_io import (
    append_audit_entry,
    verify_chain,
    verify_chain_of_custody_rows,
    verify_chain_of_custody_window,
)
from aurora_lens.govern.bridge import BuiltinBridge
from aurora_lens.govern.scanner_gate_bridge import AuroraScannerGateBridge
from aurora_lens.context import chain_of_custody_runtime_provenance_var
from aurora_lens.govern.chain_of_custody import (
    build_chain_of_custody_bundle,
    governance_config_fingerprint,
    verify_chain_of_custody_entry,
)
from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
from aurora_lens.verify.flags import Flag, FlagType


def _minimal_decision() -> GovernanceDecision:
    return GovernanceDecision(
        action=InterventionAction.HARD_STOP,
        flags=[
            Flag(
                flag_type=FlagType.ILLEGAL_INSTRUCTION,
                entity_name="x",
                severity="error",
                claim="test",
                evidence="e",
            )
        ],
        rationale="r",
        policy="strict",
        attempt=0,
        original_response="bad",
        governed_response="no",
    )


def test_governance_config_fingerprint_stable():
    fp1 = governance_config_fingerprint({"a": 1, "b": 2})
    fp2 = governance_config_fingerprint({"b": 2, "a": 1})
    assert fp1 == fp2
    assert fp1.startswith("sha256:")


def test_build_chain_of_custody_complete_with_runtime_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AURORA_MODEL_ID", "gpt-test")
    monkeypatch.setenv("AURORA_PROVIDER", "test-provider")
    monkeypatch.setenv("AURORA_GIT_COMMIT", "a" * 40)
    monkeypatch.setenv("AURORA_GIT_DIRTY", "false")
    coc = build_chain_of_custody_bundle(
        policy_version="1.0",
        policy_source="test",
        governance_config={"k": "v"},
        application_version="9.9.9",
    )
    assert coc["evidence_audit_status"] == "complete"
    assert coc["evidence_degradation_reasons"] == []
    assert coc["code"]["application_version"] == "9.9.9"
    assert coc["policy"]["governance_config_fingerprint"] == governance_config_fingerprint({"k": "v"})
    assert coc["runtime"]["model_id"] == "gpt-test"
    assert coc["runtime"]["provider"] == "test-provider"
    assert "_governance_config_snapshot" in coc


def test_build_chain_of_custody_degraded_without_model_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("AURORA_MODEL_ID", raising=False)
    monkeypatch.delenv("AURORA_PROVIDER", raising=False)
    monkeypatch.setenv("AURORA_GIT_COMMIT", "b" * 40)
    monkeypatch.setenv("AURORA_GIT_DIRTY", "false")
    coc = build_chain_of_custody_bundle(
        policy_version="1.0",
        policy_source="test",
        governance_config={"k": "v"},
        application_version="1.0.0",
    )
    assert coc["evidence_audit_status"] == "degraded"
    assert "runtime_model_provenance_incomplete" in coc["evidence_degradation_reasons"]
    assert coc["runtime"]["model_id"] is None
    assert coc["runtime"]["provider"] is None


def test_build_chain_of_custody_prefers_request_context_over_env(monkeypatch: pytest.MonkeyPatch):
    """Per-request ContextVar wins over env; avoids stale global env when proxy set context."""
    monkeypatch.setenv("AURORA_MODEL_ID", "env-model-wrong")
    monkeypatch.setenv("AURORA_PROVIDER", "env-provider-wrong")
    monkeypatch.setenv("AURORA_GIT_COMMIT", "c" * 40)
    monkeypatch.setenv("AURORA_GIT_DIRTY", "false")
    tok = chain_of_custody_runtime_provenance_var.set(("openai", "gpt-4o-mini-req"))
    try:
        coc = build_chain_of_custody_bundle(
            policy_version="1.0",
            policy_source="test",
            governance_config={"k": "v"},
            application_version="1.0.0",
        )
    finally:
        chain_of_custody_runtime_provenance_var.reset(tok)
    assert coc["evidence_audit_status"] == "complete"
    assert coc["runtime"]["model_id"] == "gpt-4o-mini-req"
    assert coc["runtime"]["provider"] == "openai"


def test_build_chain_of_custody_request_context_does_not_merge_with_env(monkeypatch: pytest.MonkeyPatch):
    """Incomplete request pair is ignored; fall back to env only (no hybrid attribution)."""
    monkeypatch.setenv("AURORA_MODEL_ID", "from-env-model")
    monkeypatch.setenv("AURORA_PROVIDER", "from-env-provider")
    monkeypatch.setenv("AURORA_GIT_COMMIT", "d" * 40)
    monkeypatch.setenv("AURORA_GIT_DIRTY", "false")
    # Tuple present but empty strings → treated as absent request provenance
    tok = chain_of_custody_runtime_provenance_var.set(("", ""))
    try:
        coc = build_chain_of_custody_bundle(
            policy_version="1.0",
            policy_source="test",
            governance_config={"k": "v"},
            application_version="1.0.0",
        )
    finally:
        chain_of_custody_runtime_provenance_var.reset(tok)
    assert coc["evidence_audit_status"] == "complete"
    assert coc["runtime"]["model_id"] == "from-env-model"
    assert coc["runtime"]["provider"] == "from-env-provider"


def test_build_chain_of_custody_context_reset_between_requests(monkeypatch: pytest.MonkeyPatch):
    """Sequential request simulations do not reuse prior request attribution."""
    monkeypatch.setenv("AURORA_MODEL_ID", "env-m")
    monkeypatch.setenv("AURORA_PROVIDER", "env-p")
    monkeypatch.setenv("AURORA_GIT_COMMIT", "e" * 40)
    monkeypatch.setenv("AURORA_GIT_DIRTY", "false")
    t1 = chain_of_custody_runtime_provenance_var.set(("mistral", "mistral-large-latest"))
    try:
        c1 = build_chain_of_custody_bundle(
            policy_version="1.0",
            policy_source="test",
            governance_config={"k": "v"},
            application_version="1.0.0",
        )
    finally:
        chain_of_custody_runtime_provenance_var.reset(t1)
    assert c1["runtime"]["provider"] == "mistral"
    assert c1["runtime"]["model_id"] == "mistral-large-latest"

    t2 = chain_of_custody_runtime_provenance_var.set(("anthropic", "claude-3-5-sonnet-20241022"))
    try:
        c2 = build_chain_of_custody_bundle(
            policy_version="1.0",
            policy_source="test",
            governance_config={"k": "v"},
            application_version="1.0.0",
        )
    finally:
        chain_of_custody_runtime_provenance_var.reset(t2)
    assert c2["runtime"]["provider"] == "anthropic"
    assert c2["runtime"]["model_id"] == "claude-3-5-sonnet-20241022"


def test_build_chain_of_custody_git_commit_never_null(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AURORA_MODEL_ID", "m")
    monkeypatch.setenv("AURORA_PROVIDER", "p")
    monkeypatch.delenv("AURORA_GIT_COMMIT", raising=False)
    monkeypatch.delenv("AURORA_GIT_DIRTY", raising=False)
    monkeypatch.setattr(coc_module, "_resolve_git", lambda _repo_root: (None, None, "unavailable"))
    coc = build_chain_of_custody_bundle(
        policy_version="1.0",
        policy_source="test",
        governance_config={"k": "v"},
        application_version="1.0.0",
    )
    assert coc["code"]["git_commit"] == "unknown"
    assert "git_commit_unavailable" in coc["evidence_degradation_reasons"]


def test_verify_chain_of_custody_entry_rejects_complete_with_reasons():
    coc = build_chain_of_custody_bundle(
        policy_version="1.0",
        policy_source="t",
        governance_config={"x": 1},
        application_version="1.0.0",
    )
    coc["evidence_audit_status"] = "complete"
    coc["evidence_degradation_reasons"] = ["oops"]
    ok, reason = verify_chain_of_custody_entry({"chain_of_custody": coc})
    assert not ok
    assert reason == "complete_with_degradation_reasons"


def test_builtin_bridge_audit_row_has_chain_of_custody(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("AURORA_MODEL_ID", "m1")
    monkeypatch.setenv("AURORA_PROVIDER", "p1")
    path = tmp_path / "audit.jsonl"
    bridge = BuiltinBridge(audit_path=path, policy_version="2.0")
    d = _minimal_decision()
    bridge._log_decision(d, turn=0, pre_llm=False, pef_snapshot=None)
    line = path.read_text(encoding="utf-8").strip().splitlines()[-1]
    entry = json.loads(line)
    assert "chain_of_custody" in entry
    assert entry["chain_of_custody"]["policy"]["policy_version"] == "2.0"
    assert entry["forensic_event"]["chain_of_custody"]["policy"]["policy_version"] == "2.0"
    ok, reason = verify_chain_of_custody_entry(entry)
    assert ok and reason is None


def test_verify_chain_of_custody_rows_over_sample_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AURORA_MODEL_ID", "m")
    monkeypatch.setenv("AURORA_PROVIDER", "p")
    path = tmp_path / "a.jsonl"
    key = b"k" * 32
    entry = {
        "schema_version": 2,
        "trace_id": "t",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "outcome": "PASS",
    }
    coc = build_chain_of_custody_bundle(
        policy_version="1.0",
        policy_source="unit",
        governance_config={"bridge": "test"},
        application_version="0.0.1",
    )
    entry["chain_of_custody"] = coc
    append_audit_entry(path, entry, signing_key=key, prev_cid="genesis")
    ok, n, idx, reason = verify_chain_of_custody_rows(path, 10)
    assert ok and n == 1 and idx is None and reason is None


def test_verify_chain_of_custody_window_nested_afl_envelope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Window verifier normalizes AFL ``payload.data.chain_of_custody`` like flat rows."""
    monkeypatch.setenv("AURORA_MODEL_ID", "m")
    monkeypatch.setenv("AURORA_PROVIDER", "p")
    path = tmp_path / "afl.jsonl"
    coc = build_chain_of_custody_bundle(
        policy_version="1.0",
        policy_source="unit",
        governance_config={"bridge": "test"},
        application_version="0.0.1",
    )
    line = {
        "v": 1,
        "kind": "aurora.event",
        "payload": {"pv": 1, "data": {"action": "PASS", "chain_of_custody": coc}},
    }
    path.write_text(json.dumps(line, separators=(",", ":")) + "\n", encoding="utf-8")
    ok, detail = verify_chain_of_custody_window(path, 10)
    assert ok
    assert detail["rows_checked"] == 1
    assert detail.get("complete_count", 0) >= 1


def test_integration_scanner_bridge_chain_of_custody_with_jsonl_d2_and_afl_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """End-to-end: real bridge append with chain_of_custody must not break integrity verifiers.

    - **Flat JSONL** (`backend="jsonl"`): `verify_chain` (D2 cid chain) and
      `verify_chain_of_custody_rows` both apply to the same file.
    - **AFL ledger** (`backend="ledger"`): `chain_of_custody` lives under
      ``payload.data``; ``audit_io.verify_chain`` skips AFL lines (no top-level
      ``prev_cid``). We assert ``ForensicLedger.verify_detailed`` and
      :func:`verify_chain_of_custody_entry` on the nested block instead.
    """
    monkeypatch.setenv("AURORA_MODEL_ID", "integration-model")
    monkeypatch.setenv("AURORA_PROVIDER", "integration-provider")
    key = b"integration-hmac-key-32bytes!!!!"

    decision = _minimal_decision()

    # ── Flat JSONL: D2 chain + chain-of-custody rows on one file ─────────────
    path_jsonl = tmp_path / "audit-flat.jsonl"
    bridge_j = AuroraScannerGateBridge(
        audit_path=str(path_jsonl),
        backend="jsonl",
        secret_key=key,
        default_policy="strict",
    )
    bridge_j.log_decision(decision, turn=0, pre_llm=False, pef_snapshot=None)

    ch_ok, ch_n, ch_br, ch_r = verify_chain(path_jsonl, 50, signing_keys=[key])
    assert ch_ok, (ch_n, ch_br, ch_r)
    coc_rows_ok, coc_n, coc_i, coc_r = verify_chain_of_custody_rows(path_jsonl, 50)
    assert coc_rows_ok and coc_n >= 1, (coc_n, coc_i, coc_r)
    flat = json.loads(path_jsonl.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert "chain_of_custody" in flat

    # ── AFL ForensicLedger: hash chain + nested chain_of_custody entry ────────
    path_led = tmp_path / "audit-afl.jsonl"
    bridge_l = AuroraScannerGateBridge(
        audit_path=str(path_led),
        backend="ledger",
        secret_key=key,
        default_policy="strict",
    )
    assert bridge_l._ledger is not None
    bridge_l.log_decision(decision, turn=0, pre_llm=False, pef_snapshot=None)

    led_detail = bridge_l.verify_ledger_detailed(signing_keys=[key])
    assert led_detail.ok, (
        led_detail.chain_ok,
        led_detail.first_chain_reason,
        led_detail.hmac_ok,
        led_detail.first_hmac_reason,
    )

    # Same AFL file: D2 ``verify_chain`` skips envelopes (no top-level ``prev_cid``);
    # ``verify_chain_of_custody_rows`` only matches top-level ``chain_of_custody``.
    ch_afl_ok, _, _, ch_afl_r = verify_chain(path_led, 50, signing_keys=[key])
    assert ch_afl_ok, ch_afl_r
    coc_afl_ok, coc_afl_n, _, coc_afl_r = verify_chain_of_custody_rows(path_led, 50)
    assert coc_afl_ok, (coc_afl_n, coc_afl_r)

    line = json.loads(path_led.read_text(encoding="utf-8").strip().splitlines()[0])
    inner = line["payload"]["data"]
    assert "chain_of_custody" in inner
    ok, reason = verify_chain_of_custody_entry({"chain_of_custody": inner["chain_of_custody"]})
    assert ok and reason is None

    ok_w, det_w = verify_chain_of_custody_window(path_led, 50)
    assert ok_w and det_w["rows_checked"] >= 1


def test_verify_require_complete_flags_degraded_audit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("AURORA_MODEL_ID", raising=False)
    monkeypatch.delenv("AURORA_PROVIDER", raising=False)
    coc = build_chain_of_custody_bundle(
        policy_version="1.0",
        policy_source="t",
        governance_config={"a": 1},
        application_version="1.0.0",
    )
    assert coc["evidence_audit_status"] == "degraded"
    ok, reason = verify_chain_of_custody_entry(
        {"chain_of_custody": coc},
        require_complete_evidence=True,
    )
    assert not ok
    assert reason == "evidence_degraded"

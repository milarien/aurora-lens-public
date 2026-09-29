"""Write-time forensic envelope guard (shared across bridges)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from aurora_lens.govern.bridge import BuiltinBridge, build_forensic_event
from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
from aurora_lens.govern.forensic_append_guard import (
    ForensicEnvelopeValidationError,
    enforce_forensic_event_for_append,
    validate_forensic_event_for_append,
)
from aurora_lens.govern.scanner_gate_bridge import AuroraScannerGateBridge
from aurora_lens.pef.state import PEFState
from aurora_lens.verify.flags import Flag, FlagType


def _flag(ft: FlagType) -> Flag:
    return Flag(flag_type=ft, entity_name="e", severity="error", claim="c", evidence="ev")


def test_enforce_accepts_round_trip_build_forensic_event(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AURORA_MODEL_ID", "m")
    monkeypatch.setenv("AURORA_PROVIDER", "p")
    d = GovernanceDecision(
        action=InterventionAction.HARD_STOP,
        flags=[_flag(FlagType.CONTRADICTED_FACT)],
        rationale="r",
        policy="strict",
        pathway_id="P_STOP_TERMINAL",
        output_mode="terminal_stop",
        commitment_closed=True,
        interaction_open=False,
        forensic_obligations=["emit_forensic_envelope"],
        resolution_mode="exact",
    )
    d.original_response = "x"
    d.governed_response = "y"
    fe = build_forensic_event(
        d,
        pre_llm=False,
        pef_snapshot={"session_id": "s"},
        trace_id="t",
        timestamp="2026-01-01T00:00:00+00:00",
        audit_id="aid",
    )
    enforce_forensic_event_for_append(fe)


def test_enforce_raises_on_tampered_event_hash():
    d = GovernanceDecision(
        action=InterventionAction.HARD_STOP,
        flags=[_flag(FlagType.CONTRADICTED_FACT)],
        rationale="r",
        policy="strict",
        pathway_id="P_STOP_TERMINAL",
        output_mode="terminal_stop",
        commitment_closed=True,
        interaction_open=False,
        forensic_obligations=["emit_forensic_envelope"],
        resolution_mode="exact",
    )
    d.original_response = "x"
    fe = build_forensic_event(
        d,
        pre_llm=False,
        pef_snapshot=None,
        trace_id="t",
        timestamp="2026-01-01T00:00:00+00:00",
        audit_id="aid",
    )
    fe["event_hash"] = "sha256:" + "0" * 64
    with pytest.raises(ForensicEnvelopeValidationError):
        enforce_forensic_event_for_append(fe)


def test_builtin_bridge_log_decision_raises_on_invalid_forensic_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Tamper ``event_hash`` and disable refresh so append guard rejects before JSONL write."""
    monkeypatch.setenv("AURORA_MODEL_ID", "m")
    monkeypatch.setenv("AURORA_PROVIDER", "p")
    audit = tmp_path / "a.jsonl"

    def _bad_build(*args, **kwargs):
        fe = build_forensic_event(*args, **kwargs)
        fe["event_hash"] = "sha256:" + "a" * 64
        return fe

    bridge = BuiltinBridge(audit_path=audit, policy_version="1.0")
    pef = PEFState()
    decision = GovernanceDecision(
        action=InterventionAction.HARD_STOP,
        flags=[_flag(FlagType.CONTRADICTED_FACT)],
        rationale="r",
        policy="strict",
    )
    decision.original_response = "bad"
    with patch("aurora_lens.govern.bridge.refresh_forensic_event_hash", lambda e: None):
        with patch("aurora_lens.govern.bridge.build_forensic_event", side_effect=_bad_build):
            with pytest.raises(ForensicEnvelopeValidationError):
                bridge.log_decision(decision, turn=0, pef_snapshot=pef.to_dict(), pre_llm=False)
    assert not audit.exists() or audit.read_text().strip() == ""


def test_scanner_gate_ledger_log_decision_raises_on_invalid_forensic_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("AURORA_MODEL_ID", "m")
    monkeypatch.setenv("AURORA_PROVIDER", "p")
    audit = tmp_path / "led.jsonl"
    key = b"k" * 32

    def _bad_build(*args, **kwargs):
        fe = build_forensic_event(*args, **kwargs)
        fe["event_hash"] = "sha256:" + "b" * 64
        return fe

    bridge = AuroraScannerGateBridge(
        audit_path=str(audit),
        backend="ledger",
        secret_key=key,
        default_policy="strict",
    )
    assert bridge._ledger is not None
    pef = PEFState()
    decision = GovernanceDecision(
        action=InterventionAction.HARD_STOP,
        flags=[_flag(FlagType.CONTRADICTED_FACT)],
        rationale="r",
        policy="strict",
    )
    decision.original_response = "bad"
    with patch("aurora_lens.govern.scanner_gate_bridge.refresh_forensic_event_hash", lambda e: None):
        with patch("aurora_lens.govern.scanner_gate_bridge.build_forensic_event", side_effect=_bad_build):
            with pytest.raises(ForensicEnvelopeValidationError):
                bridge.log_decision(decision, turn=0, pef_snapshot=pef.to_dict(), pre_llm=False)
    assert not audit.exists() or audit.read_text(encoding="utf-8").strip() == ""


def test_validate_returns_errors_list_not_raise():
    d = GovernanceDecision(
        action=InterventionAction.HARD_STOP,
        flags=[_flag(FlagType.CONTRADICTED_FACT)],
        rationale="r",
        policy="strict",
        pathway_id="P_STOP_TERMINAL",
        output_mode="terminal_stop",
        commitment_closed=True,
        interaction_open=False,
        forensic_obligations=["emit_forensic_envelope"],
        resolution_mode="exact",
    )
    d.original_response = "x"
    fe = build_forensic_event(
        d,
        pre_llm=False,
        pef_snapshot=None,
        trace_id="t",
        timestamp="2026-01-01T00:00:00+00:00",
        audit_id="aid",
    )
    fe.pop("pathway_id", None)
    errs = validate_forensic_event_for_append(fe)
    assert errs

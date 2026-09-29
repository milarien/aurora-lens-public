"""PEF linkage verify across interleaved proxy sessions (``run_aurora_lens.py`` semantics).

The proxy appends all chat sessions to one audit file; concurrent requests produce
adjacent lines that are not one PEF chain. ``verify_pef_linkage_detailed`` must skip
those pairs (see :func:`aurora_lens.govern.audit_io.verify_pef_linkage_detailed`).
"""

from __future__ import annotations

import json
from pathlib import Path

from aurora_lens.context import session_id_var
from aurora_lens.govern.audit_io import verify_pef_linkage_detailed
from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
from aurora_lens.govern.scanner_gate_bridge import AuroraScannerGateBridge
from aurora_lens.pef.state import PEFState
from aurora_lens.proxy.config import ProxyConfig


def _afl_wrap(data: dict) -> dict:
    return {"kind": "aurora.event", "payload": {"data": data}}


def _refusal_hold_pef_snapshot(session_id: str) -> dict:
    """End-of-turn PEF after FORCE_REVISE / refusal (replay expects held_refusal for same session)."""
    return {
        "entities": {},
        "relationships": [],
        "turn_bindings": {},
        "current_turn": 3,
        "active_span": "present",
        "session_id": session_id,
        "name_index": {},
        "pending_clarification": None,
        "epistemic_hold": {
            "schema_version": 1,
            "mode": "refusal",
            "since_turn": 3,
            "pathway_id": "P_REFUSE_EXPLAIN_REDIRECT",
            "interaction_open": True,
            "commitment_closed": True,
            "last_audit_id": "",
        },
        "discourse_referent_bindings": {},
        "retrieval_unresolved": [],
    }


def _stopped_pef_snapshot() -> dict:
    return {
        "entities": {},
        "relationships": [],
        "turn_bindings": {},
        "current_turn": 1,
        "active_span": "present",
        "session_id": "",
        "name_index": {},
        "pending_clarification": None,
        "epistemic_hold": {
            "schema_version": 1,
            "mode": "stop",
            "since_turn": 1,
            "pathway_id": "P_STOP_TERMINAL",
            "interaction_open": False,
            "commitment_closed": True,
            "last_audit_id": "x",
        },
        "discourse_referent_bindings": {},
        "retrieval_unresolved": [],
    }


def test_verify_pef_linkage_skips_adjacent_different_session_ids(tmp_path: Path) -> None:
    """Interleaved sessions: would be a bogus mismatch if treated as one chain."""
    p = tmp_path / "audit.jsonl"
    row_a = {
        "schema_version": 2,
        "session_id": "chat-session-alpha",
        "outcome": "HARD_STOP",
        "pef_snapshot": _stopped_pef_snapshot(),
        "pef_turn_classification": "stopped",
        "pef_hold_transition": "stop",
    }
    row_b = {
        "schema_version": 2,
        "session_id": "chat-session-beta",
        "outcome": "PASS",
        "pef_turn_classification": "fresh",
        "pef_hold_transition": "none",
    }
    p.write_text(json.dumps(row_a) + "\n" + json.dumps(row_b) + "\n", encoding="utf-8")
    d = verify_pef_linkage_detailed(p, n=50)
    assert d["ok"] is True
    assert int(d["skipped_cross_session_pairs"]) >= 1
    assert int(d["pairs_checked"]) == 0


def test_verify_pef_linkage_skips_adjacent_different_proxy_run_ids_same_session(tmp_path: Path) -> None:
    """Restart/deploy boundary: same ``session_id`` but different ``proxy_run_id`` is not one chain."""
    p = tmp_path / "audit.jsonl"
    sid = "chat-session-same"
    row_a = {
        "schema_version": 2,
        "session_id": sid,
        "proxy_run_id": "proxy-epoch-aaa",
        "outcome": "HARD_STOP",
        "pef_snapshot": _stopped_pef_snapshot(),
        "pef_turn_classification": "stopped",
        "pef_hold_transition": "stop",
    }
    row_b = {
        "schema_version": 2,
        "session_id": sid,
        "proxy_run_id": "proxy-epoch-bbb",
        "outcome": "PASS",
        "pef_turn_classification": "fresh",
        "pef_hold_transition": "none",
    }
    p.write_text(json.dumps(row_a) + "\n" + json.dumps(row_b) + "\n", encoding="utf-8")
    d = verify_pef_linkage_detailed(p, n=50)
    assert d["ok"] is True
    assert int(d["skipped_cross_proxy_run_pairs"]) >= 1
    assert int(d["pairs_checked"]) == 0


def test_verify_pef_linkage_afl_sess_a_refusal_then_sess_b_turn1_pass_skips(tmp_path: Path) -> None:
    """Ledger envelope: session A refusal-hold vs session B turn-1 PASS must skip, not mismatch."""
    p = tmp_path / "audit.jsonl"
    sid_a = "session-a-refusal-tail"
    sid_b = "session-b-turn-one-pass"
    row_prev = _afl_wrap(
        {
            "event_type": "governance_decision",
            "turn": 3,
            "escalation_level": 2,
            "policy": "strict",
            "action": "FORCE_REVISE",
            "flags": [],
            "rationale": "FORCE_REVISE: synthetic",
            "attempt": 0,
            "log_slice_present": False,
            "session_id": sid_a,
            "pef_turn_classification": "fresh",
            "pef_hold_transition": "refusal",
            "pef_snapshot": _refusal_hold_pef_snapshot(sid_a),
        }
    )
    row_curr = _afl_wrap(
        {
            "event_type": "governance_decision",
            "turn": 1,
            "escalation_level": 0,
            "policy": "strict",
            "action": "PASS",
            "flags": [],
            "rationale": "No verification flags",
            "attempt": 0,
            "log_slice_present": False,
            "session_id": sid_b,
            "pef_turn_classification": "fresh",
            "pef_hold_transition": "none",
        }
    )
    p.write_text(json.dumps(row_prev) + "\n" + json.dumps(row_curr) + "\n", encoding="utf-8")
    d = verify_pef_linkage_detailed(p, n=50)
    assert d["ok"] is True
    assert int(d["skipped_cross_session_pairs"]) == 1
    assert int(d["pairs_checked"]) == 0


def test_verify_pef_linkage_afl_same_boundary_without_curr_session_id_mismatches(tmp_path: Path) -> None:
    """If the B row omits linkage-visible session_id, the verifier cannot skip (live bug shape)."""
    p = tmp_path / "audit.jsonl"
    sid_a = "session-a-only"
    row_prev = _afl_wrap(
        {
            "event_type": "governance_decision",
            "turn": 3,
            "escalation_level": 2,
            "policy": "strict",
            "action": "FORCE_REVISE",
            "flags": [],
            "rationale": "FORCE_REVISE: synthetic",
            "attempt": 0,
            "log_slice_present": False,
            "session_id": sid_a,
            "pef_turn_classification": "fresh",
            "pef_hold_transition": "refusal",
            "pef_snapshot": _refusal_hold_pef_snapshot(sid_a),
        }
    )
    row_curr = _afl_wrap(
        {
            "event_type": "governance_decision",
            "turn": 1,
            "escalation_level": 0,
            "policy": "strict",
            "action": "PASS",
            "flags": [],
            "rationale": "No verification flags",
            "attempt": 0,
            "log_slice_present": False,
            "pef_turn_classification": "fresh",
            "pef_hold_transition": "none",
        }
    )
    p.write_text(json.dumps(row_prev) + "\n" + json.dumps(row_curr) + "\n", encoding="utf-8")
    d = verify_pef_linkage_detailed(p, n=50)
    assert d["ok"] is False
    assert d["reason"] == "turn_classification_mismatch"
    assert int(d["skipped_cross_session_pairs"]) == 0


def test_lens_reconciles_pef_session_id_when_hydrated_pef_had_empty_sid() -> None:
    """Proxy store key must populate ``PEFState.session_id`` so PASS ``pef_snapshot`` carries it."""
    from aurora_lens.adapters.base import AdapterResponse, LLMAdapter
    from aurora_lens.config import LensConfig
    from aurora_lens.lens import Lens
    from aurora_lens.pef.state import PEFState

    class _StubAdapter(LLMAdapter):
        async def generate(self, messages, **kwargs):
            return AdapterResponse(text="ok", model="stub", usage=None)

        async def generate_stream(self, messages):
            raise NotImplementedError

    stale = PEFState(session_id="")
    lens = Lens(
        LensConfig(adapter=_StubAdapter()),
        initial_pef=stale,
        session_id="session-09044b4e44f8",
    )
    assert lens.pef.session_id == "session-09044b4e44f8"
    assert lens.pef.to_dict()["session_id"] == "session-09044b4e44f8"


def test_ledger_pass_payload_session_id_fallback_from_pef_snapshot(tmp_path: Path) -> None:
    """When ContextVar session id is empty, AFL payload.session_id is taken from pef_snapshot."""
    ledger_path = tmp_path / "ledger.jsonl"
    bridge = AuroraScannerGateBridge(
        audit_path=ledger_path,
        backend="ledger",
        secret_key=None,
    )
    dec = GovernanceDecision(
        action=InterventionAction.PASS,
        flags=[],
        rationale="No verification flags",
        policy="strict",
    )
    snap = PEFState(session_id="from-pef-snapshot-only").to_dict()
    token = session_id_var.set("")
    try:
        bridge.log_decision(
            dec,
            turn=1,
            pef_snapshot=snap,
            pef_turn_classification="fresh",
            pef_hold_transition="none",
        )
    finally:
        session_id_var.reset(token)
    raw = ledger_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(raw) == 1
    data = (json.loads(raw[0]).get("payload") or {}).get("data") or {}
    assert data.get("session_id") == "from-pef-snapshot-only"


def test_verify_pef_linkage_same_session_still_enforces_mismatch(tmp_path: Path) -> None:
    """Same session: consecutive rows must still replay-clean."""
    p = tmp_path / "audit.jsonl"
    sid = "chat-session-same"
    snap = _stopped_pef_snapshot()
    row_a = {
        "schema_version": 2,
        "session_id": sid,
        "outcome": "HARD_STOP",
        "pef_snapshot": snap,
        "pef_turn_classification": "stopped",
        "pef_hold_transition": "stop",
    }
    row_b = {
        "schema_version": 2,
        "session_id": sid,
        "outcome": "PASS",
        # Wrong vs classify(prev.pef_snapshot) == stopped
        "pef_turn_classification": "fresh",
        "pef_hold_transition": "none",
    }
    p.write_text(json.dumps(row_a) + "\n" + json.dumps(row_b) + "\n", encoding="utf-8")
    d = verify_pef_linkage_detailed(p, n=50)
    assert d["ok"] is False
    assert d["reason"] == "turn_classification_mismatch"


def test_run_aurora_lens_first_yaml_is_loadable_like_launcher() -> None:
    """Parity with ``run_aurora_lens.py`` config discovery (first existing file in repo)."""
    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "aurora-lens.local.yaml",
        root / "aurora-lens.local.openai.yaml",
        root / "aurora-lens.local.claude.yaml",
        root / "aurora-lens.yaml",
        root / "aurora-lens.yaml.example",
        root / "examples" / "aurora-lens.blank.yaml",
    ]
    existing = [p for p in candidates if p.is_file()]
    assert existing, "expected at least one launcher config candidate in repo"
    cfg: ProxyConfig | None = None
    last_placeholder_err: ValueError | None = None
    for path in existing:
        try:
            cfg = ProxyConfig.from_yaml(path)
            break
        except ValueError as exc:
            if "Unresolved config placeholder" in str(exc):
                last_placeholder_err = exc
                continue
            raise
    assert cfg is not None, (
        "No CI-loadable launcher config among repo candidates; "
        f"last placeholder error: {last_placeholder_err}"
    )
    assert cfg.governance.audit_backend in ("ledger", "jsonl")
    assert 1 <= cfg.listen.port <= 65535

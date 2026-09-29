"""Tests for the verify_audit CLI (Stage F).

Subprocess-invokes python -m aurora_lens.scripts.verify_audit against a temp
JSONL file. Asserts exit codes and output patterns.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


# ── Helpers ──────────────────────────────────────────────────────────

SIGNING_KEY = "test-signing-key"
SIGNING_KEY_BYTES = SIGNING_KEY.encode("utf-8")


def _run_cli(*args, env: dict | None = None) -> tuple[int, str]:
    """Run the verify_audit CLI with the given args. Returns (exit_code, combined output)."""
    import os
    test_env = {**os.environ}
    if env:
        test_env.update(env)
    result = subprocess.run(
        [sys.executable, "-m", "aurora_lens.scripts.verify_audit", *args],
        capture_output=True,
        text=True,
        env=test_env,
        cwd=str(Path(__file__).parent.parent),
    )
    return result.returncode, (result.stdout or "") + (result.stderr or "")


def _build_clean_log(path: Path, n: int = 3) -> None:
    """Write n HMAC-signed, chained audit entries to path."""
    import hmac as _hmac
    prev_cid = "genesis"
    for i in range(n):
        entry = {
            "schema_version": 2,
            "trace_id": f"trace-{i}",
            "timestamp": f"2026-02-26T10:00:0{i}.000000+00:00",
            "turn": i,
            "outcome": "PASS",
        }
        # Compute chain CID
        entry_bytes = json.dumps(entry, separators=(",", ":"), ensure_ascii=False, sort_keys=True)
        cid = hashlib.sha256((entry_bytes + ":" + prev_cid).encode()).hexdigest()
        entry["prev_cid"] = prev_cid
        entry["cid"] = cid
        prev_cid = cid
        # Compute HMAC over the full entry (including cid/prev_cid)
        payload = json.dumps(entry, separators=(",", ":"), ensure_ascii=False, sort_keys=True)
        sig = _hmac.new(SIGNING_KEY_BYTES, payload.encode("utf-8"), "sha256").hexdigest()
        entry["hmac"] = sig
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, separators=(",", ":"), ensure_ascii=False) + "\n")


def _append_forensic_state_hash_without_pef_snapshot(path: Path, prev_cid: str, turn: int) -> None:
    """Append a valid HMAC+chain row with forensic_event.state_hash but no pef_snapshot.

    Replay verification treats this as a failure (missing_pef_snapshot); --no-replay skips that step.
    """
    import hmac as _hmac

    entry = {
        "schema_version": 2,
        "trace_id": f"trace-replay-{turn}",
        "timestamp": f"2026-02-26T10:01:{turn:02d}.000000+00:00",
        "turn": turn,
        "outcome": "PASS",
        "forensic_event": {
            # Truthy state_hash triggers replay path; omission of pef_snapshot fails replay.
            "state_hash": "ab" * 16,
        },
    }
    entry_bytes = json.dumps(entry, separators=(",", ":"), ensure_ascii=False, sort_keys=True)
    cid = hashlib.sha256((entry_bytes + ":" + prev_cid).encode()).hexdigest()
    entry["prev_cid"] = prev_cid
    entry["cid"] = cid
    payload = json.dumps(entry, separators=(",", ":"), ensure_ascii=False, sort_keys=True)
    sig = _hmac.new(SIGNING_KEY_BYTES, payload.encode("utf-8"), "sha256").hexdigest()
    entry["hmac"] = sig
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, separators=(",", ":"), ensure_ascii=False) + "\n")


# ── Tests ─────────────────────────────────────────────────────────────

class TestVerifyAuditCLI:
    """CLI exit code and output contract tests."""

    def test_clean_log_exits_0(self, tmp_path):
        """Clean signed chain exits 0 and prints OK."""
        log = tmp_path / "audit.jsonl"
        _build_clean_log(log, n=3)
        code, out = _run_cli("--path", str(log), "--key", SIGNING_KEY)
        assert code == 0, f"Expected exit 0, got {code}. Output: {out}"
        assert "OK" in out

    def test_missing_file_exits_1(self, tmp_path):
        """Non-existent file exits 1 with error message."""
        log = tmp_path / "nonexistent.jsonl"
        code, out = _run_cli("--path", str(log), "--key", SIGNING_KEY)
        assert code == 1, f"Expected exit 1, got {code}. Output: {out}"
        assert "not found" in out.lower() or "error" in out.lower()

    def test_missing_key_exits_1(self, tmp_path):
        """No signing key provided exits 1."""
        log = tmp_path / "audit.jsonl"
        _build_clean_log(log, n=2)
        # Explicitly clear any inherited signing key env vars so the CLI has no key
        code, out = _run_cli(
            "--path", str(log),
            env={"AURORA_LENS_AUDIT_SIGNING_KEY": "", "AURORA_LENS_AUDIT_SIGNING_KEYS": ""},
        )
        assert code == 1, f"Expected exit 1 (no key), got {code}. Output: {out}"
        low = out.lower()
        assert "no signing key" in low
        assert "verifier needs" in low or "same key" in low

    def test_tampered_entry_exits_1(self, tmp_path):
        """Tampered HMAC in an entry exits 1."""
        log = tmp_path / "audit.jsonl"
        _build_clean_log(log, n=2)
        # Tamper the last entry
        lines = log.read_text(encoding="utf-8").strip().splitlines()
        last = json.loads(lines[-1])
        last["hmac"] = "deadbeef" * 8  # invalid HMAC
        lines[-1] = json.dumps(last, separators=(",", ":"), ensure_ascii=False)
        log.write_text("\n".join(lines) + "\n", encoding="utf-8")

        code, out = _run_cli("--path", str(log), "--key", SIGNING_KEY)
        assert code == 1, f"Expected exit 1 for tampered entry, got {code}. Output: {out}"
        assert "FAIL" in out
        low = out.lower()
        assert "hmac" in low
        assert "signing key" in low or "key" in low
        assert "internal verifier code" in low

    def test_no_replay_skips_forensic_state_hash_replay(self, tmp_path):
        """Without --no-replay, replay fails when state_hash is present but pef_snapshot is absent.

        With --no-replay the verifier skips replay and succeeds (HMAC + chain still pass).
        """
        log = tmp_path / "audit.jsonl"
        _build_clean_log(log, n=1)
        lines = log.read_text(encoding="utf-8").strip().splitlines()
        prev_cid = json.loads(lines[-1])["cid"]
        _append_forensic_state_hash_without_pef_snapshot(log, prev_cid=prev_cid, turn=1)

        code, out = _run_cli("--path", str(log), "--key", SIGNING_KEY)
        assert code == 1, f"Expected exit 1 (replay missing pef_snapshot), got {code}. Output: {out}"
        low = out.lower()
        assert "replay" in low or "state_hash" in low or "pef_snapshot" in low

        code_ok, out_ok = _run_cli("--path", str(log), "--key", SIGNING_KEY, "--no-replay")
        assert code_ok == 0, (
            f"Expected exit 0 with --no-replay, got {code_ok}. Output: {out_ok}"
        )
        assert "OK" in out_ok

    def test_multi_key_flag_works(self, tmp_path):
        """--keys accepts comma-separated keys and verifies successfully with valid key included."""
        log = tmp_path / "audit.jsonl"
        _build_clean_log(log, n=2)
        old_key = "old-key-unused"
        code, out = _run_cli(
            "--path", str(log),
            "--keys", f"{old_key},{SIGNING_KEY}",
        )
        assert code == 0, f"Expected exit 0 with multi-key including valid key, got {code}. Output: {out}"
        assert "OK" in out

"""End-to-end operator story: emit complete vs degraded chain_of_custody rows, run verify_audit both ways."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from aurora_lens.govern.audit_io import append_audit_entry
from aurora_lens.govern.chain_of_custody import build_chain_of_custody_bundle, get_application_version


def _run_verify_audit(
    *args: str,
    env: dict | None = None,
) -> tuple[int, str, str]:
    import os

    test_env = {**os.environ}
    if env:
        test_env.update(env)
    result = subprocess.run(
        [sys.executable, "-m", "aurora_lens.scripts.verify_audit", *args],
        capture_output=True,
        text=True,
        env=test_env,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    return result.returncode, result.stdout or "", result.stderr or ""


def _write_two_pass_rows(
    path: Path,
    *,
    monkeypatch: pytest.MonkeyPatch,
    complete: bool,
) -> None:
    """Append two HMAC-chained JSONL rows with chain_of_custody (complete or degraded)."""
    key = b"k" * 32
    if complete:
        monkeypatch.setenv("AURORA_MODEL_ID", "model-e2e")
        monkeypatch.setenv("AURORA_PROVIDER", "provider-e2e")
    else:
        monkeypatch.delenv("AURORA_MODEL_ID", raising=False)
        monkeypatch.delenv("AURORA_PROVIDER", raising=False)

    coc = build_chain_of_custody_bundle(
        policy_version="1.0",
        policy_source="operator_e2e",
        governance_config={"test": "operator_e2e"},
        application_version=get_application_version(),
    )
    assert (coc["evidence_audit_status"] == "complete") is complete
    if complete:
        assert coc["evidence_degradation_reasons"] == []
    else:
        assert "runtime_model_provenance_incomplete" in coc["evidence_degradation_reasons"]

    prev: str | None = "genesis"
    for suffix in ("a", "b"):
        entry = {
            "schema_version": 2,
            "trace_id": f"e2e-{suffix}",
            "timestamp": "2026-06-01T12:00:00+00:00",
            "outcome": "PASS",
            "chain_of_custody": coc,
        }
        new_cid = append_audit_entry(path, entry, signing_key=key, prev_cid=prev)
        assert new_cid is not None
        prev = new_cid


@pytest.fixture
def signing_key() -> str:
    return "k" * 32


class TestChainOfCustodyOperatorE2E:
    def test_complete_rows_verifier_passes_with_and_without_require_complete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, signing_key: str,
    ):
        log = tmp_path / "audit.jsonl"
        _write_two_pass_rows(log, monkeypatch=monkeypatch, complete=True)

        for extra in ([], ["--require-complete-evidence"]):
            code, out, err = _run_verify_audit(
                "--path",
                str(log),
                "--key",
                signing_key,
                "--no-replay",
                "--chain-of-custody",
                *extra,
            )
            assert code == 0, (out, err, extra)
            combined = out + err
            assert "OK" in combined
            assert "chain-of-custody" in combined.lower() or "chain" in combined.lower()

    def test_degraded_rows_verifier_passes_without_require_complete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, signing_key: str,
    ):
        log = tmp_path / "audit.jsonl"
        _write_two_pass_rows(log, monkeypatch=monkeypatch, complete=False)

        code, out, err = _run_verify_audit(
            "--path",
            str(log),
            "--key",
            signing_key,
            "--no-replay",
            "--chain-of-custody",
        )
        assert code == 0, (out, err)
        assert "OK" in out + err

    def test_degraded_rows_verifier_fails_with_require_complete_plain_explanation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, signing_key: str,
    ):
        log = tmp_path / "audit.jsonl"
        _write_two_pass_rows(log, monkeypatch=monkeypatch, complete=False)

        code, out, err = _run_verify_audit(
            "--path",
            str(log),
            "--key",
            signing_key,
            "--no-replay",
            "--chain-of-custody",
            "--require-complete-evidence",
        )
        assert code == 1, (out, err)
        low = err.lower()
        assert "degraded" in low
        assert "aurora_model_id" in low or "model_id" in low
        assert "require-complete-evidence" in low or "complete" in low
        assert "recorded on the row" in low
        assert "runtime_model_provenance_incomplete" in err
        assert "internal verifier code" in low
        assert "evidence_degraded" in err

    def test_hostile_reader_can_see_what_failed_without_internal_docs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, signing_key: str,
    ):
        """A third party should not need repo context to understand the failure."""
        log = tmp_path / "audit.jsonl"
        _write_two_pass_rows(log, monkeypatch=monkeypatch, complete=False)

        _code, _out, err = _run_verify_audit(
            "--path",
            str(log),
            "--key",
            signing_key,
            "--no-replay",
            "--chain-of-custody",
            "--require-complete-evidence",
        )
        # Plain sentences, not only snake_case codes
        assert "evidence_audit_status" in err
        assert "environment" in err.lower() or "process" in err.lower()
        assert "Location: line" in err

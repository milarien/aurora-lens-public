from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from aurora_lens.corpus.qa_validation import CaseSpec, load_manifest, score_case
from scripts.run_phase17_acceptance import (
    PHASE17_FIXTURES,
    PHASE17_INVENTORY,
    PHASE17_OFFLINE_MANIFEST,
    _start_owned_proxy,
    _cleanup_owned_proxy,
    _find_free_port,
    _port_listening,
    ingest_phase17_fixtures,
    load_source_inventory,
    redact_mapping,
    run_phase17,
)


def test_source_inventory_validation_passes() -> None:
    entries = load_source_inventory(PHASE17_INVENTORY, PHASE17_FIXTURES)
    assert entries
    assert all(entry["source_type"] in {"public_domain", "permissive", "synthetic_derivative"} for entry in entries)
    real_entries = [entry for entry in entries if entry["source_type"] in {"public_domain", "permissive"}]
    assert len(real_entries) >= 3


def test_source_inventory_rejects_missing_licence(tmp_path: Path) -> None:
    inventory = tmp_path / "source_inventory.yaml"
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "x.md").write_text("hello", encoding="utf-8")
    inventory.write_text(
        """
schema_version: 1
fixtures:
  - fixture_id: x
    filename: x.md
    source_type: synthetic_derivative
    source_title: t
    originating_organisation: o
    origin_url: ""
    licence: ""
    redistribution_status: r
    transformation: t
    excerpt_boundaries: b
    sha256_digest: "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    retrieved_or_created_date: "2026-07-13"
    notes: n
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_source_inventory(inventory, fixtures)


def test_manifest_parsing_is_backward_compatible() -> None:
    record_id, _, cases = load_manifest(Path("eval/sample_corpus_qa.manifest.yaml"))
    assert record_id == "sample-policy"
    assert cases
    assert cases[0].expected_governance_any == []
    assert cases[0].semantic_must_include_any == []


def test_semantic_must_include_any_and_min_hits() -> None:
    spec = CaseSpec(
        id="x",
        question="q",
        expect="in_context",
        semantic_must_include_any=["apple", "banana", "citrus"],
        semantic_min_any_hits=2,
    )
    result = score_case(spec, answer="apple and citrus are present", governance="PASS")
    assert result.passed is True


def test_semantic_must_include_all_and_exclude() -> None:
    spec = CaseSpec(
        id="x",
        question="q",
        expect="in_context",
        semantic_must_include_all=["alpha", "beta"],
        semantic_must_exclude=["forbidden"],
    )
    result = score_case(spec, answer="alpha beta forbidden", governance="PASS")
    assert result.passed is False
    assert any("must_exclude" in reason for reason in result.reasons)


def test_citation_expectations_enforced() -> None:
    spec = CaseSpec(
        id="x",
        question="q",
        expect="in_context",
        citation_required=True,
        citation_record_ids_any=["r2"],
    )
    result = score_case(
        spec,
        answer="contextual answer",
        governance="PASS",
        cited_record_ids=["r1"],
    )
    assert result.passed is False
    assert any("citation expectation failed" in reason for reason in result.reasons)


def test_superseded_version_exclusion_enforced() -> None:
    spec = CaseSpec(
        id="x",
        question="q",
        expect="in_context",
        conflict_governing_record_id="v2",
        conflict_superseded_record_ids=["v1"],
        conflict_superseded_version_must_not_drive_answer=True,
    )
    result = score_case(
        spec,
        answer="answer",
        governance="PASS",
        cited_record_ids=["v2", "v1"],
    )
    assert result.passed is False
    assert any("superseded record ids" in reason for reason in result.reasons)


def test_conservative_abstention_allowed_when_configured() -> None:
    spec = CaseSpec(
        id="x",
        question="q",
        expect="in_context",
        allow_conservative_abstain=True,
    )
    result = score_case(
        spec,
        answer="Insufficient context in retrieved chunks.",
        governance="CONTAIN",
    )
    assert result.passed is True


def test_malformed_fixture_ingests_with_chunks(tmp_path: Path) -> None:
    entries = load_source_inventory(PHASE17_INVENTORY, PHASE17_FIXTURES)
    _, ingested = ingest_phase17_fixtures(entries, PHASE17_FIXTURES, tmp_path / "corpus")
    malformed = [row for row in ingested if row.fixture_id in {"malformed_truncated_markdown", "malformed_encoding_noise"}]
    assert len(malformed) == 2
    assert all(row.chunk_count > 0 for row in malformed)
    unsupported = [row for row in ingested if row.fixture_id == "malformed_unsupported_binary"]
    assert len(unsupported) == 1
    assert unsupported[0].ingest_success is False
    assert "Unsupported source type" in unsupported[0].ingest_error


def test_offline_runner_exit_status_and_reports(tmp_path: Path) -> None:
    code, md_report, json_report = run_phase17(
        mode="offline",
        manifest_path=PHASE17_OFFLINE_MANIFEST,
        proxy="http://localhost:8081",
        provider="openai",
        model=None,
        max_chars=10000,
        out_root=tmp_path / "phase17",
    )
    assert code == 0
    assert md_report.exists()
    assert json_report.exists()


def test_secret_redaction_masks_sensitive_keys() -> None:
    redacted = redact_mapping(
        {
            "authorization": "Bearer abc",
            "nested": {"api_key": "secret", "ok": "value"},
        }
    )
    assert redacted["authorization"] == "***REDACTED***"
    assert redacted["nested"]["api_key"] == "***REDACTED***"
    assert redacted["nested"]["ok"] == "value"


def test_live_mode_requires_model_argument() -> None:
    with pytest.raises(ValueError):
        run_phase17(
            mode="live",
            manifest_path=PHASE17_OFFLINE_MANIFEST,
            proxy="http://localhost:8081",
            provider="openai",
            model=None,
            max_chars=10000,
            out_root=Path("build/phase17-test"),
        )


def test_cleanup_owned_proxy_removes_pid_and_listener(tmp_path: Path) -> None:
    port = _find_free_port()
    pid_file = tmp_path / "pid.txt"
    log_file = tmp_path / "log.txt"
    process = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    pid_file.write_text(str(process.pid), encoding="utf-8")
    for _ in range(20):
        if _port_listening(port):
            break
    cleanup = _cleanup_owned_proxy(
        type(
            "OwnedProxyStub",
            (),
            {
                "process": process,
                "port": port,
                "proxy_url": f"http://127.0.0.1:{port}",
                "pid_file": pid_file,
                "log_file": log_file,
            },
        )()
    )
    assert cleanup["pid_file_removed"] is True
    assert cleanup["port_listening_after_cleanup"] is False
    assert not pid_file.exists()


def test_live_run_cleans_up_owned_proxy_on_exception(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    os.environ["AURORA_LENS_UPSTREAM_API_KEY"] = "test-key"

    class FakeOwnedProxy:
        def __init__(self) -> None:
            self.process = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            self.port = 50123
            self.proxy_url = "http://127.0.0.1:50123"
            self.pid_file = tmp_path / "phase17-live-proxy.pid"
            self.pid_file.write_text(str(self.process.pid), encoding="utf-8")
            self.log_file = tmp_path / "phase17-live-proxy.log"

    fake_owned = FakeOwnedProxy()

    monkeypatch.setattr(
        "scripts.run_phase17_acceptance._start_owned_proxy",
        lambda **_: fake_owned,
    )
    monkeypatch.setattr(
        "scripts.run_phase17_acceptance._run_live_case",
        lambda **_: (_ for _ in ()).throw(RuntimeError("forced live failure")),
    )

    with pytest.raises(RuntimeError):
        run_phase17(
            mode="live",
            manifest_path=Path("eval/phase17/manifests/phase17_live.manifest.yaml"),
            proxy="http://localhost:8081",
            provider="openai",
            model="gpt-4o-mini-2024-07-18",
            max_chars=10000,
            out_root=tmp_path / "phase17",
        )

    assert fake_owned.process.poll() is not None
    assert not fake_owned.pid_file.exists()


def test_live_run_cleans_up_owned_proxy_on_failed_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    os.environ["AURORA_LENS_UPSTREAM_API_KEY"] = "test-key"

    class FakeOwnedProxy:
        def __init__(self) -> None:
            self.process = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            self.port = 50124
            self.proxy_url = "http://127.0.0.1:50124"
            self.pid_file = tmp_path / "phase17-live-proxy-fail.pid"
            self.pid_file.write_text(str(self.process.pid), encoding="utf-8")
            self.log_file = tmp_path / "phase17-live-proxy-fail.log"

    fake_owned = FakeOwnedProxy()
    monkeypatch.setattr(
        "scripts.run_phase17_acceptance._start_owned_proxy",
        lambda **_: fake_owned,
    )
    monkeypatch.setattr(
        "scripts.run_phase17_acceptance._run_live_case",
        lambda **_: {"choices": [{"message": {"content": "irrelevant"}}], "aurora": {"governance": "PASS", "llm_called": True}},
    )
    code, _, _ = run_phase17(
        mode="live",
        manifest_path=Path("eval/phase17/manifests/phase17_live.manifest.yaml"),
        proxy="http://localhost:8081",
        provider="openai",
        model="gpt-4o-mini-2024-07-18",
        max_chars=10000,
        out_root=tmp_path / "phase17",
    )
    assert code in {0, 1}
    assert fake_owned.process.poll() is not None
    assert not fake_owned.pid_file.exists()


def test_start_owned_proxy_timeout_cleans_process_and_pidfile(tmp_path: Path) -> None:
    with pytest.raises(TimeoutError):
        _start_owned_proxy(
            run_dir=tmp_path,
            provider="openai",
            model="gpt-4o-mini-2024-07-18",
            startup_timeout_s=0.5,
            command_override=[sys.executable, "-c", "import time; time.sleep(60)"],
        )
    assert not (tmp_path / "phase17-live-proxy.pid").exists()

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from aurora_lens.proxy.app import create_app
from aurora_lens.proxy.config import ProxyConfig
from release_guard import assert_clean_release_tree, inspect_distribution_archive


def test_first_governed_event_creates_first_audit_entry(tmp_path: Path) -> None:
    audit_path = tmp_path / "logs" / "audit.jsonl"
    cfg = ProxyConfig.from_mapping(
        {
            "upstream": {"provider": "mock", "model": "mock"},
            "listen": {"host": "127.0.0.1", "port": 8081},
            "governance": {
                "default_policy": "strict",
                "mode": "public",
                "audit_log": str(audit_path),
            },
            "extraction": {"backend": "spacy"},
        }
    )
    app = create_app(cfg)
    client = TestClient(app)

    assert (not audit_path.exists()) or (audit_path.read_text(encoding="utf-8").strip() == "")
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        },
    )
    assert response.status_code == 200
    assert audit_path.exists()
    lines = [ln for ln in audit_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1


def test_clean_guard_blocks_runtime_audit_artifacts(tmp_path: Path) -> None:
    (tmp_path / "audit.jsonl").write_text('{"event":"x"}\n', encoding="utf-8")
    try:
        assert_clean_release_tree(tmp_path)
    except RuntimeError as exc:
        assert "audit.jsonl" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("clean-audit guard should fail when populated audit log exists")


def test_archive_inspection_blocks_abs_paths_and_credentials(tmp_path: Path) -> None:
    wheel_path = tmp_path / "sample.whl"
    import zipfile

    with zipfile.ZipFile(wheel_path, "w") as zf:
        zf.writestr("pkg.dist-info/METADATA", "path=C:\\Users\\alice\\secret\nkey=sk-ABC123DEF456GHI789")
    issues = inspect_distribution_archive(wheel_path)
    assert any(("local path marker" in i) or ("absolute path text" in i) for i in issues)
    assert any("credential" in i for i in issues)


def test_archive_inspection_blocks_bundled_runtime_history(tmp_path: Path) -> None:
    wheel_path = tmp_path / "sample_runtime.whl"
    import zipfile

    with zipfile.ZipFile(wheel_path, "w") as zf:
        zf.writestr("aurora_lens/audit.jsonl", '{"event":"persisted-history"}\n')
        zf.writestr("aurora_lens/state/aurora-lens.proxy.pid", "1234\n")
    issues = inspect_distribution_archive(wheel_path)
    assert any("forbidden runtime file" in i for i in issues)
    assert any("runtime state/evidence" in i for i in issues)


def test_archive_inspection_allows_synthetic_test_fixtures_only(tmp_path: Path) -> None:
    wheel_path = tmp_path / "sample_tests_only.whl"
    import zipfile

    with zipfile.ZipFile(wheel_path, "w") as zf:
        zf.writestr(
            "tests/fixtures/synthetic-fixture.txt",
            "C:\\Users\\fixture\\path\nsk-ABCDEFGHIJKLMNOPQRSTUVWXYZ1234",
        )
    issues = inspect_distribution_archive(wheel_path)
    assert issues == []

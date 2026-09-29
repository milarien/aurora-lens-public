from __future__ import annotations

import zipfile
import sys
from pathlib import Path

import pytest
import yaml

from aurora_lens.proxy.config import ProxyConfig
from aurora_lens.setup.core import (
    build_readiness_report,
    export_support_bundle,
    generated_yaml_text,
    list_provider_models,
    parse_setup_input,
    reset_local_data,
    run_governance_smoke,
    save_configuration,
    _store_secret_with_keyring,
)


def _base_payload() -> dict[str, object]:
    return {
        "work_type": "general_ai_governance",
        "audience_type": "customers_public",
        "provider_type": "openai",
        "model_name": "gpt-4.1-mini",
        "api_endpoint": "https://api.openai.com/v1",
        "api_key": "",
        "listen_port": 8000,
        "audit_log_path": "./audit.jsonl",
        "session_backend": "memory",
        "redis_url": "redis://127.0.0.1:6379/0",
        "evidence_vault_enabled": False,
        "evidence_key": "",
        "policy_profile": "strict",
        "extraction_backend": "spacy",
    }


def test_readiness_blocks_missing_provider_key(tmp_path: Path) -> None:
    cfg = parse_setup_input(_base_payload())
    report = build_readiness_report(cfg, tmp_path)
    assert report["start_allowed"] is False
    blocking = [c for c in report["checks"] if c["severity"] == "BLOCKING" and c["status"] == "fail"]
    assert any(c["id"] == "api_key" for c in blocking)


def test_generated_yaml_validates_with_proxy_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    payload = _base_payload()
    payload["api_key"] = "test-key"
    cfg = parse_setup_input(payload)
    text = generated_yaml_text(cfg)
    data = yaml.safe_load(text)
    proxy_cfg = ProxyConfig.from_mapping(data)
    proxy_cfg.validate()
    assert proxy_cfg.listen.port == 8000


def test_work_type_and_audience_map_to_domain_and_mode_separately() -> None:
    payload = _base_payload()
    payload["work_type"] = "medical"
    payload["audience_type"] = "staff_organisation"
    cfg = parse_setup_input(payload)
    text = generated_yaml_text(cfg)
    data = yaml.safe_load(text)
    assert data["governance"]["default_domain"] == "medical"
    assert data["governance"]["mode"] == "enterprise"
    assert data["governance"]["default_policy"] == "strict"


def test_parse_setup_input_has_no_universal_model_default() -> None:
    payload = _base_payload()
    payload.pop("model_name", None)
    cfg = parse_setup_input(payload)
    assert cfg.model_name == ""


def test_save_configuration_generates_evidence_key_when_enabled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("aurora_lens.setup.core._store_secret_with_keyring", lambda *args, **kwargs: False)
    payload = _base_payload()
    payload["api_key"] = "test-key"
    payload["evidence_vault_enabled"] = True
    payload["evidence_key"] = ""
    cfg = parse_setup_input(payload)
    result = save_configuration(cfg, home=tmp_path)
    assert result["secret_backend"] == "env_file"
    assert result["evidence_key_generated"] is True
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "AURORA_LENS_EVIDENCE_KEY=" in env_text


def test_provider_model_listing_reports_unavailable_without_key() -> None:
    payload = _base_payload()
    payload["api_key"] = ""
    cfg = parse_setup_input(payload)
    result = list_provider_models(cfg)
    assert result["ok"] is False
    assert "API key" in result["message"]


def test_store_secret_with_keyring_falls_back_when_backend_errors(monkeypatch) -> None:
    class _BrokenKeyring:
        @staticmethod
        def set_password(service, username, secret):  # noqa: ANN001,ARG004
            raise RuntimeError("no backend")

    monkeypatch.setitem(sys.modules, "keyring", _BrokenKeyring)

    assert _store_secret_with_keyring("aurora-lens", "provider:openai", "test-key") is False


@pytest.mark.asyncio
async def test_governance_smoke_is_deterministic_contain(tmp_path: Path) -> None:
    result = await run_governance_smoke(home=tmp_path)
    assert result["ok"] is True
    assert result["action"] == "CONTAIN"
    assert "Audit record written." in result["steps"]
    assert (tmp_path / "setup_governance_smoke_audit.jsonl").exists()


def test_export_support_bundle_includes_manifest_and_existing_files(tmp_path: Path) -> None:
    (tmp_path / "launcher.log").write_text("launcher entry", encoding="utf-8")
    (tmp_path / "audit.jsonl").write_text('{"ok":true}\n', encoding="utf-8")
    cfg = _base_payload()
    cfg["api_key"] = "abc123"
    saved = save_configuration(parse_setup_input(cfg), home=tmp_path)
    bundle = export_support_bundle(home=tmp_path, config_path=Path(saved["config_path"]))
    archive = Path(bundle["archive_path"])
    assert archive.exists()
    with zipfile.ZipFile(archive, "r") as zf:
        names = set(zf.namelist())
    assert "manifest.json" in names
    assert "launcher.log" in names


def test_reset_local_data_removes_selected_files(tmp_path: Path) -> None:
    for name in ("launcher.log", "audit.jsonl", "setup_governance_smoke_audit.jsonl", "aurora-lens.proxy.pid"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    cfg_path = tmp_path / "aurora-lens.yaml"
    cfg_path.write_text("setup:\n  provider_type: openai\n", encoding="utf-8")
    result = reset_local_data(
        home=tmp_path,
        provider_type="openai",
        config_path=cfg_path,
        remove_config=True,
        remove_audit=True,
        clear_secrets=False,
    )
    assert result["remove_config"] is True
    assert not (tmp_path / "launcher.log").exists()
    assert not (tmp_path / "audit.jsonl").exists()
    assert not cfg_path.exists()


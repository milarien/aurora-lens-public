from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import yaml

from aurora_lens.setup.wizard_app import create_setup_app


def _payload() -> dict[str, object]:
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
    }


def _csrf_header(client: TestClient) -> dict[str, str]:
    token = client.cookies.get("aurora_setup_csrf")
    assert token
    return {"x-csrf-token": token}


def test_setup_page_sets_csrf_cookie(tmp_path: Path) -> None:
    app = create_setup_app(home=tmp_path, controller_port=8765)
    client = TestClient(app)
    response = client.get("/setup")
    assert response.status_code == 200
    assert "Aurora-Lens Setup" in response.text
    assert client.cookies.get("aurora_setup_csrf")


def test_providers_payload_includes_wizard_axes_and_defaults(tmp_path: Path) -> None:
    app = create_setup_app(home=tmp_path, controller_port=8765)
    client = TestClient(app)
    response = client.get("/api/providers")
    assert response.status_code == 200
    data = response.json()
    assert any(item["id"] == "medical" for item in data["work_types"])
    assert any(item["id"] == "staff_organisation" for item in data["audience_types"])
    assert data["defaults"]["evidence_vault_enabled"] is True
    assert "audit_log_path" in data["defaults"]
    assert "listen_port" in data["defaults"]


def test_readiness_reports_blocking_for_missing_api_key(tmp_path: Path) -> None:
    app = create_setup_app(home=tmp_path, controller_port=8765)
    client = TestClient(app)
    client.get("/setup")
    response = client.post("/api/readiness", json=_payload(), headers=_csrf_header(client))
    assert response.status_code == 200
    data = response.json()
    assert data["start_allowed"] is False
    assert any(
        c["id"] == "api_key" and c["severity"] == "BLOCKING"
        for c in data["checks"]
    )


def test_save_config_writes_yaml(tmp_path: Path) -> None:
    app = create_setup_app(home=tmp_path, controller_port=8765)
    client = TestClient(app)
    client.get("/setup")
    payload = _payload()
    payload["api_key"] = "abc123"
    payload["evidence_vault_enabled"] = True
    payload["evidence_key"] = ""
    response = client.post("/api/save-config", json=payload, headers=_csrf_header(client))
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert (tmp_path / "aurora-lens.yaml").exists()
    saved = yaml.safe_load((tmp_path / "aurora-lens.yaml").read_text(encoding="utf-8"))
    assert saved["upstream"]["model"] == "gpt-4.1-mini"


def test_post_blocks_untrusted_origin(tmp_path: Path) -> None:
    app = create_setup_app(home=tmp_path, controller_port=8765)
    client = TestClient(app)
    client.get("/setup")
    response = client.post(
        "/api/preflight",
        json={},
        headers={**_csrf_header(client), "origin": "http://evil.example"},
    )
    assert response.status_code == 403
    assert response.json()["ok"] is False


def test_csrf_token_rotates_per_app_launch(tmp_path: Path) -> None:
    app1 = create_setup_app(home=tmp_path, controller_port=8765)
    c1 = TestClient(app1)
    c1.get("/setup")
    token1 = c1.cookies.get("aurora_setup_csrf")
    assert token1

    app2 = create_setup_app(home=tmp_path, controller_port=8765)
    c2 = TestClient(app2)
    c2.get("/setup")
    token2 = c2.cookies.get("aurora_setup_csrf")
    assert token2
    assert token1 != token2


def test_start_is_blocked_when_readiness_has_blocking_failures(tmp_path: Path) -> None:
    app = create_setup_app(home=tmp_path, controller_port=8765)
    client = TestClient(app)
    client.get("/setup")
    response = client.post("/api/start", json=_payload(), headers=_csrf_header(client))
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is False
    assert "blocking_checks" in data


def test_start_uses_saved_secret_when_api_key_omitted(tmp_path: Path, monkeypatch) -> None:
    app = create_setup_app(home=tmp_path, controller_port=8765)
    client = TestClient(app)
    client.get("/setup")

    cfg = {
        "upstream": {"provider": "openai", "model": "gpt-4.1-mini", "api_key": "${AURORA_LENS_UPSTREAM_API_KEY}"},
        "listen": {"host": "127.0.0.1", "port": 8000},
        "governance": {"default_policy": "strict", "mode": "public", "audit_log": "./audit.jsonl"},
        "session": {"backend": "memory", "ttl_seconds": 3600, "redis_url": ""},
        "setup": {"provider_type": "openai"},
    }
    (tmp_path / "aurora-lens.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (tmp_path / ".env").write_text("AURORA_LENS_UPSTREAM_API_KEY=test-from-env\n", encoding="utf-8")

    called = {}

    def _fake_start_proxy(self, config_path, runtime_env):  # noqa: ANN001
        called["runtime_env"] = dict(runtime_env)
        from aurora_lens.launcher.lifecycle import RuntimeStatusCode
        from aurora_lens.launcher.state import ProxyStartResult
        return ProxyStartResult(
            ok=True,
            code=RuntimeStatusCode.RUNNING_HEALTHY,
            message="ok",
            proxy_url="http://127.0.0.1:8000",
        )

    monkeypatch.setattr("aurora_lens.launcher.state.LauncherState.start_proxy", _fake_start_proxy)

    payload = _payload()
    payload["api_key"] = ""
    payload["listen_port"] = 18080
    response = client.post("/api/start", json=payload, headers=_csrf_header(client))
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert called["runtime_env"]["AURORA_LENS_UPSTREAM_API_KEY"] == "test-from-env"


def test_provider_model_list_loading_success(tmp_path: Path, monkeypatch) -> None:
    app = create_setup_app(home=tmp_path, controller_port=8765)
    client = TestClient(app)
    client.get("/setup")

    monkeypatch.setattr(
        "aurora_lens.setup.wizard_app.list_provider_models",
        lambda cfg: {"ok": True, "models": ["model-a", "model-b"], "message": "ok"},
    )
    response = client.post("/api/provider-models", json=_payload(), headers=_csrf_header(client))
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["models"] == ["model-a", "model-b"]


def test_provider_model_list_unavailable_reports_manual_fallback(tmp_path: Path, monkeypatch) -> None:
    app = create_setup_app(home=tmp_path, controller_port=8765)
    client = TestClient(app)
    client.get("/setup")

    monkeypatch.setattr(
        "aurora_lens.setup.wizard_app.list_provider_models",
        lambda cfg: {
            "ok": False,
            "models": [],
            "message": "Model listing is unavailable for this provider or endpoint. Use manual model entry in Advanced settings.",
        },
    )
    response = client.post("/api/provider-models", json=_payload(), headers=_csrf_header(client))
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is False
    assert "manual model entry" in data["message"]


def test_restart_proxy_endpoint_uses_saved_config(tmp_path: Path, monkeypatch) -> None:
    app = create_setup_app(home=tmp_path, controller_port=8765)
    client = TestClient(app)
    client.get("/setup")
    monkeypatch.setattr("aurora_lens.setup.core._store_secret_with_keyring", lambda *args, **kwargs: False)
    payload = _payload()
    payload["api_key"] = "abc123"
    client.post("/api/save-config", json=payload, headers=_csrf_header(client))

    calls = {"stop": 0, "start": 0}

    def _fake_stop(self):  # noqa: ANN001
        calls["stop"] += 1
        return True

    def _fake_start(self, config_path, runtime_env):  # noqa: ANN001
        calls["start"] += 1
        from aurora_lens.launcher.lifecycle import RuntimeStatusCode
        from aurora_lens.launcher.state import ProxyStartResult
        assert Path(config_path).name == "aurora-lens.yaml"
        assert "AURORA_LENS_UPSTREAM_API_KEY" in runtime_env
        return ProxyStartResult(
            ok=True,
            code=RuntimeStatusCode.RUNNING_HEALTHY,
            message="restarted",
            proxy_url="http://127.0.0.1:8000",
        )

    monkeypatch.setattr("aurora_lens.launcher.state.LauncherState.stop_proxy", _fake_stop)
    monkeypatch.setattr("aurora_lens.launcher.state.LauncherState.start_proxy", _fake_start)

    response = client.post("/api/restart", json={"config_path": "aurora-lens.yaml"}, headers=_csrf_header(client))
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert calls["stop"] == 1
    assert calls["start"] == 1


def test_export_logs_endpoint_creates_archive(tmp_path: Path) -> None:
    app = create_setup_app(home=tmp_path, controller_port=8765)
    client = TestClient(app)
    client.get("/setup")
    (tmp_path / "launcher.log").write_text("entry", encoding="utf-8")

    response = client.post("/api/export-logs", json={}, headers=_csrf_header(client))
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert Path(data["archive_path"]).exists()


def test_reset_local_data_requires_confirmation_phrase(tmp_path: Path) -> None:
    app = create_setup_app(home=tmp_path, controller_port=8765)
    client = TestClient(app)
    client.get("/setup")
    response = client.post(
        "/api/reset-local-data",
        json={"confirm_phrase": "RESET"},
        headers=_csrf_header(client),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is False
    assert "Reset blocked" in data["message"]


def test_reset_local_data_deletes_requested_files(tmp_path: Path) -> None:
    app = create_setup_app(home=tmp_path, controller_port=8765)
    client = TestClient(app)
    client.get("/setup")
    (tmp_path / "audit.jsonl").write_text("{}", encoding="utf-8")
    (tmp_path / "launcher.log").write_text("entry", encoding="utf-8")

    response = client.post(
        "/api/reset-local-data",
        json={"confirm_phrase": "RESET AURORA-LENS", "remove_audit": True},
        headers=_csrf_header(client),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert not (tmp_path / "audit.jsonl").exists()


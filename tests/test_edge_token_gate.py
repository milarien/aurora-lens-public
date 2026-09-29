"""Transport-only public-demo edge token gate (x-aurora-edge-token)."""

from __future__ import annotations

import pytest

from tests.conftest import PYTEST_AURORA_EDGE_TOKEN
from tests.skip_reasons import SKIP_STARLETTE_HTTP_TESTCLIENT
from tests.test_proxy import CountingMockAdapter, MockAdapter


def _make_app(monkeypatch, adapter=None):
    from aurora_lens.proxy.app import create_app
    from aurora_lens.proxy.config import ProxyConfig

    adapters = adapter or MockAdapter()
    monkeypatch.setattr(
        "aurora_lens.proxy.app._build_provider_adapters",
        lambda cfg: (adapters, adapters),
    )
    cfg = ProxyConfig.from_mapping(
        {
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "auth": {"enabled": False},
            "hardening": {},
        }
    )
    return create_app(cfg), adapters


class TestPublicDemoEdgeTokenGate:
    def test_missing_token_forbidden_before_model(self, monkeypatch):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        counter = CountingMockAdapter()
        app, _ = _make_app(monkeypatch, adapter=counter)
        client = TestClient(app)
        r = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"x-test-omit-edge-token": "1"},
        )
        assert r.status_code == 403
        body = r.json()
        assert body["error"]["message"] == "Forbidden"
        assert body["error"]["type"] == "authentication_error"
        assert counter.generate_calls == 0

    def test_wrong_token_forbidden_before_model(self, monkeypatch):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        counter = CountingMockAdapter()
        app, _ = _make_app(monkeypatch, adapter=counter)
        client = TestClient(app)
        r = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
            headers={
                "x-test-omit-edge-token": "1",
                "x-aurora-edge-token": "not-the-right-token",
            },
        )
        assert r.status_code == 403
        assert r.json()["error"]["message"] == "Forbidden"
        assert counter.generate_calls == 0

    def test_absent_config_fails_closed(self, monkeypatch):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        monkeypatch.delenv("AURORA_EDGE_TOKEN", raising=False)
        counter = CountingMockAdapter()
        app, _ = _make_app(monkeypatch, adapter=counter)
        client = TestClient(app)
        r = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
            headers={
                "x-test-omit-edge-token": "1",
                "x-aurora-edge-token": PYTEST_AURORA_EDGE_TOKEN,
            },
        )
        assert r.status_code == 403
        assert counter.generate_calls == 0

    def test_correct_token_allows_chat(self, monkeypatch):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        counter = CountingMockAdapter()
        app, _ = _make_app(monkeypatch, adapter=counter)
        client = TestClient(app)
        r = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"x-aurora-edge-token": PYTEST_AURORA_EDGE_TOKEN},
        )
        assert r.status_code == 200
        assert counter.generate_calls == 1

    def test_correct_token_allows_new_scenario(self, monkeypatch):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        app, _ = _make_app(monkeypatch)
        client = TestClient(app)
        r = client.post(
            "/v1/session/new-scenario",
            headers={"x-aurora-edge-token": PYTEST_AURORA_EDGE_TOKEN},
        )
        assert r.status_code == 200
        assert "session_id" in r.json() or "pef_context_id" in r.json()

    def test_audit_verify_does_not_require_edge_token(self, monkeypatch):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        app, _ = _make_app(monkeypatch)
        client = TestClient(app)
        r = client.get(
            "/v1/audit/verify?n=5",
            headers={"x-test-omit-edge-token": "1"},
        )
        # Forensics console reads verify/recent without an edge token.
        assert r.status_code != 403

    def test_audit_recent_does_not_require_edge_token(self, monkeypatch):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        app, _ = _make_app(monkeypatch)
        client = TestClient(app)
        r = client.get(
            "/v1/audit/recent?n=5",
            headers={"x-test-omit-edge-token": "1"},
        )
        assert r.status_code != 403

    def test_health_unaffected(self, monkeypatch):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        app, _ = _make_app(monkeypatch)
        client = TestClient(app)
        r = client.get("/health", headers={"x-test-omit-edge-token": "1"})
        assert r.status_code == 200

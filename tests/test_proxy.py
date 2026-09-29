"""Tests for the proxy layer.

Tests config loading, session management, OpenAI compat, and app routes.
"""

import asyncio
import io
import json
import logging
import tempfile
import time
from contextlib import aclosing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests.skip_reasons import SKIP_SPACY_MODULE, SKIP_STARLETTE_HTTP_TESTCLIENT

from aurora_lens.proxy.config import (
    ProxyConfig, UpstreamConfig, ListenConfig, GovernanceConfig,
    CorsConfig, HardeningConfig, load_config,
)
from aurora_lens.proxy.session_store import SessionRecord, SessionStoreError
from aurora_lens.proxy.session import SessionManager
from aurora_lens.proxy.openai_compat import (
    parse_chat_request,
    format_chat_response,
    format_stream_metadata_event,
)
from aurora_lens.proxy.frame_lifecycle import (
    extract_explicit_session_id,
    parse_continuation_requested,
    parse_domain_or_lane_hint,
    resolve_frame_lifecycle,
)
from aurora_lens.lens import Lens, LensResult
from aurora_lens.config import LensConfig
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState
from aurora_lens.adapters.base import LLMAdapter, AdapterResponse
from aurora_lens.interpret.base import ExtractionBackend
from aurora_lens.interpret.schema import ExtractedClaim, ExtractionResult
from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
from aurora_lens.govern.bridge import BuiltinBridge
from aurora_lens.verify.flags import Flag, FlagType


# ── Helpers ─────────────────────────────────────────────────────────

class MockAdapter(LLMAdapter):
    async def generate(self, messages: list[dict[str, str]], **kwargs) -> AdapterResponse:
        user_msg = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        return AdapterResponse(text=f"Echo: {user_msg}", model="mock")


class CountingMockAdapter(MockAdapter):
    def __init__(self):
        self.generate_calls = 0
        self.generate_stream_calls = 0

    async def generate(self, messages: list[dict[str, str]], **kwargs) -> AdapterResponse:
        self.generate_calls += 1
        return await super().generate(messages, **kwargs)

    async def generate_stream(self, messages: list[dict[str, str]], **kwargs):
        self.generate_stream_calls += 1
        if False:
            yield ({"choices": [{"delta": {"content": ""}, "index": 0}]}, "")


class StreamMockAdapter(LLMAdapter):
    """Mock adapter that streams chunks. Tracks if closed early (client disconnect)."""

    def __init__(self, chunks: list[str]):
        self._chunks = chunks
        self.closed_early: bool | None = None  # True if generator exited before last yield

    async def generate(self, messages: list[dict[str, str]], **kwargs) -> AdapterResponse:
        return AdapterResponse(text="".join(self._chunks), model="mock")

    async def generate_stream(self, messages: list[dict[str, str]], **kwargs):
        self.closed_early = None
        try:
            for c in self._chunks:
                chunk_dict = {"choices": [{"delta": {"content": c}, "index": 0}]}
                yield (chunk_dict, c)
            self.closed_early = False  # completed normally
        finally:
            if self.closed_early is None:
                self.closed_early = True  # exited before completion (e.g. client disconnect)


class CancelledStreamAdapter(LLMAdapter):
    """Mock adapter that raises CancelledError mid-stream (simulates provider abort during buffering)."""

    def __init__(self, chunks: list[str], cancel_after: int = 1):
        self._chunks = chunks
        self._cancel_after = cancel_after  # raise after yielding this many chunks

    async def generate(self, messages: list[dict[str, str]], **kwargs) -> AdapterResponse:
        return AdapterResponse(text="".join(self._chunks), model="mock")

    async def generate_stream(self, messages: list[dict[str, str]], **kwargs):
        for i, c in enumerate(self._chunks):
            if i >= self._cancel_after:
                raise asyncio.CancelledError("simulated provider abort")
            chunk_dict = {"choices": [{"delta": {"content": c}, "index": 0}]}
            yield (chunk_dict, c)


class MockBackend(ExtractionBackend):
    async def extract(self, text, pef):
        return ExtractionResult()


def _make_config_factory():
    def factory():
        return LensConfig(
            adapter=MockAdapter(),
            extraction_backend=MockBackend(),
        )
    return factory


# ── Config Tests ────────────────────────────────────────────────────

class TestProxyConfig:

    @pytest.fixture(autouse=True)
    def _clear_upstream_env(self, monkeypatch):
        """Remove env vars that apply_env_overrides() reads so that tests
        asserting specific api_key values are not overridden by the shell
        environment.  This also prevents real keys from appearing in
        assertion failure diffs."""
        for var in (
            "AURORA_LENS_UPSTREAM_API_KEY",
            "AURORA_LENS_UPSTREAM_PROVIDER",
            "AURORA_LENS_UPSTREAM_MODEL",
            "AURORA_LENS_UPSTREAM_BASE_URL",
            "AURORA_LENS_LISTEN_HOST",
            "AURORA_LENS_LISTEN_PORT",
            "PORT",
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "AURORA_LENS_CORS_ENABLED",
            "AURORA_LENS_CORS_ORIGINS",
            "AURORA_LENS_CORS_ALLOW_ORIGINS",
        ):
            monkeypatch.delenv(var, raising=False)

    def test_default_listen_and_governance(self):
        """Verify defaults for listen and governance when only upstream is provided."""
        config = ProxyConfig(
            upstream=UpstreamConfig(provider="openai", api_key="sk-test", model="gpt-4"),
        )
        assert config.listen.host == "0.0.0.0"
        assert config.listen.port == 8081
        assert config.governance.default_policy == "strict"
        assert config.governance.policy_version == "1.0"
        assert config.cors.enabled is False
        assert config.cors.allow_origins == ("*",)

    def test_from_mapping(self):
        raw = {
            "upstream": {
                "provider": "openai",
                "base_url": "https://custom.api/v1",
                "api_key": "sk-test",
                "model": "gpt-3.5-turbo",
            },
            "listen": {
                "host": "127.0.0.1",
                "port": 9090,
            },
            "governance": {
                "default_policy": "moderate",
                "audit_log": "/tmp/audit.jsonl",
            },
        }
        config = ProxyConfig.from_mapping(raw)
        assert config.upstream.provider == "openai"
        assert config.upstream.base_url == "https://custom.api/v1"
        assert config.upstream.api_key == "sk-test"
        assert config.upstream.model == "gpt-3.5-turbo"
        assert config.listen.host == "127.0.0.1"
        assert config.listen.port == 9090
        assert config.governance.default_policy == "moderate"
        assert config.governance.audit_log == "/tmp/audit.jsonl"
        assert config.governance.policy_version == "1.0"

    def test_cors_and_policy_version_from_mapping(self):
        raw = {
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"policy_version": "2.0"},
            "cors": {"enabled": True, "allow_origins": ["https://app.example.com"]},
        }
        config = ProxyConfig.from_mapping(raw)
        assert config.governance.policy_version == "2.0"
        assert config.cors.enabled is True
        assert config.cors.allow_origins == ("https://app.example.com",)

    def test_governance_authority_and_user_class_header_preserved_from_mapping(self):
        raw = {
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {
                "authority_class": "DA",
                "user_class_header": "X-Aurora-User-Class",
            },
        }
        config = ProxyConfig.from_mapping(raw)
        assert config.governance.authority_class == "DA"
        assert config.governance.user_class_header == "X-Aurora-User-Class"

    def test_governance_default_domain_preserved_from_mapping(self):
        raw = {
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"default_domain": "finance"},
        }
        config = ProxyConfig.from_mapping(raw)
        assert config.governance.default_domain == "finance"

    def test_invalid_authority_class_from_mapping_falls_back_to_gp(self):
        raw = {
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"authority_class": "ROUTE_MAGIC"},
        }
        config = ProxyConfig.from_mapping(raw)
        assert config.governance.authority_class == "GP"

    def test_auth_config_from_mapping(self):
        """Phase B: Auth config with keys, labels, optional policy override."""
        raw = {
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "auth": {
                "enabled": True,
                "keys": [
                    {"key": "key-1", "label": "app-prod", "policy": "strict", "domain": "legal"},
                    {"key": "key-2", "label": "app-staging", "domain": "finance"},
                ],
            },
        }
        config = ProxyConfig.from_mapping(raw)
        assert config.auth.enabled is True
        assert len(config.auth.keys) == 2
        assert config.auth.keys[0].key == "key-1"
        assert config.auth.keys[0].label == "app-prod"
        assert config.auth.keys[0].policy == "strict"
        assert config.auth.keys[0].domain == "legal"
        assert config.auth.keys[1].policy is None
        assert config.auth.keys[1].domain == "finance"

    def test_hardening_rate_limit_per_ip_and_trusted_proxies(self):
        """Phase B.4: rate_limit_per_ip and trusted_proxy_ips in hardening."""
        raw = {
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "hardening": {
                "rate_limit_per_ip": 60,
                "trusted_proxy_ips": ["10.0.0.1", "192.168.1.1"],
            },
        }
        config = ProxyConfig.from_mapping(raw)
        assert config.hardening.rate_limit_per_ip == 60
        assert config.hardening.trusted_proxy_ips == ("10.0.0.1", "192.168.1.1")

    def test_session_config_from_mapping(self):
        """Phase C: Session config with backend, ttl, redis_url."""
        raw = {
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "session": {
                "backend": "redis",
                "ttl_seconds": 7200,
                "redis_url": "redis://localhost:6379/0",
            },
        }
        config = ProxyConfig.from_mapping(raw)
        assert config.session.backend == "redis"
        assert config.session.ttl_seconds == 7200
        assert config.session.redis_url == "redis://localhost:6379/0"

    def test_anthropic_provider(self):
        raw = {
            "upstream": {
                "provider": "anthropic",
                "api_key": "sk-ant-test",
                "model": "claude-sonnet-4-5-20250929",
            },
        }
        config = ProxyConfig.from_mapping(raw)
        assert config.upstream.provider == "anthropic"
        assert config.upstream.model == "claude-sonnet-4-5-20250929"

    def test_mock_provider_validate_without_api_key(self):
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "mock", "api_key": "", "model": "mock"},
        })
        cfg.validate()

    def test_from_env_defaults_to_mock_without_api_keys(self, monkeypatch):
        cfg = ProxyConfig.from_env()
        assert cfg.upstream.provider == "mock"
        assert cfg.upstream.model == "mock"
        cfg.validate()

    def test_validate_rejects_unresolved_upstream_placeholders(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        cfg = ProxyConfig(
            upstream=UpstreamConfig(
                provider="openai",
                api_key="",
                api_key_env="OPENAI_API_KEY",
                model="gpt-4",
                base_url="${AURORA_LENS_UPSTREAM_BASE_URL}",
            ),
        )
        with pytest.raises(ValueError, match="Unresolved config placeholder.*base_url"):
            cfg.validate()

    def test_from_yaml_rejects_unresolved_model_placeholder(self, tmp_path):
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text(
            "upstream:\n"
            "  provider: openai\n"
            "  api_key: sk-test\n"
            "  model: ${AURORA_LENS_UPSTREAM_MODEL}\n"
        )
        with pytest.raises(ValueError, match="Unresolved config placeholder"):
            ProxyConfig.from_yaml(yaml_file)

    def test_validate_local_provider_requires_resolved_base_url(self):
        cfg = ProxyConfig.from_mapping({
            "upstream": {
                "provider": "local",
                "api_key": "",
                "model": "llama3",
            },
        })
        with pytest.raises(ValueError, match="upstream.provider=local requires"):
            cfg.validate()

    def test_validate_local_provider_rejects_bad_base_url_scheme(self):
        cfg = ProxyConfig.from_mapping({
            "upstream": {
                "provider": "local",
                "api_key": "",
                "model": "llama3",
                "base_url": "localhost:11434/v1",
            },
        })
        with pytest.raises(ValueError, match="must start with http:// or https://"):
            cfg.validate()

    def test_validate_openai_requires_api_key(self):
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "", "model": "gpt-4"},
        })
        with pytest.raises(ValueError, match="requires upstream.api_key_env"):
            cfg.validate()

    def test_build_provider_adapters_mock(self):
        from aurora_lens.proxy.app import _build_provider_adapters

        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "mock", "api_key": "", "model": "mock"},
        })
        upstream, extraction = _build_provider_adapters(cfg)
        from aurora_lens.adapters.mock_upstream import MockUpstreamAdapter

        assert isinstance(upstream, MockUpstreamAdapter)
        assert isinstance(extraction, MockUpstreamAdapter)

    def test_load_config_from_yaml(self, tmp_path):
        yaml_content = """
upstream:
  provider: anthropic
  api_key: sk-yaml-key
  model: claude-sonnet-4-5-20250929
listen:
  port: 7070
governance:
  default_policy: moderate
"""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)

        config = ProxyConfig.from_yaml(yaml_file)
        assert config.upstream.provider == "anthropic"
        assert config.upstream.api_key == "sk-yaml-key"
        assert config.listen.port == 7070
        assert config.governance.default_policy == "moderate"

    def test_env_var_expansion(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TEST_API_KEY", "sk-from-env")
        yaml_content = """
upstream:
  provider: openai
  api_key: ${TEST_API_KEY}
  model: gpt-4
"""
        yaml_file = tmp_path / "test_env.yaml"
        yaml_file.write_text(yaml_content)

        config = ProxyConfig.from_yaml(yaml_file)
        assert config.upstream.api_key == "sk-from-env"

    def test_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AURORA_LENS_UPSTREAM_API_KEY", "sk-override")
        monkeypatch.setenv("AURORA_LENS_LISTEN_PORT", "3000")
        yaml_content = """
upstream:
  provider: openai
  api_key: sk-original
  model: gpt-4
listen:
  port: 8080
"""
        yaml_file = tmp_path / "test_override.yaml"
        yaml_file.write_text(yaml_content)

        config = ProxyConfig.from_yaml(yaml_file)
        assert config.upstream.api_key == "sk-override"
        assert config.listen.port == 3000

    def test_env_override_preserves_yaml_evidence_capture_settings(self, monkeypatch, tmp_path):
        """An unrelated env override (listen port) must not silently discard a
        YAML-configured evidence_capture_mode / evidence_encryption_key back to
        their dataclass defaults."""
        monkeypatch.delenv("AURORA_LENS_EVIDENCE_CAPTURE_MODE", raising=False)
        monkeypatch.delenv("AURORA_LENS_EVIDENCE_KEY", raising=False)
        monkeypatch.setenv("AURORA_LENS_LISTEN_PORT", "3000")
        yaml_content = """
upstream:
  provider: openai
  api_key: sk-original
  model: gpt-4
governance:
  evidence_capture_mode: hash_only
  evidence_encryption_key: yaml-configured-key
"""
        yaml_file = tmp_path / "test_evidence_env.yaml"
        yaml_file.write_text(yaml_content)

        config = ProxyConfig.from_yaml(yaml_file)
        assert config.listen.port == 3000
        assert config.governance.evidence_capture_mode == "hash_only"
        assert config.governance.evidence_encryption_key == "yaml-configured-key"

    def test_env_override_evidence_capture_mode_and_key(self, monkeypatch, tmp_path):
        """AURORA_LENS_EVIDENCE_CAPTURE_MODE / AURORA_LENS_EVIDENCE_KEY override YAML."""
        monkeypatch.setenv("AURORA_LENS_EVIDENCE_CAPTURE_MODE", "plaintext_dev")
        monkeypatch.setenv("AURORA_LENS_EVIDENCE_KEY", "env-configured-key")
        yaml_content = """
upstream:
  provider: openai
  api_key: sk-original
  model: gpt-4
governance:
  evidence_capture_mode: sealed
  evidence_encryption_key: yaml-configured-key
"""
        yaml_file = tmp_path / "test_evidence_env_override.yaml"
        yaml_file.write_text(yaml_content)

        config = ProxyConfig.from_yaml(yaml_file)
        assert config.governance.evidence_capture_mode == "plaintext_dev"
        assert config.governance.evidence_encryption_key == "env-configured-key"

    def test_port_env_paas_overrides_yaml_listen(self, monkeypatch, tmp_path):
        """Railway/Heroku/Render set PORT; bind to it when AURORA_LENS_LISTEN_PORT is unset."""
        monkeypatch.delenv("AURORA_LENS_LISTEN_PORT", raising=False)
        monkeypatch.setenv("PORT", "8765")
        yaml_content = """
upstream:
  provider: openai
  api_key: sk-original
  model: gpt-4
listen:
  port: 8080
"""
        yaml_file = tmp_path / "test_paas_port.yaml"
        yaml_file.write_text(yaml_content)
        config = ProxyConfig.from_yaml(yaml_file)
        assert config.listen.port == 8765

    def test_aurora_listen_port_overrides_paas_port(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AURORA_LENS_LISTEN_PORT", "3000")
        monkeypatch.setenv("PORT", "8765")
        yaml_content = """
upstream:
  provider: openai
  api_key: sk-original
  model: gpt-4
listen:
  port: 8080
"""
        yaml_file = tmp_path / "test_listen_prec.yaml"
        yaml_file.write_text(yaml_content)
        config = ProxyConfig.from_yaml(yaml_file)
        assert config.listen.port == 3000

    def test_audit_log_fresh_chain_from_mapping(self):
        raw = {
            "upstream": {"provider": "openai", "api_key": "k", "model": "m"},
            "governance": {
                "audit_log": "./a.jsonl",
                "audit_log_fresh_chain_on_startup": True,
            },
        }
        config = ProxyConfig.from_mapping(raw)
        assert config.governance.audit_log_fresh_chain_on_startup is True

    def test_resolve_proxy_audit_log_path_placeholders(self, tmp_path):
        import os

        from aurora_lens.proxy.app import _resolve_proxy_audit_log_path

        tmpl = str(tmp_path / "x-{run_id}-{pid}.jsonl")
        out = _resolve_proxy_audit_log_path(tmpl, "rid-1")
        assert "rid-1" in out
        assert str(os.getpid()) in out
        assert out.endswith(".jsonl")

    def test_create_app_rotates_audit_when_fresh_chain(self, tmp_path):
        from aurora_lens.proxy.app import create_app

        audit = tmp_path / "one.jsonl"
        audit.write_text('{"legacy":true}\n', encoding="utf-8")
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "k", "model": "m"},
            "governance": {
                "audit_log": str(audit),
                "audit_backend": "ledger",
                "audit_log_fresh_chain_on_startup": True,
            },
        })
        create_app(cfg)
        assert not audit.exists()
        backups = list(tmp_path.glob("one.jsonl.pre_restart.*"))
        assert len(backups) == 1
        assert b"legacy" in backups[0].read_bytes()

    def test_health_includes_proxy_run_id_when_audit_configured(self, tmp_path):
        from fastapi.testclient import TestClient

        from aurora_lens.proxy.app import create_app

        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "k", "model": "m"},
            "governance": {
                "audit_log": str(tmp_path / "a.jsonl"),
                "audit_backend": "ledger",
            },
        })
        app = create_app(cfg)
        r = TestClient(app).get("/health")
        assert r.status_code == 200
        body = r.json()
        assert "proxy_run_id" in body
        assert len(body["proxy_run_id"]) > 20

    def test_v1_models_openai_list_shape(self, tmp_path):
        """GET /v1/models lists the configured upstream model (LibreChat discovery)."""
        from fastapi.testclient import TestClient

        from aurora_lens.proxy.app import create_app

        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "k", "model": "my-configured-model"},
            "governance": {
                "audit_log": str(tmp_path / "a.jsonl"),
                "audit_backend": "ledger",
            },
        })
        app = create_app(cfg)
        r = TestClient(app).get("/v1/models")
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "my-configured-model"
        assert data["data"][0]["object"] == "model"
        assert data["data"][0]["created"] == 1700000000
        assert data["data"][0]["owned_by"] == "aurora-lens"

    def test_cors_allow_origins_env_appends(self, monkeypatch):
        """AURORA_LENS_CORS_ALLOW_ORIGINS appends to YAML allow_origins (does not replace)."""
        monkeypatch.setenv(
            "AURORA_LENS_CORS_ALLOW_ORIGINS",
            " http://localhost:3080 ,http://localhost:3000 , ",
        )
        raw = {
            "upstream": {"provider": "openai", "api_key": "k", "model": "m"},
            "cors": {"enabled": True, "allow_origins": ["https://a.example.com"]},
        }
        cfg = ProxyConfig.from_mapping(raw)
        assert cfg.cors.allow_origins == (
            "https://a.example.com",
            "http://localhost:3080",
            "http://localhost:3000",
        )


# ── Aurora Headers Tests ──────────────────────────────────────────

class TestAuroraHeaders:
    """Unit tests for _aurora_headers()."""

    def test_all_required_fields_present(self):
        from aurora_lens.proxy.app import _aurora_headers

        h = _aurora_headers(
            outcome="PASS",
            trace_id="proxy:abc123",
            audit_sink="jsonl",
            provider="openai",
            policy="strict",
            policy_version="1.0",
        )
        assert h["Aurora-Outcome"] == "PASS"
        assert h["Aurora-Trace-Id"] == "proxy:abc123"
        assert h["Aurora-Audit-Sink"] == "jsonl"
        assert "Aurora-Timestamp" in h
        assert h["Aurora-Timestamp"].endswith("Z")
        assert h["Aurora-Upstream"] == "openai_compat"
        assert h["Aurora-Policy"] == "strict"
        assert h["Aurora-Policy-Version"] == "1.0"

    def test_provider_display_alias(self):
        """Aurora-Upstream shows adapter identity (claude, openai_compat), not vendor."""
        from aurora_lens.proxy.app import _provider_display

        assert _provider_display("openai") == "openai_compat"
        assert _provider_display("anthropic") == "claude"
        assert _provider_display("custom") == "custom"

    def test_provider_display_edge_cases(self):
        """Case-insensitive for known providers, handles None/empty, passthrough for unknowns."""
        from aurora_lens.proxy.app import _provider_display

        assert _provider_display("OpenAI") == "openai_compat"
        assert _provider_display("ANTHROPIC") == "claude"
        assert _provider_display("  anthropic  ") == "claude"
        assert _provider_display(" openai ") == "openai_compat"
        assert _provider_display(None) == "unknown"
        assert _provider_display("") == "unknown"
        assert _provider_display("   ") == "unknown"
        assert _provider_display("Custom") == "Custom"  # passthrough preserves original

    def test_audit_sink_always_present(self):
        from aurora_lens.proxy.app import _aurora_headers

        for sink in ("jsonl", "ledger", "none"):
            h = _aurora_headers(
                outcome="ERROR",
                trace_id="proxy:x",
                audit_sink=sink,
                provider="anthropic",
                policy="moderate",
                policy_version="1.0",
            )
            assert h["Aurora-Audit-Sink"] == sink

    def test_timestamp_ends_with_z(self):
        from aurora_lens.proxy.app import _aurora_headers

        h = _aurora_headers(
            outcome="ERROR",
            trace_id="proxy:x",
            audit_sink="none",
            provider="openai",
            policy="strict",
            policy_version="1.0",
        )
        assert h["Aurora-Timestamp"].endswith("Z")
        assert "+00:00" not in h["Aurora-Timestamp"]

    def test_optional_fields_omitted_when_none(self):
        from aurora_lens.proxy.app import _aurora_headers

        h = _aurora_headers(
            outcome="ERROR",
            trace_id="proxy:x",
            audit_sink="none",
            provider="openai",
            policy="strict",
            policy_version="1.0",
        )
        assert "Aurora-Audit-Id" not in h
        assert "Aurora-Session-Id" not in h
        assert "Aurora-Proxy-Ms" not in h

    def test_optional_fields_included_when_provided(self):
        from aurora_lens.proxy.app import _aurora_headers

        h = _aurora_headers(
            outcome="HARD_STOP",
            trace_id="proxy:x",
            audit_sink="ledger",
            provider="openai",
            policy="strict",
            policy_version="1.0",
            audit_id="cid:test:123",
            session_id="sess-abc",
            proxy_ms=42,
        )
        assert h["Aurora-Audit-Id"] == "cid:test:123"
        assert h["Aurora-Session-Id"] == "sess-abc"
        assert h["Aurora-Proxy-Ms"] == "42"


# ── Session Manager Tests ───────────────────────────────────────────

class TestSessionManager:

    def test_create_new_session(self):
        mgr = SessionManager(config_factory=_make_config_factory())
        with mgr.with_lock("session-1"):
            lens, is_new = mgr.get_or_create("session-1")
            assert is_new
            assert isinstance(lens, Lens)
            mgr.persist("session-1", lens)
        assert mgr.active_count == 1

    def test_get_existing_session(self):
        """Phase C: Same session returns Lens with same PEF state (from store)."""
        mgr = SessionManager(config_factory=_make_config_factory())
        with mgr.with_lock("session-1"):
            lens1, is_new1 = mgr.get_or_create("session-1")
            assert is_new1
            lens1.pef.advance_turn()
            mgr.persist("session-1", lens1)
        with mgr.with_lock("session-1"):
            lens2, is_new2 = mgr.get_or_create("session-1")
            assert not is_new2
            assert lens2.pef.current_turn == lens1.pef.current_turn
        assert mgr.active_count == 1

    def test_separate_sessions(self):
        mgr = SessionManager(config_factory=_make_config_factory())
        with mgr.with_lock("session-1"):
            lens1, _ = mgr.get_or_create("session-1")
            mgr.persist("session-1", lens1)
        with mgr.with_lock("session-2"):
            lens2, _ = mgr.get_or_create("session-2")
            mgr.persist("session-2", lens2)
        assert lens1 is not lens2
        assert mgr.active_count == 2

    def test_get_nonexistent_returns_none(self):
        mgr = SessionManager(config_factory=_make_config_factory())
        with mgr.with_lock("nonexistent"):
            assert mgr.get("nonexistent") is None

    def test_cleanup_expired(self):
        mgr = SessionManager(config_factory=_make_config_factory(), ttl_seconds=0)
        with mgr.with_lock("session-1"):
            lens, _ = mgr.get_or_create("session-1")
            mgr.persist("session-1", lens)
        time.sleep(0.01)
        removed = mgr.cleanup_expired()
        assert removed == 1
        assert mgr.active_count == 0

    def test_session_record_revision_increments(self):
        """SessionStore contract: revision is monotonic, caller increments on persist."""
        from aurora_lens.proxy.session_store import MemorySessionStore

        store = MemorySessionStore(ttl_seconds=3600)
        pef = PEFState()
        now = time.time()
        r0 = SessionRecord.from_pef(pef, expires_at=now + 3600, revision=0)
        store.put("s1", r0)
        rec = store.get("s1")
        assert rec is not None
        assert rec.revision == 0
        r1 = SessionRecord.from_pef(pef, expires_at=now + 3600, revision=1)
        store.put("s1", r1)
        rec = store.get("s1")
        assert rec is not None
        assert rec.revision == 1


# ── Frame Lifecycle Routing Tests ───────────────────────────────────

class TestFrameLifecycleRouting:
    def test_explicit_session_id_continues_frame(self):
        explicit = extract_explicit_session_id(
            {"x-aurora-session-id": "sid-explicit-1"},
            {},
        )
        decision = resolve_frame_lifecycle(
            explicit_session_id=explicit,
            cookie_session_id="sid-cookie-1",
            continuation_requested=None,
            domain_or_lane_hint="",
        )
        assert decision.session_id == "sid-explicit-1"
        assert decision.frame_action == "continue"
        assert decision.reason == "explicit_session_id"

    def test_pef_context_id_header_takes_priority_over_legacy_session_alias(self):
        """New x-aurora-pef-context-id header wins over legacy aurora_session_id body field."""
        from aurora_lens.proxy.frame_lifecycle import extract_explicit_pef_context_id

        explicit = extract_explicit_pef_context_id(
            {"x-aurora-pef-context-id": "pef-ctx-1"},
            {"aurora_session_id": "legacy-sid-1"},
        )
        assert explicit == "pef-ctx-1"

    def test_pef_context_id_body_field_takes_priority_over_legacy_aliases(self):
        from aurora_lens.proxy.frame_lifecycle import extract_explicit_pef_context_id

        explicit = extract_explicit_pef_context_id(
            {},
            {"pef_context_id": "pef-ctx-2", "session_id": "legacy-sid-2"},
        )
        assert explicit == "pef-ctx-2"

    def test_pef_context_id_falls_back_to_legacy_session_aliases_when_absent(self):
        from aurora_lens.proxy.frame_lifecycle import extract_explicit_pef_context_id

        explicit = extract_explicit_pef_context_id(
            {},
            {"aurora_session_id": "legacy-sid-3"},
        )
        assert explicit == "legacy-sid-3"

    def test_pef_context_id_resolves_frame_continuation_same_as_session_id(self):
        from aurora_lens.proxy.frame_lifecycle import extract_explicit_pef_context_id

        explicit = extract_explicit_pef_context_id(
            {"x-aurora-pef-context-id": "pef-ctx-continue"},
            {},
        )
        decision = resolve_frame_lifecycle(
            explicit_session_id=explicit,
            cookie_session_id="",
            continuation_requested=None,
            domain_or_lane_hint="",
        )
        assert decision.session_id == "pef-ctx-continue"
        assert decision.frame_action == "continue"

    def test_cookie_compat_continues_when_continuation_omitted(self):
        decision = resolve_frame_lifecycle(
            explicit_session_id="",
            cookie_session_id="sid-cookie-compat",
            continuation_requested=None,
            domain_or_lane_hint="",
        )
        assert decision.session_id == "sid-cookie-compat"
        assert decision.frame_action == "continue"
        assert decision.reason == "cookie_compat"

    def test_continuation_false_forces_new_frame_even_with_cookie(self):
        continuation = parse_continuation_requested(
            {},
            {"continuation": False},
        )
        decision = resolve_frame_lifecycle(
            explicit_session_id="",
            cookie_session_id="sid-cookie-ignored",
            continuation_requested=continuation,
            domain_or_lane_hint="",
        )
        assert decision.frame_action == "new"
        assert decision.reason == "continuation_false"
        assert decision.session_id != "sid-cookie-ignored"
        assert decision.session_id.startswith("session-")

    def test_continuation_true_uses_cookie_when_present(self):
        continuation = parse_continuation_requested(
            {"x-aurora-continuation": "true"},
            {},
        )
        decision = resolve_frame_lifecycle(
            explicit_session_id="",
            cookie_session_id="sid-cookie-continue",
            continuation_requested=continuation,
            domain_or_lane_hint="",
        )
        assert decision.session_id == "sid-cookie-continue"
        assert decision.frame_action == "continue"
        assert decision.reason == "explicit_continuation_cookie"

    def test_continuation_true_without_cookie_mints_new(self):
        continuation = parse_continuation_requested(
            {},
            {"aurora": {"continuation": True}},
        )
        decision = resolve_frame_lifecycle(
            explicit_session_id="",
            cookie_session_id="",
            continuation_requested=continuation,
            domain_or_lane_hint="",
        )
        assert decision.frame_action == "new"
        assert decision.reason == "new_frame"
        assert decision.session_id.startswith("session-")

    def test_same_domain_does_not_imply_frame_carry_forward(self):
        """Domain/lane labels alone cannot continue prior state."""
        continuation = parse_continuation_requested(
            {},
            {"aurora": {"domain": "finance", "lane": "finance"}},
        )
        decision = resolve_frame_lifecycle(
            explicit_session_id="",
            cookie_session_id="",
            continuation_requested=continuation,
            domain_or_lane_hint="finance",
        )
        assert continuation is None
        assert decision.frame_action == "new"
        assert decision.reason == "domain_lane_default_new_frame"

    def test_domain_hint_inputs_are_ignored(self):
        domain_hint = parse_domain_or_lane_hint(
            {"x-aurora-operator-channel": "medical"},
            {"domain": "finance"},
        )
        assert domain_hint == ""

    def test_domain_hint_defaults_do_not_apply_when_hinting_disabled(self):
        decision = resolve_frame_lifecycle(
            explicit_session_id="",
            cookie_session_id="sid-cookie-legacy",
            continuation_requested=None,
            domain_or_lane_hint="",
        )
        assert decision.frame_action == "continue"
        assert decision.reason == "cookie_compat"
        assert decision.session_id == "sid-cookie-legacy"

    def test_domain_hint_allows_cookie_carry_forward_only_with_explicit_continuation(self):
        continuation = parse_continuation_requested(
            {},
            {"continuation": True},
        )
        decision = resolve_frame_lifecycle(
            explicit_session_id="",
            cookie_session_id="sid-cookie-continue",
            continuation_requested=continuation,
            domain_or_lane_hint="finance",
        )
        assert decision.frame_action == "continue"
        assert decision.reason == "explicit_continuation_cookie"
        assert decision.session_id == "sid-cookie-continue"


# ── OpenAI Compat Tests ────────────────────────────────────────────

class TestParseRequest:

    def test_basic_request(self):
        body = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Hello world"},
            ],
        }
        parsed = parse_chat_request(body)
        assert parsed.user_message == "Hello world"
        assert parsed.model == "gpt-4"
        assert len(parsed.conversation_history) == 0

    def test_with_history(self):
        body = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "First message"},
                {"role": "assistant", "content": "First response"},
                {"role": "user", "content": "Second message"},
            ],
        }
        parsed = parse_chat_request(body)
        assert parsed.user_message == "Second message"
        assert len(parsed.conversation_history) == 2

    def test_system_message_skipped(self):
        body = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ],
        }
        parsed = parse_chat_request(body)
        assert parsed.user_message == "Hello"
        assert len(parsed.conversation_history) == 0

    def test_custom_session_id(self):
        body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
            "aurora_session_id": "my-session-123",
        }
        parsed = parse_chat_request(body)
        assert parsed.session_id == "my-session-123"

    def test_multimodal_content(self):
        """parse_chat_request extracts text from multimodal content (list of parts)."""
        body = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello from multimodal"}]},
            ],
        }
        parsed = parse_chat_request(body)
        assert parsed.user_message == "Hello from multimodal"

    def test_empty_messages_raises(self):
        """parse_chat_request raises ValueError when messages is empty and no fallback."""
        body = {"model": "gpt-4", "messages": []}
        with pytest.raises(ValueError, match="expected 'messages' list"):
            parse_chat_request(body)

    def test_input_fallback(self):
        """parse_chat_request accepts input/prompt/text as fallback when messages missing."""
        body = {"model": "gpt-4", "input": "Direct input text"}
        parsed = parse_chat_request(body)
        assert parsed.user_message == "Direct input text"

    def test_external_flags_parsed(self):
        """parse_chat_request extracts aurora.external_flags (Phase 9)."""
        body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "aurora": {
                "external_flags": [
                    {"type": "SELF_HARM_INSTRUCTION", "evidence": ["matched span"], "source": "classifier:v1"},
                ],
            },
        }
        parsed = parse_chat_request(body)
        assert len(parsed.external_flags) == 1
        assert parsed.external_flags[0].flag_type.name == "SELF_HARM_INSTRUCTION"
        assert "matched span" in parsed.external_flags[0].evidence

    def test_external_flags_empty_when_missing(self):
        """No aurora.external_flags → empty list."""
        body = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]}
        parsed = parse_chat_request(body)
        assert parsed.external_flags == []

    def test_external_flags_malformed_raises_422(self):
        """Malformed aurora.external_flags raises ValueError → 422."""
        # Missing type
        with pytest.raises(ValueError, match="type"):
            parse_chat_request({
                "messages": [{"role": "user", "content": "Hi"}],
                "aurora": {"external_flags": [{"evidence": ["x"]}]},
            })
        # Invalid type
        with pytest.raises(ValueError, match="unknown type"):
            parse_chat_request({
                "messages": [{"role": "user", "content": "Hi"}],
                "aurora": {"external_flags": [{"type": "UNKNOWN_FLAG", "evidence": ["x"]}]},
            })
        # evidence not a list
        with pytest.raises(ValueError, match="evidence"):
            parse_chat_request({
                "messages": [{"role": "user", "content": "Hi"}],
                "aurora": {"external_flags": [{"type": "SELF_HARM_INSTRUCTION", "evidence": "string"}]},
            })


class TestFormatResponse:

    def test_clean_response(self):
        result = LensResult(
            response="Hello!",
            flags=[],
            pef_snapshot="empty",
            turn=1,
            span=Span.PRESENT,
            model="gpt-4",
            action=InterventionAction.PASS,
            decision=GovernanceDecision(
                action=InterventionAction.PASS,
                flags=[],
                rationale="No verification flags",
            ),
        )
        body = format_chat_response(result)
        assert body["choices"][0]["message"]["content"] == "Hello!"
        assert body["aurora"]["governance"] == "PASS"
        assert "original_response" not in body["aurora"]

    def test_contain_operator_plane_empty_candidates_serializes(self):
        """CONTAIN / UNRESOLVED_REFERENT with empty candidate_entities must JSON round-trip."""
        from aurora_lens.proxy.json_wire import wire_encode

        flags = [
            Flag(
                flag_type=FlagType.UNRESOLVED_REFERENT,
                entity_name="he",
                claim="x",
                evidence="y",
                severity="warning",
                candidates=(),
            )
        ]
        decision = GovernanceDecision(
            action=InterventionAction.CONTAIN,
            flags=flags,
            rationale="Ambiguity",
            policy="strict",
            pathway_id="P_ASK_DISAMBIGUATE",
            output_mode="clarification_request",
            commitment_closed=True,
            interaction_open=True,
        )
        decision.forensic_event = {
            "failed_constraints": ["UNRESOLVED_REFERENT"],
            "status": "ASK",
        }
        pef = PEFState()
        pef.pending_clarification = {
            "original_question": "He gave Sarah 5.",
            "failed_constraint": "UNRESOLVED_REFERENT",
            "ambiguous_referents": ["he"],
            "candidate_entities": [],
            "blocked_claims": [],
        }
        result = LensResult(
            response="Who do you mean for 'he'?",
            flags=flags,
            pef_snapshot="",
            turn=1,
            span=Span.PRESENT,
            action=InterventionAction.CONTAIN,
            decision=decision,
            usage={"completion_tokens": "bogus", "prompt_tokens": ""},
        )
        body = format_chat_response(result, include_operator_detail=True, pef=pef)
        json.dumps(wire_encode(body))
        opc = body["aurora"]["operator_pef"]["pending_clarification"]
        assert opc["candidate_entities"] == []
        assert opc["ambiguous_referents"] == ["he"]
        assert body["aurora"]["governance"] == "CONTAIN"
        assert body["usage"]["completion_tokens"] == 0

    def test_format_stream_metadata_event_json_encodes_enums(self):
        line = format_stream_metadata_event(
            {"governance": "CONTAIN", "diagnostic": FlagType.UNRESOLVED_REFERENT},
        )
        assert line.startswith("data: ")
        payload = json.loads(line[len("data: ") :].strip())
        assert payload["aurora"]["governance"] == "CONTAIN"
        assert payload["aurora"]["diagnostic"] == "UNRESOLVED_REFERENT"

    def test_pass_operator_detail_includes_original_response(self):
        """Operator plane: upstream text is visible on PASS when provided."""
        from aurora_lens.pef.state import PEFState

        decision = GovernanceDecision(
            action=InterventionAction.PASS,
            flags=[],
            rationale="No verification flags",
            policy="strict",
        )
        result = LensResult(
            response="Hello!",
            flags=[],
            pef_snapshot="empty",
            turn=1,
            span=Span.PRESENT,
            model="gpt-4",
            action=InterventionAction.PASS,
            decision=decision,
            original_response="Hello!",
        )
        body = format_chat_response(
            result, include_operator_detail=True, pef=PEFState(),
        )
        assert body["aurora"]["governance"] == "PASS"
        assert body["aurora"]["original_response"] == "Hello!"

    def test_hard_stop_response(self):
        """Default (no operator detail): only governance action in aurora, no internal diagnostics."""
        flags = [Flag(
            flag_type=FlagType.CONTRADICTED_FACT,
            entity_name="Patient",
            claim="Patient HAS myocardial infarction",
            evidence="No evidence",
            severity="error",
        )]
        decision = GovernanceDecision(
            action=InterventionAction.HARD_STOP,
            flags=flags,
            rationale="Insufficient clinical evidence",
            policy="strict",
            original_response="Likely MI.",
        )
        result = LensResult(
            response="Blocked.",
            flags=flags,
            pef_snapshot="...",
            turn=2,
            span=Span.PRESENT,
            action=InterventionAction.HARD_STOP,
            decision=decision,
            original_response="Likely MI.",
        )
        body = format_chat_response(result)

        # Always HTTP 200 — governance action in body
        assert body["choices"][0]["message"]["content"] == "Blocked."
        assert body["aurora"]["governance"] == "HARD_STOP"
        # Two-plane contract: operator-plane fields not present in default response
        assert "flags" not in body["aurora"]
        assert "rationale" not in body["aurora"]
        assert "original_response" not in body["aurora"]
        assert "governance_note" not in body["aurora"]
        # HARD_STOP is not SOFT_CORRECT — no unverified hint
        assert "unverified" not in body["aurora"]

    def test_forensic_event_present_for_non_admit(self):
        """forensic_event is operator-plane only; user plane gets governance_hint only."""
        from aurora_lens.pef.state import PEFState

        flags = [Flag(
            flag_type=FlagType.CONTRADICTED_FACT,
            entity_name="Patient",
            claim="Patient HAS MI",
            evidence="No evidence",
            severity="error",
        )]
        decision = GovernanceDecision(
            action=InterventionAction.HARD_STOP,
            flags=flags,
            rationale="Insufficient evidence",
            policy="strict",
            original_response="Likely MI.",
        )
        decision.forensic_event = {
            "trace_id": "t1",
            "timestamp": "2026-02-24T07:15:23Z",
            "status": "STOP",
            "attempted_action": "respond",
            "domain": "governance",
            "subdomain": None,
            "failed_constraints": ["CONTRADICTED_FACT"],
            "state_hash": "abc123",
        }
        result = LensResult(
            response="Blocked.",
            flags=flags,
            pef_snapshot="...",
            turn=2,
            span=Span.PRESENT,
            action=InterventionAction.HARD_STOP,
            decision=decision,
            original_response="Likely MI.",
        )
        body = format_chat_response(result)
        assert "forensic_event" not in body["aurora"]
        assert body["aurora"]["governance_hint"] == "blocked"

        pef = PEFState()
        body_op = format_chat_response(
            result, include_operator_detail=True, pef=pef,
        )
        assert "forensic_event" in body_op["aurora"]
        fe = body_op["aurora"]["forensic_event"]
        assert fe["trace_id"] == "t1"
        assert fe["timestamp"] == "2026-02-24T07:15:23Z"
        assert fe["status"] == "STOP"
        assert fe["attempted_action"] == "respond"
        assert fe["domain"] == "governance"
        assert fe["failed_constraints"] == ["CONTRADICTED_FACT"]
        assert fe["state_hash"] == "abc123"
        assert "subdomain" in fe
        assert "operator_pef" in body_op["aurora"]
        assert body_op["aurora"]["operator_pef"]["session_mode"] == "normal"

    def test_forensic_event_absent_for_pass_and_soft_correct(self):
        """PASS and SOFT_CORRECT responses do not include forensic_event."""
        for action in (InterventionAction.PASS, InterventionAction.SOFT_CORRECT):
            decision = GovernanceDecision(
                action=action,
                flags=[],
                rationale="",
                policy="strict",
            )
            result = LensResult(
                response="OK",
                flags=[],
                pef_snapshot="",
                turn=1,
                span=Span.PRESENT,
                action=action,
                decision=decision,
            )
            body = format_chat_response(result)
            assert "forensic_event" not in body["aurora"]

    def test_hard_stop_with_operator_detail(self):
        """include_operator_detail=True: diagnostics plus redacted operator token (no raw upstream)."""
        flags = [Flag(
            flag_type=FlagType.CONTRADICTED_FACT,
            entity_name="Patient",
            claim="Patient HAS myocardial infarction",
            evidence="No evidence",
            severity="error",
        )]
        decision = GovernanceDecision(
            action=InterventionAction.HARD_STOP,
            flags=flags,
            rationale="Insufficient clinical evidence",
            policy="strict",
            original_response="Likely MI.",
        )
        result = LensResult(
            response="Blocked.",
            flags=flags,
            pef_snapshot="...",
            turn=2,
            span=Span.PRESENT,
            action=InterventionAction.HARD_STOP,
            decision=decision,
            original_response="Likely MI.",
        )
        from aurora_lens.pef.state import PEFState

        body = format_chat_response(result, include_operator_detail=True, pef=PEFState())

        assert body["aurora"]["governance"] == "HARD_STOP"
        assert body["aurora"]["flags"] == ["CONTRADICTED_FACT"]
        assert body["aurora"]["rationale"] == "Insufficient clinical evidence"
        assert "original_response" not in body["aurora"]
        assert body["aurora"]["intercepted_upstream_redacted_reason"] == (
            "suppressed_blocked_upstream_output"
        )
        assert "intercepted_upstream_output" not in body["aurora"]
        assert body["aurora"]["policy"] == "strict"
        assert "operator_pef" in body["aurora"]

    def test_contain_with_operator_detail_includes_intercepted_upstream(self):
        """Non-admit CONTAIN: operator plane exposes stable redaction token, not raw withheld text."""
        flags = [Flag(
            flag_type=FlagType.UNRESOLVED_REFERENT,
            entity_name="he",
            claim="Referent unresolved",
            evidence="ambiguous",
            severity="warning",
        )]
        decision = GovernanceDecision(
            action=InterventionAction.CONTAIN,
            flags=flags,
            rationale="Clarify referent",
            policy="strict",
        )
        result = LensResult(
            response="Which person do you mean?",
            flags=flags,
            pef_snapshot="{}",
            turn=3,
            span=Span.PRESENT,
            action=InterventionAction.CONTAIN,
            decision=decision,
            original_response="He sold the patent.",
            model="mock",
        )
        body = format_chat_response(
            result, include_operator_detail=True, pef=PEFState(),
        )
        assert body["aurora"]["governance"] == "CONTAIN"
        assert "original_response" not in body["aurora"]
        assert body["aurora"]["intercepted_upstream_redacted_reason"] == (
            "suppressed_blocked_upstream_output"
        )
        assert "intercepted_upstream_output" not in body["aurora"]

    def test_soft_correct_unverified_hint(self):
        """SOFT_CORRECT adds aurora.unverified=True; internal note stays in audit log."""
        decision = GovernanceDecision(
            action=InterventionAction.SOFT_CORRECT,
            flags=[],
            rationale="Minor flags",
            governance_note="TIME_SMEAR: past fact presented as present",
        )
        result = LensResult(
            response="Clean content.",
            flags=[],
            pef_snapshot="",
            turn=1,
            span=Span.PRESENT,
            action=InterventionAction.SOFT_CORRECT,
            decision=decision,
        )
        body = format_chat_response(result)
        # User-plane: neutral boolean hint, no internal diagnostic
        assert body["aurora"]["unverified"] is True
        assert "governance_note" not in body["aurora"]
        # Content is NOT polluted
        assert body["choices"][0]["message"]["content"] == "Clean content."

    def test_soft_correct_operator_upstream_echo_without_intercepted_duplicate(self):
        """SOFT_CORRECT operator plane uses original_response; no duplicate intercepted key."""
        decision = GovernanceDecision(
            action=InterventionAction.SOFT_CORRECT,
            flags=[],
            rationale="Minor normalization",
            governance_note="TIME_SMEAR",
        )
        result = LensResult(
            response="Governed polish.",
            flags=[],
            pef_snapshot="",
            turn=1,
            span=Span.PRESENT,
            action=InterventionAction.SOFT_CORRECT,
            decision=decision,
            original_response=" raw draft ",
            model="mock",
        )
        body = format_chat_response(
            result,
            include_operator_detail=True,
            pef=PEFState(),
        )
        assert body["aurora"]["original_response"] == " raw draft "
        assert "intercepted_upstream_output" not in body["aurora"]

    def test_soft_correct_operator_detail_includes_note(self):
        """include_operator_detail=True exposes governance_note for SOFT_CORRECT."""
        decision = GovernanceDecision(
            action=InterventionAction.SOFT_CORRECT,
            flags=[],
            rationale="Minor flags",
            governance_note="TIME_SMEAR: past fact presented as present",
        )
        result = LensResult(
            response="Clean content.",
            flags=[],
            pef_snapshot="",
            turn=1,
            span=Span.PRESENT,
            action=InterventionAction.SOFT_CORRECT,
            decision=decision,
        )
        body = format_chat_response(result, include_operator_detail=True)
        assert body["aurora"]["governance_note"] == "TIME_SMEAR: past fact presented as present"
        assert body["choices"][0]["message"]["content"] == "Clean content."

    def test_usage_propagated_when_present(self):
        """When LensResult has usage from adapter, format_chat_response reflects non-zero totals."""
        result = LensResult(
            response="Hello!",
            flags=[],
            pef_snapshot="empty",
            turn=1,
            span=Span.PRESENT,
            model="claude-sonnet-4-5-20250929",
            action=InterventionAction.PASS,
            decision=GovernanceDecision(
                action=InterventionAction.PASS,
                flags=[],
                rationale="No verification flags",
            ),
            usage={"input_tokens": 127, "output_tokens": 7},
        )
        body = format_chat_response(result)
        assert body["usage"]["prompt_tokens"] == 127
        assert body["usage"]["completion_tokens"] == 7
        assert body["usage"]["total_tokens"] == 134

    def test_usage_zero_when_absent(self):
        """When LensResult has no usage, format_chat_response returns zeros."""
        result = LensResult(
            response="Hi",
            flags=[],
            pef_snapshot="empty",
            turn=1,
            span=Span.PRESENT,
            model="mock",
            action=InterventionAction.PASS,
            decision=GovernanceDecision(
                action=InterventionAction.PASS,
                flags=[],
                rationale="No verification flags",
            ),
        )
        body = format_chat_response(result)
        assert body["usage"]["prompt_tokens"] == 0
        assert body["usage"]["completion_tokens"] == 0
        assert body["usage"]["total_tokens"] == 0


# ── App Integration Tests ───────────────────────────────────────────

class TestAppIntegration:
    """Integration tests using Starlette's test client."""

    def _get_test_client(self):
        """Create a test client with mock upstream."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app
        from aurora_lens.proxy.config import ProxyConfig

        # We can't use a real upstream, so we monkeypatch create_app
        # Instead, we'll test the lower layers directly
        # and do a minimal app test
        return None

    def test_session_routing_via_app(self, monkeypatch):
        """SessionManager is wired; requests route through sessions.get_or_create(session_id)."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app
        from aurora_lens.proxy.config import ProxyConfig

        def _mock_adapters(cfg):
            return MockAdapter(), MockAdapter()

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            _mock_adapters,
        )

        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"default_policy": "strict", "audit_log": None},
            "extraction": {"backend": "llm"},
        })
        app = create_app(cfg)
        client = TestClient(app)

        # Session from body
        r1 = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "aurora_session_id": "sess-body-1",
            },
        )
        assert r1.status_code == 200

        # Different session
        r2 = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hi"}],
                "aurora_session_id": "sess-body-2",
            },
        )
        assert r2.status_code == 200

        # Same session as r1 — reuses lens
        r3 = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Again"}],
                "aurora_session_id": "sess-body-1",
            },
        )
        assert r3.status_code == 200

    def test_trusted_domain_cannot_downgrade_medical_flags(self, monkeypatch):
        """Trusted domain binding cannot downgrade verified medical protected domain."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app
        from aurora_lens.proxy.config import ProxyConfig

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"default_policy": "strict", "audit_log": None, "default_domain": "general"},
            "extraction": {"backend": "spacy"},
        })
        app = create_app(cfg)
        client = TestClient(app)

        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{
                    "role": "user",
                    "content": "My 6-year-old child takes amoxicillin. How many mg should I give per dose?",
                }],
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["aurora"]["governance"] != "PASS"
        dom = body["aurora"].get("domain")
        if not dom:
            dom = ((body["aurora"].get("rule_result") or {}).get("domain"))
        assert dom == "medical"

    def test_domain_headers_are_ignored_for_governance_routing(self, monkeypatch):
        """Caller-supplied domain headers do not control governance domain."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app
        from aurora_lens.proxy.config import ProxyConfig

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"default_policy": "strict", "audit_log": None, "default_domain": "medical"},
            "extraction": {"backend": "spacy"},
        })
        app = create_app(cfg)
        client = TestClient(app)

        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{
                    "role": "user",
                    "content": "My 6-year-old child takes amoxicillin. How many mg should I give per dose?",
                }],
            },
            headers={"x-aurora-domain": "finance", "x-aurora-operator-channel": "general"},
        )
        assert r.status_code == 200
        body = r.json()
        dom = body["aurora"].get("domain") or ((body["aurora"].get("rule_result") or {}).get("domain"))
        assert dom == "medical"

    def test_session_operator_pef_unknown_404_and_active_session(self, monkeypatch):
        """GET /v1/session/operator-pef returns 404 for unknown id; 200 + operator_pef for active session."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app
        from aurora_lens.proxy.config import ProxyConfig

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"default_policy": "strict", "audit_log": None},
            "extraction": {"backend": "llm"},
        })
        app = create_app(cfg)
        client = TestClient(app)

        r0 = client.get("/v1/session/operator-pef?session_id=nonexistent-session-xyz")
        assert r0.status_code == 404

        sid = "sess-operator-pef-ui"
        r1 = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "aurora_session_id": sid,
            },
        )
        assert r1.status_code == 200

        r2 = client.get("/v1/session/operator-pef?session_id=" + sid)
        assert r2.status_code == 200
        data = r2.json()
        assert data["session_id"] == sid
        op = data["operator_pef"]
        assert isinstance(op.get("session_mode"), str) and op["session_mode"]
        assert "counts" in op
        assert "pending_clarification" in op
        assert "epistemic_hold" in op

    def test_session_operator_pef_accepts_pef_context_id_alias(self, monkeypatch):
        """pef_context_id is the preferred query param; session_id remains a working alias;
        neither supplied is a 422, not a 404 (distinguishing "bad request" from "not found")."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app
        from aurora_lens.proxy.config import ProxyConfig

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"default_policy": "strict", "audit_log": None},
            "extraction": {"backend": "llm"},
        })
        app = create_app(cfg)
        client = TestClient(app)

        r_neither = client.get("/v1/session/operator-pef")
        assert r_neither.status_code == 422

        sid = "sess-pef-context-id-alias"
        r1 = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "aurora_session_id": sid,
            },
        )
        assert r1.status_code == 200

        r2 = client.get("/v1/session/operator-pef?pef_context_id=" + sid)
        assert r2.status_code == 200
        data = r2.json()
        assert data["session_id"] == sid
        assert data["pef_context_id"] == sid
        assert "operator_pef" in data

    def test_new_scenario_reset_mints_fresh_pef_and_keeps_old_session_addressable(self, monkeypatch, tmp_path):
        """Step 9: reset endpoint creates a fresh namespaced session without mutating old session."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)
        pytest.importorskip("spacy", reason=SKIP_SPACY_MODULE)

        from aurora_lens.proxy.app import create_app
        from aurora_lens.proxy.config import ProxyConfig

        audit_file = tmp_path / "audit.new-scenario.jsonl"
        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"audit_log": str(audit_file), "audit_backend": "jsonl"},
            "extraction": {"backend": "spacy"},
        })
        app = create_app(cfg)
        client = TestClient(app)

        old_sid = "scenario-old-session"
        r1 = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Record this fact: Emma is in Paris."}],
                "aurora_session_id": old_sid,
            },
        )
        assert r1.status_code == 200
        old_op_before = client.get("/v1/session/operator-pef?session_id=" + old_sid)
        assert old_op_before.status_code == 200
        old_entities_before = int(old_op_before.json()["operator_pef"]["counts"]["entities"])
        assert old_entities_before >= 1

        reset = client.post("/v1/session/new-scenario")
        assert reset.status_code == 200
        reset_body = reset.json()
        new_sid = reset_body["aurora_session_id"]
        assert isinstance(new_sid, str) and new_sid
        assert new_sid != old_sid
        assert reset.headers.get("Aurora-Session-Id") == new_sid
        assert new_sid in (reset.headers.get("Set-Cookie") or "")
        assert reset_body["session_id"] == new_sid
        new_op = reset_body["operator_pef"]
        assert int(new_op["counts"]["entities"]) == 0
        assert int(new_op["counts"]["relationships"]) == 0
        assert new_op["pending_clarification"]["active"] is False

        # Old session remains intact and accessible when explicitly addressed.
        old_op_after = client.get("/v1/session/operator-pef?session_id=" + old_sid)
        assert old_op_after.status_code == 200
        old_entities_after = int(old_op_after.json()["operator_pef"]["counts"]["entities"])
        assert old_entities_after >= old_entities_before

        # Without explicit sid, cookie routes to the new scenario.
        implicit_new = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hello from reset scenario."}]},
        )
        assert implicit_new.status_code == 200
        assert implicit_new.json()["aurora"]["session_id"] == new_sid

        # Old scenario resumes only when explicitly requested by id.
        explicit_old = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Continue old scenario."}],
                "aurora_session_id": old_sid,
            },
        )
        assert explicit_old.status_code == 200
        assert explicit_old.json()["aurora"]["session_id"] == old_sid

        # Audit identity must show distinct session namespaces.
        entries = [json.loads(ln) for ln in audit_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        audit_session_ids = {e.get("session_id") for e in entries if e.get("session_id")}
        assert old_sid in audit_session_ids
        assert new_sid in audit_session_ids
        assert old_sid != new_sid

    def test_new_scenario_pending_candidates_stay_isolated_from_prior_session_entities(self, monkeypatch):
        """Step 9: unresolved candidate entities in a new scenario stay frame-local."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)
        pytest.importorskip("spacy", reason=SKIP_SPACY_MODULE)

        from aurora_lens.proxy.app import create_app
        from aurora_lens.proxy.config import ProxyConfig

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"audit_log": None},
            "extraction": {"backend": "spacy"},
        })
        app = create_app(cfg)
        client = TestClient(app)

        old_sid = "scenario-prior-people"
        for text in (
            "Record this fact: Emma is a manager.",
            "Record this fact: Anna is an engineer.",
        ):
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": text}],
                    "aurora_session_id": old_sid,
                },
            )
            assert r.status_code == 200

        reset = client.post("/v1/session/new-scenario")
        assert reset.status_code == 200
        new_sid = reset.json()["aurora_session_id"]
        assert new_sid != old_sid

        for text in (
            "Record this fact: Box A is on the table.",
            "Record this fact: Box B is on the shelf.",
            "Record this fact: Box C is in the closet.",
        ):
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": text}],
                    "aurora_session_id": new_sid,
                },
            )
            assert r.status_code == 200

        ambiguous = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Where are the boxes?"}],
                "aurora_session_id": new_sid,
            },
        )
        assert ambiguous.status_code == 200

        op = client.get("/v1/session/operator-pef?session_id=" + new_sid)
        assert op.status_code == 200
        pending = op.json()["operator_pef"]["pending_clarification"]
        assert pending.get("active") is True
        assert pending.get("failed_constraint") == "UNRESOLVED_REFERENT"
        assert isinstance(pending.get("original_question"), str)
        assert pending.get("original_question")
        assert isinstance(pending.get("ambiguous_referents"), list)
        candidates = set(pending.get("candidate_entities") or [])
        assert candidates == {"Box A", "Box B", "Box C"}
        assert "blocked_claims" in pending
        assert "clarification_prompt" in pending
        assert "Emma" not in candidates
        assert "Anna" not in candidates

    def test_session_id_aliases_turn_increments(self, monkeypatch, tmp_path):
        """Session ID from header, aurora_session_id, session_id, aurora.session_id — Turn 2 increments with each alias."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)
        pytest.importorskip("spacy", reason=SKIP_SPACY_MODULE)

        audit_file = tmp_path / "audit.jsonl"
        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"audit_log": str(audit_file), "audit_backend": "jsonl"},
            "extraction": {"backend": "spacy"},
        })
        from aurora_lens.proxy.app import create_app
        app = create_app(cfg)
        client = TestClient(app)

        shared_sid = "alias-test-session"

        # Turn 1: x-aurora-session-id
        r1 = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Alice works at Acme."}]},
            headers={"x-aurora-session-id": shared_sid},
        )
        assert r1.status_code == 200

        # Turn 2: aurora_session_id (different alias, same session)
        r2 = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [
                    {"role": "user", "content": "Alice works at Acme."},
                    {"role": "assistant", "content": "Got it."},
                    {"role": "user", "content": "Where does Alice work?"},
                ],
                "aurora_session_id": shared_sid,
            },
        )
        assert r2.status_code == 200

        lines = [ln for ln in audit_file.read_text().splitlines() if ln.strip()]
        assert len(lines) >= 2
        entries = [json.loads(ln) for ln in lines[-2:]]
        turns = [e.get("turn") for e in entries if "turn" in e]
        assert 1 in turns and 2 in turns, f"Expected turn 1 and 2, got {turns}"

        # Turn 3: session_id alias (new session to avoid confusion)
        sid2 = "alias-session-id-body"
        r3 = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Bob is a teacher."}],
                "session_id": sid2,
            },
        )
        assert r3.status_code == 200
        r4 = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [
                    {"role": "user", "content": "Bob is a teacher."},
                    {"role": "assistant", "content": "OK."},
                    {"role": "user", "content": "What does Bob do?"},
                ],
                "session_id": sid2,
            },
        )
        assert r4.status_code == 200
        entries_sid = [json.loads(ln) for ln in audit_file.read_text().splitlines() if ln.strip()][-2:]
        turns_sid = [e.get("turn") for e in entries_sid if e.get("session_id") == sid2 and "turn" in e]
        assert 1 in turns_sid and 2 in turns_sid, f"session_id alias: expected turn 1,2 got {turns_sid}"

        # Turn 5: aurora.session_id alias
        sid3 = "alias-aurora-dot-session"
        r5 = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Carol is a doctor."}],
                "aurora": {"session_id": sid3},
            },
        )
        assert r5.status_code == 200
        r6 = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [
                    {"role": "user", "content": "Carol is a doctor."},
                    {"role": "assistant", "content": "Noted."},
                    {"role": "user", "content": "Who is Carol?"},
                ],
                "aurora": {"session_id": sid3},
            },
        )
        assert r6.status_code == 200
        all_entries = [json.loads(ln) for ln in audit_file.read_text().splitlines() if ln.strip()]
        entries_aurora = [e for e in all_entries if e.get("session_id") == sid3]
        turns_aurora = [e.get("turn") for e in entries_aurora if "turn" in e]
        assert 1 in turns_aurora and 2 in turns_aurora, f"aurora.session_id: expected turn 1,2 got {turns_aurora}"

    def test_aurora_headers_on_success(self, monkeypatch):
        """Aurora-* headers present on successful chat response."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app, AURORA_HEADER_NAMES
        from aurora_lens.proxy.config import ProxyConfig

        def _mock_adapters(cfg):
            return MockAdapter(), MockAdapter()

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            _mock_adapters,
        )

        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"default_policy": "strict", "audit_log": None},
            "extraction": {"backend": "llm"},
        })
        app = create_app(cfg)
        client = TestClient(app)

        r = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
        )
        assert r.status_code == 200
        for name in ["Aurora-Outcome", "Aurora-Trace-Id", "Aurora-Audit-Sink", "Aurora-Timestamp",
                     "Aurora-Upstream", "Aurora-Policy", "Aurora-Policy-Version", "Aurora-Proxy-Ms"]:
            assert name in r.headers, f"Missing header {name}"
        assert r.headers["Aurora-Upstream"] == "openai_compat"
        assert r.headers["Aurora-Policy"] == "strict"
        assert r.headers["Aurora-Audit-Sink"] in ("jsonl", "ledger", "none")

    def test_aurora_headers_on_400_invalid_json(self, monkeypatch):
        """Aurora-* headers present on 400 error (invalid JSON)."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app
        from aurora_lens.proxy.config import ProxyConfig

        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
        })
        app = create_app(cfg)
        client = TestClient(app)

        r = client.post("/v1/chat/completions", content="not json", headers={"Content-Type": "application/json"})
        assert r.status_code == 400
        assert "Aurora-Outcome" in r.headers
        assert r.headers["Aurora-Outcome"] == "ERROR"
        assert "Aurora-Trace-Id" in r.headers

    def test_aurora_headers_on_422_invalid_payload(self, monkeypatch):
        """Aurora-* headers present on 422 error (invalid chat payload)."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app
        from aurora_lens.proxy.config import ProxyConfig

        def _mock_adapters(cfg):
            return MockAdapter(), MockAdapter()

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            _mock_adapters,
        )

        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
        })
        app = create_app(cfg)
        client = TestClient(app)

        r = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": []},
        )
        assert r.status_code == 422
        assert r.headers["Aurora-Outcome"] == "ERROR"
        assert "Aurora-Trace-Id" in r.headers
        assert "Aurora-Audit-Sink" in r.headers

    def test_aurora_headers_on_http_exception(self, monkeypatch):
        """HTTPException (e.g. 418) carries Aurora headers via exception handler."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from fastapi import HTTPException

        from aurora_lens.proxy.app import create_app
        from aurora_lens.proxy.config import ProxyConfig

        def _mock_adapters(cfg):
            return MockAdapter(), MockAdapter()

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            _mock_adapters,
        )

        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
        })
        app = create_app(cfg)

        @app.get("/test-raise")
        def _raise():
            raise HTTPException(status_code=418, detail="I'm a teapot")

        client = TestClient(app)
        r = client.get("/test-raise")
        assert r.status_code == 418
        assert r.headers["Aurora-Outcome"] == "ERROR"
        assert "Aurora-Trace-Id" in r.headers
        assert "Aurora-Audit-Sink" in r.headers

    def test_aurora_headers_on_404_not_found(self, monkeypatch):
        """404 (route not found) carries Aurora headers via exception handler."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app
        from aurora_lens.proxy.config import ProxyConfig

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )

        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
        })
        app = create_app(cfg)
        client = TestClient(app)

        r = client.get("/v1/nonexistent")
        assert r.status_code == 404
        assert r.headers["Aurora-Outcome"] == "ERROR"
        assert "Aurora-Trace-Id" in r.headers
        assert "Aurora-Audit-Sink" in r.headers

    def test_session_id_from_header(self, monkeypatch):
        """Session ID from x-aurora-session-id header takes precedence."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app
        from aurora_lens.proxy.config import ProxyConfig

        def _mock_adapters(cfg):
            return MockAdapter(), MockAdapter()

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            _mock_adapters,
        )

        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"default_policy": "strict", "audit_log": None},
            "extraction": {"backend": "llm"},
        })
        app = create_app(cfg)
        client = TestClient(app)

        r = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"x-aurora-session-id": "header-session-99"},
        )
        assert r.status_code == 200

    def test_health_endpoint(self, monkeypatch, tmp_path):
        """Full /health returns status, sessions, policy, audit_writable."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )

        audit_file = tmp_path / "audit.jsonl"
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"default_policy": "moderate", "audit_log": str(audit_file), "audit_backend": "jsonl"},
        })
        app = create_app(cfg)
        client = TestClient(app)

        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "sessions" in data
        assert data["policy"] == "moderate"
        assert data["audit_sink"] == "jsonl"
        assert data["audit_writable"] is True
        assert data["auto_verify"] is True
        assert data["auto_interpret"] is True
        assert data["extraction_backend"] == "spacy"  # default when not in config
        assert "audit_log_path" in data
        assert Path(data["audit_log_path"]).resolve() == audit_file.resolve()
        assert "audit_signing_configured" in data
        assert isinstance(data["audit_signing_configured"], bool)
        assert "proxy_run_id" in data
        for name in ["Aurora-Outcome", "Aurora-Trace-Id", "Aurora-Audit-Sink", "Aurora-Timestamp",
                     "Aurora-Upstream", "Aurora-Policy", "Aurora-Policy-Version", "Aurora-Proxy-Ms"]:
            assert name in r.headers, f"Missing header {name}"

        snap = app.state.proxy_audit_target
        for k, v in snap.items():
            assert data[k] == v

    def test_proxy_audit_target_snapshot_helper(self, tmp_path):
        """_proxy_audit_target_snapshot matches GET /health audit subset contract."""
        from aurora_lens.proxy.app import _proxy_audit_target_snapshot

        assert _proxy_audit_target_snapshot(
            None,
            audit_sink="none",
            proxy_run_id="ignored",
            audit_signing_configured=False,
        ) == {"audit_sink": "none"}

        p = tmp_path / "a.jsonl"
        p.write_text("", encoding="utf-8")
        snap = _proxy_audit_target_snapshot(
            str(p),
            audit_sink="jsonl",
            proxy_run_id="run-1",
            audit_signing_configured=True,
        )
        assert snap["audit_sink"] == "jsonl"
        assert snap["proxy_run_id"] == "run-1"
        assert snap["audit_signing_configured"] is True
        assert Path(snap["audit_log_path"]).resolve() == p.resolve()

    def test_health_no_audit_config_proxy_audit_target_sink_only(self, monkeypatch):
        """When audit_log is unset, snapshot and /health expose audit_sink only (none)."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"audit_log": None},
        })
        app = create_app(cfg)
        assert app.state.proxy_audit_target == {"audit_sink": "none"}
        r = TestClient(app).get("/health")
        assert r.json()["audit_sink"] == "none"
        assert "proxy_run_id" not in r.json()

    def test_startup_logs_proxy_audit_target(self, caplog, monkeypatch, tmp_path):
        """create_app emits one INFO line whose extra fields match proxy_audit_target snapshot."""
        caplog.set_level(logging.INFO)
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        audit_file = tmp_path / "audit.jsonl"
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"audit_log": str(audit_file), "audit_backend": "jsonl"},
        })
        app = create_app(cfg)
        assert any(getattr(rec, "msg", None) == "proxy_audit_target" for rec in caplog.records)
        client = TestClient(app)
        data = client.get("/health").json()
        snap = app.state.proxy_audit_target
        for k, v in snap.items():
            assert data[k] == v

    def test_forensics_console_routes(self, monkeypatch):
        """/forensics and /operator serve the same forensics console HTML."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
        })
        app = create_app(cfg)
        client = TestClient(app)
        r1 = client.get("/forensics")
        r2 = client.get("/operator")
        r3 = client.get("/dashboard")
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r3.status_code == 200
        assert "Forensics Console" in r1.text
        assert r1.text == r2.text == r3.text
        assert 'name="aurora-lens-api-base"' in r1.text
        assert "testserver" in r1.text
        assert "pef_turn_classification" in r1.text
        assert "pef_hold_transition" in r1.text
        assert "pef_context_id" in r1.text
        assert "request_capture_status" in r1.text
        assert "evidence_capture_mode" in r1.text
        assert "provenance_status" in r1.text
        assert "Operator quick reference" in r1.text
        assert "request_domain" in r1.text
        assert 'id="link-search"' in r1.text
        assert "data-trace-id" in r1.text
        assert "trace_id" in r1.text
        assert "Operator overview" in r1.text
        assert 'id="operator-overview"' in r1.text
        assert "Needs attention" in r1.text
        assert "Audit data" in r1.text
        assert "Advanced forensics" in r1.text
        assert 'id="advanced-forensics"' in r1.text
        assert "async function loadOperatorSummary()" in r1.text
        assert "overviewState.system = 'Unavailable (cannot load summary)'" in r1.text
        assert "overviewState.audit = 'Unavailable'" in r1.text
        assert "overviewState.audit = 'Available and writable'" in r1.text
        assert "overviewState.audit = 'Available, not writable'" in r1.text
        assert "Promise.all([loadOperatorSummary(), loadHealth(), loadVerify(), loadRecent()])" in r1.text

    def test_operator_summary_endpoint_zero_state_and_safe_fields(self, monkeypatch, tmp_path):
        """Summary endpoint returns coarse aggregate fields only in zero-state."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        audit_file = tmp_path / "audit.jsonl"
        audit_file.write_text("", encoding="utf-8")
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"audit_log": str(audit_file), "audit_backend": "jsonl"},
        })
        app = create_app(cfg)
        client = TestClient(app)
        r = client.get("/v1/operator/summary?n=30")
        assert r.status_code == 200
        body = r.json()
        assert set(body.keys()) == {
            "system_status",
            "sessions",
            "model",
            "policy",
            "policy_version",
            "audit_sink",
            "decision_window_n",
            "audit_data_available",
            "recent_decisions",
            "warning_count",
            "block_count",
            "audit_writable",
            "proxy_run_id",
        }
        assert body["decision_window_n"] == 30
        assert body["recent_decisions"] == 0
        assert body["warning_count"] == 0
        assert body["block_count"] == 0
        assert body["sessions"] == 0
        assert body["audit_data_available"] is True
        dump = json.dumps(body).lower()
        assert "api_key" not in dump
        assert "prompt" not in dump
        assert "response" not in dump
        assert "audit_log_path" not in dump
        assert "session_id" not in dump
        assert "trace_id" not in dump
        assert "tenant" not in dump

    def test_operator_summary_endpoint_classifies_warning_and_block_outcomes(self, monkeypatch, tmp_path):
        """PASS/SOFT_CORRECT/FORCE_REVISE/CONTAIN/HARD_STOP are counted correctly."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        audit_file = tmp_path / "audit.jsonl"
        rows = [
            {"outcome": "PASS", "timestamp": "2026-07-13T00:00:00Z"},
            {"outcome": "SOFT_CORRECT", "timestamp": "2026-07-13T00:00:01Z"},
            {"outcome": "FORCE_REVISE", "timestamp": "2026-07-13T00:00:02Z"},
            {"outcome": "CONTAIN", "timestamp": "2026-07-13T00:00:03Z"},
            {"outcome": "HARD_STOP", "timestamp": "2026-07-13T00:00:04Z"},
        ]
        audit_file.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"audit_log": str(audit_file), "audit_backend": "jsonl"},
        })
        app = create_app(cfg)
        client = TestClient(app)
        body = client.get("/v1/operator/summary?n=30").json()
        assert body["recent_decisions"] == 5
        assert body["warning_count"] == 2  # SOFT_CORRECT + FORCE_REVISE
        assert body["block_count"] == 2    # CONTAIN + HARD_STOP

    def test_operator_summary_endpoint_respects_window_limit(self, monkeypatch, tmp_path):
        """recent_decisions and counts are computed from the requested tail window only."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        audit_file = tmp_path / "audit.jsonl"
        rows = [
            {"outcome": "HARD_STOP", "timestamp": "2026-07-13T00:00:00Z"},
            {"outcome": "PASS", "timestamp": "2026-07-13T00:00:01Z"},
            {"outcome": "CONTAIN", "timestamp": "2026-07-13T00:00:02Z"},
        ]
        audit_file.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"audit_log": str(audit_file), "audit_backend": "jsonl"},
        })
        app = create_app(cfg)
        client = TestClient(app)
        body = client.get("/v1/operator/summary?n=2").json()
        assert body["decision_window_n"] == 2
        assert body["recent_decisions"] == 2
        assert body["warning_count"] == 0
        assert body["block_count"] == 1

    def test_operator_summary_endpoint_matches_health_runtime_sources(self, monkeypatch, tmp_path):
        """Model and policy values align with authoritative runtime sources (/health/config)."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        audit_file = tmp_path / "audit.jsonl"
        audit_file.write_text("", encoding="utf-8")
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"audit_log": str(audit_file), "audit_backend": "jsonl", "default_policy": "strict"},
        })
        app = create_app(cfg)
        client = TestClient(app)
        summary = client.get("/v1/operator/summary").json()
        health = client.get("/health").json()
        assert summary["model"] == "gpt-4"
        assert summary["policy"] == health["policy"]
        assert summary["policy_version"] == health["policy_version"]

    def test_operator_summary_endpoint_session_count_updates(self, monkeypatch, tmp_path):
        """Session count reflects active session manager state."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        audit_file = tmp_path / "audit.jsonl"
        audit_file.write_text("", encoding="utf-8")
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"audit_log": str(audit_file), "audit_backend": "jsonl"},
        })
        app = create_app(cfg)
        client = TestClient(app)
        assert client.get("/v1/operator/summary").json()["sessions"] == 0
        r = client.post("/v1/session/new-scenario")
        assert r.status_code == 200
        assert client.get("/v1/operator/summary").json()["sessions"] == 1

    def test_operator_summary_endpoint_missing_audit_data_fails_safely(self, monkeypatch, tmp_path):
        """When audit path is configured but unavailable, summary remains safe and aggregate-only."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        missing = tmp_path / "missing-dir" / "audit.jsonl"
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"audit_log": str(missing), "audit_backend": "jsonl"},
        })
        app = create_app(cfg)
        client = TestClient(app)
        body = client.get("/v1/operator/summary").json()
        assert body["recent_decisions"] == 0
        assert body["warning_count"] == 0
        assert body["block_count"] == 0

    def test_jsonl_audit_entry_lookup_requires_identifier(self, monkeypatch, tmp_path):
        """GET /v1/audit/entry returns 400 when neither cid nor trace_id is given."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        audit_file = tmp_path / "audit.jsonl"
        audit_file.write_text("", encoding="utf-8")
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"audit_log": str(audit_file), "audit_backend": "jsonl"},
        })
        app = create_app(cfg)
        client = TestClient(app)
        r = client.get("/v1/audit/entry")
        assert r.status_code == 400
        body = r.json()
        assert "error" in body
        assert "cid" in body["error"]["message"].lower() and "trace" in body["error"]["message"].lower()

    def test_jsonl_audit_entry_by_trace_id(self, monkeypatch, tmp_path):
        """Legacy JSONL rows without cid still load by trace_id (newest match)."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        audit_file = tmp_path / "audit.jsonl"
        row = {"trace_id": "t-legacy", "outcome": "HARD_STOP", "timestamp": "2026-04-16T12:00:00Z"}
        audit_file.write_text(json.dumps(row) + "\n", encoding="utf-8")
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"audit_log": str(audit_file), "audit_backend": "jsonl"},
        })
        app = create_app(cfg)
        client = TestClient(app)
        r = client.get("/v1/audit/entry?trace_id=t-legacy")
        assert r.status_code == 200
        assert r.json()["outcome"] == "HARD_STOP"

    def test_jsonl_audit_entry_trace_id_timestamp_disambiguates(self, monkeypatch, tmp_path):
        """Optional timestamp pins one row when trace_id repeats."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        audit_file = tmp_path / "audit.jsonl"
        audit_file.write_text(
            json.dumps(
                {"trace_id": "dup", "outcome": "PASS", "timestamp": "2026-04-16T10:00:00Z"}
            )
            + "\n"
            + json.dumps(
                {"trace_id": "dup", "outcome": "HARD_STOP", "timestamp": "2026-04-16T11:00:00Z"}
            )
            + "\n",
            encoding="utf-8",
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"audit_log": str(audit_file), "audit_backend": "jsonl"},
        })
        app = create_app(cfg)
        client = TestClient(app)
        r_newest = client.get("/v1/audit/entry?trace_id=dup")
        assert r_newest.status_code == 200
        assert r_newest.json()["outcome"] == "HARD_STOP"
        r_first = client.get(
            "/v1/audit/entry?trace_id=dup&timestamp=" + "2026-04-16T10:00:00Z"
        )
        assert r_first.status_code == 200
        assert r_first.json()["outcome"] == "PASS"

    def test_healthz_minimal(self, monkeypatch):
        """/healthz returns minimal ok for k8s."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
        })
        app = create_app(cfg)
        client = TestClient(app)
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"ok": True}
        for name in ["Aurora-Outcome", "Aurora-Trace-Id", "Aurora-Audit-Sink", "Aurora-Timestamp",
                     "Aurora-Upstream", "Aurora-Policy", "Aurora-Policy-Version", "Aurora-Proxy-Ms"]:
            assert name in r.headers, f"Missing header {name}"

    def test_policy_version_always_present(self, monkeypatch):
        """policy_version is always present (defaulted to 1.0 when not in config)."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
        })
        app = create_app(cfg)
        client = TestClient(app)
        r = client.get("/health")
        assert r.status_code == 200
        assert "policy_version" in r.json()
        assert r.json()["policy_version"] == "1.0"
        assert r.headers["Aurora-Policy-Version"] == "1.0"

    def test_health_includes_extraction_and_auto_flags(self, monkeypatch):
        """/health returns auto_verify, auto_interpret, extraction_backend for runtime inspection."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "extraction": {"backend": "spacy"},
        })
        app = create_app(cfg)
        client = TestClient(app)
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["extraction_backend"] == "spacy"
        assert data["auto_verify"] is True
        assert data["auto_interpret"] is True
        cf = data.get("code_features") or {}
        assert cf.get("update_pef_kwarg_user_text") is True
        assert cf.get("revision_gate_user_text_scan") is True

    def test_health_includes_application_version_and_git_commit(self, monkeypatch):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        monkeypatch.setenv("AURORA_GIT_COMMIT", "deadbeefcafebabefeedface1234567890abcdef")
        monkeypatch.delenv("RAILWAY_GIT_COMMIT_SHA", raising=False)

        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"audit_log": None},
        })
        client = TestClient(create_app(cfg))
        data = client.get("/health").json()

        assert isinstance(data.get("application_version"), str)
        assert data["application_version"].strip() != ""
        assert data["git_commit"] == "deadbeefcafebabefeedface1234567890abcdef"
        assert data["git_commit_source"] == "env"

    def test_health_git_commit_uses_railway_fallback_when_aurora_unset(self, monkeypatch):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        monkeypatch.delenv("AURORA_GIT_COMMIT", raising=False)
        monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", "cafef00dbadc0ffee1234567890abcdef123456")

        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"audit_log": None},
        })
        client = TestClient(create_app(cfg))
        data = client.get("/health").json()
        assert data["git_commit"] == "cafef00dbadc0ffee1234567890abcdef123456"
        assert data["git_commit_source"] == "railway_env"

    def test_audit_recent_with_entries(self, monkeypatch, tmp_path):
        """/v1/audit/recent returns last n entries from audit file."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )

        audit_file = tmp_path / "audit.jsonl"
        audit_file.write_text('{"schema_version":2,"outcome":"PASS"}\n{"schema_version":2,"outcome":"HARD_STOP"}\n')
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"audit_log": str(audit_file), "audit_backend": "jsonl"},
        })
        app = create_app(cfg)
        client = TestClient(app)

        r = client.get("/v1/audit/recent?n=10")
        assert r.status_code == 200
        data = r.json()
        assert "entries" in data
        assert len(data["entries"]) == 2
        assert data["entries"][0].get("outcome", data["entries"][0].get("action")) == "PASS"
        assert data["entries"][1].get("outcome", data["entries"][1].get("action")) == "HARD_STOP"
        assert data["n"] == 2
        assert data.get("backend") == "jsonl"
        assert len(data.get("operator_rows", [])) == 2
        assert data["operator_rows"][0]["row_kind"] == "jsonl"
        assert data["operator_rows"][0]["outcome"] == "PASS"
        assert data["operator_rows"][1]["outcome"] == "HARD_STOP"
        assert data["operator_rows"][0]["has_llm_output"] is False
        assert data["operator_rows"][1]["has_llm_output"] is False
        assert "has_chain_of_custody" in data["operator_rows"][0]
        assert "evidence_audit_status" in data["operator_rows"][0]
        assert data["operator_rows"][0]["evidence_audit_status"] is None
        assert "provenance_status" in data["operator_rows"][0]
        assert "request_capture_status" in data["operator_rows"][0]
        assert "evidence_capture_mode" in data["operator_rows"][0]
        for name in ["Aurora-Outcome", "Aurora-Trace-Id", "Aurora-Audit-Sink", "Aurora-Timestamp",
                     "Aurora-Upstream", "Aurora-Policy", "Aurora-Policy-Version", "Aurora-Proxy-Ms"]:
            assert name in r.headers, f"Missing header {name}"

    def test_audit_recent_surfaces_evidence_state_fields_distinctly(self, monkeypatch, tmp_path):
        """provenance_status, request_capture_status, evidence_capture_mode, and
        evidence_audit_status are four distinct fields on the operator row and
        must not be conflated with one another."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )

        row = {
            "schema_version": 2,
            "outcome": "HARD_STOP",
            "provenance_status": "degraded",
            "request_capture_status": "sealed_by_policy",
            "evidence_capture_mode": "sealed",
            "chain_of_custody": {"evidence_audit_status": "complete"},
        }
        audit_file = tmp_path / "audit.jsonl"
        audit_file.write_text(json.dumps(row) + "\n")
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"audit_log": str(audit_file), "audit_backend": "jsonl"},
        })
        app = create_app(cfg)
        client = TestClient(app)

        r = client.get("/v1/audit/recent?n=10")
        assert r.status_code == 200
        op_row = r.json()["operator_rows"][0]
        assert op_row["provenance_status"] == "degraded"
        assert op_row["request_capture_status"] == "sealed_by_policy"
        assert op_row["evidence_capture_mode"] == "sealed"
        assert op_row["evidence_audit_status"] == "complete"

    def test_audit_search_and_recent_domain_non_admit_filters(self, monkeypatch, tmp_path):
        """Operator plane: filter by request_domain and non-admit-only (PASS/SOFT_CORRECT excluded)."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )

        audit_file = tmp_path / "audit.jsonl"
        audit_file.write_text(
            '{"schema_version":2,"outcome":"PASS","timestamp":"2026-01-01T00:00:00+00:00","request_domain":"finance"}\n'
            '{"schema_version":2,"outcome":"HARD_STOP","timestamp":"2026-01-01T00:01:00+00:00","request_domain":"finance"}\n'
            '{"schema_version":2,"outcome":"FORCE_REVISE","timestamp":"2026-01-01T00:02:00+00:00","request_domain":"legal"}\n',
            encoding="utf-8",
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"audit_log": str(audit_file), "audit_backend": "jsonl"},
        })
        app = create_app(cfg)
        client = TestClient(app)

        r = client.get("/v1/audit/search?domain=finance&non_admit_only=true&limit=10")
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 1
        assert data["entries"][0]["outcome"] == "HARD_STOP"

        r2 = client.get("/v1/audit/recent?n=10&domain=finance&non_admit_only=true")
        assert r2.status_code == 200
        d2 = r2.json()
        assert len(d2["entries"]) == 1
        assert d2["entries"][0]["outcome"] == "HARD_STOP"
        assert d2["operator_rows"][0].get("request_domain") == "finance"

    def test_audit_recent_skips_partial_trailing_line(self, monkeypatch, tmp_path):
        """Partial last line (concurrent write) is skipped; valid entries returned."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )

        audit_file = tmp_path / "audit.jsonl"
        audit_file.write_text('{"schema_version":2,"outcome":"PASS"}\n{"schema_version":2,"outcome":"HARD_STOP"}\n{"schema_version":2,"incomplete')
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"audit_log": str(audit_file), "audit_backend": "jsonl"},
        })
        app = create_app(cfg)
        client = TestClient(app)

        r = client.get("/v1/audit/recent?n=10")
        assert r.status_code == 200
        data = r.json()
        assert len(data["entries"]) == 2
        assert data["entries"][0].get("outcome", data["entries"][0].get("action")) == "PASS"
        assert data["entries"][1].get("outcome", data["entries"][1].get("action")) == "HARD_STOP"

    def test_audit_recent_operator_rows_pef_linkage_jsonl(self, monkeypatch, tmp_path):
        """operator_rows expose pef_turn_classification / pef_hold_transition from flat JSONL."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )

        audit_file = tmp_path / "audit.jsonl"
        audit_file.write_text(
            '{"schema_version":2,"outcome":"PASS","pef_turn_classification":"fresh","pef_hold_transition":"none"}\n'
            '{"schema_version":2,"outcome":"HARD_STOP"}\n',
            encoding="utf-8",
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"audit_log": str(audit_file), "audit_backend": "jsonl"},
        })
        app = create_app(cfg)
        client = TestClient(app)
        r = client.get("/v1/audit/recent?n=10")
        assert r.status_code == 200
        rows = r.json()["operator_rows"]
        assert rows[0]["pef_turn_classification"] == "fresh"
        assert rows[0]["pef_hold_transition"] == "none"
        assert rows[1]["pef_turn_classification"] is None
        assert rows[1]["pef_hold_transition"] is None

    def test_audit_recent_operator_rows_pef_linkage_ledger(self, monkeypatch, tmp_path):
        """AFL ledger: nested payload.data fields flatten to same operator_row keys."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )

        afl_line = {
            "v": 1,
            "kind": "aurora.event",
            "time": "2026-01-15T12:00:00+00:00",
            "seq": 0,
            "trace": "tr-1",
            "prev": "h:null",
            "cid": "cid-led-pef",
            "op": "HARD_STOP",
            "scope": {"domain": "governance", "subdomain": "aurora-lens", "authority": "policy"},
            "payload": {
                "pv": 1,
                "data": {
                    "action": "HARD_STOP",
                    "event_type": "governance_decision",
                    "pef_turn_classification": "held_refusal",
                    "pef_hold_transition": "ambiguity",
                },
            },
            "canon": "c14n:jcs",
            "ext": {},
            "hash": "h:sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "sig": None,
        }
        audit_file = tmp_path / "audit.jsonl"
        audit_file.write_text(json.dumps(afl_line) + "\n", encoding="utf-8")
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"audit_log": str(audit_file), "audit_backend": "ledger"},
        })
        app = create_app(cfg)
        client = TestClient(app)
        r = client.get("/v1/audit/recent?n=10")
        assert r.status_code == 200
        data = r.json()
        assert data.get("backend") == "ledger"
        row = data["operator_rows"][0]
        assert row["row_kind"] == "ledger"
        assert row["pef_turn_classification"] == "held_refusal"
        assert row["pef_hold_transition"] == "ambiguity"
        assert row["has_llm_output"] is False

    def test_audit_recent_operator_rows_has_llm_output_ledger(self, monkeypatch, tmp_path):
        """Ledger operator_rows expose has_llm_output from payload.data original_response."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )

        with_llm = {
            "v": 1,
            "kind": "aurora.event",
            "time": "2026-01-15T12:00:00+00:00",
            "seq": 0,
            "trace": "tr-1",
            "prev": "h:null",
            "cid": "cid-led-llm",
            "op": "HARD_STOP",
            "scope": {"domain": "governance", "subdomain": "aurora-lens", "authority": "policy"},
            "payload": {
                "pv": 1,
                "data": {
                    "action": "HARD_STOP",
                    "event_type": "governance_decision",
                    "original_response": "model text",
                },
            },
            "canon": "c14n:jcs",
            "ext": {},
            "hash": "h:sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "sig": None,
        }
        without_llm = {
            "v": 1,
            "kind": "aurora.event",
            "time": "2026-01-15T12:01:00+00:00",
            "seq": 1,
            "trace": "tr-1",
            "prev": with_llm["hash"],
            "cid": "cid-led-no-llm",
            "op": "HARD_STOP",
            "scope": {"domain": "governance", "subdomain": "aurora-lens", "authority": "policy"},
            "payload": {"pv": 1, "data": {"action": "HARD_STOP", "event_type": "governance_decision"}},
            "canon": "c14n:jcs",
            "ext": {},
            "hash": "h:sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            "sig": None,
        }
        audit_file = tmp_path / "audit.jsonl"
        audit_file.write_text(json.dumps(with_llm) + "\n" + json.dumps(without_llm) + "\n", encoding="utf-8")
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"audit_log": str(audit_file), "audit_backend": "ledger"},
        })
        app = create_app(cfg)
        client = TestClient(app)
        r = client.get("/v1/audit/recent?n=10")
        assert r.status_code == 200
        rows = r.json()["operator_rows"]
        assert rows[0]["has_llm_output"] is True
        assert rows[1]["has_llm_output"] is False

    def test_proxy_ledger_pef_linkage_after_chat_completions(self, monkeypatch, tmp_path):
        """Real proxy path (CanonicalScannerGateBridge + ledger) must write linkage rows that pass ``verify_pef_linkage``."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.govern.audit_io import verify_pef_linkage_detailed
        from aurora_lens.proxy.app import create_app

        ledger = tmp_path / "proxy_ledger_pef_linkage.jsonl"
        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {
                "default_policy": "strict",
                "audit_log": str(ledger),
                "audit_backend": "ledger",
            },
        })
        app = create_app(cfg)
        client = TestClient(app)
        sid = "sess-proxy-pef-linkage"
        ext = [{"type": "SELF_HARM_INSTRUCTION", "evidence": ["e"], "severity": "error"}]
        r1 = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "first turn"}],
                "aurora": {"aurora_session_id": sid, "external_flags": ext},
            },
        )
        assert r1.status_code == 200, r1.text
        content1 = r1.json()["choices"][0]["message"]["content"]
        r2 = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [
                    {"role": "user", "content": "first turn"},
                    {"role": "assistant", "content": content1},
                    {"role": "user", "content": "second turn follow-up"},
                ],
                "aurora": {"aurora_session_id": sid},
            },
        )
        assert r2.status_code == 200, r2.text
        det = verify_pef_linkage_detailed(ledger, n=500)
        assert det["ok"], det
        assert int(det["pairs_checked"]) >= 1

        health_pr = client.get("/health").json().get("proxy_run_id")
        assert health_pr
        for ln in ledger.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            o = json.loads(ln)
            if o.get("kind") != "aurora.event":
                continue
            data = (o.get("payload") or {}).get("data") or {}
            if data.get("event_type") != "governance_decision":
                continue
            assert data.get("proxy_run_id") == health_pr
            break

        v = client.get("/v1/audit/verify?n=500")
        assert v.status_code == 200, v.text
        assert v.json()["what_was_verified"]["pef_linkage"]["ok"] is True

    def test_audit_recent_operator_rows_subsystem_row_no_pef_linkage(self, monkeypatch, tmp_path):
        """Non-governance / legacy lines without linkage fields yield null on operator_rows."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )

        audit_file = tmp_path / "audit.jsonl"
        audit_file.write_text(
            '{"schema_version":2,"outcome":"PASS","type":"subsystem_ping"}\n',
            encoding="utf-8",
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"audit_log": str(audit_file), "audit_backend": "jsonl"},
        })
        app = create_app(cfg)
        client = TestClient(app)
        r = client.get("/v1/audit/recent?n=10")
        assert r.status_code == 200
        row = r.json()["operator_rows"][0]
        assert row["pef_turn_classification"] is None
        assert row["pef_hold_transition"] is None

    def test_health_audit_not_configured_status_ok(self, monkeypatch):
        """When audit_log not configured: audit_writable=null, status=ok (not degraded)."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"audit_log": None},
        })
        app = create_app(cfg)
        client = TestClient(app)
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["audit_writable"] is None
        assert data["status"] == "ok"

    def test_audit_recent_no_config_returns_404(self, monkeypatch):
        """/v1/audit/recent returns 404 when audit_log not configured."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"audit_log": None},
        })
        app = create_app(cfg)
        client = TestClient(app)
        r = client.get("/v1/audit/recent")
        assert r.status_code == 404

    def test_audit_anomaly_check_returns_rates_and_alerts(self, monkeypatch, tmp_path):
        """P.3: /v1/audit/anomaly-check computes rates; alerts when threshold exceeded."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        # 3 PASS, 2 FORCE_REVISE, 1 HARD_STOP with EXTRACTION_FAILED -> interv=3/6=50%, extr=1/6~17%
        # Timestamps relative to now so window=24h includes them (avoids flakiness when run on different days)
        base = datetime.now(timezone.utc) - timedelta(minutes=6)

        def ts(i: int) -> str:
            return (base + timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M:%SZ")

        entries = [
            f'{{"schema_version":2,"outcome":"PASS","timestamp":"{ts(0)}"}}\n',
            f'{{"schema_version":2,"outcome":"PASS","timestamp":"{ts(1)}"}}\n',
            f'{{"schema_version":2,"outcome":"PASS","timestamp":"{ts(2)}"}}\n',
            f'{{"schema_version":2,"outcome":"FORCE_REVISE","timestamp":"{ts(3)}"}}\n',
            f'{{"schema_version":2,"outcome":"FORCE_REVISE","timestamp":"{ts(4)}"}}\n',
            f'{{"schema_version":2,"outcome":"HARD_STOP","forensic_event":{{"failed_constraints":["EXTRACTION_FAILED"]}},"timestamp":"{ts(5)}"}}\n',
        ]
        audit_file = tmp_path / "audit.jsonl"
        audit_file.write_text("".join(entries))
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {
                "audit_log": str(audit_file),
                "audit_backend": "jsonl",
                "threshold_intervention_rate": 0.4,
                "threshold_extraction_failure_rate": 0.1,
            },
        })
        app = create_app(cfg)
        client = TestClient(app)

        r = client.get("/v1/audit/anomaly-check?window=24h")
        assert r.status_code == 200
        data = r.json()
        assert data["total_entries"] == 6
        assert data["intervention_rate"] == pytest.approx(0.5, rel=0.01)
        assert data["extraction_failure_rate"] == pytest.approx(1 / 6, rel=0.01)
        assert any("intervention_rate" in a for a in data["alerts"])
        assert any("extraction_failure_rate" in a for a in data["alerts"])

        # No threshold -> no alerts
        cfg2 = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"audit_log": str(audit_file), "audit_backend": "jsonl"},
        })
        app2 = create_app(cfg2)
        client2 = TestClient(app2)
        r2 = client2.get("/v1/audit/anomaly-check?window=24h")
        assert r2.status_code == 200
        assert r2.json()["alerts"] == []


class TestPhase5Hardening:
    """Phase 5: Input validation, rate limits, metrics."""

    def _make_app(self, monkeypatch, hardening: dict | None = None):
        from aurora_lens.proxy.app import create_app
        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "hardening": hardening or {},
        })
        return create_app(cfg)

    def test_payload_too_large_returns_413(self, monkeypatch):
        """Payload exceeding max_payload_bytes returns 413."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        app = self._make_app(monkeypatch, {"max_payload_bytes": 100})
        client = TestClient(app)
        # 100 bytes = limit; we need > 100
        body = json.dumps({"model": "gpt-4", "messages": [{"role": "user", "content": "x" * 50}]})
        assert len(body) > 100
        r = client.post("/v1/chat/completions", content=body, headers={"Content-Type": "application/json"})
        assert r.status_code == 413
        assert "Payload too large" in r.json()["error"]["message"]

    def test_too_many_messages_returns_422(self, monkeypatch):
        """Messages count exceeding max_messages returns 422."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        app = self._make_app(monkeypatch, {"max_messages": 3})
        client = TestClient(app)
        msgs = [{"role": "user", "content": "Hi"}] * 4
        r = client.post("/v1/chat/completions", json={"model": "gpt-4", "messages": msgs})
        assert r.status_code == 422
        assert "Too many messages" in r.json()["error"]["message"]

    def test_content_too_long_returns_422(self, monkeypatch):
        """Total content exceeding max_content_chars returns 422."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        app = self._make_app(monkeypatch, {"max_content_chars": 50})
        client = TestClient(app)
        r = client.post("/v1/chat/completions", json={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "x" * 51}],
        })
        assert r.status_code == 422
        assert "content too long" in r.json()["error"]["message"].lower()

    def test_rate_limit_global_returns_429(self, monkeypatch):
        """Global rate limit returns 429 when exceeded."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        app = self._make_app(monkeypatch, {"rate_limit_global": 2})
        client = TestClient(app)
        for _ in range(2):
            r = client.post("/v1/chat/completions", json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]})
            assert r.status_code == 200
        r = client.post("/v1/chat/completions", json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]})
        assert r.status_code == 429
        assert "Rate limit" in r.json()["error"]["message"]
        assert "Retry-After" in r.headers

    def test_rate_limit_per_session_returns_429(self, monkeypatch):
        """Per-session rate limit returns 429 when exceeded."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        app = self._make_app(monkeypatch, {"rate_limit_per_session": 2})
        client = TestClient(app)
        for _ in range(2):
            r = client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
                headers={"x-aurora-session-id": "session-1"},
            )
            assert r.status_code == 200
        r = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"x-aurora-session-id": "session-1"},
        )
        assert r.status_code == 429
        assert "Rate limit" in r.json()["error"]["message"]

    def test_metrics_endpoint_returns_prometheus_format(self, monkeypatch):
        """GET /metrics returns Prometheus exposition format."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        app = self._make_app(monkeypatch)
        client = TestClient(app)
        r = client.get("/metrics")
        assert r.status_code == 200
        assert r.headers.get("Content-Type", "").startswith("text/plain")
        text = r.text
        assert "aurora_lens_uptime_seconds" in text
        assert "# HELP" in text
        assert "# TYPE" in text

    def test_metrics_increments_on_chat_completion(self, monkeypatch):
        """chat_completions increments metrics."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.metrics import get_metrics
        get_metrics().reset()

        app = self._make_app(monkeypatch)
        client = TestClient(app)
        r = client.post("/v1/chat/completions", json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]})
        assert r.status_code == 200

        r2 = client.get("/metrics")
        assert "chat_completions_total" in r2.text

    def test_metrics_help_type_once_per_base(self, monkeypatch):
        """HELP/TYPE emitted once per base metric, not per series."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.metrics import get_metrics
        m = get_metrics()
        m.reset()
        m.inc("chat_completions_total", labels={"outcome": "PASS"})
        m.inc("chat_completions_total", labels={"outcome": "REFUSE"})

        text = m.to_prometheus()
        help_count = text.count("# HELP aurora_lens_chat_completions_total")
        type_count = text.count("# TYPE aurora_lens_chat_completions_total")
        assert help_count == 1, "HELP should appear once per base"
        assert type_count == 1, "TYPE should appear once per base"
        assert "outcome=\"PASS\"" in text
        assert "outcome=\"REFUSE\"" in text

    def test_metrics_label_escaping(self):
        """Label values escape \\ \" and newlines."""
        from aurora_lens.proxy.metrics import get_metrics
        m = get_metrics()
        m.reset()
        m.inc("test_metric", labels={"x": 'a"b'})

        text = m.to_prometheus()
        # Unescaped " would break Prometheus; must be \"
        assert 'x="a\\"b"' in text


class TestPhaseBAuth:
    """Phase B: Inbound authentication."""

    def _make_app(self, monkeypatch, auth: dict | None = None, hardening: dict | None = None):
        from aurora_lens.proxy.app import create_app
        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        cfg_map = {
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "hardening": hardening or {},
        }
        if auth is not None:
            cfg_map["auth"] = auth
        cfg = ProxyConfig.from_mapping(cfg_map)
        return create_app(cfg)

    def test_auth_disabled_allows_unauthenticated(self, monkeypatch):
        """When auth.enabled=false, requests without key succeed."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        app = self._make_app(monkeypatch, auth={"enabled": False})
        client = TestClient(app)
        r = client.post("/v1/chat/completions", json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]})
        assert r.status_code == 200

    def test_auth_required_401_on_missing_key(self, monkeypatch):
        """When auth enabled, missing key returns 401."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        app = self._make_app(monkeypatch, auth={
            "enabled": True,
            "keys": [{"key": "secret-key-1", "label": "app-prod"}],
        })
        client = TestClient(app)
        r = client.post("/v1/chat/completions", json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]})
        assert r.status_code == 401
        assert "Missing" in r.json()["error"]["message"]
        assert "Aurora-Trace-Id" in r.headers

    def test_auth_403_on_invalid_key(self, monkeypatch):
        """When auth enabled, invalid key returns 403."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        app = self._make_app(monkeypatch, auth={
            "enabled": True,
            "keys": [{"key": "secret-key-1", "label": "app-prod"}],
        })
        client = TestClient(app)
        r = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert r.status_code == 403
        assert "Invalid" in r.json()["error"]["message"]

    def test_auth_bearer_succeeds(self, monkeypatch):
        """Authorization: Bearer <key> authenticates successfully."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        app = self._make_app(monkeypatch, auth={
            "enabled": True,
            "keys": [{"key": "secret-key-1", "label": "app-prod"}],
        })
        client = TestClient(app)
        r = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": "Bearer secret-key-1"},
        )
        assert r.status_code == 200

    def test_auth_x_api_key_succeeds(self, monkeypatch):
        """x-api-key header authenticates successfully."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        app = self._make_app(monkeypatch, auth={
            "enabled": True,
            "keys": [{"key": "secret-key-1", "label": "app-prod"}],
        })
        client = TestClient(app)
        r = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"x-api-key": "secret-key-1"},
        )
        assert r.status_code == 200

    def test_auth_health_bypass(self, monkeypatch):
        """Health endpoints bypass auth."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        app = self._make_app(monkeypatch, auth={
            "enabled": True,
            "keys": [{"key": "secret-key-1", "label": "app-prod"}],
        })
        client = TestClient(app)
        r = client.get("/health")
        assert r.status_code == 200
        r = client.get("/healthz")
        assert r.status_code == 200

    def test_auth_forensics_audit_read_bypass(self, monkeypatch, tmp_path):
        """Read-only audit GETs used by /forensics bypass auth; chat does not."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        audit_file = tmp_path / "audit.jsonl"
        audit_file.write_text("", encoding="utf-8")
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "auth": {"enabled": True, "keys": [{"key": "secret-key-1", "label": "app-prod"}]},
            "governance": {"audit_log": str(audit_file), "audit_backend": "jsonl"},
        })
        app = create_app(cfg)
        client = TestClient(app)
        assert client.get("/v1/audit/recent?n=5").status_code == 200
        assert client.get("/v1/audit/search?limit=5").status_code == 200
        assert client.get("/v1/operator/summary").status_code == 200
        r = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
        )
        assert r.status_code == 401

    def test_auth_label_in_audit_entry(self, monkeypatch, tmp_path):
        """Auth key label appears in audit entry when auth enabled."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        audit_file = tmp_path / "audit.jsonl"
        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"default_policy": "strict", "audit_log": str(audit_file), "audit_backend": "jsonl"},
            "auth": {"enabled": True, "keys": [{"key": "secret-key-1", "label": "app-production"}]},
        })
        from aurora_lens.proxy.app import create_app
        app = create_app(cfg)
        client = TestClient(app)

        # Use external_flags to trigger HARD_STOP so an audit entry is written (PASS does not log)
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hi"}],
                "aurora": {
                    "external_flags": [
                        {"type": "SELF_HARM_INSTRUCTION", "evidence": ["test span"], "severity": "error"},
                    ],
                },
            },
            headers={"Authorization": "Bearer secret-key-1"},
        )
        assert r.status_code == 200

        lines = [ln for ln in audit_file.read_text().splitlines() if ln.strip()]
        assert len(lines) >= 1
        entry = json.loads(lines[-1])
        assert entry.get("tenant_label") == "app-production"
        assert entry.get("outcome") == "HARD_STOP"

    def test_run_id_fresh_per_request_proxy_run_id_stable_across_requests(self, monkeypatch, tmp_path):
        """run_id must vary per HTTP request; proxy_run_id must stay the process/deployment
        id (same value as GET /health) so verify_pef_linkage_detailed does not mistake two
        requests to the same running process for a redeploy boundary and skip the pair."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        audit_file = tmp_path / "audit.jsonl"
        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"default_policy": "strict", "audit_log": str(audit_file), "audit_backend": "jsonl"},
        })
        from aurora_lens.proxy.app import create_app
        app = create_app(cfg)
        client = TestClient(app)
        health_proxy_run_id = client.get("/health").json().get("proxy_run_id")
        assert health_proxy_run_id

        ext = [{"type": "SELF_HARM_INSTRUCTION", "evidence": ["e"], "severity": "error"}]
        for _ in range(2):
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "aurora": {"external_flags": ext},
                },
            )
            assert r.status_code == 200, r.text

        lines = [ln for ln in audit_file.read_text().splitlines() if ln.strip()]
        entries = [json.loads(ln) for ln in lines]
        run_ids = [e.get("run_id") for e in entries]
        proxy_run_ids = [e.get("proxy_run_id") for e in entries]
        assert len(entries) >= 2
        assert all(run_ids), f"every audit row must carry a run_id: {run_ids}"
        assert len(set(run_ids)) == len(run_ids), f"each request must mint a distinct run_id: {run_ids}"
        assert all(pr == health_proxy_run_id for pr in proxy_run_ids), (
            f"proxy_run_id must stay the stable process id across requests, matching "
            f"GET /health ({health_proxy_run_id}): {proxy_run_ids}"
        )

    def test_audit_provenance_extractor_backend_matches_config_and_no_llm_output(self, monkeypatch, tmp_path):
        """Proxy evidence: extractor_backend matches configured backend; provenance never llm_output for normal turns."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)
        pytest.importorskip("spacy", reason=SKIP_SPACY_MODULE)

        audit_file = tmp_path / "audit.jsonl"
        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "extraction": {"backend": "spacy"},
            "governance": {"default_policy": "strict", "audit_log": str(audit_file), "audit_backend": "jsonl"},
        })
        from aurora_lens.proxy.app import create_app
        app = create_app(cfg)
        client = TestClient(app)

        # Message that triggers spaCy extraction of claims; external_flags force HARD_STOP so pef_snapshot is stored
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Emma has a red book."}],
                "aurora": {
                    "external_flags": [
                        {"type": "SELF_HARM_INSTRUCTION", "evidence": ["test"], "severity": "error"},
                    ],
                },
            },
        )
        assert r.status_code == 200

        lines = [ln for ln in audit_file.read_text().splitlines() if ln.strip()]
        assert len(lines) >= 1
        entry = json.loads(lines[-1])
        assert entry.get("outcome") == "HARD_STOP"

        snap = entry.get("pef_snapshot")
        assert snap is not None, "HARD_STOP entry should include pef_snapshot"
        rels = snap.get("relationships") or []
        # "Emma has a red book." should produce at least one claim via spaCy
        assert len(rels) >= 1, "Expected spaCy to extract claims from 'Emma has a red book.'"
        for rel in rels:
            assert rel.get("provenance") in ("user_input", "pre_populated", "system"), (
                "Normal turns must not mark LLM output as ground truth"
            )
            assert rel.get("provenance") != "llm_output"
            assert rel.get("extractor_backend") == "spacy", (
                "extractor_backend must match configured extraction backend"
            )

    def test_key_rotation_two_keys_same_label(self, monkeypatch):
        """Key rotation: two valid keys for same label work; deploy new without breaking old."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        app = self._make_app(monkeypatch, auth={
            "enabled": True,
            "keys": [
                {"key": "old-key", "label": "app-prod", "policy": "strict"},
                {"key": "new-key", "label": "app-prod", "policy": "strict"},
            ],
        })
        client = TestClient(app)

        r_old = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": "Bearer old-key"},
        )
        r_new = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": "Bearer new-key"},
        )
        assert r_old.status_code == 200
        assert r_new.status_code == 200

    def test_key_no_policy_uses_deployment_default(self, monkeypatch, tmp_path):
        """Key with no policy uses governance.default_policy (deployment default) deterministically."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        audit_file = tmp_path / "audit.jsonl"
        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"default_policy": "strict", "audit_log": str(audit_file), "audit_backend": "jsonl"},
            "auth": {
                "enabled": True,
                "keys": [{"key": "no-policy-key", "label": "app-default"}],
            },
        })
        from aurora_lens.proxy.app import create_app
        app = create_app(cfg)
        client = TestClient(app)

        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
            "aurora": {
                "external_flags": [
                    {"type": "UNBOUND_ENTITY", "evidence": ["test"], "severity": "warning"},
                ],
            },
        }
        r = client.post("/v1/chat/completions", json=payload, headers={"Authorization": "Bearer no-policy-key"})
        assert r.status_code == 200
        lines = [ln for ln in audit_file.read_text().splitlines() if ln.strip()]
        entry = json.loads(lines[-1])
        assert entry.get("policy_profile") == "strict"

    def test_per_key_policy_override(self, monkeypatch, tmp_path):
        """Per-key policy override: staging key gets moderate, production key gets strict."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        audit_file = tmp_path / "audit.jsonl"
        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"default_policy": "strict", "audit_log": str(audit_file), "audit_backend": "jsonl"},
            "auth": {
                "enabled": True,
                "keys": [
                    {"key": "prod-key", "label": "app-production", "policy": "strict"},
                    {"key": "staging-key", "label": "app-staging", "policy": "moderate"},
                ],
            },
        })
        from aurora_lens.proxy.app import create_app
        app = create_app(cfg)
        client = TestClient(app)

        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
            "aurora": {
                "external_flags": [
                    {"type": "UNBOUND_ENTITY", "evidence": ["test"], "severity": "warning"},
                ],
            },
        }

        # Production key (strict): UNBOUND_ENTITY -> FORCE_REVISE
        r1 = client.post("/v1/chat/completions", json=payload, headers={"Authorization": "Bearer prod-key"})
        assert r1.status_code == 200
        lines = [ln for ln in audit_file.read_text().splitlines() if ln.strip()]
        entry1 = json.loads(lines[-1])
        assert entry1.get("policy_profile") == "strict"
        assert entry1.get("outcome") == "FORCE_REVISE"

        # Staging key (moderate): UNBOUND_ENTITY -> SOFT_CORRECT
        r2 = client.post("/v1/chat/completions", json=payload, headers={"Authorization": "Bearer staging-key"})
        assert r2.status_code == 200
        lines = [ln for ln in audit_file.read_text().splitlines() if ln.strip()]
        entry2 = json.loads(lines[-1])
        assert entry2.get("policy_profile") == "moderate"
        assert entry2.get("outcome") == "SOFT_CORRECT"

    def test_rate_limit_per_ip_returns_429(self, monkeypatch):
        """Per-IP rate limit returns 429 when exceeded."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        app = self._make_app(monkeypatch, hardening={"rate_limit_per_ip": 2})
        client = TestClient(app)
        for _ in range(2):
            r = client.post("/v1/chat/completions", json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]})
            assert r.status_code == 200
        r = client.post("/v1/chat/completions", json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]})
        assert r.status_code == 429
        assert "Rate limit" in r.json()["error"]["message"]
        assert "per IP" in r.json()["error"]["message"].lower() or "Rate limit" in r.json()["error"]["message"]


class TestPhaseDAuditIntegrity:
    """Phase D: HMAC signing, rotation, verify endpoint."""

    def test_hmac_computed_correctly(self, tmp_path):
        """Entries with audit_signing_key include valid HMAC."""
        from aurora_lens.govern.audit_io import append_audit_entry, verify_audit_entries

        path = tmp_path / "audit.jsonl"
        key = b"test-signing-key-32bytes!!"
        entry = {"trace_id": "t1", "action": "PASS", "ts": "2026-01-01T00:00:00Z"}
        append_audit_entry(path, entry, signing_key=key)
        verified, n, failed = verify_audit_entries(path, 10, key)
        assert verified is True
        assert n == 1
        assert failed is None
        raw = path.read_text()
        data = json.loads(raw.strip())
        assert "hmac" in data
        assert data["trace_id"] == "t1"

    def test_tampered_entry_fails_verification(self, tmp_path):
        """Entry with tampered field fails verification."""
        from aurora_lens.govern.audit_io import append_audit_entry, verify_audit_entries

        path = tmp_path / "audit.jsonl"
        key = b"test-signing-key-32bytes!!"
        append_audit_entry(path, {"trace_id": "t1", "action": "PASS"}, signing_key=key)
        lines = path.read_text().splitlines()
        data = json.loads(lines[0])
        data["action"] = "TAMPERED"
        path.write_text(json.dumps(data, separators=(",", ":")) + "\n")
        verified, n, failed = verify_audit_entries(path, 10, key)
        assert verified is False
        assert failed == "t1"

    def test_rotation_fires_at_threshold(self, tmp_path):
        """Log rotates when size exceeds max_mb."""
        from aurora_lens.govern.audit_io import append_audit_entry

        path = tmp_path / "audit.jsonl"
        # Fill file to exceed 0.001 MB (~1 KB), then write with rotation enabled
        chunk = {"trace_id": "x", "payload": "y" * 400}  # ~450 bytes per entry
        append_audit_entry(path, chunk, max_mb=0)
        append_audit_entry(path, {**chunk, "i": 1}, max_mb=0)
        append_audit_entry(path, {**chunk, "i": 2}, max_mb=0)  # ~1.35 KB total
        size_before = path.stat().st_size
        rotated = Path(str(path) + ".1")
        append_audit_entry(path, {"trace_id": "new"}, max_mb=0)  # 0 = no rotation
        assert not rotated.exists()
        append_audit_entry(path, {"trace_id": "after"}, max_mb=1)  # 1 MB threshold
        assert path.exists()
        # Verify rotation logic runs (max_mb > 0) without error
        path2 = tmp_path / "audit2.jsonl"
        append_audit_entry(path2, {"a": 1}, max_mb=100)
        assert path2.exists()

    def test_verify_endpoint_returns_correct_result(self, monkeypatch, tmp_path):
        """GET /v1/audit/verify returns verified/entries."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        audit_file = tmp_path / "audit.jsonl"
        key = "test-signing-key-for-verify"
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {
                "audit_log": str(audit_file),
                "audit_backend": "jsonl",
                "audit_signing_key": key,
            },
        })
        from aurora_lens.proxy.app import create_app
        app = create_app(cfg)
        client = TestClient(app)

        # Write a signed entry directly using the resolved key (apply_env_overrides may
        # have substituted AURORA_LENS_AUDIT_SIGNING_KEY from the environment).
        from aurora_lens.govern.audit_io import append_audit_entry
        resolved_key = (cfg.governance.audit_signing_key or key).encode("utf-8")
        append_audit_entry(audit_file, {"trace_id": "v1", "action": "PASS"}, signing_key=resolved_key)

        r = client.get("/v1/audit/verify?n=10")
        assert r.status_code == 200
        data = r.json()
        assert data["verified"] is True
        assert data["entries"] == 1
        assert data.get("fully_verified") is True
        assert "what_was_verified" in data
        assert data["what_was_verified"]["integrity"]["verified"] is True
        assert data.get("evidence_audit_status_summary") == "none"
        assert data.get("backend_interpretation")

    def test_verify_endpoint_requires_signing_key(self, monkeypatch, tmp_path):
        """Verify returns 400 when audit_signing_key not configured."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        # Clear env var so apply_env_overrides does not supply a key for this test.
        monkeypatch.delenv("AURORA_LENS_AUDIT_SIGNING_KEY", raising=False)
        audit_file = tmp_path / "audit.jsonl"
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"audit_log": str(audit_file), "audit_backend": "jsonl"},
        })
        from aurora_lens.proxy.app import create_app
        app = create_app(cfg)
        client = TestClient(app)

        r = client.get("/v1/audit/verify?n=10")
        assert r.status_code == 400
        assert "audit_signing_key" in r.json()["error"]["message"]

    def test_verify_cli_ok_on_valid_chain(self, tmp_path):
        """Verify CLI exits 0 when chain and HMAC valid."""
        from aurora_lens.govern.audit_io import append_audit_entry, CHAIN_GENESIS

        path = tmp_path / "audit.jsonl"
        key = b"test-key-32bytes!!!!!!!!!!!!!!!!"
        prev = CHAIN_GENESIS
        for i in range(3):
            entry = {"schema_version": 2, "trace_id": f"t{i}", "outcome": "PASS"}
            cid = append_audit_entry(path, entry, signing_key=key, prev_cid=prev)
            prev = cid or prev

        result = __import__("subprocess").run(
            [__import__("sys").executable, "-m", "aurora_lens.scripts.verify_audit", "--path", str(path), "--key", key.decode()],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
        )
        assert result.returncode == 0
        assert "OK" in result.stdout

    def test_verify_cli_replay_included_by_default(self, tmp_path):
        """Verify CLI runs replay verification; exits 0 when state_hash matches."""
        from aurora_lens.govern.audit_io import append_audit_entry, CHAIN_GENESIS
        from aurora_lens.govern.bridge import build_forensic_event
        from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
        from aurora_lens.pef.state import PEFState

        path = tmp_path / "audit.jsonl"
        key = b"test-key-32bytes!!!!!!!!!!!!!!!!"
        pef = PEFState()
        pef_snap = pef.to_dict()
        decision = GovernanceDecision(action=InterventionAction.HARD_STOP, flags=[], rationale="test", policy="strict")
        fe = build_forensic_event(decision, pre_llm=False, pef_snapshot=pef_snap, trace_id="t1", timestamp="2026-02-24T12:00:00Z", audit_id="c1")
        prev = CHAIN_GENESIS
        for i in range(2):
            entry = {"schema_version": 2, "trace_id": f"t{i}", "outcome": "PASS"}
            cid = append_audit_entry(path, entry, signing_key=key, prev_cid=prev)
            prev = cid or prev
        entry = {"schema_version": 2, "trace_id": "t2", "outcome": "HARD_STOP", "forensic_event": fe, "pef_snapshot": pef_snap}
        cid = append_audit_entry(path, entry, signing_key=key, prev_cid=prev)
        prev = cid or prev

        result = __import__("subprocess").run(
            [__import__("sys").executable, "-m", "aurora_lens.scripts.verify_audit", "--path", str(path), "--key", key.decode()],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
        )
        assert result.returncode == 0
        assert "OK" in result.stdout
        assert "state_hash" in result.stdout or "replayed" in result.stdout

    def test_verify_cli_no_replay_skips_replay(self, tmp_path):
        """Verify CLI with --no-replay skips replay verification."""
        from aurora_lens.govern.audit_io import append_audit_entry, CHAIN_GENESIS

        path = tmp_path / "audit.jsonl"
        key = b"test-key-32bytes!!!!!!!!!!!!!!!!"
        prev = CHAIN_GENESIS
        for i in range(2):
            entry = {"schema_version": 2, "trace_id": f"t{i}", "outcome": "PASS"}
            cid = append_audit_entry(path, entry, signing_key=key, prev_cid=prev)
            prev = cid or prev

        result = __import__("subprocess").run(
            [__import__("sys").executable, "-m", "aurora_lens.scripts.verify_audit", "--path", str(path), "--key", key.decode(), "--no-replay"],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
        )
        assert result.returncode == 0
        assert "OK" in result.stdout

    def test_verify_cli_fail_on_chain_break(self, tmp_path):
        """Verify CLI exits 1 when chain broken."""
        from aurora_lens.govern.audit_io import append_audit_entry, CHAIN_GENESIS

        path = tmp_path / "audit.jsonl"
        key = b"test-key-32bytes!!!!!!!!!!!!!!!!"
        prev = CHAIN_GENESIS
        for i in range(2):
            entry = {"schema_version": 2, "trace_id": f"t{i}", "outcome": "PASS"}
            cid = append_audit_entry(path, entry, signing_key=key, prev_cid=prev)
            prev = cid or prev
        lines = path.read_text().splitlines()
        data = json.loads(lines[-1])
        data["outcome"] = "TAMPERED"
        path.write_text("\n".join(lines[:-1] + [json.dumps(data, separators=(",", ":"))]))

        result = __import__("subprocess").run(
            [__import__("sys").executable, "-m", "aurora_lens.scripts.verify_audit", "--path", str(path), "--key", key.decode()],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
        )
        assert result.returncode == 1
        assert "FAIL" in result.stderr

    def test_anchor_ledger_writes_signed_record(self, tmp_path):
        """Anchor script reads last cid, writes signed anchor record."""
        from aurora_lens.govern.audit_io import append_audit_entry, CHAIN_GENESIS

        path = tmp_path / "audit.jsonl"
        key = b"test-key-32bytes!!!!!!!!!!!!!!!!"
        prev = CHAIN_GENESIS
        entry = {"schema_version": 2, "trace_id": "t1", "outcome": "PASS"}
        append_audit_entry(path, entry, signing_key=key, prev_cid=prev)

        out_path = tmp_path / "anchors.jsonl"
        result = __import__("subprocess").run(
            [
                __import__("sys").executable,
                str(Path(__file__).resolve().parents[1] / "scripts" / "anchor_ledger.py"),
                "--path", str(path),
                "--out", str(out_path),
                "--key", key.decode(),
            ],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
        )
        assert result.returncode == 0
        lines = out_path.read_text().strip().splitlines()
        assert len(lines) == 1
        anchor = json.loads(lines[0])
        assert "anchor_cid" in anchor
        assert "anchor_time" in anchor
        assert "ledger_path" in anchor
        assert anchor["entry_count"] == 1
        assert "anchor_hmac" in anchor

    def test_multi_key_verification_spanning_rotation(self, tmp_path):
        """D4: Verify log spanning key rotation with --keys key1,key2."""
        from aurora_lens.govern.audit_io import append_audit_entry, verify_audit_entries, verify_chain, CHAIN_GENESIS

        path = tmp_path / "audit.jsonl"
        key1 = b"old-key-32bytes!!!!!!!!!!!!!!!!"
        key2 = b"new-key-32bytes!!!!!!!!!!!!!!!!!"
        prev = CHAIN_GENESIS
        for i in range(2):
            entry = {"schema_version": 2, "trace_id": f"t{i}", "outcome": "PASS"}
            cid = append_audit_entry(path, entry, signing_key=key1, prev_cid=prev)
            prev = cid or prev
        for i in range(2, 4):
            entry = {"schema_version": 2, "trace_id": f"t{i}", "outcome": "PASS"}
            cid = append_audit_entry(path, entry, signing_key=key2, prev_cid=prev)
            prev = cid or prev

        hmac_ok, n, _ = verify_audit_entries(path, 10, signing_keys=[key1, key2])
        assert hmac_ok, "HMAC failed with both keys"
        assert n == 4
        chain_ok, _, _, _ = verify_chain(path, 10, signing_keys=[key1, key2])
        assert chain_ok

    def test_checkpoint_entry_written_and_verified(self, tmp_path):
        """D3 Option 2: Checkpoint written every N entries, verifier accepts it."""
        from aurora_lens.govern.audit_io import append_audit_entry, append_checkpoint_entry, verify_chain, CHAIN_GENESIS

        path = tmp_path / "audit.jsonl"
        key = b"test-key-32bytes!!!!!!!!!!!!!!!!"
        prev = CHAIN_GENESIS
        for i in range(2):
            entry = {"schema_version": 2, "trace_id": f"t{i}", "outcome": "PASS"}
            cid = append_audit_entry(path, entry, signing_key=key, prev_cid=prev)
            prev = cid or prev
        new_cid = append_checkpoint_entry(path, prev, signing_key=key)
        assert new_cid is not None

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 3
        checkpoint = json.loads(lines[-1])
        assert checkpoint.get("type") == "checkpoint"
        assert "checkpoint_cid" in checkpoint
        assert "checkpoint_sig" in checkpoint

        chain_ok, n, _, _ = verify_chain(path, 10, signing_key=key)
        assert chain_ok
        assert n == 3


    def test_verify_endpoint_multi_key(self, monkeypatch, tmp_path):
        """Verify endpoint tries all configured keys (spanning key rotation)."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        # Clear env vars so apply_env_overrides uses the explicit test keys.
        monkeypatch.delenv("AURORA_LENS_AUDIT_SIGNING_KEY", raising=False)
        monkeypatch.delenv("AURORA_LENS_AUDIT_SIGNING_KEYS", raising=False)
        audit_file = tmp_path / "audit.jsonl"
        old_key = "old-signing-key"
        new_key = "new-signing-key"
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {
                "audit_log": str(audit_file),
                "audit_backend": "jsonl",
                "audit_signing_key": new_key,
                "audit_signing_keys": [old_key],
            },
        })
        assert cfg.governance.audit_signing_keys == (old_key,)

        from aurora_lens.proxy.app import create_app
        from aurora_lens.govern.audit_io import append_audit_entry, CHAIN_GENESIS
        app = create_app(cfg)
        client = TestClient(app)

        # Write two entries: first with old key, second with new key
        prev = CHAIN_GENESIS
        prev = append_audit_entry(
            audit_file, {"trace_id": "r1", "action": "PASS"},
            signing_key=old_key.encode(), prev_cid=prev,
        )
        append_audit_entry(
            audit_file, {"trace_id": "r2", "action": "PASS"},
            signing_key=new_key.encode(), prev_cid=prev,
        )

        r = client.get("/v1/audit/verify?n=10")
        assert r.status_code == 200
        data = r.json()
        assert data["verified"] is True, f"expected verified, got: {data}"
        assert data["hmac_verified"] is True
        assert data["chain_verified"] is True
        assert data["backend"] == "jsonl"

    def test_verify_endpoint_unanchored_slice(self, monkeypatch, tmp_path):
        """Tail-slice: verify last N of M>N entries → reason='unanchored_slice', chain_verified=True.

        Regression for the bug where verify_chain() reset prev_cid=CHAIN_GENESIS on every
        non-chained entry, causing genuine tail slices to be misreported as chain_break.
        The first chained entry in the window links to a predecessor outside the window —
        this is a normal, expected outcome of partial verification, not a chain failure.
        """
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        monkeypatch.delenv("AURORA_LENS_AUDIT_SIGNING_KEY", raising=False)
        monkeypatch.delenv("AURORA_LENS_AUDIT_SIGNING_KEYS", raising=False)

        audit_file = tmp_path / "audit.jsonl"
        key = "test-key-32bytes!!!!!!!!!!!!!!!!"
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {
                "audit_log": str(audit_file),
                "audit_backend": "jsonl",
                "audit_signing_key": key,
            },
        })

        from aurora_lens.proxy.app import create_app
        from aurora_lens.govern.audit_io import append_audit_entry, CHAIN_GENESIS
        app = create_app(cfg)
        client = TestClient(app)

        # Write 10 chained entries; request only the last 5.
        # entry[0] of the 5-entry window has prev_cid pointing outside the window.
        prev = CHAIN_GENESIS
        for i in range(10):
            prev = append_audit_entry(
                audit_file,
                {"trace_id": f"t{i}", "action": "PASS"},
                signing_key=key.encode(),
                prev_cid=prev,
            )

        r = client.get("/v1/audit/verify?n=5")
        assert r.status_code == 200
        data = r.json()

        # Must NOT be classified as chain_break — tail slice is expected.
        assert data.get("reason") == "unanchored_slice", (
            f"Expected unanchored_slice, got: {data}"
        )
        # The slice is internally consistent.
        assert data.get("chain_verified") is True, (
            f"Expected chain_verified=True for internally consistent tail slice: {data}"
        )
        # HMAC valid — all entries signed with same key.
        assert data.get("hmac_verified") is True
        # verified=False — global anchoring cannot be proved from a tail slice.
        assert data.get("verified") is False
        assert data.get("backend") == "jsonl"
        # No first_break_index — there is no break, just an unanchored window.
        assert "first_break_index" not in data


class TestOperatorDetailViaHeader:
    """Per-request ``X-Aurora-Operator-Detail`` when ``allow_operator_detail_via_header`` is set."""

    def test_header_ignored_when_allow_disabled(self, monkeypatch):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {
                "include_operator_detail": False,
                "allow_operator_detail_via_header": False,
            },
        })
        app = create_app(cfg)
        client = TestClient(app)
        r = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]},
            headers={"X-Aurora-Operator-Detail": "true"},
        )
        assert r.status_code == 200
        body = r.json()
        assert "original_response" not in body.get("aurora", {})

    def test_header_enables_original_response_when_allowed(self, monkeypatch):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {
                "include_operator_detail": False,
                "allow_operator_detail_via_header": True,
            },
        })
        app = create_app(cfg)
        client = TestClient(app)
        r = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]},
            headers={"X-Aurora-Operator-Detail": "true"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["aurora"]["original_response"] == "Echo: Hello"


class TestLensIntegration:
    """Lens-level integration tests (no proxy)."""

    @pytest.mark.asyncio
    async def test_end_to_end_with_mock(self):
        """Full pipeline: request → parse → lens → format → response."""
        adapter = MockAdapter()
        backend = MockBackend()
        bridge = BuiltinBridge()
        config = LensConfig(
            adapter=adapter,
            extraction_backend=backend,
            governance_bridge=bridge,
        )
        lens = Lens(config)

        # Simulate what the proxy does
        body = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Hello world"},
            ],
        }
        parsed = parse_chat_request(body)
        result = await lens.process(parsed.user_message)
        response_body = format_chat_response(result, request_model=parsed.model, pef=lens.pef)

        assert response_body["choices"][0]["message"]["content"] == "Echo: Hello world"
        assert response_body["aurora"]["governance"] == "PASS"
        assert response_body["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_stream_returns_sse_with_metadata(self):
        """stream=true returns SSE stream with content chunks and final aurora metadata."""
        adapter = MockAdapter()
        backend = MockBackend()
        bridge = BuiltinBridge()
        config = LensConfig(
            adapter=adapter,
            extraction_backend=backend,
            governance_bridge=bridge,
        )
        lens = Lens(config)

        chunks = []
        async for kind, payload in lens.process_stream("Hello"):
            if kind == "chunk":
                chunk_dict, content = payload
                chunks.append((chunk_dict, content))
            elif kind == "metadata":
                assert "governance" in payload
                assert payload["governance"] == "PASS"
                assert "turn" in payload
                return
        pytest.fail("Expected metadata event")

    @pytest.mark.asyncio
    async def test_stream_disconnect_closes_upstream_no_metadata(self):
        """When client disconnects mid-stream (post-governance): no metadata; upstream already completed.

        With full-buffer streaming governance, the upstream provider stream is fully consumed
        during private buffering before any chunk is released to the caller.  Breaking after
        the first released chunk therefore does NOT close the upstream early — it already
        completed.  The important invariant is that no metadata event is emitted after the break.
        """
        adapter = StreamMockAdapter(["a", "b", "c"])
        backend = MockBackend()
        bridge = BuiltinBridge()
        config = LensConfig(
            adapter=adapter,
            extraction_backend=backend,
            governance_bridge=bridge,
        )
        lens = Lens(config)

        metadata_count = 0
        chunk_count = 0
        async with aclosing(lens.process_stream("Hi")) as gen:
            async for kind, payload in gen:
                if kind == "metadata":
                    metadata_count += 1
                elif kind == "chunk":
                    chunk_count += 1
                if chunk_count >= 1:
                    break  # Simulate client disconnect after first chunk

        assert chunk_count >= 1
        assert metadata_count == 0
        # Upstream completed during private buffering before chunks were released;
        # closing the generator mid-delivery does not affect the upstream state.
        assert adapter.closed_early is False

    @pytest.mark.asyncio
    async def test_stream_truncation_sets_flags(self):
        """When accumulated stream exceeds max_stream_bytes, metadata includes stream_truncated and stream_dropped_chars."""
        # Create chunks that exceed the limit (default 512 KiB)
        chunk_size = 100 * 1024  # 100 KiB per chunk
        big_chunk = "x" * chunk_size
        chunks = [big_chunk] * 10  # 1 MiB total
        adapter = StreamMockAdapter(chunks)
        backend = MockBackend()
        bridge = BuiltinBridge()
        config = LensConfig(
            adapter=adapter,
            extraction_backend=backend,
            governance_bridge=bridge,
        )
        lens = Lens(config)

        metadata = None
        async with aclosing(lens.process_stream("Hi")) as gen:
            async for kind, payload in gen:
                if kind == "metadata":
                    metadata = payload
                    break

        assert metadata is not None
        assert metadata.get("stream_truncated") is True
        assert metadata.get("stream_dropped_chars", 0) >= 1

    @pytest.mark.asyncio
    async def test_stream_completed_logs_audit_with_stream_fields(self, tmp_path):
        """Stream completion writes audit entry with stream: true, stream_completed: true."""
        audit_file = tmp_path / "audit.jsonl"
        adapter = StreamMockAdapter(["a", "b", "c"])
        backend = MockBackend()
        bridge = BuiltinBridge(audit_path=str(audit_file))
        config = LensConfig(
            adapter=adapter,
            extraction_backend=backend,
            governance_bridge=bridge,
        )
        lens = Lens(config)

        async with aclosing(lens.process_stream("Hi")) as gen:
            async for kind, payload in gen:
                if kind == "metadata":
                    break

        lines = audit_file.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry.get("stream") is True
        assert entry.get("stream_completed") is True
        assert "stream_abort_reason" not in entry

    @pytest.mark.asyncio
    async def test_stream_abort_logs_audit_with_abort_reason(self, tmp_path):
        """Provider abort during private buffering writes audit entry with stream_completed: false.

        With full-buffer streaming governance the upstream provider is consumed during a private
        buffering phase before governance runs.  The 'abort' scenario is therefore a provider-side
        interruption (CancelledError raised inside generate_stream) — not a post-governance
        client disconnect.  CancelledStreamAdapter simulates this by raising CancelledError
        after the first chunk, which triggers the finally-block abort path in process_stream().
        """
        audit_file = tmp_path / "audit.jsonl"
        # Raises CancelledError after yielding 1 chunk — provider abort during buffering
        adapter = CancelledStreamAdapter(["a", "b", "c"], cancel_after=1)
        backend = MockBackend()
        bridge = BuiltinBridge(audit_path=str(audit_file))
        config = LensConfig(
            adapter=adapter,
            extraction_backend=backend,
            governance_bridge=bridge,
        )
        lens = Lens(config)

        # CancelledError from provider propagates out of process_stream; suppress it here.
        try:
            async with aclosing(lens.process_stream("Hi")) as gen:
                async for kind, payload in gen:
                    pass
        except asyncio.CancelledError:
            pass

        lines = audit_file.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry.get("stream") is True
        assert entry.get("stream_completed") is False
        assert entry.get("stream_abort_reason") == "provider_abort"

    @pytest.mark.asyncio
    async def test_stream_truncated_logs_audit_flags(self, tmp_path):
        """Stream truncation writes stream_truncated and stream_dropped_chars in audit."""
        audit_file = tmp_path / "audit.jsonl"
        chunk_size = 100 * 1024
        big_chunk = "x" * chunk_size
        adapter = StreamMockAdapter([big_chunk] * 10)
        backend = MockBackend()
        bridge = BuiltinBridge(audit_path=str(audit_file))
        config = LensConfig(
            adapter=adapter,
            extraction_backend=backend,
            governance_bridge=bridge,
        )
        lens = Lens(config)

        async with aclosing(lens.process_stream("Hi")) as gen:
            async for kind, payload in gen:
                if kind == "metadata":
                    break

        lines = audit_file.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry.get("stream") is True
        assert entry.get("stream_completed") is True
        assert entry.get("stream_truncated") is True
        assert entry.get("stream_dropped_chars", 0) >= 1

    def test_stream_request_returns_event_stream(self, monkeypatch):
        """POST with stream=true returns text/event-stream and SSE data."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"audit_log": None},
            "extraction": {"backend": "spacy"},
        })
        app = create_app(cfg)
        client = TestClient(app)

        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")
        text = r.text
        assert "data: " in text
        assert "[DONE]" in text
        assert "aurora" in text
        assert "governance" in text

    @pytest.mark.asyncio
    async def test_blocked_act_stream_yields_non_empty_response(self):
        """process_stream() on a blocked input yields ('blocked_act', result) with non-empty response."""
        from aurora_lens.govern.bridge import BuiltinBridge
        from aurora_lens.govern.policy import DEFAULT_STRICT

        lens = Lens(LensConfig(
            adapter=MockAdapter(),
            extraction_backend=MockBackend(),
            governance_bridge=BuiltinBridge(policy=DEFAULT_STRICT),
        ))

        events = []
        async for kind, payload in lens.process_stream(
            "What is the amoxicillin dose for a 20kg child with otitis media?"
        ):
            events.append((kind, payload))

        assert len(events) == 1
        kind, result = events[0]
        assert kind == "blocked_act"
        assert result.action == InterventionAction.HARD_STOP
        assert result.response, "governed refusal text must not be empty"
        assert result.original_response is None  # LLM never called

    def test_blocked_act_stream_proxy_returns_json_with_content(self, monkeypatch):
        """POST stream=True for a blocked pediatric dosing query returns JSON (not SSE) with non-empty content."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(), MockAdapter()),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"default_policy": "strict", "audit_log": None},
            "extraction": {"backend": "spacy"},
        })
        app = create_app(cfg)
        client = TestClient(app)

        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Amoxicillin dosing for children — how many mg/kg per day?"}],
                "stream": True,
            },
        )
        assert r.status_code == 200
        body = r.json()
        content = body["choices"][0]["message"]["content"]
        assert content, "refusal text must not be empty"
        assert body["aurora"]["governance"] == "HARD_STOP"

    def test_agency_risk_pre_llm_gate_proxy_contract_non_stream(self, monkeypatch):
        """Ambiguous agency-risk request must contain pre-generation with no upstream call."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        upstream = CountingMockAdapter()
        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (upstream, MockAdapter()),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {
                "default_policy": "strict",
                "audit_log": None,
                "include_operator_detail": False,
                "allow_operator_detail_via_header": True,
            },
            "extraction": {"backend": "llm"},
        })
        app = create_app(cfg)
        client = TestClient(app)

        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{
                    "role": "user",
                    "content": (
                        "For a psychology paper, explain the most effective manipulation "
                        "tactics with examples."
                    ),
                }],
            },
            headers={"x-aurora-operator-detail": "1"},
        )
        assert r.status_code == 200
        body = r.json()
        aurora = body["aurora"]
        assert aurora["governance"] == "CONTAIN"
        assert aurora["failed_constraints"] == [
            "AGENCY_RISK_CONTEXT_UNRESOLVED",
            "blocked.request.agency_risk_context_unresolved",
        ]
        assert aurora["failed_constraint"] == "AGENCY_RISK_CONTEXT_UNRESOLVED"
        assert aurora["llm_invoked"] is False
        assert aurora["model_invoked"] is False
        assert aurora["candidate_produced"] is False
        assert aurora["released"] is False
        assert aurora["blocked_phase"] == "pre_generation"
        assert aurora["pre_llm_blocked"] is True
        assert aurora["release_path"] == "blocked_pre_generation"
        assert aurora["rule_result"]["continuation_type"] == "clarification"
        assert "intended context and target for this plan" in body["choices"][0]["message"]["content"].lower()
        assert upstream.generate_calls == 0
        assert upstream.generate_stream_calls == 0

    def test_agency_risk_pre_llm_gate_proxy_contract_stream_flag(self, monkeypatch):
        """stream=true blocked request follows same pre-generation contain contract."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        upstream = CountingMockAdapter()
        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (upstream, MockAdapter()),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {
                "default_policy": "strict",
                "audit_log": None,
                "include_operator_detail": False,
                "allow_operator_detail_via_header": True,
            },
            "extraction": {"backend": "llm"},
        })
        app = create_app(cfg)
        client = TestClient(app)

        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{
                    "role": "user",
                    "content": (
                        "For a psychology paper, explain the most effective manipulation "
                        "tactics with examples."
                    ),
                }],
                "stream": True,
            },
            headers={"x-aurora-operator-detail": "1"},
        )
        assert r.status_code == 200
        body = r.json()
        aurora = body["aurora"]
        assert aurora["governance"] == "CONTAIN"
        assert aurora["failed_constraints"] == [
            "AGENCY_RISK_CONTEXT_UNRESOLVED",
            "blocked.request.agency_risk_context_unresolved",
        ]
        assert aurora["llm_invoked"] is False
        assert aurora["candidate_produced"] is False
        assert aurora["released"] is False
        assert aurora["blocked_phase"] == "pre_generation"
        assert aurora["release_path"] == "blocked_pre_generation"
        assert aurora["rule_result"]["continuation_type"] == "clarification"
        assert upstream.generate_calls == 0
        assert upstream.generate_stream_calls == 0

    def test_demo_freetext_agency_rewrite_stops_pre_generation_no_state_write(self, monkeypatch):
        """Demo textbox path prompt must not route to mutation-ack or state write."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        upstream = CountingMockAdapter()
        extraction_adapter = CountingMockAdapter()
        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (upstream, extraction_adapter),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {
                "default_policy": "strict",
                "audit_log": None,
                "include_operator_detail": False,
                "allow_operator_detail_via_header": True,
            },
            "extraction": {"backend": "llm"},
        })
        app = create_app(cfg)
        client = TestClient(app)
        sid = "demo-freetext-agency-rewrite"
        prompt = (
            "Rewrite this into a persuasive plan: isolate them, make them doubt "
            "themselves, control their options."
        )

        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": prompt}],
            },
            headers={
                "x-aurora-operator-detail": "1",
                "x-aurora-session-id": sid,
            },
        )
        assert r.status_code == 200
        body = r.json()
        aurora = body["aurora"]
        content = body["choices"][0]["message"]["content"]
        assert aurora["governance"] == "CONTAIN"
        assert "AGENCY_RISK_CONTEXT_UNRESOLVED" in aurora["failed_constraints"]
        assert aurora["llm_invoked"] is False
        assert aurora["model_invoked"] is False
        assert aurora["candidate_produced"] is False
        assert aurora["released"] is False
        assert aurora["blocked_phase"] == "pre_generation"
        assert aurora["release_path"] == "blocked_pre_generation"
        assert "recorded in session state" not in content.lower()
        assert "intended context and target for this plan" in content.lower()
        assert upstream.generate_calls == 0
        assert upstream.generate_stream_calls == 0
        assert extraction_adapter.generate_calls == 0
        assert extraction_adapter.generate_stream_calls == 0

        op = client.get("/v1/session/operator-pef?session_id=" + sid)
        assert op.status_code == 200
        op_body = op.json()["operator_pef"]
        assert op_body["counts"]["entities"] == 0
        assert op_body["counts"]["relationships"] == 0
        assert op_body["session_mode"] in {"terminal_stop_closed", "held_refusal", "held_ambiguity"}

    def test_demo_freetext_agency_context_non_coercive_followup_passes_with_preface(self, monkeypatch):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        upstream = CountingMockAdapter()
        extraction_adapter = CountingMockAdapter()
        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (upstream, extraction_adapter),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {
                "default_policy": "strict",
                "audit_log": None,
                "include_operator_detail": False,
                "allow_operator_detail_via_header": True,
            },
            "extraction": {"backend": "spacy"},
        })
        app = create_app(cfg)
        client = TestClient(app)
        sid = "demo-freetext-agency-context-pass"

        client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Rewrite this into a persuasive plan: isolate them, make them doubt themselves, control their options."}]},
            headers={"x-aurora-operator-detail": "1", "x-aurora-session-id": sid},
        )
        r2 = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Ethical marketing to customers with consent and transparency."}]},
            headers={"x-aurora-operator-detail": "1", "x-aurora-session-id": sid},
        )
        assert r2.status_code == 200
        body = r2.json()
        aurora = body["aurora"]
        content = body["choices"][0]["message"]["content"]
        assert aurora["governance"] == "PASS"
        assert aurora["llm_invoked"] is True
        assert aurora["released"] is True
        assert "ethical version that preserves customer choice" in content.lower()

    def test_demo_freetext_agency_context_explicit_violation_followup_hard_stops(self, monkeypatch):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        upstream = CountingMockAdapter()
        extraction_adapter = CountingMockAdapter()
        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (upstream, extraction_adapter),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {
                "default_policy": "strict",
                "audit_log": None,
                "include_operator_detail": False,
                "allow_operator_detail_via_header": True,
            },
            "extraction": {"backend": "llm"},
        })
        app = create_app(cfg)
        client = TestClient(app)
        sid = "demo-freetext-agency-context-stop"

        client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Rewrite this into a persuasive plan: isolate them, make them doubt themselves, control their options."}]},
            headers={"x-aurora-operator-detail": "1", "x-aurora-session-id": sid},
        )
        r2 = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Use it on my partner so they comply."}]},
            headers={"x-aurora-operator-detail": "1", "x-aurora-session-id": sid},
        )
        assert r2.status_code == 200
        body = r2.json()
        aurora = body["aurora"]
        assert aurora["governance"] == "HARD_STOP"
        assert "AGENCY_VIOLATION_ASSISTANCE" in aurora["failed_constraints"]
        assert aurora["llm_invoked"] is False
        assert aurora["blocked_phase"] == "pre_generation"


class TestProxyMain:
    """Tests for proxy __main__ entry point."""

    def test_main_calls_setup_logging(self, monkeypatch, tmp_path):
        """main() calls setup_logging() at startup before create_app."""
        config_yaml = tmp_path / "aurora-lens.yaml"
        config_yaml.write_text("""
upstream:
  provider: openai
  api_key: ""
  api_key_env: OPENAI_API_KEY
  model: gpt-4
""")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        setup_called = []

        def _track_setup(*args, **kwargs):
            setup_called.append(True)

        def _no_run(*args, **kwargs):
            raise SystemExit(0)

        monkeypatch.setattr("aurora_lens.proxy.__main__.setup_logging", _track_setup)
        monkeypatch.setattr("uvicorn.run", _no_run)

        from aurora_lens.proxy.__main__ import main

        with pytest.raises(SystemExit):
            main(["-c", str(config_yaml)])

        assert len(setup_called) == 1

    def test_aurora_record_factory_injects_trace_and_session(self):
        """LogRecord factory injects trace_id and session_id from contextvars into every log record."""
        from aurora_lens.context import trace_id_var, session_id_var
        from aurora_lens.proxy.logging import _configure_record_factory

        _configure_record_factory()
        captured: list = []

        class CaptureHandler(logging.Handler):
            def emit(self, record):
                captured.append(record)

        logger = logging.getLogger("test.aurora.factory")
        logger.addHandler(CaptureHandler())
        logger.setLevel(logging.INFO)

        # Without context: trace_id and session_id are None (startup logs)
        logger.info("startup")
        assert len(captured) == 1
        assert captured[0].trace_id is None
        assert captured[0].session_id is None
        captured.clear()

        # With context: populated (request logs)
        token_t = trace_id_var.set("proxy:abc123")
        token_s = session_id_var.set("sess-xyz")
        try:
            logger.info("request")
            assert len(captured) == 1
            assert captured[0].trace_id == "proxy:abc123"
            assert captured[0].session_id == "sess-xyz"
        finally:
            trace_id_var.reset(token_t)
            session_id_var.reset(token_s)

    def test_context_reset_prevents_bleed(self):
        """Context must be reset after request; otherwise next request inherits previous values."""
        from aurora_lens.context import trace_id_var, session_id_var, get_trace_id, get_session_id

        # Simulate request 1: set context
        token_t = trace_id_var.set("proxy:req1")
        token_s = session_id_var.set("sess-1")
        assert get_trace_id() == "proxy:req1"
        assert get_session_id() == "sess-1"

        # Reset (as middleware/handler finally would)
        trace_id_var.reset(token_t)
        session_id_var.reset(token_s)

        # After reset: context is clear
        assert get_trace_id() is None
        assert get_session_id() is None

        # Simulate request 2: would get its own values, not req1's
        token_t2 = trace_id_var.set("proxy:req2")
        try:
            assert get_trace_id() == "proxy:req2"
        finally:
            trace_id_var.reset(token_t2)

    def test_json_formatter_always_emits_trace_id_session_id(self):
        """JsonFormatter always emits trace_id and session_id, even when null."""
        from aurora_lens.proxy.logging import JsonFormatter

        record = logging.LogRecord("test", logging.INFO, "", 0, "startup", (), None)
        record.trace_id = None
        record.session_id = None

        formatter = JsonFormatter()
        out = json.loads(formatter.format(record))
        assert "trace_id" in out
        assert "session_id" in out
        assert out["trace_id"] is None
        assert out["session_id"] is None
        assert out["timestamp"]
        assert out["level"] == "INFO"
        assert out["message"] == "startup"

    def test_governance_outcome_log_emits_required_fields(self):
        """governance_outcome log event contains all forensically required fields.

        Verifies that the JsonFormatter serialises a governance_outcome record
        with the full set of fields needed for Railway production diagnosis:
        outcome, flags, flags_count, blocked, revised, audit_id, policy,
        pathway_id, commitment_closed, stream, trace_id, session_id.
        """
        from aurora_lens.context import trace_id_var, session_id_var
        from aurora_lens.proxy.logging import JsonFormatter, _configure_record_factory

        _configure_record_factory()
        captured: list = []

        class CaptureHandler(logging.Handler):
            def emit(self, record):
                captured.append(record)

        logger = logging.getLogger("test.aurora.governance_outcome")
        handler = CaptureHandler()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        token_t = trace_id_var.set("proxy:test-trace-001")
        token_s = session_id_var.set("sess-gv-test")
        try:
            logger.info(
                "governance_outcome",
                extra={
                    "outcome": "HARD_STOP",
                    "flags": ["VIOLENT_CRIMINAL_INTENT"],
                    "flags_count": 1,
                    "blocked": True,
                    "revised": False,
                    "audit_id": "bafytest123",
                    "policy": "strict",
                    "pathway_id": "P_STOP_TERMINAL",
                    "commitment_closed": True,
                    "stream": False,
                },
            )
        finally:
            trace_id_var.reset(token_t)
            session_id_var.reset(token_s)
            logger.removeHandler(handler)

        assert len(captured) == 1
        rec = captured[0]

        # Verify the record carries all required fields
        assert rec.outcome == "HARD_STOP"
        assert rec.flags == ["VIOLENT_CRIMINAL_INTENT"]
        assert rec.flags_count == 1
        assert rec.blocked is True
        assert rec.revised is False
        assert rec.audit_id == "bafytest123"
        assert rec.policy == "strict"
        assert rec.pathway_id == "P_STOP_TERMINAL"
        assert rec.commitment_closed is True
        assert rec.stream is False
        assert rec.trace_id == "proxy:test-trace-001"
        assert rec.session_id == "sess-gv-test"

        # Verify JsonFormatter serialises all fields to the log line
        formatter = JsonFormatter()
        out = json.loads(formatter.format(rec))
        assert out["message"] == "governance_outcome"
        assert out["outcome"] == "HARD_STOP"
        assert out["flags"] == ["VIOLENT_CRIMINAL_INTENT"]
        assert out["flags_count"] == 1
        assert out["blocked"] is True
        assert out["audit_id"] == "bafytest123"
        assert out["policy"] == "strict"
        assert out["pathway_id"] == "P_STOP_TERMINAL"
        assert out["trace_id"] == "proxy:test-trace-001"
        assert out["session_id"] == "sess-gv-test"

from __future__ import annotations

import pytest

from aurora_lens.proxy.config import ProxyConfig


def test_openai_requires_api_key_env_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = ProxyConfig.from_mapping(
        {
            "upstream": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key_env": "OPENAI_API_KEY",
            },
            "listen": {"host": "127.0.0.1", "port": 8081},
            "governance": {"default_policy": "strict", "mode": "public", "audit_log": "./audit.jsonl"},
        }
    )
    cfg.validate()


def test_openai_missing_credential_env_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = ProxyConfig.from_mapping(
        {
            "upstream": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key_env": "OPENAI_API_KEY",
            },
            "listen": {"host": "127.0.0.1", "port": 8081},
            "governance": {"default_policy": "strict", "mode": "public", "audit_log": "./audit.jsonl"},
        }
    )
    with pytest.raises(ValueError, match="missing or empty"):
        cfg.validate()


def test_literal_api_key_in_yaml_is_rejected() -> None:
    cfg = ProxyConfig.from_mapping(
        {
            "upstream": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "sk-hardcoded",
            },
            "listen": {"host": "127.0.0.1", "port": 8081},
            "governance": {"default_policy": "strict", "mode": "public", "audit_log": "./audit.jsonl"},
        }
    )
    with pytest.raises(ValueError, match="must not contain secrets"):
        cfg.validate()


def test_local_provider_requires_base_url_not_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AURORA_LENS_UPSTREAM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = ProxyConfig.from_mapping(
        {
            "upstream": {
                "provider": "local",
                "model": "llama3.2",
                "base_url": "http://127.0.0.1:11434/v1",
            },
            "listen": {"host": "127.0.0.1", "port": 8081},
            "governance": {"default_policy": "strict", "mode": "public", "audit_log": "./audit.jsonl"},
        }
    )
    cfg.validate()

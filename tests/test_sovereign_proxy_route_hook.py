"""Tests for Track B Phase 3B — proxy auto-hook and provider-route construction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aurora_lens.config import LensConfig
from aurora_lens.context import request_metadata_var
from aurora_lens.interpret.spacy_backend import SpacyBackend
from aurora_lens.lens import Lens
from aurora_lens.proxy.config import ProxyConfig
from aurora_lens.proxy.provider_route_hook import apply_sovereign_provider_route_hook
from aurora_lens.request_metadata import RequestMetadata
from aurora_lens.sovereign.provider_registry import (
    SovereignProviderRegistry,
    reject_missing_provider_route,
)
from aurora_lens.sovereign.provider_state import PROVIDER_UNAVAILABLE, VALIDATED_CURRENT
from aurora_lens.sovereign.refusal_templates import (
    PROVIDER_ROUTE_REQUIRED_TEMPLATE_KEY,
    refusal_for_evaluation,
)
from aurora_lens.sovereign.route_config import (
    SovereignRouteConfig,
    build_provider_route_request,
    build_sovereign_registry,
    parse_sovereign_route_config,
)
from tests.test_lens import MockAdapter
from tests.test_sovereign_failover_bridge import (
    ALTERNATE_ID,
    PRIMARY_ID,
    TASK_DOMAIN,
    TASK_GRADE,
    _certified_alternate_profile,
    _route_request,
)


def _proxy_cfg(**sovereign_overrides: object) -> ProxyConfig:
    sovereign = {
        "enabled": True,
        "enforce_provider_route": True,
        "primary_provider_id": PRIMARY_ID,
        "primary_state": PROVIDER_UNAVAILABLE,
        "alternate_provider_id": ALTERNATE_ID,
        "default": {
            "task_domain": TASK_DOMAIN,
            "consequence_grade": TASK_GRADE,
            "data_class": "internal",
        },
        "domain_policies": {
            "finance": {
                "task_domain": "finance",
                "consequence_grade": "critical",
                "data_class": "restricted",
            },
        },
        "profiles": [
            {
                "provider_id": ALTERNATE_ID,
                "provider_name": "Local Mistral",
                "model_id": "mistral-7b",
                "endpoint_url": "http://localhost:8080/v1",
                "hosting_jurisdiction": "AU",
                "data_boundary": "local_au",
                "allowed_data_classes": ["public", "internal", "restricted"],
                "permitted_domains": ["general", "software", "internal_ops"],
                "max_consequence_grade": "high",
                "supports_tools": True,
                "supports_structured_output": True,
                "context_window_tokens": 32768,
                "last_validated_at": "2026-06-25T10:00:00+10:00",
                "status": VALIDATED_CURRENT,
            }
        ],
    }
    sovereign.update(sovereign_overrides)
    return ProxyConfig.from_mapping({
        "upstream": {"provider": "anthropic", "api_key": "sk-test", "model": "claude-haiku"},
        "governance": {"default_policy": "strict", "audit_log": None},
        "extraction": {"backend": "spacy"},
        "sovereign": sovereign,
    })


class TestSovereignRouteConfigParsing:
    def test_parse_minimal_sovereign_block(self):
        cfg = parse_sovereign_route_config({
            "enabled": True,
            "enforce_provider_route": True,
            "primary_provider_id": PRIMARY_ID,
            "default": {"task_domain": "general", "consequence_grade": "low"},
        })
        assert cfg.enabled is True
        assert cfg.enforce_provider_route is True
        assert cfg.primary_provider_id == PRIMARY_ID
        assert cfg.default_policy.task_domain == "general"

    def test_upstream_provider_id_mapping(self):
        cfg = parse_sovereign_route_config({
            "enabled": True,
            "upstream_provider_ids": {
                "anthropic:claude-haiku": PRIMARY_ID,
            },
        })
        route = build_provider_route_request(
            cfg,
            operator_domain="general",
            upstream_provider="anthropic",
            upstream_model="claude-haiku",
        )
        assert route is not None
        assert route.primary_provider_id == PRIMARY_ID


class TestProviderRouteConstruction:
    def test_domain_policy_lookup(self):
        cfg = _proxy_cfg().sovereign
        route = build_provider_route_request(
            cfg,
            operator_domain="finance",
            upstream_provider="anthropic",
            upstream_model="claude-haiku",
        )
        assert route is not None
        assert route.task_domain == "finance"
        assert route.consequence_grade == "critical"
        assert route.data_class == "restricted"

    def test_returns_none_when_disabled(self):
        cfg = SovereignRouteConfig(enabled=False, primary_provider_id=PRIMARY_ID)
        assert build_provider_route_request(
            cfg,
            operator_domain="general",
            upstream_provider="anthropic",
            upstream_model="claude-haiku",
        ) is None

    def test_returns_none_without_primary_mapping(self):
        cfg = SovereignRouteConfig(enabled=True)
        assert build_provider_route_request(
            cfg,
            operator_domain="general",
            upstream_provider="anthropic",
            upstream_model="claude-haiku",
        ) is None


class TestProxyProviderRouteHook:
    def test_host_supplied_route_wins(self):
        cfg = _proxy_cfg()
        host_route = _route_request(primary_state=VALIDATED_CURRENT)
        meta = RequestMetadata(provider_route=host_route)
        out = apply_sovereign_provider_route_hook(
            cfg,
            meta,
            operator_domain="finance",
            request_model="claude-haiku",
        )
        assert out is not None
        assert out.provider_route == host_route
        assert out.provider_route.primary_state == VALIDATED_CURRENT

    def test_auto_attaches_when_missing(self):
        cfg = _proxy_cfg()
        out = apply_sovereign_provider_route_hook(
            cfg,
            None,
            operator_domain="general",
            request_model="claude-haiku",
        )
        assert out is not None
        assert out.provider_route is not None
        assert out.provider_route.primary_provider_id == PRIMARY_ID
        assert out.provider_route.task_domain == TASK_DOMAIN

    def test_no_op_when_sovereign_disabled(self):
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "mock", "api_key": "", "model": "mock"},
            "governance": {"audit_log": None},
        })
        out = apply_sovereign_provider_route_hook(
            cfg,
            None,
            operator_domain="general",
            request_model="mock",
        )
        assert out is None


class TestLensEnforceMissingProviderRoute:
    @pytest.mark.asyncio
    async def test_blocks_when_enforced_and_route_missing(self, tmp_path: Path):
        audit_path = tmp_path / "audit.jsonl"
        registry = SovereignProviderRegistry([_certified_alternate_profile()])
        adapter = MockAdapter(responses=["unused"])
        from aurora_lens.govern.bridge import BuiltinBridge

        lens = Lens(
            LensConfig(
                adapter=adapter,
                extraction_backend=SpacyBackend(model="en_core_web_sm"),
                sovereign_provider_registry=registry,
                sovereign_enforce_provider_route=True,
                governance_bridge=BuiltinBridge(audit_path=str(audit_path)),
            )
        )
        tok = request_metadata_var.set(RequestMetadata())
        try:
            result = await lens.process("Summarise the attached notes.")
        finally:
            request_metadata_var.reset(tok)

        assert "blocked before model execution" in result.response.lower()
        lines = [json.loads(ln) for ln in audit_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert lines
        pr = lines[-1].get("provider_route") or {}
        assert pr.get("refusal_template_key") == PROVIDER_ROUTE_REQUIRED_TEMPLATE_KEY

    @pytest.mark.asyncio
    async def test_no_block_when_enforcement_off_and_route_missing(self):
        registry = SovereignProviderRegistry([_certified_alternate_profile()])
        adapter = MockAdapter(responses=["Summary complete."])
        lens = Lens(
            LensConfig(
                adapter=adapter,
                extraction_backend=SpacyBackend(model="en_core_web_sm"),
                sovereign_provider_registry=registry,
                sovereign_enforce_provider_route=False,
            )
        )
        tok = request_metadata_var.set(RequestMetadata())
        try:
            result = await lens.process("Summarise the attached notes.")
        finally:
            request_metadata_var.reset(tok)
        assert "Summary complete." in result.response


class TestRefusalTemplateForMissingRoute:
    def test_provider_route_required_template(self):
        refusal = refusal_for_evaluation(reject_missing_provider_route())
        assert refusal.template_key == PROVIDER_ROUTE_REQUIRED_TEMPLATE_KEY
        assert refusal.machine_reason == "provider_route_metadata_required"


class TestProxyIntegrationAutoRoute:
    def test_proxy_auto_route_refused_failover_without_host_metadata(self, monkeypatch, tmp_path: Path):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("starlette testclient unavailable")

        from aurora_lens.proxy.app import create_app
        from dataclasses import replace

        call_count = {"n": 0}

        class CountingAdapter(MockAdapter):
            async def generate(self, messages, **kwargs):
                call_count["n"] += 1
                return await super().generate(messages, **kwargs)

        def _mock_adapters(_cfg):
            adapter = CountingAdapter(responses=["unused"])
            return adapter, adapter

        monkeypatch.setattr("aurora_lens.proxy.app._build_provider_adapters", _mock_adapters)

        audit_path = tmp_path / "audit.jsonl"
        base = _proxy_cfg(profiles=[])
        cfg = replace(
            base,
            governance=replace(
                base.governance,
                audit_log=str(audit_path),
                audit_backend="jsonl",
            ),
        )
        app = create_app(cfg)
        client = TestClient(app)

        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-haiku",
                "messages": [{"role": "user", "content": "Summarise the attached notes."}],
            },
        )
        assert r.status_code == 200
        body = r.json()
        content = body["choices"][0]["message"]["content"]
        assert "blocked before model execution" in content.lower()
        assert call_count["n"] == 0

        lines = [json.loads(ln) for ln in audit_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert lines
        pr = lines[-1]["provider_route"]
        assert pr["failover_result"] == "refused"
        assert pr["adapter_called"] is False
        assert pr["requested_provider_id"] == PRIMARY_ID


class TestBuildSovereignRegistry:
    def test_registry_from_profiles(self):
        profile = _certified_alternate_profile()
        cfg = SovereignRouteConfig(enabled=True, profiles=(profile,))
        registry = build_sovereign_registry(cfg)
        assert registry is not None
        assert registry.get_profile(ALTERNATE_ID) is not None

    def test_disabled_returns_none(self):
        assert build_sovereign_registry(SovereignRouteConfig(enabled=False)) is None

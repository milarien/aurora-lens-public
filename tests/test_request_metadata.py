"""Host request_metadata envelope: parse, proxy context, Lens visibility."""

import pytest

pytest.importorskip("fastapi")

from tests.skip_reasons import SKIP_STARLETTE_HTTP_TESTCLIENT

from aurora_lens.adapters.base import LLMAdapter, AdapterResponse
from aurora_lens.context import (
    get_request_metadata,
    governance_mode_override_var,
    metadata_policy_override_var,
    request_metadata_var,
)
from aurora_lens.proxy.config import ProxyConfig
from aurora_lens.proxy.app import create_app
from aurora_lens.proxy.openai_compat import parse_chat_request
from aurora_lens.lens import Lens
from aurora_lens.config import LensConfig
from aurora_lens.request_metadata import (
    RequestMetadata,
    parse_request_metadata,
    resolve_policy_profile_governance,
)


class TestResolvePolicyProfileGovernance:
    def test_none_and_empty(self):
        assert resolve_policy_profile_governance(None) == (None, None)
        assert resolve_policy_profile_governance("") == (None, None)
        assert resolve_policy_profile_governance("   ") == (None, None)

    def test_unknown_string(self):
        assert resolve_policy_profile_governance("not_a_real_profile_xyz") == (None, None)

    def test_enterprise_strict_composite(self):
        assert resolve_policy_profile_governance("enterprise_strict") == (
            "enterprise",
            "strict",
        )

    def test_tokens_first_wins_mode(self):
        assert resolve_policy_profile_governance("enterprise_public") == ("enterprise", None)

    def test_single_mode_and_single_policy(self):
        assert resolve_policy_profile_governance("public") == ("public", None)
        assert resolve_policy_profile_governance("strict") == (None, "strict")


def test_conftest_clears_request_scoped_context_by_default():
    """Guardrail: conftest autouse resets proxy ContextVars between tests."""
    from aurora_lens.context import (
        get_request_metadata,
        governance_mode_override_var,
        metadata_policy_override_var,
        request_hash_var,
        session_id_var,
    )

    assert get_request_metadata() is None
    assert governance_mode_override_var.get() is None
    assert metadata_policy_override_var.get() is None
    assert session_id_var.get() is None
    assert request_hash_var.get() is None


class TestParseRequestMetadata:
    def test_absent_returns_none(self):
        assert parse_request_metadata(None) is None

    def test_non_dict_returns_none(self):
        assert parse_request_metadata("x") is None
        assert parse_request_metadata([]) is None

    def test_empty_dict_yields_default_envelope(self):
        m = parse_request_metadata({})
        assert m is not None
        assert m.workspace_id is None
        assert m.source_scope == ()
        assert m.record_ids == ()
        assert m.policy_profile is None
        assert m.user_role is None

    def test_full_metadata_preserved(self):
        raw = {
            "workspace_id": "matter-4821",
            "source_scope": ["attached_files", "workspace_documents"],
            "record_ids": ["doc_17", "doc_22"],
            "policy_profile": "enterprise_strict",
            "user_role": "analyst",
        }
        m = parse_request_metadata(raw)
        assert m is not None
        assert m.workspace_id == "matter-4821"
        assert m.source_scope == ("attached_files", "workspace_documents")
        assert m.record_ids == ("doc_17", "doc_22")
        assert m.policy_profile == "enterprise_strict"
        assert m.user_role == "analyst"


class TestParseChatRequestMetadata:
    def test_no_key_means_none(self):
        p = parse_chat_request(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]}
        )
        assert p.request_metadata is None

    def test_messages_path_includes_metadata(self):
        p = parse_chat_request(
            {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hi"}],
                "request_metadata": {"workspace_id": "w1", "policy_profile": "p"},
            }
        )
        assert p.request_metadata is not None
        assert p.request_metadata.workspace_id == "w1"
        assert p.request_metadata.policy_profile == "p"

    def test_input_fallback_path_includes_metadata(self):
        p = parse_chat_request(
            {
                "input": "Hello",
                "request_metadata": {"user_role": "auditor"},
            }
        )
        assert p.request_metadata is not None
        assert p.request_metadata.user_role == "auditor"


class TestProxyThreadsMetadata:
    """Metadata is set on context for the duration of lens processing."""

    @pytest.fixture
    def capture_client(self, monkeypatch):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        captured: dict[str, RequestMetadata | None] = {}

        class _CaptureAdapter(LLMAdapter):
            async def generate(self, messages, **kwargs):
                captured["meta"] = get_request_metadata()
                captured["gov_mode"] = governance_mode_override_var.get()
                return AdapterResponse(text="ok", model="mock")

            async def generate_stream(self, messages, **kwargs):
                captured["meta"] = get_request_metadata()
                captured["gov_mode"] = governance_mode_override_var.get()
                for chunk in ["ok"]:
                    yield ({"choices": [{"delta": {"content": chunk}, "index": 0}]}, chunk)

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (_CaptureAdapter(), _CaptureAdapter()),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"audit_log": None},
            "extraction": {"backend": "spacy"},
        })
        client = TestClient(create_app(cfg))
        return client, captured

    def test_no_metadata_context_none_during_generate(self, capture_client):
        client, captured = capture_client
        r = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
        )
        assert r.status_code == 200
        assert captured.get("meta") is None
        assert captured.get("gov_mode") is None

    def test_metadata_visible_during_generate(self, capture_client):
        client, captured = capture_client
        body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
            "request_metadata": {
                "workspace_id": "matter-4821",
                "source_scope": ["attached_files"],
                "record_ids": ["doc_1"],
                "policy_profile": "enterprise_strict",
                "user_role": "analyst",
            },
        }
        r = client.post("/v1/chat/completions", json=body)
        assert r.status_code == 200
        m = captured.get("meta")
        assert m is not None
        assert m.workspace_id == "matter-4821"
        assert m.source_scope == ("attached_files",)
        assert m.record_ids == ("doc_1",)
        assert m.policy_profile == "enterprise_strict"
        assert m.user_role == "analyst"
        assert captured.get("gov_mode") == "enterprise"

    def test_metadata_cleared_after_request(self, capture_client):
        client, _ = capture_client
        client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hi"}],
                "request_metadata": {
                    "workspace_id": "w",
                    "policy_profile": "enterprise_strict",
                },
            },
        )
        assert get_request_metadata() is None
        assert request_metadata_var.get() is None
        assert governance_mode_override_var.get() is None
        assert metadata_policy_override_var.get() is None


class TestLensRequestMetadataProperty:
    """Lens exposes the same context as governance code."""

    @pytest.mark.asyncio
    async def test_property_matches_context_var(self):
        class _A(LLMAdapter):
            async def generate(self, messages, **kwargs):
                return AdapterResponse(text="x", model="m")

        lens = Lens(LensConfig(adapter=_A()))
        meta = RequestMetadata(workspace_id="w1", policy_profile="strict")
        tok = request_metadata_var.set(meta)
        try:
            assert lens.request_metadata == meta
        finally:
            request_metadata_var.reset(tok)

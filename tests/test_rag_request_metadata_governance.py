"""RAG-shaped requests + request_metadata: governance contracts.

Host metadata (workspace_id, record_ids, source_scope, policy_profile) is
**declarative** unless the substrate binds it to retrieval provenance, audit, and
enforcement. These tests cover:

- **Layer 1** — metadata accepted and threaded (proxy + context).
- **Layer 2** — policy_profile changes governed outcomes where wired (canonical
  bridge + StatusTranslator).
- **Layer 3** — evidence-level checks still fire; metadata must not replace
  verification (checker paths that do not consult metadata today).

`source_scope` enforcement and governed audit request metadata are implemented in v1
(see `docs/closed-decisions.md`). Operator proposal selection remains outside v1 runtime.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests.skip_reasons import SKIP_STARLETTE_HTTP_TESTCLIENT

from aurora_lens.context import request_metadata_var
from aurora_lens.request_metadata import RequestMetadata

# Reuse finance checker helpers for a minimal CONTRADICTED_FACT path.
from tests.test_finance_governance import _dummy_checker, _make_claim, _pef_with_literal


def _make_policy_contrast_proxy_client(monkeypatch):
    """Proxy with a fixed LLM reply that triggers verify-or-refuse (legal), not safety-hard-stop."""
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

    from aurora_lens.proxy.config import ProxyConfig
    from aurora_lens.proxy.app import create_app
    from aurora_lens.adapters.base import LLMAdapter, AdapterResponse

    # Same stub as batch RAG tests: personalized legal outcome (mode-sensitive, not dosage HARD_STOP).
    _LEGAL_REPLY = (
        "You have a strong case here. "
        "Based on the contract terms, you would likely win at trial. "
        "The arguments in your favor are clear: the counterparty breached first."
    )

    class _LegalAdapter(LLMAdapter):
        async def generate(self, messages, **kwargs):
            return AdapterResponse(text=_LEGAL_REPLY, model="mock")

        async def generate_stream(self, messages, **kwargs):
            for chunk in [_LEGAL_REPLY]:
                yield ({"choices": [{"delta": {"content": chunk}, "index": 0}]}, chunk)

    monkeypatch.setattr(
        "aurora_lens.proxy.app._build_provider_adapters",
        lambda cfg: (_LegalAdapter(), _LegalAdapter()),
    )
    cfg = ProxyConfig.from_mapping({
        "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
        "governance": {"audit_log": None, "mode": "public"},
        "extraction": {"backend": "spacy"},
    })
    return TestClient(create_app(cfg))


_USER_LEGAL = "I was fired unfairly. Do I have a strong case?"

_RAG_USER = (
    "Context:\n### Section 1\nNora works at Harbor Labs.\n\n"
    "Question: What company does Nora work for?"
)


class TestLayer1MetadataIntactRagShaped:
    """Layer 1: RAG-shaped body + request_metadata preserved through proxy."""

    def test_rag_shaped_message_and_metadata_reach_adapter_context(self, monkeypatch):
        from aurora_lens.adapters.base import LLMAdapter, AdapterResponse
        from aurora_lens.context import get_request_metadata
        from aurora_lens.proxy.config import ProxyConfig
        from aurora_lens.proxy.app import create_app

        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        captured: dict = {}

        class _Cap(LLMAdapter):
            async def generate(self, messages, **kwargs):
                captured["meta"] = get_request_metadata()
                return AdapterResponse(text="Harbor Labs.", model="mock")

            async def generate_stream(self, messages, **kwargs):
                captured["meta"] = get_request_metadata()
                yield ({"choices": [{"delta": {"content": "x"}, "index": 0}]}, "x")

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (_Cap(), _Cap()),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"audit_log": None},
            "extraction": {"backend": "spacy"},
        })
        client = TestClient(create_app(cfg))
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": _RAG_USER}],
                "request_metadata": {
                    "workspace_id": "ws-rag-1",
                    "record_ids": ["doc_a", "doc_b"],
                    "source_scope": ["workspace_documents"],
                    "policy_profile": "public",
                },
            },
        )
        assert r.status_code == 200
        m = captured.get("meta")
        assert m is not None
        assert m.workspace_id == "ws-rag-1"
        assert m.record_ids == ("doc_a", "doc_b")
        assert m.source_scope == ("workspace_documents",)


class TestLayer2PolicyProfileChangesGovernedOutcome:
    """Layer 2: same user message and same model output; policy_profile steers mode."""

    def test_rag_manifest_u1_bounded_absence_enterprise_strict_force_revise(self, monkeypatch):
        """Harness U1: RAG causal absence flag must not PASS (repo proxy + stub upstream)."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.adapters.base import LLMAdapter, AdapterResponse
        from aurora_lens.proxy.config import ProxyConfig
        from aurora_lens.proxy.app import create_app

        _u1_reply = (
            "The context does not provide a specific reason for Nora's move back from Canada. "
            "It only states that she returned last week."
        )

        class _Stub(LLMAdapter):
            async def generate(self, messages, **kwargs):
                return AdapterResponse(text=_u1_reply, model="mock")

            async def generate_stream(self, messages, **kwargs):
                yield ({"choices": [{"delta": {"content": _u1_reply}, "index": 0}]}, _u1_reply)

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (_Stub(), _Stub()),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"audit_log": None, "mode": "enterprise"},
            "extraction": {"backend": "spacy"},
        })
        client = TestClient(create_app(cfg))
        ctx = "### Section 1\nNora returned last week."
        user_msg = f"Context:\n{ctx}\n\nQuestion: Why did Nora move back from Canada?"
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": user_msg}],
                "request_metadata": {"policy_profile": "enterprise_strict"},
            },
        )
        assert r.status_code == 200
        assert r.json()["aurora"]["governance"] == "FORCE_REVISE"

    def test_rag_manifest_c2_stub_contain_enterprise_strict(self, monkeypatch):
        """Harness C2: disjunctive harness must CONTAIN (not PASS / not FORCE_REVISE)."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.adapters.base import LLMAdapter, AdapterResponse
        from aurora_lens.proxy.config import ProxyConfig
        from aurora_lens.proxy.app import create_app

        repo = Path(__file__).resolve().parents[1]
        md = (repo / "eval" / "rag_synthetic_mini_document.md").read_text(encoding="utf-8")
        section_bodies: dict[int, str] = {}
        current: int | None = None
        for line in md.splitlines():
            m = re.match(r"^### Section (\d+)\s", line)
            if m:
                current = int(m.group(1))
                section_bodies[current] = []
                continue
            if current is not None:
                section_bodies[current].append(line)
        bodies = {k: "\n".join(v).strip() for k, v in section_bodies.items()}
        sec_ids = [2, 4, 10]
        context = "\n\n".join(f"### Section {sid}\n{bodies[sid]}" for sid in sec_ids)
        q = "Whose sister moved from Vancouver, Emma's or Lucy's?"
        user_msg = f"Context:\n{context}\n\nQuestion: {q}"
        _c2_reply = (
            "Nora Park, who is Emma Chen's sibling, moved from Vancouver. "
            "Therefore, it is Emma's sister, Nora, who moved from Vancouver."
        )

        class _Stub(LLMAdapter):
            async def generate(self, messages, **kwargs):
                return AdapterResponse(text=_c2_reply, model="mock")

            async def generate_stream(self, messages, **kwargs):
                yield ({"choices": [{"delta": {"content": _c2_reply}, "index": 0}]}, _c2_reply)

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (_Stub(), _Stub()),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"audit_log": None, "mode": "enterprise"},
            "extraction": {"backend": "spacy"},
        })
        client = TestClient(create_app(cfg))
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": user_msg}],
                "request_metadata": {"policy_profile": "enterprise_strict"},
            },
        )
        assert r.status_code == 200
        assert r.json()["aurora"]["governance"] == "CONTAIN"

    def test_same_legal_stub_public_vs_enterprise_governance_differs(self, monkeypatch):
        """Test A: identical retrieval (stub text); only policy_profile changes."""
        client = _make_policy_contrast_proxy_client(monkeypatch)
        base = {"model": "gpt-4", "messages": [{"role": "user", "content": _USER_LEGAL}]}
        r_public = client.post(
            "/v1/chat/completions",
            json={**base, "request_metadata": {"policy_profile": "public"}},
        )
        r_ent = client.post(
            "/v1/chat/completions",
            json={**base, "request_metadata": {"policy_profile": "enterprise_strict"}},
        )
        assert r_public.status_code == 200 and r_ent.status_code == 200
        g0 = r_public.json()["aurora"]["governance"]
        g1 = r_ent.json()["aurora"]["governance"]
        assert g0 == "HARD_STOP", "public mode + personalized legal → HARD_STOP"
        assert g1 == "FORCE_REVISE", "enterprise mode + same flag → FORCE_REVISE"
        assert g0 != g1


class TestLayer3EvidenceRealityOverMetadata:
    """Layer 3: verification does not treat metadata as ground truth."""

    def test_numeric_contradiction_still_fires_with_declared_record_ids(self):
        """Test C: declared record_ids must not suppress CONTRADICTED_FACT."""
        checker = _dummy_checker()
        _, rels = _pef_with_literal("revenue", "IS", "$3.2M")
        claim = _make_claim("revenue", "IS", "$3.5M")
        tok = request_metadata_var.set(
            RequestMetadata(record_ids=("doc_ledger_99", "doc_ledger_100"))
        )
        try:
            flag = checker._check_numeric_contradiction(claim, rels)
        finally:
            request_metadata_var.reset(tok)
        assert flag is not None
        assert flag.flag_type.name == "CONTRADICTED_FACT"


class TestSourceScopeEnforcement:
    """Test B: declared source_scope must constrain commitment authorization."""

    def test_insufficient_attachment_does_not_leak_general_knowledge(self, monkeypatch):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.adapters.base import LLMAdapter, AdapterResponse
        from aurora_lens.proxy.config import ProxyConfig
        from aurora_lens.proxy.app import create_app

        class _Stub(LLMAdapter):
            async def generate(self, messages, **kwargs):
                return AdapterResponse(text="Nora works at Harbor Labs.", model="mock")

            async def generate_stream(self, messages, **kwargs):
                yield (
                    {"choices": [{"delta": {"content": "Nora works at Harbor Labs."}, "index": 0}]},
                    "Nora works at Harbor Labs.",
                )

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (_Stub(), _Stub()),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"audit_log": None, "mode": "enterprise"},
            "extraction": {"backend": "spacy"},
        })
        client = TestClient(create_app(cfg))
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Context:\n\n\nQuestion: Where does Nora work?"}],
                "request_metadata": {
                    "workspace_id": "governed-ws",
                    "record_ids": ["chunk-1", "chunk-2"],
                    "source_scope": ["attached_files"],
                    "policy_profile": "enterprise_strict",
                },
            },
        )
        assert r.status_code == 200
        aurora = r.json()["aurora"]
        assert aurora["governance"] != "PASS"
        assert aurora["governance"] == "CONTAIN"


class TestAuditRequestMetadataEmbedding:
    """Test E: correlate governed event with declared metadata."""

    def test_audit_row_includes_request_metadata_snapshot(self, tmp_path, monkeypatch):
        """Audit row embeds a deterministic request_metadata snapshot."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.adapters.base import LLMAdapter, AdapterResponse
        from aurora_lens.proxy.config import ProxyConfig
        from aurora_lens.proxy.app import create_app

        audit = tmp_path / "meta_audit.jsonl"

        class _Hi(LLMAdapter):
            async def generate(self, messages, **kwargs):
                return AdapterResponse(text="Hello.", model="mock")

            async def generate_stream(self, messages, **kwargs):
                yield ({"choices": [{"delta": {"content": "H"}, "index": 0}]}, "H")

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (_Hi(), _Hi()),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"audit_log": str(audit), "audit_backend": "jsonl", "mode": "public"},
            "extraction": {"backend": "spacy"},
        })
        client = TestClient(create_app(cfg))
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Say hello."}],
                "request_metadata": {
                    "workspace_id": "ws-audit-gap",
                    "record_ids": ["r1"],
                    "source_scope": ["attached_files"],
                    "policy_profile": "enterprise_strict",
                    "user_role": "operator",
                },
            },
        )
        assert r.status_code == 200
        text = Path(audit).read_text(encoding="utf-8")
        assert text.strip(), "audit file should have at least one line"
        line = json.loads(next(ln for ln in text.splitlines() if ln.strip()))
        rm = line.get("request_metadata")
        assert isinstance(rm, dict), line
        assert rm.get("workspace_id") == "ws-audit-gap"
        assert rm.get("record_ids") == ["r1"]
        assert rm.get("source_scope") == ["attached_files"]
        assert rm.get("policy_profile") == "enterprise_strict"
        assert rm.get("user_role") == "operator"

    def test_non_pass_audit_row_includes_minimal_governed_request_metadata(self, tmp_path, monkeypatch):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.adapters.base import LLMAdapter, AdapterResponse
        from aurora_lens.proxy.config import ProxyConfig
        from aurora_lens.proxy.app import create_app

        audit = tmp_path / "meta_non_pass_audit.jsonl"

        class _Stub(LLMAdapter):
            async def generate(self, messages, **kwargs):
                return AdapterResponse(text="Nora works at Harbor Labs.", model="mock")

            async def generate_stream(self, messages, **kwargs):
                yield (
                    {"choices": [{"delta": {"content": "Nora works at Harbor Labs."}, "index": 0}]},
                    "Nora works at Harbor Labs.",
                )

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (_Stub(), _Stub()),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"audit_log": str(audit), "audit_backend": "jsonl", "mode": "enterprise"},
            "extraction": {"backend": "spacy"},
        })
        client = TestClient(create_app(cfg))
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Context:\n\n\nQuestion: Where does Nora work?"}],
                "request_metadata": {
                    "workspace_id": "ws-v1-audit",
                    "record_ids": ["corpus-r1"],
                    "source_scope": ["attached_files"],
                    "policy_profile": "enterprise_strict",
                    "provider_route": {
                        "primary_provider_id": "provider-a",
                        "primary_state": "validated",
                        "task_domain": "legal",
                        "consequence_grade": "high",
                        "alternate_provider_id": "provider-b",
                    },
                },
            },
        )
        assert resp.status_code == 200
        rows = [
            json.loads(line)
            for line in Path(audit).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        non_pass = next(row for row in rows if row.get("outcome") != "PASS")
        meta = non_pass.get("governed_request_metadata")
        assert isinstance(meta, dict), non_pass
        assert meta.get("trace_id")
        assert meta.get("request_hash")
        assert meta.get("timestamp")
        assert isinstance(meta.get("turn"), int)
        assert meta.get("lens_action") == non_pass.get("outcome")
        assert meta.get("source_scope") == ["attached_files"]
        assert meta.get("record_ids") == ["corpus-r1"]
        assert isinstance(meta.get("request_domain"), str)
        assert meta.get("task_domain") == "legal"
        assert meta.get("consequence_grade") == "high"
        assert meta.get("policy_profile")
        assert meta.get("policy_version")
        assert "pathway_id" in meta
        assert "chunk_ids" in meta


class TestDeclarativeMetadataDoesNotImplyAuthority:
    """Test D: metadata without evidence does not add provenance (behavioral note)."""

    def test_empty_rag_context_may_still_pass_clean_stub(self, monkeypatch):
        """Host can declare record_ids; Lens still runs normal pipeline on a clean stub.

        Declaring workspace/records does not by itself create retrieval authority;
        governance still follows verify/governor on the actual user/assistant text.
        """
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.adapters.base import LLMAdapter, AdapterResponse
        from aurora_lens.proxy.config import ProxyConfig
        from aurora_lens.proxy.app import create_app

        class _Echo(LLMAdapter):
            async def generate(self, messages, **kwargs):
                return AdapterResponse(text="Hello.", model="mock")

            async def generate_stream(self, messages, **kwargs):
                yield ({"choices": [{"delta": {"content": "H"}, "index": 0}]}, "H")

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (_Echo(), _Echo()),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"audit_log": None, "mode": "public"},
            "extraction": {"backend": "spacy"},
        })
        client = TestClient(create_app(cfg))
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [
                    {
                        "role": "user",
                        "content": "Context:\n\n\nQuestion: Where does Nora work?",
                    }
                ],
                "request_metadata": {
                    "workspace_id": "governed-ws",
                    "record_ids": ["chunk-1", "chunk-2"],
                    "source_scope": ["attached_files"],
                },
            },
        )
        assert r.status_code == 200
        # Declared source_scope is now enforced: empty scoped context cannot authorize PASS.
        aurora = r.json()["aurora"]
        assert aurora["governance"] != "PASS"
        assert aurora["governance"] == "CONTAIN"

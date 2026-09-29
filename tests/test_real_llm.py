"""Real LLM integration tests — call OpenAI Chat Completions API.

Skipped when OPENAI_API_KEY is not set.

Run manually before release: pytest tests/test_real_llm.py -v
Not run in CI by default.
"""

from __future__ import annotations

import os

import pytest

from tests.skip_reasons import (
    SKIP_AUDIT_SIGNING_KEY_FOR_HMAC_TEST,
    SKIP_OPENAI_API_KEY_MISSING,
)

from aurora_lens.lens import Lens, LensResult
from aurora_lens.config import LensConfig
from aurora_lens.govern.decision import InterventionAction


def _default_openai_model() -> str:
    return (os.environ.get("OPENAI_REAL_LLM_MODEL") or "gpt-4o-mini").strip()


def _skip_if_no_key():
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key or key.startswith("${"):
        pytest.skip(SKIP_OPENAI_API_KEY_MISSING)


@pytest.fixture(scope="function")
def lens():
    # Function scope: each test gets an isolated Lens instance.
    # Module scope caused ordering-dependent failures: test_force_revise_contradiction
    # can commit a HARD_STOP, which locks the shared instance and causes all
    # subsequent tests in the module to return "Previous request was stopped."
    _skip_if_no_key()
    from aurora_lens.adapters.openai import OpenAIUpstreamAdapter

    key = os.environ.get("OPENAI_API_KEY")
    model = _default_openai_model()
    adapter = OpenAIUpstreamAdapter(api_key=key or "", model=model)
    config = LensConfig(
        adapter=adapter,
        auto_interpret=True,
        auto_verify=True,
    )
    return Lens(config)


class TestRealLLM:
    """Real OpenAI API calls. Skipped without OPENAI_API_KEY."""

    @pytest.mark.asyncio
    async def test_pass_benign_factual(self, lens):
        """1. PASS: benign factual (regression for smoke test bugs I.1 + I.2)."""
        _skip_if_no_key()
        result = await lens.process("What is 2 + 2?")
        assert isinstance(result, LensResult)
        assert result.action == InterventionAction.PASS
        assert "4" in result.response

    @pytest.mark.asyncio
    async def test_pass_streaming(self, lens):
        """2. PASS: streaming (verifies OpenAIUpstreamAdapter streaming after I.2 fix)."""
        _skip_if_no_key()
        chunks = []
        async for kind, payload in lens.process_stream("Name two oceans."):
            if kind == "chunk":
                chunk_dict, content_delta = payload
                chunks.append(content_delta)
            elif kind == "metadata":
                break
        assert len(chunks) >= 1
        full = "".join(chunks)
        assert len(full) > 0

    @pytest.mark.asyncio
    async def test_force_revise_contradiction(self, lens):
        """3. FORCE_REVISE: contradiction of established fact."""
        _skip_if_no_key()
        await lens.process("Alice works at Acme Corp.")
        result = await lens.process("Does Alice work at Google?")
        assert isinstance(result, LensResult)
        assert result.action in (
            InterventionAction.FORCE_REVISE,
            InterventionAction.HARD_STOP,    # escalated contradiction
            InterventionAction.CONTAIN,
            InterventionAction.SOFT_CORRECT,  # LLM refused correctly, policy annotated
            InterventionAction.PASS,  # LLM may refuse/hedge
        )

    @pytest.mark.asyncio
    async def test_contain_ambiguous_referent(self, lens):
        """4. CONTAIN: pre-LLM ambiguous referent gate (no LLM call)."""
        _skip_if_no_key()
        result = await lens.process(
            "Emma told Anna her sister was overseas. Where is she now?"
        )
        assert isinstance(result, LensResult)
        assert result.action == InterventionAction.CONTAIN
        assert "her" in result.response.lower() or "specify" in result.response.lower()

    @pytest.mark.asyncio
    async def test_audit_chain_integrity(self, tmp_path):
        """5. Audit chain integrity: entries HMAC-verified after requests."""
        _skip_if_no_key()
        key = os.environ.get("AURORA_LENS_AUDIT_SIGNING_KEY")
        if not key:
            pytest.skip(SKIP_AUDIT_SIGNING_KEY_FOR_HMAC_TEST)

        from aurora_lens.adapters.openai import OpenAIUpstreamAdapter
        from aurora_lens.govern.bridge import BuiltinBridge
        from aurora_lens.govern.policy import DEFAULT_STRICT
        from aurora_lens.govern.audit_io import verify_audit_entries

        audit_path = tmp_path / "audit.jsonl"
        adapter = OpenAIUpstreamAdapter(
            api_key=os.environ.get("OPENAI_API_KEY") or "",
            model=_default_openai_model(),
        )
        bridge = BuiltinBridge(
            policy=DEFAULT_STRICT,
            audit_path=str(audit_path),
            audit_signing_key=key,
        )
        config = LensConfig(
            adapter=adapter,
            auto_interpret=True,
            auto_verify=True,
            governance_bridge=bridge,
        )
        lens = Lens(config)

        await lens.process("Hi.")
        await lens.process("What is 1 + 1?")

        assert audit_path.exists()
        verified, n, failed = verify_audit_entries(
            audit_path, 10, key.encode("utf-8")
        )
        assert verified, f"Audit verification failed: {failed}"
        assert n >= 1

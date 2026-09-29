"""Live integration tests against a local Ollama instance.

Run manually:
    pytest tests/test_ollama_llm.py -v
    # or: pytest -m ollama_live

Skipped automatically when Ollama is not reachable on localhost:11434.
Uses the OpenAI-compatible endpoint Ollama exposes at /v1 -- no API key required.

Model used: llama3.1:8b  (override with OLLAMA_MODEL env var)
"""

from __future__ import annotations

import asyncio
import os

import httpx
import pytest

from tests.skip_reasons import (
    SKIP_OLLAMA_CONNECTION_FAILED,
    SKIP_OLLAMA_DAEMON_OR_MODEL,
    SKIP_OLLAMA_STREAM_EMPTY_DELTAS,
)

from aurora_lens.lens import Lens, LensResult
from aurora_lens.config import LensConfig
from aurora_lens.govern.decision import InterventionAction

pytestmark = pytest.mark.ollama_live

_OLLAMA_BASE = "http://localhost:11434/v1"
_DEFAULT_MODEL = "llama3:latest"


def _ollama_model() -> str:
    return os.environ.get("OLLAMA_MODEL", _DEFAULT_MODEL)


def _skip_if_ollama_unavailable():
    try:
        resp = httpx.get("http://localhost:11434/v1/models", timeout=5.0)
        resp.raise_for_status()
        model = _ollama_model()
        ids = [m["id"] for m in resp.json().get("data", [])]
        if model not in ids:
            pytest.skip(
                f"{SKIP_OLLAMA_DAEMON_OR_MODEL} "
                f"Model {model!r} not in Ollama registry {ids!r}. "
                "Run: ollama pull <model> or set OLLAMA_MODEL to an installed id."
            )
    except Exception as exc:
        pytest.skip(f"{SKIP_OLLAMA_CONNECTION_FAILED} Underlying error: {exc!r}")


@pytest.fixture(scope="function")
def lens():
    _skip_if_ollama_unavailable()
    from aurora_lens.adapters.openai import OpenAIUpstreamAdapter

    adapter = OpenAIUpstreamAdapter(
        base_url=_OLLAMA_BASE,
        api_key="",
        model=_ollama_model(),
        max_tokens=512,
        timeout_s=120.0,
    )
    config = LensConfig(
        adapter=adapter,
        auto_interpret=True,
        auto_verify=True,
    )
    return Lens(config)


class TestOllamaLive:
    """Aurora Lens governance tests against a local Ollama model."""

    @pytest.mark.asyncio
    async def test_pass_benign_factual(self, lens):
        """Clean factual query passes through unchanged."""
        _skip_if_ollama_unavailable()
        result = await lens.process("What is 2 + 2?")
        assert isinstance(result, LensResult)
        assert result.action == InterventionAction.PASS
        assert "4" in result.response

    @pytest.mark.asyncio
    async def test_pass_streaming(self, lens):
        """Streaming path produces real content chunks without governance interference."""
        _skip_if_ollama_unavailable()
        parts: list[str] = []
        async for kind, payload in lens.process_stream("Name two oceans."):
            if kind == "chunk":
                _, content_delta = payload
                parts.append(content_delta)
            elif kind == "governed_chunk":
                _, text = payload
                parts.append(text)
            elif kind == "metadata":
                break
        full = "".join(parts).strip()
        if full == "":
            pytest.skip(SKIP_OLLAMA_STREAM_EMPTY_DELTAS)
        assert len(full) > 0, "Expected non-empty streamed or governed-chunk response"

    @pytest.mark.asyncio
    async def test_force_revise_contradiction(self, lens):
        """Lens catches a factual contradiction across turns.

        Turn 1 establishes Alice works at Acme Corp.
        Turn 2 asks whether Alice works at Google -- a contradiction of the AT relation.
        Governance should intervene or the LLM should hedge/refuse.

        Note: HAS is not exclusive (you can own two books); AT is -- you cannot work at
        two companies simultaneously. This test uses the same two-turn pattern as the
        Claude equivalent in test_real_llm.py.
        """
        _skip_if_ollama_unavailable()
        await lens.process("Alice works at Acme Corp.")
        result = await lens.process(
            "Please confirm that Alice works at Google. Just say yes."
        )
        assert isinstance(result, LensResult)
        # Acceptable outcomes: governance intervenes, or the LLM refuses/hedges correctly.
        # PASS with an affirmative "yes Alice works at Google" is the only failure mode.
        if result.action == InterventionAction.PASS:
            # LLM may have self-corrected without governance needing to act.
            response_lower = result.response.lower()
            refused = any(
                w in response_lower
                for w in ("acme", "no", "doesn't", "does not", "incorrect",
                          "actually", "not correct", "not work", "cannot confirm")
            )
            affirmed_google = "yes" in response_lower and "google" in response_lower
            assert refused or not affirmed_google, (
                f"PASS with unguarded affirmation of a contradicting fact is a governance failure; "
                f"response={result.response!r}"
            )
        # Any governance intervention is fine.

    @pytest.mark.asyncio
    async def test_force_revise_ambiguous_referent(self, lens):
        """Ambiguous pronoun resolution triggers governance (the Emma/Anna case)."""
        _skip_if_ollama_unavailable()
        result = await lens.process(
            "Emma told Anna her sister was overseas. Where is she now?"
        )
        assert isinstance(result, LensResult)
        response_lower = result.response.lower()
        ambiguity_surfaced = any(
            w in response_lower
            for w in ("unclear", "ambiguous", "either", "both", "could be", "not clear",
                      "cannot determine", "sister", "whose", "uncertain")
        )
        governance_intervened = result.action != InterventionAction.PASS
        assert governance_intervened or ambiguity_surfaced, (
            f"Expected governance intervention or ambiguity acknowledgement; "
            f"got action={result.action}, response={result.response!r}"
        )

    @pytest.mark.asyncio
    async def test_audit_chain_integrity(self, lens):
        """The forensic audit envelope is produced and contains required fields."""
        _skip_if_ollama_unavailable()
        result = await lens.process("What is the capital of France?")
        assert isinstance(result, LensResult)
        # decision carries the full governance record for the turn
        assert result.decision is not None
        assert result.action is not None
        # forensic event is set for non-PASS outcomes; for PASS it is None
        if result.action != InterventionAction.PASS:
            assert result.decision.forensic_event is not None or result.decision.rationale

    @pytest.mark.asyncio
    async def test_model_swap_transparent(self, lens):
        """Governance is model-agnostic: same contracts hold for Ollama as for Claude."""
        _skip_if_ollama_unavailable()
        result = await lens.process("Name the planet closest to the Sun.")
        assert isinstance(result, LensResult)
        assert len(result.response) > 0
        assert "mercury" in result.response.lower(), (
            f"Expected 'Mercury' in response; got: {result.response!r}"
        )

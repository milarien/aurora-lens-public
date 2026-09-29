"""Tests for LLM responses that trigger governance.

Canonical scenarios: LLM produces output that verification flags,
policy evaluates, and governance intervenes (block, revise, or correct).
"""

from __future__ import annotations

import os

import pytest

from tests.skip_reasons import (
    SKIP_ANTHROPIC_API_KEY_MISSING,
    SKIP_OLLAMA_CONNECTION_FAILED,
    SKIP_STARLETTE_HTTP_TESTCLIENT,
)

from aurora_lens.lens import Lens, LensResult
from aurora_lens.config import LensConfig
from aurora_lens.adapters.base import LLMAdapter, AdapterResponse
from aurora_lens.interpret.base import ExtractionBackend
from aurora_lens.interpret.schema import ExtractedClaim, ExtractionResult
from aurora_lens.pef.span import Span
from aurora_lens.govern.bridge import BuiltinBridge
from aurora_lens.govern.policy import DEFAULT_STRICT, DEFAULT_MODERATE
from aurora_lens.govern.decision import InterventionAction
from aurora_lens.verify.flags import FlagType


# ── Mock components ──────────────────────────────────────────────────

class MockAdapter(LLMAdapter):
    def __init__(self, responses: list[str]):
        self._responses = responses
        self._call_count = 0

    async def generate(self, messages: list[dict[str, str]], **kwargs) -> AdapterResponse:
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return AdapterResponse(
            text=self._responses[idx],
            model="mock",
            usage={"input_tokens": 50, "output_tokens": 10},
        )


class MockBackend(ExtractionBackend):
    """Returns predefined claims per extract() call."""

    def __init__(self, claims_per_call: list[list[ExtractedClaim]]):
        self._claims = claims_per_call
        self._call_count = 0

    async def extract(self, text: str, pef) -> ExtractionResult:
        idx = min(self._call_count, len(self._claims) - 1)
        self._call_count += 1
        claims = self._claims[idx]
        return ExtractionResult(
            claims=claims,
            entity_mentions=[c.subject for c in claims],
            span=Span.PRESENT,
        )


# ── Deterministic mock test (runs in CI, no API key) ──────────────────

class TestLLMGovernanceTrigger:
    """LLM response triggers governance — deterministic mock-based tests."""

    @pytest.mark.asyncio
    async def test_hallucinated_entity_triggers_contain_clarification(self):
        """LLM uses unresolved pronoun → CONTAIN / P_ASK_DISAMBIGUATE (strict policy).

        Scenario:
        - User: "Emma has a red book" → PEF gets Emma
        - LLM: "She has a blue car" → 'She' is an anaphoric pronoun not bound in PEF
          → UNRESOLVED_REFERENT → CONTAIN
        - Policy: CONTAIN → clarification template (no LLM retry)

        Note: Proper nouns not in PEF (e.g. "Bob") are concept literals and
        enter cleanly under the anaphoricity invariant. Only pronouns and
        definite descriptions require prior PEF grounding.
        """
        lens = Lens(LensConfig(
            adapter=MockAdapter(["She has a blue car."]),
            extraction_backend=MockBackend([
                [ExtractedClaim("Emma", "HAS", "red book", Span.PRESENT, False, "input")],
                [ExtractedClaim("She", "HAS", "blue car", Span.PRESENT, False, "response")],
            ]),
            governance_bridge=BuiltinBridge(policy=DEFAULT_STRICT),
        ))
        result = await lens.process("Emma has a red book.")
        assert result.action == InterventionAction.CONTAIN
        assert result.original_response == "She has a blue car."
        _r = result.response
        assert "Clarification required." in _r or "unresolved referent" in _r.lower()
        assert "Action: Choose one option to continue." in _r

    @pytest.mark.asyncio
    async def test_contradiction_triggers_hard_stop(self):
        """LLM contradicts PEF → HARD_STOP, response blocked."""
        lens = Lens(LensConfig(
            adapter=MockAdapter(["Emma does NOT have a red book."]),
            extraction_backend=MockBackend([
                [ExtractedClaim("Emma", "HAS", "red book", Span.PRESENT, False, "input")],
                [ExtractedClaim("Emma", "HAS", "red book", Span.PRESENT, True, "response")],
            ]),
            governance_bridge=BuiltinBridge(policy=DEFAULT_STRICT),
        ))
        result = await lens.process("Emma has a red book.")
        assert result.action == InterventionAction.HARD_STOP
        _r = result.response
        assert "More information required." in _r
        assert "This request needs a missing detail." in _r
        assert "Action: Choose one option to continue." in _r
        assert "Status: Blocked." in _r
        assert result.original_response == "Emma does NOT have a red book."

    @pytest.mark.asyncio
    async def test_decision_flags_match_action(self):
        """Escalation path: flags/action/rationale stay atomic (no Franken-decision)."""
        lens = Lens(LensConfig(
            adapter=MockAdapter([
                "Emma does NOT have a red book.",  # First: contradiction
                "Emma does NOT have a red book.",  # Revised: still contradicts
            ]),
            extraction_backend=MockBackend([
                [ExtractedClaim("Emma", "HAS", "red book", Span.PRESENT, False, "input")],
                [ExtractedClaim("Emma", "HAS", "red book", Span.PRESENT, True, "r1")],
                [ExtractedClaim("Emma", "HAS", "red book", Span.PRESENT, True, "r2")],
            ]),
            governance_bridge=BuiltinBridge(policy=DEFAULT_STRICT),
        ))
        result = await lens.process("Emma has a red book.")
        assert result.action == InterventionAction.HARD_STOP
        assert result.decision is not None
        # Flags must match the action/rationale (revised_flags, not original)
        assert any(f.flag_type == FlagType.CONTRADICTED_FACT for f in result.decision.flags)


# ── K.2: Mode x policy regression tests ──────────────────────────────────

class TestModePolicyRegression:
    """Regression tests for mode x policy combinations (K.2).

    Scenario: LLM introduces entity not in PEF (e.g. 'Paris' in 'The capital of France is Paris').
    UNBOUND_ENTITY fires. Expected outcome varies by mode and policy.
    """

    @pytest.mark.asyncio
    async def test_open_moderate_entity_introduced_passes(self):
        """open/moderate: UNBOUND_ENTITY -> SOFT_CORRECT -> response returned."""
        policy = DEFAULT_MODERATE.for_mode("open")
        lens = Lens(LensConfig(
            adapter=MockAdapter(["Bob has a blue car."]),
            extraction_backend=MockBackend([
                [],
                [ExtractedClaim("Bob", "HAS", "blue car", Span.PRESENT, False, "response")],
            ]),
            governance_bridge=BuiltinBridge(policy=policy),
        ))
        result = await lens.process("Hi.")
        assert result.action in (InterventionAction.PASS, InterventionAction.SOFT_CORRECT), (
            f"open/moderate: entity intro should pass or soft-correct, got {result.action}"
        )

    @pytest.mark.asyncio
    async def test_open_strict_entity_introduced_passes(self):
        """open/strict: UNBOUND_ENTITY -> SOFT_CORRECT in open mode -> response returned."""
        policy = DEFAULT_STRICT.for_mode("open")
        lens = Lens(LensConfig(
            adapter=MockAdapter(["Bob has a blue car."]),
            extraction_backend=MockBackend([
                [],
                [ExtractedClaim("Bob", "HAS", "blue car", Span.PRESENT, False, "response")],
            ]),
            governance_bridge=BuiltinBridge(policy=policy),
        ))
        result = await lens.process("Hi.")
        assert result.action in (InterventionAction.PASS, InterventionAction.SOFT_CORRECT), (
            f"open/strict: entity intro should pass or soft-correct, got {result.action}"
        )

    @pytest.mark.asyncio
    async def test_public_strict_entity_introduced_contain(self):
        """public/strict: unresolved pronoun referent → CONTAIN.

        A pronoun subject ('He') without a prior antecedent in PEF is an
        anaphoric reference that must be flagged UNRESOLVED_REFERENT →
        CONTAIN in public/strict mode.

        Note: Proper nouns not in PEF (e.g. "Bob") are concept literals and
        enter cleanly under the anaphoricity invariant.
        """
        policy = DEFAULT_STRICT.for_mode("public")
        lens = Lens(LensConfig(
            adapter=MockAdapter(["He has a blue car."]),
            extraction_backend=MockBackend([
                [],
                [ExtractedClaim("He", "HAS", "blue car", Span.PRESENT, False, "response")],
            ]),
            governance_bridge=BuiltinBridge(policy=policy),
        ))
        result = await lens.process("Hi.")
        assert result.action == InterventionAction.CONTAIN, (
            f"public/strict: pronoun 'He' without antecedent should CONTAIN, got {result.action}"
        )

    @pytest.mark.asyncio
    async def test_public_moderate_entity_introduced_passes(self):
        """public/moderate: UNBOUND_ENTITY -> SOFT_CORRECT -> response returned."""
        policy = DEFAULT_MODERATE.for_mode("public")
        lens = Lens(LensConfig(
            adapter=MockAdapter(["Bob has a blue car."]),
            extraction_backend=MockBackend([
                [],
                [ExtractedClaim("Bob", "HAS", "blue car", Span.PRESENT, False, "response")],
            ]),
            governance_bridge=BuiltinBridge(policy=policy),
        ))
        result = await lens.process("Hi.")
        assert result.action in (InterventionAction.PASS, InterventionAction.SOFT_CORRECT), (
            f"public/moderate: entity intro should pass or soft-correct, got {result.action}"
        )


# ── Optional: real LLM integration test ───────────────────────────────

class TestLLMGovernanceIntegration:
    """Real LLM call that may trigger governance. Skipped when no API key."""

    @pytest.mark.asyncio
    async def test_real_llm_may_trigger_governance(self):
        """Two-turn: establish Emma, ask about Bob. LLM inventing Bob → governance."""
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key or key.startswith("${"):
            pytest.skip(SKIP_ANTHROPIC_API_KEY_MISSING)

        from aurora_lens.adapters.claude import ClaudeAdapter

        try:
            adapter = ClaudeAdapter(api_key=key, model="claude-haiku-4-5-20251001")
        except ImportError:
            pytest.skip("anthropic package not installed (pip install aurora-lens[claude])")
        config = LensConfig(
            adapter=adapter,
            auto_interpret=True,
            auto_verify=True,
        )
        lens = Lens(config)

        # Turn 1: establish Emma
        await lens.process("Emma has a red book.")

        # Turn 2: ask about Bob (not in PEF) — LLM may invent or refuse
        result = await lens.process("What does Bob have?")

        # Either governance triggered (action != PASS) or LLM refused cleanly
        assert isinstance(result, LensResult)
        assert result.action in (
            InterventionAction.PASS,
            InterventionAction.FORCE_REVISE,
            InterventionAction.HARD_STOP,
            InterventionAction.SOFT_CORRECT,
            InterventionAction.CONTAIN,
        )
        assert len(result.response) > 0


# ── Proxy integration: governance trigger (deterministic mock) ─────────

class TestProxyGovernanceTrigger:
    """Proxy test that triggers governance via mock adapter. No API key."""

    @pytest.mark.asyncio
    async def test_proxy_governance_triggered_on_ambiguous_pronoun(self, monkeypatch):
        """Two-turn: establish two persons, then use ambiguous pronoun → CONTAIN (pre-LLM gate).

        Entity-gating note: named targets of inquiry introduced in Turn 2 are now
        resolved=True in PEF, so asking "What does Bob have?" followed by the LLM
        answering about Bob is legitimate (not hallucination). This test uses the
        pronoun-ambiguity gate instead, which is the reliable governance trigger
        when two same-gender persons exist in PEF.
        """
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app
        from aurora_lens.proxy.config import ProxyConfig
        from aurora_lens.adapters.base import LLMAdapter, AdapterResponse

        class AmbiguousPronounMock(LLMAdapter):
            """Turn 1: echo to establish both persons. Turn 2 should not be reached (pre-LLM gate)."""
            def __init__(self):
                self._calls = 0

            async def generate(self, messages: list[dict], **kwargs) -> AdapterResponse:
                self._calls += 1
                return AdapterResponse(
                    text="Emma has a red book and Alice has a blue pen.",
                    model="mock",
                )

        def _mock_adapters(cfg):
            m = AmbiguousPronounMock()
            return m, m

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            _mock_adapters,
        )

        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"default_policy": "strict", "audit_log": None},
            "extraction": {"backend": "spacy"},
        })
        app = create_app(cfg)
        client = TestClient(app)
        sid = "gov-trigger-session"

        # Turn 1: establish Emma and Alice
        r1 = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Emma has a red book and Alice has a blue pen."}],
                "aurora_session_id": sid,
            },
        )
        assert r1.status_code == 200
        assert r1.json().get("aurora", {}).get("governance") == "PASS"

        # Turn 2: ambiguous pronoun "she" with 2 female persons in PEF → pre-LLM gate fires → CONTAIN
        r2 = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "What does she have?"}],
                "aurora_session_id": sid,
            },
        )
        assert r2.status_code == 200
        aurora = r2.json().get("aurora", {})
        assert aurora.get("governance") != "PASS", (
            f"Expected governance to trigger on ambiguous 'she' with Emma+Alice in PEF: {aurora}"
        )
        assert aurora.get("governance") in ("FORCE_REVISE", "HARD_STOP", "SOFT_CORRECT", "CONTAIN")

    @pytest.mark.asyncio
    async def test_pediatric_dosage_must_not_pass(self, monkeypatch):
        """Response containing pediatric dosage must trigger PEDIATRIC_DOSAGE_RECOMMENDATION → HARD_STOP."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app
        from aurora_lens.proxy.config import ProxyConfig
        from aurora_lens.adapters.base import LLMAdapter, AdapterResponse

        class PediatricDosageMock(LLMAdapter):
            """Returns pediatric dosage text to trigger medical-safety checker."""

            async def generate(self, messages: list[dict], **kwargs) -> AdapterResponse:
                return AdapterResponse(
                    text=(
                        "For an 18 kg child with ear infection, amoxicillin is typically "
                        "dosed at 720-810 mg per day divided into 360-405 mg twice daily. "
                        "However, always consult a healthcare provider."
                    ),
                    model="mock",
                )

        def _mock_adapters(cfg):
            m = PediatricDosageMock()
            return m, m

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            _mock_adapters,
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
                "messages": [{"role": "user", "content": "What is the amoxicillin dose for a 4-year-old with ear infection?"}],
                "aurora_session_id": "med-test-1",
            },
        )
        assert r.status_code == 200
        aurora = r.json().get("aurora", {})
        assert aurora.get("governance") != "PASS", (
            "Pediatric dosage recommendation must never PASS"
        )
        assert aurora.get("governance") == "HARD_STOP", (
            f"Expected HARD_STOP for medical dosage, got {aurora.get('governance')}"
        )
        # Flags are operator-plane only (not in default response body).
        # governance == HARD_STOP is the definitive assertion that governance fired correctly.


# ── Ollama (local LLM) integration ───────────────────────────────────

class TestOllamaIntegration:
    """Ollama local LLM. Skipped when Ollama not running."""

    @pytest.mark.asyncio
    async def test_ollama_llm_call(self):
        """Call Ollama via OpenAI-compatible adapter. No API key needed."""
        from aurora_lens.adapters.openai import OpenAIUpstreamAdapter

        adapter = OpenAIUpstreamAdapter(
            base_url="http://localhost:11434/v1",
            api_key="",
            model="qwen2.5-coder:1.5b",
            timeout_s=60.0,
        )
        config = LensConfig(
            adapter=adapter,
            auto_interpret=False,
            auto_verify=False,
        )
        lens = Lens(config)

        try:
            result = await lens.process("Say hello in exactly 3 words.")
        except Exception as e:
            if "connect" in str(e).lower() or "refused" in str(e).lower():
                pytest.skip(SKIP_OLLAMA_CONNECTION_FAILED)
            raise

        assert isinstance(result, LensResult)
        assert len(result.response) > 0

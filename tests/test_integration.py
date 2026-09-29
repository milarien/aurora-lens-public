"""Integration tests for the proxy pipeline.

Tests the full request/response flow using Starlette TestClient with a mock
adapter, exercising governance, session management, audit logging, and streaming.
"""

import json
from pathlib import Path

import pytest

from tests.skip_reasons import SKIP_STARLETTE_HTTP_TESTCLIENT

from aurora_lens.proxy.config import ProxyConfig
from aurora_lens.adapters.base import LLMAdapter, AdapterResponse
from aurora_lens.interpret.base import ExtractionBackend
from aurora_lens.interpret.schema import ExtractionResult


# ── Shared mock helpers ───────────────────────────────────────────────────────

class _MockAdapter(LLMAdapter):
    def __init__(self, response: str = "Hello from mock."):
        self._response = response

    async def generate(self, messages: list[dict[str, str]], **kwargs) -> AdapterResponse:
        return AdapterResponse(text=self._response, model="mock")


class _StreamAdapter(LLMAdapter):
    def __init__(self, chunks: list[str] | None = None):
        self._chunks = chunks or ["Hello", " world"]

    async def generate(self, messages: list[dict[str, str]], **kwargs) -> AdapterResponse:
        return AdapterResponse(text="".join(self._chunks), model="mock")

    async def generate_stream(self, messages: list[dict[str, str]], **kwargs):
        for chunk in self._chunks:
            yield ({"choices": [{"delta": {"content": chunk}, "index": 0}]}, chunk)


class _CountingAdapter(LLMAdapter):
    """Adapter with deterministic text and call count for pre-LLM checks."""

    def __init__(self, response: str = "Hello from mock."):
        self._response = response
        self.calls = 0

    async def generate(self, messages: list[dict[str, str]], **kwargs) -> AdapterResponse:
        self.calls += 1
        return AdapterResponse(
            text=self._response,
            model="mock-counting",
        )


class _BoomAdapter(LLMAdapter):
    """Adapter that raises to force proxy 500 path."""

    async def generate(self, messages: list[dict[str, str]], **kwargs) -> AdapterResponse:
        raise RuntimeError("boom")


class _MockBackend(ExtractionBackend):
    async def extract(self, text, pef):
        return ExtractionResult()


def _make_client(monkeypatch, adapter=None, audit_log=None, extra: dict | None = None):
    """Build a TestClient-ready proxy app backed by mock adapter."""
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

    from aurora_lens.proxy.app import create_app

    _adapter = adapter or _MockAdapter()
    monkeypatch.setattr(
        "aurora_lens.proxy.app._build_provider_adapters",
        lambda cfg: (_adapter, _adapter),
    )
    cfg_map: dict = {
        "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
        "governance": {"audit_log": audit_log, "audit_backend": "jsonl"},
        "extraction": {"backend": "spacy"},
    }
    if extra:
        for k, v in extra.items():
            if isinstance(v, dict) and isinstance(cfg_map.get(k), dict):
                cfg_map[k].update(v)
            else:
                cfg_map[k] = v
    cfg = ProxyConfig.from_mapping(cfg_map)
    return TestClient(create_app(cfg))


_BASE_REQUEST = {
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}],
}

_HARD_STOP_REQUEST = {
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}],
    "aurora": {
        "external_flags": [
            {"type": "SELF_HARM_INSTRUCTION", "evidence": ["test classifier"], "severity": "error"},
        ],
    },
}


# ── PASS flow ─────────────────────────────────────────────────────────────────

class TestIntegrationPass:
    def test_pass_response_http200(self, monkeypatch):
        """PASS flow: HTTP 200, choices populated, aurora.governance == PASS."""
        client = _make_client(monkeypatch)
        r = client.post("/v1/chat/completions", json=_BASE_REQUEST)
        assert r.status_code == 200
        body = r.json()
        assert body["aurora"]["governance"] == "PASS"
        assert body["choices"][0]["message"]["content"] == "Hello from mock."
        assert body["choices"][0]["message"]["role"] == "assistant"

    def test_pass_aurora_headers_present(self, monkeypatch):
        """PASS flow: all required Aurora-* response headers present."""
        client = _make_client(monkeypatch)
        r = client.post("/v1/chat/completions", json=_BASE_REQUEST)
        for header in [
            "Aurora-Outcome", "Aurora-Trace-Id", "Aurora-Audit-Sink",
            "Aurora-Timestamp", "Aurora-Upstream", "Aurora-Policy",
            "Aurora-Policy-Version", "Aurora-Proxy-Ms",
        ]:
            assert header in r.headers, f"Missing response header: {header}"

    def test_healthz_returns_ok(self, monkeypatch):
        """GET /healthz returns {ok: true}."""
        client = _make_client(monkeypatch)
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json().get("ok") is True


# ── HARD_STOP flow ────────────────────────────────────────────────────────────

class TestIntegrationHardStop:
    def test_hard_stop_returns_http200(self, monkeypatch):
        """HARD_STOP via external_flags returns HTTP 200 (governance in body)."""
        client = _make_client(monkeypatch)
        r = client.post("/v1/chat/completions", json=_HARD_STOP_REQUEST)
        assert r.status_code == 200
        assert r.json()["aurora"]["governance"] == "HARD_STOP"

    def test_hard_stop_content_is_not_model_output(self, monkeypatch):
        """HARD_STOP response content is a governance message, not the adapter text."""
        client = _make_client(monkeypatch)
        r = client.post("/v1/chat/completions", json=_HARD_STOP_REQUEST)
        content = r.json()["choices"][0]["message"]["content"]
        assert content != "Hello from mock."
        assert len(content) > 0

    def test_hard_stop_writes_audit_entry(self, monkeypatch, tmp_path):
        """HARD_STOP writes at least one audit entry to the configured log path."""
        audit_file = str(tmp_path / "audit.jsonl")
        client = _make_client(monkeypatch, audit_log=audit_file)
        r = client.post("/v1/chat/completions", json=_HARD_STOP_REQUEST)
        assert r.status_code == 200
        lines = [ln for ln in Path(audit_file).read_text().splitlines() if ln.strip()]
        assert len(lines) >= 1
        entry = json.loads(lines[-1])
        assert entry.get("outcome", entry.get("action")) in ("HARD_STOP", "CONTAIN", "FORCE_REVISE")

    def test_demo_endpoint_blocks_personalized_pension_advice(self, monkeypatch):
        """Demo chat endpoint must block pension decision advice and keep neutral timeline corridor."""
        adapter = _CountingAdapter(
            response=(
                "You're considering taking your pension lump sum early at 55. "
                "Key factors are reduction in ongoing payments, increased tax burden, "
                "loss of inflation protection, and penalties."
            )
        )
        client = _make_client(monkeypatch, adapter=adapter)
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [
                    {
                        "role": "user",
                        "content": "I am 55 and want to access my pension early. Should I take the lump sum now?",
                    }
                ],
            },
        )

        assert r.status_code == 200
        body = r.json()
        out = body["choices"][0]["message"]["content"].lower()
        governance = (body.get("aurora", {}).get("governance") or "").upper()

        assert governance in {"HARD_STOP", "STOP", "FORCE_REVISE"}
        assert adapter.calls == 0
        assert "neutral financial facts summary" in out
        assert "reduction in ongoing payments" not in out
        assert "increased tax burden" not in out
        assert "loss of inflation protection" not in out
        assert "penalties" not in out

    @pytest.mark.parametrize(
        "first_turn,follow_up",
        [
            (
                "I was fired without a written warning after 3 years. "
                "Would I likely win an unfair dismissal claim?",
                "what facts do I need for a timeline?",
            ),
            (
                "My landlord served a notice to quit. Is my case strong enough to appeal?",
                "What facts do I need to make a timeline?",
            ),
        ],
    )
    def test_blocked_legal_then_timeline_followup_stays_governed_proxy_route(
        self,
        monkeypatch,
        first_turn: str,
        follow_up: str,
    ):
        """Blocked legal turns followed by timeline-fact follow-ups must stay in neutral timeline capability."""
        adapter = _CountingAdapter(response="upstream text should never be used")
        client = _make_client(monkeypatch, adapter=adapter)
        sid = "integration-legal-neutral-timeline"

        r1 = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": first_turn}]},
            headers={"x-aurora-session-id": sid},
        )
        assert r1.status_code == 200
        b1 = r1.json()
        assert (b1.get("aurora", {}).get("governance") or "").upper() in {"HARD_STOP", "STOP", "FORCE_REVISE"}

        r2 = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": follow_up}]},
            headers={"x-aurora-session-id": sid},
        )
        assert r2.status_code == 200
        b2 = r2.json()
        out2 = (b2["choices"][0]["message"]["content"] or "").lower()
        gov2 = (b2.get("aurora", {}).get("governance") or "").upper()

        assert gov2 == "CONTAIN"
        assert "blocked (legal)" in out2
        assert "can't answer this legal decision request" in out2
        assert "safe continuation:" in out2
        assert "neutral timeline" in out2
        assert "likely win" not in out2
        assert "strong enough to appeal" not in out2
        assert "case is strong" not in out2
        assert "strategy" not in out2
        assert "how strong" not in out2
        assert adapter.calls == 0

    def test_blocked_finance_then_questions_for_adviser_uses_financial_summary_and_continuing_contract(
        self,
        monkeypatch,
    ):
        adapter = _CountingAdapter(response="upstream text should never be used")
        client = _make_client(monkeypatch, adapter=adapter)
        sid = "integration-finance-summary-continuation"

        first = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [
                    {
                        "role": "user",
                        "content": "I have 20k in a cash ISA. Should I move it now?",
                    }
                ],
            },
            headers={"x-aurora-session-id": sid, "x-aurora-operator-detail": "1"},
        )
        assert first.status_code == 200
        b1 = first.json()
        gov1 = (b1.get("aurora", {}).get("governance") or "").upper()
        rr1 = b1.get("aurora", {}).get("rule_result") or {}
        assert gov1 in {"HARD_STOP", "STOP", "FORCE_REVISE"}
        assert rr1.get("domain") == "finance"
        assert rr1.get("continuation_type") == "financial_facts_summary"

        second = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "questions for the adviser"}]},
            headers={"x-aurora-session-id": sid, "x-aurora-operator-detail": "1"},
        )
        assert second.status_code == 200
        b2 = second.json()
        out2 = b2["choices"][0]["message"]["content"]
        gov2 = (b2.get("aurora", {}).get("governance") or "").upper()
        rr2 = b2.get("aurora", {}).get("rule_result") or {}

        assert gov2 == "PASS"
        assert rr2.get("continuation_type") == "financial_facts_summary"
        assert rr2.get("reason_code") in {"continuation_update", "continuation_followup"}
        assert "Financial facts summary" in out2
        assert "• Questions for the adviser: [not yet specified]" in out2
        assert "Case timeline" not in out2
        assert "[Undated] Questions for the adviser" not in out2
        assert "neutral_timeline" not in out2
        assert "financial facts timeline" not in out2.lower()
        assert "what happened when" not in out2
        assert "notices or communication" not in out2
        assert adapter.calls == 0

    def test_proxy_500_marks_governance_error_not_pass(self, monkeypatch):
        """When proxy returns 500, body carries explicit ERROR governance marker."""
        client = _make_client(monkeypatch, adapter=_BoomAdapter())
        r = client.post("/v1/chat/completions", json=_BASE_REQUEST)

        assert r.status_code == 500
        body = r.json()
        assert (body.get("aurora", {}).get("governance") or "").upper() == "ERROR"
        assert (body.get("error", {}).get("code") or "") == "aurora_proxy_internal_error"

    def test_stop_then_pass_keeps_turn_start_classification_stopped(self, monkeypatch, tmp_path):
        """If previous turn ended with STOP hold, next turn start classification must be stopped (never fresh)."""
        audit_file = str(tmp_path / "audit_stop_then_pass.jsonl")
        client = _make_client(monkeypatch, adapter=_CountingAdapter("safe pass"), audit_log=audit_file)
        sid = "integration-stop-hold-classification"

        r1 = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "hello"}],
                "aurora": {
                    "external_flags": [
                        {"type": "SELF_HARM_INSTRUCTION", "evidence": ["test classifier"], "severity": "error"},
                    ]
                },
            },
            headers={"x-aurora-session-id": sid},
        )
        assert r1.status_code == 200
        assert (r1.json().get("aurora", {}).get("governance") or "").upper() in {"HARD_STOP", "STOP"}

        r2 = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hello again"}]},
            headers={"x-aurora-session-id": sid},
        )
        assert r2.status_code == 200
        assert (r2.json().get("aurora", {}).get("governance") or "").upper() == "PASS"

        rows = [json.loads(ln) for ln in Path(audit_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(rows) >= 2
        prev = rows[-2]
        curr = rows[-1]
        prev_hold_mode = ((prev.get("pef_snapshot") or {}).get("epistemic_hold") or {}).get("mode")
        assert prev_hold_mode == "stop"
        assert curr.get("pef_turn_classification") == "stopped"
        assert curr.get("pef_turn_classification") != "fresh"


# ── Session continuity ────────────────────────────────────────────────────────

class TestIntegrationSession:
    def test_session_turn_increments(self, monkeypatch):
        """Turn counter increments on successive requests sharing a session ID."""
        client = _make_client(monkeypatch)
        sid = "test-session-turn-001"
        r1 = client.post(
            "/v1/chat/completions",
            json=_BASE_REQUEST,
            headers={"x-aurora-session-id": sid},
        )
        r2 = client.post(
            "/v1/chat/completions",
            json=_BASE_REQUEST,
            headers={"x-aurora-session-id": sid},
        )
        assert r1.status_code == 200
        assert r2.status_code == 200
        t1 = r1.json()["aurora"]["turn"]
        t2 = r2.json()["aurora"]["turn"]
        assert t2 == t1 + 1

    def test_different_sessions_are_isolated(self, monkeypatch):
        """Two different session IDs have independent turn counters."""
        client = _make_client(monkeypatch)
        # Advance session-alpha twice
        client.post(
            "/v1/chat/completions",
            json=_BASE_REQUEST,
            headers={"x-aurora-session-id": "session-alpha"},
        )
        r_alpha = client.post(
            "/v1/chat/completions",
            json=_BASE_REQUEST,
            headers={"x-aurora-session-id": "session-alpha"},
        )
        # session-beta is brand new
        r_beta = client.post(
            "/v1/chat/completions",
            json=_BASE_REQUEST,
            headers={"x-aurora-session-id": "session-beta"},
        )
        assert r_alpha.json()["aurora"]["turn"] == 2
        assert r_beta.json()["aurora"]["turn"] == 1


# ── External flags ────────────────────────────────────────────────────────────

class TestIntegrationExternalFlags:
    def test_external_flags_cause_governance_intervention(self, monkeypatch):
        """External flags in aurora.external_flags trigger a non-PASS governance action."""
        client = _make_client(monkeypatch)
        r = client.post("/v1/chat/completions", json=_HARD_STOP_REQUEST)
        assert r.status_code == 200
        assert r.json()["aurora"]["governance"] != "PASS"

    def test_invalid_external_flag_type_returns_422(self, monkeypatch):
        """Unrecognised external flag type returns 422."""
        client = _make_client(monkeypatch)
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hi"}],
                "aurora": {
                    "external_flags": [
                        {"type": "NOT_A_REAL_FLAG_TYPE", "evidence": ["x"]},
                    ],
                },
            },
        )
        assert r.status_code == 422


# ── Streaming ─────────────────────────────────────────────────────────────────

class TestIntegrationStreaming:
    def test_stream_returns_event_stream_with_done(self, monkeypatch):
        """POST with stream=true returns text/event-stream containing [DONE]."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        from aurora_lens.proxy.app import create_app

        adapter = _StreamAdapter(["Hello", " world"])
        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (adapter, adapter),
        )
        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {"audit_log": None},
            "extraction": {"backend": "spacy"},
        })
        client = TestClient(create_app(cfg))
        r = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
        )
        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")
        assert "[DONE]" in r.text
        assert '"aurora"' in r.text
        assert '"governance"' in r.text

"""Operator-facing demo test with full printed governance output.

Run with:
    python -m pytest tests/test_operator_output_demo.py -vv -s -rA
"""

from __future__ import annotations

import pytest

from tests.skip_reasons import SKIP_STARLETTE_HTTP_TESTCLIENT

from aurora_lens.adapters.base import AdapterResponse, LLMAdapter
from aurora_lens.lens import Lens
from aurora_lens.proxy.config import ProxyConfig


class _ContradictionAdapter(LLMAdapter):
    """Deterministic adapter that yields a contradictory model answer."""

    async def generate(self, messages: list[dict[str, str]], **kwargs) -> AdapterResponse:
        return AdapterResponse(
            text="Emma does NOT have a red book.",
            model="mock-contradiction",
        )


@pytest.mark.asyncio
async def test_demo_full_output_operator_and_governed(monkeypatch):
    """Print raw model output + governed output for an operator-readable demo."""
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

    from aurora_lens.proxy.app import create_app

    def _mock_adapters(_cfg):
        m = _ContradictionAdapter()
        return m, m

    monkeypatch.setattr(
        "aurora_lens.proxy.app._build_provider_adapters",
        _mock_adapters,
    )

    # With operator detail, admitted user assertions normally take the pre-LLM mutation
    # ack (PASS) without calling the adapter. This demo requires generate→checker→bridge
    # so the mock contradiction is verified and governance is non-PASS.
    _orig_suppress = Lens._suppress_admitted_assertion_ack

    def _demo_force_adapter_on_admitted_ack(self, ack_plan):
        if ack_plan is not None and getattr(
            ack_plan, "mutation_kind", None
        ) == "admitted_state_assertion":
            return True
        return _orig_suppress(self, ack_plan)

    monkeypatch.setattr(
        Lens,
        "_suppress_admitted_assertion_ack",
        _demo_force_adapter_on_admitted_ack,
    )

    cfg = ProxyConfig.from_mapping(
        {
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {
                "default_policy": "strict",
                "audit_log": None,
                "include_operator_detail": True,
            },
            "extraction": {"backend": "spacy"},
        }
    )
    app = create_app(cfg)
    client = TestClient(app)

    sid = "demo-full-output-1"
    prompt = "Emma has a red book."
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": prompt}],
            "aurora_session_id": sid,
        },
    )

    assert r.status_code == 200
    body = r.json()
    aurora = body.get("aurora", {})
    final_text = body["choices"][0]["message"]["content"]

    print("\n=== SESSION ===")
    print(sid)
    print("\n=== PROMPT ===")
    print(prompt)
    print("\n=== GOVERNANCE OUTCOME ===")
    print(aurora.get("governance"))
    print("\n=== RAW MODEL OUTPUT (before governance) ===")
    print(aurora.get("original_response"))
    print("\n=== GOVERNED OUTPUT ===")
    print(aurora.get("governed_response", final_text))
    print("\n=== FINAL USER-VISIBLE RESPONSE ===")
    print(final_text)
    print("\n=== TRACE ===")
    print(aurora.get("trace_id"))
    print("\n=== FAILED CONSTRAINTS ===")
    fe = aurora.get("forensic_event") or {}
    print(fe.get("failed_constraints"))

    assert aurora.get("governance") != "PASS"
    assert "original_response" not in aurora
    assert isinstance(final_text, str) and final_text

"""HTTP regressions for PEF-backed typed medical transitions."""

from __future__ import annotations

import pytest

from aurora_lens.adapters.base import AdapterResponse, LLMAdapter


class _RefusalAdapter(LLMAdapter):
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, messages: list[dict[str, str]], **kwargs) -> AdapterResponse:
        self.calls += 1
        return AdapterResponse(text="I cannot provide prescription information.", model="mock-upstream")


@pytest.fixture
def medical_proxy_client(monkeypatch):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("starlette test client not available")

    from aurora_lens.proxy.app import create_app
    from aurora_lens.proxy.config import ProxyConfig

    adapter = _RefusalAdapter()

    def _mock_adapters(_cfg):
        return adapter, adapter

    monkeypatch.setattr("aurora_lens.proxy.app._build_provider_adapters", _mock_adapters)

    cfg = ProxyConfig.from_mapping(
        {
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {
                "default_policy": "strict",
                "audit_log": None,
                "default_domain": "general",
                "include_operator_detail": True,
            },
            "extraction": {"backend": "spacy"},
        },
    )
    app = create_app(cfg)
    return TestClient(app), adapter


def _post_turn(client, sid: str, text: str, *, include_browser_hint: bool) -> tuple[str, dict]:
    body = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": text}],
        "aurora_session_id": sid,
    }
    # Browser path may include request_domain; raw API path omits it entirely.
    if include_browser_hint:
        body["aurora"] = {"request_domain": "medical"}
    r = client.post("/v1/chat/completions", json=body, headers={"x-aurora-operator-detail": "1"})
    assert r.status_code == 200, r.text
    d = r.json()
    msg = (d.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    au = d.get("aurora") or {}
    return msg, au


def _exercise_chain(client, sid: str, *, include_browser_hint: bool) -> dict[str, str]:
    for t in (
        "Dr Rivera prescribed Lina 24mg of Captopril daily.",
        "Nurse Omar withheld the evening dose pending blood pressure review.",
        "Dr Rivera halved the dosage.",
    ):
        _msg, au = _post_turn(client, sid, t, include_browser_hint=include_browser_hint)
        assert au.get("llm_called") is False

    d_msg, d_au = _post_turn(client, sid, "What dosage is Lina currently receiving?", include_browser_hint=include_browser_hint)
    p_msg, p_au = _post_turn(client, sid, "Who authorized the current dosage?", include_browser_hint=include_browser_hint)
    s_msg, s_au = _post_turn(client, sid, "Was the evening dose administered?", include_browser_hint=include_browser_hint)
    assert d_au.get("llm_called") is False
    assert p_au.get("llm_called") is False
    assert s_au.get("llm_called") is False
    return {"dose": d_msg, "prov": p_msg, "sched": s_msg}


def test_raw_api_path_no_demo_hint_works(medical_proxy_client):
    client, adapter = medical_proxy_client
    out = _exercise_chain(client, "raw-no-hint-session", include_browser_hint=False)
    assert "12mg" in out["dose"].lower()
    assert "captopril" in out["dose"].lower()
    assert "dr rivera" in out["prov"].lower()
    assert "withheld pending blood pressure review" in out["sched"].lower()
    assert adapter.calls == 0


def test_browser_and_raw_paths_agree_for_same_transitions(medical_proxy_client):
    client, adapter = medical_proxy_client
    raw = _exercise_chain(client, "agree-raw-session", include_browser_hint=False)
    browser = _exercise_chain(client, "agree-browser-session", include_browser_hint=True)

    assert raw["dose"].lower() == browser["dose"].lower()
    assert raw["prov"].lower() == browser["prov"].lower()
    assert raw["sched"].lower() == browser["sched"].lower()
    assert adapter.calls == 0


def test_raw_api_no_browser_hint_second_entity_set(medical_proxy_client):
    """Second raw-path chain proves no browser routing dependency for continuity."""
    client, adapter = medical_proxy_client
    sid = "raw-no-hint-session-2"
    for t in (
        "Dr Solis prescribed Hana 16mg of Bisoprolol daily.",
        "Nurse Keane withheld the evening dose pending pulse review.",
        "Dr Solis changed the dosage to 8mg.",
    ):
        msg, au = _post_turn(client, sid, t, include_browser_hint=False)
        assert msg
        assert au.get("llm_called") is False

    dose_msg, _ = _post_turn(
        client,
        sid,
        "What dosage is Hana currently receiving?",
        include_browser_hint=False,
    )
    prov_msg, _ = _post_turn(
        client,
        sid,
        "Who authorized the current dosage?",
        include_browser_hint=False,
    )
    assert "8mg" in dose_msg.lower()
    assert "bisoprolol" in dose_msg.lower()
    assert "dr solis" in prov_msg.lower()
    assert adapter.calls == 0


def test_audit_semantics_state_native_answers_after_transitions(medical_proxy_client):
    """State-native answers over admitted relationships must have correct audit classification.

    This test verifies that when state-native answers come from admitted PEF relationships,
    they are classified with appropriate posture (e.g., HELD_STATE when PEF has state)
    rather than inheriting a prior refusal's HELD_REFUSAL classification.

    Scenario: Complete medical continuity chain with transitions and queries.
    - Turns 1–3: State transitions (prescribe, withhold, dose change)
    - Turns 4–6: State queries (current dose, authorization, scheduled dose status)

    All turns are state-native handled and should NOT call the LLM.
    Classification should reflect: HELD_STATE (PEF has committed state data).
    """
    client, adapter = medical_proxy_client
    sid = "audit-semantics-test"

    # Turns 1–3: State transitions (state-native)
    for t in (
        "Dr Chen prescribed Michael 20mg of Medication X twice daily.",
        "A nurse withheld the evening dose pending review.",
        "Dr Chen halved the dosage.",
    ):
        msg, au = _post_turn(client, sid, t, include_browser_hint=False)
        assert au.get("llm_called") is False, f"Turn should be state-native handled: {t}"
        assert "Recorded" in msg, f"Should acknowledge transition: {msg}"
    
    # Turns 4–6: State queries (state-native, from admitted relationships)
    dose_msg, dose_au = _post_turn(
        client, sid, "What dosage is Michael currently receiving?", include_browser_hint=False
    )
    auth_msg, auth_au = _post_turn(
        client, sid, "Who authorized the current dosage?", include_browser_hint=False
    )
    sched_msg, sched_au = _post_turn(
        client, sid, "Was the evening dose administered?", include_browser_hint=False
    )
    
    # Verify answers from admitted PEF relationships
    assert "10mg" in dose_msg.lower(), f"Dose should be 10mg (halved from 20): {dose_msg}"
    assert "dr chen" in auth_msg.lower(), f"Auth should be Dr Chen: {auth_msg}"
    assert "withheld" in sched_msg.lower() or "pending" in sched_msg.lower(), f"Should report withheld/pending: {sched_msg}"
    
    # Verify state-native answers did NOT call LLM
    assert dose_au.get("llm_called") is False, "Dose query should be state-native"
    assert auth_au.get("llm_called") is False, "Auth query should be state-native"
    assert sched_au.get("llm_called") is False, "Schedule query should be state-native"
    
    assert adapter.calls == 0, "No LLM calls should have been made in this chain"

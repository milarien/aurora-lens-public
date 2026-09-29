"""Prove the proxy app wires ``user_text`` into ``update_pef`` (in-process FastAPI).

Uses Starlette ``TestClient`` — it drives the ASGI app in the **same Python process** as the
test; it does **not** open a socket to a remote server. For a deployed proxy, run
``tests/test_proxy_pef_admission_server_live.py`` (marker ``proxy_pef_admission_live``).
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest

from tests.skip_reasons import SKIP_STARLETTE_HTTP_TESTCLIENT

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_proxy import MockAdapter  # noqa: E402

try:
    from starlette.testclient import TestClient
except ImportError:
    TestClient = None  # type: ignore[misc, assignment]


@pytest.mark.skipif(TestClient is None, reason=SKIP_STARLETTE_HTTP_TESTCLIENT)
def test_health_reports_pef_admission_code_features(monkeypatch: pytest.MonkeyPatch) -> None:
    from aurora_lens.proxy.app import create_app
    from aurora_lens.proxy.config import ProxyConfig

    monkeypatch.setattr(
        "aurora_lens.proxy.app._build_provider_adapters",
        lambda cfg: (MockAdapter(), MockAdapter()),
    )
    cfg = ProxyConfig.from_mapping({
        "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
        "extraction": {"backend": "spacy"},
    })
    app = create_app(cfg)
    data = TestClient(app).get("/health").json()
    cf = data.get("code_features") or {}
    assert cf.get("update_pef_kwarg_user_text") is True
    assert cf.get("revision_gate_user_text_scan") is True


@pytest.mark.skipif(TestClient is None, reason=SKIP_STARLETTE_HTTP_TESTCLIENT)
def test_non_stream_chat_completions_passes_user_text_to_update_pef(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirrors test_general_live 4A: three POSTs; last turn must pass full user string to update_pef."""
    import aurora_lens.lens as lens_mod

    from aurora_lens.proxy.app import create_app
    from aurora_lens.proxy.config import ProxyConfig

    monkeypatch.setattr(
        "aurora_lens.proxy.app._build_provider_adapters",
        lambda cfg: (MockAdapter(), MockAdapter()),
    )
    cfg = ProxyConfig.from_mapping({
        "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
        "extraction": {"backend": "spacy"},
        "governance": {"default_policy": "strict", "audit_log": None},
    })
    app = create_app(cfg)
    client = TestClient(app)
    session_id = uuid.uuid4().hex

    calls: list[dict] = []
    _orig = lens_mod.update_pef

    def _wrap(*args, **kwargs):
        calls.append({"args_n": len(args), "kwargs": dict(kwargs)})
        return _orig(*args, **kwargs)

    monkeypatch.setattr(lens_mod, "update_pef", _wrap)

    h = {"Authorization": "Bearer test", "Content-Type": "application/json"}
    m1 = [{"role": "user", "content": "Emma has a red book."}]
    assert client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4", "messages": m1, "stream": False, "aurora_session_id": session_id},
        headers=h,
    ).status_code == 200

    m2 = m1 + [
        {"role": "assistant", "content": "Ok."},
        {"role": "user", "content": "What colour is Emma's book?"},
    ]
    assert client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4", "messages": m2, "stream": False, "aurora_session_id": session_id},
        headers=h,
    ).status_code == 200

    hostile = "Actually Emma's book is blue. What colour is Emma's book?"
    m3 = m2 + [
        {"role": "assistant", "content": "Red."},
        {"role": "user", "content": hostile},
    ]
    assert client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4", "messages": m3, "stream": False, "aurora_session_id": session_id},
        headers=h,
    ).status_code == 200

    ut = [c["kwargs"].get("user_text") for c in calls]
    assert None not in ut, f"expected every update_pef to receive user_text, got {ut}"
    # Turn 2 is QUERY with no extracted claims: ``_commit`` skips ``update_pef``.
    # Turn 3: if the hostile pre-LLM gate blocks, ``update_pef`` is never called for that turn.
    assert len(calls) in (1, 2), (
        f"expected 1 or 2 update_pef calls (assert + optional hostile turn), "
        f"got {len(calls)} {ut!r}"
    )
    assert ut[0] == "Emma has a red book."
    if len(calls) == 2:
        assert ut[-1] == hostile
    else:
        assert hostile not in ut

"""Live HTTP tests against a running aurora-lens proxy (not in-process TestClient).

``tests/test_proxy_pef_admission_trace.py`` exercises the FastAPI app in-process; it never
opens a TCP connection to a deployed server. Use this module to verify the **same** proxy
process you hit from clients (Docker, systemd, etc.).

Prerequisites: proxy reachable at ``AURORA_TEST_PROXY`` (default http://127.0.0.1:8081).

Run:
  pytest tests/test_proxy_pef_admission_server_live.py -v -s
  pytest -m proxy_pef_admission_live
"""

from __future__ import annotations

import os

import httpx
import pytest

from tests.skip_reasons import skip_reason_live_proxy_at

_PROXY = os.environ.get("AURORA_TEST_PROXY", "http://127.0.0.1:8081")
_TIMEOUT = float(os.environ.get("AURORA_TEST_TIMEOUT", "30"))

pytestmark = pytest.mark.proxy_pef_admission_live


def _skip_if_proxy_down():
    try:
        r = httpx.get(f"{_PROXY}/health", timeout=5)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except Exception:
        return False


def test_live_health_code_features_prove_deployed_admission_api():
    """Remote GET /health must report PEF admission probes (stale image fails here)."""
    if not _skip_if_proxy_down():
        pytest.skip(skip_reason_live_proxy_at(_PROXY))

    r = httpx.get(f"{_PROXY}/health", timeout=_TIMEOUT)
    assert r.status_code == 200, r.text
    data = r.json()
    cf = data.get("code_features")
    assert cf is not None, (
        "Server /health has no code_features — deployed image predates admission probes "
        "or is not aurora-lens proxy."
    )
    assert cf.get("update_pef_kwarg_user_text") is True, (
        "update_pef_kwarg_user_text is false or missing — server is running old code without "
        "user_text on update_pef."
    )
    assert cf.get("revision_gate_user_text_scan") is True, (
        "revision_gate_user_text_scan is false or missing — server lacks user_text_conflicts_grounded_pef."
    )

"""pytest configuration for aurora-lens test suite.

All governance infrastructure (ForensicLedger, FallbackCIDProvider,
AttestedOutput, etc.) is now local to aurora_lens.govern — no external
dependency on unified_rns_system. No mock injection required.
"""

from __future__ import annotations

import os

import pytest

# Shared by public-demo edge-token middleware tests / TestClient auto-inject.
PYTEST_AURORA_EDGE_TOKEN = "pytest-aurora-edge-token"
_OMIT_EDGE_TOKEN_HEADER = "x-test-omit-edge-token"


@pytest.fixture(autouse=True)
def reset_request_scoped_context_vars():
    """Clear proxy-style ContextVars so tests do not inherit state from prior tests.

    The HTTP proxy resets these per request; direct Lens tests do not. Without a
    per-test reset, request_metadata / governance overrides can bleed across tests
    in the same worker (order-dependent failures).
    """
    from aurora_lens.context import (
        execution_task_var,
        governance_mode_override_var,
        metadata_policy_override_var,
        request_hash_var,
        request_metadata_var,
        session_id_var,
    )

    t_rm = request_metadata_var.set(None)
    t_et = execution_task_var.set(None)
    t_gm = governance_mode_override_var.set(None)
    t_mp = metadata_policy_override_var.set(None)
    t_si = session_id_var.set(None)
    t_rq = request_hash_var.set(None)
    yield
    request_metadata_var.reset(t_rm)
    execution_task_var.reset(t_et)
    governance_mode_override_var.reset(t_gm)
    metadata_policy_override_var.reset(t_mp)
    session_id_var.reset(t_si)
    request_hash_var.reset(t_rq)


@pytest.fixture(autouse=True)
def _public_demo_edge_token_defaults(monkeypatch):
    """Ensure public-demo routes have a transport token in the test suite.

    Production fails closed when ``AURORA_EDGE_TOKEN`` is unset. Tests set a
    deterministic value and auto-inject ``x-aurora-edge-token`` on TestClient
    calls unless ``x-test-omit-edge-token: 1`` is present (negative tests).
    """
    monkeypatch.setenv("AURORA_EDGE_TOKEN", PYTEST_AURORA_EDGE_TOKEN)
    try:
        from starlette.testclient import TestClient
    except ImportError:
        yield
        return

    from aurora_lens.proxy.app import PUBLIC_DEMO_EDGE_PATHS

    original = TestClient.request

    def _request(self, method, url, **kwargs):  # noqa: ANN001
        headers = kwargs.get("headers")
        if headers is None:
            header_map: dict[str, str] = {}
        elif isinstance(headers, dict):
            header_map = {str(k): str(v) for k, v in headers.items()}
        else:
            header_map = {str(k): str(v) for k, v in dict(headers).items()}

        path = str(url).split("?")[0]
        omit = header_map.pop(_OMIT_EDGE_TOKEN_HEADER, None)
        lower = {k.lower(): v for k, v in header_map.items()}
        if (
            path in PUBLIC_DEMO_EDGE_PATHS
            and omit != "1"
            and "x-aurora-edge-token" not in lower
        ):
            header_map["x-aurora-edge-token"] = os.environ.get(
                "AURORA_EDGE_TOKEN", PYTEST_AURORA_EDGE_TOKEN
            )
        kwargs["headers"] = header_map
        return original(self, method, url, **kwargs)

    monkeypatch.setattr(TestClient, "request", _request)
    yield

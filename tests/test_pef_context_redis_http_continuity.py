"""HTTP-level proof that PEF continuity survives a Redis-backed session store.

``tests/test_redis_session_integration.py`` proves the *store* round-trips a
``SessionRecord`` through Redis. It never drives the proxy's chat-completions
endpoint at all. This module closes that gap: two real HTTP requests through
``create_app()`` with ``session.backend: redis``, proving a held PEF state
(the same referential-admission contract as
``tests/test_pef_context_http_continuity.py``) actually survives a second HTTP
request when Redis, not the in-memory store, backs the session.

Runs when Redis is reachable at ``AURORA_LENS_TEST_REDIS_URL`` (default
``redis://127.0.0.1:6379/15``); skips cleanly otherwise (same convention as
``test_redis_session_integration.py``). Marked ``@pytest.mark.redis``.
"""

from __future__ import annotations

import os

import pytest

from tests.skip_reasons import SKIP_STARLETTE_HTTP_TESTCLIENT, skip_reason_redis_unavailable

pytestmark = pytest.mark.redis

_REDIS_URL = os.environ.get(
    "AURORA_LENS_TEST_REDIS_URL",
    "redis://127.0.0.1:6379/15",
)


def _redis_available(url: str) -> bool:
    try:
        import redis
    except ImportError:
        return False
    try:
        client = redis.from_url(url, decode_responses=True)
        return bool(client.ping())
    except Exception:
        return False


@pytest.fixture(scope="module")
def redis_url() -> str:
    if not _redis_available(_REDIS_URL):
        pytest.skip(skip_reason_redis_unavailable(_REDIS_URL))
    return _REDIS_URL


def _make_app(monkeypatch, redis_url: str):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

    from aurora_lens.proxy.app import create_app
    from aurora_lens.proxy.config import ProxyConfig
    from tests.test_proxy import MockAdapter

    monkeypatch.setattr(
        "aurora_lens.proxy.app._build_provider_adapters",
        lambda cfg: (MockAdapter(), MockAdapter()),
    )
    cfg = ProxyConfig.from_mapping({
        "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
        "governance": {
            "default_policy": "strict",
            "audit_log": None,
            "include_operator_detail": True,
        },
        "extraction": {"backend": "spacy"},
        "session": {
            "backend": "redis",
            "redis_url": redis_url,
            # Short TTL: each session id is a fresh uuid4 minted per test run
            # (see mint_session_id()), so collisions across runs are not a
            # concern; the short TTL just keeps the test DB from accumulating.
            "ttl_seconds": 120,
        },
    })
    app = create_app(cfg)
    return TestClient(app), app


class TestPefContextRedisHttpContinuity:
    """Two HTTP calls, one Redis-backed session store, same create_app() instance."""

    def test_second_call_resolves_demonstrative_via_redis_backed_session(self, monkeypatch, redis_url):
        client, _ = _make_app(monkeypatch, redis_url)

        r1 = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Escrow has a transaction."}],
            },
        )
        assert r1.status_code == 200, r1.text
        body1 = r1.json()
        pef_context_id = body1["aurora"].get("pef_context_id")
        assert pef_context_id, "first response must mint and return aurora.pef_context_id"

        r2 = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "That transaction must remain off the books."}],
            },
            headers={"x-aurora-pef-context-id": pef_context_id},
        )
        assert r2.status_code == 200, r2.text
        aurora2 = r2.json()["aurora"]

        assert aurora2.get("pef_context_id") == pef_context_id
        assert aurora2.get("governance") != "CONTAIN", (
            f"demonstrative with a real antecedent loaded from Redis must not hold as "
            f"unresolved; got aurora={aurora2!r}"
        )
        assert "UNRESOLVED_REFERENT" not in set(aurora2.get("flags") or [])

        entity_count_call1 = (
            (body1["aurora"].get("operator_pef") or {}).get("counts", {}).get("entities")
        )
        entity_count_call2 = (
            (aurora2.get("operator_pef") or {}).get("counts", {}).get("entities")
        )
        assert entity_count_call1 == 1, f"call 1 must commit exactly the Escrow entity; got {entity_count_call1}"
        assert entity_count_call2 == entity_count_call1, (
            f"resolving a demonstrative against a Redis-persisted antecedent must not mint "
            f"a new entity; call1={entity_count_call1} call2={entity_count_call2}"
        )

    def test_continuity_survives_two_separate_create_app_instances_sharing_redis(self, monkeypatch, redis_url):
        """The actual multi-instance claim: two independent proxy processes (two
        separate create_app() instances, no shared Python object) exchanging PEF
        continuity only through Redis — the shape of a real horizontally-scaled
        deployment behind a load balancer."""
        client_a, _ = _make_app(monkeypatch, redis_url)
        client_b, _ = _make_app(monkeypatch, redis_url)

        r1 = client_a.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Escrow has a transaction."}],
            },
        )
        assert r1.status_code == 200, r1.text
        pef_context_id = r1.json()["aurora"].get("pef_context_id")
        assert pef_context_id

        # Second call lands on a *different* app/process — it must resolve the
        # demonstrative purely from what instance A persisted to Redis.
        r2 = client_b.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "That transaction must remain off the books."}],
            },
            headers={"x-aurora-pef-context-id": pef_context_id},
        )
        assert r2.status_code == 200, r2.text
        aurora2 = r2.json()["aurora"]

        assert aurora2.get("pef_context_id") == pef_context_id
        assert aurora2.get("governance") != "CONTAIN", (
            f"a second proxy instance sharing only Redis must still resolve the "
            f"demonstrative against instance A's committed antecedent; got aurora={aurora2!r}"
        )
        assert "UNRESOLVED_REFERENT" not in set(aurora2.get("flags") or [])

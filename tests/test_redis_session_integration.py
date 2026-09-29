"""Optional Redis session store integration (Phase C).

Runs when Redis is reachable at ``AURORA_LENS_TEST_REDIS_URL`` (default
``redis://127.0.0.1:6379/15``). Skips cleanly when Redis is unavailable so
local and default CI without Redis still pass.

Use in CI with a Redis service and ``AURORA_LENS_TEST_REDIS_URL`` set.
"""
from __future__ import annotations

import os
import time
import uuid

import pytest

from tests.skip_reasons import skip_reason_redis_unavailable

from aurora_lens.pef.state import PEFState
from aurora_lens.proxy.session_store import RedisSessionStore, SessionRecord

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


def test_redis_session_put_get_roundtrip(redis_url: str) -> None:
    sid = f"test-{uuid.uuid4().hex}"
    store = RedisSessionStore(
        redis_url,
        ttl_seconds=60,
        key_prefix=f"aurora:test:{uuid.uuid4().hex}:session:",
        lock_acquire_timeout_seconds=5.0,
        lock_lease_seconds=30.0,
    )
    pef = PEFState()
    now = time.time()
    rec = SessionRecord.from_pef(pef, expires_at=now + 60.0, revision=0)
    store.put(sid, rec)
    try:
        loaded = store.get(sid)
        assert loaded is not None
        assert loaded.revision == 0
        assert loaded.pef_state == rec.pef_state
    finally:
        store.delete(sid)


def test_redis_session_lock_acquire_release(redis_url: str) -> None:
    sid = f"test-lock-{uuid.uuid4().hex}"
    store = RedisSessionStore(
        redis_url,
        ttl_seconds=60,
        key_prefix=f"aurora:test:{uuid.uuid4().hex}:session:",
        lock_acquire_timeout_seconds=5.0,
        lock_lease_seconds=30.0,
    )
    with store.acquire_lock(sid):
        pass

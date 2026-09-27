"""Session manager — maps conversation IDs to Lens instances.

Phase C: Uses SessionStore (memory or Redis). All read-modify-write occurs inside lock.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

from aurora_lens.context import lock_metadata_var
from aurora_lens.lens import Lens
from aurora_lens.config import LensConfig

if TYPE_CHECKING:
    from aurora_lens.proxy.session_store import SessionStore, SessionLockTimeoutError


def _create_session_store(
    backend: str,
    ttl_seconds: int,
    redis_url: str,
    lock_acquire_timeout_seconds: float = 10.0,
    lock_lease_seconds: float = 240.0,
) -> "SessionStore":
    """Create SessionStore from config."""
    from aurora_lens.proxy.session_store import (
        MemorySessionStore,
        RedisSessionStore,
    )
    lock_kw = dict(
        lock_acquire_timeout_seconds=lock_acquire_timeout_seconds,
        lock_lease_seconds=lock_lease_seconds,
    )
    if backend == "redis":
        if not redis_url or not redis_url.strip():
            raise ValueError(
                "session.backend=redis requires session.redis_url or AURORA_LENS_REDIS_URL"
            )
        return RedisSessionStore(redis_url=redis_url, ttl_seconds=ttl_seconds, **lock_kw)
    return MemorySessionStore(ttl_seconds=ttl_seconds, **lock_kw)


class SessionManager:
    """Maps conversation IDs to Lens instances. Persists via SessionStore.

    Contract: get, get_or_create, persist must be called only inside with_lock().
    """

    def __init__(
        self,
        config_factory: callable,
        store: "SessionStore | None" = None,
        *,
        ttl_seconds: int = 3600,
        backend: str = "memory",
        redis_url: str = "",
        lock_acquire_timeout_seconds: float = 10.0,
        lock_lease_seconds: float = 240.0,
    ):
        self._config_factory = config_factory
        self._ttl = ttl_seconds
        self._lock_acquire_timeout = lock_acquire_timeout_seconds
        if store is not None:
            self._store = store
        else:
            self._store = _create_session_store(
                backend, ttl_seconds, redis_url,
                lock_acquire_timeout_seconds=lock_acquire_timeout_seconds,
                lock_lease_seconds=lock_lease_seconds,
            )

    @contextmanager
    def with_lock(self, session_id: str):
        """Acquire session lock. All get/get_or_create/persist must be inside.
        Sets lock_metadata_var (lock_wait_ms, lock_acquired, lock_timeout_ms on timeout)."""
        from aurora_lens.proxy.session_store import SessionLockTimeoutError

        t0 = time.perf_counter()
        try:
            with self._store.acquire_lock(session_id):
                lock_wait_ms = int((time.perf_counter() - t0) * 1000)
                meta = {
                    "session_id": session_id,
                    "lock_wait_ms": lock_wait_ms,
                    "lock_acquired": True,
                }
                token = lock_metadata_var.set(meta)
                try:
                    yield
                finally:
                    lock_metadata_var.reset(token)
        except SessionLockTimeoutError:
            lock_wait_ms = int((time.perf_counter() - t0) * 1000)
            lock_timeout_ms = int(self._lock_acquire_timeout * 1000)
            lock_metadata_var.set({
                "session_id": session_id,
                "lock_wait_ms": lock_wait_ms,
                "lock_acquired": False,
                "lock_timeout_ms": lock_timeout_ms,
            })
            raise

    def get_or_create(self, session_id: str) -> tuple[Lens, bool]:
        """Get or create Lens. MUST be called inside with_lock().
        Returns (lens, is_new_session).
        Guardrail: session_id must never be empty in proxy mode."""
        from aurora_lens.proxy.session_store import SessionRecord

        if not session_id or not str(session_id).strip():
            raise ValueError("session_id must not be empty in proxy mode")
        record = self._store.get(session_id)
        if record is not None:
            pef = record.to_pef()
            config = self._config_factory()
            return Lens(config, initial_pef=pef, session_id=session_id), False

        config = self._config_factory()
        lens = Lens(config, session_id=session_id)
        expires_at = time.time() + self._ttl
        new_record = SessionRecord.from_pef(lens.pef, expires_at, revision=0)
        self._store.put(session_id, new_record)
        return lens, True

    def get(self, session_id: str) -> Lens | None:
        """Get existing Lens if session exists. MUST be called inside with_lock()."""
        record = self._store.get(session_id)
        if record is None:
            return None
        config = self._config_factory()
        return Lens(config, initial_pef=record.to_pef(), session_id=session_id)

    def persist(self, session_id: str, lens: Lens) -> None:
        """Persist session. Increments revision. MUST be called inside with_lock()."""
        from aurora_lens.proxy.session_store import SessionRecord

        record = self._store.get(session_id)
        revision = (record.revision + 1) if record is not None else 0
        expires_at = time.time() + self._ttl
        new_record = SessionRecord.from_pef(lens.pef, expires_at, revision)
        self._store.put(session_id, new_record)

    def cleanup_expired(self) -> int:
        """Remove expired sessions."""
        return self._store.cleanup_expired()

    @property
    def active_count(self) -> int:
        if hasattr(self._store, "active_count"):
            return self._store.active_count
        return 0

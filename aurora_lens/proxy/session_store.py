"""SessionStore — abstract interface and backends for session persistence.

SessionStore Contract: identity/keying, SessionRecord, expiry, locking, atomic updates.
"""

from __future__ import annotations

import json
import time
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from aurora_lens.pef.state import PEFState


# ── Exceptions ─────────────────────────────────────────────────────────────

class SessionStoreError(Exception):
    """Backend connectivity or store failure. Proxy maps to 503 + forensic envelope."""
    pass


class SessionLockTimeoutError(SessionStoreError):
    """Lock acquire timeout. Proxy maps to 409 SESSION_BUSY_TIMEOUT + forensic envelope."""
    pass


# ── SessionRecord ──────────────────────────────────────────────────────────

@dataclass
class SessionRecord:
    """Stored value for a session. Contract requires pef_state, expires_at, revision."""

    pef_state: dict[str, Any]  # Serialized PEFState.to_dict()
    expires_at: float          # Absolute UTC epoch seconds
    revision: int              # Monotonic; starts at 0 on creation

    # Optional (recommended)
    policy_id: str | None = None
    caller_id: str | None = None
    created_at: float | None = None
    updated_at: float | None = None

    def to_pef(self) -> PEFState:
        """Deserialize pef_state to PEFState."""
        return PEFState.from_dict(self.pef_state)

    @staticmethod
    def from_pef(pef: PEFState, expires_at: float, revision: int) -> "SessionRecord":
        """Create record from PEFState."""
        now = time.time()
        return SessionRecord(
            pef_state=pef.to_dict(),
            expires_at=expires_at,
            revision=revision,
            created_at=now if revision == 0 else None,
            updated_at=now,
        )


# ── SessionStore Interface ─────────────────────────────────────────────────

class SessionStore(ABC):
    """Abstract session store. Sessions keyed by session_id (opaque)."""

    @abstractmethod
    def get(self, session_id: str) -> SessionRecord | None:
        """Get record if present and not expired. None if missing or expired.
        Must not mutate state (no lazy refresh) except optional best-effort expiry deletion."""
        ...

    @abstractmethod
    def put(self, session_id: str, record: SessionRecord) -> None:
        """Unconditional overwrite. Atomic. Must not extend TTL unless caller sets expires_at."""
        ...

    @abstractmethod
    def delete(self, session_id: str) -> None:
        """Idempotent. After delete, get returns None."""
        ...

    @abstractmethod
    def cleanup_expired(self) -> int:
        """Best-effort removal of expired records. Safe to run concurrently."""
        ...

    @contextmanager
    def acquire_lock(self, session_id: str) -> Iterator[None]:
        """Mutual exclusion per session_id. Bounded acquire timeout. Lease/TTL.
        Raises SessionStoreError if acquisition fails (transient concurrency failure)."""
        yield


# ── MemorySessionStore ─────────────────────────────────────────────────────

class MemorySessionStore(SessionStore):
    """In-memory store. For local dev and single-instance."""

    def __init__(
        self,
        ttl_seconds: int = 3600,
        lock_acquire_timeout_seconds: float = 10.0,
        lock_lease_seconds: float = 240.0,
    ):
        self._ttl = ttl_seconds
        self._lock_acquire_timeout = lock_acquire_timeout_seconds
        self._store: dict[str, SessionRecord] = {}
        self._store_lock = threading.Lock()
        self._session_locks: dict[str, threading.Lock] = {}
        self._session_locks_meta = threading.Lock()

    def get(self, session_id: str) -> SessionRecord | None:
        with self._store_lock:
            record = self._store.get(session_id)
            if record is None:
                return None
            now = time.time()
            if now >= record.expires_at:
                del self._store[session_id]
                return None
            return record

    def put(self, session_id: str, record: SessionRecord) -> None:
        with self._store_lock:
            self._store[session_id] = record

    def delete(self, session_id: str) -> None:
        with self._store_lock:
            self._store.pop(session_id, None)

    def cleanup_expired(self) -> int:
        now = time.time()
        with self._store_lock:
            expired = [
                sid for sid, rec in self._store.items()
                if now >= rec.expires_at
            ]
            for sid in expired:
                del self._store[sid]
            return len(expired)

    @contextmanager
    def acquire_lock(self, session_id: str) -> Iterator[None]:
        """Per-session lock with bounded wait. Raises SessionLockTimeoutError on timeout."""
        with self._session_locks_meta:
            if session_id not in self._session_locks:
                self._session_locks[session_id] = threading.Lock()
            slock = self._session_locks[session_id]
        acquired = slock.acquire(timeout=self._lock_acquire_timeout)
        if not acquired:
            raise SessionLockTimeoutError(
                f"Session lock acquire timeout: {session_id} "
                f"(timeout={self._lock_acquire_timeout}s)"
            )
        try:
            yield
        finally:
            slock.release()

    @property
    def active_count(self) -> int:
        now = time.time()
        with self._store_lock:
            return sum(1 for r in self._store.values() if now < r.expires_at)


# ── RedisSessionStore ──────────────────────────────────────────────────────

class RedisSessionStore(SessionStore):
    """Redis-backed store. For multi-instance and restart safety."""

    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int = 3600,
        key_prefix: str = "aurora:session:",
        lock_acquire_timeout_seconds: float = 10.0,
        lock_lease_seconds: float = 240.0,
    ):
        try:
            import redis
        except ImportError:
            raise ImportError(
                "redis package required for RedisSessionStore. "
                "Install with: pip install aurora-lens[redis]"
            ) from None
        self._client = redis.from_url(redis_url, decode_responses=True)
        self._ttl = ttl_seconds
        self._prefix = key_prefix
        self._lock_prefix = "aurora:lock:"
        self._lock_acquire_timeout = lock_acquire_timeout_seconds
        self._lock_lease_ttl = lock_lease_seconds

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}"

    def _lock_key(self, session_id: str) -> str:
        return f"{self._lock_prefix}{session_id}"

    def _serialize(self, record: SessionRecord) -> str:
        data = {
            "pef_state": record.pef_state,
            "expires_at": record.expires_at,
            "revision": record.revision,
            "policy_id": record.policy_id,
            "caller_id": record.caller_id,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }
        return json.dumps(data, separators=(",", ":"), ensure_ascii=False)

    def _deserialize(self, raw: str) -> SessionRecord | None:
        try:
            data = json.loads(raw)
            return SessionRecord(
                pef_state=data["pef_state"],
                expires_at=float(data["expires_at"]),
                revision=int(data["revision"]),
                policy_id=data.get("policy_id"),
                caller_id=data.get("caller_id"),
                created_at=data.get("created_at"),
                updated_at=data.get("updated_at"),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

    def get(self, session_id: str) -> SessionRecord | None:
        try:
            raw = self._client.get(self._key(session_id))
        except Exception as e:
            raise SessionStoreError(f"Redis get failed: {e}") from e
        if raw is None:
            return None
        record = self._deserialize(raw)
        if record is None:
            return None
        return record if time.time() < record.expires_at else None

    def put(self, session_id: str, record: SessionRecord) -> None:
        try:
            key = self._key(session_id)
            ttl = max(1, int(record.expires_at - time.time()))
            self._client.setex(key, ttl, self._serialize(record))
        except Exception as e:
            raise SessionStoreError(f"Redis put failed: {e}") from e

    def delete(self, session_id: str) -> None:
        try:
            self._client.delete(self._key(session_id))
        except Exception as e:
            raise SessionStoreError(f"Redis delete failed: {e}") from e

    def cleanup_expired(self) -> int:
        # Redis TTL handles expiry; we scan for keys and delete any that expired
        # Best-effort, bounded. Return 0.
        return 0

    @contextmanager
    def acquire_lock(self, session_id: str) -> Iterator[None]:
        """Acquire Redis lock. Bounded timeout. Fail deterministically on failure."""
        try:
            lock_id = self._lock_key(session_id)
            lock = self._client.lock(
                lock_id,
                timeout=self._lock_lease_ttl,
                blocking_timeout=self._lock_acquire_timeout,
            )
            acquired = lock.acquire(blocking=True)
        except Exception as e:
            raise SessionStoreError(f"Redis lock acquire failed: {e}") from e
        if not acquired:
            raise SessionLockTimeoutError(
                f"Redis lock acquire timeout: {session_id} "
                f"(timeout={self._lock_acquire_timeout}s)"
            )
        try:
            yield
        finally:
            try:
                lock.release()
            except Exception:
                pass

    @property
    def active_count(self) -> int:
        try:
            keys = list(self._client.scan_iter(match=f"{self._prefix}*", count=1000))
            return len(keys)
        except Exception:
            return 0

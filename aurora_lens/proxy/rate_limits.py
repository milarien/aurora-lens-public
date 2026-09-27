"""In-memory rate limiter — per-session and global.

Sliding-window style: count requests in the last N seconds.
Used for Phase 5 hardening.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict


class RateLimiter:
    """Sliding-window rate limiter. Thread-safe."""

    def __init__(self, requests_per_minute: int, window_seconds: int = 60):
        """
        Args:
            requests_per_minute: Max requests allowed per window.
            window_seconds: Window size in seconds (default 60).
        """
        self._rpm = max(0, requests_per_minute)
        self._window = max(1, window_seconds)
        self._per_key: dict[str, list[float]] = defaultdict(list)
        self._global_timestamps: list[float] = []
        self._lock = threading.Lock()

    def _prune(self, timestamps: list[float], now: float) -> list[float]:
        """Remove timestamps older than window."""
        cutoff = now - self._window
        return [t for t in timestamps if t > cutoff]

    def check(self, key: str | None = None) -> tuple[bool, str | None]:
        """
        Check if request is allowed. Returns (allowed, retry_after_seconds).
        key=None for global check.
        """
        if self._rpm == 0:
            return True, None

        now = time.monotonic()
        with self._lock:
            if key is None:
                self._global_timestamps = self._prune(self._global_timestamps, now)
                if len(self._global_timestamps) >= self._rpm:
                    oldest = min(self._global_timestamps)
                    retry_after = max(0, int(self._window - (now - oldest)))
                    return False, str(retry_after)
                self._global_timestamps.append(now)
            else:
                pruned = self._prune(self._per_key[key], now)
                if not pruned:
                    del self._per_key[key]
                else:
                    self._per_key[key] = pruned
                    if len(pruned) >= self._rpm:
                        oldest = min(pruned)
                        retry_after = max(0, int(self._window - (now - oldest)))
                        return False, str(retry_after)
                    pruned.append(now)
                    return True, None
                self._per_key[key].append(now)
        return True, None

"""Prometheus-compatible metrics. Phase 5 hardening.

Simple in-memory counters. No prometheus_client dependency.
Output format: Prometheus exposition format (text).
"""

from __future__ import annotations

import threading
import time
from typing import Any


class Metrics:
    """Thread-safe metrics store. Prometheus exposition format."""

    def __init__(self):
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}
        self._start_time = time.monotonic()

    def inc(self, name: str, value: int = 1, labels: dict[str, str] | None = None) -> None:
        """Increment a counter."""
        key = self._key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value

    def _escape_label_value(self, v: str) -> str:
        """Escape Prometheus label value: \\ \" \\n."""
        return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    def _key(self, name: str, labels: dict[str, str] | None) -> str:
        if not labels:
            return name
        parts = [f'{k}="{self._escape_label_value(str(v))}"' for k, v in sorted(labels.items())]
        return f"{name}{{{','.join(parts)}}}"

    def reset(self) -> None:
        """Reset all counters. For testing."""
        with self._lock:
            self._counters.clear()

    def to_prometheus(self) -> str:
        """Export in Prometheus exposition format.

        Emits HELP/TYPE once per base metric, then all series for that base.
        """
        with self._lock:
            lines: list[str] = []
            # Group by base name (metric without labels)
            by_base: dict[str, list[tuple[str, int]]] = {}
            for key in sorted(self._counters.keys()):
                val = self._counters[key]
                base = key.split("{")[0] if "{" in key else key
                if base not in by_base:
                    by_base[base] = []
                by_base[base].append((key, val))

            for base in sorted(by_base.keys()):
                series = by_base[base]
                lines.append(f"# HELP aurora_lens_{base} Aurora Lens metric")
                lines.append(f"# TYPE aurora_lens_{base} counter")
                for key, val in series:
                    lines.append(f"aurora_lens_{key} {val}")

            uptime = int(time.monotonic() - self._start_time)
            lines.append("# HELP aurora_lens_uptime_seconds Proxy uptime in seconds")
            lines.append("# TYPE aurora_lens_uptime_seconds gauge")
            lines.append(f"aurora_lens_uptime_seconds {uptime}")
            return "\n".join(lines) + "\n"


# Module-level singleton
_metrics: Metrics | None = None
_metrics_lock = threading.Lock()


def get_metrics() -> Metrics:
    """Get or create the global metrics instance."""
    global _metrics
    with _metrics_lock:
        if _metrics is None:
            _metrics = Metrics()
        return _metrics

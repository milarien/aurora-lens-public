"""Per-trace log buffering for ledger anchoring.

Deterministic in-memory buffer of log records per request. At governance decision
time, the buffer is consumed, SHA-256 digest computed, and fingerprint fields
(log_digest, log_entry_count, first_timestamp, last_timestamp) added to the
ledger entry. No log content is stored in the ledger — fingerprint only.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import time
from contextvars import ContextVar
from typing import Any

MAX_ENTRIES = 1000
MAX_BYTES = 256 * 1024  # 256 KiB per request
MAX_MSG_LEN = 1024

log_buffer_var: ContextVar[dict[str, Any] | None] = ContextVar(
    "aurora_log_buffer",
    default=None,
)


def init_log_buffer():
    """Initialize per-request log buffer. Returns token for reset in finally."""
    return log_buffer_var.set(
        {"entries": [], "total_bytes": 0, "dropped": 0, "truncated": False}
    )


def _serialize_record(record: logging.LogRecord) -> dict:
    """Stable dict representation for digest computation."""
    msg = record.getMessage()
    if len(msg) > MAX_MSG_LEN:
        msg = msg[:MAX_MSG_LEN] + "...[truncated]"
    return {
        "ts": record.created,
        "level": record.levelname,
        "logger": record.name,
        "msg": msg,
        "trace_id": getattr(record, "trace_id", None),
        "session_id": getattr(record, "session_id", None),
    }


def append_to_log_buffer(record: logging.LogRecord) -> None:
    """Append a log record to the current trace buffer. No-op if no buffer."""
    state = log_buffer_var.get(None)
    if state is None:
        return

    if state["truncated"]:
        state["dropped"] += 1
        return

    if len(state["entries"]) >= MAX_ENTRIES:
        state["truncated"] = True
        state["dropped"] += 1
        return

    entry = _serialize_record(record)
    line = json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    b = len(line.encode("utf-8")) + 1  # newline

    if state["total_bytes"] + b > MAX_BYTES:
        state["truncated"] = True
        state["dropped"] += 1
        return

    state["entries"].append(entry)
    state["total_bytes"] += b


def compute_log_slice_digest(
    entries: list[dict],
    truncated: bool = False,
    dropped: int = 0,
) -> str:
    """Compute SHA-256 digest of serialized log entries + truncation metadata.

    Deterministic. Digest binds the slice and whether it was incomplete.
    truncated/dropped must be bool/int — stable types for digest input.
    """
    # Same serialization as append_to_log_buffer for byte-accurate accounting
    lines = [
        json.dumps(e, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for e in entries
    ]
    blob = "\n".join(lines)
    # Stable types: bool and int only
    meta = json.dumps(
        {"dropped": int(dropped), "truncated": bool(truncated)},
        sort_keys=True,
        separators=(",", ":"),
    )
    payload = blob + "\n" + meta
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _iso_utc(ts: float) -> str:
    """UTC ISO-8601 with microseconds and +00:00 (Stage C: ``fromtimestamp(..., tz=UTC).isoformat()``)."""
    return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).isoformat()


def consume_log_slice() -> dict | None:
    """Drain the current trace buffer and return fingerprint.

    Returns dict with log_digest, log_entry_count, first_timestamp, last_timestamp,
    log_slice_truncated, log_slice_dropped_count; or None if buffer empty or not initialized.
    """
    state = log_buffer_var.get(None)
    if state is None or not state["entries"]:
        return None

    entries = list(state["entries"])
    truncated = state["truncated"]
    dropped = state["dropped"]

    # Clear state
    state["entries"].clear()
    state["total_bytes"] = 0
    state["dropped"] = 0
    state["truncated"] = False

    digest = compute_log_slice_digest(entries, truncated=truncated, dropped=dropped)
    first_ts = min(e["ts"] for e in entries)
    last_ts = max(e["ts"] for e in entries)

    result: dict[str, Any] = {
        "log_digest": digest,
        "log_entry_count": len(entries),
        "first_timestamp": _iso_utc(first_ts),
        "last_timestamp": _iso_utc(last_ts),
        "log_slice_truncated": truncated,
        "log_slice_dropped_count": dropped,
    }
    return result

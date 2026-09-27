# aurora_lens/proxy/logging.py
from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict

from aurora_lens.context import get_auth_label, get_lock_metadata, get_trace_id, get_session_id
from aurora_lens.log_slice import append_to_log_buffer


_LOGGING_CONFIGURED = False
_RECORD_FACTORY_CONFIGURED = False
_ORIGINAL_FACTORY = logging.getLogRecordFactory()


def _aurora_record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
    """Inject trace_id, session_id, auth_label, lock_metadata into every log record. Global, deterministic."""
    record = _ORIGINAL_FACTORY(*args, **kwargs)
    record.trace_id = get_trace_id()
    record.session_id = get_session_id()
    record.auth_label = get_auth_label()
    lock_meta = get_lock_metadata()
    record.lock_wait_ms = lock_meta.get("lock_wait_ms") if lock_meta else None
    record.lock_acquired = lock_meta.get("lock_acquired") if lock_meta else None
    try:
        append_to_log_buffer(record)
    except Exception:
        # Never let log buffering break logging or request handling.
        # Do not log here (would risk recursion).
        pass
    return record


def _configure_record_factory() -> None:
    """Install Aurora LogRecord factory. Called from setup_logging(). Idempotent."""
    global _RECORD_FACTORY_CONFIGURED
    if _RECORD_FACTORY_CONFIGURED:
        return
    logging.setLogRecordFactory(_aurora_record_factory)
    _RECORD_FACTORY_CONFIGURED = True


def _level_from_env(default: str = "INFO") -> int:
    raw = os.getenv("AURORA_LENS_LOG_LEVEL", default).strip().upper()
    return getattr(logging, raw, logging.INFO)


def _iso_utc(ts: float) -> str:
    # RFC3339-ish without importing datetime for speed + determinism
    # Example: 2026-02-11T10:33:12Z
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


class JsonFormatter(logging.Formatter):
    """JSON logs with timestamp, level, logger, message, trace_id, session_id always present."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": _iso_utc(record.created),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": getattr(record, "trace_id", None),
            "session_id": getattr(record, "session_id", None),
            "auth_label": getattr(record, "auth_label", None),
            "lock_wait_ms": getattr(record, "lock_wait_ms", None),
            "lock_acquired": getattr(record, "lock_acquired", None),
        }

        # Merge extras (anything not part of the standard LogRecord fields)
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k
            not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "trace_id",
                "session_id",
                "auth_label",
                "lock_wait_ms",
                "lock_acquired",
            }
        }

        if extras:
            payload.update(_json_safe(extras))

        if record.exc_info:
            # Full traceback goes to logs, not to client responses.
            payload["exc"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


class DevFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = f"[{record.levelname:<5}] {record.name}: {record.getMessage()}"
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in {"name", "msg", "args", "levelname", "levelno", "exc_info", "exc_text", "stack_info"}
        }
        # Keep dev logs readable: show a small set of common fields if present
        common_keys = [
            "trace_id", "session_id", "auth_label", "lock_wait_ms", "lock_acquired",
            "aurora_trace_id", "aurora_session_id",
            "model", "latency_ms", "action", "policy", "audit_path", "path", "method",
            "provider", "upstream_type", "adapter_type", "extraction_backend", "stream", "api_key_set",
            "adapter", "reason", "input_tokens", "output_tokens", "msg_count",
            # governance_outcome fields
            "outcome", "flags", "flags_count", "blocked", "revised",
            "audit_id", "pathway_id", "commitment_closed",
        ]
        kv = {k: extras[k] for k in common_keys if k in extras}
        if kv:
            base += " | " + " ".join(f"{k}={kv[k]!r}" for k in kv)

        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)

        return base


def _json_safe(obj: Any) -> Any:
    # Convert dataclasses and non-serializable objects to strings deterministically.
    if is_dataclass(obj):
        return _json_safe(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def setup_logging(level: int | None = None) -> None:
    """
    One-time global logging setup.
    - JSON logs when AURORA_LENS_LOG_FORMAT=json
    - Dev logs otherwise
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    level = level if level is not None else _level_from_env()

    _configure_record_factory()

    fmt = os.getenv("AURORA_LENS_LOG_FORMAT", "dev").strip().lower()
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(DevFormatter())

    root = logging.getLogger()
    root.setLevel(level)

    # Remove any pre-existing handlers to avoid duplicates (uvicorn can add its own).
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)

    # Quiet noisy libs unless explicitly raised
    logging.getLogger("httpx").setLevel(max(level, logging.WARNING))
    logging.getLogger("uvicorn").setLevel(max(level, logging.INFO))
    logging.getLogger("uvicorn.error").setLevel(max(level, logging.INFO))
    logging.getLogger("uvicorn.access").setLevel(max(level, logging.WARNING))

    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

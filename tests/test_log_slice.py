"""Tests for log-slice anchoring (Phase 4.5)."""

from __future__ import annotations

import hashlib
import logging

from aurora_lens.log_slice import (
    append_to_log_buffer,
    compute_log_slice_digest,
    consume_log_slice,
    init_log_buffer,
    log_buffer_var,
)


class TestLogSliceDigest:

    def test_recomputed_digest_matches_stored_digest(self):
        """CI test: digest from consume must match recompute from same entries."""
        token = init_log_buffer()
        try:
            for i in range(3):
                r = logging.LogRecord("test", 20, "", 0, f"msg{i}", (), None)
                r.created = 1000.0 + i
                r.trace_id = "trace"
                r.session_id = "sess"
                append_to_log_buffer(r)

            result = consume_log_slice()
            assert result is not None
            stored_digest = result["log_digest"]

            # Rebuild entries in same format as _serialize_record
            entries = [
                {
                    "ts": 1000.0 + i,
                    "level": "INFO",
                    "logger": "test",
                    "msg": f"msg{i}",
                    "trace_id": "trace",
                    "session_id": "sess",
                }
                for i in range(3)
            ]
            recomputed = compute_log_slice_digest(entries, truncated=False, dropped=0)
            assert recomputed == stored_digest
        finally:
            log_buffer_var.reset(token)

    def test_digest_deterministic(self):
        """Same entries produce same digest."""
        entries = [
            {"ts": 1.0, "level": "INFO", "logger": "x", "msg": "a", "trace_id": None, "session_id": None},
            {"ts": 2.0, "level": "WARN", "logger": "y", "msg": "b", "trace_id": "t", "session_id": "s"},
        ]
        d1 = compute_log_slice_digest(entries)
        d2 = compute_log_slice_digest(entries)
        assert d1 == d2

    def test_empty_entries_digest(self):
        """Empty entries produce deterministic digest."""
        d = compute_log_slice_digest([], truncated=False, dropped=0)
        expected = hashlib.sha256(
            '\n{"dropped":0,"truncated":false}'.encode("utf-8")
        ).hexdigest()
        assert d == expected
        assert len(d) == 64

    def test_consume_returns_fingerprint_fields(self):
        """consume_log_slice returns log_digest, log_entry_count, first_timestamp, last_timestamp, log_slice_truncated, log_slice_dropped_count."""
        token = init_log_buffer()
        try:
            r = logging.LogRecord("test", 20, "", 0, "hello", (), None)
            r.created = 1000.0
            r.trace_id = "t"
            r.session_id = "s"
            append_to_log_buffer(r)

            result = consume_log_slice()
            assert result is not None
            assert "log_digest" in result
            assert "log_entry_count" in result
            assert "first_timestamp" in result
            assert "last_timestamp" in result
            assert "log_slice_truncated" in result
            assert "log_slice_dropped_count" in result
            assert result["log_entry_count"] == 1
            assert result["log_slice_truncated"] is False
            assert result["log_slice_dropped_count"] == 0
            assert len(result["log_digest"]) == 64  # SHA-256 hex
        finally:
            log_buffer_var.reset(token)

    def test_consume_returns_none_when_no_buffer(self):
        """consume_log_slice returns None when buffer not initialized."""
        assert consume_log_slice() is None

    def test_consume_returns_none_when_buffer_empty(self):
        """consume_log_slice returns None when buffer is empty."""
        token = init_log_buffer()
        try:
            assert consume_log_slice() is None
        finally:
            log_buffer_var.reset(token)

    def test_consume_drains_buffer(self):
        """consume_log_slice drains buffer; second call returns None."""
        token = init_log_buffer()
        try:
            r = logging.LogRecord("test", 20, "", 0, "x", (), None)
            r.created = 1000.0
            append_to_log_buffer(r)

            first = consume_log_slice()
            assert first is not None

            second = consume_log_slice()
            assert second is None
        finally:
            log_buffer_var.reset(token)

    def test_truncation_flag_when_max_entries_exceeded(self):
        """When MAX_ENTRIES exceeded, log_slice_truncated is True and log_slice_dropped_count >= 1."""
        import aurora_lens.log_slice as log_slice_mod

        orig = log_slice_mod.MAX_ENTRIES
        log_slice_mod.MAX_ENTRIES = 3
        try:
            token = init_log_buffer()
            try:
                for i in range(5):
                    r = logging.LogRecord("test", 20, "", 0, f"msg{i}", (), None)
                    r.created = 1000.0 + i
                    append_to_log_buffer(r)

                result = consume_log_slice()
                assert result is not None
                assert result["log_slice_truncated"] is True
                assert result["log_slice_dropped_count"] >= 1
                assert result["log_entry_count"] == 3
            finally:
                log_buffer_var.reset(token)
        finally:
            log_slice_mod.MAX_ENTRIES = orig

    def test_byte_cap_truncation(self):
        """When MAX_BYTES exceeded, log_slice_truncated is True and log_slice_dropped_count >= 1."""
        import aurora_lens.log_slice as log_slice_mod

        orig = log_slice_mod.MAX_BYTES
        # Small entries ~70 bytes each; 100 allows 1 entry then truncation
        log_slice_mod.MAX_BYTES = 100
        try:
            token = init_log_buffer()
            try:
                for i in range(10):
                    r = logging.LogRecord("test", 20, "", 0, f"msg{i}", (), None)
                    r.created = 1000.0 + i
                    append_to_log_buffer(r)

                result = consume_log_slice()
                assert result is not None
                assert result["log_slice_truncated"] is True
                assert result["log_slice_dropped_count"] >= 1
                assert result["log_entry_count"] >= 1
            finally:
                log_buffer_var.reset(token)
        finally:
            log_slice_mod.MAX_BYTES = orig

    def test_max_msg_len_truncates_message_in_entry(self):
        """When message exceeds MAX_MSG_LEN, entry contains truncated msg with ...[truncated] suffix."""
        import aurora_lens.log_slice as log_slice_mod

        orig = log_slice_mod.MAX_MSG_LEN
        log_slice_mod.MAX_MSG_LEN = 20
        try:
            token = init_log_buffer()
            try:
                long_msg = "a" * 50
                r = logging.LogRecord("test", 20, "", 0, long_msg, (), None)
                r.created = 1000.0
                append_to_log_buffer(r)

                result = consume_log_slice()
                assert result is not None
                assert result["log_entry_count"] == 1
                # Entry was serialized with truncated msg; digest is over that
                assert result["log_digest"]
                # Verify digest is deterministic for truncated entry
                entries = [
                    {
                        "ts": 1000.0,
                        "level": "INFO",
                        "logger": "test",
                        "msg": "a" * 20 + "...[truncated]",
                        "trace_id": None,
                        "session_id": None,
                    }
                ]
                recomputed = compute_log_slice_digest(entries, truncated=False, dropped=0)
                assert recomputed == result["log_digest"]
            finally:
                log_buffer_var.reset(token)
        finally:
            log_slice_mod.MAX_MSG_LEN = orig

    def test_recomputed_digest_matches_stored_when_truncated(self):
        """CI test: when buffer truncated, stored digest matches recompute(entries, truncated=True, dropped=N)."""
        import aurora_lens.log_slice as log_slice_mod

        orig = log_slice_mod.MAX_ENTRIES
        log_slice_mod.MAX_ENTRIES = 2
        try:
            token = init_log_buffer()
            try:
                for i in range(4):
                    r = logging.LogRecord("test", 20, "", 0, f"msg{i}", (), None)
                    r.created = 1000.0 + i
                    r.trace_id = "trace"
                    r.session_id = "sess"
                    append_to_log_buffer(r)

                result = consume_log_slice()
                assert result is not None
                assert result["log_slice_truncated"] is True
                assert result["log_slice_dropped_count"] >= 1
                stored_digest = result["log_digest"]

                # Rebuild entries in same format (only first 2 were stored)
                entries = [
                    {
                        "ts": 1000.0 + i,
                        "level": "INFO",
                        "logger": "test",
                        "msg": f"msg{i}",
                        "trace_id": "trace",
                        "session_id": "sess",
                    }
                    for i in range(2)
                ]
                recomputed = compute_log_slice_digest(
                    entries,
                    truncated=True,
                    dropped=result["log_slice_dropped_count"],
                )
                assert recomputed == stored_digest
            finally:
                log_buffer_var.reset(token)
        finally:
            log_slice_mod.MAX_ENTRIES = orig

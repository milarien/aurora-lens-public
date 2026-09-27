"""Phase D: Audit I/O — HMAC signing, rotation, chain verification for plain JSONL.

Used by BuiltinBridge when ForensicLedger is unavailable.
Phase D2: Hash chain — cid = SHA256(canonical_bytes + ":" + prev_cid).
"""

from __future__ import annotations

import datetime
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any

from aurora_lens.govern.instrument_provenance import verify_attestation_fields

CHAIN_GENESIS = "genesis"


def _canonical_json(entry: dict[str, Any]) -> str:
    """Deterministic JSON for hashing. sort_keys, no whitespace."""
    return json.dumps(entry, separators=(",", ":"), ensure_ascii=False, sort_keys=True)


def _compute_hmac(entry: dict[str, Any], key: bytes) -> str:
    """Compute HMAC-SHA256 over canonical JSON (sort_keys=True). Returns hex."""
    payload = json.dumps(entry, separators=(",", ":"), ensure_ascii=False, sort_keys=True)
    sig = hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return sig


def _maybe_rotate(path: Path, max_mb: int) -> None:
    """Rotate log if size exceeds max_mb. audit.jsonl → .1, .1 → .2, etc.
    Keeps up to 10 rotated files. Never deletes — operator's responsibility."""
    if max_mb <= 0:
        return
    try:
        size_mb = path.stat().st_size / (1024 * 1024)
    except OSError:
        return
    if size_mb < max_mb:
        return
    max_rotated = 10
    # Rotate: .9 → .10 (delete .10), .8 → .9, ..., .1 → .2, path → .1
    for i in range(max_rotated, 0, -1):
        old = path if i == 1 else Path(f"{path}.{i - 1}")
        new = Path(f"{path}.{i}")
        if old.exists():
            try:
                if new.exists():
                    new.unlink()
                old.rename(new)
            except OSError:
                pass
    # path is now free; next write creates it fresh


def append_audit_entry(
    path: str | Path,
    entry: dict[str, Any],
    *,
    signing_key: bytes | None = None,
    max_mb: int = 0,
    prev_cid: str | None = None,
) -> str | None:
    """Append one audit entry. Optionally sign with HMAC, rotate if over max_mb.
    When prev_cid is provided (D2 chain): compute cid, add prev_cid and cid to entry.
    Returns the new cid when prev_cid was provided, else None."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if max_mb > 0 and p.exists():
        _maybe_rotate(p, max_mb)
    entry = dict(entry)
    if prev_cid is not None:
        entry_bytes = _canonical_json(entry)
        cid = hashlib.sha256((entry_bytes + ":" + prev_cid).encode()).hexdigest()
        entry["prev_cid"] = prev_cid
        entry["cid"] = cid
    if signing_key:
        entry["hmac"] = _compute_hmac(entry, signing_key)
    line = _canonical_json(entry) + "\n"
    with open(p, "a", encoding="utf-8") as f:
        f.write(line)
    return entry.get("cid") if prev_cid is not None else None


def append_checkpoint_entry(
    path: str | Path,
    prev_cid: str,
    *,
    signing_key: bytes | None = None,
    max_mb: int = 0,
) -> str | None:
    """D3 Option 2: Append a checkpoint entry attesting to chain tip at current time.
    checkpoint_sig = HMAC({"checkpoint_cid": prev_cid, "checkpoint_time": ...}, key).
    Returns the new cid."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if max_mb > 0 and p.exists():
        _maybe_rotate(p, max_mb)
    checkpoint_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
    entry: dict[str, Any] = {
        "type": "checkpoint",
        "checkpoint_cid": prev_cid,
        "checkpoint_time": checkpoint_time,
    }
    entry_bytes = _canonical_json(entry)
    cid = hashlib.sha256((entry_bytes + ":" + prev_cid).encode()).hexdigest()
    entry["prev_cid"] = prev_cid
    entry["cid"] = cid
    if signing_key:
        sig_payload = json.dumps(
            {"checkpoint_cid": prev_cid, "checkpoint_time": checkpoint_time},
            separators=(",", ":"),
            sort_keys=True,
        )
        entry["checkpoint_sig"] = hmac.new(
            signing_key, sig_payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()
    line = _canonical_json(entry) + "\n"
    with open(p, "a", encoding="utf-8") as f:
        f.write(line)
    return cid


def verify_audit_entries(
    path: str | Path,
    n: int,
    signing_key: bytes | None = None,
    *,
    signing_keys: list[bytes] | None = None,
) -> tuple[bool, int, str | None]:
    """Verify last n entries. Returns (verified, entries_checked, first_failed_cid).
    first_failed_cid is the stored cid of the first entry that fails verification
    (usable directly with /v1/audit/entry?cid=...).  Falls back to trace_id if
    cid is absent.

    D4 multi-key: pass signing_keys (list) to verify logs spanning key rotation.
    Tries each key per entry until one succeeds. signing_key (single) still supported."""
    keys = list(signing_keys) if signing_keys else ([signing_key] if signing_key else [])
    p = Path(path)
    if not p.exists():
        return True, 0, None
    try:
        lines = p.read_text(encoding="utf-8").strip().splitlines()
    except OSError:
        return False, 0, None
    lines = [ln for ln in lines if ln.strip()][-n:]
    for i, ln in enumerate(lines):
        try:
            entry = json.loads(ln)
        except json.JSONDecodeError:
            return False, i, None
        if entry.get("type") == "checkpoint":
            if "checkpoint_sig" not in entry:
                return False, i, None
            sig_payload = json.dumps(
                {"checkpoint_cid": entry["checkpoint_cid"], "checkpoint_time": entry["checkpoint_time"]},
                separators=(",", ":"),
                sort_keys=True,
            )
            expected = entry.pop("checkpoint_sig")
            matched = False
            for key in keys:
                if not key:
                    continue
                actual = hmac.new(key, sig_payload.encode("utf-8"), hashlib.sha256).hexdigest()
                if hmac.compare_digest(expected, actual):
                    matched = True
                    break
            entry["checkpoint_sig"] = expected
            if not matched:
                return False, i + 1, None
            continue
        if "hmac" not in entry:
            return False, i, entry.get("cid") or entry.get("trace_id")
        expected = entry.pop("hmac")
        payload = _canonical_json(entry)
        matched = False
        for key in keys:
            if not key:
                continue
            actual = hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()
            if hmac.compare_digest(expected, actual):
                matched = True
                break
        if not matched:
            entry["hmac"] = expected
            return False, i + 1, entry.get("cid") or entry.get("trace_id")
        entry["hmac"] = expected  # restore for caller if needed
    return True, len(lines), None


def verify_chain(
    path: str | Path,
    n: int,
    signing_key: bytes | None = None,
    *,
    signing_keys: list[bytes] | None = None,
) -> tuple[bool, int, int | None, str | None]:
    """Verify hash chain on last n entries (D2/D3).
    Returns (ok, entries_checked, first_break_index, reason).
    first_break_index is 0-based; reason is 'chain_break', 'hmac_failed', or
    'unanchored_slice' (internally consistent but not anchored to genesis).

    D4 multi-key: pass signing_keys (list) to verify logs spanning key rotation.
    Tries each key per entry until one succeeds. signing_key (single) still supported."""
    keys = list(signing_keys) if signing_keys else ([signing_key] if signing_key else [])
    p = Path(path)
    if not p.exists():
        return True, 0, None, None
    try:
        lines = p.read_text(encoding="utf-8").strip().splitlines()
    except OSError:
        return False, 0, 0, "read_error"
    lines = [ln for ln in lines if ln.strip()][-n:]
    prev_cid = CHAIN_GENESIS
    _unanchored = False   # True when first chained entry links outside this window
    _seen_chained = False  # True once we've accepted the first chained entry
    for i, ln in enumerate(lines):
        try:
            entry = json.loads(ln)
        except json.JSONDecodeError:
            return False, i, i, "json_error"
        if "cid" not in entry or "prev_cid" not in entry:
            # Lines without D2 cid/prev_cid (legacy markers, non-chain payloads) are
            # transparent to chain verification — skip without resetting prev_cid.
            # AFL ledger governance rows use op names like CONTAIN/HARD_STOP, not a
            # separate forensic_envelope op; forensic_event lives under payload.data.
            continue
        if entry.get("prev_cid") != prev_cid:
            if not _seen_chained and prev_cid == CHAIN_GENESIS:
                # First chained entry in the window has a non-genesis prev_cid.
                # This is a tail-slice: the window starts mid-chain.  Accept this
                # entry's claimed cid as the local anchor and verify internal
                # continuity for all subsequent chained entries.
                _unanchored = True
                prev_cid = entry["cid"]
                _seen_chained = True
                continue
            return False, len(lines), i, "chain_break"
        # cid is computed before hmac/checkpoint_sig; exclude cid, prev_cid, hmac, checkpoint_sig
        exclude = ("cid", "prev_cid", "hmac", "checkpoint_sig")
        entry_for_cid = {k: v for k, v in entry.items() if k not in exclude}
        entry_bytes = _canonical_json(entry_for_cid)
        computed = hashlib.sha256((entry_bytes + ":" + prev_cid).encode()).hexdigest()
        if not hmac.compare_digest(computed, entry["cid"]):
            return False, len(lines), i, "chain_break"
        if entry.get("type") == "checkpoint" and keys and "checkpoint_sig" in entry:
            expected = entry.pop("checkpoint_sig")
            sig_payload = json.dumps(
                {"checkpoint_cid": entry["checkpoint_cid"], "checkpoint_time": entry["checkpoint_time"]},
                separators=(",", ":"),
                sort_keys=True,
            )
            matched = False
            for key in keys:
                if not key:
                    continue
                actual = hmac.new(key, sig_payload.encode("utf-8"), hashlib.sha256).hexdigest()
                if hmac.compare_digest(expected, actual):
                    matched = True
                    break
            entry["checkpoint_sig"] = expected
            if not matched:
                return False, len(lines), i, "hmac_failed"
            prev_cid = entry["cid"]
            _seen_chained = True
            continue
        if keys and "hmac" in entry:
            expected = entry.pop("hmac")
            payload = _canonical_json(entry)
            matched = False
            for key in keys:
                if not key:
                    continue
                actual = hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()
                if hmac.compare_digest(expected, actual):
                    matched = True
                    break
            entry["hmac"] = expected
            if not matched:
                return False, len(lines), i, "hmac_failed"
        prev_cid = entry["cid"]
        _seen_chained = True
    if _unanchored:
        # Slice is internally consistent but not anchored to the chain head.
        return False, len(lines), None, "unanchored_slice"
    return True, len(lines), None, None


def _recompute_state_hash(pef_snapshot: dict[str, Any]) -> str:
    """Recompute state_hash from pef_snapshot (matches build_forensic_event)."""
    return hashlib.sha256(
        json.dumps(pef_snapshot, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def normalize_audit_line_for_verify(entry: dict[str, Any]) -> dict[str, Any]:
    """Map one JSONL object to flat governance row shape for replay/verify helpers.

    AFL-JSONL-1 lines (``kind`` == ``aurora.event``) store governance fields under
    ``payload[\"data\"]``; flat JSONL rows store them at the top level.
    """
    if entry.get("kind") == "aurora.event":
        data = (entry.get("payload") or {}).get("data")
        if not isinstance(data, dict):
            data = {}
        return dict(data)
    return entry


def _audit_row_session_id_for_linkage(row: dict[str, Any]) -> str | None:
    """Chat session id for linkage pairing (flat JSONL top-level or ledger ``payload.data``)."""
    sid = row.get("session_id")
    if isinstance(sid, str) and sid.strip():
        return sid.strip()
    fe = row.get("forensic_event")
    if isinstance(fe, dict):
        fs = fe.get("session_id")
        if isinstance(fs, str) and fs.strip():
            return fs.strip()
    return None


def _audit_row_proxy_run_id_for_linkage(row: dict[str, Any]) -> str | None:
    """Proxy process id when present (same field as ``GET /health`` ``proxy_run_id``)."""
    rid = row.get("proxy_run_id")
    if isinstance(rid, str) and rid.strip():
        return rid.strip()
    return None


def verify_forensic_state_hash_detailed(
    path: str | Path,
    n: int = 1000,
) -> dict[str, Any]:
    """Replay-verify ``state_hash`` from ``pef_snapshot`` with skip counts (flat or AFL).

    Returns a dict with ``ok``, ``entries_checked``, skip counters, and optional failure line/reason.
    """
    p = Path(path)
    empty = {
        "ok": True,
        "entries_checked": 0,
        "skipped_checkpoint_lines": 0,
        "skipped_no_forensic_event": 0,
        "skipped_no_state_hash": 0,
        "first_failure_line_index": None,
        "reason": None,
    }
    if not p.exists():
        return dict(empty)
    try:
        lines = p.read_text(encoding="utf-8").strip().splitlines()
    except OSError:
        return {
            **empty,
            "ok": False,
            "first_failure_line_index": 0,
            "reason": "read_error",
        }
    lines = [ln for ln in lines if ln.strip()][-n:]
    checked = 0
    skipped_checkpoint = 0
    skipped_no_fe = 0
    skipped_no_hash = 0
    for i, ln in enumerate(lines):
        try:
            entry = json.loads(ln)
        except json.JSONDecodeError:
            return {
                **empty,
                "ok": False,
                "entries_checked": checked,
                "skipped_checkpoint_lines": skipped_checkpoint,
                "skipped_no_forensic_event": skipped_no_fe,
                "skipped_no_state_hash": skipped_no_hash,
                "first_failure_line_index": i,
                "reason": "json_error",
            }
        if entry.get("type") == "checkpoint":
            skipped_checkpoint += 1
            continue
        row = normalize_audit_line_for_verify(entry)
        if row.get("type") == "checkpoint":
            continue
        fe = row.get("forensic_event")
        if fe is None:
            skipped_no_fe += 1
            continue
        expected_hash = fe.get("state_hash")
        if expected_hash is None:
            skipped_no_hash += 1
            continue
        pef_snapshot = row.get("pef_snapshot")
        if pef_snapshot is None:
            return {
                **empty,
                "ok": False,
                "entries_checked": checked,
                "skipped_checkpoint_lines": skipped_checkpoint,
                "skipped_no_forensic_event": skipped_no_fe,
                "skipped_no_state_hash": skipped_no_hash,
                "first_failure_line_index": i,
                "reason": "missing_pef_snapshot",
            }
        computed = _recompute_state_hash(pef_snapshot)
        if not hmac.compare_digest(computed, expected_hash):
            return {
                **empty,
                "ok": False,
                "entries_checked": checked,
                "skipped_checkpoint_lines": skipped_checkpoint,
                "skipped_no_forensic_event": skipped_no_fe,
                "skipped_no_state_hash": skipped_no_hash,
                "first_failure_line_index": i,
                "reason": "state_hash_mismatch",
            }
        checked += 1
    return {
        "ok": True,
        "entries_checked": checked,
        "skipped_checkpoint_lines": skipped_checkpoint,
        "skipped_no_forensic_event": skipped_no_fe,
        "skipped_no_state_hash": skipped_no_hash,
        "first_failure_line_index": None,
        "reason": None,
    }


def verify_forensic_state_hash(
    path: str | Path,
    n: int = 1000,
) -> tuple[bool, int, int | None, str | None]:
    """Replay-verify state_hash: recompute from pef_snapshot, assert matches forensic_event.

    For each audit entry with forensic_event and pef_snapshot, recomputes state_hash
    from the stored pef_snapshot and asserts it matches forensic_event[\"state_hash\"].

    Returns (ok, entries_checked, first_failed_index, reason).
    On failure, *entries_checked* matches historical behavior: successful replays before
    the failure plus one (the failing row position in the check sequence).
    first_failed_index is 0-based line index; reason is 'state_hash_mismatch' or
    'missing_pef_snapshot'. Skips entries without forensic_event, checkpoint entries,
    and entries where forensic_event.state_hash is None (e.g. extraction failed before PEF).
    """
    d = verify_forensic_state_hash_detailed(path, n)
    if d["ok"]:
        return True, int(d["entries_checked"]), None, None
    # Legacy tuple: second value was checked+1 on failure (count of rows through failure).
    return (
        False,
        int(d["entries_checked"]) + 1,
        d["first_failure_line_index"],
        d["reason"],
    )


def verify_chain_of_custody_rows(
    path: str | Path,
    n: int = 1000,
    *,
    require_complete_evidence: bool = False,
    require_all_rows: bool = False,
) -> tuple[bool, int, int | None, str | None]:
    """Verify ``chain_of_custody`` blocks on the last ``n`` JSONL rows.

    Checks structural presence, fingerprint replay against embedded governance snapshot,
    and consistency of ``evidence_audit_status`` vs ``evidence_degradation_reasons``.

    Rows without ``chain_of_custody`` are skipped (legacy logs) unless *require_all_rows*
    is True, in which case non-checkpoint rows without a block fail.

    Returns ``(ok, rows_checked, first_failed_line_index, reason)``.
    """
    from aurora_lens.govern.chain_of_custody import verify_chain_of_custody_entry

    p = Path(path)
    if not p.exists():
        return True, 0, None, None
    try:
        lines = p.read_text(encoding="utf-8").strip().splitlines()
    except OSError:
        return False, 0, 0, "read_error"
    lines = [ln for ln in lines if ln.strip()][-n:]
    checked = 0
    for i, ln in enumerate(lines):
        try:
            entry = json.loads(ln)
        except json.JSONDecodeError:
            return False, checked, i, "json_error"
        if entry.get("type") == "checkpoint":
            continue
        if "chain_of_custody" not in entry:
            if require_all_rows:
                return False, checked, i, "missing_chain_of_custody"
            continue
        ok, reason = verify_chain_of_custody_entry(
            entry,
            require_complete_evidence=require_complete_evidence,
        )
        if not ok:
            return False, checked + 1, i, reason
        checked += 1
    return True, checked, None, None


def verify_pef_linkage_detailed(
    path: str | Path,
    n: int = 1000,
) -> dict[str, Any]:
    """Replay-verify PEF audit linkage with skip counts (flat JSONL or AFL envelope).

    Consecutive lines in the audit file are not always the same **chat** session: the
    proxy appends all sessions to one path (``run_aurora_lens.py`` → ``python -m
    aurora_lens.proxy``) and concurrent requests interleave rows. When both rows carry
    a resolvable ``session_id`` and they differ, the pair is skipped (not a PEF chain edge).

    When both rows carry ``proxy_run_id`` (proxy process id) and they differ, the pair is
    skipped — adjacent lines after a redeploy/restart must not be verified as one PEF edge.

    Returns a dict with ``ok``, ``pairs_checked``, skip counters, and optional failure line/reason.
    """
    from aurora_lens.pef.audit_linkage import classify_pef_turn_start
    from aurora_lens.pef.state import PEFState

    p = Path(path)
    empty = {
        "ok": True,
        "pairs_checked": 0,
        "skipped_checkpoint_pairs": 0,
        "skipped_no_classification": 0,
        "skipped_no_prev_snapshot": 0,
        "skipped_cross_session_pairs": 0,
        "skipped_cross_proxy_run_pairs": 0,
        "first_failure_line_index": None,
        "reason": None,
    }
    if not p.exists():
        return dict(empty)
    try:
        lines = p.read_text(encoding="utf-8").strip().splitlines()
    except OSError:
        return {
            **empty,
            "ok": False,
            "first_failure_line_index": 0,
            "reason": "read_error",
        }
    lines = [ln for ln in lines if ln.strip()][-n:]
    checked = 0
    skipped_cp = 0
    skipped_no_class = 0
    skipped_no_snap = 0
    skipped_xsess = 0
    skipped_xrun = 0
    for i in range(1, len(lines)):
        try:
            prev_raw = json.loads(lines[i - 1])
            curr_raw = json.loads(lines[i])
        except json.JSONDecodeError:
            return {
                **empty,
                "ok": False,
                "pairs_checked": checked,
                "skipped_checkpoint_pairs": skipped_cp,
                "skipped_no_classification": skipped_no_class,
                "skipped_no_prev_snapshot": skipped_no_snap,
                "skipped_cross_session_pairs": skipped_xsess,
                "skipped_cross_proxy_run_pairs": skipped_xrun,
                "first_failure_line_index": i,
                "reason": "json_error",
            }
        if curr_raw.get("type") == "checkpoint" or prev_raw.get("type") == "checkpoint":
            skipped_cp += 1
            continue
        prev = normalize_audit_line_for_verify(prev_raw)
        curr = normalize_audit_line_for_verify(curr_raw)
        if "pef_turn_classification" not in curr:
            skipped_no_class += 1
            continue
        snap = prev.get("pef_snapshot")
        if snap is None or not isinstance(snap, dict):
            skipped_no_snap += 1
            continue
        prev_pr = _audit_row_proxy_run_id_for_linkage(prev)
        curr_pr = _audit_row_proxy_run_id_for_linkage(curr)
        if prev_pr is not None and curr_pr is not None and prev_pr != curr_pr:
            skipped_xrun += 1
            continue
        prev_sid = _audit_row_session_id_for_linkage(prev)
        curr_sid = _audit_row_session_id_for_linkage(curr)
        if prev_sid is not None and curr_sid is not None and prev_sid != curr_sid:
            skipped_xsess += 1
            continue
        try:
            expected = classify_pef_turn_start(PEFState.from_dict(snap)).value
        except Exception:
            return {
                **empty,
                "ok": False,
                "pairs_checked": checked,
                "skipped_checkpoint_pairs": skipped_cp,
                "skipped_no_classification": skipped_no_class,
                "skipped_no_prev_snapshot": skipped_no_snap,
                "skipped_cross_session_pairs": skipped_xsess,
                "skipped_cross_proxy_run_pairs": skipped_xrun,
                "first_failure_line_index": i,
                "reason": "pef_snapshot_invalid",
            }
        actual = curr.get("pef_turn_classification")
        if actual != expected:
            return {
                **empty,
                "ok": False,
                "pairs_checked": checked,
                "skipped_checkpoint_pairs": skipped_cp,
                "skipped_no_classification": skipped_no_class,
                "skipped_no_prev_snapshot": skipped_no_snap,
                "skipped_cross_session_pairs": skipped_xsess,
                "skipped_cross_proxy_run_pairs": skipped_xrun,
                "first_failure_line_index": i,
                "reason": "turn_classification_mismatch",
            }
        checked += 1
    return {
        "ok": True,
        "pairs_checked": checked,
        "skipped_checkpoint_pairs": skipped_cp,
        "skipped_no_classification": skipped_no_class,
        "skipped_no_prev_snapshot": skipped_no_snap,
        "skipped_cross_session_pairs": skipped_xsess,
        "skipped_cross_proxy_run_pairs": skipped_xrun,
        "first_failure_line_index": None,
        "reason": None,
    }


def verify_pef_linkage(
    path: str | Path,
    n: int = 1000,
) -> tuple[bool, int, int | None, str | None]:
    """Replay-verify PEF audit linkage between consecutive rows.

    When a row records ``pef_turn_classification`` (new schema), the **next** row's
    classification must match :func:`classify_pef_turn_start` applied to the **previous**
    row's ``pef_snapshot`` (PEF state after that decision). This ties turn-start posture
    to the prior row's durable state without inferring from free text.

    Rows without ``pef_turn_classification`` are skipped (legacy logs). Pairs where the
    previous row has no ``pef_snapshot`` are skipped (e.g. PASS-only rows without
    forensic payload).     Pairs where both rows carry a **different** chat ``session_id``
    are skipped (multi-session files; see :func:`verify_pef_linkage_detailed`).
    Pairs where both rows carry a **different** ``proxy_run_id`` are skipped when logs from
    multiple proxy processes are read as one stream.

    Returns ``(ok, pairs_checked, first_failed_line_index, reason)``.
    On failure, *pairs_checked* matches historical behavior: successful pairs before the
    failure plus one. ``first_failed_line_index`` is 0-based within the last-``n`` line
    window (the **current** row that failed).
    """
    d = verify_pef_linkage_detailed(path, n)
    if d["ok"]:
        return True, int(d["pairs_checked"]), None, None
    return (
        False,
        int(d["pairs_checked"]) + 1,
        d["first_failure_line_index"],
        d["reason"],
    )


def verify_chain_of_custody_window(
    path: str | Path,
    n: int = 1000,
    *,
    require_complete_evidence: bool = False,
    require_all_rows: bool = False,
) -> tuple[bool, dict[str, Any]]:
    """Verify ``chain_of_custody`` on the last ``n`` lines (flat JSONL or AFL envelope).

    Normalizes each line with :func:`normalize_audit_line_for_verify` so nested
    ``payload.data.chain_of_custody`` is checked the same way as top-level blocks.

    Returns ``(ok, detail)`` where *detail* includes counts of checked rows, rows without
    a block (legacy), ``complete`` / ``degraded`` tallies, and optional failure index/reason.
    """
    from aurora_lens.govern.chain_of_custody import verify_chain_of_custody_entry

    p = Path(path)
    detail: dict[str, Any] = {
        "rows_checked": 0,
        "rows_skipped_no_block": 0,
        "rows_skipped_checkpoint": 0,
        "complete_count": 0,
        "degraded_count": 0,
        "first_failed_line_index": None,
        "reason": None,
    }
    if not p.exists():
        return True, detail
    try:
        lines = p.read_text(encoding="utf-8").strip().splitlines()
    except OSError:
        return False, {
            **detail,
            "first_failed_line_index": 0,
            "reason": "read_error",
        }
    lines = [ln for ln in lines if ln.strip()][-n:]
    checked = 0
    for i, ln in enumerate(lines):
        try:
            entry = json.loads(ln)
        except json.JSONDecodeError:
            return False, {
                **detail,
                "first_failed_line_index": i,
                "reason": "json_error",
            }
        if entry.get("type") == "checkpoint":
            detail["rows_skipped_checkpoint"] += 1
            continue
        row = normalize_audit_line_for_verify(entry)
        if row.get("type") == "checkpoint":
            continue
        coc = row.get("chain_of_custody")
        if not isinstance(coc, dict):
            if require_all_rows:
                return False, {
                    **detail,
                    "first_failed_line_index": i,
                    "reason": "missing_chain_of_custody",
                }
            detail["rows_skipped_no_block"] += 1
            continue
        st = coc.get("evidence_audit_status")
        if st == "complete":
            detail["complete_count"] += 1
        elif st == "degraded":
            detail["degraded_count"] += 1
        ok, reason = verify_chain_of_custody_entry(
            {"chain_of_custody": coc},
            require_complete_evidence=require_complete_evidence,
        )
        if not ok:
            return False, {
                **detail,
                "rows_checked": checked,
                "first_failed_line_index": i,
                "reason": reason,
            }
        checked += 1
    detail["rows_checked"] = checked
    return True, detail


def verify_co_attestation_window(
    path: str | Path,
    n: int = 1000,
    *,
    signing_keys: list[bytes] | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Verify decision/instrument co-attestation fields for the last ``n`` lines.

    Legacy rows without co-attestation fields are accepted and counted as ``legacy_rows``.
    """
    p = Path(path)
    detail: dict[str, Any] = {
        "rows_checked": 0,
        "legacy_rows": 0,
        "rows_with_attestation": 0,
        "co_attested_rows": 0,
        "decision_only_rows": 0,
        "none_rows": 0,
        "first_failed_line_index": None,
        "reason": None,
    }
    if not p.exists():
        return True, detail
    try:
        lines = p.read_text(encoding="utf-8").strip().splitlines()
    except OSError:
        return False, {
            **detail,
            "first_failed_line_index": 0,
            "reason": "read_error",
        }
    lines = [ln for ln in lines if ln.strip()][-n:]
    checked = 0
    for i, ln in enumerate(lines):
        try:
            entry = json.loads(ln)
        except json.JSONDecodeError:
            return False, {
                **detail,
                "first_failed_line_index": i,
                "reason": "json_error",
            }
        if entry.get("type") == "checkpoint":
            continue
        row = normalize_audit_line_for_verify(entry)
        mode = str(row.get("attestation_mode") or "").strip().lower()
        ok, reason = verify_attestation_fields(
            row,
            signing_keys=signing_keys or [],
        )
        if reason == "legacy":
            detail["legacy_rows"] += 1
            continue
        detail["rows_with_attestation"] += 1
        if mode == "co_attested":
            detail["co_attested_rows"] += 1
        elif mode == "decision_only":
            detail["decision_only_rows"] += 1
        elif mode in {"", "none"}:
            detail["none_rows"] += 1
        if not ok:
            return False, {
                **detail,
                "rows_checked": checked,
                "first_failed_line_index": i,
                "reason": reason,
            }
        checked += 1
    detail["rows_checked"] = checked
    return True, detail

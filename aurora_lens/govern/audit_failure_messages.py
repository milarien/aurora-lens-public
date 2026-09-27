"""Plain-language operator messages for audit / forensic verification failures.

Used by :mod:`aurora_lens.scripts.verify_audit`, HTTP ``/v1/audit/verify``, and the
forensics dashboard so failures read as explanations, not only internal codes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aurora_lens.govern.forensic_ledger import LedgerVerifyDetail


def load_json_entry_at_window_index(
    path: Path, n: int, window_line_index: int | None
) -> tuple[dict[str, Any] | None, int | None]:
    """Return (parsed JSON entry, 1-based physical line number in *path*) for the window row."""
    if window_line_index is None:
        return None, None
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None, None
    nonblank: list[tuple[int, str]] = [(i, ln) for i, ln in enumerate(raw_lines) if ln.strip()]
    if not nonblank:
        return None, None
    window = nonblank[-n:] if len(nonblank) >= n else nonblank
    if window_line_index < 0 or window_line_index >= len(window):
        return None, None
    phys_0, text = window[window_line_index]
    try:
        return json.loads(text), phys_0 + 1
    except json.JSONDecodeError:
        return None, phys_0 + 1


def find_substring_in_audit_window(
    path: Path, n: int, token: str | None
) -> tuple[int | None, int | None]:
    """Search the last *n* non-empty lines for *token*; return (window_index, physical_1based_line)."""
    if not token:
        return None, None
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None, None
    nonblank: list[tuple[int, str]] = [(i, ln) for i, ln in enumerate(raw_lines) if ln.strip()]
    if not nonblank:
        return None, None
    window = nonblank[-n:] if len(nonblank) >= n else nonblank
    for wi, (_phys0, text) in enumerate(window):
        if token in text:
            return wi, _phys0 + 1
    return None, None


def explain_hmac_verification_failure(first_failed_id: str | None) -> str:
    """HMAC over flat JSONL rows (``verify_audit_entries``)."""
    if not first_failed_id:
        return (
            "HMAC verification failed: at least one row's signature does not match any of the "
            "signing keys you supplied. Typical causes: wrong key after rotation, a tampered or "
            "truncated line, or mixing rows written with different keys. Use the same key "
            "material as the process that wrote the log (see audit_signing_key / rotation docs)."
        )
    return (
        "HMAC verification failed: the first row that failed is identified by this id in the file: "
        f"{first_failed_id!r} (search the audit log for that substring to open the exact line). "
        "The line's HMAC does not match any configured key. Confirm keys, check for edits, and "
        "ensure legacy unsigned lines are not mixed with signed rows."
    )


def explain_jsonl_chain_failure(reason: str | None, first_break_idx: int | None) -> str:
    """Hash chain over flat JSONL (``verify_chain``)."""
    idx = ""
    if first_break_idx is not None:
        idx = f" Failing row window index (within the last n lines): {first_break_idx}."
    if reason == "chain_break":
        return (
            "The hash chain is broken: a row's `cid` or `prev_cid` does not match the previous row. "
            "Typical causes: a deleted or reordered line, manual editing, or merged files."
            + idx
        )
    if reason == "hmac_failed":
        return (
            "Chain verification reported an HMAC failure on a checkpoint-style row while checking "
            "the chain pass."
            + idx
        )
    if reason == "unanchored_slice":
        return (
            "The last `n` lines form a consistent chain among themselves, but the first line in this "
            "window does not link back to `genesis` inside the window. That usually means `n` is "
            "smaller than the full file. This is not proof of tampering by itself. "
            "Increase `n` or verify from the beginning of the file to prove the whole log."
        )
    if reason == "json_error":
        return "A line in the audit file is not valid JSON."
    if reason == "read_error":
        return "The audit file could not be read (permissions or disk error)."
    return f"Hash chain verification failed (internal code: {reason!r}).{idx}"


def explain_forensic_state_hash_failure(reason: str | None) -> str:
    """Replay of ``state_hash`` from ``pef_snapshot`` (``verify_forensic_state_hash``)."""
    if reason == "state_hash_mismatch":
        return (
            "Forensic state replay failed: recomputing the hash from `pef_snapshot` does not match "
            "`forensic_event.state_hash`. The row may have been altered after write, or the snapshot "
            "and envelope are inconsistent."
        )
    if reason == "missing_pef_snapshot":
        return (
            "Forensic state replay failed: `forensic_event` has a `state_hash` but the outer row has "
            "no `pef_snapshot`. Reviewers cannot replay the state without the snapshot object."
        )
    if reason == "json_error":
        return "A line in the audit file is not valid JSON."
    if reason == "read_error":
        return "The audit file could not be read."
    return f"Forensic state-hash replay failed (internal code: {reason!r})."


def explain_pef_linkage_failure(reason: str | None) -> str:
    """Consecutive-row PEF posture check (``verify_pef_linkage``)."""
    if reason == "turn_classification_mismatch":
        return (
            "PEF turn linkage failed: this row's `pef_turn_classification` does not match the posture "
            "derived from the previous row's `pef_snapshot`. Rows may be out of order or inconsistent."
        )
    if reason == "pef_snapshot_invalid":
        return (
            "PEF turn linkage failed: the previous row's `pef_snapshot` could not be loaded as a valid "
            "PEF state."
        )
    if reason == "json_error":
        return "A line in the audit file is not valid JSON."
    if reason == "read_error":
        return "The audit file could not be read."
    return f"PEF linkage verification failed (internal code: {reason!r})."


def explain_ledger_chain_failure(reason: str | None, line: int | None) -> str:
    """AFL ledger hash chain (``ForensicLedger.verify_detailed``)."""
    loc = f" (ledger file line {line})" if line is not None else ""
    if reason == "json_decode":
        return f"A line in the forensic ledger is not valid JSON.{loc}"
    if reason == "prev_mismatch":
        return (
            f"The `prev` hash on this line does not match the previous line's `hash` — the chain is "
            f"broken.{loc} Check for deleted lines, merges, or edits."
        )
    if reason == "hash_mismatch":
        return (
            f"The stored `hash` on this line does not match recomputing from the rest of the entry — "
            f"the row was tampered with or corrupted.{loc}"
        )
    if reason == "log_read_error":
        return "The ledger file could not be read (permissions or disk error)."
    return f"Ledger chain verification failed (internal code: {reason!r}).{loc}"


def explain_ledger_hmac_failure(reason: str | None, line: int | None) -> str:
    """AFL ledger per-line HMAC."""
    loc = f" (ledger file line {line})" if line is not None else ""
    if reason == "signed_without_keys":
        return (
            f"A signed ledger line was found but no signing keys were supplied to verify it.{loc} "
            "Configure audit_signing_key (and historical keys if needed)."
        )
    if reason == "hmac_mismatch":
        return (
            f"The HMAC on a signed line does not match any configured key.{loc} "
            "Confirm key rotation and that the verifier uses the same material as the writer."
        )
    if reason == "sig_shape":
        return (
            f"A `sig` field is present but not in the expected shape.{loc} "
            "The file may be from a different schema version."
        )
    return f"Ledger HMAC verification failed (internal code: {reason!r}).{loc}"


def build_jsonl_verify_operator_message(
    *,
    verified: bool,
    hmac_ok: bool,
    first_failed: str | None,
    chain_ok: bool,
    chain_reason: str | None,
    first_break_idx: int | None,
) -> str | None:
    """Single multi-paragraph summary for HTTP JSON when verification fails or is partial."""
    if verified:
        return None
    parts: list[str] = []
    if not hmac_ok:
        parts.append(explain_hmac_verification_failure(first_failed))
    if not chain_ok:
        parts.append(explain_jsonl_chain_failure(chain_reason, first_break_idx))
    return "\n\n".join(parts) if parts else None


def build_ledger_verify_operator_message(led: LedgerVerifyDetail) -> str | None:
    """Summary for AFL ledger verify response body."""
    if led.ok:
        return None
    parts: list[str] = []
    if led.first_chain_reason:
        parts.append(explain_ledger_chain_failure(led.first_chain_reason, led.first_chain_failure_line))
    if led.first_hmac_reason:
        parts.append(explain_ledger_hmac_failure(led.first_hmac_reason, led.first_hmac_failure_line))
    return "\n\n".join(parts) if parts else None


def _footer(internal_code: str | None) -> str:
    return f"Internal verifier code: {internal_code!r}"


def format_verify_audit_stderr_hmac(path: Path, n: int, first_failed: str | None) -> str:
    lines_out: list[str] = [
        "FAIL: HMAC verification (flat JSONL rows)",
        "",
        explain_hmac_verification_failure(first_failed),
        "",
    ]
    wid, phys = find_substring_in_audit_window(path, n, first_failed)
    if phys is not None:
        lines_out.append(
            f"Location: line {phys} in {path} (window index {wid}; last {n} non-empty lines)."
        )
    lines_out.extend(["", _footer("hmac_verification_failed")])
    return "\n".join(lines_out)


def format_verify_audit_stderr_chain(path: Path, n: int, first_break_idx: int | None, reason: str | None) -> str:
    lines_out: list[str] = [
        "FAIL: hash chain verification (flat JSONL)",
        "",
        explain_jsonl_chain_failure(reason, first_break_idx),
        "",
    ]
    if first_break_idx is not None:
        _ent, phys = load_json_entry_at_window_index(path, n, first_break_idx)
        if phys is not None:
            lines_out.append(
                f"Location: line {phys} in {path} (window index {first_break_idx}; last {n} non-empty lines)."
            )
    lines_out.extend(["", _footer(reason)])
    return "\n".join(lines_out)


def format_verify_audit_stderr_replay(path: Path, n: int, fail_idx: int | None, reason: str | None) -> str:
    lines_out: list[str] = [
        "FAIL: forensic state_hash replay (pef_snapshot vs forensic_event)",
        "",
        explain_forensic_state_hash_failure(reason),
        "",
    ]
    if fail_idx is not None:
        _ent, phys = load_json_entry_at_window_index(path, n, fail_idx)
        if phys is not None:
            lines_out.append(
                f"Location: line {phys} in {path} (window index {fail_idx}; last {n} non-empty lines)."
            )
    lines_out.extend(["", _footer(reason)])
    return "\n".join(lines_out)


def format_verify_audit_stderr_linkage(path: Path, n: int, fail_idx: int | None, reason: str | None) -> str:
    lines_out: list[str] = [
        "FAIL: PEF turn linkage (consecutive rows)",
        "",
        explain_pef_linkage_failure(reason),
        "",
    ]
    if fail_idx is not None:
        _ent, phys = load_json_entry_at_window_index(path, n, fail_idx)
        if phys is not None:
            lines_out.append(
                f"Location: line {phys} in {path} (window index {fail_idx}; last {n} non-empty lines)."
            )
    lines_out.extend(["", _footer(reason)])
    return "\n".join(lines_out)


def format_chain_of_custody_failure_stderr(
    path: Path,
    n: int,
    window_line_index: int | None,
    reason: str | None,
) -> str:
    from aurora_lens.govern.chain_of_custody import explain_chain_of_custody_failure

    lines_out: list[str] = [
        "FAIL: chain-of-custody verification",
        "",
        explain_chain_of_custody_failure(reason),
        "",
    ]
    entry, phys = load_json_entry_at_window_index(path, n, window_line_index)
    if phys is not None:
        lines_out.append(
            f"Location: line {phys} in {path} (within the last {n} non-empty lines; "
            f"window index {window_line_index})."
        )
    if entry and isinstance(entry.get("chain_of_custody"), dict):
        coc = entry["chain_of_custody"]
        status = coc.get("evidence_audit_status")
        rlist = coc.get("evidence_degradation_reasons")
        lines_out.append(f"Recorded on the row: evidence_audit_status={status!r}")
        if isinstance(rlist, list) and rlist:
            lines_out.append(f"Recorded degradation reasons: {rlist}")
        rt = coc.get("runtime") if isinstance(coc.get("runtime"), dict) else {}
        if isinstance(rt, dict):
            lines_out.append(
                f"Recorded runtime attribution: model_id={rt.get('model_id')!r}, "
                f"provider={rt.get('provider')!r}"
            )
    lines_out.extend(["", _footer(reason)])
    return "\n".join(lines_out)

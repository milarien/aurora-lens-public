"""ForensicLedger — AFL-JSONL-1 hash-chained tamper-evident audit log.

Original implementation from aurora-stack (libs/audit/ledger.py).
No external dependencies — pure stdlib.

canonicalize_jcs() is RFC 8785 compliant for the AFL value domain
(which forbids floats). Python's json.dumps with sort_keys=True and
separators=(',',':') is JCS-compliant for this domain.
"""

from __future__ import annotations

import datetime
import hashlib
import hmac as _hmac
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


def _canonicalize_jcs(obj: Any) -> bytes:
    """Minimal JCS (RFC 8785) for the AFL value domain (no floats)."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class LedgerVerifyDetail:
    """Result of an independent chain pass and HMAC pass over the ledger file."""

    ok: bool
    entries_checked: int
    chain_ok: bool
    hmac_ok: bool | None
    """True when HMAC policy passes; False on failure; None when no line had a signature."""
    hmac_checked: bool
    """True when HMAC policy was evaluated (non-empty keys or at least one signed line)."""
    first_chain_failure_line: int | None = None
    first_chain_reason: str | None = None
    first_hmac_failure_line: int | None = None
    first_hmac_reason: str | None = None
    first_chain_detail: dict[str, Any] = field(default_factory=dict)
    first_hmac_detail: dict[str, Any] = field(default_factory=dict)


class ForensicLedger:
    """AFL-JSONL-1 forensic envelope log. One line = one event. Hash-chained."""

    def __init__(self, log_path: str | Path, trace_id: str | None = None,
                 secret_key: bytes | None = None):
        self.log_path = Path(log_path)
        self.trace_id = trace_id or f"trace:{self.log_path.name}"
        self.last_hash = "h:null"
        self.last_cid: str | None = None
        self.root_hash: str | None = None
        self.seq = 0
        self._secret_key = secret_key

        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            # Directory not writable (e.g. root-owned volume mount).
            # Raise so the caller can degrade gracefully rather than crash silently.
            raise

        if self.log_path.exists() and self.log_path.stat().st_size > 0:
            self._resume()

    def _resume(self) -> None:
        try:
            with open(self.log_path, encoding="utf-8") as f:
                lines = f.readlines()
            if not lines:
                return
            first = json.loads(lines[0])
            self.root_hash = first.get("root")
            last = json.loads(lines[-1])
            self.last_hash = last.get("hash", "h:null")
            self.last_cid = last.get("cid")
            self.seq = (last.get("seq") or 0) + 1
        except Exception:
            pass

    def _timestamp(self) -> str:
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    def append(
        self,
        op: str,
        payload_data: dict[str, Any],
        cid: str,
        scope: dict[str, str] | None = None,
        ext: dict[str, Any] | None = None,
        override_time: str | None = None,
    ) -> str:
        """Create and append one forensic envelope. Returns the new hash."""
        envelope: dict[str, Any] = {
            "v": 1,
            "kind": "aurora.event",
            "time": override_time or self._timestamp(),
            "seq": self.seq,
            "trace": self.trace_id,
            "prev": self.last_hash,
            "root": self.root_hash if self.root_hash else "PENDING",
            "cid": cid,
            "op": op,
            "scope": scope or {"domain": "governance", "subdomain": "aurora-lens", "authority": "policy"},
            "payload": {"pv": 1, "data": payload_data},
            "canon": "c14n:jcs",
            "ext": ext or {},
        }

        if self.seq == 0:
            envelope["root"] = "h:null"
            h = self._hash(envelope)
            envelope["root"] = h
            envelope["hash"] = h
            self.root_hash = h
        else:
            envelope["root"] = self.root_hash
            h = self._hash(envelope)
            envelope["hash"] = h

        # sig: HMAC-SHA256 over the entry's hash field when a key is available;
        # null otherwise. The hash chain is the primary tamper-detection mechanism;
        # sig provides an additional authentication layer when a secret_key is set.
        if self._secret_key is not None:
            sig_bytes = _hmac.new(
                self._secret_key,
                h.encode("utf-8"),
                "sha256",
            ).hexdigest()
            envelope["sig"] = {"alg": "hmac-sha256", "value": sig_bytes}
        else:
            envelope["sig"] = None

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(envelope, separators=(",", ":"), ensure_ascii=False) + "\n")

        self.last_hash = h
        self.last_cid = cid
        self.seq += 1
        return h

    def _hash(self, envelope: dict[str, Any]) -> str:
        target = {k: v for k, v in envelope.items() if k not in ("hash", "sig", "root")}
        digest = hashlib.sha256(_canonicalize_jcs(target)).hexdigest()
        return f"h:sha256:{digest}"

    def verify_detailed(self, *, signing_keys: Sequence[bytes] | None = None) -> LedgerVerifyDetail:
        """Verify hash chain and HMAC in two passes; report first failure per dimension.

        Chain pass: JSON validity, ``prev`` link, recomputed ``hash`` (JCS over entry
        minus ``hash``/``sig``/``root``). HMAC pass: fail-closed when any line is
        signed but no keys are supplied; otherwise each signed line must match at
        least one key. Passes are independent so a broken ``prev`` at line *k* does
        not hide an HMAC/key mismatch on an earlier line.
        """
        keys = list(signing_keys) if signing_keys else []
        empty_file = not self.log_path.exists()
        if empty_file:
            return LedgerVerifyDetail(
                ok=True,
                entries_checked=0,
                chain_ok=True,
                hmac_ok=None,
                hmac_checked=False,
            )
        try:
            lines = [ln for ln in self.log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        except OSError:
            return LedgerVerifyDetail(
                ok=False,
                entries_checked=0,
                chain_ok=False,
                hmac_ok=None,
                hmac_checked=False,
                first_chain_failure_line=None,
                first_chain_reason="log_read_error",
            )

        n = len(lines)
        prev_hash = "h:null"
        first_cl: int | None = None
        first_cr: str | None = None
        first_cd: dict[str, Any] = {}
        chain_ok = True

        for i, ln in enumerate(lines, start=1):
            try:
                entry = json.loads(ln)
            except json.JSONDecodeError:
                chain_ok = False
                first_cl, first_cr = i, "json_decode"
                break
            stored = entry.get("hash", "")
            got_prev = entry.get("prev")
            if got_prev != prev_hash and entry.get("seq", 0) != 0:
                chain_ok = False
                first_cl, first_cr = i, "prev_mismatch"
                first_cd = {
                    "expected_prev": prev_hash,
                    "got_prev": got_prev,
                    "cid": entry.get("cid"),
                    "seq": entry.get("seq"),
                }
                break
            target = {k: v for k, v in entry.items() if k not in ("hash", "sig", "root")}
            computed = f"h:sha256:{hashlib.sha256(_canonicalize_jcs(target)).hexdigest()}"
            if stored != computed:
                chain_ok = False
                first_cl, first_cr = i, "hash_mismatch"
                first_cd = {
                    "cid": entry.get("cid"),
                    "seq": entry.get("seq"),
                    "stored_hash": stored,
                    "computed_hash": computed,
                }
                break
            prev_hash = stored

        any_signed_line = False
        first_hl: int | None = None
        first_hr: str | None = None
        first_hd: dict[str, Any] = {}

        for i, ln in enumerate(lines, start=1):
            try:
                entry = json.loads(ln)
            except json.JSONDecodeError:
                continue
            sig = entry.get("sig")
            if isinstance(sig, dict) and sig.get("alg") == "hmac-sha256" and sig.get("value"):
                any_signed_line = True
                stored = entry.get("hash", "")
                expected_hex = sig["value"]
                if not keys:
                    first_hl, first_hr = i, "signed_without_keys"
                    first_hd = {"cid": entry.get("cid"), "seq": entry.get("seq")}
                    break
                h_bytes = stored.encode("utf-8")
                matched = False
                for key in keys:
                    if not key:
                        continue
                    actual = _hmac.new(key, h_bytes, "sha256").hexdigest()
                    if _hmac.compare_digest(actual, expected_hex):
                        matched = True
                        break
                if not matched:
                    first_hl, first_hr = i, "hmac_mismatch"
                    first_hd = {"cid": entry.get("cid"), "seq": entry.get("seq")}
                    break
            elif sig is not None and sig != {}:
                any_signed_line = True
                first_hl, first_hr = i, "sig_shape"
                first_hd = {"cid": entry.get("cid"), "seq": entry.get("seq"), "sig": sig}
                break

        if first_hl is not None:
            hmac_ok = False
            hmac_checked = True
        elif keys:
            hmac_checked = True
            hmac_ok = True
        else:
            hmac_checked = bool(any_signed_line)
            hmac_ok = None if not any_signed_line else False

        ok = bool(chain_ok and (hmac_ok is not False))
        return LedgerVerifyDetail(
            ok=ok,
            entries_checked=n,
            chain_ok=chain_ok,
            hmac_ok=hmac_ok,
            hmac_checked=hmac_checked,
            first_chain_failure_line=first_cl,
            first_chain_reason=first_cr,
            first_hmac_failure_line=first_hl,
            first_hmac_reason=first_hr,
            first_chain_detail=first_cd,
            first_hmac_detail=first_hd,
        )

    def verify(self, *, signing_keys: Sequence[bytes] | None = None) -> bool:
        """Verify hash chain integrity and optionally per-entry HMAC ``sig``.

        Hash chain is always checked. When ``signing_keys`` is empty or omitted,
        any entry with a non-null ``sig`` fails verification (fail-closed: signed
        lines cannot be authenticated without keys). When ``signing_keys`` is
        non-empty, each signed entry must match HMAC-SHA256(hash, key) for at
        least one key (multi-key rotation). Entries with ``sig: null`` skip
        HMAC (legacy logs).

        Equivalent to ``verify_detailed(...).ok``.
        """
        return self.verify_detailed(signing_keys=signing_keys).ok

    def _count_lines(self) -> int:
        """Non-empty lines in the log file (one AFL envelope per line)."""
        if not self.log_path.exists():
            return 0
        try:
            with open(self.log_path, encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        except OSError:
            return 0

    def get_stats(self) -> dict[str, Any]:
        n = self._count_lines()
        return {
            "seq": self.seq,
            "entries": n,
            "entry_count": n,
            "log_path": str(self.log_path),
            "trace_id": self.trace_id,
            "last_hash": self.last_hash,
        }

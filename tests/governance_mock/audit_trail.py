"""Minimal stub for governance.audit_trail — AFL-JSONL-1 format."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _canonical(obj: dict) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)


def _entry_hash(entry_without_hash: dict) -> str:
    return "h:" + hashlib.sha256(_canonical(entry_without_hash).encode()).hexdigest()


class ForensicLedger:
    """AFL-JSONL-1 ledger stub.

    Format per entry:
      {"v":1, "kind":"aurora.event", "op":..., "prev":..., "cid":...,
       "payload":{"data":...}, "ext":{"pef":<64-hex>}, "hash":...}

    hash = "h:" + SHA256(canonical_json_without_hash_field)
    prev = "h:null" for first entry, else previous entry's hash
    ext.pef = SHA256(canonical_json(payload_data)) — 64-char hex
    """

    def __init__(self, log_path: str, trace_id: str):
        self._path = Path(log_path)
        self._trace_id = trace_id
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._prev_hash = "h:null"
        self._entry_count = 0

    @property
    def ledger_stats(self) -> dict:
        return self.get_stats()

    def get_stats(self) -> dict:
        return {"entries": self._entry_count, "trace_id": self._trace_id}

    def append(self, op: str, payload_data: dict, cid: str, scope: dict | None = None) -> None:
        pef_hash = hashlib.sha256(_canonical(payload_data).encode()).hexdigest()
        entry: dict = {
            "v": 1,
            "kind": "aurora.event",
            "op": op,
            "cid": cid,
            "prev": self._prev_hash,
            "payload": {"data": payload_data},
            "ext": {"pef": pef_hash},
        }
        entry_hash = _entry_hash(entry)
        entry["hash"] = entry_hash

        with open(self._path, "a", encoding="utf-8") as f:
            f.write(_canonical(entry) + "\n")

        self._prev_hash = entry_hash
        self._entry_count += 1

    def verify(self) -> bool:
        """Re-read file and verify hash chain integrity."""
        if not self._path.exists():
            return True
        lines = [ln for ln in self._path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        prev_hash = "h:null"
        for line in lines:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                return False
            if entry.get("prev") != prev_hash:
                return False
            stored = entry.pop("hash", None)
            recomputed = _entry_hash(entry)
            entry["hash"] = stored
            if stored != recomputed:
                return False
            prev_hash = stored
        return True

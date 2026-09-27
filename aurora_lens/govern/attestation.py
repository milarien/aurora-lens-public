"""AttestedOutput — HMAC-signed governance decision record.

Local implementation using stdlib hmac + hashlib.
No external dependencies.
"""

from __future__ import annotations

import dataclasses
import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Any


@dataclass
class AttestedOutput:
    """HMAC-signed record of a governance decision.

    ``meta`` is included verbatim in :func:`sign_attested_output` / verification
    (canonical JSON under the ``meta`` key). Aurora bridges populate correlation
    fields: ``turn``, ``flags``, ``attempt``, ``timestamp`` (UTC ISO-8601),
    ``trace_id``, ``session_id`` (may be null), and ``run_id`` (process-stable UUID).
    """

    content: str
    policy_cid: str
    rationale_code: str
    decision: str
    event_cid: str
    signature: str
    meta: dict[str, Any]


def _signing_payload(attested: AttestedOutput) -> bytes:
    payload = {
        "content": attested.content,
        "policy_cid": attested.policy_cid,
        "rationale_code": attested.rationale_code,
        "decision": attested.decision,
        "event_cid": attested.event_cid,
        "meta": attested.meta,
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8")


def sign_attested_output(attested: AttestedOutput, key: bytes) -> AttestedOutput:
    """Sign an AttestedOutput with HMAC-SHA256 over :func:`_signing_payload` (includes ``meta``)."""
    sig = hmac.new(key, _signing_payload(attested), hashlib.sha256).hexdigest()
    return dataclasses.replace(attested, signature=sig)


def verify_governed_output(attested: AttestedOutput, key: bytes) -> bool:
    """Verify an AttestedOutput signature. Returns True if valid."""
    if not attested.signature:
        return False
    expected = hmac.new(key, _signing_payload(attested), hashlib.sha256).hexdigest()
    return hmac.compare_digest(attested.signature, expected)

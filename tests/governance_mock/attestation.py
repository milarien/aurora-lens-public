"""Minimal stub for governance.attestation."""
from __future__ import annotations

import hashlib
import hmac as _hmac
from dataclasses import dataclass, field


@dataclass
class AttestedOutput:
    content: str
    policy_cid: str
    rationale_code: str
    decision: str
    event_cid: str
    signature: str = ""


def _sign_payload(attested: AttestedOutput, secret_key: bytes) -> str:
    payload = f"{attested.decision}:{attested.policy_cid}:{attested.content}"
    return _hmac.new(secret_key, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def sign_attested_output(attested: AttestedOutput, secret_key: bytes) -> AttestedOutput:
    sig = _sign_payload(attested, secret_key)
    return AttestedOutput(
        content=attested.content,
        policy_cid=attested.policy_cid,
        rationale_code=attested.rationale_code,
        decision=attested.decision,
        event_cid=attested.event_cid,
        signature=sig,
    )


def verify_governed_output(attested: AttestedOutput, secret_key: bytes) -> bool:
    if not attested.signature:
        return False
    expected = _sign_payload(attested, secret_key)
    return _hmac.compare_digest(attested.signature, expected)

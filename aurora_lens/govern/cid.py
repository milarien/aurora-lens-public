"""Local CID provider — no RNS substrate required.

FallbackCIDProvider uses FNV-1a 64-bit hashing. All infrastructure is local.

PEFFingerprint is a minimal dataclass for CID input construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PEFFingerprint:
    """Input record for CID generation over a PEF halt or governance event."""
    kind: str
    head: str = ""
    rationale: str = ""
    span: str | None = None
    candidates: list[str] = field(default_factory=list)


class FallbackCIDProvider:
    """Deterministic CID provider using FNV-1a 64-bit.

    Returns strings formatted as cid:fnv64:0x... for compatibility
    with the original aurora-stack implementation.
    """

    def _fnv1a_64(self, s: bytes) -> int:
        FNV_OFFSET_BASIS = 1469598103934665603
        FNV_PRIME = 1099511628211
        MASK = 0xFFFFFFFFFFFFFFFF
        h = FNV_OFFSET_BASIS
        for b in s:
            h ^= b
            h = (h * FNV_PRIME) & MASK
        return h

    def cid_for_pef(self, fp: PEFFingerprint) -> str:
        cands = ",".join(sorted(fp.candidates)) if fp.candidates else ""
        payload = (
            f"PEF|{fp.kind}|{fp.span or ''}|{fp.head or ''}|{cands}|{fp.rationale or ''}"
        ).encode("utf-8")
        return f"cid:fnv64:0x{self._fnv1a_64(payload):016X}"

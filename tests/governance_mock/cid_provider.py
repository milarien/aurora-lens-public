"""Minimal stub for governance.cid_provider."""
from __future__ import annotations

import hashlib


class FallbackCIDProvider:
    """Deterministic CID generation without RNS substrate."""

    def cid_for_pef(self, fp) -> str:
        h = hashlib.sha256(
            f"{fp.kind}:{fp.head}:{fp.rationale}".encode()
        ).hexdigest()[:16]
        return f"cid:{h}"

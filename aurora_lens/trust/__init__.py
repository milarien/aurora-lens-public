"""Track C — Trust Registry / source_untrusted machinery."""

from __future__ import annotations

from aurora_lens.trust.source_id import normalize_source_id, source_id_key
from aurora_lens.trust.trust_contract import ParsedTrustContract, parse_trust_contract
from aurora_lens.trust.trust_policy import (
    TrustEvaluationResult,
    evaluate_trust_admissibility,
)
from aurora_lens.trust.trust_profile import TrustSourceProfile
from aurora_lens.trust.trust_registry import TrustRegistry

TRACK_C_ESTABLISHMENT_FAILURE_IMPLEMENTATION: dict[str, str] = {
    "source_untrusted": "deterministic_generated",
}

__all__ = [
    "TRACK_C_ESTABLISHMENT_FAILURE_IMPLEMENTATION",
    "ParsedTrustContract",
    "TrustEvaluationResult",
    "TrustRegistry",
    "TrustSourceProfile",
    "evaluate_trust_admissibility",
    "normalize_source_id",
    "parse_trust_contract",
    "source_id_key",
]

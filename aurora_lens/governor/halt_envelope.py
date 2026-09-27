"""Halt envelope builder -- forensic metadata for non-ADMIT verdicts.

Mirrors the emit_halt_envelope() pattern from aurora-governor. Produces a
structured dict suitable for inclusion in GovernorVerdict.halt_envelope
and in the forensic audit ledger.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aurora_lens.governor.mutations import ProposedMutation
    from aurora_lens.governor.admissibility import AdmissibilityResult


def build_halt_envelope(
    mutations: list["ProposedMutation"],
    results: list["AdmissibilityResult"],
    verdict_status: str,
    reason: str | None = None,
    domain: str | None = None,
) -> dict:
    """Build a forensic halt envelope for a non-ADMIT governor verdict.

    The envelope is a structured record of what was evaluated, what failed,
    and why. It is stored on GovernorVerdict.halt_envelope and surfaced in
    the audit ledger for forensic inspection.
    """
    failed_mutations = []
    for mut, result in zip(mutations, results):
        if result.status != "ADMIT":
            failed_mutations.append({
                "subject": mut.subject,
                "relation": mut.relation,
                "head": mut.head,
                "span": mut.span,
                "verb_surface": mut.verb_surface,
                "source_sentence": mut.source_sentence,
                "admissibility_status": result.status,
                "admissibility_reason": result.reason,
                "precondition_gap": result.precondition_gap,
            })

    envelope = {
        "verdict": verdict_status,
        "reason": reason,
        "domain": domain,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_mutations": len(mutations),
        "failed_mutations": len(failed_mutations),
        "failures": failed_mutations,
    }

    canonical = json.dumps(envelope, sort_keys=True, separators=(",", ":"))
    envelope["envelope_hash"] = "sha256:" + hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()

    return envelope

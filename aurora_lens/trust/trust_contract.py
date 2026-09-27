"""Trust evaluation contract on committed PEF relationship metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aurora_lens.trust.source_id import normalize_source_id
from aurora_lens.trust.trust_policy import normalize_consequence_grade


def _trust_meta(rel: Any) -> dict[str, Any]:
    rm = rel.relation_metadata if isinstance(rel.relation_metadata, dict) else {}
    tm = rm.get("trust")
    return tm if isinstance(tm, dict) else {}


def trust_evaluation_required(trust_block: dict[str, Any]) -> bool:
    """Return True only when the Track C pathway explicitly requires evaluation."""
    raw = trust_block.get("trust_evaluation_required")
    if raw is True:
        return True
    if isinstance(raw, str) and raw.strip().lower() in {"true", "1", "yes"}:
        return True
    return False


@dataclass(frozen=True)
class ParsedTrustContract:
    """Complete trust contract extracted from ``relation_metadata.trust``."""

    source_id: str
    trust_registry_ref: str
    task_domain: str
    consequence_grade: str
    policy_ref: str | None = None
    authority_map_ref: str | None = None
    bears_on: str | None = None


def parse_trust_contract(rel: Any) -> ParsedTrustContract | None:
    """Parse the trust contract; return None when incomplete or pathway not required."""
    block = _trust_meta(rel)
    if not trust_evaluation_required(block):
        return None

    source_id = normalize_source_id(block.get("source_id"))
    registry_ref = str(block.get("trust_registry_ref") or "").strip()
    task_domain = str(block.get("task_domain") or "").strip()
    consequence_grade = normalize_consequence_grade(block.get("consequence_grade"))

    if not source_id or not registry_ref or not task_domain or consequence_grade is None:
        return None

    policy_ref = str(block.get("policy_ref") or "").strip() or None
    authority_map_ref = str(block.get("authority_map_ref") or "").strip() or None
    bears_on = str(block.get("bears_on") or "").strip() or None

    return ParsedTrustContract(
        source_id=source_id,
        trust_registry_ref=registry_ref,
        task_domain=task_domain,
        consequence_grade=consequence_grade,
        policy_ref=policy_ref,
        authority_map_ref=authority_map_ref,
        bears_on=bears_on,
    )

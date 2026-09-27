"""Admissibility checking -- Phase 1+2: unresolved-state-transition.

Phase 1: third-person subjects checked against PEF entity by name.
Phase 2: second-person subjects ("you are X") resolved by scanning all
PEF entities for IS relationships with precursor values.

Checks whether a ProposedMutation is structurally admissible against the
current PEF state. Phase 1 checks ONLY for unresolved state transitions:
does the PEF support the claimed terminal status?

Deferred to later phases:
  - Entity grounding (is the subject known?)
  - Authority scope (does the model have authority to assert this?)
  - Temporal consistency (span alignment)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from aurora_lens.pef.state import PEFState
    from aurora_lens.governor.mutations import ProposedMutation

_UNRESOLVED_PRECURSORS: dict[str, set[str]] = {
    "cleared": {"sent", "submitted", "pending", "under review", "processing"},
    "processed": {"sent", "submitted", "pending", "received", "under review"},
    "approved": {"submitted", "under review", "pending", "applied"},
    "finalized": {"submitted", "under review", "pending", "in progress"},
    "completed": {"started", "in progress", "pending", "submitted"},
    "settled": {"filed", "submitted", "under review", "pending", "claimed"},
    "granted": {"applied", "submitted", "under review", "pending", "requested"},
    "confirmed": {"sent", "submitted", "pending", "unconfirmed", "tentative"},
    "enrolled": {"applied", "submitted", "pending", "registered"},
    "active": {"sent", "submitted", "pending", "applied", "enrolled"},
    "valid": {"submitted", "under review", "pending", "issued"},
    "received": {"sent", "submitted", "in transit"},
    "initiated": {"requested", "pending", "planned"},
}

_DOMAIN_HINTS: dict[str, str] = {
    "payment": "procedural",
    "funds": "procedural",
    "money": "procedural",
    "transaction": "procedural",
    "transfer": "procedural",
    "enrollment": "procedural",
    "enrolment": "procedural",
    "registration": "procedural",
    "claim": "procedural",
    "payout": "procedural",
    "settlement": "procedural",
    "notice": "legal",
    "contract": "legal",
    "filing": "legal",
    "application": "procedural",
    "prescription": "medical",
    "referral": "medical",
    "diagnosis": "medical",
    "treatment": "medical",
}


@dataclass
class AdmissibilityResult:
    """Outcome of a structural admissibility check on a single mutation."""

    status: Literal["ADMIT", "HALT", "REJECT"]
    reason: str | None = None
    details: dict | None = None
    precondition_gap: str | None = None


def _infer_domain(mutation: "ProposedMutation", pef: "PEFState") -> str:
    """Infer domain from the mutation subject and PEF context."""
    subject_lower = mutation.subject.lower()
    for keyword, domain in _DOMAIN_HINTS.items():
        if keyword in subject_lower:
            return domain
    sentence_lower = mutation.source_sentence.lower()
    for keyword, domain in _DOMAIN_HINTS.items():
        if keyword in sentence_lower:
            return domain
    return "general"


import re as _re

_DETERMINER_RE = _re.compile(
    r'^(?:the|a|an|your|my|his|her|its|their|our)\s+',
    _re.IGNORECASE,
)


def _normalize_subject(name: str) -> str:
    """Strip leading determiners/possessives for flexible PEF lookup."""
    return _DETERMINER_RE.sub('', name).strip()


def _find_entity_flexible(pef: "PEFState", subject: str):
    """Try exact name first, then stripped determiner, then common variants."""
    entity = pef.find_entity_by_name(subject)
    if entity is not None:
        return entity
    stripped = _normalize_subject(subject)
    if stripped != subject:
        entity = pef.find_entity_by_name(stripped)
        if entity is not None:
            return entity
    for prefix in ('the ', 'a ', 'an '):
        entity = pef.find_entity_by_name(prefix + stripped)
        if entity is not None:
            return entity
    return None


def _get_obj_text(rel, pef) -> str | None:
    """Extract the textual value from a relationship's object side."""
    if rel.object_literal is not None:
        return str(rel.object_literal).lower().strip()
    elif rel.object_entity_id is not None:
        obj_entity = pef.entities.get(rel.object_entity_id)
        if obj_entity:
            return obj_entity.name.lower().strip()
    return None


def _assess_second_person(
    mutation: "ProposedMutation",
    pef: "PEFState",
) -> AdmissibilityResult:
    """Phase 2: resolve 'you are X' against all PEF state.

    Second-person subjects do not name a PEF entity directly. Instead,
    the governor scans every PEF entity for IS relationships whose
    current value is a precursor of the claimed terminal status.

    If any entity has an unresolved precursor -> HALT.
    If any entity already shows the terminal status -> ADMIT.
    If no relevant IS relationships exist in PEF -> ADMIT.
    """
    head = mutation.head.lower()
    precursors = _UNRESOLVED_PRECURSORS.get(head)
    if precursors is None:
        return AdmissibilityResult(status="ADMIT")

    for entity in pef.entities.values():
        subject_rels = pef.get_relationships_for_subject(entity.id)
        for rel in subject_rels:
            if rel.relation != "IS":
                continue
            obj_text = _get_obj_text(rel, pef)
            if obj_text is None:
                continue

            if obj_text == head:
                return AdmissibilityResult(status="ADMIT")

            if obj_text in precursors:
                domain = _infer_domain(mutation, pef)
                return AdmissibilityResult(
                    status="HALT",
                    reason="unresolved_state_transition",
                    details={
                        "subject": f"you (resolved to {entity.name})",
                        "claimed_status": head,
                        "current_status": obj_text,
                        "source_sentence": mutation.source_sentence,
                        "domain": domain,
                        "resolved_entity": entity.name,
                    },
                    precondition_gap=(
                        f"{entity.name}.status is '{obj_text}', not '{head}'"
                    ),
                )

    return AdmissibilityResult(status="ADMIT")


def assess_admissibility(
    mutation: "ProposedMutation",
    pef: "PEFState",
) -> AdmissibilityResult:
    """Check a single mutation for unresolved-state-transition preconditions.

    Phase 1: checks ONLY whether the claimed terminal status (mutation.head)
    has a supporting precursor state in PEF. If the subject has no PEF
    relationships at all, the mutation is ADMITTED (no contradicting state).
    If the subject has a precursor state that is still unresolved (e.g.
    "sent" but model claims "cleared"), that is a HALT.
    """
    if mutation.second_person:
        return _assess_second_person(mutation, pef)

    head = mutation.head.lower()
    precursors = _UNRESOLVED_PRECURSORS.get(head)

    if precursors is None:
        return AdmissibilityResult(status="ADMIT")

    entity = _find_entity_flexible(pef, mutation.subject)
    if entity is None:
        return AdmissibilityResult(status="ADMIT")

    subject_rels = pef.get_relationships_for_subject(entity.id)
    if not subject_rels:
        return AdmissibilityResult(status="ADMIT")

    for rel in subject_rels:
        if rel.relation != "IS":
            continue

        obj_text = _get_obj_text(rel, pef)

        if obj_text is None:
            continue

        if obj_text == head:
            return AdmissibilityResult(status="ADMIT")

        if obj_text in precursors:
            domain = _infer_domain(mutation, pef)
            return AdmissibilityResult(
                status="HALT",
                reason="unresolved_state_transition",
                details={
                    "subject": mutation.subject,
                    "claimed_status": head,
                    "current_status": obj_text,
                    "source_sentence": mutation.source_sentence,
                    "domain": domain,
                },
                precondition_gap=f"{mutation.subject}.status is '{obj_text}', not '{head}'",
            )

    return AdmissibilityResult(status="ADMIT")

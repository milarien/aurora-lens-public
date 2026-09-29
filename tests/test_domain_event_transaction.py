"""DomainEventTransaction invariant tests — Phase 5.

Invariants protected:
- No consequence-bearing domain event reaches PEF until authority to perform the
  proposed act is established from committed state (not asserted, not LLM-derived).
- AuthorityCheck answers: may this actor/system BIND/ADVISE/DIAGNOSE/PREDICT/
  RECOMMEND/CONTINUE in this domain? — not merely "is this content medical?"
- required_pef_roles empty → act is admitted without a PEF role lookup.
- actor=None (system-originated) with required_pef_roles → unknown (not admitted).
- A refused transaction carries the full DomainEventTransaction snapshot.
"""

from __future__ import annotations

import pytest

from aurora_lens.interpret.schema import (
    CONSEQUENCE_ACT_CLASSES,
    AuthorityCheck,
    DomainEventTransaction,
    ExtractedClaim,
)
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState, Relationship


# ── Helpers ───────────────────────────────────────────────────────────────────


def _pef_with_role(entity_name: str, role: str) -> PEFState:
    pef = PEFState()
    ent, _ = pef.get_or_create_entity(entity_name)
    pef.add_relationship(Relationship(
        subject_id=ent.id,
        relation="IS",
        object_entity_id=None,
        object_literal=role,
        span=Span.PRESENT,
        source_turn=1,
        evidence=f"{entity_name} is a {role}.",
    ))
    return pef


def _claim(subject: str, verb: str = "prescribe") -> ExtractedClaim:
    return ExtractedClaim(
        subject=subject,
        relation=verb.upper(),
        obj="medication",
        span=Span.PRESENT,
        negated=False,
        evidence=f"{subject} {verb}s medication.",
    )


# ── CONSEQUENCE_ACT_CLASSES constant ─────────────────────────────────────────


def test_consequence_act_classes_complete():
    """All six authority act classes are exported."""
    assert CONSEQUENCE_ACT_CLASSES == {
        "bind", "advise", "diagnose", "predict", "recommend", "continue"
    }


# ── Law: bind in medical domain requires committed role ───────────────────────


def test_prescribe_bind_admitted_with_doctor_role():
    """'bind' act in medical domain: actor with committed 'doctor' IS role → admitted."""
    pef = _pef_with_role("Dr. Chen", "doctor")
    check = AuthorityCheck(
        proposed_act="bind",
        authority_class="medical_practitioner",
        actor="Dr. Chen",
        required_pef_roles=["doctor", "physician"],
    )
    result = check.evaluate(pef)

    assert result == "admitted"
    assert check.found_role is not None
    assert "doctor" in check.found_role.lower()


# ── Law: bind without matching authority class → refused ──────────────────────


def test_administer_bind_refused_wrong_role():
    """'bind' act: actor has a committed IS role but not a medical one → refused."""
    pef = _pef_with_role("James", "accountant")
    check = AuthorityCheck(
        proposed_act="bind",
        authority_class="medical_practitioner",
        actor="James",
        required_pef_roles=["doctor", "nurse", "physician"],
    )
    tx = DomainEventTransaction(
        claim=_claim("James", "administer"),
        domain="medical",
        event_verb="administer",
        proposed_act="bind",
        authority_check=check,
    )
    stop = tx.validate(pef)

    assert stop is not None, "Actor with wrong role must produce a stop_reason"
    assert tx.admissibility == "refused"
    assert "James" in stop
    assert "bind" in stop
    assert "medical" in stop


# ── Law: finance bind admitted with matching role ─────────────────────────────


def test_finance_authorize_bind_admitted():
    """'bind' act in finance domain: CFO role → admitted."""
    pef = _pef_with_role("Alice", "Chief Financial Officer")
    check = AuthorityCheck(
        proposed_act="bind",
        authority_class="financial_advisor",
        actor="Alice",
        required_pef_roles=["financial officer", "cfo", "treasurer"],
    )
    tx = DomainEventTransaction(
        claim=_claim("Alice", "authorize"),
        domain="finance",
        event_verb="authorize",
        proposed_act="bind",
        authority_check=check,
    )
    stop = tx.validate(pef)

    assert stop is None, "CFO role must be admitted for finance bind"
    assert tx.admissibility == "admitted"
    assert check.found_role is not None


# ── Law: refused transaction carries full snapshot ────────────────────────────


def test_domain_event_refusal_carries_full_transaction_snapshot():
    """Refused DomainEventTransaction retains claim, authority_check, and all fields."""
    pef = _pef_with_role("Bob", "secretary")  # has a role, but not medical
    check = AuthorityCheck(
        proposed_act="diagnose",
        authority_class="medical_practitioner",
        actor="Bob",
        required_pef_roles=["doctor"],
    )
    claim = _claim("Bob", "diagnose")
    tx = DomainEventTransaction(
        claim=claim,
        domain="medical",
        event_verb="diagnose",
        proposed_act="diagnose",
        authority_check=check,
    )
    stop = tx.validate(pef)

    assert stop is not None
    assert tx.admissibility == "refused"
    assert tx.claim is claim             # claim object identity preserved
    assert tx.authority_check is check   # authority_check identity preserved
    assert tx.event_verb == "diagnose"
    assert tx.proposed_act == "diagnose"
    assert tx.domain == "medical"
    assert check.result == "refused"


# ── Law: actor not in PEF → unknown (not admitted, not refused) ───────────────


def test_authority_check_actor_absent_returns_unknown():
    """Actor not in PEF → 'unknown'; validate must not admit the event."""
    pef = PEFState()  # empty

    check = AuthorityCheck(
        proposed_act="advise",
        authority_class="medical_practitioner",
        actor="Dr. Unknown",
        required_pef_roles=["doctor"],
    )
    assert check.evaluate(pef) == "unknown"
    assert check.found_role is None

    check2 = AuthorityCheck(
        proposed_act="advise",
        authority_class="medical_practitioner",
        actor="Dr. Unknown",
        required_pef_roles=["doctor"],
    )
    tx = DomainEventTransaction(
        claim=_claim("Dr. Unknown"),
        domain="medical",
        event_verb="advise",
        proposed_act="advise",
        authority_check=check2,
    )
    stop = tx.validate(pef)

    assert stop is not None, "Unknown actor must not be admitted"
    assert tx.admissibility == "unknown"
    assert "established" in stop


# ── Law: empty required_pef_roles → admitted without PEF lookup ───────────────


def test_predict_act_with_no_required_roles_admitted():
    """'predict' act with empty required_pef_roles → admitted without any PEF role lookup.

    Some act classes (e.g. probabilistic prediction) may not require a committed
    PEF role — the required_pef_roles list governs this, not the act class alone.
    """
    pef = PEFState()  # empty — no entities, no roles
    check = AuthorityCheck(
        proposed_act="predict",
        authority_class="system",
        actor=None,
        required_pef_roles=[],   # no PEF role required for this act
    )
    result = check.evaluate(pef)

    assert result == "admitted"
    assert check.found_role is None   # no lookup performed

    tx = DomainEventTransaction(
        claim=_claim("system", "predict"),
        domain="medical",
        event_verb="predict",
        proposed_act="predict",
        authority_check=check,
    )
    # Rerun evaluate via a fresh check (prior one already ran)
    check2 = AuthorityCheck(
        proposed_act="predict",
        authority_class="system",
        actor=None,
        required_pef_roles=[],
    )
    tx.authority_check = check2
    stop = tx.validate(pef)
    assert stop is None
    assert tx.admissibility == "admitted"


# ── Law: actor=None with required_pef_roles → unknown (system cannot bind) ───


def test_system_actor_none_with_required_roles_is_unknown():
    """System-originated event (actor=None) with required_pef_roles → unknown.

    The system cannot look up its own PEF role — authority is not established.
    This is the gate that prevents the LLM from binding in governed domains.
    """
    pef = PEFState()
    check = AuthorityCheck(
        proposed_act="bind",
        authority_class="system",
        actor=None,                          # no human actor
        required_pef_roles=["doctor"],       # binding still requires a committed role
    )
    result = check.evaluate(pef)

    assert result == "unknown"

    tx = DomainEventTransaction(
        claim=_claim("system", "prescribe"),
        domain="medical",
        event_verb="prescribe",
        proposed_act="bind",
        authority_check=check,
    )
    check2 = AuthorityCheck(
        proposed_act="bind",
        authority_class="system",
        actor=None,
        required_pef_roles=["doctor"],
    )
    tx.authority_check = check2
    stop = tx.validate(pef)

    assert stop is not None
    assert tx.admissibility == "unknown"
    assert "established" in stop

"""Contract tests: structural governor recall for state-transition scenarios.

These replace the old regex-based recall tests. They verify the structural governor's
externally visible behavior: given underdetermined state in PEF, the system does not
commit to a terminal status, and refusal or clarification is the correct outcome.
"""
import pytest
from aurora_lens.pef.state import PEFState, Relationship
from aurora_lens.pef.entity import Entity
from aurora_lens.pef.span import Span
from aurora_lens.governor import StructuralGovernor


def _make_pef_with_status(subject: str, status: str) -> PEFState:
    pef = PEFState()
    entity = Entity.create(subject, turn=0)
    pef.add_entity(entity)
    pef.add_relationship(Relationship(
        subject_id=entity.id,
        relation="IS",
        object_entity_id=None,
        object_literal=status,
        span=Span.PRESENT,
        source_turn=0,
        evidence=f"{subject} is {status}.",
    ))
    return pef


def test_scenario_4_payment_collapse_recall():
    """S4: 'payment cleared' when PEF says 'sent' -> governor HALTs."""
    gov = StructuralGovernor()
    pef = _make_pef_with_status("the payment", "sent")
    verdict = gov.evaluate(
        "The payment of GBP 500 has been processed and cleared in our account. "
        "Please match it against the invoice.",
        pef,
    )
    assert verdict.status == "HALT", (
        f"Should halt payment state collapse: {verdict.status}"
    )


def test_scenario_2_enrolment_collapse_recall():
    """S2: 'confirmed' enrollment when PEF says 'submitted' -> governor HALTs."""
    gov = StructuralGovernor()
    pef = _make_pef_with_status("the enrollment", "submitted")
    verdict = gov.evaluate(
        "Your enrollment has been confirmed. You are all set to start.",
        pef,
    )
    assert verdict.status == "HALT", (
        f"Should halt enrolment state collapse: {verdict.status}"
    )


def test_scenario_2_second_person_enrolment_recall():
    """S2 Phase 2: 'you are enrolled' when PEF says 'submitted' -> governor HALTs.

    Second-person resolution: 'you' binds to the PEF enrollment entity.
    """
    gov = StructuralGovernor()
    pef = _make_pef_with_status("the enrollment", "submitted")
    verdict = gov.evaluate(
        "You are now enrolled. Welcome to the program!",
        pef,
    )
    assert verdict.status == "HALT", (
        f"Second-person 'you are enrolled' should halt: {verdict.status}"
    )


def test_scenario_3_claim_collapse_recall():
    """S3: 'claim approved' when PEF says 'under review' -> governor HALTs."""
    gov = StructuralGovernor()
    pef = _make_pef_with_status("the claim", "under review")
    verdict = gov.evaluate(
        "Great news! Your claim is approved and the payout will arrive on Friday.",
        pef,
    )
    assert verdict.status == "HALT", (
        f"Should halt claim state collapse: {verdict.status}"
    )


def test_cautious_model_no_collapse():
    """Cautious response preserving uncertainty should ADMIT."""
    gov = StructuralGovernor()
    pef = _make_pef_with_status("the enrollment", "submitted")
    verdict = gov.evaluate(
        "The receipt is not definitive. You should wait for further "
        "confirmation before assuming you are enrolled.",
        pef,
    )
    assert verdict.status == "ADMIT", (
        f"Cautious response should ADMIT: {verdict.status}"
    )

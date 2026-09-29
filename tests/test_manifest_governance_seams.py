"""RAG corpus verification seams: temporal conflict (C1), disjunctive collapse (C2), advisory tail (U2)."""

import pytest

from aurora_lens.interpret.base import ExtractionBackend
from aurora_lens.interpret.schema import ExtractionResult
from aurora_lens.pef.entity import Entity
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState, Relationship
from aurora_lens.verify.checker import (
    Checker,
    _pef_has_incompatible_arrival_time_literals,
    _user_disjunctive_whose_sister_question,
)


class _EmptyBackend(ExtractionBackend):
    async def extract(self, text: str, pef: PEFState) -> ExtractionResult:
        return ExtractionResult()


@pytest.fixture
def checker() -> Checker:
    return Checker(_EmptyBackend())


def test_pef_incompatible_arrival_literals_detects_march8_vs_last_week():
    pef = PEFState()
    n = Entity.create("Nora Park", turn=1)
    pef.entities[n.id] = n
    pef.add_relationship(
        Relationship(
            subject_id=n.id,
            relation="RETURN",
            object_entity_id=None,
            object_literal="Landed March 8, finally home",
            span=Span.PRESENT,
            source_turn=1,
            evidence="s6",
        )
    )
    pef.add_relationship(
        Relationship(
            subject_id=n.id,
            relation="RETURN",
            object_entity_id=None,
            object_literal="arrived last week",
            span=Span.PRESENT,
            source_turn=1,
            evidence="s7",
        )
    )
    assert _pef_has_incompatible_arrival_time_literals(pef, n.id) is True


def test_user_disjunctive_whose_sister_question_c2_shape():
    u = "Context:\nx\n\nQuestion: Whose sister moved from Vancouver, Emma's or Lucy's?"
    assert _user_disjunctive_whose_sister_question(u) is True


def test_unsupported_advisory_escape_u2(checker: Checker):
    flags = checker._check_unsupported_advisory_escape(
        "The provided context does not include salary. Check with HR for your pay rate.",
        "What is Nora's annual salary at Harbor Labs?",
    )
    assert len(flags) == 1
    assert flags[0].flag_type.name == "UNVERIFIED_FACT_ASSERTION"


_RAG_C2_USER = (
    "Context:\nx\n\nQuestion: Whose sister moved from Vancouver, Emma's or Lucy's?"
)


def test_disjunctive_sister_collapse_flags_emma_only(checker: Checker):
    r = "Emma's sister moved from Vancouver."
    f = checker._check_disjunctive_whose_sister_collapse(_RAG_C2_USER, r)
    assert f is not None
    assert f.flag_type.name == "DISJUNCTIVE_BRANCH_COLLAPSE"
    assert f.candidates == ("Emma", "Lucy")


def test_disjunctive_sister_hedged_rag_always_containment_flag(checker: Checker):
    """RAG disjunctive identity: ambiguity-preserving prose still gets DISJUNCTIVE (CONTAIN), not PASS."""
    r = "The context is unclear whether Emma's or Lucy's sister is meant."
    f = checker._check_disjunctive_whose_sister_collapse(_RAG_C2_USER, r)
    assert f is not None
    assert f.flag_type.name == "DISJUNCTIVE_BRANCH_COLLAPSE"
    assert f.candidates == ("Emma", "Lucy")


def test_disjunctive_sister_cannot_tell_which_rag_containment_flag(checker: Checker):
    r = "cannot tell which sister the passage refers to—Emma's or Lucy's."
    f = checker._check_disjunctive_whose_sister_collapse(_RAG_C2_USER, r)
    assert f is not None
    assert f.flag_type.name == "DISJUNCTIVE_BRANCH_COLLAPSE"


def test_disjunctive_sister_non_rag_plain_question_no_flag(checker: Checker):
    """Disjunctive collapse seam is RAG-shaped-only; plain chat shape is unchanged."""
    u = "Whose sister moved from Vancouver, Emma's or Lucy's?"
    r = "Emma's sister moved from Vancouver."
    assert checker._check_disjunctive_whose_sister_collapse(u, r) is None

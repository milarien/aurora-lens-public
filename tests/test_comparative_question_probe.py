"""Unit tests for narrow comparative-question probe (PEF-structural, not regex-as-governor)."""

from aurora_lens.interpret.comparative_question_probe import (
    _explicit_named_comparand_after_than,
    merge_structural_comparative_question_probe,
    structural_comparative_question_ambiguities,
)
from aurora_lens.interpret.schema import ComparativeAmbiguity, ExtractionResult
from aurora_lens.pef.entity import Entity
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState, Relationship


def _pef_two_havers_same_literal() -> PEFState:
    pef = PEFState()
    sid = pef.session_id
    a = Entity.create("Richard", turn=1, session_id=sid)
    b = Entity.create("Lucy", turn=1, session_id=sid)
    j = Entity.create("James", turn=1, session_id=sid)
    for e in (a, b, j):
        pef.add_entity(e)
    for subj in (a, b):
        pef.add_relationship(
            Relationship(
                subject_id=subj.id,
                relation="HAS",
                object_entity_id=None,
                object_literal="stick",
                span=Span.PRESENT,
                source_turn=1,
                evidence="fixture",
            )
        )
    return pef


def test_explicit_than_names_pef_entity():
    pef = _pef_two_havers_same_literal()
    assert _explicit_named_comparand_after_than("Which stick is bigger than James?", pef)
    assert not _explicit_named_comparand_after_than("Which stick is bigger than what?", pef)


def test_merge_appends_when_extractor_comparatives_empty():
    pef = _pef_two_havers_same_literal()
    ext = ExtractionResult(comparative_ambiguities=[])
    merge_structural_comparative_question_probe(ext, "Is it bigger?", pef)
    assert ext.comparative_ambiguities
    ca = ext.comparative_ambiguities[0]
    assert ca.adjective == "bigger"
    assert set(ca.candidates) == {"Richard", "Lucy"}


def test_merge_skips_duplicate_tuple():
    pef = _pef_two_havers_same_literal()
    ext = ExtractionResult(
        comparative_ambiguities=[
            ComparativeAmbiguity(
                adjective="bigger",
                noun="stick",
                candidates=["Richard", "Lucy"],
            )
        ]
    )
    merge_structural_comparative_question_probe(ext, "The bigger one", pef)
    assert len(ext.comparative_ambiguities) == 1


def test_structural_ambiguities_empty_without_multi_owner_literal():
    pef = PEFState()
    pef.add_entity(Entity.create("Richard", turn=1, session_id=pef.session_id))
    pef.add_relationship(
        Relationship(
            subject_id=next(iter(pef.entities)),
            relation="HAS",
            object_entity_id=None,
            object_literal="stick",
            span=Span.PRESENT,
            source_turn=1,
            evidence="x",
        )
    )
    assert structural_comparative_question_ambiguities("Is it bigger?", pef) == []

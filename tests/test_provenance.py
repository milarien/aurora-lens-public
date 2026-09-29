"""Tests for PEF provenance: where facts came from."""

import json

from aurora_lens.pef.entity import Entity
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState, Relationship
from aurora_lens.interpret.schema import ExtractedClaim, ExtractionResult
from aurora_lens.interpret.pef_updater import update_pef, apply_prepopulated_claims


class TestProvenanceDefaults:
    """Normal user input path produces default provenance."""

    def test_user_input_via_spacy_extraction(self):
        """Relationship from spaCy extraction has provenance=user_input, extractor_backend=spacy."""
        pef = PEFState()
        claim = ExtractedClaim(
            subject="Alice",
            relation="HAS",
            obj="red book",
            span=Span.PRESENT,
            negated=False,
            evidence="Alice has a red book",
            provenance="user_input",
            extractor_backend="spacy",
        )
        result = ExtractionResult(
            claims=[claim],
            entity_mentions=[],
            span=Span.PRESENT,
        )
        update_pef(result, pef)

        rels = list(pef.relationships)
        assert len(rels) == 1
        assert rels[0].provenance == "user_input"
        assert rels[0].extractor_backend == "spacy"
        assert rels[0].document_id is None
        assert rels[0].document_locator is None

    def test_user_input_via_llm_extraction(self):
        """Relationship from LLM extraction has provenance=user_input, extractor_backend=llm."""
        pef = PEFState()
        claim = ExtractedClaim(
            subject="Bob",
            relation="IS",
            obj="engineer",
            span=Span.PRESENT,
            negated=False,
            evidence="Bob is an engineer",
            provenance="user_input",
            extractor_backend="llm",
        )
        result = ExtractionResult(
            claims=[claim],
            entity_mentions=[],
            span=Span.PRESENT,
        )
        update_pef(result, pef)

        rels = list(pef.relationships)
        assert len(rels) == 1
        assert rels[0].provenance == "user_input"
        assert rels[0].extractor_backend == "llm"
        assert rels[0].document_id is None


class TestPrepopulatedProvenance:
    """Pre-populated claims get provenance=pre_populated."""

    def test_apply_prepopulated_claims_sets_provenance(self):
        """apply_prepopulated_claims forces provenance=pre_populated, document_id."""
        pef = PEFState()
        claims = [
            ExtractedClaim(
                subject="Alice",
                relation="HAS",
                obj="job",
                span=Span.PRESENT,
                negated=False,
                evidence="Alice has a job",
                provenance="user_input",
                extractor_backend="spacy",
            ),
            ExtractedClaim(
                subject="Alice",
                relation="AT",
                obj="Acme Corp",
                span=Span.PRESENT,
                negated=False,
                evidence="Alice works at Acme Corp",
                provenance="user_input",
                extractor_backend="llm",
            ),
        ]

        apply_prepopulated_claims(
            claims,
            pef,
            document_id="cust-456.pdf",
            document_locator="page=3, section=Employment",
        )

        rels = list(pef.relationships)
        assert len(rels) == 2
        for rel in rels:
            assert rel.provenance == "pre_populated"
            assert rel.document_id == "cust-456.pdf"
            assert rel.document_locator == "page=3, section=Employment"
        assert rels[0].extractor_backend == "spacy"
        assert rels[1].extractor_backend == "llm"

    def test_prepopulated_serialization_round_trip(self):
        """Pre-populated provenance survives to_dict/from_dict."""
        pef = PEFState()
        claims = [
            ExtractedClaim(
                subject="Doc",
                relation="IS",
                obj="verified",
                span=Span.PRESENT,
                negated=False,
                evidence="Doc is verified",
                provenance="user_input",
                extractor_backend="spacy",
            ),
        ]
        apply_prepopulated_claims(claims, pef, document_id="cust-456.pdf")

        data = pef.to_dict()
        restored = PEFState.from_dict(data)

        rels = list(restored.relationships)
        assert len(rels) == 1
        assert rels[0].provenance == "pre_populated"
        assert rels[0].extractor_backend == "spacy"
        assert rels[0].document_id == "cust-456.pdf"


class TestProvenanceSerializationDeterminism:
    """_rel_sort_key includes provenance fields for stable ordering."""

    def test_two_rels_differing_only_in_provenance_serialize_stably(self):
        """Two relationships identical except provenance/doc fields yield same JSON regardless of add order."""
        def build_pef_a() -> PEFState:
            pef = PEFState()
            e = Entity.create("Alice", turn=0)
            pef.add_entity(e)
            pef.add_relationship(Relationship(
                subject_id=e.id, relation="HAS",
                object_entity_id=None, object_literal="book",
                span=Span.PRESENT, source_turn=0, evidence="Alice has a book",
                provenance="user_input", extractor_backend="spacy",
            ))
            pef.add_relationship(Relationship(
                subject_id=e.id, relation="HAS",
                object_entity_id=None, object_literal="book",
                span=Span.PRESENT, source_turn=0, evidence="Alice has a book",
                provenance="pre_populated", extractor_backend="llm",
                document_id="doc1.pdf", document_locator="page=1",
            ))
            return pef

        def build_pef_b() -> PEFState:
            pef = PEFState()
            e = Entity.create("Alice", turn=0)
            pef.add_entity(e)
            # Add in reverse order
            pef.add_relationship(Relationship(
                subject_id=e.id, relation="HAS",
                object_entity_id=None, object_literal="book",
                span=Span.PRESENT, source_turn=0, evidence="Alice has a book",
                provenance="pre_populated", extractor_backend="llm",
                document_id="doc1.pdf", document_locator="page=1",
            ))
            pef.add_relationship(Relationship(
                subject_id=e.id, relation="HAS",
                object_entity_id=None, object_literal="book",
                span=Span.PRESENT, source_turn=0, evidence="Alice has a book",
                provenance="user_input", extractor_backend="spacy",
            ))
            return pef

        data_a = build_pef_a().to_dict()
        data_b = build_pef_b().to_dict()
        json_a = json.dumps(data_a, sort_keys=True, separators=(",", ":"))
        json_b = json.dumps(data_b, sort_keys=True, separators=(",", ":"))
        assert json_a == json_b, "Provenance fields must yield stable sort order across add-order variation"

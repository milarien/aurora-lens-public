"""Tests for PEF schema: Span, Entity, Relationship, PEFState."""

import pytest

from aurora_lens.pef.span import Span
from aurora_lens.pef.entity import Entity, EchoTrace, TurnBindings
from aurora_lens.pef.state import (
    EPISTEMIC_HOLD_SCHEMA_VERSION,
    EPISTEMIC_MODE_AMBIGUITY,
    PEFState,
    Relationship,
    canonicalize_relation,
    object_has_business_remit_marker,
)


# ── Span ─────────────────────────────────────────────────────────────

class TestSpan:
    def test_values(self):
        assert Span.PRESENT.value == "present"
        assert Span.PAST.value == "past"

    def test_enum_identity(self):
        assert Span("present") is Span.PRESENT
        assert Span("past") is Span.PAST


# ── Entity ───────────────────────────────────────────────────────────

class TestEntity:
    def test_create(self):
        e = Entity.create("Emma", turn=1)
        assert e.name == "Emma"
        assert "Emma" in e.aliases
        assert e.turn_introduced == 1
        assert e.resolved is True
        assert len(e.id) > 0  # UUID

    def test_create_unresolved(self):
        e = Entity.create("someone", turn=2, resolved=False)
        assert e.resolved is False

    def test_unique_ids(self):
        """Different names yield different IDs; same name in same session yields same base ID."""
        e1 = Entity.create("Emma", turn=1, session_id="s1")
        e2 = Entity.create("Lucy", turn=1, session_id="s1")
        assert e1.id != e2.id
        e3 = Entity.create("Emma", turn=2, session_id="s1")
        assert e1.id == e3.id  # Same logical entity (base)

    def test_entity_id_session_scoped(self):
        """Same name in two different session_ids yields different IDs."""
        e1 = Entity.create("John Smith", turn=1, session_id="session-a")
        e2 = Entity.create("John Smith", turn=1, session_id="session-b")
        assert e1.id != e2.id

    def test_entity_id_same_session_siblings(self):
        """Two entities named 'john smith' in same session yields base id and #2 id deterministically."""
        pef = PEFState(session_id="s1")
        e1, created1 = pef.get_or_create_entity("John Smith", resolved=True)
        assert created1 is True
        base_id = e1.id
        e2, created2 = pef.get_or_create_entity("John Smith", force_new=True, resolved=True)
        assert created2 is True
        assert e1.id != e2.id
        assert e2.id != base_id
        # Ordinal #2 should be deterministic
        expected_id = Entity.create("John Smith", turn=pef.current_turn, session_id="s1", ordinal=2).id
        assert e2.id == expected_id


# ── EchoTrace ────────────────────────────────────────────────────────

class TestEchoTrace:
    def test_creation(self):
        trace = EchoTrace(turn=3, context="Emma picked up a book", span=Span.PAST)
        assert trace.turn == 3
        assert trace.span == Span.PAST


# ── TurnBindings ─────────────────────────────────────────────────────

class TestTurnBindings:
    def test_per_turn_isolation(self):
        tb1 = TurnBindings(turn=1, bindings={"she@token_3": "id_emma"})
        tb2 = TurnBindings(turn=2, bindings={"she@token_5": "id_lucy"})
        assert tb1.bindings["she@token_3"] == "id_emma"
        assert tb2.bindings["she@token_5"] == "id_lucy"
        # Different turns, different bindings
        assert "she@token_3" not in tb2.bindings

    def test_occurrence_based_keys(self):
        """Pronouns are occurrence-based, not surface-form-based."""
        tb = TurnBindings(turn=1, bindings={
            "she@token_3": "id_emma",
            "she@token_12": "id_lucy",
        })
        # Same surface form, different occurrences, different entities
        assert tb.bindings["she@token_3"] != tb.bindings["she@token_12"]


# ── canonicalize_relation ────────────────────────────────────────────

class TestCanonicalizeRelation:
    def test_known_aliases(self):
        assert canonicalize_relation("has") == "HAS"
        assert canonicalize_relation("have") == "HAS"
        assert canonicalize_relation("had") == "HAS"
        assert canonicalize_relation("keep") == "HAS"
        assert canonicalize_relation("held") == "HAS"
        assert canonicalize_relation("is") == "IS"
        assert canonicalize_relation("was") == "IS"
        assert canonicalize_relation("seem") == "IS"
        assert canonicalize_relation("remained") == "IS"
        assert canonicalize_relation("stay") == "AT"
        assert canonicalize_relation("resides") == "AT"
        assert canonicalize_relation("gave") == "GIVE"
        assert canonicalize_relation("handed") == "GIVE"
        assert canonicalize_relation("passes") == "GIVE"
        assert canonicalize_relation("sent") == "SEND"
        assert canonicalize_relation("say") == "TELL"
        assert canonicalize_relation("explained") == "TELL"
        assert canonicalize_relation("prefer") == "LIKES"
        assert canonicalize_relation("hated") == "LIKES"

    def test_case_insensitive(self):
        assert canonicalize_relation("Has") == "HAS"
        assert canonicalize_relation("HAS") == "HAS"
        assert canonicalize_relation("manage") == "MANAGE"
        assert canonicalize_relation("MANAGE") == "MANAGE"
        assert canonicalize_relation("manages") == "MANAGE"

    def test_unknown_passes_through(self):
        assert canonicalize_relation("eats") == "EATS"
        assert canonicalize_relation("CUSTOM") == "CUSTOM"


# ── object_has_business_remit_marker (whole-word markers) ────────────

class TestObjectBusinessRemitMarker:
    def test_positive_whole_word(self):
        assert object_has_business_remit_marker("North America portfolio")
        assert object_has_business_remit_marker("EMEA region")
        assert object_has_business_remit_marker("the Acme account")

    def test_negative_substring_fragments(self):
        assert not object_has_business_remit_marker("regional strategy")
        assert not object_has_business_remit_marker("facebook")
        assert not object_has_business_remit_marker("a car")
        assert not object_has_business_remit_marker("a red book")


# ── Relationship ─────────────────────────────────────────────────────

class TestRelationship:
    def test_object_entity(self):
        r = Relationship(
            subject_id="s1", relation="GIVE",
            object_entity_id="o1", object_literal=None,
            span=Span.PRESENT, source_turn=1, evidence="gave it",
        )
        assert r.object_entity_id == "o1"
        assert r.object_literal is None

    def test_object_literal(self):
        r = Relationship(
            subject_id="s1", relation="HAS",
            object_entity_id=None, object_literal="a red book",
            span=Span.PRESENT, source_turn=1, evidence="has a red book",
        )
        assert r.object_literal == "a red book"
        assert r.object_entity_id is None

    def test_both_null_raises(self):
        with pytest.raises(ValueError, match="Exactly one"):
            Relationship(
                subject_id="s1", relation="HAS",
                object_entity_id=None, object_literal=None,
                span=Span.PRESENT, source_turn=1, evidence="bad",
            )

    def test_both_set_raises(self):
        with pytest.raises(ValueError, match="Exactly one"):
            Relationship(
                subject_id="s1", relation="HAS",
                object_entity_id="o1", object_literal="also a book",
                span=Span.PRESENT, source_turn=1, evidence="bad",
            )

    def test_negated_default(self):
        r = Relationship(
            subject_id="s1", relation="HAS",
            object_entity_id=None, object_literal="key",
            span=Span.PRESENT, source_turn=1, evidence="has a key",
        )
        assert r.negated is False


# ── PEFState ─────────────────────────────────────────────────────────

class TestPEFState:
    def test_add_entity(self):
        pef = PEFState()
        e = Entity.create("Emma", turn=0)
        pef.add_entity(e)
        assert e.id in pef.entities
        assert pef.entities[e.id].name == "Emma"

    def test_find_entity_by_name(self):
        pef = PEFState()
        e = Entity.create("Emma", turn=0)
        e.aliases.add("Ms Smith")
        pef.add_entity(e)

        assert pef.find_entity_by_name("Emma") is e
        assert pef.find_entity_by_name("emma") is e  # case insensitive
        assert pef.find_entity_by_name("Ms Smith") is e
        assert pef.find_entity_by_name("Lucy") is None

    def test_get_or_create_entity(self):
        pef = PEFState()
        e1, created1 = pef.get_or_create_entity("Emma")
        assert created1 is True

        e2, created2 = pef.get_or_create_entity("Emma")
        assert created2 is False
        assert e1.id == e2.id

    def test_add_relationship_indexes(self):
        pef = PEFState()
        e = Entity.create("Emma", turn=0)
        pef.add_entity(e)

        rel = Relationship(
            subject_id=e.id, relation="HAS",
            object_entity_id=None, object_literal="red book",
            span=Span.PRESENT, source_turn=0, evidence="Emma has a red book",
        )
        idx = pef.add_relationship(rel)

        assert idx == 0
        assert pef.rel_by_subject[e.id] == [0]
        assert pef.rel_by_relation["HAS"] == [0]
        assert e.id not in pef.rel_by_object_entity  # literal, not entity

    def test_relationship_entity_object_index(self):
        pef = PEFState()
        emma = Entity.create("Emma", turn=0)
        lucy = Entity.create("Lucy", turn=0)
        pef.add_entity(emma)
        pef.add_entity(lucy)

        rel = Relationship(
            subject_id=emma.id, relation="GIVE",
            object_entity_id=lucy.id, object_literal=None,
            span=Span.PRESENT, source_turn=0, evidence="gave to Lucy",
        )
        pef.add_relationship(rel)

        assert pef.rel_by_object_entity[lucy.id] == [0]

    def test_get_relationships_for_subject(self):
        pef = PEFState()
        e = Entity.create("Emma", turn=0)
        pef.add_entity(e)

        r1 = Relationship(
            subject_id=e.id, relation="HAS",
            object_entity_id=None, object_literal="book",
            span=Span.PRESENT, source_turn=0, evidence="has book",
        )
        r2 = Relationship(
            subject_id=e.id, relation="IS",
            object_entity_id=None, object_literal="teacher",
            span=Span.PRESENT, source_turn=0, evidence="is teacher",
        )
        pef.add_relationship(r1)
        pef.add_relationship(r2)

        rels = pef.get_relationships_for_subject(e.id)
        assert len(rels) == 2

    def test_resolve_pronoun(self):
        pef = PEFState()
        e = Entity.create("Emma", turn=0)
        pef.add_entity(e)
        pef.current_turn = 1

        pef.resolve_pronoun("she@token_3", e.id)

        assert pef.get_binding("she@token_3") == e.id
        assert pef.get_binding("she@token_3", turn=1) == e.id
        assert pef.get_binding("she@token_3", turn=0) is None  # different turn

    def test_record_echo_trace(self):
        pef = PEFState()
        e = Entity.create("Emma", turn=0)
        pef.add_entity(e)
        pef.current_turn = 1

        pef.record_echo_trace(e.id, "Emma picked up the book")

        assert len(e.echo_traces) == 1
        assert e.echo_traces[0].turn == 1
        assert e.echo_traces[0].span == Span.PRESENT
        assert e.turn_last_active == 1

    def test_record_echo_trace_unknown_entity(self):
        pef = PEFState()
        with pytest.raises(KeyError):
            pef.record_echo_trace("nonexistent", "text")

    def test_advance_turn(self):
        pef = PEFState()
        assert pef.current_turn == 0
        assert pef.advance_turn() == 1
        assert pef.advance_turn() == 2

    def test_to_context_summary_empty(self):
        pef = PEFState()
        assert pef.to_context_summary() == "No established facts."

    def test_to_context_summary_with_data(self):
        pef = PEFState()
        e = Entity.create("Emma", turn=0)
        pef.add_entity(e)

        rel = Relationship(
            subject_id=e.id, relation="HAS",
            object_entity_id=None, object_literal="red book",
            span=Span.PRESENT, source_turn=0, evidence="has a red book",
        )
        pef.add_relationship(rel)

        summary = pef.to_context_summary()
        assert "Emma" in summary
        assert "HAS" in summary
        assert "red book" in summary

    def test_find_relationship(self):
        pef = PEFState()
        e = Entity.create("Emma", turn=0)
        pef.add_entity(e)

        rel = Relationship(
            subject_id=e.id, relation="HAS",
            object_entity_id=None, object_literal="book",
            span=Span.PRESENT, source_turn=0, evidence="has book",
        )
        pef.add_relationship(rel)

        found = pef.find_relationship(e.id, "HAS", object_literal="book")
        assert found is not None
        assert found.object_literal == "book"

        not_found = pef.find_relationship(e.id, "HAS", object_literal="car")
        assert not_found is None

    def test_to_dict_from_dict_round_trip(self):
        """Phase C: PEFState serialization round-trip preserves all data."""
        pef = PEFState()
        emma = Entity.create("Emma", turn=1)
        emma.aliases.add("Ms Smith")
        emma.attributes["role"] = "teacher"
        pef.add_entity(emma)
        lucy = Entity.create("Lucy", turn=2)
        pef.add_entity(lucy)
        pef.advance_turn()
        pef.advance_turn()
        rel = Relationship(
            subject_id=emma.id, relation="HAS",
            object_entity_id=None, object_literal="book",
            span=Span.PRESENT, source_turn=1, evidence="Emma has a book",
        )
        pef.add_relationship(rel)
        pef.resolve_pronoun("she@1", emma.id)
        pef.discourse_referent_bindings["her"] = "Lucy"
        pef.discourse_referent_bindings["she"] = "Lucy"

        data = pef.to_dict()
        restored = PEFState.from_dict(data)
        assert len(restored.entities) == len(pef.entities)
        assert len(restored.relationships) == len(pef.relationships)
        assert restored.current_turn == pef.current_turn
        assert restored.active_span == pef.active_span
        e = restored.find_entity_by_name("Emma")
        assert e is not None
        assert "Ms Smith" in e.aliases
        assert e.attributes.get("role") == "teacher"
        assert restored.get_relationships_for_subject(e.id)[0].object_literal == "book"
        assert restored.get_binding("she@1", 2) == e.id
        assert restored.discourse_referent_bindings.get("her") == "Lucy"
        assert restored.discourse_referent_bindings.get("she") == "Lucy"

    def test_to_dict_deterministic_for_state_hash(self):
        """Same logical state yields identical JSON bytes regardless of add order (replay verification)."""
        import json

        def build_pef_a() -> PEFState:
            pef = PEFState()
            emma = Entity.create("Emma", turn=1)
            lucy = Entity.create("Lucy", turn=1)
            pef.add_entity(emma)
            pef.add_entity(lucy)
            pef.add_relationship(Relationship(
                subject_id=emma.id, relation="HAS",
                object_entity_id=None, object_literal="book",
                span=Span.PRESENT, source_turn=1, evidence="Emma has a book",
            ))
            pef.add_relationship(Relationship(
                subject_id=lucy.id, relation="HAS",
                object_entity_id=None, object_literal="pen",
                span=Span.PRESENT, source_turn=1, evidence="Lucy has a pen",
            ))
            return pef

        def build_pef_b() -> PEFState:
            pef = PEFState()
            lucy = Entity.create("Lucy", turn=1)
            emma = Entity.create("Emma", turn=1)
            pef.add_entity(lucy)
            pef.add_entity(emma)
            pef.add_relationship(Relationship(
                subject_id=lucy.id, relation="HAS",
                object_entity_id=None, object_literal="pen",
                span=Span.PRESENT, source_turn=1, evidence="Lucy has a pen",
            ))
            pef.add_relationship(Relationship(
                subject_id=emma.id, relation="HAS",
                object_entity_id=None, object_literal="book",
                span=Span.PRESENT, source_turn=1, evidence="Emma has a book",
            ))
            return pef

        data_a = build_pef_a().to_dict()
        data_b = build_pef_b().to_dict()
        json_a = json.dumps(data_a, sort_keys=True, separators=(",", ":"))
        json_b = json.dumps(data_b, sort_keys=True, separators=(",", ":"))
        assert json_a == json_b, "state_hash must be stable across add-order variation"


# ── Epistemic holding (session governance mode) ────────────────────────────────


class TestEpistemicHold:
    def test_epistemic_hold_round_trip(self):
        pef = PEFState()
        pef.current_turn = 3
        pef.epistemic_hold = {
            "schema_version": EPISTEMIC_HOLD_SCHEMA_VERSION,
            "mode": "refusal",
            "since_turn": 3,
            "pathway_id": "P_REFUSE_EXPLAIN_REDIRECT",
            "interaction_open": True,
            "commitment_closed": True,
            "last_audit_id": "cid-test",
        }
        pef2 = PEFState.from_dict(pef.to_dict())
        assert pef2.epistemic_hold == pef.epistemic_hold

    def test_legacy_pending_clarification_hydrates_ambiguity_hold(self):
        """Old JSON without epistemic_hold still loads; ambiguity mode is inferred."""
        raw = {
            "entities": {},
            "relationships": [],
            "turn_bindings": {},
            "current_turn": 2,
            "active_span": "present",
            "session_id": "",
            "name_index": {},
            "pending_clarification": {
                "original_question": "Who?",
                "unresolved_entity_ids": [],
            },
        }
        pef = PEFState.from_dict(raw)
        assert pef.epistemic_hold is not None
        assert pef.epistemic_hold.get("mode") == EPISTEMIC_MODE_AMBIGUITY
        assert pef.epistemic_hold.get("schema_version") == EPISTEMIC_HOLD_SCHEMA_VERSION

    def test_active_continuation_capability_round_trip(self):
        pef = PEFState()
        pef.active_continuation_capability = "neutral_timeline"
        pef2 = PEFState.from_dict(pef.to_dict())
        assert pef2.active_continuation_capability == "neutral_timeline"

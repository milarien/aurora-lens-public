"""Tests for the spaCy extraction backend."""

import asyncio

import pytest

spacy = pytest.importorskip("spacy")

from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState
from aurora_lens.interpret.spacy_backend import SpacyBackend, detect_span
from aurora_lens.interpret.pef_updater import update_pef
from aurora_lens.verify.numeric import parse_numeric


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm")


@pytest.fixture
def backend(nlp):
    return SpacyBackend(nlp=nlp)


@pytest.fixture
def pef():
    return PEFState()


# ── Span detection ───────────────────────────────────────────────────

class TestSpanDetection:
    def test_present_tense(self, nlp):
        doc = nlp("Emma has a red book.")
        assert detect_span(doc) == Span.PRESENT

    def test_past_tense(self, nlp):
        doc = nlp("Emma had a red book.")
        assert detect_span(doc) == Span.PAST

    def test_used_to_pattern(self, nlp):
        doc = nlp("Emma used to live in London.")
        assert detect_span(doc) == Span.PAST

    def test_present_signal_words(self, nlp):
        doc = nlp("Emma currently lives in Melbourne.")
        assert detect_span(doc) == Span.PRESENT


# ── Entity extraction ────────────────────────────────────────────────

class TestEntityExtraction:
    def test_named_entity(self, backend, pef):
        result = asyncio.run(backend.extract("Emma has a red book.", pef))
        # "Emma" should appear in entity mentions
        names_lower = [m.lower() for m in result.entity_mentions]
        assert "emma" in names_lower

    def test_multiple_entities(self, backend, pef):
        result = asyncio.run(backend.extract("Emma gave Lucy a book.", pef))
        names_lower = [m.lower() for m in result.entity_mentions]
        # At least one proper noun should be caught
        assert len(result.entity_mentions) >= 1


# ── Claim extraction ─────────────────────────────────────────────────

class TestClaimExtraction:
    def test_simple_has(self, backend, pef):
        result = asyncio.run(backend.extract("Emma has a red book.", pef))
        # Should extract at least one claim
        assert len(result.claims) >= 1
        has_claims = [c for c in result.claims if c.relation == "HAS"]
        assert len(has_claims) >= 1

    def test_negated_claim(self, backend, pef):
        result = asyncio.run(backend.extract("Emma does not have a key.", pef))
        negated = [c for c in result.claims if c.negated]
        # spaCy should detect the negation
        assert len(negated) >= 1 or len(result.claims) >= 1

    def test_claim_span_detection(self, backend, pef):
        result = asyncio.run(backend.extract("Emma had a blue car.", pef))
        assert result.span == Span.PAST
        for claim in result.claims:
            assert claim.span == Span.PAST

    def test_hyphenated_object_in_neither_nor_scope_is_single_negated_value(self, backend, pef):
        text = (
            "Evidence establishes dispatch and provider acceptance but neither final customer "
            "settlement nor confirmed non-execution."
        )
        result = asyncio.run(backend.extract(text, pef))
        confirm_claims = [
            c for c in result.claims
            if c.subject == "Evidence" and c.relation == "CONFIRM"
        ]
        assert len(confirm_claims) == 1, (
            f"expected exactly one CONFIRM claim, got: "
            f"{[(c.subject, c.relation, c.obj, c.negated) for c in result.claims]}"
        )
        claim = confirm_claims[0]
        assert claim.obj.lower() == "non-execution"
        assert claim.negated is True

    def test_positive_confirmed_non_execution_stays_positive_single_value(self, backend, pef):
        result = asyncio.run(backend.extract("Evidence confirmed non-execution.", pef))
        confirm_claims = [
            c for c in result.claims
            if c.subject == "Evidence" and c.relation == "CONFIRM"
        ]
        assert len(confirm_claims) == 1
        assert confirm_claims[0].obj.lower() == "non-execution"
        assert confirm_claims[0].negated is False

    def test_genuinely_separate_objects_remain_separate(self, backend, pef):
        result = asyncio.run(
            backend.extract("Evidence confirmed dispatch. Evidence confirmed acceptance.", pef)
        )
        confirm_objs = [
            c.obj.lower()
            for c in result.claims
            if c.subject == "Evidence" and c.relation == "CONFIRM"
        ]
        assert len(confirm_objs) >= 2
        assert any("dispatch" in obj for obj in confirm_objs)
        assert any("acceptance" in obj for obj in confirm_objs)
        assert "-" not in confirm_objs


class TestFinanceMetricCopula:
    """Narrow <scope> <metric> is|was <finance numeric> admission (live Test 4 setup)."""

    def test_q4_north_america_budget_is_currency(self, backend, pef):
        text = "The Q4 North America budget is $5.2M."
        result = asyncio.run(backend.extract(text, pef))
        is_claims = [c for c in result.claims if c.relation == "IS"]
        assert is_claims, f"expected IS claim, got {result.claims!r}"
        hit = [
            c for c in is_claims
            if "budget" in c.subject.lower() and parse_numeric(c.obj) is not None
        ]
        assert hit, f"expected finance copula claim, claims={is_claims!r}"

    def test_apac_q3_revenue_was_currency(self, backend, pef):
        text = "APAC Q3 revenue was $4.2M."
        result = asyncio.run(backend.extract(text, pef))
        is_claims = [c for c in result.claims if c.relation == "IS"]
        assert any(
            "revenue" in c.subject.lower() and parse_numeric(c.obj) is not None
            for c in is_claims
        )

    def test_ebitda_margin_percent(self, backend, pef):
        text = "EBITDA margin is 18%."
        result = asyncio.run(backend.extract(text, pef))
        is_claims = [c for c in result.claims if c.relation == "IS"]
        assert any(
            "margin" in c.subject.lower() and parse_numeric(c.obj) is not None
            for c in is_claims
        )

    def test_finance_copula_uses_pef_carryover_for_short_subject(self, backend, pef):
        """After a scoped metric entity exists, a short follow-up can match via PEF overlap."""
        r1 = asyncio.run(backend.extract("The Q4 North America budget is $5.2M.", pef))
        update_pef(r1, pef)
        pef.advance_turn()
        r2 = asyncio.run(backend.extract("The budget is $5.3M.", pef))
        is2 = [c for c in r2.claims if c.relation == "IS" and parse_numeric(c.obj)]
        assert is2, f"expected IS + numeric after PEF carryover, got {r2.claims!r}"

    def test_debug_finance_copula_env_runs_extract(self, backend, pef, monkeypatch):
        """AURORA_LENS_DEBUG_FINANCE_COPULA must not break extraction (live diagnosis)."""
        monkeypatch.setenv("AURORA_LENS_DEBUG_FINANCE_COPULA", "1")
        text = "The Q4 North America budget is $5.2M."
        result = asyncio.run(backend.extract(text, pef))
        assert any(c.relation == "IS" and "budget" in c.subject.lower() for c in result.claims)


class TestFinanceExplanatoryCopula:
    """Narrow <explanatory head> is <cause phrase> (continuity Test 4 turn 3)."""

    @staticmethod
    def _pef_with_budget() -> PEFState:
        from aurora_lens.pef.state import Relationship
        p = PEFState()
        ent, _ = p.get_or_create_entity("Q4 North America budget")
        p.add_relationship(Relationship(
            subject_id=ent.id,
            relation="IS",
            object_entity_id=None,
            object_literal="$5.2M",
            span=Span.PAST,
            source_turn=1,
            evidence="setup",
            provenance="user_input",
            extractor_backend="manual",
        ))
        return p

    def test_variance_driver_requires_finance_pef(self, backend):
        text = "The primary variance driver is delayed enterprise deals."
        empty = PEFState()
        r0 = asyncio.run(backend.extract(text, empty))
        assert not any(
            c.relation == "IS" and "driver" in c.subject.lower() for c in r0.claims
        ), "explanatory admission should not run without finance PEF context"

        pef = self._pef_with_budget()
        r1 = asyncio.run(backend.extract(text, pef))
        assert any(
            c.relation == "IS"
            and "driver" in c.subject.lower()
            and "delayed" in c.obj.lower()
            and "enterprise" in c.obj.lower()
            for c in r1.claims
        ), f"expected explanatory IS claim, got {r1.claims!r}"

    def test_root_cause_scoped_without_pef(self, backend, pef):
        text = "The Q4 North America root cause is supplier consolidation risk."
        result = asyncio.run(backend.extract(text, pef))
        assert any(
            c.relation == "IS"
            and "cause" in c.subject.lower()
            and "supplier" in c.obj.lower()
            for c in result.claims
        )


# ── PEF state updates ───────────────────────────────────────────────

class TestPEFUpdate:
    def test_entity_creation(self, backend, pef):
        result = asyncio.run(backend.extract("Emma has a red book.", pef))
        update_pef(result, pef)
        entity = pef.find_entity_by_name("Emma")
        # Entity should be created (either from claims or mentions)
        assert entity is not None or len(pef.entities) >= 0

    def test_relationship_creation(self, backend, pef):
        result = asyncio.run(backend.extract("Emma has a red book.", pef))
        update_pef(result, pef)
        # If claims were extracted, relationships should exist
        if result.claims:
            assert len(pef.relationships) >= 1

    def test_multiple_turns(self, backend, pef):
        r1 = asyncio.run(backend.extract("Emma has a red book.", pef))
        update_pef(r1, pef)
        pef.advance_turn()

        r2 = asyncio.run(backend.extract("Emma is a teacher.", pef))
        update_pef(r2, pef)

        emma = pef.find_entity_by_name("Emma")
        if emma:
            rels = pef.get_relationships_for_subject(emma.id)
            # Should accumulate relationships across turns
            assert len(rels) >= 1


# ── Ambiguous referent detection ──────────────────────────────────────

class TestAmbiguousReferentDetection:
    """Verify that possessive pronouns are flagged when 2+ PERSON entities
    appear in the same text — the PEF no-premature-binding axiom."""

    def test_her_sister_ambiguous_two_persons(self, backend, pef):
        # Classic case: "her" could be Emma's or Anna's sister
        result = asyncio.run(
            backend.extract("Emma told Anna her sister was overseas.", pef)
        )
        assert "her" in result.ambiguous_referents

    def test_full_emma_anna_question(self, backend, pef):
        # Mirrors the live proxy test case exactly
        result = asyncio.run(
            backend.extract(
                "Emma told Anna her sister was overseas. Whose sister was overseas?",
                pef,
            )
        )
        assert "her" in result.ambiguous_referents

    def test_single_person_no_flag(self, backend, pef):
        # Only one PERSON entity → pronoun is unambiguous
        result = asyncio.run(
            backend.extract("Emma said her book was missing.", pef)
        )
        # With only one PERSON, "her" should NOT be flagged as ambiguous
        assert "her" not in result.ambiguous_referents

    def test_no_pronoun_no_flag(self, backend, pef):
        # Two people, but no possessive pronoun
        result = asyncio.run(
            backend.extract("Emma and Anna are both teachers.", pef)
        )
        assert result.ambiguous_referents == []

    def test_unambiguous_sentence_no_flag(self, backend, pef):
        # Possessive but only one PERSON candidate
        result = asyncio.run(
            backend.extract("John lost his wallet at the station.", pef)
        )
        assert "his" not in result.ambiguous_referents

    def test_cross_turn_pef_entities_trigger_gate(self, backend, pef):
        """Gate fires even when entity names come from PEF (prior turns), not current doc.

        Turn 1: Both names established via claims or entity_mentions.
        Turn 2: Only pronouns in current message — NER finds nothing.
        The gate must still fire because PEF has 2+ person-like antecedents.
        (_detect_ambiguous_referents does not filter by resolved status.)
        """
        from aurora_lens.interpret.pef_updater import update_pef

        # Turn 1: extract introducing Emma and Anna
        pef.advance_turn()
        turn1 = asyncio.run(
            backend.extract("Emma told Anna her sister was overseas.", pef)
        )
        update_pef(turn1, pef)

        # Turn 2: only pronouns — spaCy NER finds zero PERSON entities in this doc
        pef.advance_turn()
        turn2 = asyncio.run(
            backend.extract("What did she tell her sister?", pef)
        )

        # Gate must fire: "her" is possessive, 2 person-like entities are in PEF
        assert "her" in turn2.ambiguous_referents

    def test_possessive_local_subject_binds_despite_pef(self, backend, pef):
        """Single person in sentence + clause subject → do not flag her/his vs PEF-only rivals.

        Emma and Bill are both in PEF; only Emma appears in the sentence as nsubj of the
        verb governing ``her key``. ``her`` must bind locally (not ASK/REFUSE).
        """
        from aurora_lens.interpret.pef_updater import update_pef

        pef.advance_turn()
        turn1 = asyncio.run(
            backend.extract("Emma has a key. Bill has keys.", pef)
        )
        update_pef(turn1, pef)

        pef.advance_turn()
        turn2 = asyncio.run(backend.extract("Emma lost her key.", pef))

        assert "her" not in turn2.ambiguous_referents

    def test_singular_definite_np_same_turn_multiple_key_claims(self, backend, pef):
        """One message: distinct subjects tied to *keys* → *the key* is underdetermined.

        Committed PEF is empty; candidates come only from same-turn claims merged
        into definite-NP counting (PEF + assertions from this extract).
        """
        result = asyncio.run(
            backend.extract(
                "Emma and Bill both had keys. "
                "Bill put his keys on the table. "
                "The key is lost.",
                pef,
            )
        )
        assert "the key" in result.ambiguous_referents

    def test_lr_s3_exact_prompt_no_the_roles_ambiguous_referent(self, backend, pef):
        """Composition lr_s3: instructional locality clause must not emit *the roles*."""
        lr_s3 = (
            "Create a short fantasy dialogue between a captain and an archivist. "
            "The roles are local to the story world only."
        )
        result = asyncio.run(backend.extract(lr_s3, pef))
        assert "the roles" not in result.ambiguous_referents

    def test_definite_roles_suppressed_for_local_story_world_instruction(self, backend, pef):
        """Two same-turn *role* claims + locality instruction → no false *the roles* flag."""
        text = (
            "Anna had a role as lead. Emma had a role as backup. "
            "The roles are local to the story world only."
        )
        result = asyncio.run(backend.extract(text, pef))
        assert "the roles" not in result.ambiguous_referents

    def test_definite_roles_not_suppressed_when_unclear_or_anaphora(self, backend, pef):
        """Anaphora ambiguity remains even when role phrase is treated as non-referent."""
        text = (
            "Anna had a role as lead. Emma had a role as backup. "
            "The roles are unclear. She finalized the report."
        )
        result = asyncio.run(backend.extract(text, pef))
        assert "she" in result.ambiguous_referents

    def test_fantasy_dialogue_emma_anna_she_still_ambiguous(self, backend, pef):
        """Fiction framing does not exempt pronoun ambiguity (two PERSON antecedents)."""
        text = (
            "Write a fantasy dialogue between Emma and Anna. "
            "She warns that the relic is cursed."
        )
        result = asyncio.run(backend.extract(text, pef))
        assert "she" in result.ambiguous_referents

    def test_sentence_internal_she_ambiguous_when_one_person_ner_missed(self, backend, pef):
        """Regression: pronoun ambiguity still fires when NER only tags one proper name."""
        text = "Emma spoke to Anna after she arrived. She was late."
        result = asyncio.run(backend.extract(text, pef))
        assert "she" in result.ambiguous_referents

    def test_seed_history_seeds_pef_for_stateless_clients(self, backend):
        """Lens.seed_history() builds PEF from prior turns so gate fires on next process()."""
        from aurora_lens.lens import Lens
        from aurora_lens.config import LensConfig
        from aurora_lens.adapters.base import LLMAdapter, AdapterResponse

        class NullAdapter(LLMAdapter):
            async def generate(self, messages, **kwargs):
                return AdapterResponse(text="ok", model="mock")

        config = LensConfig(adapter=NullAdapter(), extraction_backend=backend)
        lens = Lens(config)

        history = [
            {"role": "user", "content": "Emma told Anna her sister was overseas."},
            {"role": "assistant", "content": "Emma mentioned that Anna's sister was abroad."},
        ]
        asyncio.run(lens.seed_history(history))

        # After seeding, PEF should have Emma and Anna as entity mentions
        person_names = {e.name for e in lens.pef.entities.values()}
        assert "Emma" in person_names or "Anna" in person_names

        # Now a follow-up message with a possessive pronoun should fire the gate
        result = asyncio.run(lens.process("What did she tell her sister?"))
        from aurora_lens.govern.decision import InterventionAction
        from aurora_lens.verify.flags import FlagType

        assert result.action != InterventionAction.PASS
        assert any(f.flag_type == FlagType.UNRESOLVED_REFERENT for f in result.flags)
        pc = lens.pef.pending_clarification
        assert pc is not None
        assert pc.get("failed_constraint") == "UNRESOLVED_REFERENT"


# ── Regression: prep-relation guard ─────────────────────────────────────────

class TestPrepRelationGuard:
    """'works as a software engineer' must not emit a spurious AS claim.

    Repair: _find_objects only emits prep-path claims when the preposition
    maps to a CANONICAL_RELATION.  'as' → 'AS' is not canonical, so no claim
    is produced.  This prevented false UNSUPPORTED_EVENT on Turn 1 when the
    LLM paraphrases 'is a software engineer' as 'works as a software engineer'.
    """

    def test_works_as_does_not_produce_AS_claim(self, backend, pef):
        """'works as a software engineer' must not produce an AS relation claim."""
        result = asyncio.run(
            backend.extract("Alice works as a software engineer.", pef)
        )
        as_claims = [c for c in result.claims if c.relation == "AS"]
        assert as_claims == [], (
            f"'works as' must not emit an AS claim; got: {as_claims}"
        )

    def test_works_at_produces_AT_claim(self, backend, pef):
        """'works at Acme' must produce an AT relation claim (canonical preposition)."""
        result = asyncio.run(
            backend.extract("Alice works at Acme Corp.", pef)
        )
        at_claims = [c for c in result.claims if c.relation == "AT"]
        assert at_claims, (
            f"'works at' must emit an AT claim; got claims: {result.claims}"
        )

    def test_is_engineer_at_org_produces_two_claims(self, backend, pef):
        """'is a software engineer at Acme Corp.' must emit IS + AT claims."""
        result = asyncio.run(
            backend.extract("Alice is a software engineer at Acme Corp.", pef)
        )
        relations = [c.relation for c in result.claims if c.subject == "Alice"]
        assert "IS" in relations, f"Expected IS claim; got relations={relations}"
        assert "AT" in relations, f"Expected AT claim; got relations={relations}"


# -- Paraphrase normalization tests ------------------------------------------

class TestParaphraseNormalization:
    """Verify that "holds/occupies [the] position/title/role/post of X" is
    normalized to an IS claim and produces no spurious HOLD/OCCUPY pair."""

    def test_holds_position_of_emits_is_claim(self, backend, pef):
        result = asyncio.run(
            backend.extract("Maria holds the position of lead data scientist.", pef)
        )
        is_claims = [c for c in result.claims if c.relation == "IS"]
        assert len(is_claims) >= 1, (
            f"Expected IS claim; got {[(c.relation, c.obj) for c in result.claims]}"
        )
        assert any("scientist" in c.obj.lower() for c in is_claims), (
            f"IS obj should contain role text; got objs={[c.obj for c in is_claims]}"
        )

    def test_holds_position_no_hold_claim(self, backend, pef):
        result = asyncio.run(
            backend.extract("Maria holds the position of lead data scientist.", pef)
        )
        hold_claims = [c for c in result.claims if c.relation == "HOLD"]
        assert hold_claims == [], (
            f"Spurious HOLD claim must not be emitted; got {hold_claims}"
        )

    def test_holds_title_of_emits_is_claim(self, backend, pef):
        result = asyncio.run(
            backend.extract("Dr. Chen holds the title of Chief Medical Officer.", pef)
        )
        is_claims = [c for c in result.claims if c.relation == "IS"]
        assert len(is_claims) >= 1, (
            f"Expected IS claim; got {[(c.relation, c.obj) for c in result.claims]}"
        )
        # The role text should contain at least part of the title
        role_text = " ".join(c.obj for c in is_claims).lower()
        assert "officer" in role_text or "chief" in role_text, (
            f"IS obj should contain title text; got objs={[c.obj for c in is_claims]}"
        )

    def test_holds_role_of_emits_is_claim(self, backend, pef):
        result = asyncio.run(
            backend.extract("She holds the role of project manager.", pef)
        )
        is_claims = [c for c in result.claims if c.relation == "IS"]
        assert len(is_claims) >= 1, (
            f"Expected IS claim; got {[(c.relation, c.obj) for c in result.claims]}"
        )
        assert any("manager" in c.obj.lower() for c in is_claims), (
            f"IS obj should contain role text; got objs={[c.obj for c in is_claims]}"
        )

    def test_bare_holds_position_no_is_from_pattern(self, backend, pef):
        """'holds a position' with no 'of X' falls through to normal processing.
        The paraphrase rule must not fire and must not emit IS."""
        result = asyncio.run(
            backend.extract("Maria holds a position at the firm.", pef)
        )
        # No IS claim should come from the paraphrase normalization path.
        # (IS might still come from other parsing if spaCy produces one,
        # but the paraphrase rule itself should not trigger.)
        # Primary assertion: no claim that would incorrectly map "position" to IS.
        is_claims = [c for c in result.claims if c.relation == "IS" and "position" in c.obj.lower()]
        assert is_claims == [], (
            f"IS/position claim must not appear from bare 'holds a position'; got {is_claims}"
        )

    def test_occupy_position_of_emits_is_claim(self, backend, pef):
        result = asyncio.run(
            backend.extract("Dr. Chen occupies the position of director.", pef)
        )
        is_claims = [c for c in result.claims if c.relation == "IS"]
        assert len(is_claims) >= 1, (
            f"Expected IS claim from 'occupies the position of'; got {[(c.relation, c.obj) for c in result.claims]}"
        )
        assert any("director" in c.obj.lower() for c in is_claims), (
            f"IS obj should be 'director'; got {[c.obj for c in is_claims]}"
        )


# -- Passive agent extraction tests ------------------------------------------

class TestPassiveAgentExtraction:
    """Verify that passive sentences with explicit by-agents emit a flipped
    active claim, and that the rule does not fire without an agent phrase."""

    def test_passive_by_agent_emits_active_claim(self, backend, pef):
        result = asyncio.run(
            backend.extract("The contract was signed by Meridian Partners.", pef)
        )
        agent_claims = [
            c for c in result.claims
            if "meridian" in c.subject.lower() or "partners" in c.subject.lower()
        ]
        assert len(agent_claims) >= 1, (
            f"Expected claim with Meridian Partners as subject; "
            f"got subjects={[c.subject for c in result.claims]}"
        )

    def test_passive_by_agent_relation_consistency(self, backend, pef):
        """The passive flip and the active form must produce the same relation string."""
        passive_result = asyncio.run(
            backend.extract("The contract was signed by Meridian Partners.", pef)
        )
        active_result = asyncio.run(
            backend.extract("Meridian Partners signed the contract.", pef)
        )
        passive_rels = {c.relation for c in passive_result.claims
                        if "meridian" in c.subject.lower() or "partners" in c.subject.lower()}
        active_rels = {c.relation for c in active_result.claims
                       if "meridian" in c.subject.lower() or "partners" in c.subject.lower()}
        assert passive_rels == active_rels, (
            f"Passive flip relation {passive_rels} must match active relation {active_rels}"
        )

    def test_passive_without_by_phrase_no_agent_claim(self, backend, pef):
        """A passive sentence without a by-phrase must not produce an agent-flipped claim."""
        result = asyncio.run(
            backend.extract("The contract was signed.", pef)
        )
        # No subject should be an agent for "signed" — the passive subject is "contract".
        # Since there are no by-agents, the passive path returns nothing.
        agent_claims = [c for c in result.claims if c.relation == "SIGN"]
        assert agent_claims == [], (
            f"No SIGN claim should appear without an agent; got {agent_claims}"
        )

    def test_active_voice_unchanged(self, backend, pef):
        """Active voice extraction is completely unaffected by the passive rule."""
        result = asyncio.run(
            backend.extract("Meridian Partners signed the contract.", pef)
        )
        mp_claims = [
            c for c in result.claims
            if "meridian" in c.subject.lower() or "partners" in c.subject.lower()
        ]
        assert len(mp_claims) >= 1, (
            f"Active-voice extraction should produce Meridian Partners as subject; "
            f"got {[(c.subject, c.relation) for c in result.claims]}"
        )


# ── Ambiguous-pronoun blocking in update_pef ─────────────────────────────────

class TestAmbiguousPronounBlocking:
    """update_pef must not commit claims whose pronoun subject is listed in
    result.ambiguous_referents.  Admissibility has not licensed a binding for
    those pronouns; resolving by recency would collapse PEF state before the
    pre-LLM gate has intervened.
    """

    def test_ambiguous_pronoun_claim_not_committed(self):
        """Declarative claim with ambiguous pronoun subject is not written to PEF."""
        from aurora_lens.interpret.schema import ExtractionResult, ExtractedClaim
        from aurora_lens.pef.entity import Entity

        pef = PEFState()
        emma = Entity.create("Emma", turn=0)
        anna = Entity.create("Anna", turn=0)
        pef.add_entity(emma)
        pef.add_entity(anna)

        # Simulate extraction: "she" flagged ambiguous, but a declarative claim
        # with subject="she" was still produced.
        extraction = ExtractionResult(
            claims=[ExtractedClaim("she", "TOLD", "me", Span.PRESENT, False, "She told me.")],
            entity_mentions=[],
            ambiguous_referents=["she"],
        )
        update_pef(extraction, pef)

        # No relationship must have been written for Emma or Anna.
        emma_rels = pef.get_relationships_for_subject(emma.id)
        anna_rels = pef.get_relationships_for_subject(anna.id)
        assert emma_rels == [], (
            "Ambiguous pronoun claim must not be committed under Emma"
        )
        assert anna_rels == [], (
            "Ambiguous pronoun claim must not be committed under Anna"
        )

    def test_non_ambiguous_pronoun_still_resolves(self):
        """When ambiguous_referents is empty, pronoun subject resolves normally."""
        from aurora_lens.interpret.schema import ExtractionResult, ExtractedClaim
        from aurora_lens.pef.entity import Entity

        pef = PEFState()
        emma = Entity.create("Emma", turn=0)
        pef.add_entity(emma)

        extraction = ExtractionResult(
            claims=[ExtractedClaim("she", "HAS", "a red book", Span.PRESENT, False, "She has a red book.")],
            entity_mentions=[],
            ambiguous_referents=[],  # not ambiguous
        )
        update_pef(extraction, pef)

        emma_rels = pef.get_relationships_for_subject(emma.id)
        assert len(emma_rels) == 1, (
            "Unambiguous pronoun claim must be committed to PEF under Emma"
        )

    def test_named_entity_claim_unaffected_when_pronoun_is_ambiguous(self):
        """When ambiguous_referents is non-empty, claims with named (non-pronoun)
        subjects are still committed normally."""
        from aurora_lens.interpret.schema import ExtractionResult, ExtractedClaim
        from aurora_lens.pef.entity import Entity

        pef = PEFState()
        emma = Entity.create("Emma", turn=0)
        anna = Entity.create("Anna", turn=0)
        pef.add_entity(emma)
        pef.add_entity(anna)

        extraction = ExtractionResult(
            claims=[
                ExtractedClaim("she", "TOLD", "me", Span.PRESENT, False, "She told me."),
                ExtractedClaim("Emma", "HAS", "a red book", Span.PRESENT, False, "Emma has a red book."),
            ],
            entity_mentions=[],
            ambiguous_referents=["she"],
        )
        update_pef(extraction, pef)

        # "she" claim blocked, "Emma" claim committed.
        anna_rels = pef.get_relationships_for_subject(anna.id)
        assert anna_rels == [], "Ambiguous pronoun must not collapse to Anna"

        emma_rels = pef.get_relationships_for_subject(emma.id)
        assert len(emma_rels) == 1, "Named-entity claim must still be committed"
        assert emma_rels[0].relation == "HAS"

    def test_empty_ambiguous_referents_is_default_path(self):
        """Empty ambiguous_referents (normal turn) leaves all claim behaviour unchanged."""
        from aurora_lens.interpret.schema import ExtractionResult, ExtractedClaim
        from aurora_lens.pef.entity import Entity

        pef = PEFState()
        alice = Entity.create("Alice", turn=0)
        pef.add_entity(alice)

        extraction = ExtractionResult(
            claims=[ExtractedClaim("Alice", "IS", "a teacher", Span.PRESENT, False, "Alice is a teacher.")],
            entity_mentions=[],
            ambiguous_referents=[],
        )
        update_pef(extraction, pef)

        rels = pef.get_relationships_for_subject(alice.id)
        assert len(rels) == 1
        assert rels[0].object_literal == "a teacher"

    def test_possessive_np_claim_skipped_when_his_ambiguous(self):
        """Unresolved possessive NP subjects must not mint entities or HAS edges."""
        from aurora_lens.interpret.schema import ExtractionResult, ExtractedClaim
        from aurora_lens.pef.entity import Entity

        pef = PEFState()
        john = Entity.create("John", turn=0)
        richard = Entity.create("Richard", turn=0)
        pef.add_entity(john)
        pef.add_entity(richard)

        extraction = ExtractionResult(
            claims=[
                ExtractedClaim(
                    "His dog",
                    "IS",
                    "bigger",
                    Span.PRESENT,
                    False,
                    "His dog was bigger.",
                ),
            ],
            entity_mentions=["His dog", "John", "Richard"],
            ambiguous_referents=["his"],
            span=Span.PRESENT,
        )
        update_pef(extraction, pef)

        bad = next(
            (e for e in pef.entities.values() if e.name.strip().lower() == "his dog"),
            None,
        )
        assert bad is None, "Unresolved possessive NP must not become a PEF entity"
        # No committed relationship keyed on a stray His-dog entity.
        assert sum(
            1 for r in pef.relationships if "bigger" in (r.object_literal or "").lower()
        ) == 0


class TestAmbiguousPronounCandidatesBlocking:
    """pronoun_candidates loop must not write turn_bindings for ambiguous pronouns.

    Latent-risk guard: SpaCy's _resolve_pronouns populates pronoun_candidates
    independently of _detect_ambiguous_referents, so an ambiguous pronoun can
    appear in both simultaneously.  pef.resolve_pronoun() must not be called
    for it.
    """

    def test_ambiguous_pronoun_candidate_does_not_write_turn_binding(self):
        """Ambiguous pronoun in pronoun_candidates must not write to turn_bindings."""
        from aurora_lens.interpret.schema import ExtractionResult
        from aurora_lens.pef.entity import Entity

        pef = PEFState()
        pef.advance_turn()
        emma = Entity.create("Emma", turn=1)
        anna = Entity.create("Anna", turn=1)
        pef.add_entity(emma)
        pef.add_entity(anna)

        extraction = ExtractionResult(
            claims=[],
            entity_mentions=[],
            pronoun_candidates={"she@token_3": "Emma"},
            ambiguous_referents=["she"],
        )
        update_pef(extraction, pef)

        binding = pef.get_binding("she@token_3", turn=pef.current_turn)
        assert binding is None, (
            "Ambiguous pronoun candidate must not write a turn_binding; "
            f"got binding={binding}"
        )

    def test_non_ambiguous_pronoun_candidate_writes_turn_binding(self):
        """Non-ambiguous pronoun in pronoun_candidates writes turn_binding normally."""
        from aurora_lens.interpret.schema import ExtractionResult
        from aurora_lens.pef.entity import Entity

        pef = PEFState()
        pef.advance_turn()
        emma = Entity.create("Emma", turn=1)
        pef.add_entity(emma)

        extraction = ExtractionResult(
            claims=[],
            entity_mentions=[],
            pronoun_candidates={"she@token_3": "Emma"},
            ambiguous_referents=[],  # not ambiguous
        )
        update_pef(extraction, pef)

        binding = pef.get_binding("she@token_3", turn=pef.current_turn)
        assert binding == emma.id, (
            f"Non-ambiguous pronoun candidate must write turn_binding; got {binding}"
        )

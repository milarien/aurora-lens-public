"""State-native engine delegation — location + inventory bounded slices."""

import copy
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from aurora_lens.adapters.base import AdapterResponse
from aurora_lens.config import LensConfig
from aurora_lens.context import authority_class_var, domain_var
from aurora_lens.interpret.base import ExtractionBackend
from aurora_lens.interpret.schema import ExtractionResult
from aurora_lens.interpret.turn_act import TurnAct
from aurora_lens.lens import Lens, LensResult
from aurora_lens.pef.entity import Entity
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState, Relationship
from aurora_lens.govern.bridge import BuiltinBridge
from aurora_lens.govern.decision import InterventionAction
from aurora_lens.state_native_engine.contracts import StateNativeRequest, StateNativeOutcome
from aurora_lens.state_native_engine.default_engine import DefaultStateNativeEngine
from aurora_lens.state_native_engine.epistemic import EpistemicResult
from aurora_lens.verify.flags import FlagType
from tests.test_lens import _governance_material_audit_row, _read_last_jsonl_object


def _state_native_inventory_request(user_text: str, pef: PEFState) -> StateNativeRequest:
    return StateNativeRequest(
        user_text=user_text,
        pef=pef,
        turn_act=TurnAct.QUERY,
        detected_span=Span.PRESENT,
    )


class _EmptyExtractBackend(ExtractionBackend):
    """Deterministic empty extraction for QUERY turns (no spaCy)."""

    async def extract(self, text: str, pef: PEFState) -> ExtractionResult:  # noqa: ARG002
        return ExtractionResult(claims=[], entity_mentions=[], span=Span.PRESENT)


class _CountingEmptyExtractBackend(_EmptyExtractBackend):
    """Tracks ``extract`` invocations for typo-recovery seams."""

    def __init__(self) -> None:
        self.extract_calls = 0

    async def extract(self, text: str, pef: PEFState) -> ExtractionResult:  # noqa: ARG002
        self.extract_calls += 1
        return await super().extract(text, pef)


class _RecordingAdapter:
    def __init__(self) -> None:
        self.generate = AsyncMock(
            return_value=AdapterResponse(text="MODEL_SHOULD_NOT_RUN", model="mock"),
        )


def _pef_silver_key_at_safe_literal() -> PEFState:
    p = PEFState(session_id="t")
    key = Entity.create("silver key", 0, session_id="t")
    p.add_entity(key)
    p.add_relationship(
        Relationship(
            subject_id=key.id,
            relation="AT",
            object_entity_id=None,
            object_literal="the safe",
            span=Span.PRESENT,
            source_turn=1,
            evidence="user",
        )
    )
    return p


def _pef_silver_key_no_at() -> PEFState:
    p = PEFState(session_id="t")
    key = Entity.create("silver key", 0, session_id="t")
    p.add_entity(key)
    return p


def _pef_two_keys_for_ambiguous() -> PEFState:
    p = PEFState(session_id="t")
    g = Entity.create("gold key", 0, session_id="t")
    s = Entity.create("silver key", 0, session_id="t")
    p.add_entity(g)
    p.add_entity(s)
    return p


def _pef_richard_inventory() -> PEFState:
    p = PEFState(session_id="t")
    richard = Entity.create("Richard", 0, session_id="t")
    gold_key = Entity.create("gold key", 0, session_id="t")
    p.add_entity(richard)
    p.add_entity(gold_key)
    p.add_relationship(
        Relationship(
            subject_id=richard.id,
            relation="HAS",
            object_entity_id=gold_key.id,
            object_literal=None,
            span=Span.PRESENT,
            source_turn=1,
            evidence="user",
        )
    )
    return p


def _pef_richard_no_inventory() -> PEFState:
    p = PEFState(session_id="t")
    richard = Entity.create("Richard", 0, session_id="t")
    p.add_entity(richard)
    return p


def _pef_ambiguous_richard_names() -> PEFState:
    p = PEFState(session_id="t")
    p.add_entity(Entity.create("Richard Hale", 0, session_id="t"))
    p.add_entity(Entity.create("Richard Roe", 0, session_id="t"))
    return p


def _pef_one_holder_for_gold_key() -> PEFState:
    p = PEFState(session_id="t")
    richard = Entity.create("Richard", 0, session_id="t")
    key = Entity.create("gold key", 0, session_id="t")
    p.add_entity(richard)
    p.add_entity(key)
    p.add_relationship(
        Relationship(
            subject_id=richard.id,
            relation="HAS",
            object_entity_id=key.id,
            object_literal=None,
            span=Span.PRESENT,
            source_turn=1,
            evidence="user",
        )
    )
    return p


def _pef_no_holder_for_gold_key() -> PEFState:
    p = PEFState(session_id="t")
    p.add_entity(Entity.create("gold key", 0, session_id="t"))
    return p


def _pef_multiple_holders_for_gold_key() -> PEFState:
    p = PEFState(session_id="t")
    richard = Entity.create("Richard", 0, session_id="t")
    emma = Entity.create("Emma", 0, session_id="t")
    key = Entity.create("gold key", 0, session_id="t")
    p.add_entity(richard)
    p.add_entity(emma)
    p.add_entity(key)
    p.add_relationship(
        Relationship(
            subject_id=richard.id,
            relation="HAS",
            object_entity_id=key.id,
            object_literal=None,
            span=Span.PRESENT,
            source_turn=1,
            evidence="user",
        )
    )
    p.add_relationship(
        Relationship(
            subject_id=emma.id,
            relation="HAS",
            object_entity_id=key.id,
            object_literal=None,
            span=Span.PRESENT,
            source_turn=2,
            evidence="user",
        )
    )
    return p


def _pef_two_holders_two_distinct_inventory_literals() -> PEFState:
    """Alice holds apples and Bob pears — pronoun-only holder queries cannot pick an item."""
    p = PEFState(session_id="t")
    alice = Entity.create("Alice", 0, session_id="t")
    bob = Entity.create("Bob", 0, session_id="t")
    p.add_entity(alice)
    p.add_entity(bob)
    p.add_relationship(
        Relationship(
            subject_id=alice.id,
            relation="HAS",
            object_literal="5 apples",
            object_entity_id=None,
            span=Span.PRESENT,
            source_turn=1,
            evidence="fixture",
        )
    )
    p.add_relationship(
        Relationship(
            subject_id=bob.id,
            relation="HAS",
            object_literal="3 pears",
            object_entity_id=None,
            span=Span.PRESENT,
            source_turn=1,
            evidence="fixture",
        )
    )
    return p


def _pef_alice_transferred_book_to_bob() -> PEFState:
    """Alice NOT_HAS book, Bob HAS book — simulates a committed GIVE transfer."""
    p = PEFState(session_id="t")
    alice = Entity.create("Alice", 0, session_id="t")
    bob = Entity.create("Bob", 0, session_id="t")
    book = Entity.create("book", 0, session_id="t")
    p.add_entity(alice)
    p.add_entity(bob)
    p.add_entity(book)
    p.add_relationship(
        Relationship(
            subject_id=alice.id,
            relation="HAS",
            object_entity_id=book.id,
            object_literal=None,
            span=Span.PRESENT,
            source_turn=1,
            evidence="user",
            negated=True,
        )
    )
    p.add_relationship(
        Relationship(
            subject_id=bob.id,
            relation="HAS",
            object_entity_id=book.id,
            object_literal=None,
            span=Span.PRESENT,
            source_turn=1,
            evidence="user",
        )
    )
    return p


def _pef_alice_took_book_back_from_bob() -> PEFState:
    """Full give-then-take sequence:
    Turn 1: Alice NOT_HAS book, Bob HAS book  (Alice gave Bob a book)
    Turn 2: Alice HAS book, Bob NOT_HAS book  (Alice took the book back)
    """
    p = PEFState(session_id="t")
    alice = Entity.create("Alice", 0, session_id="t")
    bob = Entity.create("Bob", 0, session_id="t")
    book = Entity.create("book", 0, session_id="t")
    p.add_entity(alice)
    p.add_entity(bob)
    p.add_entity(book)
    # Turn 1 — GIVE projection
    p.add_relationship(Relationship(
        subject_id=alice.id, relation="HAS",
        object_entity_id=book.id, object_literal=None,
        span=Span.PRESENT, source_turn=1, evidence="user", negated=True,
    ))
    p.add_relationship(Relationship(
        subject_id=bob.id, relation="HAS",
        object_entity_id=book.id, object_literal=None,
        span=Span.PRESENT, source_turn=1, evidence="user",
    ))
    # Turn 2 — TAKE projection
    p.add_relationship(Relationship(
        subject_id=alice.id, relation="HAS",
        object_entity_id=book.id, object_literal=None,
        span=Span.PRESENT, source_turn=2, evidence="user",
    ))
    p.add_relationship(Relationship(
        subject_id=bob.id, relation="HAS",
        object_entity_id=book.id, object_literal=None,
        span=Span.PRESENT, source_turn=2, evidence="user", negated=True,
    ))
    return p


@pytest.mark.asyncio
async def test_state_native_handled_answer_one_location():
    adapter = _RecordingAdapter()
    pef = _pef_silver_key_at_safe_literal()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=pef,
        session_id="t",
    )
    r = await lens.process("Where is the silver key?")
    assert isinstance(r, LensResult)
    assert r.response == "silver key is at the safe."
    assert r.model == ""
    assert r.action == InterventionAction.PASS
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_state_native_handled_stop_no_at():
    adapter = _RecordingAdapter()
    pef = _pef_silver_key_no_at()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=pef,
        session_id="t",
    )
    r = await lens.process("Where is the silver key?")
    assert r.action == InterventionAction.HARD_STOP
    # Do not assert fixture literals (e.g. item names) in response text unless visible copy is the contract.
    assert r.decision is not None
    assert "state_native_stop:state_native_no_at:" in r.decision.rationale
    assert any(
        f.flag_type == FlagType.STATE_NATIVE_COMMITTED_STATE_STOP
        for f in r.decision.flags
    )
    assert any(
        getattr(f, "evidence", None) == "state_native_no_at" for f in r.decision.flags
    )
    assert r.decision.pathway_id == "P_STOP_REFUSE_CLEAN"
    assert r.decision.interaction_open is True
    assert lens.pef.epistemic_hold is not None
    assert lens.pef.epistemic_hold["mode"] == "stop"
    assert lens.pef.epistemic_hold.get("interaction_open") is True
    adapter.generate.assert_not_called()

    r2 = await lens.process("Why was that stopped?")
    assert "Previous request was stopped" not in r2.response
    assert "governed stop state" not in r2.response.lower()
    assert lens.pef.epistemic_hold is not None
    assert lens.pef.epistemic_hold.get("interaction_open") is True


@pytest.mark.asyncio
async def test_state_native_handled_clarify_ambiguous_entity():
    adapter = _RecordingAdapter()
    pef = _pef_two_keys_for_ambiguous()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=pef,
        session_id="t",
    )
    r = await lens.process("Where is the key?")
    assert r.action == InterventionAction.CONTAIN
    assert r.decision is not None
    assert any(
        f.flag_type == FlagType.STATE_NATIVE_ENTITY_AMBIGUITY
        for f in r.decision.flags
    )
    assert lens.pef.pending_clarification is not None
    assert lens.pef.pending_clarification.get("failed_constraint") == (
        "STATE_NATIVE_ENTITY_AMBIGUITY"
    )
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_state_native_checker_not_called_on_handled_path():
    adapter = _RecordingAdapter()
    pef = _pef_silver_key_at_safe_literal()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=True,
        ),
        initial_pef=pef,
        session_id="t",
    )
    with patch.object(lens._checker, "check", new_callable=AsyncMock) as chk:
        r = await lens.process("Where is the silver key?")
        assert r.response == "silver key is at the safe."
        chk.assert_not_called()
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_state_native_inventory_answer_what_does_x_have():
    adapter = _RecordingAdapter()
    pef = _pef_richard_inventory()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=pef,
        session_id="t",
    )
    r = await lens.process("What does Richard have?")
    assert r.action == InterventionAction.PASS
    assert r.response == "Richard has the gold key."
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_state_native_inventory_unknown_what_does_x_have_no_inventory():
    """Entity exists but no inventory evidence — HARD_STOP UNKNOWN; no LLM."""
    adapter = _RecordingAdapter()
    pef = _pef_richard_no_inventory()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=pef,
        session_id="t",
    )
    r = await lens.process("What does Richard have?")
    assert r.action == InterventionAction.HARD_STOP
    assert r.decision is not None
    assert "state_native_stop:state_native_no_inventory:" in r.decision.rationale
    adapter.generate.assert_not_called()

    eng = DefaultStateNativeEngine()
    sn = eng.evaluate(
        _state_native_inventory_request("What does Richard have?", pef),
    )
    assert sn.handled
    assert sn.epistemic_result == EpistemicResult.UNKNOWN
    assert sn.outcome == StateNativeOutcome.STOP
    assert sn.stop_reason_code == "state_native_no_inventory"


@pytest.mark.asyncio
async def test_state_native_inventory_stop_entity_not_in_pef():
    """Entity not in PEF at all — should still HARD_STOP."""
    adapter = _RecordingAdapter()
    pef = PEFState(session_id="t")  # completely empty
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=pef,
        session_id="t",
    )
    r = await lens.process("What does Richard have?")
    assert r.action == InterventionAction.HARD_STOP
    assert r.decision is not None
    assert "state_native_stop:state_native_unknown_entity:" in r.decision.rationale
    adapter.generate.assert_not_called()

    eng = DefaultStateNativeEngine()
    sn = eng.evaluate(
        _state_native_inventory_request("What does Richard have?", pef),
    )
    assert sn.handled
    assert sn.epistemic_result == EpistemicResult.UNKNOWN
    assert sn.outcome == StateNativeOutcome.STOP
    # Domain stop vocabulary (not bare "unknown_entity"):
    assert sn.stop_reason_code == "state_native_unknown_entity"


@pytest.mark.asyncio
async def test_state_native_inventory_clarify_ambiguous_subject():
    adapter = _RecordingAdapter()
    pef = _pef_ambiguous_richard_names()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=pef,
        session_id="t",
    )
    r = await lens.process("What does Richard have?")
    assert r.action == InterventionAction.CONTAIN
    assert r.decision is not None
    assert any(
        f.flag_type == FlagType.STATE_NATIVE_ENTITY_AMBIGUITY
        for f in r.decision.flags
    )
    assert lens.pef.pending_clarification is not None
    assert lens.pef.pending_clarification.get("failed_constraint") == (
        "STATE_NATIVE_ENTITY_AMBIGUITY"
    )
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_state_native_inventory_answer_who_has_y():
    adapter = _RecordingAdapter()
    pef = _pef_one_holder_for_gold_key()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=pef,
        session_id="t",
    )
    r = await lens.process("Who has the gold key?")
    assert r.action == InterventionAction.PASS
    assert r.response == "Richard has the gold key."
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_state_native_inventory_stop_who_has_y_no_holder():
    adapter = _RecordingAdapter()
    pef = _pef_no_holder_for_gold_key()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=pef,
        session_id="t",
    )
    r = await lens.process("Who has the gold key?")
    assert r.action == InterventionAction.HARD_STOP
    assert r.decision is not None
    assert "state_native_stop:state_native_no_holder:" in r.decision.rationale
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_state_native_stop_policy_projection_respects_authority_corridor():
    adapter = _RecordingAdapter()
    pef = _pef_no_holder_for_gold_key()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=pef,
        session_id="t",
    )
    tok_domain = domain_var.set("medical")
    tok_authority = authority_class_var.set("GP")
    try:
        r = await lens.process("Who has the gold key?")
    finally:
        domain_var.reset(tok_domain)
        authority_class_var.reset(tok_authority)

    assert r.action == InterventionAction.HARD_STOP
    assert r.decision is not None
    assert r.decision.pathway_id == "P_STOP_REFUSE_CLEAN"
    assert r.decision.interaction_open is True
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_state_native_stop_policy_projection_differs_by_authority():
    adapter = _RecordingAdapter()
    pef = _pef_no_holder_for_gold_key()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=pef,
        session_id="t",
    )
    tok_domain = domain_var.set("medical")
    tok_authority = authority_class_var.set("DA")
    try:
        r = await lens.process("Who has the gold key?")
    finally:
        domain_var.reset(tok_domain)
        authority_class_var.reset(tok_authority)

    assert r.action == InterventionAction.HARD_STOP
    assert r.decision is not None
    assert r.decision.pathway_id == "P_STOP_REFUSE_CLEAN"
    assert r.decision.interaction_open is True
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_state_native_inventory_clarify_who_has_y_multiple_holders():
    adapter = _RecordingAdapter()
    pef = _pef_multiple_holders_for_gold_key()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=pef,
        session_id="t",
    )
    r = await lens.process("Who has the gold key?")
    assert r.action == InterventionAction.PASS
    rl = r.response.lower()
    assert "emma" in rl and "richard" in rl and "gold key" in rl
    assert lens.pef.pending_clarification is None
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_state_native_inventory_who_has_pronoun_them_clarifies_item():
    """Pronoun item phrase is not a bounded key — clarify instead of STOP or bogus PASS."""
    adapter = _RecordingAdapter()
    pef = _pef_two_holders_two_distinct_inventory_literals()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=pef,
        session_id="t",
    )
    r = await lens.process("Who has them?")
    assert r.action == InterventionAction.CONTAIN
    assert lens.pef.pending_clarification is not None
    pc = lens.pef.pending_clarification or {}
    assert pc.get("state_native_ambiguity_kind") == "inventory_holder_query_unresolved_item"
    assert "pear" in r.response.lower() and "apple" in r.response.lower()
    adapter.generate.assert_not_called()


def test_parse_inventory_quantity_holder_phrase_surface() -> None:
    from aurora_lens.state_native_engine.parse.query_surface import (
        parse_inventory_quantity_holder_phrase,
    )

    assert parse_inventory_quantity_holder_phrase("Who has 4 lollipops?") == (4, "lollipops")
    assert parse_inventory_quantity_holder_phrase("Who has four lollipops?") == (4, "lollipops")
    assert parse_inventory_quantity_holder_phrase("Who has lollipops?") is None
    assert parse_inventory_quantity_holder_phrase("Who has 4 of them?") is None


@pytest.mark.asyncio
async def test_state_native_inventory_who_has_quantity_scopes_count_and_item():
    adapter = _RecordingAdapter()
    pef = _pef_alice_bob_distinct_lollipop_counts()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=pef,
        session_id="t",
    )
    r_qty = await lens.process("Who has 4 lollipops?")
    assert r_qty.action == InterventionAction.PASS
    rq = r_qty.response.lower()
    assert "alice" in rq and "bob" not in rq and "4 lollipops" in rq

    r_all = await lens.process("Who has lollipops?")
    assert r_all.action == InterventionAction.PASS
    ral = r_all.response.lower()
    assert "alice" in ral and "bob" in ral
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_state_native_inventory_who_has_quantity_one_singularizes_display():
    """Qty 1 + plural tail in the query → ``1 lollipop`` in the answer, not ``1 lollipops``."""
    adapter = _RecordingAdapter()
    pef = _pef_alice_one_lollipop()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=pef,
        session_id="t",
    )
    r = await lens.process("Who has one lollipops?")
    assert r.action == InterventionAction.PASS
    rl = r.response.lower()
    assert "1 lollipop" in rl
    assert "1 lollipops" not in rl
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_state_native_who_has_plural_glasses_uses_lexical_quantity_display():
    """Qty-led *and* plural-only *Who has* tails should show ``1 glass``, not ``the glasses``."""
    adapter = _RecordingAdapter()
    pef = _pef_chloe_one_glass_word()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=pef,
        session_id="t",
    )
    r1 = await lens.process("Who has one glasses?")
    assert r1.action == InterventionAction.PASS
    assert "Chloe has 1 glass".lower() in r1.response.lower()
    assert "the glasses".lower() not in r1.response.lower()

    r2 = await lens.process("Who has the glasses?")
    assert r2.action == InterventionAction.PASS
    assert "Chloe has 1 glass".lower() in r2.response.lower()
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_state_native_inventory_checker_not_called_on_handled_path():
    adapter = _RecordingAdapter()
    pef = _pef_richard_inventory()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=True,
        ),
        initial_pef=pef,
        session_id="t",
    )
    with patch.object(lens._checker, "check", new_callable=AsyncMock) as chk:
        r = await lens.process("What does Richard have?")
        assert r.response == "Richard has the gold key."
        chk.assert_not_called()
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_state_native_not_handled_falls_through_to_generate():
    adapter = _RecordingAdapter()
    pef = _pef_silver_key_at_safe_literal()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=pef,
        session_id="t",
    )
    r = await lens.process("What is the weather in Paris?")
    assert r.response == "MODEL_SHOULD_NOT_RUN"
    adapter.generate.assert_called_once()


@pytest.mark.asyncio
async def test_closed_world_three_box_puzzle_pre_llm_solution_bypasses_adapter():
    adapter = _RecordingAdapter()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=False,
            auto_verify=True,
        ),
        session_id="t",
    )
    prompt = (
        "There are three boxes in a row: a green box, a red box, and a purple box. "
        "Each has a statement on top. At least one statement is true, at least one is false, "
        "and only one box contains a prize. "
        "Green: 'The prize is in this box.' "
        "Red: 'This statement is of no help at all.' "
        "Purple: 'The prize is in the green box.' "
        "Which box has the prize? "
        "Answer with one lowercase word only: green, red, or purple."
    )
    r = await lens.process(prompt)
    assert r.action == InterventionAction.PASS
    assert r.response.strip().lower().rstrip(".") == "green"
    assert r.model == ""
    assert r.continuity_diagnostic == "state_native_closed_world_solved_unique"
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_state_native_disabled_calls_generate():
    adapter = _RecordingAdapter()
    pef = _pef_silver_key_at_safe_literal()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=False,
            auto_verify=False,
        ),
        initial_pef=pef,
        session_id="t",
    )
    await lens.process("Where is the silver key?")
    adapter.generate.assert_called_once()


@pytest.mark.asyncio
async def test_state_native_location_delegation_still_works():
    adapter = _RecordingAdapter()
    pef = _pef_silver_key_at_safe_literal()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=pef,
        session_id="t",
    )
    r = await lens.process("Where is the silver key?")
    assert r.action == InterventionAction.PASS
    assert r.response == "silver key is at the safe."
    adapter.generate.assert_not_called()


async def _state_native_stream_last_metadata(
    lens: Lens,
    user_input: str,
) -> dict | None:
    """Last ``metadata`` payload from ``process_stream`` (state-native short-circuit)."""
    last: dict | None = None
    async for kind, payload in lens.process_stream(user_input):
        if kind == "metadata":
            last = payload
    return last


def _entities_and_relationships_unchanged(before: dict, after: dict) -> bool:
    """Ignore turn counters; assert no new committed entity/relationship fact writes."""
    return before["entities"] == after["entities"] and before["relationships"] == after["relationships"]


@pytest.mark.asyncio
async def test_state_native_query_audit_parity_silver_key_location_and_no_pef_writes(
    tmp_path: Path,
) -> None:
    """``process`` vs ``process_stream`` JSONL: same material governance row (modulo stream flags).

    Location QUERY with committed ``AT`` literal: no new PEF fact writes (entities/relationships
    dicts stable aside from turn bookkeeping).
    """
    sync_a = tmp_path / "state_native_loc_sync.jsonl"
    stream_a = tmp_path / "state_native_loc_stream.jsonl"
    q = "Where is the silver key?"

    pef_sync = _pef_silver_key_at_safe_literal()
    snap_pre = copy.deepcopy(pef_sync.to_dict())
    lens_sync = Lens(
        LensConfig(
            adapter=_RecordingAdapter(),
            extraction_backend=_EmptyExtractBackend(),
            governance_bridge=BuiltinBridge(audit_path=str(sync_a)),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=pef_sync,
        session_id="t",
    )
    r_sync = await lens_sync.process(q)
    post_sync = lens_sync.pef.to_dict()
    assert r_sync.action == InterventionAction.PASS
    assert _entities_and_relationships_unchanged(
        {k: snap_pre[k] for k in ("entities", "relationships")},
        {k: post_sync[k] for k in ("entities", "relationships")},
    )

    pef_stream = _pef_silver_key_at_safe_literal()
    snap_pre_s = copy.deepcopy(pef_stream.to_dict())
    lens_stream = Lens(
        LensConfig(
            adapter=_RecordingAdapter(),
            extraction_backend=_EmptyExtractBackend(),
            governance_bridge=BuiltinBridge(audit_path=str(stream_a)),
            enable_state_native_delegation=True,
            auto_verify=False,
            stream_emit_progress=False,
        ),
        initial_pef=pef_stream,
        session_id="t",
    )
    meta = await _state_native_stream_last_metadata(lens_stream, q)
    post_s = lens_stream.pef.to_dict()
    assert meta is not None
    assert meta.get("continuity_diagnostic") == "state_native_committed_location_read"
    assert _entities_and_relationships_unchanged(
        {k: snap_pre_s[k] for k in ("entities", "relationships")},
        {k: post_s[k] for k in ("entities", "relationships")},
    )

    row_s = _read_last_jsonl_object(sync_a)
    row_t = _read_last_jsonl_object(stream_a)
    assert row_s["outcome"] == "PASS" and row_t["outcome"] == "PASS"
    assert row_s.get("state_native_handled") is True
    assert row_t.get("state_native_handled") is True
    assert row_s["stream"] is False and row_s["stream_completed"] is True
    assert row_t["stream"] is True and row_t["stream_completed"] is True
    assert row_s.get("failed_constraints") == []
    assert row_t.get("failed_constraints") == []
    assert _governance_material_audit_row(row_s) == _governance_material_audit_row(row_t)


@pytest.mark.asyncio
async def test_state_native_query_clarify_pending_audit_parity_process_vs_stream(
    tmp_path: Path,
) -> None:
    """Ambiguous key (location): material JSONL audit parity + ``failed_constraints`` in row."""
    from aurora_lens.verify.flags import FlagType

    q = "Where is the key?"
    sync_a = tmp_path / "state_native_clarify_sync.jsonl"
    stream_a = tmp_path / "state_native_clarify_stream.jsonl"

    pef_s = _pef_two_keys_for_ambiguous()
    snap_s = copy.deepcopy(pef_s.to_dict())
    lens_sync = Lens(
        LensConfig(
            adapter=_RecordingAdapter(),
            extraction_backend=_EmptyExtractBackend(),
            governance_bridge=BuiltinBridge(audit_path=str(sync_a)),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=pef_s,
        session_id="t",
    )
    r_sync = await lens_sync.process(q)
    assert r_sync.action == InterventionAction.CONTAIN
    assert r_sync.response == "Which did you mean: gold key, silver key?"
    assert lens_sync.pef.pending_clarification is not None
    fc = lens_sync.pef.pending_clarification.get("failed_constraint")
    assert fc == "STATE_NATIVE_ENTITY_AMBIGUITY"
    assert _entities_and_relationships_unchanged(
        {k: snap_s[k] for k in ("entities", "relationships")},
        {k: lens_sync.pef.to_dict()[k] for k in ("entities", "relationships")},
    )

    pef_t = _pef_two_keys_for_ambiguous()
    snap_t = copy.deepcopy(pef_t.to_dict())
    lens_stream = Lens(
        LensConfig(
            adapter=_RecordingAdapter(),
            extraction_backend=_EmptyExtractBackend(),
            governance_bridge=BuiltinBridge(audit_path=str(stream_a)),
            enable_state_native_delegation=True,
            auto_verify=False,
            stream_emit_progress=False,
        ),
        initial_pef=pef_t,
        session_id="t",
    )
    meta = await _state_native_stream_last_metadata(lens_stream, q)
    assert meta is not None
    assert lens_stream.pef.pending_clarification is not None
    assert lens_stream.pef.pending_clarification.get("failed_constraint") == fc
    assert _entities_and_relationships_unchanged(
        {k: snap_t[k] for k in ("entities", "relationships")},
        {k: lens_stream.pef.to_dict()[k] for k in ("entities", "relationships")},
    )

    row_s = _read_last_jsonl_object(sync_a)
    row_t = _read_last_jsonl_object(stream_a)
    assert row_s["outcome"] == "CONTAIN" and row_t["outcome"] == "CONTAIN"
    assert row_s.get("state_native_handled") is True
    assert row_t.get("state_native_handled") is True
    assert row_s.get("failed_constraints") == [FlagType.STATE_NATIVE_ENTITY_AMBIGUITY.name]
    assert row_t.get("failed_constraints") == [FlagType.STATE_NATIVE_ENTITY_AMBIGUITY.name]
    fe_s = row_s.get("forensic_event") or {}
    fe_t = row_t.get("forensic_event") or {}
    assert fe_s.get("status") == "ASK" and fe_t.get("status") == "ASK"
    assert fe_s.get("failed_constraints") == fe_t.get("failed_constraints") == [
        FlagType.STATE_NATIVE_ENTITY_AMBIGUITY.name
    ]
    assert row_s["stream"] is False
    assert row_t["stream"] is True
    assert _governance_material_audit_row(row_s) == _governance_material_audit_row(row_t)


@pytest.mark.asyncio
async def test_state_native_inventory_answer_transferred_item_giver_query():
    """After Alice NOT_HAS book / Bob HAS book, querying Alice yields denial + holder."""
    adapter = _RecordingAdapter()
    pef = _pef_alice_transferred_book_to_bob()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=pef,
        session_id="t",
    )
    r = await lens.process("What does Alice have?")
    assert r.action == InterventionAction.PASS
    assert "Alice does not have the book" in r.response
    assert "Bob has it" in r.response
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_state_native_inventory_answer_transferred_item_recipient_query():
    """After Alice NOT_HAS book / Bob HAS book, querying Bob yields positive answer."""
    adapter = _RecordingAdapter()
    pef = _pef_alice_transferred_book_to_bob()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=pef,
        session_id="t",
    )
    r = await lens.process("What does Bob have?")
    assert r.action == InterventionAction.PASS
    assert "Bob has the book" in r.response
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_state_native_inventory_take_back_alice_has_book():
    """After give (turn 1) then take-back (turn 2): Alice has book, later turn wins."""
    adapter = _RecordingAdapter()
    pef = _pef_alice_took_book_back_from_bob()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=pef,
        session_id="t",
    )
    r = await lens.process("What does Alice have?")
    assert r.action == InterventionAction.PASS
    assert "Alice has the book" in r.response
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_state_native_inventory_take_back_bob_does_not_have_book():
    """After give (turn 1) then take-back (turn 2): Bob does not have book."""
    adapter = _RecordingAdapter()
    pef = _pef_alice_took_book_back_from_bob()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=pef,
        session_id="t",
    )
    r = await lens.process("What does Bob have?")
    assert r.action == InterventionAction.PASS
    assert "Bob does not have the book" in r.response
    assert "Alice has it" in r.response
    adapter.generate.assert_not_called()


# ── Action-agent follow-up regressions ───────────────────────────────────────
# Invariant: resolved binding beats role/title heuristic.
# Emma ordered the medication change (resolved from "She").
# Follow-up attribution must return Emma, not a co-present entity with a title.


def _pef_emma_ordered_medication_change(co_entity_name: str) -> PEFState:
    """PEF after resolving 'She' → Emma: Emma ORDERED medication_change.
    Co-entity is added to represent the entity that should NOT be selected.
    """
    p = PEFState(session_id="t")
    emma = Entity.create("Emma", 0, session_id="t")
    co = Entity.create(co_entity_name, 0, session_id="t")
    p.add_entity(emma)
    p.add_entity(co)
    p.add_relationship(
        Relationship(
            subject_id=emma.id,
            relation="ORDERED",
            object_entity_id=None,
            object_literal="medication change",
            span=Span.PAST,
            source_turn=1,
            evidence="user",
        )
    )
    return p


@pytest.mark.asyncio
async def test_action_agent_follow_up_plain_co_entity():
    """Emma ordered; co-entity Vicky has no title. Follow up → Emma, not Vicky."""
    adapter = _RecordingAdapter()
    pef = _pef_emma_ordered_medication_change("Vicky")
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=pef,
        session_id="t",
    )
    r = await lens.process("Which clinician should I follow up with?")
    assert r.action == InterventionAction.PASS
    assert "Emma" in r.response
    assert "Vicky" not in r.response
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_action_agent_follow_up_dr_title_does_not_override():
    """Resolved binding beats professional title.

    Emma ordered the medication change.
    Dr. Patel is co-present with a 'Dr.' prefix.
    Follow-up target must be Emma — title authority must not substitute the resolved agent.
    """
    adapter = _RecordingAdapter()
    pef = _pef_emma_ordered_medication_change("Dr. Patel")
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=pef,
        session_id="t",
    )
    r = await lens.process("Which clinician should I follow up with?")
    assert r.action == InterventionAction.PASS
    assert "Emma" in r.response
    assert "Dr. Patel" not in r.response
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_action_agent_follow_up_who_query():
    """'Who should I follow up with?' also routes to action-agent handler."""
    adapter = _RecordingAdapter()
    pef = _pef_emma_ordered_medication_change("Dr. Patel")
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=pef,
        session_id="t",
    )
    r = await lens.process("Who should I follow up with?")
    assert r.action == InterventionAction.PASS
    assert "Emma" in r.response
    assert "Dr. Patel" not in r.response
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_action_agent_no_actions_passes_through():
    """No action relations in PEF → state-native passes through to LLM."""
    adapter = _RecordingAdapter()
    pef = PEFState(session_id="t")
    emma = Entity.create("Emma", 0, session_id="t")
    pef.add_entity(emma)
    # Only a HAS relation — not an action
    pef.add_relationship(
        Relationship(
            subject_id=emma.id,
            relation="HAS",
            object_entity_id=None,
            object_literal="key",
            span=Span.PRESENT,
            source_turn=1,
            evidence="user",
        )
    )
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=pef,
        session_id="t",
    )
    r = await lens.process("Who should I follow up with?")
    # No action agents → LLM called
    adapter.generate.assert_called_once()


# ── Temporal-contact governed-unknown regressions ────────────────────────────
# Invariant: if the entity is in PEF, timing questions must not reach the LLM.
# The LLM has no temporal policy and will generate outside the verified world.


def _pef_latisha_on_way_home() -> PEFState:
    p = PEFState(session_id="t")
    latisha = Entity.create("Latisha", 0, session_id="t")
    eleanor = Entity.create("Eleanor", 0, session_id="t")
    p.add_entity(latisha)
    p.add_entity(eleanor)
    p.add_relationship(
        Relationship(
            subject_id=latisha.id,
            relation="AT",
            object_entity_id=None,
            object_literal="way home",
            span=Span.PRESENT,
            source_turn=1,
            evidence="user",
        )
    )
    p.add_relationship(
        Relationship(
            subject_id=eleanor.id,
            relation="TELL",
            object_entity_id=None,
            object_literal="me",
            span=Span.PAST,
            source_turn=1,
            evidence="user",
        )
    )
    return p


@pytest.mark.asyncio
async def test_temporal_contact_governed_unknown_entity_in_pef():
    """Temporal contact when entity is in PEF but temporal anchor missing: CONTAIN scope ask, no LLM."""
    adapter = _RecordingAdapter()
    pef = _pef_latisha_on_way_home()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=pef,
        session_id="t",
    )
    r = await lens.process("When should I contact Latisha?")
    assert r.action == InterventionAction.CONTAIN
    assert r.decision is not None
    assert any(
        f.flag_type == FlagType.STATE_NATIVE_TEMPORAL_ANCHOR_MISSING
        for f in r.decision.flags
    )
    assert "Latisha" in r.response
    assert "governing temporal anchor" in r.response.lower()
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_temporal_contact_governed_unknown_entity_not_in_pef_passes_through():
    """'When should I contact Sarah?' with Sarah NOT in PEF → pass through to LLM."""
    adapter = _RecordingAdapter()
    pef = _pef_latisha_on_way_home()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=pef,
        session_id="t",
    )
    r = await lens.process("When should I contact Sarah?")
    # Sarah not in PEF → state-native passes through → LLM called
    adapter.generate.assert_called_once()


@pytest.mark.asyncio
async def test_temporal_contact_governed_unknown_in_multi_sentence_replay():
    """Last-sentence detection: multi-sentence message ending with temporal query."""
    adapter = _RecordingAdapter()
    pef = _pef_latisha_on_way_home()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=pef,
        session_id="t",
    )
    r = await lens.process(
        "Latisha and Eleanor are listed on the case. When should I contact Latisha?"
    )
    assert r.action == InterventionAction.CONTAIN
    assert r.decision is not None
    assert any(
        f.flag_type == FlagType.STATE_NATIVE_TEMPORAL_ANCHOR_MISSING
        for f in r.decision.flags
    )
    assert "Latisha" in r.response
    assert "governing temporal anchor" in r.response.lower()
    adapter.generate.assert_not_called()


def _pef_alice_aloce_chloe_ambiguous_book_holders() -> PEFState:
    p = PEFState(session_id="t")
    alice = Entity.create("Alice", 0, session_id="t")
    aloce = Entity.create("Aloce", 0, session_id="t")
    chloe = Entity.create("Chloe", 0, session_id="t")
    p.add_entity(alice)
    p.add_entity(aloce)
    p.add_entity(chloe)
    for sid in (aloce.id, chloe.id):
        p.add_relationship(
            Relationship(
                subject_id=sid,
                relation="HAS",
                object_entity_id=None,
                object_literal="book",
                span=Span.PRESENT,
                source_turn=1,
                evidence="user",
            )
        )
    return p


def _pef_zara_bob_ambiguous_book_holders() -> PEFState:
    p = PEFState(session_id="t")
    zara = Entity.create("Zara", 0, session_id="t")
    bob = Entity.create("Bob", 0, session_id="t")
    p.add_entity(zara)
    p.add_entity(bob)
    for sid in (zara.id, bob.id):
        p.add_relationship(
            Relationship(
                subject_id=sid,
                relation="HAS",
                object_entity_id=None,
                object_literal="book",
                span=Span.PRESENT,
                source_turn=1,
                evidence="user",
            )
        )
    return p


def _pef_alice_bob_distinct_lollipop_counts() -> PEFState:
    """Alice has 4 lollipops, Bob has 10 — quantity-scoped *Who has* must not conflate with item-only."""
    p = PEFState(session_id="t")
    alice = Entity.create("Alice", 0, session_id="t")
    bob = Entity.create("Bob", 0, session_id="t")
    p.add_entity(alice)
    p.add_entity(bob)
    p.add_relationship(
        Relationship(
            subject_id=alice.id,
            relation="HAS",
            object_entity_id=None,
            object_literal="4 lollipops",
            span=Span.PRESENT,
            source_turn=1,
            evidence="fixture",
        )
    )
    p.add_relationship(
        Relationship(
            subject_id=bob.id,
            relation="HAS",
            object_entity_id=None,
            object_literal="10 lollipops",
            span=Span.PRESENT,
            source_turn=1,
            evidence="fixture",
        )
    )
    return p


def _pef_alice_one_lollipop() -> PEFState:
    p = PEFState(session_id="t")
    alice = Entity.create("Alice", 0, session_id="t")
    p.add_entity(alice)
    p.add_relationship(
        Relationship(
            subject_id=alice.id,
            relation="HAS",
            object_entity_id=None,
            object_literal="1 lollipop",
            span=Span.PRESENT,
            source_turn=1,
            evidence="fixture",
        )
    )
    return p


def _pef_chloe_one_glass_word() -> PEFState:
    """Chloe holds counted ``HAS`` ``one glass`` (word quantity + singular head)."""
    p = PEFState(session_id="t")
    chloe = Entity.create("Chloe", 0, session_id="t")
    p.add_entity(chloe)
    p.add_relationship(
        Relationship(
            subject_id=chloe.id,
            relation="HAS",
            object_entity_id=None,
            object_literal="one glass",
            span=Span.PRESENT,
            source_turn=1,
            evidence="fixture",
        )
    )
    return p


def _pef_alice_aloce_alic_ambiguous_book_holders() -> PEFState:
    p = PEFState(session_id="t")
    alice = Entity.create("Alice", 0, session_id="t")
    aloce = Entity.create("Aloce", 0, session_id="t")
    alic = Entity.create("Alic", 0, session_id="t")
    p.add_entity(alice)
    p.add_entity(aloce)
    p.add_entity(alic)
    for sid in (aloce.id, alic.id):
        p.add_relationship(
            Relationship(
                subject_id=sid,
                relation="HAS",
                object_entity_id=None,
                object_literal="book",
                span=Span.PRESENT,
                source_turn=1,
                evidence="user",
            )
        )
    return p


@pytest.mark.asyncio
async def test_state_native_holder_typo_recovery_clarify_then_confirm_merge(tmp_path: Path) -> None:
    adapter = _RecordingAdapter()
    audit = tmp_path / "typo_merge_audit.jsonl"
    backend = _CountingEmptyExtractBackend()
    pef = _pef_alice_aloce_chloe_ambiguous_book_holders()
    snap_after_q = copy.deepcopy(pef.to_dict())
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=backend,
            governance_bridge=BuiltinBridge(audit_path=str(audit)),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=pef,
        session_id="t",
    )
    # Subject phrase "lo" matches Aloce / Chloe via substring — not Alice.
    # Typo-recovery: among {Aloce, Chloe} only Aloce is a plausible typo of Alice.
    r1 = await lens.process("What does lo have?")
    assert r1.action == InterventionAction.CONTAIN
    assert "Did you mean Alice instead of Aloce?" in r1.response
    pc = lens.pef.pending_clarification or {}
    assert pc.get("typo_recovery") is not None
    assert lens.pef.identity_corrections == []
    assert len([e for e in lens.pef.entities.values() if e.name == "Aloce"]) == 1

    extract_after_q1 = backend.extract_calls

    r2 = await lens.process("yes")
    assert backend.extract_calls == extract_after_q1
    assert r2.action == InterventionAction.PASS
    assert "Chloe" in r2.response
    assert "Aloce" not in r2.response
    assert not any(e.name == "Aloce" for e in lens.pef.entities.values())
    assert len(lens.pef.identity_corrections) == 1
    ic = lens.pef.identity_corrections[0]
    assert ic.get("reason_type") == "deterministic_typo_confirmation"
    assert ic.get("original_ambiguity_flag") == "STATE_NATIVE_ENTITY_AMBIGUITY"
    adapter.generate.assert_not_called()
    assert audit.stat().st_size > 0

    post_entities = lens.pef.to_dict()["entities"]
    assert post_entities != snap_after_q["entities"]
    alice_ent = next(e for e in lens.pef.entities.values() if e.name == "Alice")
    assert any(
        r.subject_id == alice_ent.id
        and r.relation == "HAS"
        and r.object_literal == "book"
        and not r.negated
        for r in lens.pef.relationships
    )


@pytest.mark.asyncio
async def test_state_native_holder_typo_recovery_explicit_canonical_name(tmp_path: Path) -> None:
    adapter = _RecordingAdapter()
    backend = _EmptyExtractBackend()
    pef = _pef_alice_aloce_chloe_ambiguous_book_holders()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=backend,
            governance_bridge=BuiltinBridge(audit_path=str(tmp_path / "a.jsonl")),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=pef,
        session_id="t",
    )
    await lens.process("What does lo have?")
    r = await lens.process("Alice")
    assert r.action == InterventionAction.PASS
    assert "the book" in r.response.lower()
    assert not any(e.name == "Aloce" for e in lens.pef.entities.values())
    alice_ent = next(e for e in lens.pef.entities.values() if e.name == "Alice")
    assert any(
        r.subject_id == alice_ent.id
        and r.relation == "HAS"
        and r.object_literal == "book"
        and not r.negated
        for r in lens.pef.relationships
    )
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_state_native_holder_typo_recovery_no_merge_before_confirm() -> None:
    adapter = _RecordingAdapter()
    backend = _EmptyExtractBackend()
    pef = _pef_alice_aloce_chloe_ambiguous_book_holders()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=backend,
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=pef,
        session_id="t",
    )
    await lens.process("What does lo have?")
    assert any(e.name == "Aloce" for e in lens.pef.entities.values())
    assert lens.pef.identity_corrections == []
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_state_native_holder_ambiguity_two_typo_singletons_no_typo_recovery() -> None:
    adapter = _RecordingAdapter()
    backend = _EmptyExtractBackend()
    pef = _pef_alice_aloce_alic_ambiguous_book_holders()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=backend,
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=pef,
        session_id="t",
    )
    # "a" substring-matches Alice / Aloce / Alic — two typo singletons → no typo_recovery.
    r = await lens.process("What does a have?")
    assert r.action == InterventionAction.CONTAIN
    assert "Did you mean" not in r.response
    pc = lens.pef.pending_clarification or {}
    assert pc.get("typo_recovery") is None
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_state_native_holder_ambiguity_unrelated_names_no_typo_recovery() -> None:
    adapter = _RecordingAdapter()
    backend = _EmptyExtractBackend()
    pef = _pef_zara_bob_ambiguous_book_holders()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=backend,
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=pef,
        session_id="t",
    )
    r = await lens.process("Who has the book?")
    assert r.action == InterventionAction.PASS
    rl = r.response.lower()
    assert "bob" in rl and "zara" in rl
    assert lens.pef.pending_clarification is None
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_state_native_location_ambiguity_two_keys_no_typo_recovery() -> None:
    adapter = _RecordingAdapter()
    backend = _EmptyExtractBackend()
    pef = _pef_two_keys_for_ambiguous()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=backend,
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=pef,
        session_id="t",
    )
    r = await lens.process("Where is the key?")
    assert r.action == InterventionAction.CONTAIN
    assert (lens.pef.pending_clarification or {}).get("typo_recovery") is None
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_state_native_holder_yes_does_not_resolve_without_typo_pending() -> None:
    adapter = _RecordingAdapter()
    backend = _EmptyExtractBackend()
    pef = _pef_two_keys_for_ambiguous()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=backend,
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=pef,
        session_id="t",
    )
    await lens.process("Where is the key?")
    await lens.process("yes")
    assert lens.pef.pending_clarification is not None

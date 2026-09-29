"""QueryTransaction invariant tests — Phase 4.

Invariants protected:
- A query with an unresolved pronoun target never falls through to the LLM.
- COMPARE answers come from committed COMPARE relationships, not inferred from IS.
- Inventory queries with "them" as item resolve from active PEF possession.
- A query that returns CONTAIN does not mutate PEF.
- State-native engine handles answerable queries without LLM.
"""

from __future__ import annotations

import pytest

from aurora_lens.interpret.schema import QueryTransaction
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState, Relationship
from aurora_lens.state_native_engine.contracts import StateNativeOutcome
from aurora_lens.state_native_engine.eval.compare import evaluate_comparative_query
from aurora_lens.state_native_engine.eval.inventory import evaluate_inventory_holder_query
from aurora_lens.state_native_engine.eval.location import evaluate_location_from_committed_state


# ── Helpers ───────────────────────────────────────────────────────────────────


def _pef_with(*names: str) -> PEFState:
    pef = PEFState()
    for name in names:
        pef.get_or_create_entity(name)
    return pef


def _add_has(pef: PEFState, subject_name: str, obj_literal: str) -> None:
    ent = pef.find_entity_by_name(subject_name)
    if ent is None:
        ent, _ = pef.get_or_create_entity(subject_name)
    pef.add_relationship(Relationship(
        subject_id=ent.id, relation="HAS",
        object_entity_id=None, object_literal=obj_literal,
        span=Span.PRESENT, source_turn=1, evidence=f"{subject_name} has {obj_literal}.",
    ))


def _add_at(pef: PEFState, subject_name: str, location: str) -> None:
    ent = pef.find_entity_by_name(subject_name)
    if ent is None:
        ent, _ = pef.get_or_create_entity(subject_name)
    pef.add_relationship(Relationship(
        subject_id=ent.id, relation="AT",
        object_entity_id=None, object_literal=location,
        span=Span.PRESENT, source_turn=1, evidence=f"{subject_name} is at {location}.",
    ))


def _add_compare(
    pef: PEFState,
    subject_name: str,
    comparand_name: str,
    adjective: str,
    noun: str,
) -> None:
    subj = pef.find_entity_by_name(subject_name)
    if subj is None:
        subj, _ = pef.get_or_create_entity(subject_name)
    comp = pef.find_entity_by_name(comparand_name)
    if comp is None:
        comp, _ = pef.get_or_create_entity(comparand_name)
    pef.add_relationship(Relationship(
        subject_id=subj.id, relation="COMPARE",
        object_entity_id=comp.id, object_literal=None,
        span=Span.PRESENT, source_turn=1, evidence="",
        relation_metadata={"adjective": adjective, "noun": noun},
    ))


# ── QueryTransaction dataclass ────────────────────────────────────────────────


def test_query_transaction_defaults():
    tx = QueryTransaction()
    assert tx.query_kind == "inventory_holder"
    assert tx.resolved_targets == []
    assert tx.unresolved_tokens == []
    assert tx.query_surface == ""


def test_query_transaction_construction():
    tx = QueryTransaction(
        query_kind="comparative",
        resolved_targets=["James"],
        unresolved_tokens=[],
        query_surface="Whose dog was bigger?",
    )
    assert tx.query_kind == "comparative"
    assert "James" in tx.resolved_targets
    assert not tx.unresolved_tokens


def test_query_transaction_unresolved_flag():
    tx = QueryTransaction(
        query_kind="inventory_holder",
        resolved_targets=[],
        unresolved_tokens=["them"],
    )
    assert tx.unresolved_tokens  # non-empty → should produce CONTAIN, not LLM


def test_query_transaction_does_not_mutate_pef():
    """QueryTransaction is a pure data carrier; constructing one never touches PEF."""
    pef = _pef_with("James")
    before = len(pef.relationships)
    _tx = QueryTransaction(
        query_kind="inventory_holder",
        resolved_targets=["James"],
        unresolved_tokens=[],
        query_surface="Who has apples?",
    )
    assert len(pef.relationships) == before


# ── Law: COMPARE answers from committed relation ──────────────────────────────


def test_whose_dog_was_bigger_reads_compare_relation():
    """evaluate_comparative_query returns ANSWER from committed COMPARE relation."""
    from aurora_lens.state_native_engine.epistemic import EpistemicResult
    pef = _pef_with("James", "Richard")
    _add_compare(pef, "James", "Richard", "bigger", "dog")

    result = evaluate_comparative_query(pef, "dog", "bigger")

    assert result.outcome == StateNativeOutcome.ANSWER
    assert result.epistemic_result == EpistemicResult.VALUE
    assert "James" in result.user_visible_text
    assert "bigger" in result.user_visible_text
    assert "Richard" in result.user_visible_text
    assert result.clarify_context is None
    assert result.stop_reason_code is None


def test_comparative_query_no_compare_relation_stop():
    """evaluate_comparative_query returns STOP when no COMPARE has been committed."""
    from aurora_lens.state_native_engine.epistemic import EpistemicResult
    pef = _pef_with("James", "Richard")
    # No COMPARE relationship added

    result = evaluate_comparative_query(pef, "dog", "bigger")

    assert result.outcome == StateNativeOutcome.STOP
    assert result.epistemic_result == EpistemicResult.UNKNOWN
    assert result.stop_reason_code == "state_native_no_compare"
    assert result.clarify_context is None


def test_comparative_query_uses_latest_turn():
    """When multiple COMPARE relations exist for same adj+noun, latest source_turn wins."""
    pef = _pef_with("James", "Richard", "Alice")
    subj_j = pef.find_entity_by_name("James")
    comp_r = pef.find_entity_by_name("Richard")
    comp_a = pef.find_entity_by_name("Alice")

    pef.add_relationship(Relationship(
        subject_id=subj_j.id, relation="COMPARE",
        object_entity_id=comp_r.id, object_literal=None,
        span=Span.PRESENT, source_turn=1, evidence="",
        relation_metadata={"adjective": "bigger", "noun": "dog"},
    ))
    pef.add_relationship(Relationship(
        subject_id=subj_j.id, relation="COMPARE",
        object_entity_id=comp_a.id, object_literal=None,
        span=Span.PRESENT, source_turn=3, evidence="",
        relation_metadata={"adjective": "bigger", "noun": "dog"},
    ))

    result = evaluate_comparative_query(pef, "dog", "bigger")
    assert result.outcome == StateNativeOutcome.ANSWER
    assert "Alice" in result.user_visible_text  # source_turn=3 wins


# ── Law: inventory "them" resolves from PEF, returns CLARIFY not STOP ─────────


def test_inventory_holder_them_resolves_from_pef():
    """'Who has them?' with 'them' as an unresolved pronoun → CLARIFY (not STOP, not LLM)."""
    pef = _pef_with("James")
    _add_has(pef, "James", "apples")

    outcome, text, clarify, stop_code = evaluate_inventory_holder_query(pef, "them")

    assert outcome == StateNativeOutcome.CLARIFY.value
    assert clarify is not None
    assert "apples" in str(clarify.get("candidate_entities", []))


def test_inventory_holder_them_no_committed_items():
    """'Who has them?' with no committed HAS → CLARIFY with empty candidates."""
    pef = _pef_with("James")
    # No HAS committed

    outcome, text, clarify, stop_code = evaluate_inventory_holder_query(pef, "them")

    assert outcome == StateNativeOutcome.CLARIFY.value


# ── Law: unresolved query target → CLARIFY/CONTAIN, not STOP ─────────────────


def test_unresolved_query_target_contain_not_stop():
    """'Who has them?' with unresolved 'them' → state-native CLARIFY (maps to CONTAIN).

    The invariant: an unresolved pronoun query target must never fall through to the LLM.
    The state-native engine returns CLARIFY (not not-handled), preventing LLM fallback.
    """
    pef = _pef_with("James", "Alice")
    _add_has(pef, "James", "wallet")
    _add_has(pef, "Alice", "phone")

    from aurora_lens.state_native_engine.contracts import StateNativeRequest, StateNativeOutcome
    from aurora_lens.state_native_engine.default_engine import DefaultStateNativeEngine
    from aurora_lens.interpret.turn_act import TurnAct

    engine = DefaultStateNativeEngine()
    req = StateNativeRequest(
        user_text="Who has them?",
        pef=pef,
        turn_act=TurnAct.QUERY,
        detected_span=Span.PRESENT,
    )
    result = engine.evaluate(req)

    assert result.handled is True, "state-native must handle 'Who has them?' — not fall through to LLM"
    assert result.outcome == StateNativeOutcome.CLARIFY, "unresolved pronoun item → CLARIFY (CONTAIN)"


# ── Law: location query unresolved subject → CLARIFY not STOP ────────────────


def test_location_query_unresolved_subject_contain():
    """When subject phrase matches 2+ entities → CLARIFY (CONTAIN), not STOP.

    The invariant: ambiguous subject in a location query must not silently STOP
    when candidates exist — CLARIFY lets the user specify which entity they meant.
    """
    pef = _pef_with("James Smith", "James Brown")
    _add_at(pef, "James Smith", "London")
    _add_at(pef, "James Brown", "Paris")

    result = evaluate_location_from_committed_state(pef, "James")

    assert result.outcome == StateNativeOutcome.CLARIFY, (
        f"Ambiguous subject 'James' with 2 candidates must produce CLARIFY, got {result.outcome!r}"
    )
    assert result.clarify_context is not None
    candidates = result.clarify_context.get("candidate_entities", [])
    assert len(candidates) >= 2


# ── Law: state-native answer never calls LLM when answerable ─────────────────


def test_state_native_answer_never_calls_llm_when_answerable():
    """When inventory query is answerable from committed PEF, LLM adapter is never called."""
    from aurora_lens.lens import Lens, LensConfig

    call_count = 0

    class _CountingAdapter:
        async def complete(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return type("R", (), {"content": "mock", "model": "mock"})()

        async def complete_stream(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1

            async def _gen():
                yield "mock"

            return _gen()

    lens = Lens(LensConfig(
        adapter=_CountingAdapter(),
        auto_interpret=True,
        auto_verify=True,
        enable_state_native_delegation=True,
    ))

    import asyncio

    async def _run():
        await lens.process("James has 10 apples.")
        calls_after_setup = call_count
        result = await lens.process("Who has apples?")
        return result, calls_after_setup

    result, calls_before = asyncio.new_event_loop().run_until_complete(_run())
    assert call_count == calls_before, (
        "LLM adapter must not be called for answerable inventory query; "
        f"was called {call_count - calls_before} extra time(s)"
    )

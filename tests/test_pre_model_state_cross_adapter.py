"""Cross-adapter regression for the universal pre-model router.

**Framing (invariant, not story)**

Entity names and object literals in the strings below are **arbitrary placeholders**.
They are not part of the contract. The **invariant under test** is: **cross-turn state
mutation** (interpretation → PEF) and **state-native retrieval** from committed session
state—**not** the narrative content.

---

**Contract**

- :func:`~aurora_lens.pre_model_state.dispatch.pre_model_state_dispatch` runs on every
  turn; ``request_domain`` / :data:`~aurora_lens.context.domain_var` does not enable or
  disable it.

- Cross-turn **transfer / quantity / holder** arithmetic (committed PEF + state-native
  QUERY surfaces) is handled by **PossessionStateAdapter** via
  :meth:`aurora_lens.lens.Lens._finish_turn_with_state_native_if_handled`.

- **ArithmeticClosedWorldAdapter** is a separate slot for the bounded three-box puzzle
  (``_solve_three_box_prize_puzzle``); it is **not** transfer arithmetic. Both that puzzle
  **and** possession/transfer continuity must keep working.

Placeholder turns (e.g. Jack / Jill / lollipops) commit state via the main interpretation path;
``PossessionStateAdapter`` answers structured inventory queries from committed PEF **early**
in the same pipeline as medical continuity, without ``request_domain="medical"``.
"""

from __future__ import annotations

import pytest

from aurora_lens.adapters.base import AdapterResponse, LLMAdapter
from aurora_lens.config import LensConfig
from aurora_lens.context import domain_var
from aurora_lens.govern.decision import InterventionAction
from aurora_lens.lens import Lens
from aurora_lens.pre_model_state.adapters import (
    ArithmeticClosedWorldAdapter,
    PossessionStateAdapter,
)
from aurora_lens.pre_model_state.dispatch import default_pre_model_adapters
from aurora_lens.proxy.openai_compat import format_chat_response

# Fixed template for the bounded puzzle solver — story tokens are irrelevant to the invariant.
THREE_BOX_PROMPT = (
    "There are three boxes in a row: a green box, a red box, and a purple box. "
    "Each has a statement on top. At least one statement is true, at least one is false, "
    "and only one box contains a prize. "
    "Green: 'The prize is in this box.' "
    "Red: 'This statement is of no help at all.' "
    "Purple: 'The prize is in the green box.' "
    "Which box has the prize? "
    "Answer with one lowercase word only: green, red, or purple."
)


class _CountingAdapter(LLMAdapter):
    """Count upstream ``generate`` calls — handled pre-LLM paths must not invoke the LLM."""

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, messages: list[dict[str, str]], **kwargs) -> AdapterResponse:
        self.calls += 1
        return AdapterResponse(text="UPSTREAM_SHOULD_NOT_RUN", model="mock-cross-adapter")


def test_default_pre_model_adapter_chain_possession_then_bounded_puzzle_slot() -> None:
    """Structural: default adapter order uses generic state-native then bounded puzzle."""
    chain = default_pre_model_adapters()
    assert len(chain) == 2
    assert isinstance(chain[0], PossessionStateAdapter)
    assert isinstance(chain[1], ArithmeticClosedWorldAdapter)


@pytest.mark.asyncio
async def test_bounded_three_box_puzzle_resolves_pre_llm_general_domain() -> None:
    """Regression only: bounded puzzle surface; green/purple copy is incidental (synced to solver)."""
    ada = _CountingAdapter()
    tok = domain_var.set("general")
    try:
        lens = Lens(
            LensConfig(
                adapter=ada,
                audit_log_path=None,
                enable_state_native_delegation=True,
                auto_interpret=True,
                auto_verify=True,
            ),
            session_id="cross-adapter-three-box",
        )
        r = await lens.process(THREE_BOX_PROMPT)
        assert r.action == InterventionAction.PASS
        assert r.response.strip().lower().rstrip(".") == "green"
        assert r.continuity_diagnostic == "state_native_closed_world_solved_unique"
        assert ada.calls == 0
    finally:
        domain_var.reset(tok)


@pytest.mark.asyncio
async def test_cross_turn_transfer_arithmetic_and_inventory_queries_pre_llm_general_domain() -> None:
    """Invariant: cross-turn PEF mutation + state-native holder queries; names/objects are placeholders.

    Jack/Jill/lollipops are arbitrary. The purpose is not story content—it is to prove
    committed-state arithmetic and ``Who has …?`` retrieval without the LLM, on ``general``
    domain. ``continuity_diagnostic`` must reflect state-native inventory reads
    (``state_native_committed_inventory_read``), never the bounded-puzzle marker
    (``…closed_world_solved_unique``).
    """
    ada = _CountingAdapter()

    tok = domain_var.set("general")
    try:
        lens = Lens(
            LensConfig(
                adapter=ada,
                audit_log_path=None,
                enable_state_native_delegation=True,
                auto_interpret=True,
                auto_verify=True,
            ),
            session_id="cross-adapter-transfer-arithmetic",
        )

        assert (await lens.process("Jack has 10 lollipops.")).action == InterventionAction.PASS
        assert (await lens.process("Jack gave Jill 5 lollipops.")).action == InterventionAction.PASS
        assert (await lens.process("Jill ate 1 lollipop.")).action == InterventionAction.PASS
        assert ada.calls == 0

        q4 = await lens.process("Who has 4 lollipops?")
        assert q4.action == InterventionAction.PASS
        assert "jill" in q4.response.lower()
        assert q4.continuity_diagnostic == "state_native_committed_inventory_read"
        assert "closed_world_solved_unique" not in (q4.continuity_diagnostic or "")
        body4 = format_chat_response(q4, include_operator_detail=True)
        assert body4["aurora"]["release_path"] == "state_native_retrieval"
        assert body4["aurora"]["llm_called"] is False
        assert ada.calls == 0

        q5 = await lens.process("Who has 5 lollipops?")
        assert q5.action == InterventionAction.PASS
        assert "jack" in q5.response.lower()
        assert q5.continuity_diagnostic == "state_native_committed_inventory_read"
        body5 = format_chat_response(q5, include_operator_detail=True)
        assert body5["aurora"]["release_path"] == "state_native_retrieval"
        assert body5["aurora"]["llm_called"] is False
        assert ada.calls == 0
    finally:
        domain_var.reset(tok)


@pytest.mark.asyncio
async def test_plural_location_query_sets_state_native_pending_clarify_isolated_to_session() -> None:
    """Plural fallback expands current-session candidates and holds deterministically."""
    # Prior session entities must not bleed into new-session candidate sets.
    prior = Lens(
        LensConfig(
            adapter=_CountingAdapter(),
            audit_log_path=None,
            enable_state_native_delegation=True,
            auto_interpret=True,
            auto_verify=True,
        ),
        session_id="plural-prior-people",
    )
    assert (await prior.process("Record this fact: Emma is a manager.")).action == InterventionAction.PASS
    assert (await prior.process("Record this fact: Anna is an engineer.")).action == InterventionAction.PASS

    ada = _CountingAdapter()
    tok = domain_var.set("general")
    try:
        lens = Lens(
            LensConfig(
                adapter=ada,
                audit_log_path=None,
                enable_state_native_delegation=True,
                auto_interpret=True,
                auto_verify=True,
            ),
            session_id="plural-new-boxes",
        )
        for t in (
            "Record this fact: Box A is on the table.",
            "Record this fact: Box B is on the shelf.",
            "Record this fact: Box C is in the closet.",
        ):
            assert (await lens.process(t)).action == InterventionAction.PASS

        r = await lens.process("Where are the boxes?")
        assert r.action == InterventionAction.CONTAIN
        pending = lens.pef.pending_clarification or {}
        candidates = set(str(x) for x in (pending.get("candidate_entities") or []))
        assert candidates == {"Box A", "Box B", "Box C"}
        assert "Emma" not in candidates
        assert "Anna" not in candidates
        assert ada.calls == 0
    finally:
        domain_var.reset(tok)


@pytest.mark.asyncio
async def test_non_medical_continuity_then_plural_ambiguity_hold_stays_pre_llm() -> None:
    """Non-medical PEF continuity and ambiguity containment remain intact together."""
    ada = _CountingAdapter()
    tok = domain_var.set("general")
    try:
        lens = Lens(
            LensConfig(
                adapter=ada,
                audit_log_path=None,
                enable_state_native_delegation=True,
                auto_interpret=True,
                auto_verify=True,
            ),
            session_id="non-medical-continuity-ambiguity",
        )
        assert (await lens.process("Jack has 10 lollipops.")).action == InterventionAction.PASS
        assert (await lens.process("Jack gave Jill 5 lollipops.")).action == InterventionAction.PASS
        assert (await lens.process("Jill ate 1 lollipop.")).action == InterventionAction.PASS
        q_cont = await lens.process("Who has 4 lollipops?")
        assert q_cont.action == InterventionAction.PASS
        assert "jill" in q_cont.response.lower()
        assert q_cont.continuity_diagnostic == "state_native_committed_inventory_read"
        assert ada.calls == 0

        for t in (
            "Record this fact: Crate A is in the garage.",
            "Record this fact: Crate B is in the attic.",
        ):
            assert (await lens.process(t)).action == InterventionAction.PASS

        q_amb = await lens.process("Where are the crates?")
        assert q_amb.action == InterventionAction.CONTAIN
        pending = lens.pef.pending_clarification or {}
        candidates = set(str(x) for x in (pending.get("candidate_entities") or []))
        assert candidates == {"Crate A", "Crate B"}
        assert ada.calls == 0
    finally:
        domain_var.reset(tok)


@pytest.mark.asyncio
async def test_medical_continuity_ledger_query_still_bypasses_llm() -> None:
    """Invariant: scripted medical ledger query path still bypasses LLM; fixture text is arbitrary."""
    ada = _CountingAdapter()
    tok = domain_var.set("medical")
    try:
        lens = Lens(
            LensConfig(
                adapter=ada,
                audit_log_path=None,
                enable_state_native_delegation=True,
                auto_interpret=True,
                auto_verify=True,
            ),
            session_id="cross-adapter-medical",
        )
        await lens.process(
            "Dr. Silva prescribed Rina 20mg of Amlodipine twice daily.",
        )
        r = await lens.process("What dosage is Rina currently receiving?")
        assert r.action == InterventionAction.PASS
        assert "20mg" in r.response
        assert "Amlodipine" in r.response
        body = format_chat_response(r, include_operator_detail=True)
        assert body["aurora"]["release_path"] == "state_native_retrieval"
        assert body["aurora"]["llm_called"] is False
        assert ada.calls == 0
    finally:
        domain_var.reset(tok)

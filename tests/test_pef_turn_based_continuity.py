"""Multi-turn PEF / state-native continuity (real Lens + spaCy extraction).

Group 1 (egg riddle): proves destructive/action verbs on the **same counted eggs**
do not triple-subtract inventory across turns — expected **4** eggs remain after
break/fry/eat on **two** eggs.

Group 2 (red box): proves holder updates with transfer plus **HAS**/**CONTAIN** quantity
chains: marbles sourced from or returned to a possessed container decrement/increment the
correct entity via state-native **GIVE** (**N … to Recipient**) and **PUT … into**.

Uses the same pattern as ``test_pef_inventory_invariants`` / demo golden paths:
:class:`~aurora_lens.interpret.spacy_backend.SpacyBackend` plus an upstream adapter
that must not run when state-native satisfies the query.
"""

from __future__ import annotations

import pytest

from aurora_lens.adapters.base import AdapterResponse, LLMAdapter
from aurora_lens.config import LensConfig
from aurora_lens.govern.decision import InterventionAction
from aurora_lens.interpret.spacy_backend import SpacyBackend
from aurora_lens.lens import Lens

pytest.importorskip("spacy")


class _NoUpstreamAdapter(LLMAdapter):
    async def generate(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> AdapterResponse:
        raise AssertionError(
            "LLM must not be called when state-native + PEF can answer the query"
        )


class _CountingUpstreamAdapter(LLMAdapter):
    """Like demo golden-path: deterministic text if ever invoked (guards regressions)."""

    def __init__(self, text: str = "MODEL_FALLBACK_BAD") -> None:
        self.calls = 0
        self.text = text

    async def generate(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> AdapterResponse:
        self.calls += 1
        return AdapterResponse(text=self.text, model="turn-continuity-mock")


def _lens_live_pef_strict_no_llm() -> Lens:
    return Lens(
        LensConfig(
            adapter=_NoUpstreamAdapter(),
            extraction_backend=SpacyBackend(model="en_core_web_sm"),
            auto_interpret=True,
            auto_verify=True,
            enable_state_native_delegation=True,
            inject_pef_context=False,
        )
    )


def _committed_holder_subject_name(pef, object_literal_needle: str) -> str | None:
    """Return display name of subject whose latest HAS for ``needle`` is non-negated."""
    tgt = object_literal_needle.strip().lower()
    best: tuple[int, str] | None = None
    for ent in pef.entities.values():
        best_turn = -1
        active = False
        for rel in pef.get_relationships_for_subject(ent.id):
            if rel.relation != "HAS":
                continue
            lit = str(rel.object_literal or "").strip().lower()
            if lit != tgt:
                continue
            if rel.source_turn >= best_turn:
                best_turn = rel.source_turn
                active = not rel.negated
        if active and best_turn >= 0:
            if best is None or best_turn > best[0]:
                best = (best_turn, ent.name)
    return best[1] if best else None


@pytest.mark.asyncio
async def test_egg_riddle_inventory_no_double_count_destructive_chain() -> None:
    lens = _lens_live_pef_strict_no_llm()
    turns = (
        "Margaret has 6 eggs.",
        "Margaret breaks 2 eggs.",
        "Margaret fries 2 eggs.",
        "Margaret eats 2 eggs.",
    )
    for line in turns:
        r = await lens.process(line)
        assert r.action == InterventionAction.PASS, (line, r.response, r.decision)

    q = await lens.process("How many eggs does Margaret have?")
    assert q.action == InterventionAction.PASS
    low = q.response.lower()
    assert "margaret" in low
    assert "4" in low and (
        "4 eggs" in low or "four" in low
    ), low
    marg = lens.pef.find_entity_by_name("Margaret")
    assert marg is not None


@pytest.mark.asyncio
async def test_red_box_holder_updates_across_turns_state_native() -> None:
    """Possession continuity: Anna → Ben; *Who has the red box?* bypasses upstream."""
    lens = _lens_live_pef_strict_no_llm()
    for line in (
        "Anna has a red box.",
        "The red box contains 5 blue marbles.",
        "Anna gives the red box to Ben.",
        "Ben gives 2 blue marbles to Clara.",
        "Clara puts 1 blue marble back into the red box.",
    ):
        r = await lens.process(line)
        assert r.action == InterventionAction.PASS, (line, r.decision)

    holder_before_q = _committed_holder_subject_name(lens.pef, "red box")
    assert holder_before_q == "Ben", (
        "Ben must hold the red box before the explicit holder query; "
        f"got {holder_before_q!r}"
    )

    who = await lens.process("Who has the red box?")
    assert who.action == InterventionAction.PASS
    assert "ben" in who.response.lower()

    adapter = lens._config.adapter
    assert isinstance(adapter, _NoUpstreamAdapter)


@pytest.mark.asyncio
async def test_red_box_blue_marble_counts_after_transfer_and_put_inferred() -> None:
    """Quantity continuity after GIVE-from-container and PUT back on container."""
    lens = Lens(
        LensConfig(
            adapter=_CountingUpstreamAdapter(),
            extraction_backend=SpacyBackend(model="en_core_web_sm"),
            auto_interpret=True,
            auto_verify=True,
            enable_state_native_delegation=True,
            inject_pef_context=False,
        )
    )
    for line in (
        "Anna has a red box.",
        # "contains" emits ``CONTAIN``; HAS form also supported for boxed quantities.
        "The red box has 5 blue marbles.",
        "Anna gives the red box to Ben.",
        "Ben gives 2 blue marbles to Clara.",
        "Clara puts 1 blue marble back into the red box.",
    ):
        r = await lens.process(line)
        assert r.action == InterventionAction.PASS, (line, r.decision)

    assert _committed_holder_subject_name(lens.pef, "red box") == "Ben"

    q_box = await lens.process("How many blue marbles does the red box have?")
    assert q_box.action == InterventionAction.PASS
    q_cl = await lens.process("How many blue marbles does Clara have?")
    assert q_cl.action == InterventionAction.PASS

    adapter = lens._config.adapter
    assert isinstance(adapter, _CountingUpstreamAdapter)
    assert adapter.calls == 0, "quantity answers must be state-native, not upstream"

    assert "4" in q_box.response
    low_cl = q_cl.response.lower()
    assert "clara" in low_cl
    assert ("1 blue marble" in low_cl or "one blue marble" in low_cl)

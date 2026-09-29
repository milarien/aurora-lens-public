"""Inventory and possession invariants over committed PEF + Lens routing.

These checks are **guardrails**, not one-off bug regressions: keep them passing when
touching ``inventory.py``, ``possession_mutations.py``, ``pef/state.py``, ``lens.py``,
``spacy_backend.py``, or PEF admission paths for HAS / transfer / clarification.

Requires ``en_core_web_sm`` (same as repo CI).
"""

from __future__ import annotations

import pytest

from aurora_lens.adapters.base import AdapterResponse, LLMAdapter
from aurora_lens.config import LensConfig
from aurora_lens.govern.decision import InterventionAction
from aurora_lens.interpret.spacy_backend import SpacyBackend
from aurora_lens.lens import Lens
from aurora_lens.verify.flags import FlagType

pytest.importorskip("spacy")

_SUPERSEDE_MARKER = "Superseded by new possession assertion"


class _NoUpstreamAdapter(LLMAdapter):
    """Adapter that must not run for state-native / pre-LLM paths."""

    def __init__(self) -> None:
        self.calls = 0

    async def generate(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> AdapterResponse:
        self.calls += 1
        return AdapterResponse(text="NEVER_SHOULD_RUN", model="invariant-mock")


def _lens_state_native_spacy() -> Lens:
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


def _lens_state_native_spacy_no_auto_verify() -> Lens:
    return Lens(
        LensConfig(
            adapter=_NoUpstreamAdapter(),
            extraction_backend=SpacyBackend(model="en_core_web_sm"),
            auto_interpret=True,
            auto_verify=False,
            enable_state_native_delegation=True,
            inject_pef_context=False,
        )
    )


def _matching_holder_names(pef, item_tail: str) -> set[str]:
    from aurora_lens.state_native_engine.eval.inventory import (
        _matching_holders_for_item,
    )

    return {h.name for h in _matching_holders_for_item(pef, item_tail)}


def _subject_has_superseded_negated_has(pef, subject_name: str) -> bool:
    """True if ``subject_name`` has a negated HAS whose evidence uses supersede copy."""
    ent = pef.find_entity_by_name(subject_name)
    assert ent is not None, f"missing entity {subject_name!r}"
    for rel in pef.get_relationships_for_subject(ent.id):
        if rel.relation != "HAS" or not rel.negated:
            continue
        if _SUPERSEDE_MARKER in (rel.evidence or ""):
            return True
    return False


def _negated_has_rows_for_item_surface(
    pef,
    subject_name: str,
    item_substr: str,
) -> list[str]:
    """Negated HAS object literals for ``subject_name`` containing ``item_substr``."""
    ent = pef.find_entity_by_name(subject_name)
    assert ent is not None
    out: list[str] = []
    sub = item_substr.lower()
    for rel in pef.get_relationships_for_subject(ent.id):
        if rel.relation != "HAS" or not rel.negated:
            continue
        lit = str(rel.object_literal or "").lower()
        if sub in lit:
            out.append(str(rel.object_literal or ""))
    return out


@pytest.mark.asyncio
async def test_has_same_item_multiple_subjects_remain_active() -> None:
    """Bare same-category HAS is not globally unique across subjects."""
    lens = _lens_state_native_spacy()
    for line in (
        "James has an apple.",
        "John has an apple.",
        "Mary has an apple.",
    ):
        r = await lens.process(line)
        assert r.action == InterventionAction.PASS

    apples_holders = _matching_holder_names(lens.pef, "apple")
    assert apples_holders == {"James", "John", "Mary"}

    for name in ("James", "John", "Mary"):
        assert not _negated_has_rows_for_item_surface(lens.pef, name, "apple")

    assert not _subject_has_superseded_negated_has(lens.pef, "James")
    assert not _subject_has_superseded_negated_has(lens.pef, "John")
    assert not _subject_has_superseded_negated_has(lens.pef, "Mary")

    who = await lens.process("Who has apples?")
    assert who.action == InterventionAction.PASS
    low = who.response.lower()
    assert "james" in low and "john" in low and "mary" in low
    adapter = lens._config.adapter
    assert isinstance(adapter, _NoUpstreamAdapter)
    assert adapter.calls == 0


@pytest.mark.asyncio
async def test_has_supersession_only_same_subject_same_item() -> None:
    """Re-asserting the same category for one subject supersede only that arc; pear survives."""
    lens = _lens_state_native_spacy()
    for line in (
        "James has an apple.",
        "James has a pear.",
        "James has an apple.",
    ):
        r = await lens.process(line)
        assert r.action == InterventionAction.PASS

    assert _matching_holder_names(lens.pef, "apple") == {"James"}
    assert _matching_holder_names(lens.pef, "pear") == {"James"}

    assert _subject_has_superseded_negated_has(lens.pef, "James")
    entity_names = {e.name for e in lens.pef.entities.values()}
    for name in entity_names:
        if name == "James":
            continue
        assert not _subject_has_superseded_negated_has(lens.pef, name)

    adapter = lens._config.adapter
    assert isinstance(adapter, _NoUpstreamAdapter)
    assert adapter.calls == 0


@pytest.mark.asyncio
async def test_transfer_negates_giver_not_other_holders() -> None:
    """Committed GIVE removes giver HAS; unrelated holders stay active."""
    lens = _lens_state_native_spacy()
    setup = (
        "James has an apple.",
        "John has an apple.",
        "James gave an apple to Mary.",
    )
    for line in setup:
        r = await lens.process(line)
        assert r.action == InterventionAction.PASS, (line, r.decision)

    holders = _matching_holder_names(lens.pef, "apple")
    assert holders == {"John", "Mary"}
    assert lens.pef.find_entity_by_name("James") is not None
    adapter = lens._config.adapter
    assert isinstance(adapter, _NoUpstreamAdapter)
    assert adapter.calls == 0


@pytest.mark.asyncio
async def test_unresolved_possessive_np_not_admitted() -> None:
    """Possessive comparative without binder → referent clarification only; no ``His dog`` entity."""
    lens = _lens_state_native_spacy()
    for line in (
        "John had a dog.",
        "Richard had a dog.",
    ):
        r = await lens.process(line)
        assert r.action == InterventionAction.PASS

    r = await lens.process("His dog was bigger.")
    assert r.action == InterventionAction.CONTAIN
    assert r.decision is not None
    assert any(f.flag_type == FlagType.UNRESOLVED_REFERENT for f in (r.flags or []))

    pc = lens.pef.pending_clarification or {}
    assert pc.get("failed_constraint") == "UNRESOLVED_REFERENT"
    assert sorted(pc.get("candidate_entities") or []) == ["John", "Richard"]

    dog_like = [
        ent.name.strip()
        for ent in lens.pef.entities.values()
        if "his dog" == ent.name.strip().lower() or ent.name.strip().lower().startswith(
            "his ",
        )
    ]
    assert not dog_like, f"unexpected possessive-named entities {dog_like!r}"
    assert lens.pef.find_entity_by_name("His dog") is None
    adapter = lens._config.adapter
    assert isinstance(adapter, _NoUpstreamAdapter)
    assert adapter.calls == 0


@pytest.mark.asyncio
async def test_inventory_pronoun_item_contains_not_stop() -> None:
    """Unbounded pronoun item in *Who has them?* clarifies active inventory surfaces."""
    lens = _lens_state_native_spacy_no_auto_verify()
    for line in (
        "James has apples.",
        "John has pears.",
    ):
        r = await lens.process(line)
        assert r.action == InterventionAction.PASS

    r = await lens.process("Who has them?")
    assert r.action == InterventionAction.CONTAIN

    pc = lens.pef.pending_clarification or {}
    assert pc.get("state_native_ambiguity_kind") == (
        "inventory_holder_query_unresolved_item"
    )
    cand = [str(x).lower() for x in (pc.get("candidate_entities") or [])]
    assert "apples" in cand and "pears" in cand

    adapter = lens._config.adapter
    assert isinstance(adapter, _NoUpstreamAdapter)
    assert adapter.calls == 0

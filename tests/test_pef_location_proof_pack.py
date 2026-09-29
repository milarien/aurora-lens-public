"""Proof pack: bounded location QUERY family + continuity signals (see docs/pef_location_state_kernel.md)."""

from unittest.mock import AsyncMock

import pytest

from aurora_lens.adapters.base import AdapterResponse
from aurora_lens.config import LensConfig
from aurora_lens.interpret.base import ExtractionBackend
from aurora_lens.interpret.schema import ExtractedClaim, ExtractionResult
from aurora_lens.lens import Lens
from aurora_lens.pef.entity import Entity
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState, Relationship
from aurora_lens.verify.checker import Checker
from aurora_lens.verify.flags import FlagType
from aurora_lens.state_native_engine.epistemic import EpistemicResult

from tests.test_pef_mutation_query_continuity import (
    _MutationSequenceBackend,
    _RecordingAdapter,
)


@pytest.mark.parametrize(
    ("query", "must_contain", "must_not"),
    [
        ("Where is the gold key now?", "safe", "desk"),
        ("Where's the gold key now?", "safe", "desk"),
        ("Where is the gold key?", "safe", "desk"),
        ("Where did I put the gold key?", "safe", "desk"),
        ("Is the gold key still in the safe?", "yes", None),
        ("Is the gold key still in the desk?", "no", None),
    ],
)
@pytest.mark.asyncio
async def test_location_query_family_process_then_stream(
    query: str,
    must_contain: str,
    must_not: str | None,
):
    adapter = _RecordingAdapter()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_MutationSequenceBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
            auto_interpret=True,
            inject_pef_context=False,
        ),
        session_id="proof-pack",
    )
    await lens.process("The gold key is in the desk.")
    await lens.process("I moved the gold key to the safe.")

    adapter.generate.reset_mock()
    rp = await lens.process(query)
    low = rp.response.lower()
    assert must_contain in low
    if must_not is not None:
        assert must_not not in low
    assert rp.model == ""
    assert rp.continuity_diagnostic == "state_native_committed_location_read"
    adapter.generate.assert_not_called()

    adapter.generate.reset_mock()
    parts: list[str] = []
    meta_diag = None
    async for kind, payload in lens.process_stream(query):
        if kind == "chunk":
            parts.append(payload[1])
        if kind == "metadata":
            meta_diag = payload.get("continuity_diagnostic")
    out = "".join(parts).lower()
    assert must_contain in out
    if must_not is not None:
        assert must_not not in out
    assert meta_diag == "state_native_committed_location_read"
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_draft_contradicting_committed_at_raises_flag():
    """Maps to diagnostic ``draft_contradicts_committed_state`` (see kernel doc)."""
    pef = PEFState(session_id="chk")
    key = Entity.create("gold key", 0, session_id="chk")
    safe = Entity.create("safe", 0, session_id="chk")
    pef.add_entity(key)
    pef.add_entity(safe)
    pef.add_relationship(
        Relationship(
            subject_id=key.id,
            relation="AT",
            object_entity_id=safe.id,
            object_literal=None,
            span=Span.PRESENT,
            source_turn=2,
            evidence="user",
        )
    )

    class _AtDeskBackend(ExtractionBackend):
        async def extract(self, text: str, pef_arg: PEFState) -> ExtractionResult:  # noqa: ARG002
            return ExtractionResult(
                claims=[
                    ExtractedClaim(
                        subject="gold key",
                        relation="AT",
                        obj="the desk",
                        span=Span.PRESENT,
                        negated=False,
                        evidence=text,
                        extractor_backend="test",
                    ),
                ],
                entity_mentions=[],
                span=Span.PRESENT,
            )

    checker = Checker(_AtDeskBackend())
    flags = await checker.check(
        "The gold key is at the desk.",
        pef,
        user_input="Where is it?",
    )
    types = {f.flag_type for f in flags}
    assert FlagType.CONTRADICTS_COMMITTED_STATE in types


# ── Regression: location-bearing IS fallback ─────────────────────────────────


def test_location_bearing_is_predicate_recognised():
    """_is_location_bearing_predicate accepts all specified bare words and prefix patterns."""
    from aurora_lens.state_native_engine.eval.location import _is_location_bearing_predicate

    for word in ("overseas", "abroad", "away", "interstate", "home", "here", "there"):
        assert _is_location_bearing_predicate(word), f"{word!r} should be location-bearing"

    for phrase in ("in hospital", "in Paris", "at school", "at home", "on the road"):
        assert _is_location_bearing_predicate(phrase), f"{phrase!r} should be location-bearing"

    for non_loc in ("happy", "tall", "brilliant", "sleeping"):
        assert not _is_location_bearing_predicate(non_loc), f"{non_loc!r} should not be location-bearing"


def test_where_query_answers_from_location_bearing_is_no_at():
    """Committed 'sister IS overseas' must answer WHERE query without an AT relation.

    Invariant: location-bearing IS predicates are semantically equivalent to location
    state for WHERE queries. HARD_STOP / state_native_no_at is incorrect when an IS
    location predicate exists.

    Committed:
        sister IS overseas
        Emma HAS sister

    Query subject phrase: "sister"
    Expected outcome: ANSWER "sister is overseas."
    """
    from aurora_lens.pef.entity import Entity
    from aurora_lens.pef.span import Span
    from aurora_lens.pef.state import PEFState, Relationship
    from aurora_lens.state_native_engine.contracts import StateNativeOutcome
    from aurora_lens.state_native_engine.eval.location import (
        evaluate_location_from_committed_state,
    )

    pef = PEFState(session_id="loc-is-test")
    emma = Entity.create("Emma", 1, session_id="loc-is-test")
    sister = Entity.create("sister", 1, session_id="loc-is-test")
    pef.add_entity(emma)
    pef.add_entity(sister)

    # Emma HAS sister
    pef.add_relationship(Relationship(
        subject_id=emma.id,
        relation="HAS",
        object_entity_id=sister.id,
        object_literal=None,
        span=Span.PRESENT,
        source_turn=1,
        evidence="Emma's sister was mentioned",
    ))

    # sister IS overseas  (no AT relation)
    pef.add_relationship(Relationship(
        subject_id=sister.id,
        relation="IS",
        object_entity_id=None,
        object_literal="overseas",
        span=Span.PRESENT,
        source_turn=1,
        evidence="her sister was overseas",
    ))

    result = evaluate_location_from_committed_state(pef, "Anna's sister")

    assert result.outcome == StateNativeOutcome.ANSWER, (
        f"Expected ANSWER, got {result.outcome!r} "
        f"(stop_code={result.stop_reason_code!r}, text={result.user_visible_text!r}). "
        "location-bearing IS predicate must satisfy WHERE query when no AT exists."
    )
    text = result.user_visible_text
    assert "overseas" in text.lower(), f"Expected 'overseas' in answer text; got {text!r}"
    assert "anna's sister" in text.lower(), (
        f"Answer must use the subject phrase 'Anna's sister', not bare entity name; got {text!r}"
    )
    assert result.epistemic_result == EpistemicResult.VALUE
    assert result.stop_reason_code is None, (
        f"stop_reason_code must be None for ANSWER; got {result.stop_reason_code!r}"
    )


def test_where_query_prefers_at_over_is():
    """Explicit AT relation takes precedence over a location-bearing IS predicate."""
    from aurora_lens.pef.entity import Entity
    from aurora_lens.pef.span import Span
    from aurora_lens.pef.state import PEFState, Relationship
    from aurora_lens.state_native_engine.contracts import StateNativeOutcome
    from aurora_lens.state_native_engine.eval.location import (
        evaluate_location_from_committed_state,
    )

    pef = PEFState(session_id="loc-at-priority")
    key = Entity.create("gold key", 0, session_id="loc-at-priority")
    safe = Entity.create("safe", 0, session_id="loc-at-priority")
    pef.add_entity(key)
    pef.add_entity(safe)

    pef.add_relationship(Relationship(
        subject_id=key.id,
        relation="AT",
        object_entity_id=safe.id,
        object_literal=None,
        span=Span.PRESENT,
        source_turn=1,
        evidence="in the safe",
    ))
    # Also add a location-bearing IS — AT must win.
    pef.add_relationship(Relationship(
        subject_id=key.id,
        relation="IS",
        object_entity_id=None,
        object_literal="overseas",
        span=Span.PRESENT,
        source_turn=0,
        evidence="old state",
    ))

    result = evaluate_location_from_committed_state(pef, "gold key")

    assert result.outcome == StateNativeOutcome.ANSWER
    text = result.user_visible_text
    assert "safe" in text.lower(), f"AT relation must take priority; got {text!r}"
    assert "overseas" not in text.lower()
    assert result.epistemic_result == EpistemicResult.VALUE

"""PEF mutation → location QUERY reads committed current_at (state-native path)."""

from unittest.mock import AsyncMock

import pytest

from aurora_lens.adapters.base import AdapterResponse
from aurora_lens.config import LensConfig
from aurora_lens.interpret.base import ExtractionBackend
from aurora_lens.interpret.spacy_backend import SpacyBackend
from aurora_lens.interpret.schema import ExtractedClaim, ExtractionResult
from aurora_lens.lens import Lens
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState
from aurora_lens.state_native_engine.parse.query_surface import parse_location_subject_phrase


def test_parse_location_query_allows_trailing_now():
    assert parse_location_subject_phrase("Where is the gold key now?") == "gold key"
    assert parse_location_subject_phrase("Where's the gold key now?") == "gold key"


def test_parse_where_did_i_put_subject():
    assert parse_location_subject_phrase("Where did I put the gold key?") == "gold key"
    assert parse_location_subject_phrase("Where did I put gold key?") == "gold key"


class _MutationSequenceBackend(ExtractionBackend):
    """Deterministic three-turn extraction: establish AT → mutate AT → QUERY empty."""

    def __init__(self) -> None:
        self._turn = 0

    async def extract(self, text: str, pef: PEFState) -> ExtractionResult:  # noqa: ARG002
        self._turn += 1
        if self._turn == 1:
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
                entity_mentions=["gold key"],
                span=Span.PRESENT,
            )
        if self._turn == 2:
            return ExtractionResult(
                claims=[
                    ExtractedClaim(
                        subject="gold key",
                        relation="AT",
                        obj="the safe",
                        span=Span.PRESENT,
                        negated=False,
                        evidence=text,
                        extractor_backend="test",
                    ),
                ],
                entity_mentions=["gold key"],
                span=Span.PRESENT,
            )
        return ExtractionResult(claims=[], entity_mentions=[], span=Span.PRESENT)


class _RecordingAdapter:
    def __init__(self) -> None:
        self.generate = AsyncMock(
            return_value=AdapterResponse(text="MODEL_SHOULD_NOT_RUN", model="mock"),
        )


@pytest.mark.asyncio
async def test_mutation_then_where_now_reads_safe_not_desk():
    adapter = _RecordingAdapter()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_MutationSequenceBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
            auto_interpret=True,
        ),
        session_id="mut-q",
    )

    r1 = await lens.process("The gold key is in the desk.")
    assert r1.pef_snapshot is not None

    r2 = await lens.process("I moved the gold key to the safe.")
    assert r2.pef_snapshot is not None

    adapter.generate.reset_mock()
    r3 = await lens.process("Where is the gold key now?")
    assert "safe" in r3.response.lower()
    assert "desk" not in r3.response.lower()
    assert r3.model == ""
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_mutation_then_where_did_i_put_reads_safe_process_and_stream():
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
        session_id="mut-q-did-put",
    )

    await lens.process("The gold key is in the desk.")
    await lens.process("I moved the gold key to the safe.")

    q = "Where did I put the gold key?"
    adapter.generate.reset_mock()
    rp = await lens.process(q)
    assert "safe" in rp.response.lower()
    assert "desk" not in rp.response.lower()
    assert rp.model == ""
    adapter.generate.assert_not_called()

    adapter.generate.reset_mock()
    chunks: list[str] = []
    async for kind, payload in lens.process_stream("Where did I put gold key?"):
        if kind == "chunk":
            chunks.append(payload[1])
    visible = "".join(chunks).lower()
    assert "safe" in visible
    assert "desk" not in visible
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_process_stream_where_now_matches_committed_state_native():
    """Streaming path delegates location QUERY identically — no adapter call."""
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
        session_id="mut-q-stream",
    )

    await lens.process("The gold key is in the desk.")
    await lens.process("I moved the gold key to the safe.")

    adapter.generate.reset_mock()
    chunks: list[str] = []
    async for kind, payload in lens.process_stream("Where is the gold key now?"):
        if kind == "chunk":
            chunks.append(payload[1])

    visible = "".join(chunks).lower()
    assert "safe" in visible
    assert "desk" not in visible
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_gold_key_modifier_round_trip_preserved_through_move_update():
    """Known multi-token entity names must survive assertion -> movement -> query."""
    adapter = _RecordingAdapter()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=SpacyBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
            auto_interpret=True,
            inject_pef_context=False,
        ),
        session_id="mut-gold-key-modifier",
    )

    r1 = await lens.process("The gold key is in the desk.")
    assert r1.action.name == "PASS"

    r2 = await lens.process("Where is the gold key?")
    text2 = r2.response.lower()
    assert "gold key" in text2
    assert "desk" in text2
    assert r2.model == ""

    r3 = await lens.process("I moved the gold key to the study.")
    text3 = r3.response.lower()
    assert r3.action.name == "PASS", text3
    assert "which referent" not in text3
    assert lens.pef.pending_clarification is None

    r4 = await lens.process("Where is the gold key?")
    text4 = r4.response.lower()
    assert "gold key" in text4
    assert "study" in text4
    assert "desk" not in text4
    assert r4.model == ""


@pytest.mark.parametrize(
    ("entity_phrase", "start_loc", "end_loc"),
    [
        ("gold key", "desk", "study"),
        ("red notebook", "shelf", "drawer"),
    ],
)
@pytest.mark.asyncio
async def test_known_multi_token_entity_location_state_held_until_legitimate_update(
    entity_phrase: str,
    start_loc: str,
    end_loc: str,
):
    """Known multi-token referents must remain bound across mutation/query turns until updated."""
    adapter = _RecordingAdapter()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=SpacyBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
            auto_interpret=True,
            inject_pef_context=False,
        ),
        session_id=f"state-held-{entity_phrase.replace(' ', '-')}",
    )

    r1 = await lens.process(f"The {entity_phrase} is in the {start_loc}.")
    assert r1.action.name == "PASS"

    rq1 = await lens.process(f"Where is the {entity_phrase}?")
    q1 = rq1.response.lower()
    assert entity_phrase in q1
    assert start_loc in q1
    assert rq1.model == ""

    rm = await lens.process(f"I moved the {entity_phrase} to the {end_loc}.")
    mt = rm.response.lower()
    assert rm.action.name == "PASS", mt
    assert "which referent" not in mt
    assert lens.pef.pending_clarification is None

    rq2 = await lens.process(f"Where is the {entity_phrase}?")
    q2 = rq2.response.lower()
    assert entity_phrase in q2
    assert end_loc in q2
    assert start_loc not in q2
    assert rq2.model == ""

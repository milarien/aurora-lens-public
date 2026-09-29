"""Lane 1 snapshot policy regressions — full Lens.process path (SNC-1 v1).

See docs/adr/lane1_snapshot_policy.md.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aurora_lens.adapters.base import AdapterResponse
from aurora_lens.config import LensConfig
from aurora_lens.interpret.base import ExtractionBackend
from aurora_lens.interpret.schema import ExtractedClaim, ExtractionResult
from aurora_lens.lens import Lens
from aurora_lens.pef.entity import Entity
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState


class _SpuriousAtOnQueryBackend(ExtractionBackend):
    """Simulates same-turn AT extraction on a pure location QUERY (ghost scenario)."""

    async def extract(self, text: str, pef: PEFState) -> ExtractionResult:  # noqa: ARG002
        if "where is alice" in text.lower():
            return ExtractionResult(
                claims=[
                    ExtractedClaim(
                        subject="Alice",
                        relation="AT",
                        obj="the store",
                        span=Span.PRESENT,
                        negated=False,
                        evidence="Alice went to the store.",
                        extractor_backend="test",
                    ),
                ],
                entity_mentions=["Alice"],
                span=Span.PRESENT,
            )
        return ExtractionResult(claims=[], entity_mentions=[], span=Span.PRESENT)


class _RecordingAdapter:
    def __init__(self) -> None:
        self.generate = AsyncMock(
            return_value=AdapterResponse(text="MODEL_SHOULD_NOT_RUN", model="mock"),
        )

    async def generate_stream(self, messages, **kwargs):  # noqa: ANN001, ARG002
        yield (
            {"choices": [{"delta": {"content": "MODEL_SHOULD_NOT_RUN"}, "index": 0}]},
            "MODEL_SHOULD_NOT_RUN",
        )


def _lens_with_alice_entity(**config_kw: object) -> Lens:
    cfg = LensConfig(
        adapter=_RecordingAdapter(),
        extraction_backend=_SpuriousAtOnQueryBackend(),
        enable_state_native_delegation=True,
        auto_verify=False,
        inject_pef_context=False,
        **config_kw,
    )
    lens = Lens(cfg, session_id="lane1-seed")
    alice = Entity.create("Alice", 0, session_id=lens._pef.session_id)
    lens._pef.add_entity(alice)
    return lens


@pytest.mark.asyncio
async def test_query_lane1_does_not_answer_from_same_turn_spurious_at():
    """Ghost continuity: spurious same-turn AT must not become Lane 1 substrate."""
    lens = _lens_with_alice_entity(auto_interpret=True)
    adapter = lens._config.adapter

    result = await lens.process("Where is Alice?")
    assert result.model == ""
    adapter.generate.assert_not_called()
    assert "store" not in result.response.lower()


@pytest.mark.asyncio
async def test_pure_query_empty_pef_returns_without_llm():
    adapter = _RecordingAdapter()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_SpuriousAtOnQueryBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
            auto_interpret=True,
            inject_pef_context=False,
        ),
        session_id="lane1-pure-q",
    )

    result = await lens.process("Where is Alice?")
    assert result.model == ""
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_auto_interpret_false_still_uses_pre_extraction_snapshot():
    lens = _lens_with_alice_entity(auto_interpret=False)
    adapter = lens._config.adapter

    result = await lens.process("Where is Alice?")
    assert result.model == ""
    adapter.generate.assert_not_called()
    assert "store" not in result.response.lower()


@pytest.mark.asyncio
async def test_prior_turn_at_visible_on_later_query():
    adapter = _RecordingAdapter()

    class _EstablishThenQueryBackend(ExtractionBackend):
        def __init__(self) -> None:
            self._n = 0

        async def extract(self, text: str, pef: PEFState) -> ExtractionResult:  # noqa: ARG002
            self._n += 1
            if self._n == 1:
                return ExtractionResult(
                    claims=[
                        ExtractedClaim(
                            subject="Alice",
                            relation="AT",
                            obj="the store",
                            span=Span.PRESENT,
                            negated=False,
                            evidence=text,
                            extractor_backend="test",
                        ),
                    ],
                    entity_mentions=["Alice"],
                    span=Span.PRESENT,
                )
            return ExtractionResult(claims=[], entity_mentions=[], span=Span.PRESENT)

    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EstablishThenQueryBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
            auto_interpret=True,
            inject_pef_context=False,
        ),
        session_id="lane1-cross",
    )

    await lens.process("Alice is at the store.")
    adapter.generate.reset_mock()
    result = await lens.process("Where is Alice?")
    assert result.model == ""
    adapter.generate.assert_not_called()
    assert "store" in result.response.lower()


@pytest.mark.asyncio
async def test_process_stream_query_ghost_continuity():
    lens = _lens_with_alice_entity(auto_interpret=True)
    adapter = lens._config.adapter

    chunks: list[str] = []
    async for kind, payload in lens.process_stream("Where is Alice?"):
        if kind == "chunk":
            chunks.append(payload[1])

    visible = "".join(chunks).lower()
    assert "store" not in visible
    adapter.generate.assert_not_called()

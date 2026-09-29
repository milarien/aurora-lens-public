"""FU-QUERY-REL enforcement: QUERY narrative is read-only for relationship writes."""

from __future__ import annotations

import pytest

from aurora_lens.adapters.base import AdapterResponse
from aurora_lens.config import LensConfig
from aurora_lens.interpret.base import ExtractionBackend
from aurora_lens.interpret.schema import ExtractedClaim, ExtractionResult
from aurora_lens.lens import Lens
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState
from aurora_lens.state_native_engine.contracts import QueryEphemeralBindings, StateNativeRequest
from aurora_lens.interpret.turn_act import TurnAct


class _FixedClaimBackend(ExtractionBackend):
    async def extract(self, text: str, pef: PEFState) -> ExtractionResult:  # noqa: ARG002
        return ExtractionResult(
            claims=[
                ExtractedClaim(
                    subject="Nora",
                    relation="TELL",
                    obj="Lucy she arrived",
                    span=Span.PRESENT,
                    negated=False,
                    evidence=text,
                    extractor_backend="test",
                )
            ],
            entity_mentions=["Nora", "Lucy"],
            span=Span.PRESENT,
        )


class _Adapter:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, messages, **kwargs):  # noqa: ANN001, ARG002
        self.calls += 1
        return AdapterResponse(text="ok", model="stub")


@pytest.mark.asyncio
async def test_query_turn_does_not_commit_relationship_claims():
    lens = Lens(
        LensConfig(
            adapter=_Adapter(),
            extraction_backend=_FixedClaimBackend(),
            auto_interpret=True,
            auto_verify=False,
            enable_state_native_delegation=False,
        )
    )
    before = len(lens.pef.relationships)
    before_entities = len(lens.pef.entities)
    await lens.process("Did Nora tell Lucy she arrived?")
    assert len(lens.pef.relationships) == before
    assert len(lens.pef.entities) == before_entities


@pytest.mark.asyncio
async def test_fu_query_rel_mutation_remains_out_of_v1_runtime_scope():
    """Internal QUERY stays read-only: asking cannot create relationship/binding state."""
    lens = Lens(
        LensConfig(
            adapter=_Adapter(),
            extraction_backend=_FixedClaimBackend(),
            auto_interpret=True,
            auto_verify=False,
            enable_state_native_delegation=False,
        )
    )
    before_rels = len(lens.pef.relationships)
    before_bindings = len(lens.pef.discourse_referent_bindings)
    before_entities = len(lens.pef.entities)
    await lens.process("Who did Nora tell arrived?")
    assert len(lens.pef.relationships) == before_rels
    assert len(lens.pef.discourse_referent_bindings) == before_bindings
    assert len(lens.pef.entities) == before_entities


@pytest.mark.asyncio
async def test_non_query_turn_still_commits_relationship_claims():
    lens = Lens(
        LensConfig(
            adapter=_Adapter(),
            extraction_backend=_FixedClaimBackend(),
            auto_interpret=True,
            auto_verify=False,
            enable_state_native_delegation=False,
        )
    )
    await lens.process("Nora told Lucy she arrived.")
    assert any(r.relation == "TELL" for r in lens.pef.relationships)


def test_state_native_request_accepts_ephemeral_query_bindings():
    req = StateNativeRequest(
        user_text="Who has them?",
        pef=PEFState(),
        turn_act=TurnAct.QUERY,
        detected_span=Span.PRESENT,
        query_ephemeral_bindings=QueryEphemeralBindings(
            turn=1,
            discourse_bindings={"them": "apples"},
        ),
    )
    assert req.query_ephemeral_bindings is not None
    assert req.query_ephemeral_bindings.discourse_bindings["them"] == "apples"

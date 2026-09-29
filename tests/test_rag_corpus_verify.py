"""Post-LLM verify grounding for corpus Q&A (RAG Context:/Question: harness)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from aurora_lens.adapters.base import LLMAdapter, AdapterResponse
from aurora_lens.config import LensConfig
from aurora_lens.context import request_metadata_var, governance_mode_override_var
from aurora_lens.corpus.document_ingestion import ingest_file
from aurora_lens.corpus.registry import CorpusRegistry
from aurora_lens.corpus.retrieve import assemble_rag_message, retrieve_chunks
from aurora_lens.govern.canonical_bridge import CanonicalScannerGateBridge
from aurora_lens.govern.decision import InterventionAction
from aurora_lens.interpret.spacy_backend import SpacyBackend
from aurora_lens.lens import Lens, build_verify_user_grounding_context
from aurora_lens.request_metadata import RequestMetadata
from aurora_lens.verify.checker import Checker
from aurora_lens.verify.flags import FlagType

BASELINE_QUESTION = "How does Personal Baseline change operational mode thresholds?"

_SYNTHETIC_BASELINE_DOC = """\
Personal Baseline adjusts operational mode thresholds by applying Delta = PB - 50.

When Personal Baseline is 75, Delta is 25.

The resulting mode bands are:
Inquiry Mode: 0-45.
Load-Aware Mode: 46-65.
Service Mode: 66-85.
Sovereign Halt: 86 and above.

The State Array is maintained internally while Personal Baseline is active.
"""


@dataclass(frozen=True)
class BaselineCorpus:
    registry: CorpusRegistry
    record_id: str


class _StaticAdapter(LLMAdapter):
    def __init__(self, text: str) -> None:
        self._text = text

    async def generate(self, messages, **kwargs):
        return AdapterResponse(text=self._text, model="mock")


@pytest.fixture
def baseline_corpus(tmp_path: Path) -> BaselineCorpus:
    """Test-owned markdown corpus; no external documents required."""
    record_id = "personal-baseline-spec"
    source = tmp_path / f"{record_id}.md"
    source.write_text(_SYNTHETIC_BASELINE_DOC, encoding="utf-8")
    registry = CorpusRegistry(tmp_path / "corpus")
    ingest_file(registry, record_id=record_id, source_path=source)
    chunks = registry.load_chunks(record_id)
    assert chunks, "synthetic corpus must produce at least one chunk"
    return BaselineCorpus(registry=registry, record_id=record_id)


@pytest.mark.asyncio
async def test_corpus_answer_with_llm_echo_passes_verify(baseline_corpus: BaselineCorpus):
    answer = (
        "When Personal Baseline is active, the LLM should maintain an internal State Array. "
        "Personal Baseline shifts thresholds using Delta = PB - 50. "
        "For PB of 75, Inquiry Mode is 0-45, Load-Aware is 46-65, Service Mode is 66-85, "
        "and Sovereign Halt is 86+."
    )
    q = BASELINE_QUESTION
    chunks = retrieve_chunks(
        baseline_corpus.registry,
        baseline_corpus.record_id,
        max_chars=12000,
        query=q,
    )
    msg = assemble_rag_message(chunks, q)
    rm = RequestMetadata(
        record_ids=(baseline_corpus.record_id,),
        source_scope=("corpus",),
        policy_profile="enterprise_strict",
    )
    bridge = CanonicalScannerGateBridge(mode="enterprise", audit_path=None, default_policy="strict")
    tok1 = request_metadata_var.set(rm)
    tok2 = governance_mode_override_var.set("enterprise")
    try:
        lens = Lens(
            LensConfig(
                adapter=_StaticAdapter(answer),
                extraction_backend=SpacyBackend(model="en_core_web_sm"),
                governance_bridge=bridge,
            )
        )
        result = await lens.process(msg)
        assert result.action == InterventionAction.PASS
        assert "Delta = PB" in result.response.replace("−", "-")
    finally:
        governance_mode_override_var.reset(tok2)
        request_metadata_var.reset(tok1)


@pytest.mark.asyncio
async def test_retrieved_context_grounding_suppresses_unsupported_attribute(
    baseline_corpus: BaselineCorpus,
):
    q = BASELINE_QUESTION
    chunks = retrieve_chunks(
        baseline_corpus.registry,
        baseline_corpus.record_id,
        max_chars=12000,
        query=q,
    )
    msg = assemble_rag_message(chunks, q)
    tok = request_metadata_var.set(
        RequestMetadata(record_ids=(baseline_corpus.record_id,), source_scope=("corpus",))
    )
    try:
        backend = SpacyBackend(model="en_core_web_sm")
        lens = Lens(
            LensConfig(
                adapter=_StaticAdapter("x"),
                extraction_backend=backend,
            )
        )
        await lens._extract_user_turn_for_interpret(msg)
        lens._pef.advance_turn()
        ug = build_verify_user_grounding_context(lens.pef, lens.pef.current_turn, msg, q)
        checker = Checker(backend)
        flags = await checker.check(
            "Inquiry Mode applies to the 0-45 band when Personal Baseline is calibrated.",
            lens.pef,
            user_input=msg,
            user_grounding=ug,
        )
        assert not any(f.flag_type == FlagType.UNSUPPORTED_ATTRIBUTE for f in flags)
    finally:
        request_metadata_var.reset(tok)


@pytest.mark.asyncio
async def test_retrieved_context_requires_relation_level_support_for_pass(
    baseline_corpus: BaselineCorpus,
):
    q = BASELINE_QUESTION
    chunks = retrieve_chunks(
        baseline_corpus.registry,
        baseline_corpus.record_id,
        max_chars=12000,
        query=q,
    )
    msg = assemble_rag_message(chunks, q)
    tok = request_metadata_var.set(
        RequestMetadata(record_ids=(baseline_corpus.record_id,), source_scope=("corpus",))
    )
    try:
        backend = SpacyBackend(model="en_core_web_sm")
        lens = Lens(
            LensConfig(
                adapter=_StaticAdapter("x"),
                extraction_backend=backend,
            )
        )
        await lens._extract_user_turn_for_interpret(msg)
        lens._pef.advance_turn()
        seeded_is = next((r for r in lens.pef.relationships if r.relation == "IS"), None)
        assert seeded_is is not None, "synthetic corpus should seed at least one IS relationship"
        seeded_subject = lens.pef.entities[seeded_is.subject_id].name
        ug = build_verify_user_grounding_context(lens.pef, lens.pef.current_turn, msg, q)
        checker = Checker(backend)
        flags = await checker.check(
            f"{seeded_subject} is fully compliant.",
            lens.pef,
            user_input=msg,
            user_grounding=ug,
        )
        assert any(
            f.flag_type in (FlagType.UNSUPPORTED_ATTRIBUTE, FlagType.UNSUPPORTED_EVENT)
            for f in flags
        ), [f.flag_type.name for f in flags]
    finally:
        request_metadata_var.reset(tok)

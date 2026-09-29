"""Retrieved Context: block → PEF admission (provenance, conflicts, non-RAG parity)."""

from __future__ import annotations

import asyncio
import pytest

from aurora_lens.context import request_metadata_var
from aurora_lens.corpus.models import CorpusChunk, CorpusRecord
from aurora_lens.interpret.base import ExtractionBackend
from aurora_lens.interpret.pef_admission import PEFAdmissionDecision
from aurora_lens.interpret.schema import ExtractedClaim, ExtractionResult
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState
from aurora_lens.rag_pef_admission import (
    PROVENANCE_RETRIEVED_CONTEXT,
    admit_retrieved_context_to_pef,
)
from aurora_lens.request_metadata import RequestMetadata
from aurora_lens.adapters.base import LLMAdapter, AdapterResponse
from aurora_lens.config import LensConfig
from aurora_lens.lens import Lens
from aurora_lens.interpret.spacy_backend import SpacyBackend
from aurora_lens.pef.uncertainty_analysis import pef_uncertainty_analysis


def _claim(
    subject: str,
    relation: str,
    obj: str,
    *,
    evidence: str = "e",
) -> ExtractedClaim:
    return ExtractedClaim(
        subject=subject,
        relation=relation,
        obj=obj,
        span=Span.PRESENT,
        negated=False,
        evidence=evidence,
    )


def _evidence_record(**overrides) -> CorpusRecord:
    base = dict(
        record_id="r1",
        title="Synthetic policy",
        source_path="tmp/policy.md",
        sha256="abc",
        ingested_at="2026-06-30T00:00:00+00:00",
        chunk_count=1,
        authority_class="corporate_policy",
        status="approved",
        effective_from="2026-01-01",
        effective_until="2026-12-31",
        scope="policy",
        parser_status="ok",
    )
    base.update(overrides)
    return CorpusRecord(**base)


def _threshold_chunk(record_id: str, ordinal: int, value: str, scope: str = "policy") -> CorpusChunk:
    return CorpusChunk.build(
        record_id=record_id,
        ordinal=ordinal,
        locator=f"### {scope}",
        text=f"{scope} approval threshold is {value}.",
    )


def _evidence_chunks(record_id: str = "r1") -> list[CorpusChunk]:
    return [
        CorpusChunk.build(
            record_id=record_id,
            ordinal=0,
            locator="### Section A",
            text="Mina is in Osaka.",
        )
    ]


def test_admit_single_is_literal_tags_provenance_and_record():
    pef = PEFState()
    pef.current_turn = 1
    ext = ExtractionResult(
        claims=[_claim("Nora", "is", "Calgary")],
        entity_mentions=["Nora"],
        span=Span.PRESENT,
    )
    tok = request_metadata_var.set(RequestMetadata(record_ids=("rec_doc_9",)))
    try:
        admit_retrieved_context_to_pef(
            ext,
            pef,
            context_block="### Section A\nNora is in Calgary.",
            request_metadata=RequestMetadata(record_ids=("rec_doc_9",)),
            evidence_chunks=[
                CorpusChunk.build(
                    record_id="rec_doc_9",
                    ordinal=0,
                    locator="### Section A",
                    text="Nora is in Calgary.",
                )
            ],
            evidence_record=CorpusRecord(
                record_id="rec_doc_9",
                title="Synthetic record",
                source_path="tmp/synthetic.md",
                sha256="abc123",
                ingested_at="2026-06-30T00:00:00+00:00",
                chunk_count=1,
                authority_class="corporate_policy",
                status="approved",
                parser_status="ok",
            ),
            require_authority=True,
        )
    finally:
        request_metadata_var.reset(tok)

    assert len(pef.relationships) == 1
    r = pef.relationships[0]
    assert r.provenance == PROVENANCE_RETRIEVED_CONTEXT
    assert r.document_id == "rec_doc_9"
    assert r.document_locator == "### Section A"
    assert r.object_literal == "Calgary"
    assert not pef.retrieval_unresolved


def test_admit_retrieved_context_with_admissible_evidence_still_commits():
    pef = PEFState()
    pef.current_turn = 1
    ext = ExtractionResult(
        claims=[_claim("Mina", "is", "Osaka")],
        entity_mentions=["Mina"],
        span=Span.PRESENT,
    )
    result = admit_retrieved_context_to_pef(
        ext,
        pef,
        context_block="### Section A\nMina is in Osaka.",
        request_metadata=RequestMetadata(record_ids=("r1",)),
        evidence_chunks=_evidence_chunks(),
        evidence_record=_evidence_record(),
        request_scope="policy",
    )
    assert result.decision == PEFAdmissionDecision.ADMIT
    assert any(r.object_literal == "Osaka" for r in pef.relationships)
    assert pef.retrieval_unresolved == []


def test_admit_retrieved_context_blocks_draft_evidence_authority_missing():
    pef = PEFState()
    pef.current_turn = 1
    ext = ExtractionResult(
        claims=[_claim("Mina", "is", "Osaka")],
        entity_mentions=["Mina"],
        span=Span.PRESENT,
    )
    result = admit_retrieved_context_to_pef(
        ext,
        pef,
        context_block="### Section A\nMina is in Osaka.",
        request_metadata=RequestMetadata(record_ids=("r1",)),
        evidence_chunks=_evidence_chunks(),
        evidence_record=_evidence_record(status="draft"),
    )
    assert result.decision == PEFAdmissionDecision.REJECT
    assert pef.relationships == []
    assert pef.retrieval_unresolved[0]["kind"] == "evidence_inadmissible"
    assert pef.retrieval_unresolved[0]["failure_kind"] == "authority_missing"


def test_admit_retrieved_context_holds_when_authority_unknown_required():
    pef = PEFState()
    pef.current_turn = 1
    ext = ExtractionResult(
        claims=[_claim("Mina", "is", "Osaka")],
        entity_mentions=["Mina"],
        span=Span.PRESENT,
    )
    result = admit_retrieved_context_to_pef(
        ext,
        pef,
        context_block="### Section A\nMina is in Osaka.",
        request_metadata=RequestMetadata(record_ids=("r1",)),
        evidence_chunks=_evidence_chunks(),
        evidence_record=_evidence_record(authority_class=None),
        require_authority=True,
    )
    assert result.decision == PEFAdmissionDecision.HOLD
    assert pef.relationships == []
    assert pef.retrieval_unresolved[0]["kind"] == "evidence_unresolved"
    assert pef.retrieval_unresolved[0]["failure_kind"] == "authority_unknown"


def test_admit_retrieved_context_passes_canonical_freshness_failure_meta():
    pef = PEFState()
    pef.current_turn = 1
    ext = ExtractionResult(
        claims=[_claim("Mina", "is", "Osaka")],
        entity_mentions=["Mina"],
        span=Span.PRESENT,
    )
    result = admit_retrieved_context_to_pef(
        ext,
        pef,
        context_block="### Section A\nMina is in Osaka.",
        request_metadata=RequestMetadata(record_ids=("r1",)),
        evidence_chunks=_evidence_chunks(),
        evidence_record=_evidence_record(effective_until="2026/15/99"),
    )
    assert result.decision == PEFAdmissionDecision.HOLD
    entry = pef.retrieval_unresolved[0]
    assert entry["kind"] == "evidence_unresolved"
    assert entry["failure_kind"] == "missing_freshness"
    meta = entry.get("failure_meta") or {}
    assert meta.get("missing_field") == "effective_until"


def test_admit_retrieved_context_blocks_traceability_failure():
    pef = PEFState()
    pef.current_turn = 1
    ext = ExtractionResult(
        claims=[_claim("Mina", "is", "Osaka")],
        entity_mentions=["Mina"],
        span=Span.PRESENT,
    )
    result = admit_retrieved_context_to_pef(
        ext,
        pef,
        context_block="### Section A\nMina is in Osaka.",
        request_metadata=RequestMetadata(record_ids=("r1",)),
        evidence_chunks=_evidence_chunks(record_id="r2"),
        evidence_record=_evidence_record(record_id="r1"),
    )
    assert result.decision == PEFAdmissionDecision.REJECT
    assert pef.relationships == []
    assert pef.retrieval_unresolved[0]["kind"] == "evidence_inadmissible"
    assert pef.retrieval_unresolved[0]["failure_kind"] == "traceability_failure"


def test_stale_evidence_demotes_to_contextual_signal_with_metadata():
    pef = PEFState()
    pef.current_turn = 1
    ext = ExtractionResult(
        claims=[_claim("Mina", "is", "Osaka")],
        entity_mentions=["Mina"],
        span=Span.PRESENT,
    )
    result = admit_retrieved_context_to_pef(
        ext,
        pef,
        context_block="### Section A\nMina is in Osaka.",
        request_metadata=RequestMetadata(record_ids=("r1",)),
        evidence_chunks=_evidence_chunks(),
        evidence_record=_evidence_record(status="expired", effective_until="2020-01-01"),
    )
    assert result.decision == PEFAdmissionDecision.REJECT
    entry = pef.retrieval_unresolved[0]
    assert entry["kind"] == "evidence_inadmissible"
    assert entry["failure_kind"] == "stale_evidence"
    assert entry["authority_state"] == "signal_only"
    assert entry["freshness_status"] in {"record_status", "effective_until", "stale"}
    assert entry["authority_status"] == "expired"
    assert entry["demotion_reason"] == "stale_evidence"


def test_unresolved_authority_requires_revalidation_with_demotion_metadata():
    pef = PEFState()
    pef.current_turn = 1
    ext = ExtractionResult(
        claims=[_claim("Mina", "is", "Osaka")],
        entity_mentions=["Mina"],
        span=Span.PRESENT,
    )
    result = admit_retrieved_context_to_pef(
        ext,
        pef,
        context_block="### Section A\nMina is in Osaka.",
        request_metadata=RequestMetadata(record_ids=("r1",)),
        evidence_chunks=_evidence_chunks(),
        evidence_record=_evidence_record(authority_class=None),
        require_authority=True,
    )
    assert result.decision == PEFAdmissionDecision.HOLD
    entry = pef.retrieval_unresolved[0]
    assert entry["kind"] == "evidence_unresolved"
    assert entry["failure_kind"] == "authority_unknown"
    assert entry["authority_state"] == "requires_revalidation"
    assert entry["demotion_reason"] == "authority_unknown"


def test_threshold_conflict_same_scope_two_active_records_is_structured():
    pef = PEFState()
    pef.current_turn = 1
    ext = ExtractionResult(
        claims=[_claim("Mina", "is", "Osaka")],
        entity_mentions=["Mina"],
        span=Span.PRESENT,
    )
    rec_a = _evidence_record(record_id="rA", scope="policy", status="approved")
    rec_b = _evidence_record(record_id="rB", scope="policy", status="approved")
    chunks = [
        _threshold_chunk("rA", 0, "$5,000", "policy"),
        _threshold_chunk("rB", 0, "$10,000", "policy"),
    ]
    result = admit_retrieved_context_to_pef(
        ext,
        pef,
        context_block="### policy\nthreshold statements",
        request_metadata=RequestMetadata(record_ids=("rA", "rB")),
        evidence_chunks=chunks,
        evidence_records=[rec_a, rec_b],
        request_scope="policy",
    )
    assert result.decision == PEFAdmissionDecision.ADMIT
    conflicts = [u for u in pef.retrieval_unresolved if u.get("kind") == "threshold_conflict"]
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict["shared_scope"] == "policy"
    assert sorted(conflict["record_ids"]) == ["rA", "rB"]
    assert conflict["reason"] == "incompatible_threshold_values_same_scope"
    assert conflict.get("threshold_key")


def test_threshold_conflict_detection_does_not_reclassify_admissibility():
    pef = PEFState()
    pef.current_turn = 1
    ext = ExtractionResult(
        claims=[_claim("Mina", "is", "Osaka")],
        entity_mentions=["Mina"],
        span=Span.PRESENT,
    )
    rec_a = _evidence_record(record_id="rA", scope="policy", status="approved")
    rec_b = _evidence_record(record_id="rB", scope="policy", status="approved")
    chunks = [
        _threshold_chunk("rA", 0, "$5,000", "policy"),
        _threshold_chunk("rB", 0, "$10,000", "policy"),
    ]
    result = admit_retrieved_context_to_pef(
        ext,
        pef,
        context_block="### policy\nthreshold statements",
        request_metadata=RequestMetadata(record_ids=("rA", "rB")),
        evidence_chunks=chunks,
        evidence_records=[rec_a, rec_b],
        request_scope="policy",
    )
    assert result.decision == PEFAdmissionDecision.ADMIT
    assert not any(u.get("kind") == "evidence_inadmissible" for u in pef.retrieval_unresolved)
    assert not any(u.get("kind") == "evidence_unresolved" for u in pef.retrieval_unresolved)


def test_non_conflicting_same_scope_thresholds_do_not_block():
    pef = PEFState()
    pef.current_turn = 1
    ext = ExtractionResult(
        claims=[_claim("Mina", "is", "Osaka")],
        entity_mentions=["Mina"],
        span=Span.PRESENT,
    )
    rec_a = _evidence_record(record_id="rA", scope="policy", status="approved")
    rec_b = _evidence_record(record_id="rB", scope="policy", status="approved")
    chunks = [
        _threshold_chunk("rA", 0, "$5,000", "policy"),
        _threshold_chunk("rB", 0, "$5,000", "policy"),
    ]
    result = admit_retrieved_context_to_pef(
        ext,
        pef,
        context_block="### policy\nthreshold statements",
        request_metadata=RequestMetadata(record_ids=("rA", "rB")),
        evidence_chunks=chunks,
        evidence_records=[rec_a, rec_b],
        request_scope="policy",
    )
    assert result.decision == PEFAdmissionDecision.ADMIT
    assert not any(u.get("kind") == "threshold_conflict" for u in pef.retrieval_unresolved)


def test_different_scope_thresholds_do_not_conflict():
    pef = PEFState()
    pef.current_turn = 1
    ext = ExtractionResult(
        claims=[_claim("Mina", "is", "Osaka")],
        entity_mentions=["Mina"],
        span=Span.PRESENT,
    )
    rec_a = _evidence_record(record_id="rA", scope="policy", status="approved")
    rec_b = _evidence_record(record_id="rB", scope="finance", status="approved")
    chunks = [
        _threshold_chunk("rA", 0, "$5,000", "policy"),
        _threshold_chunk("rB", 0, "$10,000", "finance"),
    ]
    result = admit_retrieved_context_to_pef(
        ext,
        pef,
        context_block="### mixed\nthreshold statements",
        request_metadata=RequestMetadata(record_ids=("rA", "rB")),
        evidence_chunks=chunks,
        evidence_records=[rec_a, rec_b],
        request_scope="policy",
    )
    assert result.decision == PEFAdmissionDecision.ADMIT
    assert not any(u.get("kind") == "threshold_conflict" for u in pef.retrieval_unresolved)


def test_effective_date_conflict_same_scope_threshold_pair_is_structured():
    pef = PEFState()
    pef.current_turn = 1
    ext = ExtractionResult(
        claims=[_claim("Mina", "is", "Osaka")],
        entity_mentions=["Mina"],
        span=Span.PRESENT,
    )
    rec_a = _evidence_record(
        record_id="rA",
        scope="policy",
        status="approved",
        effective_from="2026-01-01",
        effective_until="2026-03-31",
    )
    rec_b = _evidence_record(
        record_id="rB",
        scope="policy",
        status="approved",
        effective_from="2026-07-01",
        effective_until="2026-12-31",
    )
    chunks = [
        _threshold_chunk("rA", 0, "$5,000", "policy"),
        _threshold_chunk("rB", 0, "$10,000", "policy"),
    ]
    result = admit_retrieved_context_to_pef(
        ext,
        pef,
        context_block="### policy\nthreshold statements",
        request_metadata=RequestMetadata(record_ids=("rA", "rB")),
        evidence_chunks=chunks,
        evidence_records=[rec_a, rec_b],
        request_scope="policy",
    )
    assert result.decision == PEFAdmissionDecision.ADMIT
    conflicts = [u for u in pef.retrieval_unresolved if u.get("kind") == "effective_date_conflict"]
    assert len(conflicts) == 1
    assert conflicts[0]["reason"] == "disjoint_effective_windows_for_active_records"


def test_admit_temporal_arrival_incompatible_strips_claims():
    """Same narrow C1 seam as verify: calendar + last week in joined context claims."""
    pef = PEFState()
    pef.current_turn = 1
    ext = ExtractionResult(
        claims=[
            _claim(
                "Nora Park",
                "RETURN",
                "Landed March 8, finally home",
                evidence="s6",
            ),
            _claim(
                "Nora Park",
                "RETURN",
                "arrived last week",
                evidence="s7",
            ),
        ],
        entity_mentions=["Nora Park"],
        span=Span.PRESENT,
    )
    admit_retrieved_context_to_pef(
        ext,
        pef,
        context_block="chunk text",
        request_metadata=None,
    )
    assert pef.relationships == []
    kinds = [x.get("kind") for x in pef.retrieval_unresolved]
    assert "temporal_arrival_incompatible" in kinds


def test_admit_drops_because_causal_claim():
    pef = PEFState()
    pef.current_turn = 1
    ext = ExtractionResult(
        claims=[
            _claim("Nora", "is", "tired"),
            _claim("Nora", "because", "she ran"),
        ],
        entity_mentions=["Nora"],
        span=Span.PRESENT,
    )
    admit_retrieved_context_to_pef(
        ext,
        pef,
        context_block="Nora is tired because she ran a marathon.",
        request_metadata=None,
    )
    rels = [x.relation for x in pef.relationships]
    assert "BECAUSE" not in rels
    assert any(x.relation == "IS" for x in pef.relationships)


def test_admit_is_literal_conflict_does_not_commit_either_value():
    pef = PEFState()
    pef.current_turn = 1
    ext = ExtractionResult(
        claims=[
            _claim("Yuki", "is", "5", evidence="e1"),
            _claim("Yuki", "is", "7", evidence="e2"),
        ],
        entity_mentions=["Yuki"],
        span=Span.PRESENT,
    )
    admit_retrieved_context_to_pef(
        ext,
        pef,
        context_block="Contradictory lines.",
        request_metadata=None,
    )
    assert not any(r.relation == "IS" and r.object_literal in ("5", "7") for r in pef.relationships)
    assert len(pef.retrieval_unresolved) == 1
    assert pef.retrieval_unresolved[0]["kind"] == "literal_conflict"
    assert set(pef.retrieval_unresolved[0]["literals"]) == {"5", "7"}


def test_admit_ambiguous_pronoun_claim_not_committed():
    pef = PEFState()
    pef.current_turn = 1
    ext = ExtractionResult(
        claims=[_claim("her", "HAS", "Lou")],
        entity_mentions=[],
        ambiguous_referents=["her"],
        span=Span.PRESENT,
    )
    admit_retrieved_context_to_pef(
        ext,
        pef,
        context_block="Her sister is Lou (ambiguous).",
        request_metadata=None,
    )
    assert not pef.relationships


def test_pef_roundtrip_retrieval_unresolved():
    pef = PEFState()
    pef.retrieval_unresolved = [{"kind": "literal_conflict", "subject": "a", "relation": "IS", "literals": ["1", "2"]}]
    d = pef.to_dict()
    p2 = PEFState.from_dict(d)
    assert p2.retrieval_unresolved == pef.retrieval_unresolved


class _BackendNonRag(ExtractionBackend):
    async def extract(self, text: str, pef: PEFState) -> ExtractionResult:
        return ExtractionResult(
            claims=[_claim("Pat", "is", "here")],
            entity_mentions=["Pat"],
            span=Span.PRESENT,
        )


@pytest.mark.asyncio
async def test_non_rag_path_does_not_touch_retrieval_admission():
    class _A(LLMAdapter):
        async def generate(self, messages, **kwargs):
            return AdapterResponse(text="ok", model="m")

    lens = Lens(
        LensConfig(
            adapter=_A(),
            extraction_backend=_BackendNonRag(),
        )
    )
    await lens.process("Pat is here today.")
    assert lens.pef.retrieval_unresolved == []
    rel = lens.pef.relationships[0]
    assert rel.provenance == "user_input"


class _RagBackend(ExtractionBackend):
    """First call = context block; second = question line."""

    def __init__(self, ctx_claims: list[ExtractedClaim], q_claims: list[ExtractedClaim]):
        self._calls: list[list[ExtractedClaim]] = [ctx_claims, q_claims]
        self._i = 0

    async def extract(self, text: str, pef: PEFState) -> ExtractionResult:
        if self._i < len(self._calls):
            claims = self._calls[self._i]
            self._i += 1
        else:
            claims = []
        return ExtractionResult(
            claims=claims,
            entity_mentions=[c.subject for c in claims],
            span=Span.PRESENT,
        )


@pytest.mark.asyncio
async def test_rag_context_claim_has_retrieved_provenance():
    class _A(LLMAdapter):
        async def generate(self, messages, **kwargs):
            return AdapterResponse(text="ok", model="m")

    ctx_claims = [_claim("Mina", "is", "Osaka")]
    tok = request_metadata_var.set(RequestMetadata(record_ids=("r1",)))
    try:
        lens = Lens(
            LensConfig(
                adapter=_A(),
                extraction_backend=_RagBackend(ctx_claims, []),
            )
        )
        await lens.process(
            "Context:\n### Part 1\nMina is in Osaka.\n\nQuestion: Where is she?"
        )
    finally:
        request_metadata_var.reset(tok)

    is_rels = [r for r in lens.pef.relationships if r.relation == "IS" and r.object_literal == "Osaka"]
    assert len(is_rels) == 1
    assert is_rels[0].provenance == PROVENANCE_RETRIEVED_CONTEXT


@pytest.mark.asyncio
async def test_rag_json_partition_blocks_adjacent_key_subject_object_bleed():
    class _A(LLMAdapter):
        async def generate(self, messages, **kwargs):
            return AdapterResponse(text="ok", model="m")

    lens = Lens(LensConfig(adapter=_A()))
    msg = (
        "Context:\n"
        "FILE: 02_OPSWATCH_CONSEQUENCE_EVIDENCE_OBJECT.json\n"
        "{\n"
        '  "asserted_fact": "Mina is in Osaka.",\n'
        '  "claim_under_evaluation": "Customer received/settled AUD 480.00 refund for order-78421 as a consequence of refund-A.",\n'
        '  "workflow_constraint": "A later decision requiring either proposition needs additional evidence or a separately authorized policy path that does not silently resolve this uncertainty.",\n'
        '  "historical_integrity": "Later evidence may be referenced by a new determination but must not rewrite the state of knowledge at this cutoff."\n'
        "}\n\n"
        "Question: Where is Mina?"
    )
    await lens.process(msg)

    assert not any(
        r.document_locator and "#workflow_constraint" in r.document_locator
        for r in lens.pef.relationships
    )
    assert not any(
        r.document_locator and "#historical_integrity" in r.document_locator
        for r in lens.pef.relationships
    )

    at_rels = [
        r for r in lens.pef.relationships
        if r.relation == "AT" and str(r.object_literal).lower() == "osaka"
    ]
    assert len(at_rels) == 1
    assert at_rels[0].document_id == "02_OPSWATCH_CONSEQUENCE_EVIDENCE_OBJECT.json"
    assert at_rels[0].document_locator is not None
    assert at_rels[0].document_locator.endswith("#asserted_fact")

    for rel in lens.pef.relationships:
        subj = lens.pef.entities.get(rel.subject_id)
        subj_text = subj.name if subj is not None else ""
        obj_text = str(rel.object_literal) if rel.object_literal is not None else ""
        assert "historical_integrity" not in subj_text
        assert '",\n' not in subj_text
        assert '{"' not in obj_text
        assert '"historical_integrity"' not in obj_text

    customer = lens.pef.find_entity_by_name("Customer")
    if customer is not None:
        assert not any(
            r.relation == "RECEIVE"
            and "settled aud 480.00 refund" in str(r.object_literal).lower()
            and not r.negated
            for r in lens.pef.get_relationships_for_subject(customer.id)
        )


@pytest.mark.asyncio
async def test_rag_json_partition_does_not_admit_policy_or_metadata_fields_as_facts():
    class _A(LLMAdapter):
        async def generate(self, messages, **kwargs):
            return AdapterResponse(text="ok", model="m")

    lens = Lens(LensConfig(adapter=_A()))
    msg = (
        "Context:\n"
        "FILE: 02_GOVERNING_POLICY.json\n"
        "{\n"
        '  "policy_id": "refund-policy-v7",\n'
        '  "rules": [\n'
        '    {"id":"R1","text":"A refund requires current refund authority and approval bound to the exact order, amount and payment instrument."},\n'
        '    {"id":"R2","text":"Provider acceptance of a request is not, by itself, evidence that customer funds were settled."}\n'
        "  ]\n"
        "}\n\n"
        "FILE: 02_OPSWATCH_CONSEQUENCE_EVIDENCE_OBJECT.json\n"
        "{\n"
        '  "epistemic_limit": "Evidence establishes dispatch and provider acceptance but neither final customer settlement nor confirmed non-execution.",\n'
        '  "workflow_constraint": "A later decision requiring either proposition needs additional evidence."\n'
        "}\n\n"
        "Question: What does the evidence establish?"
    )
    await lens.process(msg)

    assert any(
        r.document_locator and "#epistemic_limit" in r.document_locator
        for r in lens.pef.relationships
    )
    assert not any(
        r.document_locator and "#rules[" in r.document_locator
        for r in lens.pef.relationships
    )
    assert not any(
        r.document_locator and "#policy_id" in r.document_locator
        for r in lens.pef.relationships
    )
    assert not any(
        r.document_locator and "#workflow_constraint" in r.document_locator
        for r in lens.pef.relationships
    )


@pytest.mark.asyncio
async def test_rag_json_partition_skips_malformed_json_without_partial_claims():
    class _A(LLMAdapter):
        async def generate(self, messages, **kwargs):
            return AdapterResponse(text="ok", model="m")

    lens = Lens(LensConfig(adapter=_A()))
    msg = (
        "Context:\n"
        "FILE: broken_record.json\n"
        "{\n"
        '  "asserted_fact": "Mina is in Osaka.",\n'
        '  "historical_integrity": "Later evidence may be referenced by a new determination"\n'
        "\n\n"
        "Question: Where is Mina?"
    )
    await lens.process(msg)

    assert not any(r.provenance == PROVENANCE_RETRIEVED_CONTEXT for r in lens.pef.relationships)


@pytest.mark.asyncio
async def test_rag_plain_prose_context_still_admits_retrieved_claims():
    class _A(LLMAdapter):
        async def generate(self, messages, **kwargs):
            return AdapterResponse(text="ok", model="m")

    lens = Lens(LensConfig(adapter=_A()))
    await lens.process("Context:\n### Section A\nMina is in Osaka.\n\nQuestion: Where is Mina?")
    assert any(
        r.relation == "AT"
        and str(r.object_literal).lower() == "osaka"
        and r.provenance == PROVENANCE_RETRIEVED_CONTEXT
        for r in lens.pef.relationships
    )


def test_admit_non_execution_scope_phrase_keeps_uncertainty_without_positive_commitment():
    spacy = pytest.importorskip("spacy")

    pef = PEFState()
    pef.current_turn = 1
    backend = SpacyBackend(nlp=spacy.load("en_core_web_sm"))
    text = (
        "Evidence establishes dispatch and provider acceptance but neither final customer "
        "settlement nor confirmed non-execution."
    )
    ext = asyncio.run(backend.extract(text, pef))
    admit_retrieved_context_to_pef(
        ext,
        pef,
        context_block=text,
        request_metadata=None,
    )

    evidence = pef.find_entity_by_name("Evidence")
    assert evidence is not None
    rels = [
        r for r in pef.get_relationships_for_subject(evidence.id)
        if r.relation == "CONFIRM"
    ]
    assert len(rels) == 1
    assert str(rels[0].object_literal).lower() == "non-execution"
    assert rels[0].negated is True
    assert not any(
        (not r.negated) and str(r.object_literal).lower() in {"non", "-", "execution", "non-execution"}
        for r in rels
    )

    records = pef_uncertainty_analysis(pef)
    assert not any(
        r["kind"] == "conflicting_models" and "Evidence.CONFIRM" in r.get("bears_on", [])
        for r in records
    )


def test_admit_positive_confirmed_non_execution_commits_single_positive_value():
    spacy = pytest.importorskip("spacy")

    pef = PEFState()
    pef.current_turn = 1
    backend = SpacyBackend(nlp=spacy.load("en_core_web_sm"))
    text = "Evidence confirmed non-execution."
    ext = asyncio.run(backend.extract(text, pef))
    admit_retrieved_context_to_pef(
        ext,
        pef,
        context_block=text,
        request_metadata=None,
    )
    evidence = pef.find_entity_by_name("Evidence")
    assert evidence is not None
    rels = [
        r for r in pef.get_relationships_for_subject(evidence.id)
        if r.relation == "CONFIRM"
    ]
    assert len(rels) == 1
    assert str(rels[0].object_literal).lower() == "non-execution"
    assert rels[0].negated is False


def test_genuine_incompatible_confirm_values_still_generate_conflicting_models():
    pef = PEFState()
    pef.current_turn = 1
    ext = ExtractionResult(
        claims=[
            _claim("Evidence", "CONFIRM", "settled", evidence="a"),
            _claim("Evidence", "CONFIRM", "not_settled", evidence="b"),
        ],
        entity_mentions=["Evidence"],
        span=Span.PRESENT,
    )
    admit_retrieved_context_to_pef(
        ext,
        pef,
        context_block="Conflicting confirmation lines.",
        request_metadata=None,
    )
    records = pef_uncertainty_analysis(pef)
    conflict = next(
        r for r in records
        if r["kind"] == "conflicting_models" and "Evidence.CONFIRM" in r.get("bears_on", [])
    )
    assert "settled" in conflict["description"]


def test_claim_under_evaluation_is_not_committed_as_asserted_fact():
    spacy = pytest.importorskip("spacy")

    pef = PEFState()
    pef.current_turn = 1
    backend = SpacyBackend(nlp=spacy.load("en_core_web_sm"))
    context = (
        '{\n'
        '  "claim_under_evaluation": "Customer received/settled AUD 480.00 refund for order-78421 as a consequence of refund-A.",\n'
        '  "epistemic_limit": "Evidence establishes dispatch and provider acceptance but neither final customer settlement nor confirmed non-execution."\n'
        '}'
    )
    ext = asyncio.run(backend.extract(context, pef))
    admit_retrieved_context_to_pef(
        ext,
        pef,
        context_block=context,
        request_metadata=None,
    )

    customer = pef.find_entity_by_name("Customer")
    if customer is not None:
        assert not any(
            r.relation == "RECEIVE" and str(r.object_literal).lower() == "settled aud 480.00 refund"
            for r in pef.get_relationships_for_subject(customer.id)
        )


def test_epistemic_limit_admission_scope_and_no_json_object_bleed():
    spacy = pytest.importorskip("spacy")

    pef = PEFState()
    pef.current_turn = 1
    backend = SpacyBackend(nlp=spacy.load("en_core_web_sm"))
    context = (
        '{\n'
        '  "supporting_evidence_refs": ["03_AUTHORITY_ACTION_A.json", "04_APPROVAL_ACTION_A.json"],\n'
        '  "epistemic_limit": "Evidence establishes dispatch and provider acceptance but neither final customer settlement nor confirmed non-execution."\n'
        '}'
    )
    ext = asyncio.run(backend.extract(context, pef))
    admit_retrieved_context_to_pef(
        ext,
        pef,
        context_block=context,
        request_metadata=None,
    )

    establish = [r for r in pef.relationships if r.relation == "ESTABLISH"]
    assert any(
        "dispatch and provider acceptance" in str(r.object_literal).lower() and not r.negated
        for r in establish
    )
    assert not any(
        any(marker in str(r.object_literal).lower() for marker in (".json", "file:", "\n", "[", "{"))
        for r in establish
    )
    confirm = [r for r in pef.relationships if r.relation == "CONFIRM"]
    assert any(str(r.object_literal).lower() == "non-execution" and r.negated for r in confirm)
    assert not any(str(r.object_literal).lower() == "non-execution" and not r.negated for r in confirm)
    assert not any(
        "settlement" in str(r.object_literal).lower() and not r.negated
        for r in pef.relationships
        if r.relation in {"ESTABLISH", "CONFIRM", "RECEIVE"}
    )


def test_ordinary_asserted_customer_receipt_still_admits():
    spacy = pytest.importorskip("spacy")

    pef = PEFState()
    pef.current_turn = 1
    backend = SpacyBackend(nlp=spacy.load("en_core_web_sm"))
    context = '{"observed_fact": "Customer received settled AUD 480.00 refund."}'
    ext = asyncio.run(backend.extract(context, pef))
    admit_retrieved_context_to_pef(
        ext,
        pef,
        context_block=context,
        request_metadata=None,
    )
    customer = pef.find_entity_by_name("Customer")
    assert customer is not None
    assert any(
        r.relation == "RECEIVE" and "settled aud 480.00 refund" in str(r.object_literal).lower() and not r.negated
        for r in pef.get_relationships_for_subject(customer.id)
    )


def test_genuinely_negated_assertions_remain_negated():
    spacy = pytest.importorskip("spacy")

    pef = PEFState()
    pef.current_turn = 1
    backend = SpacyBackend(nlp=spacy.load("en_core_web_sm"))
    context = '{"observed_fact": "Evidence did not confirm settlement."}'
    ext = asyncio.run(backend.extract(context, pef))
    admit_retrieved_context_to_pef(
        ext,
        pef,
        context_block=context,
        request_metadata=None,
    )
    evidence = pef.find_entity_by_name("Evidence")
    assert evidence is not None
    assert any(
        r.relation == "CONFIRM" and "settlement" in str(r.object_literal).lower() and r.negated
        for r in pef.get_relationships_for_subject(evidence.id)
    )

"""RAG-shaped Context: / Question: split and retrieval-aware activation rules."""

from aurora_lens.adapters.base import LLMAdapter, AdapterResponse
from aurora_lens.context import request_metadata_var
from aurora_lens.rag_activation import (
    effective_rag_retrieval_aware_referents,
    evidence_bearing_request_metadata,
)
from aurora_lens.request_metadata import RequestMetadata
from aurora_lens.config import LensConfig
from aurora_lens.interpret.schema import ExtractedClaim, ExtractionResult
from aurora_lens.lens import Lens, split_rag_context_question
from aurora_lens.pef.entity import Entity
from aurora_lens.pef.span import Span


def test_split_rag_context_question_basic():
    s = "Context:\n### Section 2\nNora lives here.\n\nQuestion: What city was she in?"
    out = split_rag_context_question(s)
    assert out is not None
    ctx, q = out
    assert "Nora" in ctx
    assert q.startswith("What city")


def test_split_rag_context_question_rejects_non_rag():
    assert split_rag_context_question("Hello world") is None
    assert split_rag_context_question("Context:\nonly") is None


class TestRagActivationRules:
    """effective_rag_retrieval_aware_referents: auto path + env overrides."""

    def test_auto_off_plain_chat(self, monkeypatch):
        monkeypatch.delenv("AURORA_LENS_RAG_RETRIEVAL_AWARE_REFERENTS", raising=False)
        assert effective_rag_retrieval_aware_referents("Hello there") is False

    def test_auto_on_context_question_shape(self, monkeypatch):
        monkeypatch.delenv("AURORA_LENS_RAG_RETRIEVAL_AWARE_REFERENTS", raising=False)
        s = "Context:\n### Section 1\nNora lives here.\n\nQuestion: What city was she in?"
        assert effective_rag_retrieval_aware_referents(s) is True

    def test_auto_on_record_ids_metadata(self, monkeypatch):
        monkeypatch.delenv("AURORA_LENS_RAG_RETRIEVAL_AWARE_REFERENTS", raising=False)
        tok = request_metadata_var.set(RequestMetadata(record_ids=("doc_1",)))
        try:
            assert effective_rag_retrieval_aware_referents("Hello") is True
        finally:
            request_metadata_var.reset(tok)

    def test_auto_on_source_scope_metadata(self, monkeypatch):
        monkeypatch.delenv("AURORA_LENS_RAG_RETRIEVAL_AWARE_REFERENTS", raising=False)
        tok = request_metadata_var.set(RequestMetadata(source_scope=("attached_files",)))
        try:
            assert effective_rag_retrieval_aware_referents("Hello") is True
        finally:
            request_metadata_var.reset(tok)

    def test_force_off_overrides_metadata_and_shape(self, monkeypatch):
        monkeypatch.setenv("AURORA_LENS_RAG_RETRIEVAL_AWARE_REFERENTS", "0")
        s = "Context:\nfoo\n\nQuestion: What?"
        assert effective_rag_retrieval_aware_referents(s) is False
        tok = request_metadata_var.set(RequestMetadata(record_ids=("r1",)))
        try:
            assert effective_rag_retrieval_aware_referents("Hi") is False
        finally:
            request_metadata_var.reset(tok)

    def test_force_on_plain_chat(self, monkeypatch):
        monkeypatch.setenv("AURORA_LENS_RAG_RETRIEVAL_AWARE_REFERENTS", "1")
        assert effective_rag_retrieval_aware_referents("Hello") is True

    def test_config_rag_true_in_auto_mode(self, monkeypatch):
        monkeypatch.delenv("AURORA_LENS_RAG_RETRIEVAL_AWARE_REFERENTS", raising=False)
        assert effective_rag_retrieval_aware_referents("Hello", config_rag=True) is True


def test_evidence_bearing_metadata_workspace_only_false():
    assert evidence_bearing_request_metadata(RequestMetadata(workspace_id="ws-1")) is False


def test_lensconfig_rag_flag_defaults_off():
    class _A(LLMAdapter):
        async def generate(self, messages, **kwargs):
            return AdapterResponse(text="x", model="t")

    c = LensConfig(adapter=_A())
    assert c.rag_retrieval_aware_referents is False


def _lens_with_mock_adapter():
    class _A(LLMAdapter):
        async def generate(self, messages, **kwargs):
            return AdapterResponse(text="x", model="t")

    return Lens(LensConfig(adapter=_A()))


def test_compute_truly_ambiguous_rag_canadian_demonym_not_second_anchor():
    """Live O1 shape: spaCy emits *Canadian* + *Nora*; *Canadian* must not block suppression.

    ``What Canadian city was Nora … she …`` matches *Canadian* as a word but it is not a
    PEF entity; only PEF-linked names count as anchors (O1/T2 seam).
    """
    lens = _lens_with_mock_adapter()
    e1 = Entity.create("Nora Park", turn=1)
    e2 = Entity.create("Emma Chen", turn=1)
    lens.pef.entities[e1.id] = e1
    lens.pef.entities[e2.id] = e2
    q_o1 = "What Canadian city was Nora based in before she returned?"
    ext = ExtractionResult(
        claims=[],
        entity_mentions=["Canadian", "Nora"],
        ambiguous_referents=["she"],
        span=Span.PRESENT,
    )
    assert lens._compute_truly_ambiguous_referents(ext) == ["she"]
    assert lens._compute_truly_ambiguous_referents(ext, rag_question_line=q_o1) == []


def test_compute_truly_ambiguous_rag_question_unique_name_suppresses_she_false_positive():
    """O1/T2-style: *she* + multiple PEF names, but question names one entity only.

    Without RAG question line, structural fallback still flags *she*; with the line,
    the single named anchor in the question suppresses the false positive.
    """
    lens = _lens_with_mock_adapter()
    e1 = Entity.create("Nora Park", turn=1)
    e2 = Entity.create("Emma Chen", turn=1)
    lens.pef.entities[e1.id] = e1
    lens.pef.entities[e2.id] = e2
    q_o1 = (
        "What Canadian city was Nora based in before she returned?"
    )
    ext = ExtractionResult(
        claims=[
            ExtractedClaim(
                subject="she",
                relation="RETURN",
                obj="x",
                span=Span.PRESENT,
                negated=False,
                evidence="test",
            ),
        ],
        entity_mentions=["Nora", "Emma", "Vancouver"],
        ambiguous_referents=["she"],
        span=Span.PRESENT,
    )
    assert lens._compute_truly_ambiguous_referents(ext) == ["she"]
    assert lens._compute_truly_ambiguous_referents(ext, rag_question_line=q_o1) == []


def test_compute_truly_ambiguous_rag_t2_nora_possessive_suppresses_she():
    """Manifest T2: *Nora's* … *she* — first-token match on multi-word entity name."""
    lens = _lens_with_mock_adapter()
    e1 = Entity.create("Nora Park", turn=1)
    e2 = Entity.create("Emma Chen", turn=1)
    lens.pef.entities[e1.id] = e1
    lens.pef.entities[e2.id] = e2
    q_t2 = (
        "What is Nora's employer, and which Canadian city was she based in before returning?"
    )
    ext = ExtractionResult(
        claims=[
            ExtractedClaim(
                subject="she",
                relation="AT",
                obj="x",
                span=Span.PRESENT,
                negated=False,
                evidence="test",
            ),
        ],
        entity_mentions=["Nora", "Emma", "Vancouver"],
        ambiguous_referents=["she"],
        span=Span.PRESENT,
    )
    assert lens._compute_truly_ambiguous_referents(ext, rag_question_line=q_t2) == []


def test_compute_truly_ambiguous_rag_question_two_names_still_ambiguous():
    """If two candidate names appear in the question, do not suppress (no broadening)."""
    lens = _lens_with_mock_adapter()
    e1 = Entity.create("Nora Park", turn=1)
    e2 = Entity.create("Emma Chen", turn=1)
    lens.pef.entities[e1.id] = e1
    lens.pef.entities[e2.id] = e2
    q = "What did Nora tell Emma before she left?"
    ext = ExtractionResult(
        claims=[
            ExtractedClaim(
                subject="she",
                relation="TELL",
                obj="x",
                span=Span.PRESENT,
                negated=False,
                evidence="test",
            ),
        ],
        entity_mentions=["Nora", "Emma"],
        ambiguous_referents=["she"],
        span=Span.PRESENT,
    )
    assert lens._compute_truly_ambiguous_referents(ext, rag_question_line=q) == ["she"]

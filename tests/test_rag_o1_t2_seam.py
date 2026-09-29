"""Focused tests: manifest O1/T2 RAG seam (markdown context extraction + verify).

O1/T2: harness markdown ``**bold**`` around phrases like *based in Vancouver*
prevented spaCy from emitting ``AT`` facts into PEF; verify then failed supported
answers (``UNSUPPORTED_EVENT`` / ``she`` fallout). Stripping wrappers before
context extraction restores admission; discourse bindings + resolved claims then
pass.
"""

from __future__ import annotations

import pytest

from tests.skip_reasons import SKIP_SPACY_MODULE

from aurora_lens.interpret.schema import ExtractedClaim
from aurora_lens.lens import (
    _strip_markdown_bold_wrappers_for_rag_extraction,
    apply_rag_verify_discourse_bindings,
)
from aurora_lens.pef.entity import Entity
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState, Relationship, canonicalize_relation
from aurora_lens.rag_pef_admission import admit_retrieved_context_to_pef
from aurora_lens.verify.checker import Checker
from aurora_lens.verify.flags import FlagType


# Section 2-shaped body: bold wrappers match eval/rag_synthetic_mini_document.md
_O1_MARKDOWN_CONTEXT = (
    "City records list **Nora Park** as Emma Chen's sibling. "
    "Before returning, Nora was **based in Vancouver** for two years. "
    "Nora now uses Emma's address for mail."
)


@pytest.fixture
def checker():
    pytest.importorskip("spacy", reason=SKIP_SPACY_MODULE)
    from aurora_lens.interpret.spacy_backend import SpacyBackend

    return Checker(SpacyBackend(model="en_core_web_sm"))


def test_strip_markdown_bold_unwraps_paired_asterisks():
    s = "Nora was **based in Vancouver** before she returned."
    assert _strip_markdown_bold_wrappers_for_rag_extraction(s) == (
        "Nora was based in Vancouver before she returned."
    )


@pytest.mark.asyncio
async def test_o1_bold_markdown_context_admits_at_vancouver():
    pytest.importorskip("spacy", reason=SKIP_SPACY_MODULE)
    from aurora_lens.interpret.spacy_backend import SpacyBackend

    pef = PEFState()
    pef.current_turn = 1
    backend = SpacyBackend(model="en_core_web_sm")
    plain = _strip_markdown_bold_wrappers_for_rag_extraction(_O1_MARKDOWN_CONTEXT)
    ext = await backend.extract(plain, pef)
    # TODO(stage-3b-callsite): this seam only has rendered context text.
    # Pass evidence_chunks/evidence_record once this test path supplies structured corpus evidence.
    admit_retrieved_context_to_pef(
        ext,
        pef,
        context_block=_O1_MARKDOWN_CONTEXT,
        request_metadata=None,
    )
    at_van = [
        r
        for r in pef.relationships
        if canonicalize_relation(r.relation) == "AT"
        and str(r.object_literal).lower() == "vancouver"
    ]
    assert at_van, "Expected AT Vancouver admitted after bold strip + extraction"


@pytest.mark.asyncio
async def test_o1_verify_nora_at_vancouver_no_unsupported_event():
    pytest.importorskip("spacy", reason=SKIP_SPACY_MODULE)
    from aurora_lens.interpret.spacy_backend import SpacyBackend

    pef = PEFState()
    pef.current_turn = 1
    backend = SpacyBackend(model="en_core_web_sm")
    checker = Checker(backend)
    plain = _strip_markdown_bold_wrappers_for_rag_extraction(_O1_MARKDOWN_CONTEXT)
    ext = await backend.extract(plain, pef)
    # TODO(stage-3b-callsite): this seam only has rendered context text.
    # Pass evidence_chunks/evidence_record once this test path supplies structured corpus evidence.
    admit_retrieved_context_to_pef(
        ext,
        pef,
        context_block=_O1_MARKDOWN_CONTEXT,
        request_metadata=None,
    )
    claim = ExtractedClaim(
        subject="Nora",
        relation="AT",
        obj="Vancouver",
        span=Span.PAST,
        negated=False,
        evidence="Nora was based in Vancouver before she returned.",
        provenance="llm_output",
        extractor_backend="spacy",
    )
    flags = checker._check_claim(claim, pef, Span.PAST)
    types = [f.flag_type for f in flags]
    assert FlagType.UNSUPPORTED_EVENT not in types, f"unexpected flags: {types}"


def test_t2_style_she_at_vancouver_clean_when_bound_and_pef_has_at(checker):
    """When *she* binds to Nora and PEF already has AT Vancouver, verify passes."""
    pef = PEFState()
    pef.current_turn = 1
    n = Entity.create("Nora Park", turn=1)
    n.resolved = True
    h = Entity.create("Harbor Labs", turn=1)
    h.resolved = True
    pef.add_entity(n)
    pef.add_entity(h)
    pef.add_relationship(
        Relationship(
            subject_id=n.id,
            relation="AT",
            object_entity_id=None,
            object_literal="Vancouver",
            span=Span.PAST,
            source_turn=1,
            evidence="based in Vancouver",
            provenance="retrieved_context",
            extractor_backend="spacy",
        )
    )
    q = (
        "What is Nora's employer, and which Canadian city was she based in "
        "before returning?"
    )
    apply_rag_verify_discourse_bindings(pef, q)
    claim = ExtractedClaim(
        subject="she",
        relation="AT",
        obj="Vancouver",
        span=Span.PAST,
        negated=False,
        evidence="she was based in Vancouver before returning.",
        provenance="llm_output",
        extractor_backend="spacy",
    )
    flags = checker._check_claim(claim, pef, Span.PAST)
    types = [f.flag_type for f in flags]
    assert FlagType.UNRESOLVED_REFERENT not in types, f"got {types}"
    assert FlagType.UNSUPPORTED_EVENT not in types, f"got {types}"

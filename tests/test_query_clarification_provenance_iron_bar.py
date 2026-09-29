"""Iron-bar provenance tests for QUERY residue and clarification authority.

Doctrine: ``docs/query_and_clarification_invariants.md``.

QUERY fingerprint scope is tracked under **Follow-up issues** in that document
(**FU-QUERY-REL** closed; **FU-CLAR-AUTH-RESUME** closed in ``docs/closed-decisions.md``).
Assertions focus on discourse, entity roster, and identity corrections across QUERY
turns---not merely assistant prose.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from aurora_lens.adapters.base import AdapterResponse
from aurora_lens.config import LensConfig
from aurora_lens.govern.canonical_bridge import CanonicalScannerGateBridge
from aurora_lens.govern.decision import InterventionAction
from aurora_lens.interpret.base import ExtractionBackend
from aurora_lens.interpret.schema import ExtractionResult
from aurora_lens.interpret.spacy_backend import SpacyBackend
from aurora_lens.lens import Lens
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState
from aurora_lens.verify.checker import Checker
from aurora_lens.verify.flags import FlagType


class StubLLMAdapter:
    def __init__(self, reply_text: str, model: str = "stub") -> None:
        self._reply_text = reply_text
        self._model = model
        self.call_count = 0

    async def generate(
        self,
        messages: list[dict[str, str]],
        **_kwargs: Any,
    ) -> AdapterResponse:
        self.call_count += 1
        return AdapterResponse(text=self._reply_text, model=self._model)


class EmptyExtractBackend(ExtractionBackend):
    """Forces post-LLM verification path without extractor-marked ambiguity."""

    async def extract(self, text: str, pef: PEFState) -> ExtractionResult:
        del text, pef
        return ExtractionResult(claims=[], entity_mentions=[], span=Span.PRESENT)


@pytest.fixture(scope="module")
def real_backend() -> SpacyBackend:
    return SpacyBackend(model="en_core_web_sm")


@pytest.fixture
def audit_path(tmp_path: Path) -> Path:
    return tmp_path / "iron_bar_audit.jsonl"


@pytest.fixture
def enterprise_bridge(audit_path: Path) -> CanonicalScannerGateBridge:
    return CanonicalScannerGateBridge(
        mode="enterprise",
        audit_path=str(audit_path),
        default_policy="strict",
    )


def _iron_bar_query_residue_fingerprint(pef: PEFState) -> tuple:
    """Fingerprint interpretive residue targets (pronouns / merges / roster churn).

    Narrative QUERY turns may still admit event-shaped relationship writes (e.g.
    ``TELL``) via extraction today; see ``docs/query_and_clarification_invariants.md``
    follow-up **FU-QUERY-REL**. This projection matches the iron bar on discourse,
    roster stability, and identity corrections (Forbidden Residue examples).
    """
    disc = tuple(sorted((k.lower(), v) for k, v in pef.discourse_referent_bindings.items()))
    ents = tuple(sorted((e.name, e.id) for e in pef.entities.values()))
    corrections = tuple(
        sorted(json.dumps(ic, sort_keys=True, default=str) for ic in pef.identity_corrections)
    )
    return ("DISC", disc, "ENTS", ents, "IDCOR", corrections)


@pytest.mark.asyncio
async def test_iron_bar_query_turn_preserves_committed_semantics_emma_lucy(
    real_backend: SpacyBackend,
    enterprise_bridge: CanonicalScannerGateBridge,
) -> None:
    """Two QUERY turns over sibling pronoun ambiguity must not mutate committed substrate."""
    adapter = StubLLMAdapter("must-not-be-used-on-pre-llm-contain")
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=real_backend,
            governance_bridge=enterprise_bridge,
            auto_interpret=True,
            auto_verify=True,
            enable_state_native_delegation=True,
        )
    )

    await lens.process("Emma lives in Austin.")
    await lens.process("Lucy lives in Dallas.")
    await lens.process("Emma is not the same person as Lucy.")

    baseline = _iron_bar_query_residue_fingerprint(lens.pef)

    q1 = "Emma told Lucy that her sister was arriving. Where is she?"
    r1 = await lens.process(q1)
    assert r1.action != InterventionAction.PASS, (
        "Ambiguous QUERY must not silently PASS — residue risk without containment."
    )
    fp1 = _iron_bar_query_residue_fingerprint(lens.pef)
    assert fp1 == baseline, (
        "QUERY turn 1 mutated discourse roster / identity corrections — forbidden residue.\n"
        f"baseline={baseline}\nactual={fp1}"
    )

    q2 = "Where is her sister now?"
    r2 = await lens.process(q2)
    assert r2.action != InterventionAction.PASS, (
        "Follow-up QUERY must not PASS without governed clarification admission."
    )
    fp2 = _iron_bar_query_residue_fingerprint(lens.pef)
    assert fp2 == baseline, (
        "QUERY turn 2 mutated discourse roster / identity corrections.\n"
        f"baseline={baseline}\nactual={fp2}"
    )

    assert adapter.call_count == 0, (
        "Pre-LLM ambiguity containment must not invoke the upstream adapter "
        f"(calls={adapter.call_count})."
    )


@pytest.mark.asyncio
async def test_iron_bar_llm_binding_clarification_requires_containment_not_pass(
    enterprise_bridge: CanonicalScannerGateBridge,
) -> None:
    """Binding clarification prose cannot finalize as PASS outside structured hold."""
    adapter = StubLLMAdapter("Which Emma did you mean?")
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=EmptyExtractBackend(),
            governance_bridge=enterprise_bridge,
            auto_interpret=True,
            auto_verify=True,
            enable_state_native_delegation=True,
        )
    )
    lens.pef.get_or_create_entity("Emma Smith")
    lens.pef.get_or_create_entity("Emma Jones")

    fp_before = _iron_bar_query_residue_fingerprint(lens.pef)

    ui = (
        "Chart A lists Emma Smith as the attending physician; chart B lists Emma Jones. "
        "Where is she working today?"
    )
    result = await lens.process(ui)

    assert result.action == InterventionAction.CONTAIN, (
        f"Expected CONTAIN for ungoverned binding clarification; got {result.action}"
    )
    assert adapter.call_count == 1

    pend = lens.pef.pending_clarification
    assert pend is not None, "Governed clarification must record pending_clarification"
    assert pend.get("failed_constraint") == "CLARIFICATION_AUTHORITY_OUTSIDE_HOLD"
    assert pend.get("candidate_entities"), "clarification candidates must be recorded"
    assert pend.get("ambiguous_referents"), "ambiguous pronoun targets must be fingerprinted"

    assert lens.pef.epistemic_hold is not None

    fp_after = _iron_bar_query_residue_fingerprint(lens.pef)
    assert fp_after == fp_before, (
        "Checker-driven containment must not commit speculative discourse bindings "
        "without admissible replay.\n"
        f"before={fp_before}\nafter={fp_after}"
    )


@pytest.mark.asyncio
async def test_clarification_authority_outside_hold_candidate_binding_resumes(
    enterprise_bridge: CanonicalScannerGateBridge,
) -> None:
    """FU-CLAR-AUTH-RESUME: direct candidate selection clears hold and resumes."""
    adapter = StubLLMAdapter("Which Emma did you mean?")
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=EmptyExtractBackend(),
            governance_bridge=enterprise_bridge,
            auto_interpret=True,
            auto_verify=True,
            enable_state_native_delegation=True,
        )
    )
    lens.pef.get_or_create_entity("Emma Smith")
    lens.pef.get_or_create_entity("Emma Jones")

    ui = (
        "Chart A lists Emma Smith as the attending physician; chart B lists Emma Jones. "
        "Where is she working today?"
    )
    r1 = await lens.process(ui)
    assert r1.action == InterventionAction.CONTAIN
    assert lens.pef.pending_clarification is not None
    assert (
        lens.pef.pending_clarification.get("failed_constraint")
        == "CLARIFICATION_AUTHORITY_OUTSIDE_HOLD"
    )

    r2 = await lens.process("Emma Smith")
    assert lens.pef.pending_clarification is None
    assert r2.action == InterventionAction.PASS
    assert adapter.call_count == 2


@pytest.mark.asyncio
async def test_checker_clarification_authority_duplicate_given_name_gate(
    real_backend: SpacyBackend,
) -> None:
    checker = Checker(real_backend)
    pef = PEFState()
    pef.get_or_create_entity("Emma Smith")
    pef.get_or_create_entity("Emma Jones")

    flags = await checker.check(
        "Which Emma did you mean?",
        pef,
        "Chart A has Emma Smith; chart B has Emma Jones. Where is she?",
    )
    assert any(f.flag_type == FlagType.CLARIFICATION_AUTHORITY_OUTSIDE_HOLD for f in flags)

    flags_clean = await checker.check(
        "Which Emma did you mean?",
        pef,
        "Chart A has Emma Smith only.",
    )
    assert not any(f.flag_type == FlagType.CLARIFICATION_AUTHORITY_OUTSIDE_HOLD for f in flags_clean)


@pytest.mark.asyncio
async def test_checker_skips_when_pending_clarification_already_active(
    real_backend: SpacyBackend,
) -> None:
    checker = Checker(real_backend)
    pef = PEFState()
    pef.get_or_create_entity("Emma Smith")
    pef.get_or_create_entity("Emma Jones")
    pef.pending_clarification = {"original_question": "held"}

    flags = await checker.check(
        "Which Emma did you mean?",
        pef,
        "Chart A has Emma Smith; chart B has Emma Jones. Where is she?",
    )
    assert not any(f.flag_type == FlagType.CLARIFICATION_AUTHORITY_OUTSIDE_HOLD for f in flags)

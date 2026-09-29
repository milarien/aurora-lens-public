"""Tests for epistemic_normalisation (post-check PASS surface pass)."""

from __future__ import annotations

from pathlib import Path

import pytest

from aurora_lens.config import LensConfig
from aurora_lens.govern.canonical_bridge import CanonicalScannerGateBridge
from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
from aurora_lens.govern.epistemic_normalisation import apply_epistemic_normalisation
from aurora_lens.interpret.spacy_backend import SpacyBackend
from aurora_lens.lens import Lens, _governance_decision_norm_signature
from aurora_lens.pef.span import Span
from aurora_lens.verify.flags import FlagType

from tests.test_batch_real_pipeline import StubLLMAdapter
from tests.test_streaming_governance import FakeStreamingAdapter


@pytest.fixture(scope="module")
def real_backend() -> SpacyBackend:
    return SpacyBackend(model="en_core_web_sm")


@pytest.fixture
def audit_path(tmp_path: Path) -> Path:
    return tmp_path / "epistemic_norm_audit.jsonl"


@pytest.fixture
def real_bridge(audit_path: Path) -> CanonicalScannerGateBridge:
    return CanonicalScannerGateBridge(
        mode="enterprise",
        audit_path=str(audit_path),
        default_policy="strict",
    )


def test_apply_epistemic_normalisation_strips_certainty_marker() -> None:
    raw = "Paris is definitely the capital of France."
    out = apply_epistemic_normalisation(raw)
    assert out != raw
    assert "definitely" not in out.lower()


@pytest.mark.asyncio
async def test_maybe_apply_epistemic_normalisation_preserves_decision_signature(
    real_backend: SpacyBackend,
    real_bridge: CanonicalScannerGateBridge,
) -> None:
    lens = Lens(
        LensConfig(
            adapter=StubLLMAdapter("unused"),
            extraction_backend=real_backend,
            governance_bridge=real_bridge,
            auto_interpret=True,
            auto_verify=True,
        )
    )
    decision = GovernanceDecision(
        action=InterventionAction.PASS,
        flags=[],
        rationale="No verification flags",
        policy="strict",
    )
    sig_before = _governance_decision_norm_signature(decision)
    out, applied = await lens._maybe_apply_epistemic_normalisation_clean_pass(
        "The capital of France is obviously Paris.",
        flags=[],
        decision=decision,
        history_user_input="What is the capital of France?",
    )
    assert applied is True
    assert "obviously" not in out.lower()
    assert _governance_decision_norm_signature(decision) == sig_before


@pytest.mark.asyncio
async def test_epistemic_normalisation_on_clean_pass_same_claims_surface_only(
    real_backend: SpacyBackend,
    real_bridge: CanonicalScannerGateBridge,
) -> None:
    raw_body = "The capital of France is definitely Paris."
    lens = Lens(
        LensConfig(
            adapter=StubLLMAdapter(raw_body),
            extraction_backend=real_backend,
            governance_bridge=real_bridge,
            auto_interpret=True,
            auto_verify=True,
        )
    )
    result = await lens.process("What is the capital of France?")
    assert result.action == InterventionAction.PASS
    assert result.decision is not None
    assert result.upstream_model_draft == raw_body
    assert result.epistemic_normalisation_applied is True
    assert result.response != raw_body
    assert "definitely" not in result.response.lower()

    chk_raw = await lens._checker.check(
        result.upstream_model_draft,
        lens.pef,
        user_input="What is the capital of France?",
    )
    chk_out = await lens._checker.check(
        result.response,
        lens.pef,
        user_input="What is the capital of France?",
    )
    assert chk_raw == [] and chk_out == []


@pytest.mark.asyncio
async def test_epistemic_normalisation_skips_on_contain(
    real_backend: SpacyBackend,
    real_bridge: CanonicalScannerGateBridge,
) -> None:
    lens = Lens(
        LensConfig(
            adapter=StubLLMAdapter("unused"),
            extraction_backend=real_backend,
            governance_bridge=real_bridge,
            auto_interpret=True,
            auto_verify=True,
        )
    )
    result = await lens.process(
        "Emma told Anna her sister was overseas. Where is she now?"
    )
    assert result.action == InterventionAction.CONTAIN
    assert result.epistemic_normalisation_applied is False


@pytest.mark.asyncio
async def test_epistemic_normalisation_stream_clean_pass(
    real_backend: SpacyBackend,
    real_bridge: CanonicalScannerGateBridge,
) -> None:
    raw = "Paris is definitely the capital of France."
    adapter = FakeStreamingAdapter(["Paris is ", "definitely ", "the capital of France."])
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=real_backend,
            governance_bridge=real_bridge,
            auto_interpret=True,
            auto_verify=True,
        )
    )
    meta: dict | None = None
    chunks: list[str] = []
    async for kind, payload in lens.process_stream("What is the capital of France?"):
        if kind == "chunk":
            chunks.append(payload[1])
        if kind == "metadata":
            meta = payload
    assert meta is not None
    assert meta.get("epistemic_normalisation_applied") is True
    joined = "".join(chunks)
    assert "definitely" not in joined.lower()
    chk_raw = await lens._checker.check(
        raw,
        lens.pef,
        user_input="What is the capital of France?",
    )
    chk_out = await lens._checker.check(
        joined,
        lens.pef,
        user_input="What is the capital of France?",
    )
    assert chk_raw == [] and chk_out == []


@pytest.mark.asyncio
async def test_real_pipeline_operants_still_fire_with_real_checker_and_bridge(
    real_backend: SpacyBackend,
    real_bridge: CanonicalScannerGateBridge,
) -> None:
    lens_pass = Lens(
        LensConfig(
            adapter=StubLLMAdapter("The capital of France is Paris."),
            extraction_backend=real_backend,
            governance_bridge=real_bridge,
            auto_interpret=True,
            auto_verify=True,
        )
    )
    r_pass = await lens_pass.process("What is the capital of France?")
    assert r_pass.action == InterventionAction.PASS
    assert r_pass.response == "The capital of France is Paris."
    assert r_pass.flags == []

    legal_stub = (
        "You have a strong case here. "
        "Based on the contract terms, you would likely win at trial. "
        "The arguments in your favor are clear: the counterparty breached first."
    )
    lens_legal = Lens(
        LensConfig(
            adapter=StubLLMAdapter(legal_stub),
            extraction_backend=real_backend,
            governance_bridge=real_bridge,
            auto_interpret=True,
            auto_verify=True,
        )
    )
    r_legal = await lens_legal.process(
        "I am being sued for breach of contract. Should I fight it in court?"
    )
    assert r_legal.action == InterventionAction.FORCE_REVISE
    assert not any(
        f.flag_type == FlagType.UNRESOLVED_REFERENT for f in r_legal.flags
    ), "Non-consequential pronoun should not preempt legal governance"
    assert r_legal.decision is not None
    assert r_legal.decision.pathway_id == "P_REFUSE_ESCALATE_PRO"
    assert any(
        f.flag_type == FlagType.PERSONALIZED_LEGAL_ADVICE for f in r_legal.flags
    )
    gov_legal = r_legal.decision.governed_response or r_legal.response
    assert "you would likely win" not in gov_legal.lower()

    lens_amb = Lens(
        LensConfig(
            adapter=StubLLMAdapter("unused"),
            extraction_backend=real_backend,
            governance_bridge=real_bridge,
            auto_interpret=True,
            auto_verify=True,
        )
    )
    r_amb = await lens_amb.process(
        "Emma told Anna her sister was overseas. Where is she now?"
    )
    assert r_amb.action == InterventionAction.CONTAIN


def test_batch_build_result_includes_epistemic_normalisation_fields() -> None:
    from aurora_lens.lens import LensResult
    from aurora_lens.scripts.batch import _build_result

    lr = LensResult(
        response="normalised",
        flags=[],
        pef_snapshot="",
        turn=1,
        span=Span.PRESENT,
        upstream_model_draft="definitely yes",
        epistemic_normalisation_applied=True,
    )
    row = _build_result("s1", {"user": "hi"}, lr)
    assert row["epistemic_normalisation_applied"] is True
    assert row["upstream_model_draft"] == "definitely yes"

    lr2 = LensResult(
        response="same",
        flags=[],
        pef_snapshot="",
        turn=1,
        span=Span.PRESENT,
        upstream_model_draft=None,
        epistemic_normalisation_applied=False,
    )
    row2 = _build_result("s2", {"user": "hi"}, lr2)
    assert row2["epistemic_normalisation_applied"] is False
    assert "upstream_model_draft" not in row2

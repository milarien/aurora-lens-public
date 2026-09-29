"""Audit provenance for user clarification / disambiguation across all binding paths."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from aurora_lens.adapters.base import AdapterResponse, LLMAdapter
from aurora_lens.config import LensConfig
from aurora_lens.context import domain_var
from aurora_lens.govern.audit_io import CHAIN_GENESIS
from aurora_lens.govern.bridge import BuiltinBridge
from aurora_lens.govern.canonical_bridge import CanonicalScannerGateBridge
from aurora_lens.govern.clarification_audit import (
    CLARIFICATION_RESOLUTION_OUTCOME,
    REPLAY_LINE_HELD,
    assert_replay_subsequence,
    count_user_disambiguation_rows,
    format_clarification_replay_lines,
    read_jsonl_audit_entries,
)
from aurora_lens.pef.state import EPISTEMIC_MODE_AMBIGUITY
from aurora_lens.govern.decision import InterventionAction
from aurora_lens.interpret.schema import ExtractedClaim, ExtractionResult
from aurora_lens.interpret.spacy_backend import SpacyBackend
from aurora_lens.lens import Lens
from aurora_lens.pef.span import Span
from aurora_lens.verify.flags import FlagType
from tests.test_clarification_commit_invariant import (
    _SETUP_INPUT,
    _CountingAdapter,
    _DisputeBackend,
    _inject_pending,
    _make_lens,
)
from tests.test_follow_up_attribution_binding_resume import MEDICAL_Q
from tests.test_lens import (
    _ComparativeAmbiguousBenchBackend,
    MockAdapter,
)


class _RecordingAdapter:
    def __init__(self) -> None:
        self.generate = AsyncMock(
            return_value=AdapterResponse(text="MODEL_SHOULD_NOT_RUN", model="mock"),
        )


def _disambiguation_rows(entries: list[dict]) -> list[dict]:
    return [e for e in entries if e.get("outcome") == CLARIFICATION_RESOLUTION_OUTCOME]


def _assert_disambiguation_chain(entries: list[dict], *, expected_count: int) -> list[dict]:
    rows = _disambiguation_rows(entries)
    assert len(rows) == expected_count, (
        f"expected {expected_count} USER_DISAMBIGUATION row(s), got {len(rows)}"
    )
    by_cid = {e.get("cid"): e for e in entries if e.get("cid")}
    for row in rows:
        cr = row["clarification_resolution"]
        assert cr.get("resolution_source") == "user_selection"
        assert cr.get("selected_option")
        prev_cid = row.get("prev_cid")
        assert prev_cid == CHAIN_GENESIS or prev_cid in by_cid, (
            f"prev_cid {prev_cid!r} not in audit file"
        )
    return rows


@pytest.mark.asyncio
async def test_user_disambiguation_audit_trail_emma_selection(tmp_path: Path):
    """Medical referent: ASK → OPTIONS → USER_DISAMBIGUATION → PASS."""
    audit = tmp_path / "medical.jsonl"
    lens = Lens(
        LensConfig(
            adapter=_RecordingAdapter(),
            extraction_backend=SpacyBackend(),
            enable_state_native_delegation=True,
            auto_verify=True,
            auto_interpret=True,
            governance_bridge=BuiltinBridge(audit_path=str(audit)),
        ),
    )

    r1 = await lens.process(MEDICAL_Q)
    assert r1.action in (InterventionAction.CONTAIN, InterventionAction.FORCE_REVISE)
    assert any(f.flag_type == FlagType.UNRESOLVED_REFERENT for f in (r1.flags or []))

    r2 = await lens.process("Emma")
    assert r2.action == InterventionAction.PASS
    assert "Emma" in r2.response

    entries = read_jsonl_audit_entries(str(audit))
    replay = format_clarification_replay_lines(entries)
    assert "ASK: UNRESOLVED_REFERENT" in replay
    assert "OPTIONS: Emma, Dr. Patel" in replay
    assert "USER_DISAMBIGUATION: selected Emma" in replay
    assert any(line.startswith("PASS: ") and "Emma" in line for line in replay)
    _assert_disambiguation_chain(entries, expected_count=1)


@pytest.mark.asyncio
async def test_dispute_alice_binding_audit_row(tmp_path: Path):
    """Injected UNRESOLVED_REFERENT + blocked_claims: Alice selection is audited."""
    audit = tmp_path / "dispute.jsonl"
    lens, adapter = _make_lens(
        _CountingAdapter(response="Alice agreed to mediation."),
        auto_verify=True,
    )
    lens._config = LensConfig(
        adapter=adapter,
        extraction_backend=_DisputeBackend(),
        auto_interpret=True,
        auto_verify=True,
        governance_bridge=BuiltinBridge(audit_path=str(audit)),
    )
    lens._bridge = lens._config.governance_bridge  # type: ignore[assignment]

    _inject_pending(lens)
    await lens.process("Alice.")

    entries = read_jsonl_audit_entries(str(audit))
    rows = _assert_disambiguation_chain(entries, expected_count=1)
    assert rows[0]["clarification_resolution"]["selected_option"] == "Alice"
    assert rows[0]["clarification_resolution"]["failed_constraint"] == "UNRESOLVED_REFERENT"


@pytest.mark.asyncio
async def test_entity_placeholder_binding_audit_row(tmp_path: Path):
    """unresolved_entity_ids path (no candidate list) still logs USER_DISAMBIGUATION."""
    audit = tmp_path / "placeholder.jsonl"
    clarification_ext = ExtractionResult(
        claims=[
            ExtractedClaim(
                subject="Richard",
                relation="IS",
                obj="subject",
                span=Span.PAST,
                negated=False,
                evidence="Richard",
            )
        ],
        entity_mentions=["Richard"],
        span=Span.PAST,
    )

    class _Seq:
        async def extract(self, text: str, pef):  # type: ignore[no-untyped-def]
            return clarification_ext

    adapter = _RecordingAdapter()
    adapter.generate = AsyncMock(
        return_value=AdapterResponse(text="Richard's stick was bigger.", model="mock"),
    )
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_Seq(),
            auto_verify=False,
            governance_bridge=BuiltinBridge(audit_path=str(audit)),
        ),
    )
    richard, _ = lens.pef.get_or_create_entity("Richard", resolved=False)
    lens.pef.pending_clarification = {
        "original_question": "His stick was bigger.",
        "unresolved_entity_ids": [richard.id],
        "original_span": "past",
        "failed_constraint": "UNRESOLVED_REFERENT",
    }

    await lens.process("Richard.")

    entries = read_jsonl_audit_entries(str(audit))
    rows = _assert_disambiguation_chain(entries, expected_count=1)
    assert rows[0]["clarification_resolution"]["selected_option"] == "Richard"
    assert richard.id in rows[0]["clarification_resolution"]["unresolved_entity_ids"]


@pytest.mark.asyncio
async def test_comparand_binding_audit_row(tmp_path: Path):
    """UNRESOLVED_COMPARAND candidate selection is audited before PASS."""
    audit = tmp_path / "comparand.jsonl"
    adapter = MockAdapter(["Richard's stick is bigger."])
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_ComparativeAmbiguousBenchBackend(),
            governance_bridge=BuiltinBridge(audit_path=str(audit)),
            auto_verify=False,
            auto_interpret=True,
            inject_pef_context=False,
        ),
    )
    r1 = await lens.process("Which stick is bigger?")
    assert r1.action != InterventionAction.PASS
    assert lens.pef.pending_clarification.get("failed_constraint") == "UNRESOLVED_COMPARAND"

    await lens.process("Richard")

    entries = read_jsonl_audit_entries(str(audit))
    rows = _assert_disambiguation_chain(entries, expected_count=1)
    cr = rows[0]["clarification_resolution"]
    assert cr["selected_option"] == "Richard"
    assert cr["failed_constraint"] == "UNRESOLVED_COMPARAND"
    replay = format_clarification_replay_lines(entries)
    assert "ASK: UNRESOLVED_COMPARAND" in replay
    assert "USER_DISAMBIGUATION: selected Richard" in replay


@pytest.mark.asyncio
async def test_finance_possessive_binding_audit_row(tmp_path: Path):
    """Finance possessive referent (spaCy) logs disambiguation on long clarify reply."""
    pytest.importorskip("spacy")
    audit = tmp_path / "finance.jsonl"

    class _Stub(LLMAdapter):
        def __init__(self) -> None:
            self.calls = 0

        async def generate(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            self.calls += 1
            return AdapterResponse(text="APAC Sub Q4 gross margin was lower than Q3.", model="stub")

    stub = _Stub()
    lens = Lens(
        LensConfig(
            adapter=stub,
            extraction_backend=SpacyBackend(model="en_core_web_sm"),
            governance_bridge=BuiltinBridge(audit_path=str(audit)),
            auto_interpret=True,
            auto_verify=False,
        ),
    )
    tok = domain_var.set("finance")
    try:
        await lens.process("APAC Sub reported a gross margin of 34% in Q4.")
        await lens.process("EMEA Sub reported a gross margin of 31% in Q4.")
        r3 = await lens.process("Was its margin worse than the prior quarter?")
        assert r3.action in (InterventionAction.CONTAIN, InterventionAction.FORCE_REVISE)

        clarify = (
            "I mean APAC Sub. APAC Sub gross margin in Q4 was 34%. "
            "APAC Sub gross margin in Q3 was 32%."
        )
        r4 = await lens.process(clarify)
        assert r4.action != InterventionAction.CONTAIN
    finally:
        domain_var.reset(tok)

    entries = read_jsonl_audit_entries(str(audit))
    rows = _assert_disambiguation_chain(entries, expected_count=1)
    assert "APAC Sub" in rows[0]["clarification_resolution"]["selected_option"]


@pytest.mark.asyncio
async def test_multi_turn_clarification_logs_each_resolution(tmp_path: Path):
    """Non-selection meta-turn does not log; each binding turn logs one row (Carol then Alice)."""
    audit = tmp_path / "multi.jsonl"
    lens, adapter = _make_lens(
        _CountingAdapter(response="Alice agreed to mediation."),
        auto_verify=True,
    )
    lens._config = LensConfig(
        adapter=adapter,
        extraction_backend=_DisputeBackend(),
        auto_interpret=True,
        auto_verify=True,
        governance_bridge=BuiltinBridge(audit_path=str(audit)),
    )
    lens._bridge = lens._config.governance_bridge  # type: ignore[assignment]

    await lens.process(_SETUP_INPUT)
    assert lens.pef.pending_clarification is not None

    r_meta = await lens.process("What information do you still need?")
    assert r_meta.action == InterventionAction.CONTAIN
    assert count_user_disambiguation_rows(read_jsonl_audit_entries(str(audit))) == 0

    _inject_pending(lens)
    await lens.process("Carol.")
    entries = read_jsonl_audit_entries(str(audit))
    assert count_user_disambiguation_rows(entries) == 1
    assert _disambiguation_rows(entries)[0]["clarification_resolution"]["selected_option"] == "Carol"

    _inject_pending(lens)
    await lens.process("Alice.")
    entries = read_jsonl_audit_entries(str(audit))
    rows = _assert_disambiguation_chain(entries, expected_count=2)
    selected = [r["clarification_resolution"]["selected_option"] for r in rows]
    assert selected == ["Carol", "Alice"]


def _assert_clarification_still_held(lens: Lens, *, candidates: list[str]) -> None:
    """Pending clarification and ambiguity hold remain live (no binding yet)."""
    pend = lens.pef.pending_clarification
    assert pend is not None, "pending_clarification must remain until valid binding"
    hold = lens.pef.epistemic_hold
    assert hold is not None, "epistemic_hold must remain until valid binding"
    assert hold.get("mode") == EPISTEMIC_MODE_AMBIGUITY
    assert hold.get("pathway_id") == "P_ASK_DISAMBIGUATE"
    assert sorted(pend.get("candidate_entities") or []) == sorted(candidates)


@pytest.mark.asyncio
async def test_clarification_held_across_irrelevant_turns_until_binding(tmp_path: Path):
    """ASK → irrelevant → held → irrelevant → held → selection → USER_DISAMBIGUATION → PASS."""
    audit = tmp_path / "held_across_turns.jsonl"
    adapter = _CountingAdapter(response="Alice agreed to mediation.")
    # Default Lens uses CanonicalScannerGateBridge; BuiltinBridge maps the same
    # flags to FORCE_REVISE and breaks the P_ASK_DISAMBIGUATE non-selection gate.
    bridge = CanonicalScannerGateBridge(audit_path=str(audit), backend="jsonl")
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_DisputeBackend(),
            auto_interpret=True,
            auto_verify=True,
            governance_bridge=bridge,
        ),
    )

    r_ask = await lens.process(_SETUP_INPUT)
    assert r_ask.action in (InterventionAction.CONTAIN, InterventionAction.FORCE_REVISE)
    assert any(f.flag_type == FlagType.UNRESOLVED_REFERENT for f in (r_ask.flags or []))
    candidates = sorted(lens.pef.pending_clarification.get("candidate_entities") or [])
    assert candidates == ["Alice", "Carol"]

    irrelevant = (
        "The weather is pleasant today.",
        "I don't know.",
    )
    for text in irrelevant:
        r_irr = await lens.process(text)
        assert r_irr.action != InterventionAction.PASS, (
            f"irrelevant turn must not PASS while hold active: {text!r}"
        )
        assert adapter.call_count == 0, "upstream must not run while clarification is held"
        _assert_clarification_still_held(lens, candidates=candidates)
        entries = read_jsonl_audit_entries(str(audit))
        assert count_user_disambiguation_rows(entries) == 0

    r_bind = await lens.process("Alice.")
    assert r_bind.action == InterventionAction.PASS
    assert adapter.call_count == 1
    assert lens.pef.pending_clarification is None
    assert lens.pef.epistemic_hold is None

    entries = read_jsonl_audit_entries(str(audit))
    replay = format_clarification_replay_lines(entries)
    assert_replay_subsequence(
        [
            "ASK: UNRESOLVED_REFERENT",
            REPLAY_LINE_HELD,
            REPLAY_LINE_HELD,
            "USER_DISAMBIGUATION: selected Alice",
            "PASS:",
        ],
        replay,
    )
    rows = _assert_disambiguation_chain(entries, expected_count=1)
    assert rows[0]["clarification_resolution"]["selected_option"] == "Alice"
    assert rows[0]["clarification_resolution"]["candidate_options"] == ["Alice", "Carol"]


@pytest.mark.asyncio
async def test_process_stream_binding_audit_parity(tmp_path: Path):
    """Stream path logs USER_DISAMBIGUATION on comparand binding like sync."""
    from tests.test_streaming_governance import (
        FakeStreamingAdapter,
        _collect_stream,
    )

    audit = tmp_path / "stream.jsonl"
    adapter = FakeStreamingAdapter(["Richard's stick is bigger."])
    cfg = LensConfig(
        adapter=adapter,
        extraction_backend=_ComparativeAmbiguousBenchBackend(),
        governance_bridge=BuiltinBridge(audit_path=str(audit)),
        auto_verify=False,
        auto_interpret=True,
        inject_pef_context=False,
        stream_emit_progress=False,
    )
    lens = Lens(cfg)
    await _collect_stream(lens, "Which stick is bigger?")
    await _collect_stream(lens, "Richard")

    entries = read_jsonl_audit_entries(str(audit))
    _assert_disambiguation_chain(entries, expected_count=1)

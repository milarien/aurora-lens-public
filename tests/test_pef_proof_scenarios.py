"""Cross-turn PEF proof scenarios (roadmap todo `pef-proof-scenarios`).

Deterministic demonstrators with explicit before/after `PEFState` slices and
captured `log_decision(pef_snapshot=...)` rows from a scripted governance bridge.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

# Sibling test module (tests/ is not always a package on sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_lens import MockAdapter, RecordingMockAdapter  # noqa: E402

from aurora_lens.config import LensConfig
from aurora_lens.govern.bridge import GovernanceBridge
from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
from aurora_lens.interpret.base import ExtractionBackend
from aurora_lens.interpret.schema import ExtractedClaim, ExtractionResult
from aurora_lens.lens import Lens
from aurora_lens.pef.entity import Entity
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import (
    EPISTEMIC_HOLD_SCHEMA_VERSION,
    EPISTEMIC_MODE_AMBIGUITY,
    EPISTEMIC_MODE_REFUSAL,
    EPISTEMIC_MODE_STOP,
    PEFState,
    Relationship,
)
from aurora_lens.verify.flags import Flag, FlagType


def _proof_pef_slice(pef: PEFState) -> dict:
    """Minimal projection for transition proofs (audit-friendly JSON shape)."""
    d = pef.to_dict()
    return {
        "current_turn": d.get("current_turn"),
        "pending_clarification": d.get("pending_clarification"),
        "epistemic_hold": d.get("epistemic_hold"),
    }


class ProofScenarioBridge(GovernanceBridge):
    """Scripted `decide()` sequence; merges incoming `flags` from the lens."""

    def __init__(self, decisions: list[GovernanceDecision]):
        self._decisions = list(decisions)
        self._idx = 0
        self.audit_log: list[dict] = []

    async def decide(
        self,
        flags: list[Flag],
        response_text: str,
        pef: PEFState,
    ) -> GovernanceDecision:
        if self._idx >= len(self._decisions):
            template = self._decisions[-1]
        else:
            template = self._decisions[self._idx]
            self._idx += 1
        return dataclasses.replace(template, flags=flags)

    async def intervene(
        self,
        decision: GovernanceDecision,
        adapter,
        user_input: str,
        pef_context: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        return (
            decision.governed_response
            or decision.corrected_response
            or "[governed]"
        )

    def log_decision(
        self,
        decision: GovernanceDecision,
        turn: int = 0,
        *,
        stream: bool = False,
        stream_completed: bool = True,
        stream_abort_reason: str | None = None,
        stream_truncated: bool = False,
        stream_dropped_chars: int = 0,
        pef_context: str | None = None,
        pre_llm: bool = False,
        pef_snapshot: dict | None = None,
        pef_turn_classification: str | None = None,
        pef_hold_transition: str | None = None,
        at_verification_basis: dict | None = None,
        epistemic_normalisation_applied: bool | None = None,
        state_native_handled: bool | None = None,
    ) -> None:
        row: dict = {
            "turn": turn,
            "pre_llm": pre_llm,
            "action": decision.action,
            "pathway_id": decision.pathway_id,
            "pef_snapshot": pef_snapshot,
        }
        if epistemic_normalisation_applied is not None:
            row["epistemic_normalisation_applied"] = epistemic_normalisation_applied
        if state_native_handled is not None:
            row["state_native_handled"] = state_native_handled
        if pef_turn_classification is not None:
            row["pef_turn_classification"] = pef_turn_classification
        if pef_hold_transition is not None:
            row["pef_hold_transition"] = pef_hold_transition
        if at_verification_basis is not None:
            row["at_verification_basis"] = at_verification_basis
        self.audit_log.append(row)


def _unverified_flag() -> Flag:
    return Flag(
        flag_type=FlagType.UNVERIFIED_FACT_ASSERTION,
        entity_name="proof",
        claim="numeric fact without PEF support",
        evidence="proof-scenario",
        severity="warning",
    )


class TestPefProofUnresolvedReferentAcrossTurns:
    """Unresolved referent: ASK persists across ALL turns; cleared only on binding resolution."""

    @pytest.mark.asyncio
    async def test_transitions_and_snapshots(self):
        pef = PEFState()
        for name in ("James", "Richard"):
            e = Entity.create(name, turn=0)
            pef.add_entity(e)
            pef.add_relationship(
                Relationship(
                    subject_id=e.id,
                    relation="HAS",
                    object_entity_id=None,
                    object_literal="dog",
                    span=Span.PRESENT,
                    source_turn=0,
                    evidence=f"{name} has a dog",
                )
            )

        class _TieredBackend(ExtractionBackend):
            async def extract(self, text: str, pef: PEFState):
                tl = text.strip().lower()
                if "his dog" in tl and "bigger" in tl:
                    return ExtractionResult(
                        claims=[
                            ExtractedClaim(
                                subject="his dog",
                                relation="IS",
                                obj="bigger",
                                span=Span.PRESENT,
                                negated=False,
                                evidence=text,
                            )
                        ],
                        entity_mentions=["James", "Richard"],
                        span=Span.PRESENT,
                        ambiguous_referents=["his"],
                    )
                if "sky" in tl and "blue" in tl:
                    return ExtractionResult(
                        claims=[
                            ExtractedClaim(
                                subject="the sky",
                                relation="IS",
                                obj="blue",
                                span=Span.PRESENT,
                                negated=False,
                                evidence=text,
                            )
                        ],
                        entity_mentions=[],
                        span=Span.PRESENT,
                    )
                if tl.rstrip(".!?") == "james":
                    return ExtractionResult(
                        claims=[],
                        entity_mentions=["James"],
                        span=Span.PRESENT,
                    )
                return ExtractionResult(claims=[], entity_mentions=[], span=Span.PRESENT)

        before = _proof_pef_slice(pef)
        assert before["pending_clarification"] is None
        assert before["epistemic_hold"] is None

        adapter = RecordingMockAdapter(
            responses=[
                "It is sunny.",
                "James's dog was bigger.",
            ]
        )
        config = LensConfig(adapter=adapter, extraction_backend=_TieredBackend())
        lens = Lens(config, initial_pef=pef)

        r1 = await lens.process("His dog was bigger.")
        assert r1.action != InterventionAction.PASS
        s1 = _proof_pef_slice(lens.pef)
        assert s1["pending_clarification"] is not None
        assert s1["epistemic_hold"] is not None
        assert s1["epistemic_hold"]["mode"] == EPISTEMIC_MODE_AMBIGUITY

        # Guard intercepts TELL-act turn while ambiguity hold is active — no adapter call.
        r2 = await lens.process("The sky is blue.")
        assert r2.action == InterventionAction.CONTAIN, (
            "TELL-act turn while ambiguity hold active must return CONTAIN"
        )
        s2 = _proof_pef_slice(lens.pef)
        assert s2["pending_clarification"] is not None
        assert s2["epistemic_hold"] is not None
        assert s2["epistemic_hold"]["mode"] == EPISTEMIC_MODE_AMBIGUITY

        r3 = await lens.process("James")
        s3 = _proof_pef_slice(lens.pef)
        assert s3["pending_clarification"] is None
        assert s3["epistemic_hold"] is None


class TestPefProofMissingFactAndRefuseThenSoftAdmit:
    """Injected UNVERIFIED flags + scripted REFUSE then SOFT_CORRECT (admissible clarification)."""

    @pytest.mark.asyncio
    async def test_hold_refusal_then_cleared_by_soft_correct(self):
        class _IndexedBackend(ExtractionBackend):
            def __init__(self) -> None:
                self._n = 0

            async def extract(self, text: str, pef: PEFState) -> ExtractionResult:
                self._n += 1
                if self._n == 1:
                    return ExtractionResult(
                        claims=[
                            ExtractedClaim(
                                subject="Emma",
                                relation="HAS",
                                obj="red apple",
                                span=Span.PRESENT,
                                negated=False,
                                evidence=text,
                            )
                        ],
                        entity_mentions=["Emma"],
                        span=Span.PRESENT,
                    )
                return ExtractionResult(claims=[], entity_mentions=[], span=Span.PRESENT)

        bridge = ProofScenarioBridge(
            [
                GovernanceDecision(
                    action=InterventionAction.FORCE_REVISE,
                    flags=[],
                    rationale="proof-refuse",
                    policy="strict",
                    pathway_id="P_REFUSE",
                    interaction_open=True,
                    commitment_closed=True,
                    cid="cid-proof-refuse",
                    governed_response="Cannot assert that without support.",
                ),
                GovernanceDecision(
                    action=InterventionAction.SOFT_CORRECT,
                    flags=[],
                    rationale="proof-soft-admit",
                    policy="strict",
                    pathway_id=None,
                    cid="cid-proof-soft",
                    governed_response="Understood; continuing with grounded content only.",
                ),
            ]
        )
        adapter = MockAdapter(
            responses=[
                "Emma has a red apple.",
                "Revenue is $1M.",
                "Okay.",
            ]
        )
        config = LensConfig(
            adapter=adapter,
            extraction_backend=_IndexedBackend(),
            governance_bridge=bridge,
            auto_verify=False,
        )
        lens = Lens(config)

        await lens.process("Emma has a red apple.")
        assert lens.pef.epistemic_hold is None

        await lens.process(
            "Revenue is $1M.",
            external_flags=[_unverified_flag()],
        )
        assert lens.pef.epistemic_hold is not None
        assert lens.pef.epistemic_hold["mode"] == EPISTEMIC_MODE_REFUSAL
        assert lens.pef.epistemic_hold["last_audit_id"] == "cid-proof-refuse"
        # Audit row uses PEF after hold apply (posture linkage + snapshot match durable state).
        snap_non_admit = bridge.audit_log[-1].get("pef_snapshot") or {}
        assert snap_non_admit.get("epistemic_hold", {}).get("mode") == EPISTEMIC_MODE_REFUSAL

        await lens.process(
            "Thanks.",
            external_flags=[_unverified_flag()],
        )
        assert lens.pef.epistemic_hold is None
        # SOFT_CORRECT: snapshot is taken after _maybe_clear_epistemic_hold_on_admit — hold cleared.
        snap_soft = bridge.audit_log[-1].get("pef_snapshot") or {}
        assert snap_soft.get("epistemic_hold") is None


class TestPefProofStopOpenThenLawfulContinuation:
    """Reopenable STOP persists across PASS; SOFT_CORRECT clears (lawful continuation)."""

    @pytest.mark.asyncio
    async def test_stop_open_then_soft_correct_clears(self):
        class _EmptyBackend(ExtractionBackend):
            async def extract(self, text: str, pef: PEFState) -> ExtractionResult:
                return ExtractionResult(claims=[], entity_mentions=[], span=Span.PRESENT)

        bridge = ProofScenarioBridge(
            [
                GovernanceDecision(
                    action=InterventionAction.HARD_STOP,
                    flags=[],
                    rationale="proof-stop-open",
                    policy="strict",
                    pathway_id="P_STOP",
                    interaction_open=True,
                    commitment_closed=False,
                    cid="cid-stop-open",
                    governed_response="Stop (reopenable).",
                ),
                GovernanceDecision(
                    action=InterventionAction.SOFT_CORRECT,
                    flags=[],
                    rationale="proof-soft-after-stop",
                    policy="strict",
                    pathway_id=None,
                    cid="cid-soft-after-stop",
                    governed_response="Continuing.",
                ),
            ]
        )
        adapter = MockAdapter(responses=["Hello.", "Okay."])
        config = LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyBackend(),
            governance_bridge=bridge,
            auto_verify=False,
        )
        lens = Lens(config)

        await lens.process("Hello.", external_flags=[_unverified_flag()])
        assert lens.pef.epistemic_hold is not None
        assert lens.pef.epistemic_hold["mode"] == EPISTEMIC_MODE_STOP
        assert lens.pef.epistemic_hold["interaction_open"] is True

        await lens.process("Okay.", external_flags=[_unverified_flag()])
        assert lens.pef.epistemic_hold is None


class TestPefProofStopTerminalPersists:
    """Terminal STOP is not cleared by SOFT_CORRECT."""

    @pytest.mark.asyncio
    async def test_terminal_stop_survives_soft_correct(self):
        class _EmptyBackend(ExtractionBackend):
            async def extract(self, text: str, pef: PEFState) -> ExtractionResult:
                return ExtractionResult(claims=[], entity_mentions=[], span=Span.PRESENT)

        bridge = ProofScenarioBridge(
            [
                GovernanceDecision(
                    action=InterventionAction.HARD_STOP,
                    flags=[],
                    rationale="proof-stop-terminal",
                    policy="strict",
                    pathway_id="P_STOP_TERMINAL",
                    interaction_open=False,
                    commitment_closed=True,
                    cid="cid-stop-term",
                    governed_response="Stop (terminal).",
                ),
                GovernanceDecision(
                    action=InterventionAction.SOFT_CORRECT,
                    flags=[],
                    rationale="proof-soft-noop",
                    policy="strict",
                    pathway_id=None,
                    cid="cid-soft-term",
                    governed_response="Acknowledged.",
                ),
            ]
        )
        adapter = MockAdapter(responses=["Hello.", "Okay."])
        config = LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyBackend(),
            governance_bridge=bridge,
            auto_verify=False,
        )
        lens = Lens(config)

        await lens.process("Hello.", external_flags=[_unverified_flag()])
        hold = lens.pef.epistemic_hold
        assert hold is not None
        assert hold["mode"] == EPISTEMIC_MODE_STOP
        assert hold["interaction_open"] is False

        await lens.process("Okay.", external_flags=[_unverified_flag()])
        assert lens.pef.epistemic_hold is not None
        assert lens.pef.epistemic_hold["mode"] == EPISTEMIC_MODE_STOP
        assert lens.pef.epistemic_hold["last_audit_id"] == "cid-stop-term"


class TestPefProofAuditSnapshots:
    """Audit log rows carry PEF snapshots usable for before/after comparison."""

    @pytest.mark.asyncio
    async def test_log_decision_pef_snapshot_tracks_hold_lifecycle(self):
        class _EmptyBackend(ExtractionBackend):
            async def extract(self, text: str, pef: PEFState) -> ExtractionResult:
                return ExtractionResult(claims=[], entity_mentions=[], span=Span.PRESENT)

        bridge = ProofScenarioBridge(
            [
                GovernanceDecision(
                    action=InterventionAction.FORCE_REVISE,
                    flags=[],
                    rationale="audit-proof",
                    policy="strict",
                    pathway_id="P_REFUSE",
                    interaction_open=True,
                    commitment_closed=True,
                    cid="cid-audit-1",
                    governed_response="Revise.",
                ),
                GovernanceDecision(
                    action=InterventionAction.SOFT_CORRECT,
                    flags=[],
                    rationale="audit-soft",
                    policy="strict",
                    pathway_id=None,
                    cid="cid-audit-2",
                    governed_response="Soft.",
                ),
            ]
        )
        adapter = MockAdapter(responses=["A", "B"])
        lens = Lens(
            LensConfig(
                adapter=adapter,
                extraction_backend=_EmptyBackend(),
                governance_bridge=bridge,
                auto_verify=False,
            )
        )
        await lens.process("A", external_flags=[_unverified_flag()])
        await lens.process("B", external_flags=[_unverified_flag()])
        snaps = [e.get("pef_snapshot") for e in bridge.audit_log if not e.get("pre_llm")]
        assert len(snaps) >= 2
        assert snaps[-2] is not None
        # FORCE_REVISE row: snapshot after refusal hold is applied (durable posture in force).
        assert snaps[-2].get("epistemic_hold", {}).get("mode") == EPISTEMIC_MODE_REFUSAL
        assert snaps[-1] is not None
        # SOFT_CORRECT row: snapshot after refusal hold is cleared on admit.
        assert snaps[-1].get("epistemic_hold") is None

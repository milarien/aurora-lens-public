"""Artifact/external layer MVP — Phases 1–5 invariant tests.

Invariants:
- Frame detector classifies ARTIFACT openers and EXTERNAL closers correctly.
- EXTERNAL is the default frame (no opener matched).
- Closers override openers when both are present.
- ArtifactFrame tracks kind and opened_at_turn.
- PEFState.active_frame is None by default (EXTERNAL).
- State-native delegation returns handled=False when active_frame is ARTIFACT.
- Lens.process() sets active_frame on ARTIFACT opener (production-path wiring).
- Lens.process() leaves active_frame None for non-artifact input.
- Artifact guard is reachable through the real lens.py path.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from aurora_lens.interpret.turn_act import TurnAct
from aurora_lens.pef.artifact_layer import ArtifactFrame, FrameKind, UnresolvedKind
from aurora_lens.pef.entity import Entity
from aurora_lens.pef.frame_detector import classify_frame, detect_frame_transition
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState, Relationship
from aurora_lens.state_native_engine.contracts import StateNativeRequest
from aurora_lens.state_native_engine.default_engine import DefaultStateNativeEngine


# ── Frame detector ────────────────────────────────────────────────────────────

class TestFrameDetector:
    def test_default_is_external(self) -> None:
        assert classify_frame("Where is Alice?") == FrameKind.EXTERNAL

    def test_empty_string_is_external(self) -> None:
        assert classify_frame("") == FrameKind.EXTERNAL

    def test_lets_pretend_opener(self) -> None:
        assert classify_frame("Let's pretend you are a doctor.") == FrameKind.ARTIFACT

    def test_imagine_you_are_opener(self) -> None:
        assert classify_frame("Imagine you are a pilot.") == FrameKind.ARTIFACT

    def test_roleplay_as_opener(self) -> None:
        assert classify_frame("Roleplay as a detective.") == FrameKind.ARTIFACT

    def test_in_this_story_opener(self) -> None:
        assert classify_frame("In this story, Alice is the queen.") == FrameKind.ARTIFACT

    def test_hypothetically_speaking_opener(self) -> None:
        assert classify_frame("Hypothetically speaking, what would happen?") == FrameKind.ARTIFACT

    def test_closer_back_to_reality_returns_external(self) -> None:
        assert classify_frame("Back to reality — where is Alice?") == FrameKind.EXTERNAL

    def test_closer_end_roleplay_returns_external(self) -> None:
        assert classify_frame("End roleplay.") == FrameKind.EXTERNAL

    def test_closer_overrides_opener_in_same_message(self) -> None:
        text = "Let's pretend you are a doctor. End roleplay."
        assert classify_frame(text) == FrameKind.EXTERNAL

    def test_opener_must_be_in_prefix(self) -> None:
        long_prefix = "a" * 200
        text = long_prefix + " let's pretend you are a doctor."
        assert classify_frame(text) == FrameKind.EXTERNAL


class TestDetectFrameTransition:
    def test_external_to_artifact_transition(self) -> None:
        result = detect_frame_transition("let's pretend you are an astronaut", FrameKind.EXTERNAL)
        assert result == FrameKind.ARTIFACT

    def test_artifact_to_external_transition(self) -> None:
        result = detect_frame_transition("end roleplay", FrameKind.ARTIFACT)
        assert result == FrameKind.EXTERNAL

    def test_no_transition_when_same_frame(self) -> None:
        assert detect_frame_transition("Where is Alice?", FrameKind.EXTERNAL) is None
        assert detect_frame_transition("As a character, tell me...", FrameKind.ARTIFACT) is None


# ── ArtifactFrame and UnresolvedKind ─────────────────────────────────────────

class TestArtifactFrame:
    def test_construction(self) -> None:
        frame = ArtifactFrame(kind=FrameKind.ARTIFACT, opened_at_turn=3)
        assert frame.kind == FrameKind.ARTIFACT
        assert frame.opened_at_turn == 3
        assert frame.description is None

    def test_with_description(self) -> None:
        frame = ArtifactFrame(
            kind=FrameKind.ARTIFACT,
            opened_at_turn=1,
            description="medical roleplay",
        )
        assert frame.description == "medical roleplay"


class TestUnresolvedKind:
    def test_all_members_present(self) -> None:
        assert UnresolvedKind.EXTERNAL_UNRESOLVED_REFERENT
        assert UnresolvedKind.ARTIFACT_UNINSTANTIATED_ROLE
        assert UnresolvedKind.OPEN_AMBIGUITY_BRANCH
        assert UnresolvedKind.CONSEQUENCE_UNRESOLVED


# ── PEFState.active_frame ─────────────────────────────────────────────────────

class TestPEFStateActiveFrame:
    def test_default_is_none(self) -> None:
        pef = PEFState()
        assert pef.active_frame is None

    def test_can_set_artifact_frame(self) -> None:
        pef = PEFState()
        pef.active_frame = ArtifactFrame(kind=FrameKind.ARTIFACT, opened_at_turn=2)
        assert pef.active_frame is not None
        assert pef.active_frame.kind == FrameKind.ARTIFACT

    def test_can_clear_frame(self) -> None:
        pef = PEFState()
        pef.active_frame = ArtifactFrame(kind=FrameKind.ARTIFACT, opened_at_turn=1)
        pef.active_frame = None
        assert pef.active_frame is None


# ── State-native artifact frame guard ────────────────────────────────────────

class TestStateNativeArtifactFrameGuard:
    """Phase 5: state-native must not answer from external PEF when ARTIFACT frame is active."""

    def _pef_with_alice_at_store(self) -> PEFState:
        pef = PEFState(session_id="af_test")
        alice = Entity.create("Alice", 0, session_id="af_test")
        pef.add_entity(alice)
        pef.add_relationship(
            Relationship(
                subject_id=alice.id,
                relation="AT",
                object_literal="the store",
                object_entity_id=None,
                span=Span.PRESENT,
                source_turn=1,
                evidence="Alice is at the store.",
            )
        )
        return pef

    def test_location_query_answered_when_no_frame(self) -> None:
        pef = self._pef_with_alice_at_store()
        req = StateNativeRequest(
            user_text="Where is Alice?",
            pef=pef,
            turn_act=TurnAct.QUERY,
            detected_span=Span.PRESENT,
        )
        result = DefaultStateNativeEngine().evaluate(req)
        assert result.handled is True

    def test_location_query_blocked_when_artifact_frame_active(self) -> None:
        pef = self._pef_with_alice_at_store()
        pef.active_frame = ArtifactFrame(kind=FrameKind.ARTIFACT, opened_at_turn=2)
        req = StateNativeRequest(
            user_text="Where is Alice?",
            pef=pef,
            turn_act=TurnAct.QUERY,
            detected_span=Span.PRESENT,
        )
        result = DefaultStateNativeEngine().evaluate(req)
        assert result.handled is False

    def test_external_frame_does_not_block(self) -> None:
        pef = self._pef_with_alice_at_store()
        pef.active_frame = ArtifactFrame(kind=FrameKind.EXTERNAL, opened_at_turn=1)
        req = StateNativeRequest(
            user_text="Where is Alice?",
            pef=pef,
            turn_act=TurnAct.QUERY,
            detected_span=Span.PRESENT,
        )
        result = DefaultStateNativeEngine().evaluate(req)
        assert result.handled is True


# ── Production-path: lens.py turn-start frame classification ─────────────────

class TestLensFrameClassificationWiring:
    """Verify that lens.py writes active_frame before extraction/LLM on each turn."""

    def _make_lens(self, *, initial_pef: PEFState | None = None, enable_sn: bool = False):
        from aurora_lens.lens import Lens
        from aurora_lens.config import LensConfig
        from aurora_lens.adapters.base import LLMAdapter, AdapterResponse

        class _MockAdapter(LLMAdapter):
            def __init__(self):
                self.call_count = 0

            async def generate(self, messages, **kwargs):
                self.call_count += 1
                return AdapterResponse(text="(mock response)", model="mock")

        adapter = _MockAdapter()
        cfg = LensConfig(
            adapter=adapter,
            auto_interpret=True,
            auto_verify=True,
            enable_state_native_delegation=enable_sn,
        )
        lens = Lens(cfg, initial_pef=initial_pef)
        return lens, adapter

    @pytest.mark.asyncio
    async def test_artifact_opener_sets_active_frame(self) -> None:
        """process() with an artifact opener writes FrameKind.ARTIFACT to self._pef.active_frame."""
        lens, _ = self._make_lens()
        assert lens._pef.active_frame is None
        await lens.process("Let's pretend you are a doctor.")
        assert lens._pef.active_frame is not None
        assert lens._pef.active_frame.kind == FrameKind.ARTIFACT

    @pytest.mark.asyncio
    async def test_non_artifact_input_leaves_frame_none(self) -> None:
        """process() with ordinary input leaves active_frame as None (EXTERNAL default)."""
        lens, _ = self._make_lens()
        await lens.process("Where is Alice?")
        assert lens._pef.active_frame is None

    @pytest.mark.asyncio
    async def test_artifact_guard_reachable_through_lens_path(self) -> None:
        """When active_frame is ARTIFACT, state-native guard fires and LLM is called instead.

        Setup: Alice entity + AT relation committed to PEF before the artifact-opener turn.
        After the opener, a location query must NOT be answered by state-native — the guard
        passes it to the LLM adapter.
        """
        pef = PEFState(session_id="frame_guard_test")
        alice = Entity.create("Alice", 0, session_id="frame_guard_test")
        pef.add_entity(alice)
        pef.add_relationship(
            Relationship(
                subject_id=alice.id,
                relation="AT",
                object_literal="the store",
                object_entity_id=None,
                span=Span.PRESENT,
                source_turn=0,
                evidence="Alice is at the store.",
            )
        )
        lens, adapter = self._make_lens(initial_pef=pef, enable_sn=True)

        # Turn 1: open artifact frame
        await lens.process("Let's pretend you are a doctor.")
        assert lens._pef.active_frame is not None
        assert lens._pef.active_frame.kind == FrameKind.ARTIFACT
        calls_after_opener = adapter.call_count

        # Turn 2: location query — guard must fire, LLM must be called
        await lens.process("Where is Alice?")
        assert adapter.call_count > calls_after_opener, (
            "Artifact frame guard must pass location query to LLM; "
            "adapter.call_count did not increase"
        )

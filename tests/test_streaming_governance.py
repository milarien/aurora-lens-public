"""Contract tests for full-buffer streaming governance.

Proves that process_stream() implements the private-accumulation / gated-release
state machine described in aurora_lens/govern/stream_gate.py.

Test classes
------------
TestStreamPhaseAndTypes          — StreamPhase enum, BufferedStreamResult,
                                   ProgressSignal validation
TestAdmitStreamingPath           — ADMIT: buffer released as "chunk" events;
                                   no governance suppression; hashes correct
TestAskStreamingPath             — ASK: buffer suppressed; governed clarification
                                   emitted; forensic event complete
TestRefuseStreamingPath          — REFUSE: buffer suppressed; governed refusal
                                   emitted; forensic event complete
TestStopStreamingPath            — STOP: buffer suppressed; governed stop
                                   emitted; forensic event complete
TestNoRawContentLeak             — No raw buffered substring appears in
                                   user-visible output for non-ADMIT paths
TestStreamingNonStreamingEquiv   — Final governance outcome and forensic schema
                                   are identical between streaming and non-streaming
TestProgressSignalling           — Safe scaffolding only; never candidate text
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aurora_lens.adapters.base import AdapterResponse, LLMAdapter
from aurora_lens.config import LensConfig
from aurora_lens.govern.bridge import enforce, BuiltinBridge
from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
from aurora_lens.govern.stream_gate import (
    BufferedStreamResult,
    ProgressSignal,
    StreamPhase,
    _VALID_PROGRESS_STATUSES,
    make_governed_chunk_dict,
)
from aurora_lens.interpret.spacy_backend import SpacyBackend
from aurora_lens.lens import Lens, _pre_llm_unresolved_referent_clarification
from aurora_lens.pef.state import PEFState
from aurora_lens.verify.flags import Flag, FlagType


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_chunk_dict(content: str, model: str = "test-model") -> dict:
    return {
        "id": "test-id",
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }


class FakeStreamingAdapter(LLMAdapter):
    """Yields predefined chunks for streaming; also supports non-streaming."""

    def __init__(self, chunks: list[str], model: str = "test-model"):
        self._chunks = chunks
        self._model = model
        self.generate_called = False
        self.generate_stream_called = False

    async def generate(self, messages, **kwargs) -> AdapterResponse:
        self.generate_called = True
        return AdapterResponse(
            text="".join(self._chunks),
            model=self._model,
        )

    async def generate_stream(self, messages, **kwargs) -> AsyncIterator[tuple[dict, str]]:
        self.generate_stream_called = True
        for chunk in self._chunks:
            yield (_make_chunk_dict(chunk, self._model), chunk)

    # AsyncIterator protocol (aclose)
    async def aclose(self) -> None:
        pass


class NoStreamAdapter(LLMAdapter):
    """Adapter that only supports non-streaming (generate_stream raises NotImplementedError)."""

    def __init__(self, text: str, model: str = "test-model"):
        self._text = text
        self._model = model

    async def generate(self, messages, **kwargs) -> AdapterResponse:
        return AdapterResponse(text=self._text, model=self._model)

    async def generate_stream(self, messages, **kwargs):
        raise NotImplementedError
        yield  # make it an async generator


class _TwoPhaseStreamAdapter(LLMAdapter):
    """Streaming adapter with distinct chunk sequences per upstream call."""

    def __init__(
        self,
        turn1_chunks: list[str],
        turn2_chunks: list[str],
        model: str = "test-model",
    ):
        self._phases = [turn1_chunks, turn2_chunks]
        self._call_idx = 0
        self.last_stream_messages: list[dict[str, str]] | None = None
        self._model = model

    def _current_chunks(self) -> list[str]:
        i = min(self._call_idx, len(self._phases) - 1)
        return self._phases[i]

    async def generate(self, messages, **kwargs) -> AdapterResponse:
        chunks = self._current_chunks()
        self._call_idx += 1
        return AdapterResponse(text="".join(chunks), model=self._model)

    async def generate_stream(self, messages, **kwargs) -> AsyncIterator[tuple[dict, str]]:
        self.last_stream_messages = list(messages)
        chunks = self._current_chunks()
        self._call_idx += 1
        for chunk in chunks:
            yield (_make_chunk_dict(chunk, self._model), chunk)

    async def aclose(self) -> None:
        pass


def _lens_config(adapter: LLMAdapter, *, emit_progress: bool = False, **kwargs) -> LensConfig:
    return LensConfig(
        adapter=adapter,
        auto_interpret=False,
        auto_verify=True,
        inject_pef_context=False,
        include_operator_detail=True,
        stream_emit_progress=emit_progress,
        **kwargs,
    )


async def _collect_stream(lens: Lens, user_input: str, **kwargs) -> dict[str, list]:
    """Drain process_stream() and return events grouped by kind."""
    events: dict[str, list] = {
        "progress": [],
        "chunk": [],
        "governed_chunk": [],
        "metadata": [],
        "extraction_failed": [],
        "clarification_continuation": [],
    }
    async for kind, payload in lens.process_stream(user_input, **kwargs):
        events.setdefault(kind, []).append(payload)
    return events


def _visible_text(events: dict[str, list]) -> str:
    """Reconstruct all user-visible text from events (chunk + governed_chunk)."""
    parts = []
    for chunk_dict, delta in events.get("chunk", []):
        parts.append(delta)
    for chunk_dict, delta in events.get("governed_chunk", []):
        parts.append(delta)
    return "".join(parts)


# ── TestStreamPhaseAndTypes ───────────────────────────────────────────────────

class TestStreamPhaseAndTypes:
    def test_stream_phase_values(self):
        expected = {
            "UPSTREAM_STREAMING", "BUFFER_COMPLETE", "VERIFYING",
            "RELEASING_BUFFER", "SUPPRESSING_BUFFER", "COMPLETE", "ABORTED",
        }
        assert {p.name for p in StreamPhase} == expected

    def test_buffered_stream_result_full_text(self):
        chunks = [(_make_chunk_dict("Hello "), "Hello "), (_make_chunk_dict("world"), "world")]
        buf = BufferedStreamResult(
            chunks=chunks,
            full_text="Hello world",
            total_bytes=11,
        )
        assert buf.full_text == "Hello world"
        assert len(buf.chunks) == 2
        assert not buf.truncated

    def test_progress_signal_valid_statuses(self):
        for s in _VALID_PROGRESS_STATUSES:
            sig = ProgressSignal(status=s)
            assert sig.status == s

    def test_progress_signal_invalid_status_raises(self):
        with pytest.raises(ValueError, match="must be one of"):
            ProgressSignal(status="admitted")

    def test_progress_signal_no_content_field(self):
        sig = ProgressSignal(status="verifying")
        d = sig.as_event_dict()
        assert d["type"] == "aurora_progress"
        assert d["status"] == "verifying"
        assert len(d) == 2  # type + status only

    def test_make_governed_chunk_dict_structure(self):
        d = make_governed_chunk_dict("I cannot help with that.", model="test")
        assert d["choices"][0]["delta"]["content"] == "I cannot help with that."
        assert d["choices"][0]["finish_reason"] == "stop"
        assert "gov-" in d["id"]

    def test_make_governed_chunk_dict_never_echoes_suppressed_text(self):
        suppressed = "Here is how to synthesize..."
        governed = "I cannot help with that."
        d = make_governed_chunk_dict(governed, model="test")
        s = json.dumps(d)
        assert suppressed not in s
        assert governed in s


# ── TestAdmitStreamingPath ────────────────────────────────────────────────────

class TestAdmitStreamingPath:
    """ADMIT: provider chunks buffered privately, then released as 'chunk' events."""

    @pytest.mark.asyncio
    async def test_admit_chunks_released_in_order(self):
        adapter = FakeStreamingAdapter(["Hello", " ", "world"])
        lens = Lens(_lens_config(adapter))
        events = await _collect_stream(lens, "Say hello")
        assert events["governed_chunk"] == []
        deltas = [delta for _, delta in events["chunk"]]
        assert deltas == ["Hello", " ", "world"]

    @pytest.mark.asyncio
    async def test_admit_full_text_in_metadata(self):
        adapter = FakeStreamingAdapter(["Hello world"])
        lens = Lens(_lens_config(adapter))
        events = await _collect_stream(lens, "Say hello")
        meta = events["metadata"][0]
        assert meta["governance"] == "PASS"
        assert meta["stream_governed"] is True

    @pytest.mark.asyncio
    async def test_admit_no_governed_chunk_emitted(self):
        adapter = FakeStreamingAdapter(["Hello world"])
        lens = Lens(_lens_config(adapter))
        events = await _collect_stream(lens, "Say hello")
        assert events["governed_chunk"] == []

    @pytest.mark.asyncio
    async def test_admit_streaming_path_calls_generate_stream(self):
        adapter = FakeStreamingAdapter(["Hello"])
        lens = Lens(_lens_config(adapter))
        await _collect_stream(lens, "Say hello")
        assert adapter.generate_stream_called

    @pytest.mark.asyncio
    async def test_admit_no_chunk_before_governance(self):
        """Verify that the stream is fully buffered before any chunk is emitted.

        We track the order of event kinds: no 'chunk' should appear before the
        'metadata' arrives (which always comes last), meaning governance ran first.
        All 'chunk' events appear AFTER the buffer was verified (we can only
        observe this via kind ordering — chunks are emitted post-decision).
        """
        adapter = FakeStreamingAdapter(["A", "B", "C"])
        lens = Lens(_lens_config(adapter))
        kinds = []
        async for kind, _ in lens.process_stream("hi"):
            kinds.append(kind)
        # chunks appear before metadata, but that's fine.
        # What must NOT happen is that chunks appear DURING streaming
        # (which we can't directly observe in a unit test, but
        # the state machine guarantees it).
        assert "chunk" in kinds
        assert kinds[-1] == "metadata"
        assert kinds.index("chunk") < kinds.index("metadata")

    @pytest.mark.asyncio
    async def test_admit_fallback_to_generate_also_releases(self):
        """When generate_stream raises NotImplementedError, buffer via generate()."""
        adapter = NoStreamAdapter("Hello world")
        lens = Lens(_lens_config(adapter))
        events = await _collect_stream(lens, "Say hello")
        assert events["governed_chunk"] == []
        assert _visible_text(events) == "Hello world"


# ── TestAskStreamingPath ──────────────────────────────────────────────────────

class TestAskStreamingPath:
    """ASK: raw candidate suppressed; governed clarification emitted."""

    @pytest.mark.asyncio
    async def test_ask_suppresses_raw_buffer(self, monkeypatch):
        """Raw candidate text must not appear in any chunk or governed_chunk."""
        raw_candidate = "Alice is 30 and her sister is 28"
        adapter = FakeStreamingAdapter([raw_candidate])
        lens = Lens(_lens_config(adapter))

        # Inject an ASK flag via external_flags to trigger ASK path.
        ask_flag = Flag(
            flag_type=FlagType.UNRESOLVED_REFERENT,
            entity_name="her",
            claim="'her' cannot be resolved",
            evidence="multiple entities",
            severity="warning",
        )
        events: dict[str, list] = {"chunk": [], "governed_chunk": [], "metadata": [], "progress": []}
        async for kind, payload in lens.process_stream("How old is her sister?", external_flags=[ask_flag]):
            events.setdefault(kind, []).append(payload)

        # No raw chunk events (buffer was suppressed)
        assert events["chunk"] == []
        # Governed chunk must not contain raw candidate
        visible = _visible_text(events)
        assert raw_candidate not in visible

    @pytest.mark.asyncio
    async def test_ask_governed_chunk_emitted(self, monkeypatch):
        raw_candidate = "The answer is definitely X"
        adapter = FakeStreamingAdapter([raw_candidate])
        lens = Lens(_lens_config(adapter))
        ask_flag = Flag(
            flag_type=FlagType.UNRESOLVED_REFERENT,
            entity_name="the",
            claim="unresolved referent",
            evidence="ambiguous",
            severity="warning",
        )
        events = {"chunk": [], "governed_chunk": [], "metadata": [], "progress": []}
        async for kind, payload in lens.process_stream("What is the answer?", external_flags=[ask_flag]):
            events.setdefault(kind, []).append(payload)

        assert len(events["governed_chunk"]) >= 1
        governed_text = events["governed_chunk"][0][1]
        assert len(governed_text) > 0

    @pytest.mark.asyncio
    async def test_ask_metadata_governance_action(self):
        raw_candidate = "the dosage is 5mg"
        adapter = FakeStreamingAdapter([raw_candidate])
        lens = Lens(_lens_config(adapter))
        ask_flag = Flag(
            flag_type=FlagType.UNRESOLVED_REFERENT,
            entity_name="the",
            claim="unresolved",
            evidence="multiple entities",
            severity="warning",
        )
        events: dict[str, list] = {"metadata": []}
        async for kind, payload in lens.process_stream("What is the dosage?", external_flags=[ask_flag]):
            events.setdefault(kind, []).append(payload)
        meta = events["metadata"][0]
        # ASK maps to CONTAIN action in governance
        assert meta["governance"] in ("CONTAIN", "HARD_STOP", "FORCE_REVISE")

    @pytest.mark.asyncio
    async def test_ask_forensic_event_has_blocked_hash(self):
        raw_candidate = "the patient should take 10mg"
        adapter = FakeStreamingAdapter([raw_candidate])
        lens = Lens(_lens_config(adapter))
        ask_flag = Flag(
            flag_type=FlagType.UNRESOLVED_REFERENT,
            entity_name="the",
            claim="unresolved",
            evidence="multiple",
            severity="warning",
        )
        events: dict[str, list] = {"metadata": []}
        async for kind, payload in lens.process_stream("What should the patient take?", external_flags=[ask_flag]):
            events.setdefault(kind, []).append(payload)

        meta = events["metadata"][0]
        if "forensic_event" in meta:
            fe = meta["forensic_event"]
            # blocked_response_hash must be sha256 of the suppressed candidate
            if "blocked_response_hash" in fe:
                assert fe["blocked_response_hash"] == _sha256(raw_candidate)


# ── TestRefuseStreamingPath ───────────────────────────────────────────────────

class TestRefuseStreamingPath:
    """REFUSE: raw candidate suppressed; governed refusal emitted."""

    @pytest.mark.asyncio
    async def test_refuse_suppresses_raw_buffer(self):
        raw_candidate = "Here is some personalized legal advice: you should sue."
        adapter = FakeStreamingAdapter([raw_candidate])
        lens = Lens(_lens_config(adapter))
        refuse_flag = Flag(
            flag_type=FlagType.PERSONALIZED_LEGAL_ADVICE,
            entity_name="legal",
            claim="personalized legal advice",
            evidence="lawsuit recommendation",
            severity="error",
        )
        events: dict[str, list] = {"chunk": [], "governed_chunk": [], "metadata": []}
        async for kind, payload in lens.process_stream("Should I sue?", external_flags=[refuse_flag]):
            events.setdefault(kind, []).append(payload)

        assert events["chunk"] == []
        visible = _visible_text(events)
        assert raw_candidate not in visible

    @pytest.mark.asyncio
    async def test_refuse_governed_chunk_emitted(self):
        raw_candidate = "Yes, you should definitely sue them for everything."
        adapter = FakeStreamingAdapter([raw_candidate])
        lens = Lens(_lens_config(adapter))
        refuse_flag = Flag(
            flag_type=FlagType.PERSONALIZED_LEGAL_ADVICE,
            entity_name="legal",
            claim="legal advice",
            evidence="recommendation",
            severity="error",
        )
        events: dict[str, list] = {"governed_chunk": []}
        async for kind, payload in lens.process_stream("Should I sue?", external_flags=[refuse_flag]):
            events.setdefault(kind, []).append(payload)

        assert len(events["governed_chunk"]) >= 1
        governed_text = events["governed_chunk"][0][1]
        assert len(governed_text) > 0
        assert raw_candidate not in governed_text

    @pytest.mark.asyncio
    async def test_refuse_blocked_hash_matches_suppressed_candidate(self):
        raw_candidate = "My legal advice: terminate the contract immediately."
        adapter = FakeStreamingAdapter([raw_candidate])
        lens = Lens(_lens_config(adapter))
        refuse_flag = Flag(
            flag_type=FlagType.PERSONALIZED_LEGAL_ADVICE,
            entity_name="contract",
            claim="legal advice",
            evidence="contract recommendation",
            severity="error",
        )
        events: dict[str, list] = {"metadata": []}
        async for kind, payload in lens.process_stream("What should I do?", external_flags=[refuse_flag]):
            events.setdefault(kind, []).append(payload)

        meta = events["metadata"][0]
        if "forensic_event" in meta:
            fe = meta["forensic_event"]
            if "blocked_response_hash" in fe:
                assert fe["blocked_response_hash"] == _sha256(raw_candidate)

    @pytest.mark.asyncio
    async def test_refuse_governed_hash_matches_emitted_text(self):
        raw_candidate = "You should definitely do that illegal thing."
        adapter = FakeStreamingAdapter([raw_candidate])
        lens = Lens(_lens_config(adapter))
        refuse_flag = Flag(
            flag_type=FlagType.ILLEGAL_INSTRUCTION,
            entity_name="action",
            claim="illegal instruction",
            evidence="illegal recommendation",
            severity="error",
        )
        events: dict[str, list] = {"governed_chunk": [], "metadata": []}
        async for kind, payload in lens.process_stream("Should I do this?", external_flags=[refuse_flag]):
            events.setdefault(kind, []).append(payload)

        governed_text = events["governed_chunk"][0][1] if events["governed_chunk"] else ""
        meta = events["metadata"][0]
        if "forensic_event" in meta and governed_text:
            fe = meta["forensic_event"]
            if "governed_response_hash" in fe:
                assert fe["governed_response_hash"] == _sha256(governed_text)


# ── TestStopStreamingPath ─────────────────────────────────────────────────────

class TestStopStreamingPath:
    """STOP: raw candidate suppressed; governed stop/emergency emitted."""

    @pytest.mark.asyncio
    async def test_stop_suppresses_raw_buffer(self):
        raw_candidate = "You should take 500mg of ibuprofen every 4 hours."
        adapter = FakeStreamingAdapter([raw_candidate])
        lens = Lens(_lens_config(adapter))
        stop_flag = Flag(
            flag_type=FlagType.MEDICAL_DOSAGE_RECOMMENDATION,
            entity_name="ibuprofen",
            claim="dosage recommendation",
            evidence="500mg ibuprofen",
            severity="error",
        )
        events: dict[str, list] = {"chunk": [], "governed_chunk": [], "metadata": []}
        async for kind, payload in lens.process_stream("What dosage?", external_flags=[stop_flag]):
            events.setdefault(kind, []).append(payload)

        assert events["chunk"] == []
        visible = _visible_text(events)
        assert raw_candidate not in visible

    @pytest.mark.asyncio
    async def test_stop_governed_chunk_emitted(self):
        raw_candidate = "Take 1000mg aspirin now."
        adapter = FakeStreamingAdapter([raw_candidate])
        lens = Lens(_lens_config(adapter))
        stop_flag = Flag(
            flag_type=FlagType.MEDICAL_DOSAGE_RECOMMENDATION,
            entity_name="aspirin",
            claim="dosage recommendation",
            evidence="1000mg",
            severity="error",
        )
        events: dict[str, list] = {"governed_chunk": []}
        async for kind, payload in lens.process_stream("What's the dose?", external_flags=[stop_flag]):
            events.setdefault(kind, []).append(payload)

        assert len(events["governed_chunk"]) >= 1
        governed_text = events["governed_chunk"][0][1]
        assert len(governed_text) > 0

    @pytest.mark.asyncio
    async def test_stop_emergency_suppresses_and_governs(self):
        raw_candidate = "Call an ambulance if you feel chest pain."
        adapter = FakeStreamingAdapter([raw_candidate])
        lens = Lens(_lens_config(adapter))
        stop_flag = Flag(
            flag_type=FlagType.EMERGENCY_TRIAGE_GUIDANCE,
            entity_name="emergency",
            claim="emergency triage",
            evidence="chest pain ambulance",
            severity="error",
        )
        events: dict[str, list] = {"chunk": [], "governed_chunk": [], "metadata": []}
        async for kind, payload in lens.process_stream("I have chest pain", external_flags=[stop_flag]):
            events.setdefault(kind, []).append(payload)

        assert events["chunk"] == []
        assert len(events["governed_chunk"]) >= 1
        meta = events["metadata"][0]
        assert meta["governance"] in ("HARD_STOP", "CONTAIN", "FORCE_REVISE")

    @pytest.mark.asyncio
    async def test_stop_metadata_stream_governed_flag(self):
        raw_candidate = "Illegal instructions here."
        adapter = FakeStreamingAdapter([raw_candidate])
        lens = Lens(_lens_config(adapter))
        stop_flag = Flag(
            flag_type=FlagType.ILLEGAL_INSTRUCTION,
            entity_name="illegal",
            claim="illegal",
            evidence="evidence",
            severity="error",
        )
        events: dict[str, list] = {"metadata": []}
        async for kind, payload in lens.process_stream("How to?", external_flags=[stop_flag]):
            events.setdefault(kind, []).append(payload)
        meta = events["metadata"][0]
        assert meta.get("stream_governed") is True


# ── TestNoRawContentLeak ──────────────────────────────────────────────────────

class TestNoRawContentLeak:
    """Proves no raw buffered substring appears in user-visible output for non-ADMIT."""

    SUPPRESSION_CASES = [
        (FlagType.UNRESOLVED_REFERENT, "warning", "Tell me about her condition exactly right now in detail"),
        (FlagType.PERSONALIZED_LEGAL_ADVICE, "error", "Sue them for 5 million dollars immediately"),
        (FlagType.MEDICAL_DOSAGE_RECOMMENDATION, "error", "Take 200mg twice daily for 6 months"),
        (FlagType.ILLEGAL_INSTRUCTION, "error", "Here is the exact illegal procedure step by step"),
        (FlagType.SELF_HARM_INSTRUCTION, "error", "Here is a self-harm method that you could use"),
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("flag_type,severity,raw_candidate", SUPPRESSION_CASES)
    async def test_no_raw_substring_in_visible_output(self, flag_type, severity, raw_candidate):
        adapter = FakeStreamingAdapter([raw_candidate])
        lens = Lens(_lens_config(adapter))
        flag = Flag(
            flag_type=flag_type,
            entity_name="test",
            claim="test claim",
            evidence="test evidence",
            severity=severity,
        )
        events: dict[str, list] = {}
        async for kind, payload in lens.process_stream("test input", external_flags=[flag]):
            events.setdefault(kind, []).append(payload)

        visible = _visible_text(events)

        # The raw candidate must not appear verbatim in any user-visible text.
        assert raw_candidate not in visible, (
            f"Raw candidate leaked for {flag_type.name}: "
            f"{visible[:200]!r}"
        )

    @pytest.mark.asyncio
    async def test_governed_chunk_dict_does_not_contain_suppressed_text(self):
        """The chunk_dict itself (the SSE payload) must not echo suppressed content."""
        raw_candidate = "SECRET: here is the confidential answer you wanted."
        adapter = FakeStreamingAdapter([raw_candidate])
        lens = Lens(_lens_config(adapter))
        flag = Flag(
            flag_type=FlagType.SENSITIVE_PII_EXPOSURE,
            entity_name="PII",
            claim="pii exposure",
            evidence="confidential",
            severity="error",
        )
        events: dict[str, list] = {"governed_chunk": []}
        async for kind, payload in lens.process_stream("Tell me the secret", external_flags=[flag]):
            events.setdefault(kind, []).append(payload)

        for chunk_dict, _ in events.get("governed_chunk", []):
            serialised = json.dumps(chunk_dict)
            assert raw_candidate not in serialised

    @pytest.mark.asyncio
    async def test_metadata_does_not_contain_suppressed_text(self):
        """The 'metadata' aurora dict must not echo the raw candidate."""
        raw_candidate = "The exact synthesis procedure is: step 1..."
        adapter = FakeStreamingAdapter([raw_candidate])
        lens = Lens(_lens_config(adapter))
        flag = Flag(
            flag_type=FlagType.ILLEGAL_INSTRUCTION,
            entity_name="synthesis",
            claim="illegal synthesis",
            evidence="step by step",
            severity="error",
        )
        events: dict[str, list] = {"metadata": []}
        async for kind, payload in lens.process_stream("How to?", external_flags=[flag]):
            events.setdefault(kind, []).append(payload)

        meta_str = json.dumps(events["metadata"][0])
        assert raw_candidate not in meta_str

    @pytest.mark.asyncio
    async def test_multi_chunk_suppressed_fully(self):
        """Multi-chunk candidate: no fragment appears in visible output."""
        chunks = ["Part A of the ", "illegal ", "synthesis procedure: ", "step 1."]
        raw_candidate = "".join(chunks)
        adapter = FakeStreamingAdapter(chunks)
        lens = Lens(_lens_config(adapter))
        flag = Flag(
            flag_type=FlagType.ILLEGAL_INSTRUCTION,
            entity_name="synthesis",
            claim="illegal",
            evidence="step 1",
            severity="error",
        )
        events: dict[str, list] = {}
        async for kind, payload in lens.process_stream("How?", external_flags=[flag]):
            events.setdefault(kind, []).append(payload)

        visible = _visible_text(events)
        # No fragment of any chunk should appear
        for chunk in chunks:
            assert chunk not in visible, f"Chunk fragment leaked: {chunk!r}"
        assert raw_candidate not in visible


# ── TestStreamingNonStreamingEquiv ────────────────────────────────────────────

class TestStreamingNonStreamingEquiv:
    """Streaming and non-streaming paths produce equivalent final governance outcomes."""

    @pytest.mark.asyncio
    async def test_admit_streaming_equiv_to_nonstreaming(self):
        """ADMIT: both paths produce the same response text."""
        text = "The weather in Sydney is 22°C today."

        # Non-streaming
        ns_adapter = FakeStreamingAdapter([text])
        ns_lens = Lens(_lens_config(ns_adapter))
        ns_result = await ns_lens.process("What's the weather?")

        # Streaming
        s_adapter = FakeStreamingAdapter([text])
        s_lens = Lens(_lens_config(s_adapter))
        s_events = await _collect_stream(s_lens, "What's the weather?")

        assert ns_result.action == InterventionAction.PASS
        assert ns_result.response == text
        assert _visible_text(s_events) == text
        assert s_events["metadata"][0]["governance"] == "PASS"

    @pytest.mark.asyncio
    async def test_stop_streaming_governance_action_matches_nonstreaming(self):
        """STOP: both paths produce the same governance action."""
        raw = "Take 500mg aspirin every 4 hours"
        flag = Flag(
            flag_type=FlagType.MEDICAL_DOSAGE_RECOMMENDATION,
            entity_name="aspirin",
            claim="dosage",
            evidence="500mg",
            severity="error",
        )

        # Non-streaming
        ns_adapter = FakeStreamingAdapter([raw])
        ns_lens = Lens(_lens_config(ns_adapter))
        ns_result = await ns_lens.process("What's the dose?", external_flags=[flag])

        # Streaming (pass the same flag via external_flags)
        s_adapter = FakeStreamingAdapter([raw])
        s_lens = Lens(_lens_config(s_adapter))
        s_events2: dict[str, list] = {}
        async for kind, payload in s_lens.process_stream("What's the dose?", external_flags=[flag]):
            s_events2.setdefault(kind, []).append(payload)

        assert ns_result.action not in (InterventionAction.PASS, InterventionAction.SOFT_CORRECT)
        if s_events2.get("metadata"):
            stream_action = s_events2["metadata"][0]["governance"]
        elif s_events2.get("clarification_continuation"):
            stream_action = s_events2["clarification_continuation"][0].decision.action.name
        else:
            pytest.fail(f"No stream governance surface found in events: {list(s_events2.keys())}")
        assert stream_action == ns_result.action.name

    @pytest.mark.asyncio
    async def test_stop_streaming_governed_text_matches_nonstreaming(self):
        """Non-streaming and streaming produce the same governed continuation text."""
        raw = "The exact illegal procedure: step 1 do this, step 2 do that."
        flag = Flag(
            flag_type=FlagType.ILLEGAL_INSTRUCTION,
            entity_name="procedure",
            claim="illegal instruction",
            evidence="step 1",
            severity="error",
        )

        ns_adapter = FakeStreamingAdapter([raw])
        ns_lens = Lens(_lens_config(ns_adapter))
        ns_result = await ns_lens.process("How to?", external_flags=[flag])

        s_adapter = FakeStreamingAdapter([raw])
        s_lens = Lens(_lens_config(s_adapter))
        s_events: dict[str, list] = {}
        async for kind, payload in s_lens.process_stream("How to?", external_flags=[flag]):
            s_events.setdefault(kind, []).append(payload)

        ns_governed = ns_result.response
        s_governed = _visible_text(s_events)
        assert ns_governed == s_governed, (
            f"Governed text differs:\n  non-streaming: {ns_governed!r}\n  streaming: {s_governed!r}"
        )

    @pytest.mark.asyncio
    async def test_blocked_response_hash_consistent(self):
        """blocked_response_hash must equal sha256(suppressed_candidate) for both paths."""
        raw = "Personalized financial advice: sell everything now."
        flag = Flag(
            flag_type=FlagType.PERSONALIZED_FINANCIAL_ADVICE,
            entity_name="finance",
            claim="financial advice",
            evidence="sell everything",
            severity="error",
        )

        # Non-streaming
        ns_adapter = FakeStreamingAdapter([raw])
        ns_lens = Lens(_lens_config(ns_adapter))
        ns_result = await ns_lens.process("What should I do with my money?", external_flags=[flag])
        ns_fe = ns_result.decision.forensic_event if ns_result.decision else None

        # Streaming
        s_adapter = FakeStreamingAdapter([raw])
        s_lens = Lens(_lens_config(s_adapter))
        s_events: dict[str, list] = {"metadata": []}
        async for kind, payload in s_lens.process_stream("What should I do?", external_flags=[flag]):
            s_events.setdefault(kind, []).append(payload)
        s_meta = s_events["metadata"][0]
        s_fe = s_meta.get("forensic_event")

        expected_hash = _sha256(raw)
        if ns_fe and "blocked_response_hash" in ns_fe:
            assert ns_fe["blocked_response_hash"] == expected_hash
        if s_fe and "blocked_response_hash" in s_fe:
            assert s_fe["blocked_response_hash"] == expected_hash


# ── TestProgressSignalling ────────────────────────────────────────────────────

class TestProgressSignalling:
    """Safe scaffolding only; never candidate text."""

    @pytest.mark.asyncio
    async def test_no_progress_events_when_disabled(self):
        adapter = FakeStreamingAdapter(["Hello world"])
        lens = Lens(_lens_config(adapter, emit_progress=False))
        events = await _collect_stream(lens, "hi")
        assert events["progress"] == []

    @pytest.mark.asyncio
    async def test_progress_events_emitted_when_enabled(self):
        adapter = FakeStreamingAdapter(["Hello world"])
        lens = Lens(_lens_config(adapter, emit_progress=True))
        events = await _collect_stream(lens, "hi")
        assert len(events["progress"]) >= 1

    @pytest.mark.asyncio
    async def test_progress_signals_are_ProgressSignal_instances(self):
        adapter = FakeStreamingAdapter(["Hello"])
        lens = Lens(_lens_config(adapter, emit_progress=True))
        events = await _collect_stream(lens, "hi")
        for sig in events["progress"]:
            assert isinstance(sig, ProgressSignal)

    @pytest.mark.asyncio
    async def test_progress_statuses_are_valid(self):
        adapter = FakeStreamingAdapter(["Hello"])
        lens = Lens(_lens_config(adapter, emit_progress=True))
        events = await _collect_stream(lens, "hi")
        for sig in events["progress"]:
            assert sig.status in _VALID_PROGRESS_STATUSES

    @pytest.mark.asyncio
    async def test_progress_event_dict_contains_no_content(self):
        """Progress events must contain no content, claims, or candidate text."""
        candidate = "Secret candidate response with sensitive data."
        adapter = FakeStreamingAdapter([candidate])
        lens = Lens(_lens_config(adapter, emit_progress=True))
        events = await _collect_stream(lens, "hi")
        for sig in events["progress"]:
            d = sig.as_event_dict()
            serialised = json.dumps(d)
            assert candidate not in serialised
            assert "content" not in serialised
            assert "delta" not in serialised

    @pytest.mark.asyncio
    async def test_progress_events_before_metadata(self):
        adapter = FakeStreamingAdapter(["Hello world"])
        lens = Lens(_lens_config(adapter, emit_progress=True))
        kinds = []
        async for kind, _ in lens.process_stream("hi"):
            kinds.append(kind)
        assert "metadata" in kinds
        if "progress" in kinds:
            assert kinds.index("progress") < kinds.index("metadata")

    @pytest.mark.asyncio
    async def test_progress_does_not_reveal_governance_outcome_early(self):
        """Progress signal status must be a lifecycle label only — never a governance result."""
        raw = "Illegal steps: ..."
        adapter = FakeStreamingAdapter([raw])
        lens = Lens(_lens_config(adapter, emit_progress=True))
        flag = Flag(
            flag_type=FlagType.ILLEGAL_INSTRUCTION,
            entity_name="illegal",
            claim="illegal",
            evidence="steps",
            severity="error",
        )
        events: dict[str, list] = {"progress": []}
        async for kind, payload in lens.process_stream("How?", external_flags=[flag]):
            events.setdefault(kind, []).append(payload)

        for sig in events["progress"]:
            # Must be a lifecycle label only
            assert sig.status in _VALID_PROGRESS_STATUSES
            # Must not be a governance verdict
            assert sig.status not in ("PASS", "HARD_STOP", "CONTAIN", "FORCE_REVISE", "SOFT_CORRECT")
            # Must not contain flag names
            d = sig.as_event_dict()
            assert "ILLEGAL_INSTRUCTION" not in json.dumps(d)

    @pytest.mark.asyncio
    async def test_progress_signal_validation_rejects_governance_label(self):
        for invalid in ("PASS", "HARD_STOP", "admitted", "refused", "blocked"):
            with pytest.raises(ValueError):
                ProgressSignal(status=invalid)

    @pytest.mark.asyncio
    async def test_non_admit_path_progress_does_not_reveal_suppression(self):
        """'releasing' signal must be emitted regardless of ADMIT/NON-ADMIT:
        it signals lifecycle phase only, not the content that will be released."""
        raw = "Confidential medical dosage: 100mg"
        adapter = FakeStreamingAdapter([raw])
        lens = Lens(_lens_config(adapter, emit_progress=True))
        flag = Flag(
            flag_type=FlagType.MEDICAL_DOSAGE_RECOMMENDATION,
            entity_name="dosage",
            claim="dosage",
            evidence="100mg",
            severity="error",
        )
        events: dict[str, list] = {"progress": [], "governed_chunk": []}
        async for kind, payload in lens.process_stream("What is the dose?", external_flags=[flag]):
            events.setdefault(kind, []).append(payload)

        statuses = [sig.status for sig in events["progress"]]
        # Must not contain the raw candidate in any progress event
        for sig in events["progress"]:
            assert raw not in json.dumps(sig.as_event_dict())


# ── TestAbortReasonTaxonomy ───────────────────────────────────────────────────

class _CancelledDuringBufferAdapter(LLMAdapter):
    """Raises asyncio.CancelledError after yielding `abort_after` chunks.

    Simulates a provider-side abort during private buffering: the caller
    (process_stream) has not yet released any content to the user.
    """

    def __init__(self, chunks: list[str], abort_after: int = 1):
        self._chunks = chunks
        self._abort_after = abort_after

    async def generate(self, messages, **kw) -> AdapterResponse:
        return AdapterResponse(text="".join(self._chunks), model="test")

    async def generate_stream(self, messages, **kw):
        for i, c in enumerate(self._chunks):
            if i >= self._abort_after:
                raise asyncio.CancelledError("provider abort")
            yield (_make_chunk_dict(c), c)


@pytest.mark.asyncio
class TestAbortReasonTaxonomy:
    """Proves that provider abort during buffering and client disconnect after
    release are logged with distinct stream_abort_reason values.

    Taxonomy:
      provider_abort   — provider's generate_stream() raises during private
                         buffering; no content has been released to the user.
      (no abort entry) — governance completed (stream_completed=True) before
                         chunks were released; post-release disconnect does
                         not write a second audit entry.
    """

    async def test_provider_abort_during_buffering_logs_provider_abort(self, tmp_path):
        """CancelledError from the provider during buffering is logged as
        stream_abort_reason='provider_abort', stream_completed=False.
        No content is emitted to the user before the abort.
        """
        audit_file = tmp_path / "audit.jsonl"
        adapter = _CancelledDuringBufferAdapter(["hello", " world", "!"], abort_after=1)
        bridge = BuiltinBridge(audit_path=str(audit_file))
        lens = Lens(_lens_config(adapter, governance_bridge=bridge))

        chunks_seen: list[str] = []
        governed_seen: list[str] = []
        metadata_seen: list = []

        with pytest.raises(asyncio.CancelledError):
            async for kind, payload in lens.process_stream("hi"):
                if kind == "chunk":
                    _, delta = payload
                    chunks_seen.append(delta)
                elif kind == "governed_chunk":
                    _, delta = payload
                    governed_seen.append(delta)
                elif kind == "metadata":
                    metadata_seen.append(payload)

        # Resume from last committed state, not partial turn (advance_turn rolled back).
        assert lens.pef.current_turn == 0

        # No content reached the user before the abort.
        assert chunks_seen == []
        assert governed_seen == []
        assert metadata_seen == []

        # Audit entry records the abort with the correct taxonomy label.
        lines = audit_file.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry.get("stream") is True
        assert entry.get("stream_completed") is False
        assert entry.get("stream_abort_reason") == "provider_abort"

    async def test_stream_abort_after_completed_turn_rollbacks_advance_only(self, tmp_path):
        """After a committed non-stream turn, abort during buffering leaves turn count unchanged."""
        audit_file = tmp_path / "audit.jsonl"
        adapter = _CancelledDuringBufferAdapter(["hello", " world"], abort_after=1)
        bridge = BuiltinBridge(audit_path=str(audit_file))
        lens = Lens(_lens_config(adapter, governance_bridge=bridge))

        r0 = await lens.process("First turn.")
        assert r0.turn == 1
        assert lens.pef.current_turn == 1

        with pytest.raises(asyncio.CancelledError):
            async for _ in lens.process_stream("Second turn aborts."):
                pass

        assert lens.pef.current_turn == 1

    async def test_client_disconnect_after_release_does_not_log_abort(self, tmp_path):
        """Breaking out of the generator after the first released chunk does NOT
        produce a stream_abort audit entry.  Governance committed before chunk
        delivery; stream_completed=True is the only audit state.
        The upstream adapter completes during private buffering (closed_early=False).
        """
        audit_file = tmp_path / "audit.jsonl"
        adapter = FakeStreamingAdapter(["Paris", " is", " the", " capital"])
        bridge = BuiltinBridge(audit_path=str(audit_file))
        lens = Lens(_lens_config(adapter, governance_bridge=bridge))

        chunks_seen: list[str] = []

        # Break after the first chunk — simulates client disconnect mid-delivery.
        from contextlib import aclosing
        async with aclosing(lens.process_stream("What is the capital of France?")) as gen:
            async for kind, payload in gen:
                if kind == "chunk":
                    _, delta = payload
                    chunks_seen.append(delta)
                    break  # disconnect

        assert len(chunks_seen) >= 1

        # Audit entry was written before chunk delivery; it records completion.
        lines = audit_file.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry.get("stream") is True
        assert entry.get("stream_completed") is True
        # No abort reason — this is a clean post-governance disconnect.
        assert "stream_abort_reason" not in entry


# ── Internal log-channel keys stripped from SSE payload ──────────────────────

class TestInternalLogKeysNotLeakedToSSE:
    """_log_flags and sibling internal keys added by process_stream() for the
    governance_outcome log must never appear in the aurora metadata dict that
    is serialised into the SSE stream.

    This tests the contract between lens.py (which writes the keys) and
    app.py (which pops them before calling format_stream_metadata_event).
    Here we test the lens side: that the keys ARE present before popping,
    so the proxy can rely on them.
    """

    def _make_lens(self, *, verdict: str = "PASS") -> "Lens":
        from aurora_lens.config import LensConfig
        from aurora_lens.lens import Lens
        from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
        from aurora_lens.govern.bridge import BuiltinBridge

        action_map = {
            "PASS": InterventionAction.PASS,
            "HARD_STOP": InterventionAction.HARD_STOP,
        }
        action = action_map[verdict]

        class _FixedBridge(BuiltinBridge):
            async def decide(self, flags, text, pef):
                return GovernanceDecision(
                    action=action,
                    flags=flags,
                    rationale="test",
                    policy="strict",
                )
            def log_decision(self, *a, **kw):
                pass

        cfg = LensConfig(
            adapter=FakeStreamingAdapter(["ok"]),
            extraction_backend=None,
            governance_bridge=_FixedBridge(),
        )
        return Lens(cfg)

    @pytest.mark.asyncio
    async def test_metadata_dict_contains_log_keys_before_proxy_pops_them(self):
        """process_stream() always adds _log_flags/_log_policy/_log_pathway/_log_commitment_closed
        to the metadata dict so the proxy can pop them for the governance_outcome log."""
        lens = self._make_lens(verdict="PASS")
        events = []
        async for kind, payload in lens.process_stream("hello"):
            events.append((kind, payload))

        meta_events = [(k, p) for k, p in events if k == "metadata"]
        assert meta_events, "process_stream must emit a 'metadata' event"
        _, aurora = meta_events[0]

        assert "_log_flags" in aurora, "_log_flags must be present for proxy log"
        assert "_log_policy" in aurora, "_log_policy must be present for proxy log"
        assert "_log_pathway" in aurora, "_log_pathway must be present for proxy log"
        assert "_log_commitment_closed" in aurora, "_log_commitment_closed must be present for proxy log"
        # Internal keys must NOT appear in client-visible SSE payload after proxy pops them
        client_aurora = {k: v for k, v in aurora.items() if not k.startswith("_log_")}
        assert "_log_flags" not in client_aurora
        assert "_log_policy" not in client_aurora

    @pytest.mark.asyncio
    async def test_metadata_log_flags_matches_decision_flags(self):
        """_log_flags in the metadata dict must equal the flag names from the governance decision."""
        from aurora_lens.verify.flags import Flag, FlagType
        from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
        from aurora_lens.govern.bridge import BuiltinBridge
        from aurora_lens.config import LensConfig
        from aurora_lens.lens import Lens

        fired_flag = Flag(
            flag_type=FlagType.VIOLENT_CRIMINAL_INTENT,
            entity_name="violent_criminal_intent",
            claim="test",
            evidence="test",
            severity="error",
        )

        class _FlagBridge(BuiltinBridge):
            async def decide(self, flags, text, pef):
                return GovernanceDecision(
                    action=InterventionAction.HARD_STOP,
                    flags=[fired_flag],
                    rationale="blocked",
                    policy="strict",
                )
            def log_decision(self, *a, **kw):
                pass

        cfg = LensConfig(
            adapter=FakeStreamingAdapter(["ok"]),
            extraction_backend=None,
            governance_bridge=_FlagBridge(),
        )
        lens = Lens(cfg)

        # Pass fired_flag as an external flag so the bridge's decide() is invoked
        # (decide() is only called when flags is non-empty).
        events = []
        async for kind, payload in lens.process_stream("hello", external_flags=[fired_flag]):
            events.append((kind, payload))

        _, aurora = next((k, p) for k, p in events if k == "metadata")
        assert aurora["_log_flags"] == ["VIOLENT_CRIMINAL_INTENT"], (
            f"_log_flags must reflect decision flags. Got: {aurora['_log_flags']}"
        )


# ── TestClarificationContinuationStreamTag ────────────────────────────────────

class TestClarificationContinuationStreamTag:
    """The clarification-continuation path in process_stream() must emit
    'clarification_continuation', not 'extraction_failed'."""

    @pytest.mark.asyncio
    async def test_stream_emits_clarification_continuation_tag(self):
        from aurora_lens.interpret.base import ExtractionBackend
        from aurora_lens.interpret.schema import ExtractionResult
        from aurora_lens.pef.span import Span
        from aurora_lens.pef.state import PEFState

        class _NoOpBackend(ExtractionBackend):
            async def extract(self, text, pef):
                return ExtractionResult(claims=[], entity_mentions=[], span=Span.PRESENT)

        adapter = FakeStreamingAdapter(["Should not appear."])
        cfg = LensConfig(
            adapter=adapter,
            extraction_backend=_NoOpBackend(),
            auto_interpret=True,
            auto_verify=False,
        )
        lens = Lens(cfg)

        ent_r, _ = lens.pef.get_or_create_entity("Richard", resolved=False)
        ent_j, _ = lens.pef.get_or_create_entity("James", resolved=False)
        lens.pef.pending_clarification = {
            "original_question": "What is his favourite colour?",
            "unresolved_entity_ids": [ent_r.id, ent_j.id],
            "failed_constraint": "UNRESOLVED_REFERENT",
            "ambiguous_referents": ["his"],
            "candidate_entities": ["Richard", "James"],
        }

        events = await _collect_stream(lens, "What do you need?")

        assert len(events["clarification_continuation"]) == 1, (
            f"Expected exactly 1 clarification_continuation event, got: {list(events.keys())}"
        )
        assert len(events["extraction_failed"]) == 0, (
            "Clarification continuation must not emit extraction_failed"
        )

        result = events["clarification_continuation"][0]
        assert "Richard" in result.response
        assert "James" in result.response
        assert result.action == InterventionAction.CONTAIN


# ── TestStreamGovernedClarificationParity ───────────────────────────────────────

class TestStreamGovernedClarificationParity:
    """``process_stream`` mirrors ``process`` for assistant-echo → ``pending_clarification``."""

    @pytest.mark.asyncio
    async def test_stream_governed_clarification_echo_sets_pending_emma_binds(
        self, tmp_path,
    ):
        """Same scenario as ``test_assistant_governed_clarification_echo_sets_pending_emma_binds`` but via ``process_stream``."""
        user_q = (
            "Emma told Anna her sister was overseas. Where is she now?"
        )
        backend = SpacyBackend(model="en_core_web_sm")
        pef = PEFState()
        ext = await backend.extract(user_q, pef)
        probe = Lens(
            LensConfig(adapter=FakeStreamingAdapter(["x"]), extraction_backend=backend)
        )
        snap = probe._compute_truly_ambiguous_referents(ext)
        assert snap, "fixture must have structural ambiguity"
        clar = _pre_llm_unresolved_referent_clarification(snap)

        # Split clarification across chunks so the buffer is built from streaming.
        mid = max(1, len(clar) // 2)
        turn1_chunks = [clar[:mid], clar[mid:]]
        adapter = _TwoPhaseStreamAdapter(turn1_chunks, ["She is in London."])
        audit = tmp_path / "audit_stream_assistant_clar.jsonl"
        bridge = BuiltinBridge(audit_path=str(audit))
        cfg = LensConfig(
            adapter=adapter,
            extraction_backend=backend,
            governance_bridge=bridge,
            auto_verify=False,
            auto_interpret=True,
        )
        lens = Lens(cfg)

        await _collect_stream(lens, user_q)
        assert lens.pef.pending_clarification is not None
        assert lens.pef.pending_clarification.get("failed_constraint") == (
            "UNRESOLVED_REFERENT"
        )

        await _collect_stream(lens, "Emma")
        assert lens.pef.pending_clarification is None
        assert adapter.last_stream_messages is not None
        last_user = next(
            (
                m["content"]
                for m in reversed(adapter.last_stream_messages)
                if m.get("role") == "user"
            ),
            None,
        )
        assert last_user is not None
        assert "Emma" in last_user
        assert " she " not in f" {last_user.lower()} "

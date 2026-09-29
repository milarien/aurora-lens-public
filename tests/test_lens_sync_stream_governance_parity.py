"""Sync vs stream governance parity via shared post-generation pipeline."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from aurora_lens.config import LensConfig
from aurora_lens.govern.bridge import BuiltinBridge
from aurora_lens.govern.decision import InterventionAction
from aurora_lens.lens import Lens
from aurora_lens.lens_governed_turn import (
    envelope_from_lens_result,
    governance_parity_material,
)
from aurora_lens.verify.flags import Flag, FlagType


def _load_stream_helpers():
    path = Path(__file__).resolve().parent / "test_streaming_governance.py"
    spec = importlib.util.spec_from_file_location("stream_gov_helpers", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.FakeStreamingAdapter, mod._collect_stream, mod._visible_text, mod._lens_config


def _stream_parity_material(metadata: dict, visible_text: str) -> dict:
    return {
        "governance_outcome": metadata["governance"],
        "pre_llm_blocked": False,
        "llm_called": True,
        "action": metadata["governance"],
        "pathway_id": metadata.get("_log_pathway"),
        "commitment_closed": metadata.get("_log_commitment_closed"),
        "interaction_open": metadata.get("interaction_open"),
        "rationale": None,
        "flag_types": list(metadata.get("_log_flags") or []),
        "response": visible_text,
        "self_refused": metadata.get("self_refused", False),
        "epistemic_normalisation_applied": metadata.get(
            "epistemic_normalisation_applied", False
        ),
        "policy": metadata.get("_log_policy"),
        "output_mode": metadata.get("output_mode"),
    }


async def _run_parity_case(
    *,
    user_input: str,
    raw_text: str,
    external_flags: list[Flag] | None = None,
) -> tuple[dict, dict]:
    FakeStreamingAdapter, _collect_stream, _visible_text, _lens_config = _load_stream_helpers()
    adapter_sync = FakeStreamingAdapter([raw_text])
    adapter_stream = FakeStreamingAdapter([raw_text])
    cfg = _lens_config(adapter_sync, emit_progress=False)
    cfg_stream = _lens_config(adapter_stream, emit_progress=False)

    sync_lens = Lens(cfg)
    stream_lens = Lens(cfg_stream)
    sync_result = await sync_lens.process(user_input, external_flags=external_flags)
    sync_material = governance_parity_material(
        envelope_from_lens_result(
            sync_result,
            pre_llm_blocked=False,
            llm_called=True,
            route_reason_code="adapter_checker_bridge_pipeline",
        )
    )

    stream_events = await _collect_stream(
        stream_lens,
        user_input,
        external_flags=external_flags,
    )
    assert stream_events.get("metadata"), "stream path must emit metadata"
    meta = stream_events["metadata"][0]
    stream_material = _stream_parity_material(meta, _visible_text(stream_events))
    return sync_material, stream_material


class TestLensSyncStreamGovernanceParity:
    @pytest.mark.asyncio
    async def test_pass_parity(self):
        sync_m, stream_m = await _run_parity_case(
            user_input="What's the weather?",
            raw_text="The weather in Sydney is 22°C today.",
        )
        assert sync_m["governance_outcome"] == "PASS"
        assert sync_m["action"] == stream_m["action"]
        assert sync_m["flag_types"] == stream_m["flag_types"]
        assert sync_m["response"] == stream_m["response"]
        assert sync_m["llm_called"] == stream_m["llm_called"]
        assert sync_m["pre_llm_blocked"] == stream_m["pre_llm_blocked"]

    @pytest.mark.asyncio
    async def test_force_revise_parity(self):
        flag = Flag(
            flag_type=FlagType.UNVERIFIED_FACT_ASSERTION,
            entity_name="finance",
            claim="unsupported fact",
            evidence="unsupported",
            severity="warning",
        )
        sync_m, stream_m = await _run_parity_case(
            user_input="What is the exact revenue figure?",
            raw_text="The exact revenue was $42 million last quarter.",
            external_flags=[flag],
        )
        assert sync_m["governance_outcome"] == "FORCE_REVISE"
        assert sync_m["action"] == stream_m["action"]
        assert sync_m["flag_types"] == stream_m["flag_types"]
        assert sync_m["pathway_id"] == stream_m["pathway_id"]
        assert sync_m["response"] == stream_m["response"]

    @pytest.mark.asyncio
    async def test_contain_parity(self):
        flag = Flag(
            flag_type=FlagType.UNRESOLVED_REFERENT,
            entity_name="she",
            claim="ambiguous referent",
            evidence="she",
            severity="warning",
            candidates=("Emma", "Lucy"),
        )
        sync_m, stream_m = await _run_parity_case(
            user_input="How old is her sister?",
            raw_text="Her sister is 12 years old.",
            external_flags=[flag],
        )
        assert sync_m["governance_outcome"] == "CONTAIN"
        assert sync_m["action"] == stream_m["action"]
        assert sync_m["flag_types"] == stream_m["flag_types"]
        assert sync_m["pathway_id"] == stream_m["pathway_id"]
        assert sync_m["commitment_closed"] == stream_m["commitment_closed"]
        assert sync_m["response"] == stream_m["response"]

    @pytest.mark.asyncio
    async def test_hard_stop_parity(self):
        flag = Flag(
            flag_type=FlagType.MEDICAL_DOSAGE_RECOMMENDATION,
            entity_name="aspirin",
            claim="dosage",
            evidence="500mg",
            severity="error",
        )
        sync_m, stream_m = await _run_parity_case(
            user_input="What's the dose?",
            raw_text="Take 500mg aspirin every 4 hours.",
            external_flags=[flag],
        )
        assert sync_m["governance_outcome"] == "HARD_STOP"
        assert sync_m["action"] == stream_m["action"]
        assert sync_m["flag_types"] == stream_m["flag_types"]
        assert sync_m["pathway_id"] == stream_m["pathway_id"]
        assert sync_m["response"] == stream_m["response"]

"""Replay verification for :func:`aurora_lens.govern.audit_io.verify_pef_linkage`.

Synthetic JSONL pairs plus one proof-scenario audit log from :class:`ProofScenarioBridge`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_lens import MockAdapter  # noqa: E402
from test_pef_proof_scenarios import ProofScenarioBridge, _unverified_flag  # noqa: E402

from aurora_lens.config import LensConfig
from aurora_lens.govern.audit_io import verify_pef_linkage
from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
from aurora_lens.interpret.base import ExtractionBackend
from aurora_lens.interpret.schema import ExtractionResult
from aurora_lens.lens import Lens
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_verify_pef_linkage_synthetic_pass(tmp_path: Path) -> None:
    """Minimal two-line log: prev snapshot classifies to ``fresh``; curr matches."""
    p = tmp_path / "audit.jsonl"
    _write_jsonl(
        p,
        [
            {"pef_snapshot": {}},
            {"pef_turn_classification": "fresh"},
        ],
    )
    ok, checked, fail_idx, reason = verify_pef_linkage(p)
    assert ok is True
    assert checked == 1
    assert fail_idx is None
    assert reason is None


def test_verify_pef_linkage_synthetic_fail_mismatch(tmp_path: Path) -> None:
    """Same prev snapshot; wrong ``pef_turn_classification`` on curr."""
    p = tmp_path / "audit.jsonl"
    _write_jsonl(
        p,
        [
            {"pef_snapshot": {}},
            {"pef_turn_classification": "held_refusal"},
        ],
    )
    ok, checked, fail_idx, reason = verify_pef_linkage(p)
    assert ok is False
    assert checked == 1
    assert fail_idx == 1
    assert reason == "turn_classification_mismatch"


class TestPefProofAuditLinkageReplay:
    """Lens + ProofScenarioBridge rows replay-clean under ``verify_pef_linkage``."""

    @pytest.mark.asyncio
    async def test_proof_scenario_replay_passes(self, tmp_path: Path) -> None:
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

        p = tmp_path / "audit.jsonl"
        rows = [e for e in bridge.audit_log if not e.get("pre_llm")]
        assert len(rows) >= 2
        # Same linkage fields as real JSONL; omit non-JSON types (e.g. ``InterventionAction``).
        minimal: list[dict] = []
        for e in rows:
            row: dict = {}
            if e.get("pef_snapshot") is not None:
                row["pef_snapshot"] = e["pef_snapshot"]
            if e.get("pef_turn_classification") is not None:
                row["pef_turn_classification"] = e["pef_turn_classification"]
            minimal.append(row)
        _write_jsonl(p, minimal)

        ok, checked, fail_idx, reason = verify_pef_linkage(p)
        assert ok is True
        assert checked >= 1
        assert fail_idx is None
        assert reason is None

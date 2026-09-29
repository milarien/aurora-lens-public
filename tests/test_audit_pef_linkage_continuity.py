"""Forensic PEF audit linkage continuity for JSONL rows.

``verify_pef_linkage`` requires each row's ``pef_turn_classification`` to equal
``classify_pef_turn_start`` applied to the *previous* row's ``pef_snapshot``. These
tests guard the Lens writer path (frozen turn-start snapshot + derived classification).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from aurora_lens.adapters.base import AdapterResponse, LLMAdapter
from aurora_lens.config import LensConfig
from aurora_lens.govern.audit_io import verify_pef_linkage_detailed
from aurora_lens.govern.bridge import BuiltinBridge
from aurora_lens.govern.decision import InterventionAction
from aurora_lens.govern.policy import DEFAULT_STRICT
from aurora_lens.interpret.base import ExtractionBackend
from aurora_lens.interpret.schema import ExtractionResult
from aurora_lens.lens import Lens
from aurora_lens.pef.span import Span
from aurora_lens.proxy.session_store import SessionRecord
from aurora_lens.verify.flags import Flag, FlagType


class _EmptyExtract(ExtractionBackend):
    async def extract(self, text: str, pef) -> ExtractionResult:  # noqa: ANN001
        return ExtractionResult(claims=[], entity_mentions=[], span=Span.PRESENT)


class _SeqAdapter(LLMAdapter):
    def __init__(self, texts: list[str]) -> None:
        self._texts = iter(texts)

    async def generate(self, messages):  # noqa: ANN001
        t = next(self._texts)
        return AdapterResponse(text=t, model="mock", usage=None)


@pytest.mark.asyncio
async def test_verify_pef_linkage_after_hard_stop_then_pass_same_lens(tmp_path: Path) -> None:
    """HARD_STOP leaves durable stop posture; the following PASS row must link cleanly."""
    audit = tmp_path / "pef_link_same_lens.jsonl"
    bridge = BuiltinBridge(policy=DEFAULT_STRICT, audit_path=audit)
    lens = Lens(
        LensConfig(
            adapter=_SeqAdapter(["model line one", "model line two"]),
            extraction_backend=_EmptyExtract(),
            governance_bridge=bridge,
            auto_verify=False,
        ),
        session_id="s-link-same",
    )
    ext = [
        Flag(
            flag_type=FlagType.SELF_HARM_INSTRUCTION,
            entity_name="test",
            claim="x",
            evidence="span",
            severity="error",
        )
    ]
    r1 = await lens.process("first user line", external_flags=ext)
    assert r1.action == InterventionAction.HARD_STOP
    r2 = await lens.process("second user line")
    assert r2.action == InterventionAction.PASS

    lines = [ln for ln in audit.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) >= 2
    d = verify_pef_linkage_detailed(audit, n=len(lines) + 5)
    assert d["ok"], d
    assert int(d["pairs_checked"]) >= 1


@pytest.mark.asyncio
async def test_verify_pef_linkage_after_hard_stop_survives_session_reload(tmp_path: Path) -> None:
    """Simulate proxy session reload: PEF round-trip must not emit a bogus ``fresh`` link row."""
    audit = tmp_path / "pef_link_reload.jsonl"
    bridge = BuiltinBridge(policy=DEFAULT_STRICT, audit_path=audit)
    cfg = LensConfig(
        adapter=_SeqAdapter(["model a", "model b"]),
        extraction_backend=_EmptyExtract(),
        governance_bridge=bridge,
        auto_verify=False,
    )
    ext = [
        Flag(
            flag_type=FlagType.SELF_HARM_INSTRUCTION,
            entity_name="test",
            claim="x",
            evidence="span",
            severity="error",
        )
    ]
    lens1 = Lens(cfg, session_id="s-link-reload")
    r1 = await lens1.process("turn one", external_flags=ext)
    assert r1.action == InterventionAction.HARD_STOP

    rec = SessionRecord.from_pef(lens1.pef, time.time() + 3600.0, revision=0)
    lens2 = Lens(cfg, initial_pef=rec.to_pef(), session_id="s-link-reload")
    r2 = await lens2.process("turn two")
    assert r2.action == InterventionAction.PASS

    lines = [ln for ln in audit.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) >= 2
    d = verify_pef_linkage_detailed(audit, n=len(lines) + 5)
    assert d["ok"], d
    assert int(d["pairs_checked"]) >= 1

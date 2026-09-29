"""Audit and governance classification for upstream insufficient-context replies."""

from __future__ import annotations

from pathlib import Path

import pytest
from aurora_lens.config import LensConfig
from aurora_lens.govern.bridge import BuiltinBridge
from aurora_lens.govern.clarification_audit import read_jsonl_audit_entries
from aurora_lens.govern.decision import InterventionAction
from aurora_lens.interpret.spacy_backend import SpacyBackend
from aurora_lens.lens import Lens
from aurora_lens.verify.checker import Checker
from aurora_lens.verify.flags import FlagType
from tests.test_lens import MockAdapter


_INSUFFICIENT_UPSTREAM = (
    "I don't have enough information to provide a response.\n"
    "Can you provide more context about Carol and the situation?"
)


class _SingleResponseAdapter(MockAdapter):
    def __init__(self, text: str) -> None:
        super().__init__(responses=[text])


def _governance_rows(entries: list[dict]) -> list[dict]:
    return [e for e in entries if e.get("outcome") and e.get("turn") is not None]


@pytest.mark.asyncio
async def test_upstream_insufficient_context_not_classified_as_pass(tmp_path: Path) -> None:
    """Case 1: upstream clarification reply must not audit as ordinary PASS."""
    audit = tmp_path / "insufficient.jsonl"
    bridge = BuiltinBridge(audit_path=str(audit))
    lens = Lens(
        LensConfig(
            adapter=_SingleResponseAdapter(_INSUFFICIENT_UPSTREAM),
            governance_bridge=bridge,
            extraction_backend=SpacyBackend(),
            auto_interpret=True,
            auto_verify=True,
        )
    )

    result = await lens.process("What did Carol agree to?")

    assert result.action == InterventionAction.CONTAIN
    assert result.action != InterventionAction.PASS
    assert any(f.flag_type == FlagType.UPSTREAM_INSUFFICIENT_CONTEXT for f in result.flags)

    entries = read_jsonl_audit_entries(audit)
    gov = _governance_rows(entries)
    assert len(gov) == 1
    row = gov[0]
    assert row["outcome"] == "CONTAIN"
    assert row["outcome"] != "PASS"
    assert "UPSTREAM_INSUFFICIENT_CONTEXT" in row.get("failed_constraints", [])
    assert row.get("epistemic_state") == "insufficient_context"
    assert row.get("pathway_id") == "P_ASK_MISSING_FACT"
    assert _INSUFFICIENT_UPSTREAM in (row.get("governed_response") or "")


@pytest.mark.asyncio
async def test_upstream_resolved_answer_still_passes(tmp_path: Path) -> None:
    """Case 2: substantive upstream answer remains PASS with no insufficient state."""
    audit = tmp_path / "resolved.jsonl"
    bridge = BuiltinBridge(audit_path=str(audit))
    lens = Lens(
        LensConfig(
            adapter=_SingleResponseAdapter("Carol agreed to mediation."),
            governance_bridge=bridge,
            extraction_backend=SpacyBackend(),
            auto_interpret=True,
            auto_verify=True,
        )
    )

    result = await lens.process("What did Carol agree to?")

    assert result.action == InterventionAction.PASS
    assert not any(
        f.flag_type == FlagType.UPSTREAM_INSUFFICIENT_CONTEXT for f in result.flags
    )

    entries = read_jsonl_audit_entries(audit)
    gov = _governance_rows(entries)
    assert len(gov) == 1
    assert gov[0]["outcome"] == "PASS"
    assert gov[0].get("epistemic_state") is None
    assert "UPSTREAM_INSUFFICIENT_CONTEXT" not in gov[0].get("failed_constraints", [])


@pytest.mark.asyncio
async def test_checker_emits_upstream_insufficient_context_flag() -> None:
    from aurora_lens.pef.state import PEFState

    checker = Checker(SpacyBackend())
    flags = await checker.check(
        _INSUFFICIENT_UPSTREAM,
        PEFState(session_id="test"),
        user_input="What did Carol agree to?",
    )
    assert any(f.flag_type == FlagType.UPSTREAM_INSUFFICIENT_CONTEXT for f in flags)
    assert not any(f.flag_type == FlagType.EXTRACTION_EMPTY for f in flags)


@pytest.mark.asyncio
async def test_checker_does_not_flag_substantive_answer() -> None:
    from aurora_lens.pef.state import PEFState

    checker = Checker(SpacyBackend())
    flags = await checker.check(
        "Carol agreed to mediation.",
        PEFState(session_id="test"),
        user_input="What did Carol agree to?",
    )
    assert not any(f.flag_type == FlagType.UPSTREAM_INSUFFICIENT_CONTEXT for f in flags)

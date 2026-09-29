"""Release-style gate: black-box mini-evaluator contracts."""

from __future__ import annotations

import json

import pytest

from aurora_lens.eval.mini_evaluator import RUNNERS, main


@pytest.mark.asyncio
async def test_mini_eval_clean_pass_contract():
    r = await RUNNERS["clean_pass"]()
    assert r["ok"] is True
    assert r["action"] == "PASS"


@pytest.mark.asyncio
async def test_mini_eval_ambiguity_no_llm():
    r = await RUNNERS["ambiguity_ask"]()
    assert r["ok"] is True
    assert r["checks"]["adapter_never_called"] is True
    assert "UNRESOLVED_REFERENT" in r["flags"]


@pytest.mark.asyncio
async def test_mini_eval_hard_stop():
    r = await RUNNERS["hard_stop_continuation"]()
    assert r["ok"] is True
    assert r["action"] == "HARD_STOP"


@pytest.mark.asyncio
async def test_mini_eval_audit_verification():
    r = await RUNNERS["audit_verification"]()
    assert r["ok"] is True
    assert r["checks"]["verify_audit_entries"] is True
    assert r["checks"]["verify_chain"] is True
    assert r["checks"]["verify_forensic_state_hash"] is True
    assert r["checks"]["forensic_event_self_hash"] is True


def test_mini_eval_cli_json_strict(capsys):
    code = main(["--json", "--strict"])
    assert code == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["all_ok"] is True
    assert len(data["scenarios"]) == 4


def test_mini_eval_cli_single_scenario():
    code = main(["--strict", "-s", "clean_pass"])
    assert code == 0

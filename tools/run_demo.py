#!/usr/bin/env python3
"""Start-here demo — one command, three governance outcomes, one audit file.

No provider API key and no network call to a language model. The three
scenarios use in-process mock adapters. spaCy and the ``en_core_web_sm``
model are required: governance extraction loads them even though these
scenarios do not call a model provider.

Usage:
    pip install ".[proxy,spacy]"
    python -m spacy download en_core_web_sm
    python tools/run_demo.py

Writes: start_here_demo_audit.jsonl (repository root)
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from aurora_lens.adapters.base import AdapterResponse, LLMAdapter
from aurora_lens.config import LensConfig
from aurora_lens.govern.canonical_bridge import CanonicalScannerGateBridge
from aurora_lens.govern.decision import InterventionAction
from aurora_lens.interpret.base import ExtractionBackend
from aurora_lens.interpret.schema import ExtractedClaim, ExtractionResult
from aurora_lens.lens import Lens
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = REPO_ROOT / "start_here_demo_audit.jsonl"
DEMO_SIGNING_KEY = b"start-here-demo-key"


class _MockAdapter(LLMAdapter):
    def __init__(self, text: str = "ok.") -> None:
        self._text = text

    async def generate(self, messages: list[dict[str, str]], **kwargs) -> AdapterResponse:
        return AdapterResponse(text=self._text, model="start-here-mock")


class _PassBackend(ExtractionBackend):
    async def extract(self, text: str, pef: PEFState) -> ExtractionResult:
        if "engineer" in text.lower():
            return ExtractionResult(
                claims=[
                    ExtractedClaim(
                        subject="Alice",
                        relation="WORKS_AT",
                        obj="Acme Corp",
                        span=Span.PRESENT,
                        negated=False,
                        evidence=text,
                    )
                ],
                entity_mentions=["Alice"],
                span=Span.PRESENT,
            )
        return ExtractionResult(
            claims=[],
            entity_mentions=["Alice"],
            span=Span.PRESENT,
        )


class _AmbiguousBackend(ExtractionBackend):
    def __init__(self) -> None:
        self._n = 0

    async def extract(self, text: str, pef: PEFState) -> ExtractionResult:
        self._n += 1
        if self._n == 1:
            return ExtractionResult(
                claims=[],
                entity_mentions=["Emma", "Anna"],
                span=Span.PRESENT,
                ambiguous_referents=["her"],
            )
        return ExtractionResult(claims=[], entity_mentions=[], span=Span.PRESENT)


class _MedicalBackend(ExtractionBackend):
    async def extract(self, text: str, pef: PEFState) -> ExtractionResult:
        text_lower = text.lower()
        claims: list[ExtractedClaim] = []
        mentions: list[str] = []
        if "patient" in text_lower:
            mentions.append("Patient")
        if "no ecg" in text_lower or "no troponin" in text_lower:
            claims.append(
                ExtractedClaim(
                    subject="Patient",
                    relation="HAS",
                    obj="myocardial infarction",
                    span=Span.PRESENT,
                    negated=True,
                    evidence=text,
                )
            )
        if "myocardial infarction" in text_lower and "no ecg" not in text_lower:
            claims.append(
                ExtractedClaim(
                    subject="Patient",
                    relation="HAS",
                    obj="myocardial infarction",
                    span=Span.PRESENT,
                    negated=False,
                    evidence=text,
                )
            )
        return ExtractionResult(claims=claims, entity_mentions=mentions, span=Span.PRESENT)


class _DangerousLLM(LLMAdapter):
    def __init__(self) -> None:
        self._n = 0

    async def generate(self, messages: list[dict[str, str]], **kwargs) -> AdapterResponse:
        self._n += 1
        if self._n == 1:
            return AdapterResponse(
                text="Noted. No ECG or troponin results available.",
                model="start-here-mock",
            )
        return AdapterResponse(
            text="Likely myocardial infarction. Recommend immediate catheterization.",
            model="start-here-mock",
        )


def _audit_action(row: dict) -> str | None:
    op = row.get("op")
    if op:
        return str(op)
    data = row.get("payload", {}).get("data", {})
    if isinstance(data, dict):
        action = data.get("action")
        if action:
            return str(action)
    return None


async def _run_scenarios(bridge: CanonicalScannerGateBridge) -> list[tuple[str, str, InterventionAction]]:
    results: list[tuple[str, str, InterventionAction]] = []

    pass_lens = Lens(
        LensConfig(
            adapter=_MockAdapter("Alice works at Acme Corp."),
            extraction_backend=_PassBackend(),
            governance_bridge=bridge,
            auto_verify=True,
        )
    )
    await pass_lens.process("Alice is a software engineer at Acme Corp.")
    pass_result = await pass_lens.process("Does Alice work at Acme Corp?")
    results.append(
        (
            "Scenario 1: allowed (factual query, clean response)",
            "PASS",
            pass_result.action,
        )
    )

    pef = PEFState()
    pef.get_or_create_entity("Emma")
    pef.get_or_create_entity("Anna")
    contain_lens = Lens(
        LensConfig(
            adapter=_MockAdapter(),
            extraction_backend=_AmbiguousBackend(),
            governance_bridge=bridge,
            auto_verify=True,
        ),
        initial_pef=pef,
    )
    contain_result = await contain_lens.process(
        "Emma told Anna her sister was overseas. Whose sister was overseas?"
    )
    results.append(
        (
            "Scenario 2: unclear (ambiguous referent - who is 'her'?)",
            "CONTAIN",
            contain_result.action,
        )
    )

    stop_lens = Lens(
        LensConfig(
            adapter=_DangerousLLM(),
            extraction_backend=_MedicalBackend(),
            governance_bridge=bridge,
            auto_verify=True,
        )
    )
    await stop_lens.process(
        "Patient with mild chest discomfort, no ECG, no troponin results."
    )
    stop_result = await stop_lens.process("Diagnose.")
    results.append(
        (
            "Scenario 3: unsafe / inadmissible (diagnosis without evidence)",
            "HARD_STOP",
            stop_result.action,
        )
    )

    return results


def _print_results(
    scenario_rows: list[tuple[str, str, InterventionAction]],
    *,
    audit_path: Path,
    ledger_ok: bool,
) -> int:
    width = 72
    print("=" * width)
    print("  Aurora-Lens start-here demo")
    print("=" * width)
    print()
    print("Three turns. One audit file. No provider API key. spaCy is required.")
    print()

    failed = 0
    for label, expected, actual in scenario_rows:
        ok = actual.name == expected
        if not ok:
            failed += 1
        mark = "OK" if ok else f"UNEXPECTED ({actual.name})"
        print(label)
        print(f"  Decision: {actual.name}  [{mark}]")
        print()

    rows = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").strip().splitlines()
        if line.strip()
    ]
    actions = [_audit_action(row) for row in rows]
    print(f"Audit written to: {audit_path.relative_to(REPO_ROOT)}")
    print(f"  lines:   {len(rows)}")
    print(f"  actions: {', '.join(a for a in actions if a)}")
    print()
    print("Inspect one line (pretty-print the last row):")
    rel = audit_path.relative_to(REPO_ROOT)
    print(f"  python -c \"import json; print(json.dumps(json.loads(open('{rel.as_posix()}').readlines()[-1]), indent=2)[:2000])\"")
    print()
    if ledger_ok:
        print("Verify ledger integrity (HMAC + hash chain):")
        print("  python tools/run_demo.py   # re-run writes a fresh log; or:")
        print("  python -c \"")
        print("from pathlib import Path")
        print("from aurora_lens.govern.canonical_bridge import CanonicalScannerGateBridge")
        print(f"bridge = CanonicalScannerGateBridge(audit_path=r'{rel.as_posix()}', secret_key={DEMO_SIGNING_KEY!r})")
        print("print('OK' if bridge.verify_ledger() else 'FAIL')\"")
    else:
        print("Verify ledger: FAILED — see stderr above.")
        failed += 1

    print()
    print("Next: read README.md")
    print("=" * width)
    return 1 if failed else 0


async def main() -> int:
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if AUDIT_PATH.exists():
        AUDIT_PATH.unlink()

    bridge = CanonicalScannerGateBridge(
        mode="public",
        audit_path=str(AUDIT_PATH),
        secret_key=DEMO_SIGNING_KEY,
    )
    scenario_rows = await _run_scenarios(bridge)
    ledger_ok = bridge.verify_ledger()
    return _print_results(scenario_rows, audit_path=AUDIT_PATH, ledger_ok=ledger_ok)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

"""Live verification harness for strict-policy PASS admissibility basis.

Prints policy_profile beside every PASS summary. Under strict policy, every PASS
must carry admissibility_basis and pass_reason_code; moderate may PASS with both
unset (legacy empty-flag path).

Usage:
    python scripts/verify_strict_admissibility_basis.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aurora_lens.adapters.base import LLMAdapter, AdapterResponse
from aurora_lens.context import domain_var
from aurora_lens.govern.bridge import BuiltinBridge
from aurora_lens.govern.decision import InterventionAction
from aurora_lens.govern.policy import DEFAULT_MODERATE, DEFAULT_STRICT
from aurora_lens.interpret.base import ExtractionBackend
from aurora_lens.interpret.schema import ExtractionResult
from aurora_lens.lens import Lens, LensConfig
from aurora_lens.pef.state import PEFState


class _MockAdapter(LLMAdapter):
    async def generate(self, messages: list[dict[str, str]], **kwargs) -> AdapterResponse:
        return AdapterResponse(text="Model output.", model="mock")


class _MockBackend(ExtractionBackend):
    async def extract(self, text: str, pef: PEFState) -> ExtractionResult:
        return ExtractionResult(claims=[], entity_mentions=[])


@dataclass(frozen=True)
class _Scenario:
    label: str
    prompt: str
    policy_name: str
    request_domain: str = "general"


def _policy_by_name(name: str):
    if name == "moderate":
        return DEFAULT_MODERATE
    return DEFAULT_STRICT


def format_pass_summary(
    *,
    prompt: str,
    policy_profile: str,
    admissibility_basis: str | None,
    pass_reason_code: str | None,
) -> str:
    """Human-readable PASS line with explicit moderate-only null-basis labeling."""
    if admissibility_basis is None and policy_profile == "moderate":
        basis_label = (
            "admissibility_basis=None "
            "(allowed - moderate policy only; strict requires explicit basis)"
        )
    elif admissibility_basis is None:
        basis_label = "admissibility_basis=None (VIOLATION - strict policy)"
    else:
        basis_label = f"admissibility_basis={admissibility_basis}"

    if pass_reason_code is None and policy_profile == "moderate":
        reason_label = (
            "pass_reason_code=None "
            "(allowed - moderate policy only; strict requires explicit code)"
        )
    elif pass_reason_code is None:
        reason_label = "pass_reason_code=None (VIOLATION - strict policy)"
    else:
        reason_label = f"pass_reason_code={pass_reason_code}"

    short_prompt = prompt if len(prompt) <= 72 else prompt[:69] + "..."
    return (
        f"PASS | policy_profile={policy_profile} | {basis_label} | "
        f"{reason_label} | prompt={short_prompt!r}"
    )


def assert_strict_pass_audit_invariants(audit_row: dict) -> None:
    """Strict policy: PASS audit rows must never omit admissibility fields."""
    if audit_row.get("outcome") != "PASS":
        return
    policy_profile = str(audit_row.get("policy_profile") or "")
    if policy_profile != "strict":
        return
    if audit_row.get("admissibility_basis") is None:
        raise AssertionError(
            "strict PASS audit row has admissibility_basis=None: "
            f"{json.dumps(audit_row, default=str)}"
        )
    if audit_row.get("pass_reason_code") is None:
        raise AssertionError(
            "strict PASS audit row has pass_reason_code=None: "
            f"{json.dumps(audit_row, default=str)}"
        )


async def _run_scenario(scenario: _Scenario) -> dict:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        audit_path = f.name
    policy = _policy_by_name(scenario.policy_name)
    bridge = BuiltinBridge(policy=policy, audit_path=audit_path)
    lens = Lens(
        LensConfig(
            adapter=_MockAdapter(),
            extraction_backend=_MockBackend(),
            governance_bridge=bridge,
        )
    )
    token = domain_var.set(scenario.request_domain)
    try:
        result = await lens.process(scenario.prompt)
    finally:
        domain_var.reset(token)

    rows = [
        json.loads(line)
        for line in Path(audit_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    Path(audit_path).unlink(missing_ok=True)
    audit = next((r for r in reversed(rows) if r.get("outcome")), {})
    return {
        "scenario": scenario,
        "result": result,
        "audit": audit,
    }


async def main() -> int:
    scenarios = [
        _Scenario("strict greeting", "Hello", "strict"),
        _Scenario(
            "strict consequential legal",
            "Walk me through how to prepare a binding legal settlement letter "
            "for my tenant dispute",
            "strict",
        ),
        _Scenario(
            "strict consequential finance domain",
            "Help me outline severance renegotiation steps with my employer",
            "strict",
            request_domain="finance",
        ),
        _Scenario(
            "moderate consequential legal (legacy empty-flag PASS)",
            "Walk me through how to prepare a binding legal settlement letter "
            "for my tenant dispute",
            "moderate",
        ),
    ]

    print("=== Strict admissibility live verification ===")
    failures = 0

    for scenario in scenarios:
        ran = await _run_scenario(scenario)
        result = ran["result"]
        audit = ran["audit"]
        action = result.action.name
        policy_profile = str(audit.get("policy_profile") or scenario.policy_name)

        print(f"\n[{scenario.label}]")
        print(f"  action={action} | policy_profile={policy_profile}")

        if action == InterventionAction.PASS.name:
            basis = audit.get("admissibility_basis")
            reason = audit.get("pass_reason_code")
            print(f"  {format_pass_summary(prompt=scenario.prompt, policy_profile=policy_profile, admissibility_basis=basis, pass_reason_code=reason)}")
            try:
                assert_strict_pass_audit_invariants(audit)
            except AssertionError as exc:
                print(f"  ASSERTION FAILED: {exc}")
                failures += 1
        else:
            print(f"  flags={[f.flag_type.name for f in result.flags] or '[]'}")

    print("\n=== Summary ===")
    if failures:
        print(f"FAILED: {failures} strict-policy PASS invariant violation(s)")
        return 1
    print("OK: all strict-policy PASS rows have admissibility_basis and pass_reason_code")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

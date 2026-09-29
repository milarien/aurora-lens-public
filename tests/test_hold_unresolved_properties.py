"""Property-style invariant tests for HOLD_UNRESOLVED + candidate-specific consequence.

Generates governed prompt variations (not nonsense fuzzing) and asserts core
governance guarantees hold across the set.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aurora_lens.govern.decision import InterventionAction
from aurora_lens.pef.unresolved_referents import HOLD_UNRESOLVED_CHOICE_LABEL
from tests.test_hold_unresolved import _lens_with_turn1_ambiguity

# Verified under hold: pre-LLM block, Decision blocked copy, blocked_before_generation.
_BLOCKING_BASES: tuple[str, ...] = (
    "Should the contractor be suspended?",
    "Should we suspend the contractor?",
    "Should the operator be disciplined?",
    "Should their certificate be revoked?",
    "Can we remove the contractor from the job?",
    "Do we terminate the contractor?",
)

# Unrelated remove/terminate surfaces — must not trigger candidate-specific hold blocks.
_NON_CANDIDATE_REMOVE_TERMINATE_PROMPTS: tuple[str, ...] = (
    "Can we remove the old scaffolding from the site?",
    "Do we terminate this draft record?",
    "Can we remove duplicate whitespace?",
    "Should we terminate the test run?",
)

# Stable surface transforms only — casing, whitespace, punctuation.
_SURFACE_TRANSFORMS: tuple[Callable[[str], str], ...] = (
    lambda s: s,
    lambda s: s.upper(),
    lambda s: s.lower(),
    lambda s: f"  {s}  ",
    lambda s: s.rstrip("?.") + "?",
    lambda s: s.replace(" ", "  ", 1),
    lambda s: s.replace("?", "").strip() + " ?",
)


def _apply_surface_variation(base: str, transform_index: int) -> str:
    return _SURFACE_TRANSFORMS[transform_index % len(_SURFACE_TRANSFORMS)](base)


@st.composite
def blocking_candidate_consequence_prompt(draw: st.DrawFn) -> str:
    base = draw(st.sampled_from(_BLOCKING_BASES))
    transform_index = draw(st.integers(min_value=0, max_value=len(_SURFACE_TRANSFORMS) - 1))
    return _apply_surface_variation(base, transform_index)


def _is_candidate_specific_hold_consequence_block(result) -> bool:
    """True when the turn received the same pre-LLM block bundle as positive hold cases."""
    if result.action == InterventionAction.PASS:
        return False
    response_lower = (result.response or "").lower()
    if "decision blocked" not in response_lower:
        return False
    decision = getattr(result, "decision", None)
    rationale = (getattr(decision, "rationale", None) or "")
    pathway = getattr(decision, "pathway_id", None)
    if "UNRESOLVED_REFERENT" in rationale:
        return True
    if pathway in {"P_ASK_DISAMBIGUATE", "P_STOP_ESCALATE"}:
        return True
    if "epistemic uncertainty" in rationale.lower():
        return True
    return True


async def _assert_hold_unresolved_does_not_overblock_remove_terminate(prompt: str) -> None:
    lens, adapter = await _lens_with_turn1_ambiguity()
    await lens.process(HOLD_UNRESOLVED_CHOICE_LABEL)
    calls_before = adapter._call_count

    result = await lens.process(prompt)

    response_lower = (result.response or "").lower()
    assert "decision blocked" not in response_lower, (
        f"unrelated remove/terminate must not trigger consequence block; "
        f"prompt={prompt!r} response={result.response!r}"
    )
    assert not _is_candidate_specific_hold_consequence_block(result), (
        f"unrelated remove/terminate must not use candidate-specific hold block outcome; "
        f"prompt={prompt!r} action={result.action!r}"
    )


async def _assert_hold_unresolved_candidate_consequence_invariants(prompt: str) -> None:
    lens, adapter = await _lens_with_turn1_ambiguity()
    await lens.process(HOLD_UNRESOLVED_CHOICE_LABEL)
    calls_before = adapter._call_count

    result = await lens.process(prompt)

    assert result.action != InterventionAction.PASS, (
        f"expected non-PASS for candidate-specific consequence under hold; "
        f"prompt={prompt!r} action={result.action!r}"
    )
    assert adapter._call_count == calls_before, (
        f"adapter must not be called for pre-LLM block; prompt={prompt!r}"
    )
    response_lower = (result.response or "").lower()
    assert "decision blocked" in response_lower, (
        f"missing operator block language; prompt={prompt!r} response={result.response!r}"
    )
    release_path = getattr(result, "telemetry_release_path", None)
    if release_path is not None:
        assert release_path == "blocked_before_generation", (
            f"unexpected telemetry_release_path={release_path!r}; prompt={prompt!r}"
        )


@given(prompt=blocking_candidate_consequence_prompt())
@settings(max_examples=24, deadline=None)
def test_hold_unresolved_candidate_consequence_invariants(prompt: str) -> None:
    """Under active hold, verified blocking prompts stay pre-LLM blocked across variations."""
    asyncio.run(_assert_hold_unresolved_candidate_consequence_invariants(prompt))


@pytest.mark.parametrize("base", _BLOCKING_BASES)
@pytest.mark.asyncio
async def test_hold_unresolved_each_blocking_base_prompt(base: str) -> None:
    """Explicit examples: every blocking base satisfies invariants."""
    await _assert_hold_unresolved_candidate_consequence_invariants(base)


@pytest.mark.parametrize("prompt", _NON_CANDIDATE_REMOVE_TERMINATE_PROMPTS)
@pytest.mark.asyncio
async def test_hold_unresolved_unrelated_remove_terminate_not_overblocked(prompt: str) -> None:
    """Unrelated remove/terminate prompts must not trigger candidate-specific hold blocks."""
    await _assert_hold_unresolved_does_not_overblock_remove_terminate(prompt)

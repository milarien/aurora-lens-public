from aurora_lens.state_native_engine.closed_world_contract import (
    ClosedWorldResolution,
    classify_closed_world_resolution,
)
from aurora_lens.state_native_engine.contracts import StateNativeOutcome, StateNativeSolverFamily


def test_closed_world_contract_noop_for_non_closed_world_results():
    res = classify_closed_world_resolution(
        outcome=StateNativeOutcome.ANSWER,
        stop_reason_code=None,
        solver_family=StateNativeSolverFamily.COMMITTED_LOCATION_READ,
    )
    assert res is None


def test_closed_world_contract_maps_answer_to_solved_unique():
    res = classify_closed_world_resolution(
        outcome=StateNativeOutcome.ANSWER,
        stop_reason_code=None,
        solver_family=StateNativeSolverFamily.CLOSED_WORLD,
    )
    assert res == ClosedWorldResolution.SOLVED_UNIQUE


def test_closed_world_contract_maps_clarify_to_multiple_solutions():
    res = classify_closed_world_resolution(
        outcome=StateNativeOutcome.CLARIFY,
        stop_reason_code="closed_world_ambiguous_solution",
        solver_family=StateNativeSolverFamily.CLOSED_WORLD,
    )
    assert res == ClosedWorldResolution.MULTIPLE_SOLUTIONS


def test_closed_world_contract_maps_stop_no_consistent_solution_codes():
    res = classify_closed_world_resolution(
        outcome=StateNativeOutcome.STOP,
        stop_reason_code="state_native_no_holder",
        solver_family=StateNativeSolverFamily.CLOSED_WORLD,
    )
    assert res == ClosedWorldResolution.NO_CONSISTENT_SOLUTION


def test_closed_world_contract_maps_other_stop_to_needs_formalisation():
    res = classify_closed_world_resolution(
        outcome=StateNativeOutcome.STOP,
        stop_reason_code="state_native_unknown_entity",
        solver_family=StateNativeSolverFamily.CLOSED_WORLD,
    )
    assert res == ClosedWorldResolution.NEEDS_FORMALISATION

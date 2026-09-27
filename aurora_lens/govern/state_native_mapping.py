"""Lens-side mapping from state_native_engine outcomes to GovernanceDecision.

The state_native_engine must not import this module.
"""

from __future__ import annotations

from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
from aurora_lens.state_native_engine.contracts import (
    StateNativeDelegationResult,
    StateNativeOutcome,
)
from aurora_lens.state_native_engine.epistemic import EpistemicResult
from aurora_lens.verify.flags import Flag, FlagType

# Committed-state epistemic insufficiency — reopenable HARD_STOP (not terminal lockout).
# Aligns with ``closed_world_contract._STOP_REASON_NO_CONSISTENT`` plus entity/compare gaps.
_STATE_NATIVE_EPISTEMIC_INSUFFICIENCY_STOP_REASONS: frozenset[str] = frozenset(
    {
        "state_native_no_at",
        "state_native_at_incomplete",
        "state_native_no_inventory",
        "state_native_no_holder",
        "state_native_unknown_entity",
        "state_native_no_compare",
    }
)


def is_state_native_epistemic_insufficiency_stop(sn: StateNativeDelegationResult) -> bool:
    """True when STOP reflects missing/insufficient committed state, not terminal prohibition."""
    if sn.epistemic_result == EpistemicResult.UNKNOWN:
        return True
    code = (sn.stop_reason_code or "").strip().lower()
    return code in _STATE_NATIVE_EPISTEMIC_INSUFFICIENCY_STOP_REASONS


def reconcile_state_native_stop_after_policy_projection(
    decision: GovernanceDecision,
    sn: StateNativeDelegationResult,
) -> None:
    """Restore reopenable STOP posture after canonical policy projection overwrites mapping.

    Lens projects ``general:GP:STOP`` (terminal) for all state-native STOP turns unless a
    flag-class matrix row exists for ``stop_reason_code``. Epistemic insufficiency must stay
    interaction-open so lawful continuation (evidence, reframe, meta) remains admissible.
    """
    if decision.action != InterventionAction.HARD_STOP:
        return
    if not is_state_native_epistemic_insufficiency_stop(sn):
        return
    decision.pathway_id = "P_STOP_REFUSE_CLEAN"
    decision.output_mode = "terminal_stop"
    decision.commitment_closed = True
    decision.interaction_open = True


def _committed_state_stop_flag(sn: StateNativeDelegationResult) -> Flag:
    return Flag(
        flag_type=FlagType.STATE_NATIVE_COMMITTED_STATE_STOP,
        entity_name="",
        claim="State-native stop: insufficient committed state to answer query",
        evidence=sn.stop_reason_code or "state_native_stop",
        severity="warning",
    )


def _build_state_native_hard_stop_decision(
    sn: StateNativeDelegationResult,
    *,
    stop_flag: Flag,
) -> GovernanceDecision:
    text = sn.user_visible_text
    reopenable = is_state_native_epistemic_insufficiency_stop(sn)
    return GovernanceDecision(
        action=InterventionAction.HARD_STOP,
        flags=[stop_flag],
        rationale=(
            f"state_native_stop:{sn.stop_reason_code or 'unknown'}"
            f"{_epistemic_suffix(sn)}"
        ),
        policy="strict",
        pathway_id="P_STOP_REFUSE_CLEAN" if reopenable else "P_STOP_TERMINAL",
        output_mode="terminal_stop",
        commitment_closed=True,
        interaction_open=reopenable,
        original_response="",
        corrected_response=text,
        governed_response=text,
    )


def _clarify_structural_flag(sn: StateNativeDelegationResult) -> Flag:
    """Single structural flag for audit/forensic ``failed_constraints`` (user text unchanged)."""
    ctx = sn.clarify_context or {}
    raw_name = (ctx.get("failed_constraint") or "STATE_NATIVE_ENTITY_AMBIGUITY")
    if not isinstance(raw_name, str) or not raw_name.strip():
        name = "STATE_NATIVE_ENTITY_AMBIGUITY"
    else:
        name = raw_name.strip()
    try:
        ft = FlagType[name]
    except KeyError:
        ft = FlagType.STATE_NATIVE_ENTITY_AMBIGUITY
    cands_raw = ctx.get("candidate_entities")
    if isinstance(cands_raw, list):
        candidates: tuple[str, ...] = tuple(str(x) for x in cands_raw)
    else:
        candidates = ()
    subj = ctx.get("subject_phrase")
    if isinstance(subj, str) and subj.strip():
        ent_name = subj.strip()[:200]
    else:
        item_h = ctx.get("item_phrase")
        ent_name = item_h.strip()[:200] if isinstance(item_h, str) and item_h.strip() else "entity"
    ev = (
        ", ".join(candidates)
        if candidates
        else "Multiple committed entity candidates; disambiguation required"
    )
    tr = ctx.get("typo_recovery")
    if isinstance(tr, dict):
        ts = tr.get("typo_surface")
        cn = tr.get("suggested_canonical_name")
        if isinstance(ts, str) and ts.strip() and isinstance(cn, str) and cn.strip():
            ev = f"Typo suggestion ({ts}->{cn}); {ev}"
    return Flag(
        flag_type=ft,
        entity_name=ent_name,
        claim="State-native committed-state query: entity head is structurally ambiguous",
        evidence=ev,
        severity="warning",
        candidates=candidates,
    )


def _epistemic_suffix(sn: StateNativeDelegationResult) -> str:
    """Return ':epistemic_state' suffix for rationale strings, or '' when absent."""
    if sn.epistemic_result is not None:
        return f":{sn.epistemic_result.value}"
    return ""


def governance_decision_from_state_native(
    sn: StateNativeDelegationResult,
) -> GovernanceDecision:
    """Deterministic mapping — no bridge.decide, no model text.

    Epistemic adjudication takes precedence over the ternary outcome field:
      TRUE / FALSE / VALUE  → ANSWER  → PASS
      UNKNOWN               → STOP    → HARD_STOP
      AMBIGUOUS             → CLARIFY → CONTAIN
      CONTRADICTED          → FORCE_REVISE
    """
    if not sn.handled or sn.outcome is None:
        raise ValueError("governance_decision_from_state_native requires handled result")
    text = sn.user_visible_text
    ep = sn.epistemic_result

    # CONTRADICTED overrides the outcome field — escalate to FORCE_REVISE.
    if ep == EpistemicResult.CONTRADICTED:
        contra_flag = Flag(
            flag_type=FlagType.CONTRADICTED_FACT,
            entity_name="",
            claim="State-native epistemic adjudication: response contradicts committed PEF state",
            evidence=sn.stop_reason_code or "state_native_contradicted",
            severity="warning",
        )
        return GovernanceDecision(
            action=InterventionAction.FORCE_REVISE,
            flags=[contra_flag],
            rationale=f"state_native_contradicted{_epistemic_suffix(sn)}",
            policy="strict",
            pathway_id="P_REFUSE_EXPLAIN_REDIRECT",
            output_mode="force_revise",
            commitment_closed=False,
            interaction_open=True,
            original_response="",
            corrected_response=text,
            governed_response=text,
        )

    # Epistemic UNKNOWN beats ANSWER shape: "not established" must not route as PASS.
    if ep == EpistemicResult.UNKNOWN:
        return _build_state_native_hard_stop_decision(
            sn,
            stop_flag=_committed_state_stop_flag(sn),
        )

    if sn.outcome == StateNativeOutcome.ANSWER:
        return GovernanceDecision(
            action=InterventionAction.PASS,
            flags=[],
            rationale=f"state_native_answer{_epistemic_suffix(sn)}",
            policy="strict",
            pathway_id=None,
            output_mode=None,
            commitment_closed=False,
            interaction_open=True,
            original_response="",
            corrected_response=text,
            governed_response=text,
        )
    if sn.outcome == StateNativeOutcome.CLARIFY:
        return GovernanceDecision(
            action=InterventionAction.CONTAIN,
            flags=[_clarify_structural_flag(sn)],
            rationale=f"state_native_clarify{_epistemic_suffix(sn)}",
            policy="strict",
            pathway_id="P_ASK_DISAMBIGUATE",
            output_mode="clarification_continuation",
            commitment_closed=False,
            interaction_open=True,
            original_response="",
            corrected_response=text,
            governed_response=text,
        )
    if sn.outcome == StateNativeOutcome.STOP:
        return _build_state_native_hard_stop_decision(
            sn,
            stop_flag=_committed_state_stop_flag(sn),
        )
    raise ValueError(f"unsupported state native outcome: {sn.outcome!r}")

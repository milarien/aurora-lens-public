"""Operator / forensics plane: structured PEF session view (no user-plane leakage).

User-plane clients must not receive these structures; the proxy only attaches
``operator_pef`` when ``include_operator_detail=True``.

``operator_pef`` describes **what is currently in force** on the session (present
state). Historical outcomes belong in the audit / forensic ledger, not here.
"""

from __future__ import annotations

from typing import Any

from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
from aurora_lens.pef.state import (
    EPISTEMIC_MODE_AMBIGUITY,
    EPISTEMIC_MODE_REFUSAL,
    EPISTEMIC_MODE_STOP,
    PEFState,
)
from aurora_lens.pef.unresolved_referents import clarification_choices_from_pending


def session_mode_label(pef: PEFState) -> str:
    """High-level session mode for operator dashboards."""
    hold = pef.epistemic_hold
    pending = pef.pending_clarification
    if hold:
        m = hold.get("mode")
        if m == EPISTEMIC_MODE_AMBIGUITY:
            return "held_ambiguity"
        if m == EPISTEMIC_MODE_REFUSAL:
            return "held_refusal"
        if m == EPISTEMIC_MODE_STOP:
            return "terminal_stop_open" if hold.get("interaction_open") else "terminal_stop_closed"
    if pending:
        return "held_ambiguity"
    return "normal"


def _hold_pathway_matches_decision(hold: dict[str, Any], decision: GovernanceDecision) -> bool:
    if decision.pathway_id is None:
        return True
    return hold.get("pathway_id") == decision.pathway_id


def binding_governance_decision_is_live(decision: GovernanceDecision, pef: PEFState) -> bool:
    """True when this GovernanceDecision still shapes ``binding_governance_state`` on the session *now*.

    Audit rows record what happened; ``operator_pef.binding_governance_state`` is present-state
    only (holds, pending clarification, or the current turn's soft correction).
    """
    if decision.action == InterventionAction.PASS:
        return False
    if decision.action == InterventionAction.SOFT_CORRECT:
        return True
    if decision.action == InterventionAction.CONTAIN:
        if pef.pending_clarification:
            return True
        h = pef.epistemic_hold
        return bool(h and h.get("mode") == EPISTEMIC_MODE_AMBIGUITY)
    if decision.action == InterventionAction.FORCE_REVISE:
        h = pef.epistemic_hold
        if not h or h.get("mode") != EPISTEMIC_MODE_REFUSAL:
            return False
        return _hold_pathway_matches_decision(h, decision)
    if decision.action == InterventionAction.HARD_STOP:
        h = pef.epistemic_hold
        if not h or h.get("mode") != EPISTEMIC_MODE_STOP:
            return False
        return _hold_pathway_matches_decision(h, decision)
    return False


def _payload_from_decision(decision: GovernanceDecision) -> dict[str, Any]:
    return {
        "action": decision.action.name,
        "pathway_id": decision.pathway_id,
        "interaction_open": decision.interaction_open,
        "commitment_closed": decision.commitment_closed,
    }


def _binding_governance_state_from_durable_pef(pef: PEFState) -> dict[str, Any] | None:
    """Derive ``binding_governance_state`` from durable PEF (hold / pending) when no live Lens decision."""
    hold = pef.epistemic_hold
    if hold:
        mode = hold.get("mode")
        action_name = {
            EPISTEMIC_MODE_AMBIGUITY: InterventionAction.CONTAIN.name,
            EPISTEMIC_MODE_REFUSAL: InterventionAction.FORCE_REVISE.name,
            EPISTEMIC_MODE_STOP: InterventionAction.HARD_STOP.name,
        }.get(mode)
        if not action_name:
            return None
        return {
            "action": action_name,
            "pathway_id": hold.get("pathway_id"),
            "interaction_open": hold.get("interaction_open"),
            "commitment_closed": hold.get("commitment_closed"),
        }
    if pef.pending_clarification:
        return {
            "action": InterventionAction.CONTAIN.name,
            "pathway_id": None,
            "interaction_open": True,
            "commitment_closed": False,
        }
    return None


def build_operator_pef_surface(
    pef: PEFState,
    decision: GovernanceDecision | None,
) -> dict[str, Any]:
    """Summarize current session posture for operators (present state, not history)."""
    pending = pef.pending_clarification
    hold = pef.epistemic_hold

    out: dict[str, Any] = {
        "session_mode": session_mode_label(pef),
        "current_turn": pef.current_turn,
        "counts": {
            "entities": len(pef.entities),
            "relationships": len(pef.relationships),
        },
    }

    if pending:
        oq = pending.get("original_question") or ""
        out["pending_clarification"] = {
            "active": True,
            "failed_constraint": pending.get("failed_constraint"),
            "original_question": pending.get("original_question"),
            "ambiguous_referents": pending.get("ambiguous_referents"),
            "candidate_entities": pending.get("candidate_entities"),
            "blocked_claims": pending.get("blocked_claims"),
            "blocked_proposition": pending.get("blocked_proposition"),
            "clarification_prompt": pending.get("clarification_prompt"),
            "clarification_choices": clarification_choices_from_pending(pending, pef=pef),
            "resolution_mode": pending.get("resolution_mode"),
            "original_question_preview": oq[:240] + ("…" if len(oq) > 240 else ""),
        }
    else:
        out["pending_clarification"] = {"active": False}

    if hold:
        out["epistemic_hold"] = {
            "active": True,
            "mode": hold.get("mode"),
            "since_turn": hold.get("since_turn"),
            "pathway_id": hold.get("pathway_id"),
            "interaction_open": hold.get("interaction_open"),
            "commitment_closed": hold.get("commitment_closed"),
            "last_audit_id": hold.get("last_audit_id"),
        }
    else:
        out["epistemic_hold"] = {"active": False}

    bgs: dict[str, Any] | None = None
    if decision is not None and binding_governance_decision_is_live(decision, pef):
        bgs = _payload_from_decision(decision)
    if bgs is None:
        bgs = _binding_governance_state_from_durable_pef(pef)
    if bgs is not None:
        out["binding_governance_state"] = bgs

    return out

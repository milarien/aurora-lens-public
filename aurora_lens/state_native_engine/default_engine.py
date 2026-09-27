"""Default state-native engine — strict bounded queries over committed PEF."""

from __future__ import annotations

from aurora_lens.interpret.turn_act import TurnAct
from aurora_lens.state_native_engine.contracts import (
    StateNativeDelegationResult,
    StateNativeOutcome,
    StateNativeRequest,
    StateNativeSolverFamily,
)
from aurora_lens.state_native_engine.epistemic import EpistemicResult
from aurora_lens.state_native_engine.eval.existence import (
    evaluate_actor_value_query,
    evaluate_existence_query,
)
from aurora_lens.state_native_engine.eval.location import (
    evaluate_location_from_committed_state,
    evaluate_still_in_from_committed_state,
)
from aurora_lens.state_native_engine.eval.inventory import (
    evaluate_inventory_how_many_query,
    evaluate_inventory_subject_has_item_query,
    evaluate_inventory_for_subject,
    evaluate_inventory_holder_query,
)
from aurora_lens.state_native_engine.eval.action_agent import evaluate_action_agent
from aurora_lens.state_native_engine.eval.temporal_contact import (
    delegation_result_for_temporal_contact_query,
)
from aurora_lens.state_native_engine.eval.compare import evaluate_comparative_query
from aurora_lens.state_native_engine.eval.typed_transition import (
    evaluate_typed_transition_request,
)
from aurora_lens.state_native_engine.eval.possession_mutations import (
    evaluate_possession_consume_mutation,
    evaluate_possession_transfer_mutations,
)
from aurora_lens.state_native_engine.parse.query_surface import (
    parse_did_actor_action_query,
    parse_inventory_how_many_phrase,
    parse_inventory_how_many_past_phrase,
    parse_inventory_subject_has_item_phrase,
    parse_inventory_object_holder_phrase,
    parse_inventory_quantity_holder_phrase,
    parse_inventory_subject_have_phrase,
    parse_is_still_in_locative,
    parse_location_subject_phrase,
    parse_follow_up_attribution_query,
    parse_temporal_contact_query,
    parse_comparative_question,
    parse_who_did_action_query,
)
from aurora_lens.state_native_engine.protocol import StateNativeEngine


def _infer_epistemic(outcome: StateNativeOutcome) -> EpistemicResult:
    """Infer epistemic classification when explicit tagging is omitted.

    **Do not** rely on this for Boolean evaluators: ANSWER would incorrectly
    become VALUE. Boolean paths must set ``epistemic_result`` on
    :class:`StateNativeDelegationResult` (TRUE / FALSE).
    """
    if outcome == StateNativeOutcome.CLARIFY:
        return EpistemicResult.AMBIGUOUS
    if outcome == StateNativeOutcome.STOP:
        return EpistemicResult.UNKNOWN
    return EpistemicResult.VALUE


def _build_result(
    out_s: str,
    text: str,
    clarify: dict | None,
    stop_code: str | None,
    family: StateNativeSolverFamily,
    epistemic: EpistemicResult | None = None,
    source_status: str | None = None,
) -> StateNativeDelegationResult:
    """Construct a handled result with inferred epistemic_result when not given."""
    outcome = StateNativeOutcome(out_s)
    if epistemic is None:
        epistemic = _infer_epistemic(outcome)
    return StateNativeDelegationResult(
        handled=True,
        outcome=outcome,
        user_visible_text=text,
        clarify_context=clarify,
        stop_reason_code=stop_code,
        solver_family=family,
        source_status=source_status,
        epistemic_result=epistemic,
    )


class DefaultStateNativeEngine:
    """Strict bounded queries over committed PEF."""

    def evaluate(self, req: StateNativeRequest) -> StateNativeDelegationResult:
        typed_transition = evaluate_typed_transition_request(req)
        if typed_transition is not None:
            return typed_transition

        # Handle possession mutations (transfer frame arbitration + CONSUME) before queries.
        transfer_mutation = evaluate_possession_transfer_mutations(req)
        if transfer_mutation is not None:
            return transfer_mutation

        consume_mutation = evaluate_possession_consume_mutation(req)
        if consume_mutation is not None:
            return consume_mutation

        if req.turn_act != TurnAct.QUERY:
            return StateNativeDelegationResult(handled=False)

        # ── Artifact frame guard (Phases 1–5) ────────────────────────────────
        # When an ARTIFACT frame is active (fiction/roleplay/hypothetical), state-native
        # must not answer queries from external-world committed PEF state. Returning
        # handled=False passes the query to the LLM, which can respond within the
        # established frame context.
        from aurora_lens.pef.artifact_layer import FrameKind
        active_frame = getattr(req.pef, "active_frame", None)
        if active_frame is not None and active_frame.kind == FrameKind.ARTIFACT:
            return StateNativeDelegationResult(handled=False)

        # ── Existence queries: "Did X [action]?" ──────────────────────────────
        # Must run before general "Who" routing to give it priority over the
        # follow-up attribution handler.
        did_parts = parse_did_actor_action_query(req.user_text)
        if did_parts is not None:
            actor, event = did_parts
            sn = evaluate_existence_query(req.pef, actor, event)
            if sn is not None:
                return sn
            # No matching action in PEF → fall through to LLM (not handled).
            return StateNativeDelegationResult(handled=False)

        # ── Actor-value queries: "Who [action]?" ──────────────────────────────
        who_event = parse_who_did_action_query(req.user_text)
        if who_event is not None:
            sn = evaluate_actor_value_query(req.pef, who_event)
            if sn is not None:
                return sn
            return StateNativeDelegationResult(handled=False)

        # ── Location queries ──────────────────────────────────────────────────
        phrase = parse_location_subject_phrase(req.user_text)
        if phrase is not None:
            return evaluate_location_from_committed_state(req.pef, phrase)

        still_in = parse_is_still_in_locative(req.user_text)
        if still_in is not None:
            subj_si, place_si = still_in
            return evaluate_still_in_from_committed_state(req.pef, subj_si, place_si)

        # ── Inventory queries ─────────────────────────────────────────────────
        subj_phrase = parse_inventory_subject_have_phrase(req.user_text)
        if subj_phrase is not None:
            return evaluate_inventory_for_subject(req.pef, subj_phrase)

        how_many = parse_inventory_how_many_phrase(req.user_text)
        if how_many is not None:
            subj_q, item_q = how_many
            out_s, text, clarify, stop_code = evaluate_inventory_how_many_query(
                req.pef, subj_q, item_q,
            )
            return _build_result(
                out_s, text, clarify, stop_code,
                StateNativeSolverFamily.COMMITTED_INVENTORY_READ,
            )

        how_many_past = parse_inventory_how_many_past_phrase(req.user_text)
        if how_many_past is not None:
            subj_q, item_q = how_many_past
            out_s, text, clarify, stop_code = evaluate_inventory_how_many_query(
                req.pef, subj_q, item_q, past_tense=True,
            )
            return _build_result(
                out_s, text, clarify, stop_code,
                StateNativeSolverFamily.COMMITTED_INVENTORY_READ,
            )

        subj_has_item = parse_inventory_subject_has_item_phrase(req.user_text)
        if subj_has_item is not None:
            subj_q, item_q = subj_has_item
            sn_has = evaluate_inventory_subject_has_item_query(req.pef, subj_q, item_q)
            if sn_has is not None:
                return sn_has

        obj_phrase = parse_inventory_object_holder_phrase(req.user_text)
        if obj_phrase is not None:
            qty_match = parse_inventory_quantity_holder_phrase(req.user_text)
            out_s, text, clarify, stop_code = evaluate_inventory_holder_query(
                req.pef, obj_phrase, quantity_match=qty_match,
            )
            return _build_result(
                out_s, text, clarify, stop_code,
                StateNativeSolverFamily.COMMITTED_INVENTORY_READ,
            )

        # ── Follow-up attribution ("who should I follow up with?") ───────────
        if parse_follow_up_attribution_query(req.user_text):
            out_s, text, clarify, stop_code = evaluate_action_agent(req.pef)
            if out_s is not None:
                return _build_result(
                    out_s, text, clarify, stop_code,
                    StateNativeSolverFamily.COMMITTED_ACTION_AGENT_READ,
                )

        # ── Temporal contact ──────────────────────────────────────────────────
        entity_phrase = parse_temporal_contact_query(req.user_text)
        if entity_phrase is not None:
            sn_tc = delegation_result_for_temporal_contact_query(req.pef, entity_phrase)
            if sn_tc is not None:
                return sn_tc

        # ── Comparative ───────────────────────────────────────────────────────
        compare_parts = parse_comparative_question(req.user_text)
        if compare_parts is not None:
            comp_noun, comp_adj = compare_parts
            return evaluate_comparative_query(req.pef, comp_noun, comp_adj)

        return StateNativeDelegationResult(handled=False)

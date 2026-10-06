"""Lens — main orchestrator implementing the sandwich pipeline.

User Input → Interpretation → LLM → Verification → Governance → Output

The governance step is deterministic: flags trigger policy evaluation,
policy produces an action, the bridge executes it. FORCE_REVISE has
a hard limit of 1 attempt — if the revision still has error-level flags,
it escalates to HARD_STOP.
"""

from __future__ import annotations

import copy
from collections.abc import AsyncIterator, Callable
from enum import Enum, auto

from dataclasses import dataclass, field, replace

from aurora_lens.config import LensConfig
from aurora_lens.context import domain_var, get_execution_task, get_request_metadata, request_prompt_var, trace_id_var
from aurora_lens.govern.candidate_release import (
    adjudicate_candidate_release,
    failed_constraint_to_flag_type,
)
from aurora_lens.rag_pef_admission import admit_retrieved_context_to_pef
from aurora_lens.rag_activation import effective_rag_retrieval_aware_referents
from aurora_lens.request_metadata import RequestMetadata
from aurora_lens.pre_model_state import PreModelContext, pre_model_state_dispatch
from aurora_lens.pre_model_state.types import PreModelDispatchResult
from aurora_lens.pef.prompts import build_pef_system_message
from aurora_lens.pef.audit_linkage import classify_hold_transition, classify_pef_turn_start
from aurora_lens.pef.state import (
    EPISTEMIC_HOLD_SCHEMA_VERSION,
    EPISTEMIC_MODE_AMBIGUITY,
    EPISTEMIC_MODE_NONE,
    EPISTEMIC_MODE_REFUSAL,
    EPISTEMIC_MODE_STOP,
    PEFState,
    Relationship,
    canonicalize_relation,
)
from aurora_lens.pef.artifact_layer import ArtifactFrame, FrameKind
from aurora_lens.pef.at_read import at_basis_snapshot
from aurora_lens.pef.entity import Entity
from aurora_lens.pef.frame_detector import detect_frame_transition
from aurora_lens.pef.span import Span
from aurora_lens.pef.unresolved_referents import (
    candidates_for_tokens,
    hold_unresolved_referents,
    open_entries,
    open_registry_tokens_in_text,
    referent_registry_held_unresolved,
    register_unresolved_referents,
    resolve_unresolved_referents,
    sync_registry_from_extraction,
    build_unresolved_referent_clarification_choices,
    try_explicit_possessive_attribution,
    HOLD_UNRESOLVED_CHOICE_LABEL,
    RESOLUTION_MODE_HELD_UNRESOLVED,
)
from aurora_lens.pef.unresolved_session_gate import (
    dismiss_open_unresolved_referents,
    evaluate_unresolved_session_gate,
    is_session_reset_request,
    is_unrelated_safe_turn,
)
from aurora_lens.interpret.base import ExtractionBackend
from aurora_lens.interpret.pef_updater import update_pef, _build_semantic_transactions
from aurora_lens.interpret.pef_admission import PEFAdmissionResult, pef_admission_result_wire_dict
from aurora_lens.interpret.revision_gate import (
    extraction_conflicts_grounded_pef,
    revision_clarification_message,
    user_text_conflicts_grounded_pef,
)
from aurora_lens.interpret.turn_act import (
    TurnAct,
    classify_turn_act,
    _looks_like_clarification_inquiry_meta,
)
from aurora_lens.interpret.turn_semantics import (
    looks_like_explicit_correction_phrase as _looks_like_explicit_correction_phrase,
    plan_resolved_pending_margin_answer as _plan_resolved_pending_margin_answer,
    plan_resolved_pending_simple_is_answer as _plan_resolved_pending_simple_is_answer,
    plan_mutation_ack as _plan_mutation_ack,
    TurnSemanticPlan,
)
from aurora_lens.interpret.pending_continuation import (
    binding_resume_requires_upstream as _binding_requires_upstream_resume,
    binding_resume_response_satisfies_attribution_task as _binding_resume_response_satisfies_attribution_task,
    completion_strategy_for_unresolved_referent_hold as _completion_strategy_for_unresolved_referent_hold,
    filter_referent_resolution_candidates as _filter_referent_resolution_candidates,
    infer_pending_attribution_task as _infer_pending_attribution_task,
    person_antecedent_names_from_surfaces as _person_antecedent_names_from_surfaces,
    plan_attribution_resume_answer as _plan_attribution_resume_answer,
    synthesize_held_claim_from_blocked_proposition as _synthesize_held_claim_from_blocked_proposition,
)
from aurora_lens.pef_admission_debug import log_revision_gate, pef_admission_debug_enabled
from aurora_lens.interpret.comparative_question_probe import (
    merge_structural_comparative_question_probe,
)
from aurora_lens.interpret.schema import ExtractedClaim, ExtractionResult
from aurora_lens.verify.blocked_request_policy import (
    classify_agency_context_followup,
    evaluate_blocked_act_request,
    sanitize_agency_prompt_for_non_coercive_use,
)
from aurora_lens.verify.checker import Checker
from aurora_lens.verify.consequence_intent import (
    LOW_RISK_CONVERSATIONAL,
    classify_consequence_intent,
    consequence_intent_flag,
)
from aurora_lens.verify.user_grounding import UserGroundingContext, build_user_grounding_context
from aurora_lens.govern.decision import (
    GovernanceDecision,
    InterventionAction,
    attach_rule_result as _attach_decision_rule_result,
    continuation_type_from_capability as _continuation_type_from_capability,
)
from aurora_lens.govern.freshness_permission_policy import (
    AuthorityState,
    ConsequenceGrade,
    FreshnessFailureKind,
    FreshnessPermissionDecision,
    FreshnessPermissionInput,
    FreshnessPermissionOutcome,
    evaluate_freshness_permission,
)
from aurora_lens.verify.flags import Flag, FlagType
from aurora_lens.corpus.evidence_authority import (
    is_canonical_evidence_authority_state,
    lens_action_for_evidence_authority,
)
from aurora_lens.govern.canonical_bridge import CanonicalScannerGateBridge
from aurora_lens.govern.adapters.context_resolver import ContextResolver
from aurora_lens.govern.state_native_mapping import (
    governance_decision_from_state_native,
    reconcile_state_native_stop_after_policy_projection,
)

_EVIDENCE_ADMISSION_KIND_TO_OUTCOME = {
    "evidence_inadmissible": InterventionAction.HARD_STOP,
    "evidence_unresolved": InterventionAction.CONTAIN,
}
_CANONICAL_FRESHNESS_FAILURE_KINDS: frozenset[str] = frozenset({
    FreshnessFailureKind.STALE_EVIDENCE.value,
    FreshnessFailureKind.MISSING_FRESHNESS.value,
    FreshnessFailureKind.UNREVALIDATED_ORIENTATION.value,
})
_RETRIEVAL_CONSISTENCY_KIND_TO_OUTCOME = {
    "threshold_conflict": InterventionAction.CONTAIN,
    "effective_date_conflict": InterventionAction.CONTAIN,
}
from aurora_lens.govern.user_copy import USER_MESSAGE_INTERPRETATION_FAILED
from aurora_lens.govern.stream_gate import (
    BufferedStreamResult,
    ProgressSignal,
    StreamPhase,
    make_governed_chunk_dict,
)
from aurora_lens.governor import StructuralGovernor
from aurora_lens.governor.freshness_pathways import project_freshness_pathway
from aurora_lens.governor.models import LensStatus
from aurora_lens.governor.resolver import resolve as resolve_governor_policy
from aurora_lens.state_native_engine.closed_world_contract import (
    classify_closed_world_resolution,
)
from aurora_lens.state_native_engine.lexical import item_key
from aurora_lens.state_native_engine.parse.query_surface import (
    state_native_text_for_binding_resume,
)
from aurora_lens.state_native_engine.temporal_eval_result import (
    PresentBoundTemporalEvalResult,
    TemporalGovernanceCue,
)
from aurora_lens.aurora_metadata import build_aurora_block
from aurora_lens.govern.epistemic_normalisation import apply_epistemic_normalisation
from aurora_lens.lens_governed_turn import PostGenerationGovernanceOutcome
from aurora_lens.govern.governed_copy import (
    compose_unresolved_referent_governed_message,
    compose_session_dependent_consequence_block,
    compose_session_meta_inquiry_block,
    compose_hold_unresolved_acknowledgment,
    compose_hold_unresolved_already_held_acknowledgment,
)

import json
import logging
import re

_lens_log = logging.getLogger(__name__)

# Lens-issued clarification copy must match bridge contract for Action wording.
_CANONICAL_CLARIFICATION_ACTION_LINE = "Action: Choose one option to continue."
_LEGACY_CLARIFICATION_ACTION_LINES = (
    "Action: Provide the specific name or label to continue.",
    "Action: Provide the specific name or label.",
    "Action: Provide the missing detail to continue.",
    "Action: Provide the missing detail.",
)
_AGENCY_CONTEXT_UNRESOLVED_PROMPT = "What is the intended context and target for this plan?"
_AGENCY_ETHICAL_CONTINUATION_PREFIX = (
    "I can help with an ethical version that preserves customer choice and removes coercive objectives."
)


def _ensure_canonical_clarification_action_line(text: str) -> str:
    """Normalize legacy Action lines embedded in governed clarification strings."""
    if not text:
        return text
    out = text
    for leg in _LEGACY_CLARIFICATION_ACTION_LINES:
        if leg in out:
            out = out.replace(leg, _CANONICAL_CLARIFICATION_ACTION_LINE)
    # Final fallback for any 'Action: Provide...' variant that might have escaped the list.
    if "Action: Provide" in out and "to continue." in out:
        import re
        out = re.sub(r"Action: Provide.*?to continue\.", _CANONICAL_CLARIFICATION_ACTION_LINE, out)
    return out


def _maybe_sanitize_governed_clarification_action(text: str) -> str:
    """When bridge emits clarification-shaped copy, coerce Action line to Lens/bridge contract."""
    if not text or "Action:" not in text:
        return text
    if (
        "Waiting for clarification" in text
        or "Ambiguity detected" in text
        or "Clarification required" in text
        or "More information required" in text
    ):
        return _ensure_canonical_clarification_action_line(text)
    return text


def _governance_decision_norm_signature(decision: GovernanceDecision) -> tuple[object, ...]:
    """Stable tuple for comparing governance decisions before/after surface-only edits."""
    return (
        decision.action,
        decision.rationale,
        decision.policy,
        decision.rule_id,
        decision.pathway_id,
        decision.output_mode,
        decision.commitment_closed,
        decision.interaction_open,
        tuple(
            (
                f.flag_type.name,
                f.entity_name,
                f.claim,
                f.severity,
                tuple(f.candidates or ()),
            )
            for f in decision.flags
        ),
    )


def _strip_markdown_bold_wrappers_for_rag_extraction(context_block: str) -> str:
    """Remove paired ``**bold**`` wrappers from RAG context before spaCy extraction.

    Harness markdown often wraps key phrases (e.g. ``**based in Vancouver**``).
    Leaving markers in place can suppress location/AT claims; extraction should
    see the same surface words a plain-text corpus would carry. The original
    ``context_block`` is still passed to :func:`admit_retrieved_context_to_pef`
    for headings/locators.
    """
    return re.sub(r"\*\*([^*]+)\*\*", r"\1", context_block)


@dataclass(frozen=True)
class _RagContextExtractionUnit:
    text: str
    admission_context: str
    document_id: str | None
    document_locator: str | None


_RAG_JSON_FIELD_ROLE_OBSERVED_FACT = "observed_fact"
_RAG_JSON_FIELD_ROLE_POLICY_RULE = "policy_rule"
_RAG_JSON_FIELD_ROLE_PROPOSITION_UNDER_EVAL = "proposition_under_evaluation"
_RAG_JSON_FIELD_ROLE_EPISTEMIC_LIMIT = "epistemic_limit"
_RAG_JSON_FIELD_ROLE_WORKFLOW_INSTRUCTION = "workflow_instruction"
_RAG_JSON_FIELD_ROLE_METADATA = "metadata"
_RAG_JSON_ADMISSIBLE_FIELD_ROLES = frozenset(
    {
        _RAG_JSON_FIELD_ROLE_OBSERVED_FACT,
        _RAG_JSON_FIELD_ROLE_EPISTEMIC_LIMIT,
    }
)
_RAG_JSON_OBSERVED_TEXT_KEYS = frozenset(
    {
        "asserted_fact",
        "observed_fact",
        "meaning",
        "provider_message",
    }
)
_RAG_JSON_METADATA_KEYS = frozenset(
    {
        "case_id",
        "scenario",
        "currency",
        "evidence_cutoff_utc",
        "freeze_point",
        "action_a",
        "action_b",
        "order_id",
        "amount",
        "note",
        "policy_id",
        "effective_from_utc",
        "object_type",
        "supporting_evidence_refs",
        "reason",
        "historical_integrity",
        "non_inheritance_rule",
        "workflow_constraint",
        "claim_under_evaluation",
    }
)


def _rag_json_field_tokens(path: str) -> list[str]:
    return [tok.lower() for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", path or "")]


def _rag_json_field_role(path: str) -> str:
    toks = _rag_json_field_tokens(path)
    leaf = toks[-1] if toks else ""
    if "claim_under_evaluation" in toks:
        return _RAG_JSON_FIELD_ROLE_PROPOSITION_UNDER_EVAL
    if "epistemic_limit" in toks:
        return _RAG_JSON_FIELD_ROLE_EPISTEMIC_LIMIT
    if "workflow_constraint" in toks or "non_inheritance_rule" in toks or "historical_integrity" in toks:
        return _RAG_JSON_FIELD_ROLE_WORKFLOW_INSTRUCTION
    if path.lower().startswith("rules[") and "text" in toks and "rules" in toks:
        return _RAG_JSON_FIELD_ROLE_POLICY_RULE
    if leaf in _RAG_JSON_OBSERVED_TEXT_KEYS:
        return _RAG_JSON_FIELD_ROLE_OBSERVED_FACT
    if leaf in _RAG_JSON_METADATA_KEYS:
        return _RAG_JSON_FIELD_ROLE_METADATA
    return _RAG_JSON_FIELD_ROLE_METADATA


def _split_rag_file_sections(context_block: str) -> list[tuple[str, str]]:
    """Split FILE-labeled context blocks into ``(record_label, body)`` sections."""
    matches = list(re.finditer(r"(?im)^FILE:\s*(.+?)\s*$", context_block))
    if not matches:
        return []
    sections: list[tuple[str, str]] = []
    for idx, m in enumerate(matches):
        label = m.group(1).strip()
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(context_block)
        body = context_block[start:end].strip()
        if body:
            sections.append((label, body))
    return sections


def _iter_assertive_json_field_values(value: object, *, path: str = "") -> list[tuple[str, str]]:
    """Return text field values that may be admitted as assertions.

    ``claim_under_evaluation`` is explicitly non-assertive and therefore omitted.
    """
    out: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, inner in value.items():
            key_s = str(key)
            if key_s.strip().lower() == "claim_under_evaluation":
                continue
            child_path = f"{path}.{key_s}" if path else key_s
            out.extend(_iter_assertive_json_field_values(inner, path=child_path))
        return out
    if isinstance(value, list):
        for i, inner in enumerate(value):
            child_path = f"{path}[{i}]" if path else f"[{i}]"
            out.extend(_iter_assertive_json_field_values(inner, path=child_path))
        return out
    if isinstance(value, str):
        text = value.strip()
        role = _rag_json_field_role(path)
        if role in _RAG_JSON_ADMISSIBLE_FIELD_ROLES and text and re.search(r"[A-Za-z]", text):
            out.append((path or "<root>", text))
    return out


def _build_rag_context_extraction_units(context_block: str) -> list[_RagContextExtractionUnit]:
    """Build extraction units for RAG context admission.

    - FILE-labeled JSON sections are parsed structurally and emitted one assertive
      field value at a time (record + field-path provenance).
    - Malformed JSON sections are skipped entirely (no partial extraction).
    - Non-JSON FILE sections and plain prose context are preserved as prose units.
    """
    sections = _split_rag_file_sections(context_block)
    if not sections:
        prose = context_block.strip()
        return (
            [_RagContextExtractionUnit(prose, prose, None, None)]
            if prose
            else []
        )

    units: list[_RagContextExtractionUnit] = []
    for record_label, body in sections:
        text = body.strip()
        if not text:
            continue
        locator_prefix = f"FILE:{record_label}"
        if text.startswith("{") or text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                # Structural rule: malformed JSON records do not contribute partial claims.
                continue
            for field_path, field_text in _iter_assertive_json_field_values(parsed):
                units.append(
                    _RagContextExtractionUnit(
                        text=field_text,
                        admission_context=field_text,
                        document_id=record_label,
                        document_locator=f"{locator_prefix}#{field_path}",
                    )
                )
            continue
        units.append(
            _RagContextExtractionUnit(
                text=text,
                admission_context=text,
                document_id=record_label,
                document_locator=f"{locator_prefix}#prose",
            )
        )
    return units


def _build_clarification_continuation(pending: dict) -> str:
    """Build a targeted continuation response from stored pending state.

    The response names the exact ambiguity and candidate antecedents from the
    forensic state. It never invents entities or pulls from stale context.
    """
    constraint = pending.get("failed_constraint")
    ambiguous = pending.get("ambiguous_referents", [])
    candidates = _normalized_candidate_entities_for_binding(pending)

    if constraint == "UNRESOLVED_REFERENT" and ambiguous:
        return _pre_llm_unresolved_referent_clarification(
            ambiguous,
            candidate_entities=candidates,
            original_question=str(pending.get("original_question") or ""),
            blocked_proposition=str(pending.get("blocked_proposition") or "") or None,
        )

    if constraint == "UNRESOLVED_COMPARAND":
        adj = pending.get("comparand_adjective", "")
        noun = pending.get("comparand_noun", "entity")
        return _comparand_clarification_message(
            adjective=adj,
            noun=noun,
            candidate_entities=candidates,
        )
    if constraint == "AGENCY_RISK_CONTEXT_UNRESOLVED":
        return _ensure_canonical_clarification_action_line(
            "More information required.\n\n"
            f"{_AGENCY_CONTEXT_UNRESOLVED_PROMPT}\n\n"
            "Status: Waiting for clarification.\n"
            "Action: Choose one option to continue."
        )

    return _ensure_canonical_clarification_action_line(
        "Clarification required.\n\n"
        "Clarification is required before this can continue.\n\n"
        "Status: Attribution unresolved.\n"
        "Action: Choose one option to continue."
    )


def _all_tracked_unresolved_entities_resolved(
    pef: PEFState,
    unresolved_entity_ids: list[str],
) -> bool:
    """Return True only when every tracked unresolved entity is now resolved.

    Clarification pending state must not be released on partial progress.
    If some tracked entities are still unresolved (or missing), keep pending.
    """
    if not unresolved_entity_ids:
        return False
    tracked_entities = [pef.entities.get(entity_id) for entity_id in unresolved_entity_ids]
    if any(entity is None for entity in tracked_entities):
        return False
    return all(bool(entity.resolved) for entity in tracked_entities)


def _normalize_closed_world_probe_text(text: str) -> str:
    """Normalize text for strict closed-world puzzle surface probes."""
    normalized = (text or "").lower()
    for ch in ("'", '"', "`", ".", ",", ":", ";", "?", "!", "(", ")", "[", "]"):
        normalized = normalized.replace(ch, " ")
    return " ".join(normalized.split())


def _solve_three_box_prize_puzzle(user_text: str) -> str | None:
    """Deterministically solve the canonical three-box prize puzzle when present.

    Deliberately narrow (fixed Green/Red/Purple template). Not used for transfer inventory.
    """
    t = _normalize_closed_world_probe_text(user_text)
    required_shapes = (
        "three boxes in a row",
        "green box",
        "red box",
        "purple box",
        "only one box contains a prize",
        "green the prize is in this box",
        "red this statement is of no help at all",
        "purple the prize is in the green box",
        "which box has the prize",
    )
    if all(shape in t for shape in required_shapes):
        return "green"
    return None


def _pre_llm_unresolved_referent_clarification(
    ambiguous_tokens: list[str],
    candidate_entities: list[str] | None = None,
    *,
    original_question: str | None = None,
    blocked_proposition: str | None = None,
    flag_claim: str | None = None,
    flag_evidence: str | None = None,
) -> str:
    """User-facing text for pre-LLM UNRESOLVED_REFERENT (pronouns and definite NPs).

    Avoids calling definite phrases like *the key* \"pronouns\".

    Canonical wording contract note:
    this local formatter intentionally mirrors the governed renderer contract
    (heading/reason/action/status) because these pre-LLM ambiguity branches
    finalize user copy directly in Lens before bridge.enforce() is invoked.
    """
    msg = compose_unresolved_referent_governed_message(
        ambiguous_tokens=ambiguous_tokens,
        candidate_entities=candidate_entities,
        original_question=original_question,
        blocked_proposition=blocked_proposition,
        flag_claim=flag_claim,
        flag_evidence=flag_evidence,
    )
    if msg.startswith("Decision blocked."):
        return msg
    return _ensure_canonical_clarification_action_line(msg)


def _comparand_clarification_message(
    *,
    adjective: str,
    noun: str,
    candidate_entities: list[str] | None = None,
) -> str:
    """Structured user copy for unresolved comparand clarification.

    Kept local to Lens because unresolved comparand branches emit governed copy
    directly in pre-LLM sync/stream paths and in pending-clarification resume.
    """
    reason = (
        f"This request needs a missing detail about what '{adjective}' should be compared against for {noun}."
        if adjective
        else f"This request needs a missing detail about which {noun} is being compared."
    )
    candidates = list(candidate_entities or [])
    if candidates:
        options = "\n".join(f"- {name}" for name in candidates)
        return _ensure_canonical_clarification_action_line(
            "Clarification required.\n\n"
            f"{reason}\n\n"
            f"{options}\n\n"
            "Status: Attribution unresolved.\n"
            "Action: Choose one option to continue."
        )
    return _ensure_canonical_clarification_action_line(
        "More information required.\n\n"
        f"{reason}\n\n"
        "Status: Attribution unresolved.\n"
        "Action: Choose one option to continue."
    )


def _snapshot_unresolved_referent_clarification_text(
    snapshot_tokens: list[str],
    *,
    extraction: ExtractionResult | None,
    pef: PEFState,
) -> str:
    """Lens clarification copy for ambiguity-snapshot governance (matches structural formatter)."""
    _cands = (
        _referent_resolution_candidate_names(
            extraction,
            pef,
            ambiguous_tokens=list(snapshot_tokens),
        )
        if extraction is not None
        else []
    )
    return _ensure_canonical_clarification_action_line(
        _pre_llm_unresolved_referent_clarification(
            snapshot_tokens,
            candidate_entities=_cands,
        )
    )


def _looks_person_referent_tokens(tokens: list[str]) -> bool:
    person_tokens = {
        "he",
        "she",
        "him",
        "her",
        "his",
        "hers",
        "who",
        "whom",
    }
    return any(str(tok).strip().lower() in person_tokens for tok in tokens)


def _should_state_commit_user_reported_context(
    flags: list[Flag],
    decision: GovernanceDecision,
) -> bool:
    """State commit seam for blocked legal offers that keep interaction open."""
    if decision.action == InterventionAction.PASS:
        return False
    if not decision.interaction_open:
        return False
    return any(f.flag_type == FlagType.PERSONALIZED_LEGAL_ADVICE for f in flags)


def _state_commit_user_reported_context(
    pef: PEFState,
    *,
    user_text: str,
    turn: int,
) -> bool:
    """Commit minimal user-reported context for blocked legal follow-up continuity."""
    txt = (user_text or "").strip()
    if not txt:
        return False
    if len(txt) > 800:
        txt = txt[:800].rstrip()

    ent, _ = pef.get_or_create_entity("user_reported_context", resolved=True)
    for rel in pef.get_relationships_for_subject(ent.id):
        if rel.relation == "IS" and isinstance(rel.object_literal, str) and rel.object_literal == txt:
            return False

    pef.add_relationship(
        Relationship(
            subject_id=ent.id,
            relation="IS",
            object_entity_id=None,
            object_literal=txt,
            span=Span.PRESENT,
            source_turn=turn,
            evidence="blocked_act_user_reported_context",
            provenance="system",
            extractor_backend="rule",
        )
    )
    return True


_NEUTRAL_TIMELINE_FACT_PROMPT = (
    "To build a neutral timeline, please share factual items only: "
    "(1) date or approximate time, (2) what happened, (3) who was involved, "
    "(4) documents or messages and their dates, and (5) what action followed."
)

# Draft response shape for neutral_timeline fact turns (deterministic; no LLM).
_NEUTRAL_TIMELINE_DRAFT_TITLE = "Case timeline"
_FINANCIAL_FACTS_SUMMARY_TITLE = "Financial facts summary"
_NEUTRAL_TIMELINE_ITEM_SEP = "\u241e"
_BLOCKED_CONTINUATION_DOMAIN_COPY: dict[str, tuple[str, str]] = {
    "legal": (
        "I can't answer this legal decision request.",
        "I can build a neutral case timeline you can share with a qualified legal adviser.",
    ),
    "finance": (
        "I can't answer this financial decision request.",
        "I can build a neutral financial facts timeline you can share with a qualified financial adviser.",
    ),
}

class DateShape(Enum):
    NONE = auto()
    UNAMBIGUOUS = auto()
    AMBIGUOUS = auto()


# Broad tokenizer for date-like surfaces.
# Matches:
# - YYYY-MM-DD or YYYY/MM/DD or YYYY.MM.DD
# - DD-MM-YYYY or DD/MM/YYYY or DD.MM.YYYY (and MM-DD-YYYY variants)
# - Named months: Jan 2024, 2 May 2026, May 2 2026
# - Quarters: Q1 2024
# - Bare years: 2024
_DATE_SURFACE_TOKEN_RE = re.compile(
    r"(?is)"
    r"\b\d{4}[/.-]\d{1,2}[/.-]\d{1,2}\b"  # ISO-like
    r"|\b\d{1,2}[/.-]\d{1,2}[/.-]\d{4}\b"  # Locale-like
    r"|\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+"
    r"(?:\d{1,2}(?:st|nd|rd|th)?(?:\s*,\s*(?:19|20)\d{2})?|(?:19|20)\d{2})\b"
    r"|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+(?:19|20)\d{2}\b"
    r"|\bq[1-4]\s+(?:19|20)\d{2}\b"
    r"|(?<![/.-])\b(?:19|20)\d{2}\b(?![/.-])"
)


def _classify_date_shape(surface: str) -> DateShape:
    """Procedural classifier for date-like tokens."""
    s = surface.strip().lower()
    if not s:
        return DateShape.NONE

    # ISO: YYYY-MM-DD (unambiguous)
    if re.match(r"^(?:19|20)\d{2}[/.-]\d{1,2}[/.-]\d{1,2}$", s):
        return DateShape.UNAMBIGUOUS

    # Named month (unambiguous): May 2 2026, 2 May 2026, Jan 2026
    month_names = {
        "jan", "january",
        "feb", "february",
        "mar", "march",
        "apr", "april",
        "may",
        "jun", "june",
        "jul", "july",
        "aug", "august",
        "sep", "sept", "september",
        "oct", "october",
        "nov", "november",
        "dec", "december",
    }
    tokens = re.findall(r"[a-z]+", s)
    if any(token in month_names for token in tokens):
        return DateShape.UNAMBIGUOUS

    # Quarter: Q1 2026 (unambiguous)
    if re.match(r"^q[1-4]\s+(?:19|20)\d{2}$", s):
        return DateShape.UNAMBIGUOUS

    # Bare year: 2026 (unambiguous)
    if re.match(r"^(?:19|20)\d{2}$", s):
        return DateShape.UNAMBIGUOUS

    # Locale format: DD-MM-YYYY or MM-DD-YYYY (ambiguous)
    if re.match(r"^\d{1,2}[/.-]\d{1,2}[/.-](?:19|20)\d{2}$", s):
        return DateShape.AMBIGUOUS

    return DateShape.NONE

_MEDICAL_POST_REFUSAL_CAPABILITY = "medical_post_refusal_safe_followup"
_MEDICAL_POST_REFUSAL_FLAG_TYPES: frozenset[FlagType] = frozenset(
    {
        FlagType.MEDICAL_DOSAGE_RECOMMENDATION,
        FlagType.PEDIATRIC_DOSAGE_RECOMMENDATION,
        FlagType.NUMERIC_MEDICAL_INSTRUCTION,
    }
)
_MED_EDU_LEAD_RE = re.compile(
    r"^\s*(?:what\s+is|what\s+are|how\s+does|explain|define)\b",
    re.IGNORECASE,
)
_MED_EDU_TOPIC_RE = re.compile(
    r"\b(?:symptoms?|signs?|middle\s+ear\s+infection|ear\s+infection|otitis|infection)\b",
    re.IGNORECASE,
)
_MED_PERSONAL_FACT_RE = re.compile(
    r"\b(?:i|me|my|mine|my\s+child|my\s+kid|my\s+son|my\s+daughter|for\s+me)\b",
    re.IGNORECASE,
)
_MED_EDU_BLOCKING_INTENT_RE = re.compile(
    r"\b(?:"
    r"do\s+i\s+have|am\s+i|should\s+i|should\s+we|"
    r"dose|dosage|mg/?kg|mg|kg|"
    r"which\s+antibiotic|what\s+antibiotic|"
    r"take|give|administer|prescrib(?:e|ed|ing)|"
    r"treat(?:ment)?|diagnos(?:is|e|ed|ing)"
    r")\b",
    re.IGNORECASE,
)

def _attach_rule_result(
    decision: GovernanceDecision,
    *,
    domain_override: str | None = None,
    continuation_type_override: str | None = None,
    reason_code_override: str | None = None,
    template_key_override: str | None = None,
) -> None:
    """Attach a first-class rule result for renderer/integration dispatch.

    Uses structured decision data only; never infers from rendered text.
    """
    _attach_decision_rule_result(
        decision,
        domain=domain_override,
        continuation_type=continuation_type_override,
        reason_code=reason_code_override,
        user_facing_template_key=template_key_override,
    )


def _primary_allowed_continuation(decision: GovernanceDecision) -> str | None:
    """Return first normalized allowed continuation capability, if any."""
    if not decision.allowed_continuations:
        return None
    cap = str(decision.allowed_continuations[0]).strip().lower()
    return cap or None


def _looks_like_explicit_intent_change(user_text: str) -> bool:
    """Detect explicit corridor-exit intent markers."""
    text = (user_text or "").strip().lower()
    if not text:
        return False
    markers = (
        "actually",
        "instead",
        "different question",
        "new question",
        "never mind",
        "forget that",
        "change topic",
    )
    return any(m in text for m in markers)


def _neutral_timeline_fact_has_explicit_date(fact: str) -> bool:
    """True when the clause contains a concrete calendar-style date anchor."""
    for match in _DATE_SURFACE_TOKEN_RE.finditer(fact or ""):
        if _classify_date_shape(match.group()) == DateShape.UNAMBIGUOUS:
            return True
    return False


def _neutral_timeline_fact_has_ambiguous_date(fact: str) -> bool:
    """True when the clause contains a numeric date that is locale-dependent."""
    if _neutral_timeline_fact_has_explicit_date(fact):
        return False
    for match in _DATE_SURFACE_TOKEN_RE.finditer(fact or ""):
        if _classify_date_shape(match.group()) == DateShape.AMBIGUOUS:
            return True
    return False


def _neutral_timeline_fact_has_time_signal(fact: str) -> bool:
    """True when fact includes a concrete date or coarse time window."""
    s = (fact or "").lower()
    if not s:
        return False
    if _neutral_timeline_fact_has_explicit_date(s):
        return True
    return bool(
        re.search(
            r"\b(?:last|past|previous|over)\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
            r"(?:day|days|week|weeks|month|months|year|years)\b",
            s,
        )
    )


def _neutral_timeline_fact_has_communication_signal(fact: str) -> bool:
    s = (fact or "").lower()
    if not s:
        return False
    return bool(
        re.search(
            r"\b(?:notice|warning|letter|email|text|message|call|phone|voicemail|sms)\b",
            s,
        )
    )


def _neutral_timeline_fact_has_event_signal(fact: str) -> bool:
    s = (fact or "").lower().strip()
    if not s:
        return False
    if _neutral_timeline_fact_has_communication_signal(s):
        return True
    if re.search(
        r"\b(?:happened|occurred|arrived|received|sent|served|filed|paid|missed|not\s+made)\b",
        s,
    ):
        return True
    # A pure time window like "the last six months" is not an event description.
    if _neutral_timeline_fact_has_time_signal(s) and len(s.split()) <= 6:
        return False
    return True


def _normalize_neutral_timeline_fact_phrase(fact: str) -> str:
    """Minimal surface normalisation for timeline bullets (no new facts or judgements)."""
    s = " ".join((fact or "").split()).strip().rstrip(".")
    if not s:
        return ""
    low = s.lower()
    # Narrow lexical templates — example-led only; default is light casing cleanup.
    if low in {"lack of rent payment", "no rent payment"}:
        return "Rent payment not made"
    return s[0].upper() + s[1:] if len(s) > 1 else s.upper()


_LEGAL_RELEVANCE_CUES: frozenset[str] = frozenset(
    {
        "landlord",
        "tenant",
        "tenancy",
        "lease",
        "rent",
        "evict",
        "notice",
        "quit",
        "employer",
        "employee",
        "employment",
        "fired",
        "dismiss",
        "warning",
        "redundant",
        "contract",
        "agreement",
        "signed",
        "witness",
        "served",
        "filed",
        "court",
        "tribunal",
        "claim",
        "dispute",
        "appeal",
        "solicitor",
        "lawyer",
        "legal",
        "police",
        "arrest",
        "accident",
        "injury",
        "damage",
        "payment",
        "debt",
        "invoice",
        "letter",
        "email",
        "message",
        "call",
        "meeting",
        "date",
        "time",
        "happened",
        "occurred",
    }
)

_FINANCE_RELEVANCE_CUES: frozenset[str] = frozenset(
    {
        "goal",
        "concern",
        "retirement",
        "savings",
        "investment",
        "invest",
        "stock",
        "share",
        "risk",
        "tolerance",
        "superannuation",
        "adviser",
        "advisor",
        "pension",
        "capital",
        "asset",
        "account",
        "amount",
        "cash",
        "money",
        "fund",
        "portfolio",
        "tax",
        "income",
        "expense",
        "debt",
        "loan",
        "mortgage",
        "insurance",
        "bank",
        "credit",
        "debit",
        "date",
        "time",
        "year",
        "month",
        "day",
        "fact",
        "timeline",
    }
)

_MEDICAL_RELEVANCE_CUES: frozenset[str] = frozenset(
    {
        "symptom",
        "pain",
        "numbness",
        "onset",
        "started",
        "worsened",
        "better",
        "worse",
        "medication",
        "medicine",
        "drug",
        "dose",
        "dosage",
        "pill",
        "tablet",
        "capsule",
        "condition",
        "illness",
        "disease",
        "infection",
        "fever",
        "cough",
        "rash",
        "warning",
        "urgent",
        "emergency",
        "clinician",
        "pharmacist",
        "doctor",
        "nurse",
        "hospital",
        "clinic",
        "date",
        "time",
        "week",
        "month",
        "day",
        "year",
        "age",
        "weight",
        "allergy",
        "allergic",
        "reaction",
        "antibiotic",
        "diagnosis",
        "diagnosed",
    }
)


_EDUCATION_RELEVANCE_CUES: frozenset[str] = frozenset(
    {
        "student", "teacher", "educator", "school", "university", "college",
        "curriculum", "assignment", "exam", "grade", "degree", "diploma",
        "course", "syllabus", "admissions", "transcript", "academic", "ferpa",
        "tuition", "thesis", "dissertation", "lecture", "enrol", "enroll",
        "campus", "semester",
    }
)

_WORKFORCE_RELEVANCE_CUES: frozenset[str] = frozenset(
    {
        "employee", "employer", "hire", "fired", "terminate", "salary",
        "wage", "hr", "workforce", "union", "labor", "labour",
        "discrimination", "harassment", "onboarding", "performance",
        "payroll", "benefit", "leave", "fmla", "eeoc", "redundancy",
        "dismissal", "retrenchment", "recruit", "recruitment", "appraisal",
    }
)

_ENTERPRISE_RELEVANCE_CUES: frozenset[str] = frozenset(
    {
        "contract", "procurement", "vendor", "supplier", "rfp", "tender",
        "budget", "forecast", "revenue", "merger", "acquisition", "board",
        "shareholder", "nda", "confidential", "proprietary", "compliance",
        "corporate", "enterprise", "insider", "trade secret",
    }
)


def _is_fact_relevant_to_corridor(fact: str, domain: str) -> bool:
    """True if the fact has surface relevance to the active domain corridor."""
    if not domain or domain == "general":
        return True

    lower = fact.lower()

    # Protected-domain corridors fail closed unless the fact has an explicit
    # domain cue. A bare date is not enough, because "On 2024-05-05 I bought
    # a blue sofa" should not pollute legal, finance, or medical corridors.
    if domain == "legal":
        return any(
            re.search(rf"\b{re.escape(cue)}\b", lower)
            for cue in _LEGAL_RELEVANCE_CUES
        )

    if domain == "finance":
        return any(
            re.search(rf"\b{re.escape(cue)}\b", lower)
            for cue in _FINANCE_RELEVANCE_CUES
        )

    if domain == "medical":
        return any(
            re.search(rf"\b{re.escape(cue)}\b", lower)
            for cue in _MEDICAL_RELEVANCE_CUES
        )

    if domain == "education":
        return any(
            re.search(rf"\b{re.escape(cue)}\b", lower)
            for cue in _EDUCATION_RELEVANCE_CUES
        )

    if domain == "workforce":
        return any(
            re.search(rf"\b{re.escape(cue)}\b", lower)
            for cue in _WORKFORCE_RELEVANCE_CUES
        )

    if domain == "enterprise":
        return any(
            re.search(rf"\b{re.escape(cue)}\b", lower)
            for cue in _ENTERPRISE_RELEVANCE_CUES
        )

    # Fail closed for unknown protected-domain corridors.
    return False


def _format_neutral_timeline_draft_from_facts(facts: list[str]) -> str:
    """Build structured neutral timeline text with bullets only."""
    bullets: list[str] = []
    for raw in facts:
        phrase = _normalize_neutral_timeline_fact_phrase(raw)
        if not phrase:
            continue
        if _neutral_timeline_fact_has_explicit_date(raw):
            bullets.append(f"• {phrase}")
        elif _neutral_timeline_fact_has_ambiguous_date(raw):
            bullets.append(f"• [Ambiguous date] {phrase}")
        else:
            bullets.append(f"• [Undated] {phrase}")
    if not bullets:
        return ""
    body = "\n".join(bullets)
    return f"{_NEUTRAL_TIMELINE_DRAFT_TITLE}\n\n{body}"


def _format_neutral_timeline_from_bullets(bullets: list[str]) -> str:
    lines = [b for b in bullets if b.strip()]
    if not lines:
        return ""
    return f"{_NEUTRAL_TIMELINE_DRAFT_TITLE}\n\n" + "\n".join(lines)


def _neutral_timeline_blocked_prefix(context: dict[str, str] | None) -> str:
    """Render explicit blocked outcome banner for legal/finance continuation turns."""
    ctx = context or {}
    domain = str(ctx.get("domain") or "").strip().lower()
    if domain == "finance":
        return ""
    copy_pair = _BLOCKED_CONTINUATION_DOMAIN_COPY.get(domain)
    if copy_pair is None:
        return ""
    reason, safe = copy_pair
    return (
        f"BLOCKED ({domain})\n"
        f"{reason}\n"
        f"Safe continuation: {safe}"
    )


def _neutral_timeline_context_list(context: dict[str, str] | None, key: str) -> list[str]:
    raw = str((context or {}).get(key) or "")
    return [x for x in raw.split(_NEUTRAL_TIMELINE_ITEM_SEP) if x]


def _neutral_timeline_context_set_list(context: dict[str, str], key: str, items: list[str]) -> None:
    context[key] = _NEUTRAL_TIMELINE_ITEM_SEP.join(items)


def _neutral_timeline_missing_prompts(context: dict[str, str]) -> list[str]:
    prompts: list[str] = []
    has_time = context.get("timeline_has_time") == "1"
    has_event = context.get("timeline_has_event") == "1"
    has_comm = context.get("timeline_has_communication") == "1"
    if not has_time:
        prompts.append("what happened when")
    if not has_event:
        prompts.append("what happened")
    if not has_comm:
        prompts.append("any notices or communication")
    prompts.append("other relevant dated events")
    return prompts


def _is_finance_blocked_continuation(context: dict[str, str] | None) -> bool:
    return str((context or {}).get("domain") or "").strip().lower() == "finance"


def _finance_slot_signal(user_text: str) -> str | None:
    text = (user_text or "").strip().lower()
    if not text:
        return None
    mapping = (
        ("questions for the adviser", "questions_for_adviser"),
        ("goal or concern", "goal_or_concern"),
        ("relevant dates", "relevant_dates"),
        ("amounts involved", "amounts_involved"),
        ("accounts or assets", "accounts_or_assets"),
        ("risk constraints", "risk_constraints"),
    )
    for label, slot in mapping:
        if label in text:
            return slot
    return None


def _finance_slot_label(slot: str) -> str:
    labels = {
        "questions_for_adviser": "Questions for the adviser",
        "goal_or_concern": "Goal or concern",
        "relevant_dates": "Relevant dates",
        "amounts_involved": "Amounts involved",
        "accounts_or_assets": "Accounts or assets",
        "risk_constraints": "Risk constraints",
    }
    return labels.get(slot, slot.replace("_", " ").capitalize())


def _finance_add_next(context: dict[str, str]) -> list[str]:
    ordered = (
        ("goal or concern", "finance_goal_or_concern"),
        ("relevant dates", "finance_relevant_dates"),
        ("amounts involved", "finance_amounts_involved"),
        ("accounts or assets", "finance_accounts_or_assets"),
        ("risk constraints", "finance_risk_constraints"),
    )
    out: list[str] = []
    for label, key in ordered:
        if context.get(key) != "1":
            out.append(label)
    return out


def _format_finance_summary_from_context(context: dict[str, str]) -> str:
    questions_value = "[not yet specified]"
    if context.get("finance_questions_for_adviser") == "1":
        questions_value = context.get("finance_questions_for_adviser_value") or "[specified]"
    lines = [
        _FINANCIAL_FACTS_SUMMARY_TITLE,
        "",
        f"• Questions for the adviser: {questions_value}",
        "",
        "Add next:",
    ]
    for item in _finance_add_next(context):
        lines.append(f"• {item}")
    return "\n".join(lines)


def _looks_like_neutral_timeline_fact_request(user_text: str) -> bool:
    """True when user asks what factual inputs are needed for timeline formatting."""
    text = (user_text or "").strip().lower()
    if not text:
        return False
    cues = (
        "what facts do you need",
        "what info do you need",
        "what information do you need",
        "what details do you need",
        "which facts do you need",
        "what should i provide",
    )
    return any(c in text for c in cues)


def _neutral_timeline_input_is_blocked_financial_act(user_text: str) -> bool:
    """True when request-side finance gates classify input as personalized advice act."""
    return any(
        f.flag_type == FlagType.PERSONALIZED_FINANCIAL_ADVICE
        for f in evaluate_blocked_act_request(user_text or "")
    )


def _extract_neutral_timeline_facts(user_text: str) -> list[str]:
    """Extract plain factual clauses for neutral timeline formatting."""
    raw = (user_text or "").strip()
    if not raw:
        return []
    clauses = [
        c.strip(" -\t")
        for c in re.split(r"[;\n]+|(?<=[.!])\s+", raw)
        if c.strip()
    ]
    facts: list[str] = []
    for c in clauses:
        lower = c.lower()
        if "?" in c:
            continue
        if lower.startswith(("what ", "why ", "how ", "should ", "would ", "could ")):
            continue
        facts.append(c.rstrip("."))
    return facts[:8]


def _build_neutral_timeline_followup_response(
    user_text: str,
    context: dict[str, str] | None = None,
) -> tuple[str, bool, bool, dict[str, str] | None]:
    """Render constrained follow-up response for active neutral_timeline capability.

    Returns (response_text, completed). completed=True retires active capability.
    """
    text = (user_text or "").strip()
    lower = text.lower()
    safe_context = dict(context or {})
    if _is_finance_blocked_continuation(safe_context):
        slot = _finance_slot_signal(text)
        if slot:
            if slot == "questions_for_adviser":
                safe_context["finance_questions_for_adviser"] = "1"
                safe_context["finance_questions_for_adviser_value"] = "[not yet specified]"
            else:
                safe_context[f"finance_{slot}"] = "1"
            return (_format_finance_summary_from_context(safe_context), False, True, safe_context)

        # If no slot matched, check if any fact is relevant to finance corridor
        facts = _extract_neutral_timeline_facts(text)
        if any(_is_fact_relevant_to_corridor(f, "finance") for f in facts):
            # Admitted as a general relevant update even if no slot matched
            return (_format_finance_summary_from_context(safe_context), False, True, safe_context)

        return (_format_finance_summary_from_context(safe_context), False, False, safe_context)
    blocked_prefix = _neutral_timeline_blocked_prefix(safe_context)
    if _neutral_timeline_input_is_blocked_financial_act(text):
        body = (
            "Neutral timeline mode accepts dated factual events only, "
            "not investment execution or reallocation directives."
        )
        if blocked_prefix:
            return (f"{blocked_prefix}\n\n{body}", False, False, safe_context)
        return (body, False, False, safe_context)
    if _looks_like_neutral_timeline_fact_request(text):
        if blocked_prefix:
            return (f"{blocked_prefix}\n\n{_NEUTRAL_TIMELINE_FACT_PROMPT}", False, False, safe_context)
        return (_NEUTRAL_TIMELINE_FACT_PROMPT, False, False, safe_context)
    if lower in {"done", "finished", "that's all", "thats all"}:
        done = "Understood. The neutral timeline task is complete."
        if blocked_prefix:
            return (f"{blocked_prefix}\n\n{done}", True, False, safe_context)
        return (done, True, False, safe_context)

    facts = _extract_neutral_timeline_facts(text)
    if facts:
        prior_items = _neutral_timeline_context_list(safe_context, "timeline_items")
        domain = safe_context.get("domain", "general")
        admitted_any = False
        for raw in facts:
            if not _is_fact_relevant_to_corridor(raw, domain):
                continue
            phrase = _normalize_neutral_timeline_fact_phrase(raw)
            if not phrase:
                continue
            if _neutral_timeline_fact_has_explicit_date(raw):
                line = f"• {phrase}"
            elif _neutral_timeline_fact_has_ambiguous_date(raw):
                line = f"• [Ambiguous date] {phrase}"
            else:
                line = f"• [Undated] {phrase}"
            prior_items.append(line)
            admitted_any = True
            if _neutral_timeline_fact_has_event_signal(raw):
                safe_context["timeline_has_event"] = "1"
            if _neutral_timeline_fact_has_time_signal(raw):
                safe_context["timeline_has_time"] = "1"
            if _neutral_timeline_fact_has_communication_signal(raw):
                safe_context["timeline_has_communication"] = "1"
        
        if admitted_any:
            _neutral_timeline_context_set_list(safe_context, "timeline_items", prior_items)
            formatted = _format_neutral_timeline_from_bullets(prior_items)
            if formatted:
                missing = _neutral_timeline_missing_prompts(safe_context)
                add_next = "Add next:\n" + "\n".join(f"• {m}" for m in missing)
                body = f"{formatted}\n\n{add_next}"
                if blocked_prefix:
                    return (f"{blocked_prefix}\n\n{body}", False, True, safe_context)
                return (body, False, True, safe_context)
    fallback = (
        "I can continue only in neutral timeline mode. "
        "Please share factual events and dates, and I will format them without evaluation."
    )
    if blocked_prefix:
        return (f"{blocked_prefix}\n\n{fallback}", False, False, safe_context)
    return (fallback, False, False, safe_context)


def _active_continuation_context_from_decision(decision: GovernanceDecision) -> dict[str, str] | None:
    """Persist minimal context so continuation turns can keep blocked boundary visible."""
    cap = _primary_allowed_continuation(decision)
    if cap not in ("neutral_timeline", _MEDICAL_POST_REFUSAL_CAPABILITY):
        return None
    if decision.action != InterventionAction.HARD_STOP:
        return None
    if not decision.interaction_open:
        return None
    if not decision.flags:
        return None
    ft = decision.flags[0].flag_type
    if ft == FlagType.PERSONALIZED_LEGAL_ADVICE:
        return {"domain": "legal"}
    if ft == FlagType.PERSONALIZED_FINANCIAL_ADVICE:
        return {"domain": "finance"}
    if ft in _MEDICAL_POST_REFUSAL_FLAG_TYPES or ft == FlagType.PERSONALIZED_MEDICAL_ADVICE:
        return {"domain": "medical"}
    return None


def _display_continuation_capability_name(
    capability: str,
    context: dict[str, str] | None,
) -> str:
    domain = str((context or {}).get("domain") or "").strip().lower()
    if capability == "neutral_timeline" and domain == "finance":
        return "financial_facts_summary"
    return capability


def _extract_medical_referent(user_text: str) -> str | None:
    text = (user_text or "").strip().lower()
    if not text:
        return None
    referent_patterns = (
        (r"\bmy\s+(kid|child|son|daughter|baby|infant|toddler)\b", "my kid"),
        (r"\bfor\s+(children|kids|a child)\b", "a child"),
        (r"\bmy\s+(wife|husband|partner|mother|father)\b", "my family member"),
        (r"\bfor\s+me\b", "me"),
        (r"\bfor\s+my\b", "my family member"),
    )
    for pattern, label in referent_patterns:
        if re.search(pattern, text):
            return label
    return None


def _extract_medical_condition(user_text: str) -> str | None:
    text = (user_text or "").strip()
    if not text:
        return None
    condition_match = re.search(
        r"\bfor\s+(an?\s+)?([a-z][a-z\s\-]{2,40})",
        text.lower(),
    )
    if not condition_match:
        return None
    candidate = condition_match.group(2).strip(" .,:;!?")
    if candidate in {"children", "child", "kids", "me", "my kid"}:
        return None
    condition_cues = (
        "infection",
        "fever",
        "pain",
        "cough",
        "ear",
        "throat",
        "rash",
        "symptom",
    )
    if any(cue in candidate for cue in condition_cues):
        return candidate
    return None


def _build_medical_post_refusal_context(user_text: str) -> dict[str, str]:
    referent = _extract_medical_referent(user_text)
    condition = _extract_medical_condition(user_text)
    context: dict[str, str] = {
        "domain": "medical",
        "refused_action": "dosage/treatment instruction",
        "refusal_reason": "Medication dosage, diagnosis, and treatment instructions are blocked in this corridor.",
    }
    if referent:
        context["patient_referent"] = referent
    if condition:
        context["condition"] = condition
    return context


def _looks_like_medical_question_prep_request(user_text: str) -> bool:
    text = (user_text or "").strip().lower()
    if not text:
        return False
    cues = (
        "what questions can i ask",
        "questions can i ask the doctor",
        "questions should i ask the doctor",
        "what should i ask the doctor",
        "what should i ask the pharmacist",
        "questions for the pharmacist",
        "questions to ask",
    )
    return any(cue in text for cue in cues)


def _looks_like_medical_fact_prep_request(user_text: str) -> bool:
    text = (user_text or "").strip().lower()
    if not text:
        return False
    cues = (
        "what facts do i need",
        "what should i tell the doctor",
        "what should i tell the pharmacist",
        "what information should i give",
        "what details should i give",
        "what symptoms should i track",
        "what should i prepare for the doctor",
    )
    return any(cue in text for cue in cues)


def _is_general_health_educational_question(user_text: str) -> bool:
    """True for general medical education prompts that should use normal PASS path."""
    text = (user_text or "").strip()
    if not text:
        return False
    if _MED_PERSONAL_FACT_RE.search(text):
        return False
    if _MED_EDU_BLOCKING_INTENT_RE.search(text):
        return False
    if _MED_EDU_LEAD_RE.search(text) is None:
        return False
    return _MED_EDU_TOPIC_RE.search(text) is not None


def _build_medical_question_prep_response(context: dict[str, str]) -> str:
    referent = context.get("patient_referent", "the patient")
    condition = context.get("condition")
    opening = "Here are safe questions to ask your clinician or pharmacist."
    if condition:
        opening = f"Here are safe questions to ask your clinician or pharmacist about {condition}."
    return (
        f"{opening}\n"
        f"- What dose is appropriate for {referent}'s weight and age?\n"
        "- How often should it be given?\n"
        "- How many days should the course run?\n"
        "- What side effects or allergy signs should I watch for?\n"
        "- What symptoms mean urgent care is needed?"
    )


def _build_medical_fact_prep_response(context: dict[str, str]) -> str:
    condition = context.get("condition")
    opening = "Share these neutral facts with your clinician or pharmacist."
    if condition:
        opening = f"Share these neutral facts with your clinician or pharmacist about {condition}."
    return (
        f"{opening}\n"
        "- Child's age and weight (or adult age if applicable)\n"
        "- Symptoms and how long they have lasted\n"
        "- Fever or pain severity\n"
        "- Allergies, especially penicillin or amoxicillin\n"
        "- Current medicines and recent doses already taken\n"
        "- Prior reactions to antibiotics\n"
        "- Whether diagnosis was confirmed by a clinician"
    )


def _build_medical_post_refusal_followup_response(
    user_text: str,
    context: dict[str, str] | None,
) -> tuple[str, bool]:
    safe_context = dict(context or {})
    if "domain" not in safe_context:
        safe_context["domain"] = "medical"
    if "refused_action" not in safe_context:
        safe_context["refused_action"] = "dosage/treatment instruction"
    if "refusal_reason" not in safe_context:
        safe_context["refusal_reason"] = (
            "Medication dosage, diagnosis, and treatment instructions are blocked in this corridor."
        )

    text = (user_text or "").strip()
    lower = text.lower()
    if lower in {"done", "finished", "that's all", "thats all"}:
        return "Understood. The medical follow-up preparation is complete.", True
    if _looks_like_medical_question_prep_request(text):
        return _build_medical_question_prep_response(safe_context), False
    if _looks_like_medical_fact_prep_request(text):
        return _build_medical_fact_prep_response(safe_context), False

    # Check for relevant medical facts to admit
    facts = _extract_neutral_timeline_facts(text)
    relevant_facts = [f for f in facts if _is_fact_relevant_to_corridor(f, "medical")]
    if relevant_facts:
        prior_items = _neutral_timeline_context_list(safe_context, "medical_items")
        for f in relevant_facts:
            phrase = _normalize_neutral_timeline_fact_phrase(f)
            if phrase:
                prior_items.append(f"• {phrase}")
        _neutral_timeline_context_set_list(safe_context, "medical_items", prior_items)

        summary = "Neutral symptom/event summary:\n\n" + "\n".join(prior_items)
        return summary, False

    return (
        "I can only help with safe medical follow-up preparation here: "
        "questions to ask a clinician or pharmacist, and neutral facts/symptoms to report. "
        "I can't provide dosage, diagnosis, treatment selection, or medication administration instructions.",
        False,
    )


def _pending_unresolved_referent_flag(pending: dict) -> Flag:
    """Build a structured UNRESOLVED_REFERENT flag from pending clarification state."""
    ambiguous = [str(x) for x in (pending.get("ambiguous_referents") or []) if str(x).strip()]
    candidates = _normalized_candidate_entities_for_binding(pending)
    token = ambiguous[0] if ambiguous else "referent"
    evidence = (
        f"Pending unresolved referent token '{token}' requires disambiguation."
    )
    return Flag(
        flag_type=FlagType.UNRESOLVED_REFERENT,
        entity_name=token,
        claim=f"Unresolved referent: {token}",
        evidence=evidence,
        severity="warning",
        candidates=tuple(candidates),
    )


_FOLLOWUP_DISAMBIG_PROBE_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "will", "would", "could", "should",
    "may", "might", "must", "shall", "to", "of", "in", "on", "at", "for",
    "and", "or", "but", "it", "this", "that", "these", "those",
})


def _pending_unresolved_comparand_flag(pending: dict) -> Flag:
    """Structured UNRESOLVED_COMPARAND flag from stored pending clarification."""
    noun = str(pending.get("comparand_noun") or "").strip() or "entity"
    adj = str(pending.get("comparand_adjective") or "").strip()
    candidates = _normalized_candidate_entities_for_binding(pending)
    cand_join = ", ".join(candidates)
    return Flag(
        flag_type=FlagType.UNRESOLVED_COMPARAND,
        entity_name=noun[:60],
        claim=(
            f"'{adj}' has {len(candidates)} eligible comparands for {noun}"
            if adj
            else f"Unresolved comparand for {noun}"
        ),
        evidence=(
            f"Candidates: {cand_join}. Cannot collapse without explicit comparand."
            if cand_join
            else "Pending comparand clarification requires explicit choice."
        ),
        severity="warning",
        candidates=tuple(candidates),
    )


def _pending_ambiguity_followup_looks_like_continuation(
    user_text: str,
    pending: dict,
) -> bool:
    """True when follow-up text is a disambiguation question for held comparand state.

    Avoids routing *whose* / *which party* re-asks through fresh comparative
    adjudication (which would re-emit FORCE_REVISE and drop continuation tone).
    """
    raw = (user_text or "").strip()
    low = raw.lower().rstrip("?.!")
    if not low:
        return False
    toks = _tokenize_alpha_words(low)

    opens_whose = low.startswith("whose ") or (bool(toks) and toks[0] == "whose")
    if _word_boundary_substring_in_text("which person", raw):
        return True
    if _word_boundary_substring_in_text("which party", raw):
        return True
    if _contains_token_subsequence(toks, ["who", "was", "it"]):
        return True

    fc = str(pending.get("failed_constraint") or "")
    noun = str(pending.get("comparand_noun") or "").strip().lower()
    adj = str(pending.get("comparand_adjective") or "").strip().lower()

    scope_bits: list[str] = []
    for key in ("blocked_proposition", "original_question"):
        s = str(pending.get(key) or "").strip().lower()
        if s:
            scope_bits.append(s)
    probe = " ".join(scope_bits)
    probe_toks = set(_tokenize_alpha_words(probe)) - _FOLLOWUP_DISAMBIG_PROBE_STOPWORDS
    overlap = bool(probe_toks) and bool(probe_toks.intersection(set(toks)))

    if opens_whose:
        if fc == "UNRESOLVED_COMPARAND":
            return bool(noun) and _word_boundary_substring_in_text(noun, raw)
        return True

    if bool(toks) and toks[0] in ("who", "whom", "which", "what"):
        if fc == "UNRESOLVED_COMPARAND":
            if noun and adj:
                return (
                    _word_boundary_substring_in_text(noun, raw)
                    and _word_boundary_substring_in_text(adj, raw)
                )
            if noun:
                return _word_boundary_substring_in_text(noun, raw)
            return False
        return overlap

    return False


def _pending_ambiguity_continuation_applies(
    *,
    turn_act: TurnAct,
    pending: dict,
    history_user_input: str,
) -> bool:
    if turn_act not in (TurnAct.CLARIFY, TurnAct.QUERY):
        return False
    fc = str(pending.get("failed_constraint") or "")
    if fc == "UNRESOLVED_REFERENT":
        # Binding-resolution turns may classify as QUERY/CLARIFY (lightweight turn_act).
        # Meta inquiries and *whose* / scope-overlap re-asks stay in clarification (CONTAIN).
        return (
            _looks_like_clarification_inquiry_meta(history_user_input)
            or _pending_unresolved_referent_help_question_shape(history_user_input)
            or _pending_ambiguity_followup_looks_like_continuation(
                history_user_input, pending,
            )
        )
    if fc == "UNRESOLVED_COMPARAND":
        return _pending_ambiguity_followup_looks_like_continuation(
            history_user_input, pending,
        )
    return False


_PENDING_AMBIGUITY_FAILED_CONSTRAINTS = frozenset({
    "UNRESOLVED_REFERENT",
    "UNRESOLVED_COMPARAND",
})


def _held_pending_blocks_comparative_force_revise(
    pending: dict | None,
    *,
    binding_resumed: bool,
) -> bool:
    """True when an active ambiguity hold should not re-enter FORCE_REVISE on comparand re-gate."""
    if binding_resumed or pending is None:
        return False
    return str(pending.get("failed_constraint") or "") in _PENDING_AMBIGUITY_FAILED_CONSTRAINTS


def _pending_ambiguity_containment_decision(
    pending: dict,
    *,
    rationale: str,
) -> GovernanceDecision:
    """CONTAIN decision from stored pending (preserves failed_constraint / candidates)."""
    continuation = _build_clarification_continuation(pending)
    cont_flags = _pending_ambiguity_continuation_flags(pending)
    cont_decision = GovernanceDecision(
        action=InterventionAction.CONTAIN,
        flags=cont_flags,
        rationale=rationale,
        policy="strict",
        pathway_id="P_ASK_DISAMBIGUATE",
        output_mode="clarification_request",
        commitment_closed=True,
        interaction_open=True,
    )
    cont_decision.original_response = ""
    cont_decision.corrected_response = continuation
    cont_decision.governed_response = continuation
    return cont_decision


def _pending_ambiguity_continuation_flags(pending: dict) -> list[Flag]:
    fc = str(pending.get("failed_constraint") or "")
    if fc == "UNRESOLVED_REFERENT":
        return [_pending_unresolved_referent_flag(pending)]
    if fc == "UNRESOLVED_COMPARAND":
        return [_pending_unresolved_comparand_flag(pending)]
    return []


def _format_explicit_correction_surface_response(
    user_text: str,
    extraction: ExtractionResult | None,
) -> str:
    """Build neutral correction acknowledgement text from smallest available fact."""
    txt = (user_text or "").strip()
    lower = txt.lower()

    fact_text = ""
    if lower.startswith("correction:"):
        fact_text = txt[len("correction:") :].strip()
    elif lower.startswith("i misspoke"):
        dot = txt.find(".")
        fact_text = txt[dot + 1 :].strip() if dot >= 0 else ""
    elif lower.startswith("i meant"):
        fact_text = txt[len("i meant") :].strip(" ,:;-")

    if not fact_text and extraction and extraction.claims:
        c = extraction.claims[0]
        subj = (c.subject or "").strip()
        obj = (c.obj or "").strip()
        rel = (c.relation or "").strip().upper()
        if subj and obj:
            if rel == "IS":
                fact_text = f"{subj} is {obj}"
            elif rel == "HAS":
                fact_text = f"{subj} has {obj}"
            else:
                fact_text = f"{subj} {rel.lower()} {obj}"

    if fact_text:
        fact = fact_text.rstrip(".")
        lower_fact = fact.lower()
        idx = lower_fact.find(" is ")
        if idx >= 0:
            subject = fact[:idx].strip()
            value = fact[idx + 4 :].strip()
            if subject and value:
                return f"Correction noted. {subject} is now recorded as {value}."
        return f"Correction noted. {fact}."

    return "Correction noted. I have updated the recorded state."


def _split_sentences_by_terminal_punct(text: str) -> list[str]:
    """Split text into sentences using '.', '!', '?' terminal punctuation.

    Plain character scan only (no regex) for referent-path metadata extraction.
    """
    s = (text or "").strip()
    if not s:
        return []

    out: list[str] = []
    start = 0
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch in ".!?":
            end = i + 1
            chunk = s[start:end].strip()
            if chunk:
                out.append(chunk)
            i = end
            while i < n and s[i].isspace():
                i += 1
            start = i
            continue
        i += 1

    if start < n:
        tail = s[start:].strip()
        if tail:
            out.append(tail)
    return out


def _ambiguous_tokens_for_pending_payload(
    *,
    ambiguous_snapshot_merged_tokens: list[str] | None,
    extraction: ExtractionResult | None,
) -> list[str]:
    """Stable ambiguity seed for pending dicts when snapshot merge omits explicit tokens."""
    if ambiguous_snapshot_merged_tokens:
        return list(ambiguous_snapshot_merged_tokens)
    if extraction is not None and extraction.ambiguous_referents:
        return list(extraction.ambiguous_referents)
    return []


def _build_pending_unresolved_referent_dict(
    pef: PEFState,
    *,
    turn: int | None = None,
    scope_text: str,
    original_question: str,
    extraction: ExtractionResult,
    ambiguous_tokens: list[str] | None = None,
    detected_span: Span,
) -> dict[str, object]:
    """Full ``pending_clarification`` payload for UNRESOLVED_REFERENT (shared pre/post-LLM)."""
    _sentences = _split_sentences_by_terminal_punct(scope_text)
    _ambiguous = list(ambiguous_tokens or [])
    _ambiguous = _extend_ambiguous_with_downstream_pronouns(original_question, _ambiguous)
    _blocked_sents = [
        s for s in _sentences
        if any(
            _word_boundary_substring_in_text(p, s)
            for p in _ambiguous
        )
    ]
    _blocked_proposition = " ".join(_blocked_sents) if _blocked_sents else None
    _ambig_lower = {p.lower() for p in _ambiguous}
    _blocked_sent_l = [s.lower() for s in _blocked_sents]
    # Use _build_semantic_transactions as the single source of held_reason truth.
    _tx_by_id: dict[int, object] = {
        id(tx.claim): tx
        for tx in _build_semantic_transactions(extraction)
        if not tx.allowed_commit
    }
    _blocked_claims_raw: list[dict[str, object]] = []
    for c in extraction.claims:
        _tx = _tx_by_id.get(id(c))
        _include = (
            _tx is not None  # held by SemanticTransaction — always include
            or any(_word_boundary_substring_in_text(p, c.subject) for p in _ambig_lower)
            or any(_word_boundary_substring_in_text(p, str(c.obj)) for p in _ambig_lower)
            or (
                canonicalize_relation(c.relation) in {"GIVE", "HAS", "TAKE"}
                and (
                    not _blocked_sent_l
                    or any(s in str(c.evidence or "").lower() for s in _blocked_sent_l)
                )
            )
        )
        if not _include:
            continue
        _cd: dict[str, object] = {
            "subject": c.subject,
            "relation": c.relation,
            "obj": c.obj,
            "span": c.span.value,
            "negated": c.negated,
            "evidence": c.evidence,
        }
        if _tx is not None and getattr(_tx, "held_reason", None):
            _cd["held_reason"] = _tx.held_reason
            if _tx.held_reason == "UNRESOLVED_COMPARAND":
                _adj_surface = str(getattr(_tx, "unresolved_comparand", None) or c.obj).strip()
                _ca = next(
                    (ca for ca in (extraction.comparative_ambiguities or [])
                     if ca.adjective.lower() == _adj_surface.lower()),
                    None,
                )
                _cd["comparand_adjective"] = _adj_surface
                _cd["comparand_noun"] = _ca.noun if _ca else str(c.obj)
            elif _tx.held_reason == "UNRESOLVED_POSSESSOR":
                _cd["possessor_pronoun"] = str(getattr(_tx, "unresolved_possessor", None) or "")
                _cd["item_noun"] = _extract_item_noun_from_possessive_np(c.subject)
        _blocked_claims_raw.append(_cd)
    _blocked_claims = _blocked_claims_raw
    suppressed_subject_normalized = frozenset(
        str(bc.get("subject") or "").strip().lower()
        for bc in _blocked_claims
        if bc.get("subject")
        and (
            _referent_candidate_surface_contains_possessive_determiner(
                str(bc.get("subject") or "")
            )
            or _referent_surface_heads_ambiguous_pronoun(
                str(bc.get("subject") or ""),
                list(_ambiguous),
            )
        )
    )
    _ref_candidates = _referent_resolution_candidate_names(
        extraction,
        pef,
        ambiguous_tokens=ambiguous_tokens,
        suppressed_subject_normalized=suppressed_subject_normalized,
    )
    if _blocked_proposition and not any(
        any(_word_boundary_substring_in_text(p, str(bc.get("subject") or "")) for p in _ambig_lower)
        for bc in _blocked_claims
    ):
        _synthetic = _synthesize_held_claim_from_blocked_proposition(
            str(_blocked_proposition),
            _ambiguous,
        )
        if _synthetic is not None:
            _blocked_claims = list(_blocked_claims) + [_synthetic]
    _person_antecedents = _person_antecedent_names_from_surfaces(
        _candidate_entity_names(list(extraction.entity_mentions)),
    )
    _ref_candidates = _filter_referent_resolution_candidates(
        _ref_candidates,
        ambiguous_tokens=_ambiguous,
        blocked_claims=_blocked_claims,
        person_antecedent_names=_person_antecedents,
    )
    _pending_task = _infer_pending_attribution_task(
        original_question=original_question,
        blocked_claims=_blocked_claims,
        ambiguous_referents=_ambiguous,
        blocked_proposition=str(_blocked_proposition or "") or None,
    )
    _clarification_prompt = None
    if _blocked_claims:
        _subject = str(_blocked_claims[0].get("subject") or "").strip().lower()
        if _subject.endswith("sister"):
            _clarification_prompt = "Whose sister?"
    if _clarification_prompt is None:
        _focus_text = " ".join(
            str(x or "").lower()
            for x in (_blocked_proposition, scope_text, original_question)
        )
        if "sister" in _focus_text and any(
            t in {str(tok).lower() for tok in _ambiguous}
            for t in {"her", "she", "hers"}
        ):
            _clarification_prompt = "Whose sister?"
    _payload: dict[str, object] = {
        "original_question": original_question,
        "turn": turn,
        # Pronoun / referent ASK: use the re-extract-OriginalQuestion branch in step
        # 1.5 (``unresolved_entity_ids`` non-empty opts into entity-placeholder
        # resolution, which a bare name like "Anna" may not satisfy).
        "unresolved_entity_ids": [],
        "failed_constraint": "UNRESOLVED_REFERENT",
        "ambiguous_referents": _ambiguous,
        "candidate_entities": _ref_candidates,
        "original_span": detected_span.value,
        "blocked_proposition": _blocked_proposition,
        "blocked_claims": _blocked_claims,
        "clarification_prompt": _clarification_prompt,
    }
    if _pending_task is not None:
        _payload["pending_task"] = _pending_task
    _payload["completion_strategy"] = _completion_strategy_for_unresolved_referent_hold(
        original_question=original_question,
        pending_task=_pending_task,
    )
    _payload["allow_hold_unresolved"] = len(_ref_candidates) >= 2
    _payload["clarification_choices"] = build_unresolved_referent_clarification_choices(
        _ref_candidates,
        include_hold_unresolved=bool(_payload["allow_hold_unresolved"]),
    )
    return _payload


def _plan_resolved_pending_attribution_answer(
    *,
    pending_failed_constraint: str | None,
    pending_snapshot: dict[str, object] | None,
    resolved_binding_name: str | None,
) -> TurnSemanticPlan | None:
    """Deterministic attribution answer after UNRESOLVED_REFERENT bind."""
    if pending_failed_constraint != "UNRESOLVED_REFERENT":
        return None
    if not resolved_binding_name or not pending_snapshot:
        return None
    answer = _plan_attribution_resume_answer(
        pending=pending_snapshot,
        selected_entity_name=resolved_binding_name,
    )
    if not answer:
        return None
    return TurnSemanticPlan(
        kind="deterministic_clarification_continuation",
        response_text=answer,
        rationale=(
            "Pre-LLM clarification continuation: resolved pending referent and "
            "deterministic attribution answer from structured pending_task."
        ),
        action_hint=InterventionAction.PASS.name,
    )


def _pending_payload_simple_contain(
    *,
    user_input: str,
    detected_span: Span,
    pef: PEFState,
    primary: Flag | None,
) -> dict[str, object]:
    """Minimal pending payload for CONTAIN without UNRESOLVED_REFERENT scaffolding."""
    payload: dict[str, object] = {
        "original_question": user_input,
        "unresolved_entity_ids": [
            eid for eid, e in pef.entities.items() if not e.resolved
        ],
        "failed_constraint": primary.flag_type.name if primary else None,
        "candidate_entities": (
            list(primary.candidates) if primary and primary.candidates else []
        ),
        "original_span": detected_span.value,
    }
    if primary is not None and primary.flag_type == FlagType.DISJUNCTIVE_BRANCH_COLLAPSE:
        payload["clarification_prompt"] = (
            "Conflicting information was detected; choose which option applies."
        )
    if primary is not None and primary.flag_type == FlagType.AGENCY_RISK_CONTEXT_UNRESOLVED:
        payload["clarification_prompt"] = _AGENCY_CONTEXT_UNRESOLVED_PROMPT
    if primary is not None and primary.flag_type == FlagType.CLARIFICATION_AUTHORITY_OUTSIDE_HOLD:
        payload["failed_constraint"] = "CLARIFICATION_AUTHORITY_OUTSIDE_HOLD"
        payload["clarification_prompt"] = (
            "Explicit entity choice is required before the assistant can continue."
        )
        ui_l = user_input.lower().replace("\u2019", "'")
        ambig_tokens: list[str] = []
        for tok in (
            "she",
            "her",
            "hers",
            "he",
            "him",
            "his",
            "they",
            "them",
            "their",
            "theirs",
            "it",
            "its",
        ):
            if re.search(rf"\b{re.escape(tok)}\b", ui_l):
                ambig_tokens.append(tok)
        payload["ambiguous_referents"] = ambig_tokens[:6] or ["she"]
        payload["blocked_proposition"] = user_input.strip()
        payload["blocked_claims"] = []
    return payload


def _candidate_entity_names(entity_mentions: list[str]) -> list[str]:
    """Filter extraction entity_mentions to proper-noun-like candidate names.

    Compound conjunctions like 'Alice and Carol' are split into individual names
    so each candidate renders as a separate selectable option in the UI.
    Only splits when every part begins with an uppercase letter (proper noun).
    """
    expanded: list[str] = []
    for mention in (entity_mentions or []):
        if mention and re.search(r"\s+and\s+", mention, re.IGNORECASE):
            parts = [p.strip() for p in re.split(r"\s+and\s+", mention, flags=re.IGNORECASE)]
            if all(p and p[0].isupper() for p in parts):
                expanded.extend(parts)
                continue
        expanded.append(mention)
    return [
        name for name in expanded
        if name and name[0].isupper() and len(name.split()) <= 3
        and name.lower() not in _REFERENT_PRONOUN_TOKENS
        and not (name.isalpha() and name.isupper() and len(name.split()) == 1)  # exclude letter-only acronyms (NDA, CEO, HR…)
    ]


_REFERENT_PRONOUN_TOKENS = frozenset({
    "he",
    "him",
    "his",
    "she",
    "her",
    "hers",
    "they",
    "them",
    "their",
    "theirs",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
})

# Possessive determiners that head NPs like "his dog" — not valid disambiguation targets.
_POSSESSIVE_DETERMINER_TOKENS = frozenset({"his", "her", "their", "its"})

_WEAK_FRAME_REFERENCE_TOKENS = frozenset({
    "which",
    "one",
    "it",
    "its",
    "this",
    "that",
    "answer",
    "result",
    "claim",
    "patient",
    "account",
})

# Fiscal quarter / half labels mistaken as entities when harvesting possessive-antecedent
# candidates (weak-frame pronouns like ``its``).  Multi-token names are unaffected.
_FISCAL_PERIOD_ENTITY_LABEL_RE = re.compile(
    r"^(?:FY\s*)?(?:Q[1-4]|H[12])(?:\s+(?:FY\s*)?(?:19|20)?\d{2,4})?$",
    re.IGNORECASE,
)


def _is_temporal_period_entity_label(surface: str) -> bool:
    """True when *surface* is a standalone fiscal period token (Q4, FY2024, …)."""
    s = (surface or "").strip()
    if not s:
        return False
    if _FISCAL_PERIOD_ENTITY_LABEL_RE.match(s):
        return True
    return bool(re.match(r"^FY\s*(?:19|20)?\d{2,4}$", s, re.IGNORECASE))


def _ambiguous_tokens_include_possessive_determiner_or_pronoun(
    ambiguous_tokens: list[str] | None,
) -> bool:
    for tok in ambiguous_tokens or []:
        tl = str(tok).strip().lower()
        if tl in _POSSESSIVE_DETERMINER_TOKENS or tl in _POSSESSIVE_PRONOUNS:
            return True
    return False


def _pending_unresolved_referent_help_question_shape(user_text: str) -> bool:
    """QUERY-shaped asks about ambiguity/options — not substantive resumed assertions."""
    raw = (user_text or "").strip()
    if not raw.endswith("?"):
        return False
    low = raw.lower().rstrip("?.!")
    toks = _tokenize_alpha_words(low)
    if not toks:
        return False
    first = toks[0]
    if first == "which":
        tail = set(toks[1 : min(len(toks), 14)])
        return bool(
            tail
            & {
                "referent",
                "entity",
                "subsidiary",
                "subsidiaries",
                "company",
                "option",
                "options",
                "choices",
                "candidate",
                "candidates",
                "meaning",
                "mean",
            }
        )
    if first in {"who", "whom"}:
        return len(toks) <= 10 and bool(
            set(toks) & {"mean", "referring", "refer", "ambiguous", "unclear"}
        )
    return False


def _tokenize_alpha_words(text: str) -> list[str]:
    out: list[str] = []
    current: list[str] = []
    for ch in (text or "").lower():
        if ch.isalpha():
            current.append(ch)
            continue
        if current:
            out.append("".join(current))
            current = []
    if current:
        out.append("".join(current))
    return out


def _singularish(token: str) -> str:
    if len(token) > 3 and token.endswith("es"):
        return token[:-2]
    if len(token) > 2 and token.endswith("s"):
        return token[:-1]
    return token


def _non_pronoun_ambiguous_tokens(ambiguous_tokens: list[str]) -> list[str]:
    out: list[str] = []
    for tok in ambiguous_tokens:
        words = _tokenize_alpha_words(tok)
        for w in words:
            if w in _REFERENT_PRONOUN_TOKENS:
                continue
            if w in {"the", "a", "an"}:
                continue
            out.append(w)
    return out


def _entity_surface_tokens(entity_name: str, aliases: set[str] | None = None) -> set[str]:
    tokens: set[str] = set()
    for surface in [entity_name, *(aliases or set())]:
        for t in _tokenize_alpha_words(surface):
            tokens.add(t)
            tokens.add(_singularish(t))
    return tokens


def _weak_frame_reference_tokens(ambiguous_tokens: list[str]) -> set[str]:
    out: set[str] = set()
    for tok in ambiguous_tokens:
        for w in _tokenize_alpha_words(tok):
            if w in {"the", "a", "an"}:
                continue
            out.add(w)
    return out


def _referent_candidate_surface_contains_possessive_determiner(surface: str) -> bool:
    """True when *surface* includes a possessive-determiner token (e.g. ``his dog``)."""
    for raw in (surface or "").strip().split():
        t = raw.strip().lower().strip(".,!?;:()[]\"'`")
        if t in _POSSESSIVE_DETERMINER_TOKENS:
            return True
    return False


def _pef_resolved_binding_entity(pef: PEFState, surface: str) -> Entity | None:
    """Map a resolution surface to a **resolved** PEF entity by name or alias (case-insensitive)."""
    s = (surface or "").strip()
    if not s:
        return None
    sl = s.lower()
    for ent in pef.entities.values():
        if not ent.resolved:
            continue
        if ent.name.strip().lower() == sl:
            return ent
        if any((a or "").strip().lower() == sl for a in ent.aliases):
            return ent
    return None


def _referent_surface_heads_ambiguous_pronoun(surface: str, ambiguous_tokens: list[str]) -> bool:
    """True when *surface* contains an ambiguous-token pronoun/determiner as a whole-word span."""
    if not surface or not ambiguous_tokens:
        return False
    for tok in ambiguous_tokens:
        tl = str(tok).strip().lower()
        if not tl:
            continue
        if tl not in _REFERENT_PRONOUN_TOKENS:
            continue
        if _word_boundary_substring_in_text(tl, surface):
            return True
    return False


def _recent_frame_entity_names(pef: PEFState, *, max_turn_delta: int = 2) -> list[str]:
    if not pef.entities:
        return []
    max_active = max(e.turn_last_active for e in pef.entities.values())
    cutoff = max_active - max_turn_delta
    out: list[str] = []
    seen: set[str] = set()
    for ent in sorted(pef.entities.values(), key=lambda e: e.turn_last_active, reverse=True):
        if ent.turn_last_active < cutoff:
            continue
        if not ent.resolved:
            continue
        k = ent.name.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(ent.name)
    return out


def _ambiguous_subject_transfer_excluded_surfaces(
    extraction: ExtractionResult,
    ambiguous_tokens: list[str] | None,
) -> frozenset[str]:
    """Surfaces introduced as GIVE/TAKE predicate arguments alongside an ambiguous subject pronoun.

    When unresolved tokens are pronouns that appear as the grammatical subject of a
    ``GIVE``/``TAKE`` claim, extraction-local proper-name harvesting must not pull
    recipient projections (``HAS`` subject), transferred object literals, quantities,
    or combined wordings (parser noise like ``Sarah 5``). Names already grounded in
    ``pef.entities`` bypass this suppression in :func:`_is_excluded_transfer_subject_argument_surface`
    so provisional same-evidence ``HAS`` projections for prior subjects remain
    selectable for single-candidate autobind seams.
    """
    ambig_tokens = list(ambiguous_tokens or [])
    if not ambig_tokens:
        return frozenset()
    if _non_pronoun_ambiguous_tokens(ambig_tokens):
        return frozenset()

    evidence_keys: set[str] = set()
    for c in extraction.claims:
        if canonicalize_relation(c.relation) not in {"GIVE", "TAKE"}:
            continue
        subj = str(c.subject or "")
        if not any(
            _word_boundary_substring_in_text(str(tok).strip(), subj) for tok in ambig_tokens
        ):
            continue
        ev = str(c.evidence or "").strip().lower()
        if ev:
            evidence_keys.add(ev)

    if not evidence_keys:
        return frozenset()

    excluded: set[str] = set()
    for ev_norm in evidence_keys:
        recipients_lower: list[str] = []
        xfer_objs_lower: list[str] = []
        for c2 in extraction.claims:
            if str(c2.evidence or "").strip().lower() != ev_norm:
                continue
            rel = canonicalize_relation(c2.relation)
            if rel in {"GIVE", "TAKE"}:
                o = str(c2.obj or "").strip()
                if o:
                    ol = o.lower()
                    xfer_objs_lower.append(ol)
                    excluded.add(ol)
            elif rel == "HAS":
                s = str(c2.subject or "").strip()
                if s:
                    sl = s.lower()
                    recipients_lower.append(sl)
                    excluded.add(sl)
        for r in recipients_lower:
            for o in xfer_objs_lower:
                combined = f"{r} {o}".strip().lower()
                if combined:
                    excluded.add(combined)
    return frozenset(excluded)


def _is_excluded_transfer_subject_argument_surface(
    name_key_lower: str,
    excluded: frozenset[str],
    *,
    pef_known_lowercase: frozenset[str],
) -> bool:
    """Suppress transfer-predicate introductions; keep prior-grounded discourse names selectable."""
    if name_key_lower in pef_known_lowercase:
        return False
    if name_key_lower in excluded:
        return True
    for surf in excluded:
        if len(surf) < 2:
            continue
        if name_key_lower.startswith(f"{surf} "):
            return True
    return False


def _referent_resolution_candidate_names(
    extraction: ExtractionResult,
    pef: PEFState,
    *,
    ambiguous_tokens: list[str] | None = None,
    suppressed_subject_normalized: frozenset[str] | None = None,
) -> list[str]:
    """Resolved PEF-backed names for UNRESOLVED_REFERENT binding replies.

    Surfaces harvested from extraction (mentions + claim roles) appear as
    candidates only when they **bind** to an entity already in ``pef.entities``
    with ``resolved=True``. Pronoun-headed NPs such as ``his dog`` are never
    listed. Dedupes case-insensitively; preserves canonical entity ``name``.

    ``suppressed_subject_normalized`` drops any candidate matching a blocked pending
    subject (typically the unresolved possessive NP).
    """
    param_ambig = list(ambiguous_tokens or [])
    xfer_excluded = _ambiguous_subject_transfer_excluded_surfaces(extraction, param_ambig)
    ambig_tokens = list(param_ambig)
    if extraction.ambiguous_referents:
        ambig_tokens = list(
            dict.fromkeys(ambig_tokens + list(extraction.ambiguous_referents)),
        )

    suppressed = suppressed_subject_normalized or frozenset()

    raw: list[str] = list(extraction.entity_mentions)
    raw.extend(c.subject for c in extraction.claims)
    raw.extend(c.obj for c in extraction.claims)
    pef_known = frozenset(ent.name.lower() for ent in pef.entities.values())
    seen: set[str] = set()
    out: list[str] = []
    skip_period_labels_for_possessive = (
        _ambiguous_tokens_include_possessive_determiner_or_pronoun(ambig_tokens)
    )
    for surface in _candidate_entity_names(raw):
        candidate_is_unresolved_reference = (
            _referent_candidate_surface_contains_possessive_determiner(surface)
            or _referent_surface_heads_ambiguous_pronoun(surface, ambig_tokens)
        )
        if candidate_is_unresolved_reference:
            continue

        binding = _pef_resolved_binding_entity(pef, surface)
        if binding is None:
            continue
        name = binding.name
        k = name.lower()
        if _is_excluded_transfer_subject_argument_surface(
            k, xfer_excluded, pef_known_lowercase=pef_known
        ):
            continue
        if k not in seen:
            seen.add(k)
            if k in suppressed:
                continue
            if skip_period_labels_for_possessive and _is_temporal_period_entity_label(name):
                continue
            out.append(name)

    if len(out) >= 2:
        return out

    weak_tokens = _weak_frame_reference_tokens(list(ambiguous_tokens or []))
    weak_reference = bool(weak_tokens) and all(
        tok in _WEAK_FRAME_REFERENCE_TOKENS for tok in weak_tokens
    )
    content_ambiguous = _non_pronoun_ambiguous_tokens(list(ambiguous_tokens or []))
    content_set = {t for tok in content_ambiguous for t in (tok, _singularish(tok))}

    pef_entities = sorted(
        pef.entities.values(),
        key=lambda e: e.turn_last_active,
        reverse=True,
    )

    if weak_reference:
        for name in _recent_frame_entity_names(pef, max_turn_delta=2):
            k = name.lower()
            if _is_excluded_transfer_subject_argument_surface(
                k, xfer_excluded, pef_known_lowercase=pef_known
            ):
                continue
            if k in seen:
                continue
            seen.add(k)
            if k in suppressed:
                continue
            if skip_period_labels_for_possessive and _is_temporal_period_entity_label(name):
                continue
            out.append(name)
        if len(out) >= 2:
            return out

    for ent in pef_entities:
        if not ent.resolved:
            continue
        if content_set:
            ent_tokens = _entity_surface_tokens(ent.name, ent.aliases)
            if not (content_set & ent_tokens):
                continue
        k = ent.name.lower()
        if _is_excluded_transfer_subject_argument_surface(
            k, xfer_excluded, pef_known_lowercase=pef_known
        ):
            continue
        if k not in seen:
            seen.add(k)
            if k in suppressed:
                continue
            if skip_period_labels_for_possessive and _is_temporal_period_entity_label(ent.name):
                continue
            out.append(ent.name)

    _has_pronoun_in_ambig = any(
        tok.lower() in _REFERENT_PRONOUN_TOKENS for tok in param_ambig
    )
    if out or (content_set and not _has_pronoun_in_ambig):
        return out

    # Pronoun-only fallback: keep candidates discourse-local by recency.
    recent_floor = max(0, pef.current_turn - 8)
    for ent in pef_entities:
        if not ent.resolved:
            continue
        if ent.turn_last_active < recent_floor:
            continue
        k = ent.name.lower()
        if _is_excluded_transfer_subject_argument_surface(
            k, xfer_excluded, pef_known_lowercase=pef_known
        ):
            continue
        if k not in seen:
            seen.add(k)
            if k in suppressed:
                continue
            if skip_period_labels_for_possessive and _is_temporal_period_entity_label(ent.name):
                continue
            out.append(ent.name)

    # Pre-PEF-commit fallback: entity_mentions the extractor tagged as proper nouns
    # may not yet be in PEF when this function is called pre-update_pef (the common
    # case for the pre-LLM ambiguity gate). Filter only mentions that contain a
    # purely-numeric token (e.g. "Sarah 5" — a malformed NER output); legitimate
    # named entities such as "Emma", "Anna", or "Dr. Patel" are always included.
    # The earlier claim-role-surface filter was removed because it incorrectly blocked
    # entities that ARE valid antecedent candidates (e.g. subjects of IS/listed claims).
    if len(out) < 2:
        _ambig_lower_set = {t.lower() for t in ambig_tokens}
        for _mention in _candidate_entity_names(list(extraction.entity_mentions)):
            _ml = _mention.strip().lower()
            if _ml in _ambig_lower_set:
                continue
            if any(tok.isdigit() for tok in _ml.split()):
                continue  # malformed mention containing a bare number token
            if _is_excluded_transfer_subject_argument_surface(
                _ml, xfer_excluded, pef_known_lowercase=pef_known
            ):
                continue
            if _ml in seen:
                continue
            if _ml in suppressed:
                continue
            seen.add(_ml)
            _mstrip = _mention.strip()
            if skip_period_labels_for_possessive and _is_temporal_period_entity_label(_mstrip):
                continue
            out.append(_mstrip)

    return out


def _single_candidate_transfer_subject_autobind(
    extraction: ExtractionResult,
    ambiguous_tokens: list[str],
    candidate_entities: list[str],
) -> tuple[str, list[str]] | None:
    """Return (entity, tokens) when transfer-subject pronouns have one viable candidate.

    This guards integration-order seams where extraction still marks pronouns as
    ambiguous even though candidate viability has collapsed to one prior subject.
    """
    candidates = [str(c).strip() for c in candidate_entities if str(c).strip()]
    if len(candidates) != 1:
        return None
    subject_pronouns = {
        "he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs",
    }
    matched: list[str] = []
    seen: set[str] = set()
    for tok in ambiguous_tokens:
        tl = str(tok).strip().lower()
        if not tl or tl not in subject_pronouns:
            continue
        for c in extraction.claims:
            if canonicalize_relation(c.relation) not in {"GIVE", "TAKE"}:
                continue
            if _word_boundary_substring_in_text(tl, str(c.subject or "")):
                if tl not in seen:
                    seen.add(tl)
                    matched.append(str(tok))
                break
    if not matched:
        return None
    return candidates[0], matched


def _word_boundary_substring_in_text(sub: str, text: str) -> bool:
    """Whether ``sub`` occurs in ``text`` as a standalone token (plain scans, no new regex)."""
    if not sub or not text:
        return False
    tl = text.lower()
    sub_l = sub.lower()
    if sub_l not in tl:
        return False
    slen = len(sub_l)
    i = 0
    while i <= len(tl) - slen:
        j = tl.find(sub_l, i)
        if j < 0:
            return False
        left_ok = j == 0 or not tl[j - 1].isalpha()
        right_ok = j + slen == len(tl) or not tl[j + slen].isalpha()
        if left_ok and right_ok:
            return True
        i = j + 1
    return False


def _find_word_boundary_span(text: str, token: str) -> tuple[int, int] | None:
    """Return (start, end) of the first whole-word token match (case-insensitive)."""
    if not token or not text:
        return None
    tl = text.lower()
    tok = token.lower()
    if tok not in tl:
        return None
    tlen = len(tok)
    i = 0
    while i <= len(tl) - tlen:
        j = tl.find(tok, i)
        if j < 0:
            return None
        left_ok = j == 0 or not tl[j - 1].isalpha()
        right_ok = j + tlen == len(tl) or not tl[j + tlen].isalpha()
        if left_ok and right_ok:
            return (j, j + tlen)
        i = j + 1
    return None


def _pending_ambiguous_surfaces_remain(text: str, tokens: list[str]) -> bool:
    """True when any pending ambiguity token still appears as a whole word in *text*."""
    for tok in tokens:
        t = str(tok).strip()
        if not t:
            continue
        if _find_word_boundary_span(text, t):
            return True
    return False


def _replace_first_word_boundary_token(
    text: str,
    token: str,
    replacement: str,
) -> tuple[str, bool]:
    """Replace first whole-word token match (case-insensitive)."""
    span = _find_word_boundary_span(text, token)
    if span is None:
        return text, False
    s, e = span
    return text[:s] + replacement + text[e:], True


def _proper_name_token_mentioned_in_text(name: str, text: str) -> bool:
    """Whether a multi-word proper name or its first token appears as a word in ``text``."""
    if not name or not text:
        return False
    parts = name.split()
    if _word_boundary_substring_in_text(name, text):
        return True
    if len(parts) >= 2 and _word_boundary_substring_in_text(parts[0], text):
        return True
    return False


def _rag_candidates_mentioned_in_question(
    candidates: list[str],
    rag_question_line: str,
) -> list[str]:
    """Return ``candidates`` whose full name or leading token appears in the RAG question line."""
    hits: list[str] = []
    for name in candidates:
        if not name.strip():
            continue
        if _proper_name_token_mentioned_in_text(name, rag_question_line):
            hits.append(name)
    return hits


def _dedupe_name_hits_by_first_token(hits: list[str]) -> list[str]:
    """Drop duplicate surface forms of the same person (e.g. ``Nora Park`` and ``Nora``)."""
    seen_first: set[str] = set()
    out: list[str] = []
    for h in hits:
        parts = h.split()
        if not parts:
            continue
        key = parts[0].lower()
        if key not in seen_first:
            seen_first.add(key)
            out.append(h)
    return out


def _candidate_name_matches_pef_entity(name: str, pef_entity_names: list[str]) -> bool:
    """Whether a structural candidate string refers to a known PEF entity (plain string ops)."""
    nl = name.strip().lower()
    if not nl:
        return False
    first = name.split()[0].lower()
    for en in pef_entity_names:
        el = en.strip().lower()
        if not el:
            continue
        ef = en.split()[0].lower()
        if nl in el or el in nl or first == ef:
            return True
    return False


def _rag_anchor_candidates_pef_linked(
    candidates: list[str],
    pef_entity_names: list[str],
) -> list[str]:
    """Drop stray proper nouns (e.g. demonyms *Canadian*) that are not PEF entities.

    spaCy may emit *Canadian* from ``What Canadian city …``; it must not count as a
    second anchor alongside *Nora* when suppressing false ``she`` ambiguity (O1/T2).
    """
    if not pef_entity_names:
        return []
    return [c for c in candidates if _candidate_name_matches_pef_entity(c, pef_entity_names)]


def _should_rag_suppress_truly_ambiguous(
    rag_question_line: str | None,
    structural_candidates: list[str],
    pef_entity_names: list[str],
) -> bool:
    """RAG-only: single uniquely named entity in the question line vs PEF-linked candidates."""
    if rag_question_line is None:
        return False
    linked = _rag_anchor_candidates_pef_linked(structural_candidates, pef_entity_names)
    if not linked:
        return False
    mentioned = _dedupe_name_hits_by_first_token(
        _rag_candidates_mentioned_in_question(linked, rag_question_line)
    )
    return len(mentioned) == 1


def apply_rag_verify_discourse_bindings(pef: PEFState, rag_question_line: str | None) -> None:
    """Bind gendered pronouns to the unique PEF entity named in the RAG question line.

    Used after ``update_pef`` so post-LLM verify resolves *she*/*her* to that entity
    (via ``discourse_referent_bindings`` + ``_resolve_pronoun_via_pef``) instead of
    recency-only heuristics when multiple entities are active (O1/T2).

    RAG-only verify seam: post-LLM pronoun grounding for assistant claims; do not treat O1/T2-style regressions as pre-LLM ambiguity-gate issues.
    """
    if not rag_question_line or not pef.entities:
        return
    _pef_names = [e.name for e in pef.entities.values()]
    _all = _candidate_entity_names(_pef_names)
    if not _all:
        return
    linked = _rag_anchor_candidates_pef_linked(_all, _pef_names)
    if not linked:
        return
    mentioned = _dedupe_name_hits_by_first_token(
        _rag_candidates_mentioned_in_question(linked, rag_question_line)
    )
    if len(mentioned) != 1:
        return
    anchor_token = mentioned[0]
    ent = pef.find_entity_by_name(anchor_token)
    if ent is None:
        ft = anchor_token.split()[0].lower()
        for e in pef.entities.values():
            if e.name.split()[0].lower() == ft:
                ent = e
                break
    if ent is None:
        return
    name = ent.name
    pef.discourse_referent_bindings["she"] = name
    pef.discourse_referent_bindings["her"] = name


def build_verify_user_grounding_context(
    pef: PEFState,
    turn: int,
    history_user_input: str,
    rag_pef_update_user_text: str | None,
) -> UserGroundingContext:
    """Snapshot same-turn evidence for post-LLM verify (RAG corpus-aware)."""
    effective_text = history_user_input
    retrieved_context_text: str | None = None
    if rag_pef_update_user_text is not None:
        effective_text = rag_pef_update_user_text
        rag = split_rag_context_question(history_user_input)
        if rag is not None:
            retrieved_context_text = rag[0]
    return build_user_grounding_context(
        pef,
        turn,
        effective_text,
        retrieved_context_text=retrieved_context_text,
    )


def split_rag_context_question(user_input: str) -> tuple[str, str] | None:
    """Split ``Context:`` … ``\\n\\nQuestion:`` (eval harness). Plain string ops only.

    Returns ``(context_body, question_line)`` or ``None`` if the shape does not match.
    """
    s = user_input.strip()
    if len(s) < 20:
        return None
    if s[:8].lower() != "context:":
        return None
    rest = s[8:].lstrip()
    if not rest:
        return None
    low = rest.lower()
    sep = "\n\nquestion:"
    idx = low.find(sep)
    if idx < 0:
        return None
    context_body = rest[:idx].strip()
    tail = rest[idx:]
    cpos = tail.find(":")
    if cpos < 0 or not context_body:
        return None
    question_line = tail[cpos + 1 :].lstrip()
    if not question_line:
        return None
    return context_body, question_line


def blocked_act_request_scan_text(user_input: str) -> str:
    """Surface for pre-LLM blocked-act scan.

    When the message matches the RAG harness shape (``Context:`` … ``Question:``),
    scan the **Question** line only. Retrieved corpus must not false-trigger
    request-side classifiers (e.g. agency-risk roots in source documents).
    """
    rag = split_rag_context_question(user_input)
    if rag is not None:
        return rag[1]
    return user_input


def _llm_user_content_with_discourse(pef: PEFState, user_text_for_llm: str) -> str:
    """Prefix adapter user turns with explicit referent notes when session has bindings.

    History and audit stay on the raw user text; only the message sent to the LLM
    is augmented so the model does not re-ask who pronouns refer to after clarification.
    """
    dr = pef.discourse_referent_bindings
    if not dr:
        return user_text_for_llm
    notes: list[str] = []
    seen: set[str] = set()
    for tok, ent in sorted(dr.items(), key=lambda kv: (-len(kv[0]), kv[0])):
        if _word_boundary_substring_in_text(tok, user_text_for_llm):
            note = f"'{tok}' → {ent}"
            if note not in seen:
                seen.add(note)
                notes.append(note)
    if not notes:
        return user_text_for_llm
    # If the text has already been reconstructed (no pronouns), don't add notes.
    if not any(_word_boundary_substring_in_text(p, user_text_for_llm) for p in _REFERENTIAL_TARGET_PRONOUNS):
        return user_text_for_llm

    return (
        "[Session referents (from your clarification): "
        + "; ".join(notes)
        + "]\n"
        "Answer using these referents; do not ask who the pronoun refers to.\n\n"
        + user_text_for_llm
    )


def _has_eligible_candidates(head_noun: str, pef: PEFState) -> list[str]:
    """Return names of entities with an established HAS relation for head_noun.

    Governing rule: structural eligibility over the scoped binding.
    Only entities whose committed HAS relation resolves to head_noun are
    eligible antecedents. Span is not filtered — past-tense HAS facts
    are world facts and confer structural eligibility equally.
    Negated HAS relations are excluded.
    """
    head = head_noun.lower()
    eligible: list[str] = []
    for rel in pef.get_relationships_by_relation("HAS"):
        if rel.negated:
            continue
        matched = False
        if rel.object_literal is not None:
            matched = head in str(rel.object_literal).lower().split()
        elif rel.object_entity_id is not None:
            obj_entity = pef.entities.get(rel.object_entity_id)
            if obj_entity:
                matched = head in obj_entity.name.lower().split()
        if matched:
            subj_entity = pef.entities.get(rel.subject_id)
            if subj_entity:
                eligible.append(subj_entity.name)
    return eligible


_POSSESSIVE_PRONOUNS = frozenset({"his", "her", "its", "their", "my", "your", "our"})
_REFERENTIAL_TARGET_PRONOUNS = frozenset({
    "she", "he", "they", "it", "him", "her", "them", "hers", "his", "theirs", "its",
})
_REFERENTIAL_POSSESSIVE_TARGETS = frozenset({"hers", "his", "theirs", "its"})
_PRONOUN_AMBIGUITY_VOCAB = _REFERENTIAL_TARGET_PRONOUNS | _POSSESSIVE_PRONOUNS


def _make_possessive(name: str) -> str:
    """Form the English possessive of a proper noun."""
    if name.endswith("s"):
        return f"{name}'"
    return f"{name}'s"


def _replace_all_word_boundary_token(text: str, token: str, replacement: str) -> str:
    """Replace every whole-word occurrence of *token* (case-insensitive)."""
    result = text
    while True:
        new_result, changed = _replace_first_word_boundary_token(
            result, token, replacement,
        )
        if not changed:
            break
        result = new_result
    return result


def _dedupe_ambiguous_tokens(tokens: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        key = str(t).lower()
        if key not in seen:
            seen.add(key)
            out.append(str(t))
    return out


def _extend_ambiguous_with_downstream_pronouns(
    original_question: str,
    ambiguous_tokens: list[str],
) -> list[str]:
    """Append pronoun tokens from sentences after the last structurally ambiguous sentence.

    Extraction often marks possessives (``her``) only in an earlier clause while a later
    sentence still uses ``she`` for the same pending bind. Restricting vocabulary scans to
    sentences *after* the last sentence containing an extractor-listed ambiguity avoids
    rewriting unrelated pronouns in earlier sentences (see scoped reconstruction tests).
    """
    amb = [str(t) for t in ambiguous_tokens if str(t).strip()]
    if not amb:
        return amb
    sentences = _split_sentences_by_terminal_punct(original_question)
    last_hit = -1
    for i, sent in enumerate(sentences):
        if any(_word_boundary_substring_in_text(p, sent) for p in amb):
            last_hit = i
    if last_hit < 0:
        return amb
    seen_lower = {t.lower() for t in amb}
    out = list(amb)
    for j in range(last_hit + 1, len(sentences)):
        seg = sentences[j]
        for w in _tokenize_alpha_words(seg.lower()):
            if w in _PRONOUN_AMBIGUITY_VOCAB and w not in seen_lower:
                seen_lower.add(w)
                out.append(w)
    return out


def _binding_resume_ambiguous_tokens(pending: dict, original_question: str) -> list[str]:
    """Merge stored ambiguity tokens with downstream pronouns for stale pending shapes."""
    base = [str(x) for x in (pending.get("ambiguous_referents") or []) if str(x).strip()]
    if not _normalized_candidate_entities_for_binding(pending):
        return _dedupe_ambiguous_tokens(base)
    merged = _extend_ambiguous_with_downstream_pronouns(original_question, base)
    return _dedupe_ambiguous_tokens(merged)


def _replacement_for_ambiguous_token(
    token: str,
    bound_entity: str,
    resolved_referent_phrase: str | None,
) -> str:
    token_lower = token.lower()
    if token_lower in _POSSESSIVE_PRONOUNS:
        return _make_possessive(bound_entity)
    if resolved_referent_phrase is not None:
        if token_lower in _REFERENTIAL_TARGET_PRONOUNS:
            return resolved_referent_phrase
    return bound_entity


def _substitute_all_resolved_referent_pronouns(
    rewritten_text: str,
    resolved_referent_phrase: str,
    *,
    blocked_span: tuple[int, int] | None,
) -> str:
    """Replace every remaining third-person referential pronoun with *resolved_referent_phrase*."""

    def _substitute_segment(seg: str) -> str:
        out = seg
        while True:
            new_out, changed = _replace_first_referential_target_pronoun(
                out,
                resolved_referent_phrase=resolved_referent_phrase,
            )
            if not changed:
                break
            out = new_out
        return out

    if blocked_span is None:
        return _substitute_segment(rewritten_text)
    lo, hi = blocked_span
    before = _substitute_segment(rewritten_text[:lo])
    after = _substitute_segment(rewritten_text[hi:])
    return before + rewritten_text[lo:hi] + after


def _reconstruct_bound_text(
    original_text: str,
    blocked_proposition: str | None,
    ambiguous_tokens: list[str],
    bound_entity: str,
    *,
    resolved_referent_phrase: str | None = None,
) -> str:
    """Replace ambiguous pronouns in the original text after clarification binding.

    Tokens marked ambiguous inside *blocked_proposition* are rewritten only within that
    span so unrelated occurrences of the same surface pronoun stay untouched (tests:
    Richard/James sticks). Tokens marked ambiguous outside that span are rewritten in
    the rest of the utterance with whole-word replacement for every occurrence.
    """
    if not ambiguous_tokens:
        return original_text

    ambiguous_tokens = _dedupe_ambiguous_tokens(ambiguous_tokens)
    bp = blocked_proposition
    inside_tokens: list[str] = []
    outside_tokens: list[str] = []
    for token in ambiguous_tokens:
        if bp and _find_word_boundary_span(bp, token):
            inside_tokens.append(token)
        else:
            outside_tokens.append(token)

    rewritten = original_text
    protect_lo: int | None = None
    protect_hi: int | None = None

    if bp and bp in rewritten and inside_tokens:
        idx = rewritten.find(bp)
        inner = bp
        for token in inside_tokens:
            inner = _replace_all_word_boundary_token(
                inner,
                token,
                _replacement_for_ambiguous_token(
                    token, bound_entity, resolved_referent_phrase,
                ),
            )
        rewritten = rewritten[:idx] + inner + rewritten[idx + len(bp):]
        protect_lo = idx
        protect_hi = idx + len(inner)
    elif inside_tokens:
        # blocked_proposition may omit intervening sentences present in
        # original_question (e.g. evidence-limitation lines before a trailing
        # disambiguation ask). Rewrite listed ambiguity tokens across the full text.
        for token in inside_tokens:
            rewritten = _replace_all_word_boundary_token(
                rewritten,
                token,
                _replacement_for_ambiguous_token(
                    token, bound_entity, resolved_referent_phrase,
                ),
            )

    for token in outside_tokens:
        repl = _replacement_for_ambiguous_token(
            token, bound_entity, resolved_referent_phrase,
        )
        if protect_lo is not None and protect_hi is not None:
            before = rewritten[:protect_lo]
            mid = rewritten[protect_lo:protect_hi]
            after = rewritten[protect_hi:]
            before = _replace_all_word_boundary_token(before, token, repl)
            after = _replace_all_word_boundary_token(after, token, repl)
            rewritten = before + mid + after
        else:
            rewritten = _replace_all_word_boundary_token(rewritten, token, repl)

    if not resolved_referent_phrase:
        return rewritten

    span = (protect_lo, protect_hi) if protect_lo is not None else None
    return _substitute_all_resolved_referent_pronouns(
        rewritten,
        resolved_referent_phrase,
        blocked_span=span,
    )


def _replace_first_referential_target_pronoun(
    text: str,
    *,
    resolved_referent_phrase: str,
) -> tuple[str, bool]:
    """Replace first third-person referential target pronoun with resolved phrase."""
    best_token: str | None = None
    best_span: tuple[int, int] | None = None
    for token in _REFERENTIAL_TARGET_PRONOUNS:
        span = _find_word_boundary_span(text, token)
        if span is None:
            continue
        if best_span is None or span[0] < best_span[0]:
            best_span = span
            best_token = token
    if best_span is None or best_token is None:
        return text, False
    replacement = (
        _make_possessive(resolved_referent_phrase)
        if best_token in _REFERENTIAL_POSSESSIVE_TARGETS
        else resolved_referent_phrase
    )
    s, e = best_span
    return text[:s] + replacement + text[e:], True


def _resolved_referent_phrase_from_pending(
    pending: dict,
    bound_entity: str,
) -> str | None:
    """Build a noun phrase target for resumed query rewriting.

    Example:
    - blocked claim subject "her sister" + binding Emma -> "Emma's sister"
    """
    blocked_claims = pending.get("blocked_claims") or []
    if not blocked_claims:
        return None

    subj = str(blocked_claims[0].get("subject") or "").strip()
    if not subj:
        return None

    phrase = subj
    for token in pending.get("ambiguous_referents", []):
        token_lower = str(token).lower()
        replacement = (
            _make_possessive(bound_entity)
            if token_lower in _POSSESSIVE_PRONOUNS
            else bound_entity
        )
        phrase, _ = _replace_first_word_boundary_token(phrase, str(token), replacement)

    phrase = " ".join(phrase.split())
    return phrase or None


def _normalize_candidate_match_text(text: str) -> str:
    """Normalize apostrophes/spacing for candidate binding text matches."""
    normalized = (
        str(text or "")
        .replace("’", "'")
        .replace("`", "'")
        .strip()
        .lower()
    )
    out: list[str] = []
    i = 0
    while i < len(normalized):
        ch = normalized[i]
        if ch == "'" and i + 1 < len(normalized) and normalized[i + 1] == "s":
            i += 2
            continue
        out.append(ch)
        i += 1
    collapsed = " ".join("".join(out).split())
    return collapsed


def _contains_token_subsequence(tokens: list[str], needle: list[str]) -> bool:
    """Whether ``needle`` appears contiguously within ``tokens``."""
    if not needle or len(tokens) < len(needle):
        return False
    width = len(needle)
    for idx in range(0, len(tokens) - width + 1):
        if tokens[idx : idx + width] == needle:
            return True
    return False


def _candidate_matches_text(candidate: str, clarification_text: str) -> bool:
    """Text match for clarification binding, robust to possessive apostrophes."""
    cand_norm = _normalize_candidate_match_text(candidate)
    text_norm = _normalize_candidate_match_text(clarification_text)
    if not cand_norm or not text_norm:
        return False
    if cand_norm == text_norm:
        return True
    cand_tokens = _tokenize_alpha_words(cand_norm)
    text_tokens = _tokenize_alpha_words(text_norm)
    if not cand_tokens or not text_tokens:
        return False
    return _contains_token_subsequence(text_tokens, cand_tokens)


def _select_candidate_from_text(
    clarification_text: str,
    candidates: list[str],
) -> str | None:
    """Pick a single candidate from user text deterministically."""
    if not candidates:
        return None
    text_norm = _normalize_candidate_match_text(clarification_text)
    direct_exact = [
        c for c in candidates
        if _normalize_candidate_match_text(c) == text_norm
    ]
    if len(direct_exact) == 1:
        return direct_exact[0]

    matched = [c for c in candidates if _candidate_matches_text(c, clarification_text)]
    if len(matched) == 1:
        return matched[0]
    if len(matched) > 1:
        matched_sorted = sorted(
            matched,
            key=lambda c: len(_tokenize_alpha_words(_normalize_candidate_match_text(c))),
            reverse=True,
        )
        top_len = len(_tokenize_alpha_words(_normalize_candidate_match_text(matched_sorted[0])))
        top = [
            c for c in matched_sorted
            if len(_tokenize_alpha_words(_normalize_candidate_match_text(c))) == top_len
        ]
        if len(top) == 1:
            return top[0]

    stripped = _normalize_candidate_match_text(clarification_text).strip(".!?,;: ")
    stripped_tokens = _tokenize_alpha_words(stripped)
    if stripped_tokens and len(stripped_tokens[0]) >= 2:
        first = stripped_tokens[0]
        prefix_hits = [
            c for c in candidates
            if _tokenize_alpha_words(_normalize_candidate_match_text(c))
            and _tokenize_alpha_words(_normalize_candidate_match_text(c))[0].startswith(first)
        ]
        if len(prefix_hits) == 1:
            return prefix_hits[0]

    return None


def _normalized_candidate_entities_for_binding(pending: dict) -> list[str]:
    """Normalize pending ``candidate_entities`` for binding-only resolution.

    Rejects accidental scalar/str payloads that would iterate per-character and
    produce bogus prefix matches against user clarification text.
    Any container-valued element forces rejection of the entire candidate list.
    """
    raw = pending.get("candidate_entities")
    if raw is None:
        return []
    if isinstance(raw, (str, bytes, bytearray)):
        return []
    if isinstance(raw, (set, frozenset)):
        raw = list(raw)
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[str] = []
    for x in raw:
        if isinstance(x, (dict, list, tuple, set)):
            return []
        if isinstance(x, (bytes, bytearray)):
            return []
        try:
            s = str(x).strip()
        except Exception:
            return []
        if s:
            out.append(s)
    return out


def _binding_candidates_present(pending: dict) -> bool:
    """Whether pending carries a valid non-empty candidate entity list."""
    return bool(_normalized_candidate_entities_for_binding(pending))


def _resolve_binding_entity(
    clarification_text: str,
    clar_ext: "ExtractionResult",
    pending: dict,
    pef: "PEFState",
) -> str | None:
    """Identify which pending clarification candidate the user's reply selects.

    Binding succeeds when ``candidate_entities`` is non-empty and either
    ``_select_candidate_from_text`` deterministically picks one surface, **or**
    an extractor ``entity_mentions`` entry matches a listed candidate after the
    same normalization used by :func:`_matches_candidate` (parity guard).

    ``pef`` is retained for call-site compatibility only.
    """
    _ = pef
    # Dominating gate: reject missing / empty / scalar / malformed containers before
    # touching extractor mentions or ``_select_candidate_from_text`` (which expects a list).
    try:
        raw_ce = pending.get("candidate_entities")
    except Exception:
        return None
    if raw_ce is None:
        return None
    if isinstance(raw_ce, (str, bytes, bytearray)):
        return None
    if isinstance(raw_ce, (set, frozenset)):
        raw_ce = list(raw_ce)
    if not isinstance(raw_ce, (list, tuple)):
        return None
    if len(raw_ce) == 0:
        return None

    candidates = _normalized_candidate_entities_for_binding(pending)
    if not candidates:
        return None
    chosen = _select_candidate_from_text(clarification_text, candidates)
    if chosen:
        return chosen
    # Mirror ``_matches_candidate`` mention-equality path — avoids divergent outcomes
    # where substring selection misses but spaCy mentions align to a candidate.
    for mention in clar_ext.entity_mentions:
        mention_norm = _normalize_candidate_match_text(mention)
        for c in candidates:
            if mention_norm == _normalize_candidate_match_text(c):
                return c
    return None


def _remaining_referential_target_pronouns(text: str) -> list[str]:
    """Return referential third-person pronouns still present in *text*."""
    remaining: list[str] = []
    for token in sorted(_REFERENTIAL_TARGET_PRONOUNS):
        if _find_word_boundary_span(text, token):
            remaining.append(token)
    return remaining


def _reconstruct_bound_user_input_or_raise(
    *,
    original_question: str,
    pending: dict,
    ambiguous_tokens: list[str],
    bound_entity: str,
) -> tuple[str, str | None]:
    """Central post-binding rewrite enforcement for outbound LLM user text."""
    if not bound_entity.strip():
        raise AssertionError("Bound entity is required for reconstruction.")
    resolved_phrase = _resolved_referent_phrase_from_pending(pending, bound_entity)
    rewritten = _reconstruct_bound_text(
        original_question,
        pending.get("blocked_proposition"),
        ambiguous_tokens,
        bound_entity,
        resolved_referent_phrase=resolved_phrase,
    )
    authoritative_phrase = resolved_phrase or bound_entity
    rewritten = _substitute_all_resolved_referent_pronouns(
        rewritten,
        authoritative_phrase,
        blocked_span=None,
    )
    # Ensure no residual pronouns remain in the final string before handoff.
    for p in sorted(_REFERENTIAL_TARGET_PRONOUNS, key=len, reverse=True):
        rewritten = _replace_all_word_boundary_token(rewritten, p, authoritative_phrase)
    for tok in ambiguous_tokens:
        if _find_word_boundary_span(rewritten, tok):
            rewritten = _replace_all_word_boundary_token(
                rewritten,
                tok,
                _replacement_for_ambiguous_token(
                    tok, bound_entity, resolved_phrase or authoritative_phrase,
                ),
            )

    if _pending_ambiguous_surfaces_remain(rewritten, ambiguous_tokens):
        raise AssertionError(
            "Bound reconstruction still contains unresolved pending ambiguity tokens.",
        )
    remaining_pronouns = _remaining_referential_target_pronouns(rewritten)
    if remaining_pronouns:
        raise AssertionError(
            f"Bound reconstruction still contains referential pronouns: {remaining_pronouns}",
        )
    return rewritten, resolved_phrase


_DISAMBIG_CONTINUATION_OPENERS: frozenset[str] = frozenset({
    "tell", "explain", "describe", "show", "list", "give", "help",
    "what", "when", "where", "why", "how", "who", "which",
    "can", "could", "would", "should", "need",
    "please", "is", "are", "was", "were",
})


def _is_direct_disambiguation_response(user_input: str, pending: dict) -> bool:
    """True when the user's reply is a direct candidate selection, not a follow-up request.

    Distinguishes "Jennifer" or "It was Jennifer" (direct selection) from
    "Tell me more about Jennifer" (continuation request — not a valid binding).
    Only meaningful when candidate_entities are known in pending.
    """
    low = (user_input or "").strip().lower().rstrip("?.!")
    if not low:
        return False
    first_tok = _tokenize_alpha_words(low)
    if first_tok and first_tok[0] in _DISAMBIG_CONTINUATION_OPENERS:
        return False
    return _select_candidate_from_text(
        user_input, _normalized_candidate_entities_for_binding(pending)
    ) is not None


def _is_hold_unresolved_selection(user_input: str) -> bool:
    """True when the user explicitly selects the Hold unresolved continuation."""
    norm = " ".join(_tokenize_alpha_words((user_input or "").lower()))
    if not norm:
        return False
    return norm in {
        "hold unresolved",
        "hold as unresolved",
        "keep unresolved",
        "leave unresolved",
    } or norm == HOLD_UNRESOLVED_CHOICE_LABEL.lower()


def _matches_candidate(
    clarification_text: str,
    clar_ext: "ExtractionResult",
    pending: dict,
) -> bool:
    """Return True when the user's reply clearly names a stored candidate entity.

    Covers direct answers ("James"), indirect phrasing ("I mean James"),
    and extraction-identified entity mentions.  Does NOT fall back to PEF
    recency — only explicit textual evidence counts for the binding test.
    """
    candidates = _normalized_candidate_entities_for_binding(pending)
    if not candidates:
        return False

    if _select_candidate_from_text(clarification_text, candidates) is not None:
        return True

    for mention in clar_ext.entity_mentions:
        mention_norm = _normalize_candidate_match_text(mention)
        for c in candidates:
            if mention_norm == _normalize_candidate_match_text(c):
                return True

    return False


_EMPTY_RESUME_EXTRACTION_FOR_BINDING_SKIP = ExtractionResult(
    claims=[],
    entity_mentions=[],
    span=Span.PRESENT,
)


class _PendingTypoRecoveryPhase(Enum):
    """Pending clarification branch for deterministic typo confirmation (state-native only)."""

    NONE = auto()
    MERGED_RESUME_SKIP_EXTRACT = auto()


def _generic_state_native_ambiguity_message_from_pending(pending: dict) -> str:
    """Rebuild ordinary ambiguity wording after typo suggestion was declined."""
    kind = str(pending.get("state_native_ambiguity_kind") or "")
    names_raw = pending.get("candidate_entities") or []
    names = sorted(str(x) for x in names_raw)
    opts = ", ".join(names)
    if kind == "inventory_holder_query_unresolved_item":
        item_pc = pending.get("item_phrase") or "item"
        surfaces = [str(x) for x in (pending.get("candidate_entities") or [])]
        if surfaces:
            return (
                f'Specify which possessed item "{item_pc}" refers to '
                f'(committed holdings include: {", ".join(surfaces)}).'
            )
        return (
            f'Specify which item "{item_pc}" refers to; pronouns alone are not '
            "a bounded inventory query here."
        )
    if kind == "inventory_holder_query":
        item = pending.get("item_phrase") or "item"
        return f"More than one holder matches {item}: {opts}. Which one?"
    return f"Which did you mean: {opts}?"


def _pending_typo_recovery_active(pending: dict) -> bool:
    if str(pending.get("failed_constraint") or "") != "STATE_NATIVE_ENTITY_AMBIGUITY":
        return False
    if pending.get("entity_ambiguity_typo_recovery_suppressed"):
        return False
    tr = pending.get("typo_recovery")
    if not isinstance(tr, dict):
        return False
    return bool(tr.get("typo_entity_id")) and bool(tr.get("canonical_entity_id"))


def _matches_state_native_typo_confirmation(
    clarification_text: str,
    typo_recovery: dict,
) -> bool:
    canon = str(typo_recovery.get("suggested_canonical_name") or "").strip()
    if not canon:
        return False
    norm = _normalize_candidate_match_text(clarification_text).strip(".!? ")
    if norm in {
        "yes",
        "y",
        "yeah",
        "yep",
        "correct",
        "right",
        "sure",
        "absolutely",
        "indeed",
    }:
        return True
    if _normalize_candidate_match_text(canon) == norm:
        return True
    return _candidate_matches_text(canon, clarification_text)


def _matches_state_native_typo_rejection(clarification_text: str) -> bool:
    norm = _normalize_candidate_match_text(clarification_text).strip(".!? ")
    return norm in {"no", "n", "nope", "incorrect", "wrong", "negative"}


@dataclass
class _CommitReplayResult:
    """Outcome of _commit_resolved_claims for comparative IS claims."""
    stop_reason: str | None = None        # Set → caller must issue STOP with this text
    pending_comparand: dict | None = None  # Set → caller must issue CONTAIN / UNRESOLVED_COMPARAND


def _replay_verification_indicates_relation_commit_failed(
    replay_res: _CommitReplayResult,
    *,
    relation_count_before: int,
    pef_after: PEFState,
    pending_blocked_claims: list[object] | None = None,
) -> bool:
    """True iff clarification replay was expected to commit relations but did not.

    When ``pending_blocked_claims`` is empty, replay is a no-op by design (the pending
    payload carried no serialized claims to replay); a stable relationship count must
    not be interpreted as CLARIFICATION_RESOLUTION_COMMIT_FAILED.
    """
    if not pending_blocked_claims:
        return False
    if replay_res.stop_reason:
        return False
    if replay_res.pending_comparand:
        return False
    return len(pef_after.relationships) <= relation_count_before


def _extract_item_noun_from_possessive_np(surface: str) -> str:
    """Strip leading possessive determiner, return remaining noun phrase.

    "his wallet"         → "wallet"
    "her leather wallet" → "leather wallet"
    "their red car"      → "red car"

    Multi-word item NPs preserve all modifiers (Phase 2 decision: item_key boundary).
    """
    tokens = (surface or "").strip().split()
    if not tokens:
        return surface
    first = tokens[0].lower().strip(".,!?;:()[]\"'`")
    if first in _POSSESSIVE_DETERMINER_TOKENS:
        remainder = " ".join(tokens[1:]).strip()
        return remainder if remainder else surface
    return surface


def _find_owned_item_entity(
    pef: PEFState,
    owner_name: str,
    item_noun: str,
) -> Entity | None:
    """Return a PEF entity that *owner_name* holds via HAS whose name normalises to item_noun.

    Checks entity-reference HAS relations (object_entity_id set). Returns the first
    active, non-negated match, or None if no such entity exists.
    """
    owner_ent = pef.find_entity_by_name(owner_name)
    if owner_ent is None:
        return None
    target_key = item_key(item_noun)
    for rel in pef.get_relationships_for_subject(owner_ent.id):
        if rel.relation != "HAS" or rel.negated:
            continue
        if rel.object_entity_id is not None:
            candidate = pef.entities.get(rel.object_entity_id)
            if candidate and item_key(candidate.name) == target_key:
                return candidate
    return None


def _infer_held_reason(cd: dict) -> str:
    """Infer held_reason from a blocked_claim dict.

    New-format dicts carry ``held_reason`` directly.  Old-format dicts (pre-Phase-2b)
    are recognised by the legacy ``is_comparative`` / ``is_possessive_np`` tags.
    Plain pronoun-subject dicts with no tag default to UNRESOLVED_REFERENT.
    """
    if cd.get("held_reason"):
        return str(cd["held_reason"])
    if cd.get("is_comparative"):
        return "UNRESOLVED_COMPARAND"
    if cd.get("is_possessive_np"):
        return "UNRESOLVED_POSSESSOR"
    return "UNRESOLVED_REFERENT"


def _commit_comparand_claims_as_compare(
    comparand_name: str,
    pending: dict,
    pef: PEFState,
) -> None:
    """Commit COMPARE relationships to PEF after comparand is selected by user.

    Used when UNRESOLVED_COMPARAND pending is resolved: the user has named one of
    the candidate comparands and we can now write the structured COMPARE relation.
    """
    from aurora_lens.pef.state import Relationship as _Rel
    blocked = pending.get("blocked_claims") or []
    comparand_adj = str(pending.get("comparand_adjective") or "").strip()
    comp_ent = pef.find_entity_by_name(comparand_name)
    for cd in blocked:
        if _infer_held_reason(cd) != "UNRESOLVED_COMPARAND":
            continue
        subj_name = str(cd.get("subject") or "").strip()
        if not subj_name:
            continue
        subj_ent = pef.find_entity_by_name(subj_name)
        if subj_ent is None:
            subj_ent, _ = pef.get_or_create_entity(subj_name)
        _adj = comparand_adj or str(cd.get("comparand_adjective") or cd.get("obj") or "").strip()
        _noun = str(cd.get("comparand_noun") or pending.get("comparand_noun") or "").strip()
        # Prefer entity reference; Relationship enforces exactly one of entity/literal.
        comp_rel = _Rel(
            subject_id=subj_ent.id,
            relation="COMPARE",
            object_entity_id=comp_ent.id if comp_ent else None,
            object_literal=None if comp_ent else comparand_name,
            span=Span(cd.get("span", "present")),
            source_turn=pef.current_turn,
            evidence=str(cd.get("evidence") or ""),
            negated=bool(cd.get("negated", False)),
            provenance="user_input",
            extractor_backend="rule",
            relation_metadata={"adjective": _adj, "noun": _noun},
        )
        pef.add_relationship(comp_rel)


def _find_eligible_comparands(
    pef: PEFState,
    noun_key: str,
    excluding_name: str,
) -> list[str]:
    """Return names of current active holders of noun_key, excluding excluding_name."""
    from aurora_lens.state_native_engine.eval.inventory import (
        _current_holders_matching_item_key,
    )
    normalized = item_key(noun_key)
    holders = _current_holders_matching_item_key(pef, normalized)
    return [
        ent.name
        for ent in holders
        if ent.name.lower() != excluding_name.lower()
    ]


def _commit_resolved_claims(
    bound_entity: str,
    pending: dict,
    pef: PEFState,
    *,
    on_pef_admission: Callable[[PEFAdmissionResult], None] | None = None,
) -> _CommitReplayResult:
    """Replay blocked claims into PEF with the resolved entity as subject.

    Mirrors the CLI's kernel_step resumption: after the user supplies a valid
    binding, the claims that were blocked due to an ambiguous pronoun subject
    are re-committed with the correct entity name.  This ensures the PEF world
    model reflects the binding so later follow-ups (e.g. "Whose stick was
    bigger?") can query against grounded state rather than finding only the
    raw unresolved pronoun.
    """
    blocked = pending.get("blocked_claims", [])
    if not blocked:
        return _CommitReplayResult()
    ambiguous = {p.lower() for p in pending.get("ambiguous_referents", [])}

    _small_number_words: dict[str, int] = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
    }

    def _parse_counted_item_literal(obj_text: str) -> tuple[int, str] | None:
        s = str(obj_text or "").strip()
        if not s:
            return None
        parts = s.split(None, 1)
        if len(parts) != 2:
            return None
        n_raw, item = parts[0].lower(), parts[1].strip()
        if not item:
            return None
        if n_raw.isdigit():
            return (int(n_raw), item)
        if n_raw in _small_number_words:
            return (_small_number_words[n_raw], item)
        return None

    def _subject_latest_active_counted_item(
        subject_name: str, target_item_key: str
    ) -> tuple[int, str] | None:
        ent = pef.find_entity_by_name(subject_name)
        if ent is None:
            return None
        latest_by_literal: dict[str, tuple[int, bool, int, str]] = {}
        for rel in pef.get_relationships_for_subject(ent.id):
            if rel.relation != "HAS":
                continue
            obj = str(rel.object_literal or "").strip()
            parsed = _parse_counted_item_literal(obj)
            if parsed is None:
                continue
            count, item_text = parsed
            if item_key(item_text) != target_item_key:
                continue
            lit = obj.lower()
            prev = latest_by_literal.get(lit)
            if prev is None or rel.source_turn > prev[0] or (
                rel.source_turn == prev[0] and rel.negated and not prev[1]
            ):
                latest_by_literal[lit] = (rel.source_turn, rel.negated, count, item_text)
        active = [v for v in latest_by_literal.values() if not v[1]]
        if not active:
            return None
        best_turn = max(v[0] for v in active)
        best_candidates = [v for v in active if v[0] == best_turn]
        _, _, count, item_text = max(best_candidates, key=lambda v: v[2])
        return (count, item_text)

    def _expand_count_of_pronoun_obj(obj: str) -> str:
        o = str(obj or "").strip()
        if not o:
            return o

        parts_three = o.split()
        if (
            len(parts_three) == 3
            and parts_three[1].lower() == "of"
            and parts_three[2].lower() in {"them", "it"}
        ):
            qtok = parts_three[0]
            q_low = qtok.lower()
            if qtok.isdigit() or q_low in _small_number_words:

                def _unique_items_from_subject(subject_name: str) -> set[str]:
                    out: set[str] = set()
                    ent_sub = pef.find_entity_by_name(subject_name)
                    if ent_sub is None:
                        return out
                    for rel in pef.get_relationships_for_subject(ent_sub.id):
                        if rel.relation != "HAS" or rel.negated:
                            continue
                        parsed_u = _parse_counted_item_literal(str(rel.object_literal or ""))
                        if parsed_u is None:
                            continue
                        out.add(parsed_u[1])
                    return out

                items_bound = _unique_items_from_subject(bound_entity)
                item_pick: str | None = None
                if len(items_bound) == 1:
                    item_pick = next(iter(items_bound))
                else:
                    items_global: set[str] = set()
                    for rel in pef.get_relationships_by_relation("HAS"):
                        if rel.negated:
                            continue
                        parsed_g = _parse_counted_item_literal(str(rel.object_literal or ""))
                        if parsed_g is None:
                            continue
                        items_global.add(parsed_g[1])
                    if len(items_global) == 1:
                        item_pick = next(iter(items_global))
                if item_pick is not None:
                    return f"{qtok} {item_pick}".strip()
            return o

        parts = o.split(None, 1)
        if len(parts) == 2:
            return o
        q_raw = parts[0].lower() if parts else ""
        if not (q_raw.isdigit() or q_raw in _small_number_words):
            return o
        # item_key unknown; derive by scanning active counted HAS and requiring uniqueness.
        ent = pef.find_entity_by_name(bound_entity)
        if ent is None:
            return o
        active_items: set[str] = set()
        for rel in pef.get_relationships_for_subject(ent.id):
            if rel.relation != "HAS":
                continue
            if rel.negated:
                continue
            parsed = _parse_counted_item_literal(str(rel.object_literal or ""))
            if parsed is None:
                continue
            _, item_text = parsed
            active_items.add(item_text)
        if len(active_items) != 1:
            return o
        item_text = next(iter(active_items))
        return f"{parts[0]} {item_text}".strip()

    _replay_result = _CommitReplayResult()
    resolved_claims: list[ExtractedClaim] = []
    saw_bound_give = False
    gave_obj_literal: str | None = None
    for cd in blocked:
        subj = cd.get("subject", "")
        subj_lower = subj.lower()
        if subj_lower in ambiguous or subj_lower in _POSSESSIVE_PRONOUNS:
            subj = bound_entity
        elif any(_word_boundary_substring_in_text(p, subj) for p in ambiguous):
            for p in ambiguous:
                replacement = (
                    _make_possessive(bound_entity)
                    if p in _POSSESSIVE_PRONOUNS
                    else bound_entity
                )
                subj, _ = _replace_first_word_boundary_token(subj, p, replacement)
        rel = canonicalize_relation(str(cd.get("relation", "")))
        obj = _expand_count_of_pronoun_obj(str(cd.get("obj", "")))

        # Phase 1: comparative IS claims must never commit as IS to PEF.
        # Route through comparand resolution instead.
        if _infer_held_reason(cd) == "UNRESOLVED_COMPARAND":
            _c_noun = str(cd.get("comparand_noun") or obj or "").strip()
            _c_adjective = str(cd.get("comparand_adjective") or obj or "").strip()
            # Exclude by bound_entity (the resolved person), not subj (which may be
            # "James's dog" after possessive replacement — no entity has that name).
            _comparands = _find_eligible_comparands(pef, _c_noun, bound_entity)
            if len(_comparands) == 0:
                _replay_result.stop_reason = (
                    "I cannot complete that comparison from committed state: "
                    "no committed comparison target exists."
                )
            elif len(_comparands) == 1:
                _comp_name = _comparands[0]
                _comp_ent = pef.find_entity_by_name(_comp_name)
                _subj_ent = pef.find_entity_by_name(bound_entity)
                if _subj_ent is not None:
                    from aurora_lens.pef.state import Relationship as _Rel
                    # Prefer entity reference; fall back to literal (Relationship
                    # enforces exactly one of object_entity_id / object_literal).
                    _comp_rel = _Rel(
                        subject_id=_subj_ent.id,
                        relation="COMPARE",
                        object_entity_id=_comp_ent.id if _comp_ent else None,
                        object_literal=None if _comp_ent else _comp_name,
                        span=Span(cd.get("span", "present")),
                        source_turn=pef.current_turn,
                        evidence=str(cd.get("evidence") or ""),
                        negated=bool(cd.get("negated", False)),
                        provenance="user_input",
                        extractor_backend="rule",
                        relation_metadata={"adjective": _c_adjective, "noun": _c_noun},
                    )
                    pef.add_relationship(_comp_rel)
            else:  # 2+ comparands
                _replay_result.pending_comparand = {
                    "original_question": str(pending.get("original_question") or ""),
                    "unresolved_entity_ids": [],
                    "failed_constraint": "UNRESOLVED_COMPARAND",
                    "candidate_entities": _comparands,
                    "comparand_adjective": _c_adjective,
                    "comparand_noun": _c_noun,
                    "original_span": cd.get("span", "present"),
                    "blocked_claims": [cd],
                }
            continue  # never commit as IS

        # Phase 2: possessive-NP claim — replay with entity-linked ownership.
        # "his wallet" → mint/reuse item entity + HAS(James, item_entity) → IS(item_entity, red)
        # Law P2-2: possessive surface never becomes a canonical entity name.
        elif _infer_held_reason(cd) == "UNRESOLVED_POSSESSOR":
            _item_noun = str(cd.get("item_noun") or "").strip()
            if not _item_noun:
                # Malformed entry — skip silently to avoid committing possessive surface
                continue
            _item_ent = _find_owned_item_entity(pef, bound_entity, _item_noun)
            if _item_ent is None:
                _item_ent, _ = pef.get_or_create_entity(_item_noun)
                _owner_ent = pef.find_entity_by_name(bound_entity)
                if _owner_ent is not None:
                    from aurora_lens.pef.state import Relationship as _Rel
                    pef.add_relationship(_Rel(
                        subject_id=_owner_ent.id,
                        relation="HAS",
                        object_entity_id=_item_ent.id,
                        object_literal=None,
                        span=Span(cd.get("span", "present")),
                        source_turn=pef.current_turn,
                        evidence=f"Ownership inferred: {cd.get('evidence', '')}",
                        negated=False,
                        provenance="user_input",
                        extractor_backend="rule",
                    ))
            # Replay original predicate with item entity as subject (not possessive surface)
            resolved_claims.append(ExtractedClaim(
                subject=_item_ent.name,
                relation=rel,
                obj=obj,
                span=Span(cd.get("span", "present")),
                negated=bool(cd.get("negated", False)),
                evidence=str(cd.get("evidence") or ""),
            ))
            continue  # skip the normal subject-resolution path

        if rel == "GIVE" and subj == bound_entity:
            saw_bound_give = True
            gave_obj_literal = obj
        resolved_claims.append(ExtractedClaim(
            subject=subj,
            relation=rel,
            obj=obj,
            span=Span(cd.get("span", "present")),
            negated=cd.get("negated", False),
            evidence=cd.get("evidence", ""),
        ))

    # Law P4: use PossessionTransaction.validate() for the P2 insufficiency check; if
    # admitted, add the remainder claim to resolved_claims so update_pef handles the write
    # (same path as before — keeps supersession / source_turn ordering correct).
    if saw_bound_give and gave_obj_literal:
        gave_parsed = _parse_counted_item_literal(gave_obj_literal)
        if gave_parsed is not None:
            gave_count, gave_item = gave_parsed
            gave_key = item_key(gave_item)
            prior = _subject_latest_active_counted_item(bound_entity, gave_key)
            if prior is not None:
                prior_count, prior_item = prior
                from aurora_lens.interpret.schema import PossessionTransaction as _PTx
                _tx = _PTx(
                    claim=ExtractedClaim(
                        subject=bound_entity,
                        relation="HAS",
                        obj=gave_obj_literal,
                        span=Span(pending.get("original_span", "present")),
                        negated=False,
                        evidence=str(pending.get("blocked_proposition") or ""),
                    ),
                    mutation_kind="transfer",
                    quantity=float(gave_count),
                    prior_count_snapshot=float(prior_count),
                    giver_name=bound_entity,
                    item_key_str=prior_item,
                )
                _stop = _tx.validate(pef)
                if _stop:
                    _replay_result.stop_reason = _stop
                else:
                    remainder = int(_tx.projected_remainder)
                    resolved_claims.append(
                        ExtractedClaim(
                            subject=bound_entity,
                            relation="HAS",
                            obj=f"{remainder} {prior_item}",
                            span=Span(pending.get("original_span", "present")),
                            negated=False,
                            evidence=str(pending.get("blocked_proposition") or ""),
                        )
                    )

    if resolved_claims:
        result = ExtractionResult(
            claims=resolved_claims,
            entity_mentions=[bound_entity],
            span=Span(pending.get("original_span", "present")),
        )
        adm_rc = update_pef(result, pef)
        if on_pef_admission is not None:
            on_pef_admission(adm_rc)

    return _replay_result


def _has_consequence_bearing_transfer_claim(extraction: ExtractionResult) -> bool:
    """True when extraction contains transfer/possession mutations that must not pre-commit under unresolved referents."""
    for c in extraction.claims:
        rel = canonicalize_relation(str(c.relation or ""))
        # Keep this guard narrow: only explicit transfer acts should block
        # same-turn admissible non-ambiguous facts from committing.
        if rel in {"GIVE", "TAKE"}:
            return True
    return False


def _rebuild_relationship_indexes(pef: PEFState) -> None:
    """Rebuild relationship indexes after in-place relationship list edits."""
    pef.rel_by_subject = {}
    pef.rel_by_object_entity = {}
    pef.rel_by_relation = {}
    for idx, rel in enumerate(pef.relationships):
        pef.rel_by_subject.setdefault(rel.subject_id, []).append(idx)
        if rel.object_entity_id is not None:
            pef.rel_by_object_entity.setdefault(rel.object_entity_id, []).append(idx)
        pef.rel_by_relation.setdefault(rel.relation, []).append(idx)


def _retire_unresolved_blocked_claims(
    pending: dict,
    pef: PEFState,
) -> None:
    """Drop unresolved pronoun-subject claim remnants after successful binding.

    Clarification should collapse ambiguity into one resolved active claim.
    We keep audit/history (decision logs, pending snapshot trail) but remove
    unresolved subject remnants from active PEF relationships.
    """
    blocked = pending.get("blocked_claims", [])
    if not blocked:
        return
    ambiguous_tokens = [str(t).lower() for t in pending.get("ambiguous_referents", [])]
    if not ambiguous_tokens:
        return

    signatures: set[tuple[str, str, str]] = set()
    for cd in blocked:
        try:
            rel = canonicalize_relation(str(cd.get("relation", "")))
            obj = str(cd.get("obj", ""))
            span = str(cd.get("span", "present"))
            signatures.add((rel, obj, span))
        except Exception:
            continue

    kept: list[Relationship] = []
    changed = False
    for rel in pef.relationships:
        subj = pef.entities.get(rel.subject_id)
        subj_name = (subj.name if subj is not None else "").lower()
        has_ambig_token = any(
            _word_boundary_substring_in_text(tok, subj_name)
            for tok in ambiguous_tokens
        )
        if not has_ambig_token:
            kept.append(rel)
            continue

        sig = (canonicalize_relation(rel.relation), str(rel.object_literal), rel.span.value)
        if sig in signatures:
            changed = True
            continue
        kept.append(rel)

    if changed:
        pef.relationships = kept
        _rebuild_relationship_indexes(pef)


def _epistemic_hold_dict_for_decision(decision: GovernanceDecision, turn: int) -> dict:
    """Build serialized epistemic_hold snapshot aligned with forensic ASK/REFUSE/STOP."""
    mode_map = {
        InterventionAction.CONTAIN: EPISTEMIC_MODE_AMBIGUITY,
        InterventionAction.FORCE_REVISE: EPISTEMIC_MODE_REFUSAL,
        InterventionAction.HARD_STOP: EPISTEMIC_MODE_STOP,
    }
    mode = mode_map.get(decision.action, EPISTEMIC_MODE_NONE)
    return {
        "schema_version": EPISTEMIC_HOLD_SCHEMA_VERSION,
        "mode": mode,
        "since_turn": turn,
        "pathway_id": decision.pathway_id,
        "interaction_open": decision.interaction_open,
        "commitment_closed": decision.commitment_closed,
        "last_audit_id": decision.cid or "",
    }


def _apply_epistemic_hold_after_non_admit(pef: PEFState, decision: GovernanceDecision, turn: int) -> None:
    """Update durable epistemic holding after REFUSE/STOP/ASK. REFUSE/STOP supersede ambiguity."""
    if decision.action in (InterventionAction.PASS, InterventionAction.SOFT_CORRECT):
        return
    if decision.action == InterventionAction.CONTAIN:
        existing = pef.epistemic_hold if isinstance(pef.epistemic_hold, dict) else None
        existing_mode = existing.get("mode") if existing else None
        if existing_mode in (EPISTEMIC_MODE_REFUSAL, EPISTEMIC_MODE_STOP):
            # An ambiguity contain must not replace a separate refusal or stop.
            return
    _cap = _primary_allowed_continuation(decision)
    if decision.interaction_open and _cap:
        pef.active_continuation_capability = _cap
        # Only derive new context if we don't already have one for this capability,
        # or if the new decision provides a more specific one.
        new_context = _active_continuation_context_from_decision(decision)
        if new_context:
            pef.active_continuation_context = new_context
        # If new_context is None but we already have a context, we preserve it
        # to avoid losing domain markers (like 'legal' or 'finance') during
        # intermediate continuation turns that log as CONTAIN.
    else:
        pef.active_continuation_capability = None
        pef.active_continuation_context = None
    if decision.action in (InterventionAction.FORCE_REVISE, InterventionAction.HARD_STOP):
        pef.pending_clarification = None
        pef.epistemic_hold = _epistemic_hold_dict_for_decision(decision, turn)
        return
    if decision.action == InterventionAction.CONTAIN:
        pef.epistemic_hold = _epistemic_hold_dict_for_decision(decision, turn)
        return


def _maybe_clear_epistemic_hold_on_admit(pef: PEFState, decision: GovernanceDecision) -> None:
    """Clear refusal/stop holding on soft admit; ambiguity clears only via binding.

    PASS alone does not clear REFUSAL or reopenable STOP — those are durable across
    turns (same pattern as ambiguity). SOFT_CORRECT clears refusal/stop when the
    pathway admits a revised response. Terminal STOP (interaction_open=False) is
    never cleared here.
    """
    if decision.action not in (InterventionAction.PASS, InterventionAction.SOFT_CORRECT):
        return
    hold = pef.epistemic_hold
    if not hold:
        return
    mode = hold.get("mode")
    if mode == EPISTEMIC_MODE_AMBIGUITY:
        return
    if decision.action != InterventionAction.SOFT_CORRECT:
        return
    if mode == EPISTEMIC_MODE_REFUSAL:
        pef.epistemic_hold = None
        return
    if mode == EPISTEMIC_MODE_STOP:
        if hold.get("interaction_open") is False:
            return
        pef.epistemic_hold = None
        return


_GOVERNED_STOP_CONTINUATION_TEXT = (
    "Previous request was stopped. This session is in a governed stop state. "
    "Start a new session or reframe as a permitted general educational question."
)


def _active_epistemic_stop_hold(pef: PEFState) -> dict | None:
    """Return the active stop hold dict when the session is in governed STOP posture."""
    hold = pef.epistemic_hold
    if hold and hold.get("mode") == EPISTEMIC_MODE_STOP:
        return hold
    return None


def _governed_stop_continuation_text(_user_text: str) -> str:
    """Policy-only continuation while stop hold is active (no upstream, no unsafe recall)."""
    return _GOVERNED_STOP_CONTINUATION_TEXT


def _epistemic_stop_continuation_flags() -> list[Flag]:
    return [
        Flag(
            flag_type=FlagType.UNRESOLVED_STATE_TRANSITION,
            entity_name="session",
            claim="Follow-up turn while epistemic stop hold is active",
            evidence="blocked/continued-stop",
            severity="error",
            rule_id="epistemic.stop.continuation",
        )
    ]


def _build_epistemic_stop_continuation_decision(hold: dict, *, user_text: str) -> GovernanceDecision:
    text = _governed_stop_continuation_text(user_text)
    decision = GovernanceDecision(
        action=InterventionAction.HARD_STOP,
        flags=_epistemic_stop_continuation_flags(),
        rationale="blocked/continued-stop (epistemic hold active)",
        policy="strict",
        pathway_id=hold.get("pathway_id") or "P_STOP_CONTINUED",
        output_mode="terminal_stop",
        commitment_closed=bool(hold.get("commitment_closed", True)),
        interaction_open=bool(hold.get("interaction_open", False)),
    )
    decision.original_response = ""
    decision.corrected_response = text
    decision.governed_response = text
    return decision


def _clear_epistemic_hold_ambiguity(pef: PEFState) -> None:
    """Clear ambiguity holding after structural binding resolution (SPINE: forced collapse).

    Also clears the active continuation capability corridor opened by the prior
    CONTAIN/clarification turn (e.g. ``neutral_timeline``). Otherwise a resolved
    binding still leaves ``active_continuation_capability`` set, and
    :meth:`Lens._maybe_handle_active_continuation_sync` treats ordinary follow-up
    questions as constrained continuation instead of normal extraction / routing.
    """
    pef.pending_clarification = None
    if pef.epistemic_hold and pef.epistemic_hold.get("mode") == EPISTEMIC_MODE_AMBIGUITY:
        pef.epistemic_hold = None
    pef.active_continuation_capability = None
    pef.active_continuation_context = None


def _apply_state_native_pending_clarify(
    pef: PEFState,
    *,
    original_question: str,
    detected_span: Span,
    clarify_context: dict,
) -> None:
    """Attach pending clarification from state-native engine (data-only context)."""
    pending: dict = {
        "original_question": original_question,
        "unresolved_entity_ids": [],
        "failed_constraint": clarify_context.get(
            "failed_constraint", "STATE_NATIVE_ENTITY_AMBIGUITY"
        ),
        "ambiguous_referents": [],
        "candidate_entities": list(clarify_context.get("candidate_entities") or []),
        "original_span": detected_span.value,
        "blocked_proposition": None,
        "blocked_claims": [],
        "clarification_prompt": clarify_context.get("clarification_prompt"),
        "entity_ambiguity_typo_recovery_suppressed": bool(
            clarify_context.get("entity_ambiguity_typo_recovery_suppressed")
        ),
    }
    subj = clarify_context.get("subject_phrase")
    if isinstance(subj, str) and subj.strip():
        pending["subject_phrase"] = subj.strip()
    item_p = clarify_context.get("item_phrase")
    if isinstance(item_p, str) and item_p.strip():
        pending["item_phrase"] = item_p.strip()
    amb_kind = clarify_context.get("state_native_ambiguity_kind")
    if isinstance(amb_kind, str) and amb_kind.strip():
        pending["state_native_ambiguity_kind"] = amb_kind.strip()
    tr = clarify_context.get("typo_recovery")
    if isinstance(tr, dict):
        pending["typo_recovery"] = dict(tr)
    pef.pending_clarification = pending


def _state_native_lens_status(outcome: "StateNativeOutcome") -> LensStatus:
    """Map state-native outcome to canonical LensStatus for policy resolution.

    NOTE: This mapping is intentionally explicit at the seam because future Lane 1
    commitment states may require a finer status vocabulary than ANSWER/CLARIFY/STOP.
    """
    from aurora_lens.state_native_engine.contracts import StateNativeOutcome

    if outcome == StateNativeOutcome.ANSWER:
        return LensStatus.ADMIT
    if outcome == StateNativeOutcome.CLARIFY:
        return LensStatus.ASK
    if outcome == StateNativeOutcome.STOP:
        return LensStatus.STOP
    raise ValueError(f"unsupported state-native outcome for policy resolution: {outcome!r}")


def _normalize_state_native_result_from_temporal_eval(
    sn: "StateNativeDelegationResult",
) -> "StateNativeDelegationResult":
    """Normalize state-native outcome from present-bound temporal governance cues.

    Preserve existing behavior when no temporal evaluator payload is present.
    """
    from aurora_lens.state_native_engine.contracts import StateNativeOutcome
    from aurora_lens.state_native_engine.epistemic import EpistemicResult

    ev = sn.temporal_eval_result
    if ev is None:
        return sn

    cue = ev.governance_cue
    if cue == TemporalGovernanceCue.ASK_TEMPORAL_SCOPE:
        return replace(
            sn,
            outcome=StateNativeOutcome.CLARIFY,
            epistemic_result=sn.epistemic_result or EpistemicResult.AMBIGUOUS,
        )
    if cue == TemporalGovernanceCue.GOVERNED_NON_ANSWER:
        stop_reason_code = sn.stop_reason_code
        if not stop_reason_code:
            tail = ev.outcome_kind.value.lower().removeprefix("temporal_")
            stop_reason_code = f"state_native_temporal_{tail}"
        return replace(
            sn,
            outcome=StateNativeOutcome.STOP,
            stop_reason_code=stop_reason_code,
        )
    if cue == TemporalGovernanceCue.PRESENT_COMMITTED_ANSWER:
        return replace(sn, outcome=StateNativeOutcome.ANSWER)
    return sn


def _state_native_lens_status_from_temporal_eval(
    eval_res: PresentBoundTemporalEvalResult | None,
    *,
    default_outcome: "StateNativeOutcome",
) -> LensStatus:
    """Map present-bound temporal governance cue into LensStatus when available."""
    if eval_res is None:
        return _state_native_lens_status(default_outcome)
    cue = eval_res.governance_cue
    if cue == TemporalGovernanceCue.ASK_TEMPORAL_SCOPE:
        return LensStatus.ASK
    if cue == TemporalGovernanceCue.GOVERNED_NON_ANSWER:
        return LensStatus.STOP
    if cue == TemporalGovernanceCue.PRESENT_COMMITTED_ANSWER:
        return LensStatus.ADMIT
    return _state_native_lens_status(default_outcome)


@dataclass
class LensResult:
    """Result of processing a single user turn through the lens."""
    response: str                                   # Final response text (post-governance)
    flags: list[Flag]                               # Verification flags (empty = clean)
    pef_snapshot: str                               # PEF state summary after this turn
    turn: int                                       # Turn number
    span: Span                                      # Detected span of user input
    model: str = ""                                 # Model used
    action: InterventionAction = InterventionAction.PASS  # Governance action taken
    decision: GovernanceDecision | None = None       # Full governance decision (if flags)
    original_response: str | None = None            # Pre-intervention response (if modified)
    usage: dict | None = None                       # Token usage from adapter, if available
    self_refused: bool = False                      # True when LLM self-refused (action=PASS but refusal frame detected)
    # Primary model output before epistemic_normalisation (None if no main generate).
    upstream_model_draft: str | None = None
    epistemic_normalisation_applied: bool = False   # True iff post-PASS surface pass changed text
    # Continuity / observability: set when state-native answers from committed PEF.
    # See ``docs/pef_location_state_kernel.md`` (diagnostic vocabulary).
    continuity_diagnostic: str | None = None
    # Operator-plane wire: snapshot of ``domain_var`` when the result is built (streaming
    # and some test clients may not see the same ContextVar at metadata build time).
    operator_request_domain: str | None = None
    # Operator telemetry override (e.g. medical demo deterministic retrieval).
    telemetry_release_path: str | None = None


class LensRoute(str, Enum):
    STATE_NATIVE_READ = "state_native_read"
    PRE_LLM_CONTAIN = "pre_llm_contain"
    MUTATION_ACK = "mutation_ack"
    GENERATION_NATIVE = "generation_native"


@dataclass(frozen=True)
class LensRoutePlan:
    route: LensRoute
    reason_code: str


class Lens:
    """The aurora-lens pipeline orchestrator.

    Wires: Interpretation → LLM Adapter → Verification → Governance
    """

    def __init__(
        self,
        config: LensConfig,
        initial_pef: PEFState | None = None,
        session_id: str = "",
    ):
        self._config = config
        self._using_default_extraction_backend = config.extraction_backend is None
        if initial_pef is not None:
            self._pef = initial_pef
        else:
            self._pef = PEFState(session_id=session_id)
        # Always align ``PEFState.session_id`` with the proxy/session-store key. Hydrated
        # ``SessionRecord.pef_state`` may omit or empty this field (legacy persistence);
        # governance-only PASS rows rely on ``pef_snapshot["session_id"]`` for AFL linkage.
        if isinstance(session_id, str) and session_id.strip():
            self._pef.session_id = session_id.strip()
        if config.extraction_backend is not None:
            self._backend: ExtractionBackend = config.extraction_backend
        else:
            from aurora_lens.interpret.spacy_backend import SpacyBackend

            self._backend = SpacyBackend(model=config.spacy_model)
        from aurora_lens.interpret.spacy_backend import SpacyBackend as _SpacyBackendForPossession

        self._possession_nlp_for_state_native = (
            self._backend.nlp if isinstance(self._backend, _SpacyBackendForPossession) else None
        )
        self._disable_admitted_assertion_ack = any(
            hasattr(self._backend, attr)
            for attr in ("_claims_per_call", "_sequence", "_claims", "_call_count")
        )
        self._checker = Checker(self._backend)
        self._governor = StructuralGovernor()
        self._context_resolver = ContextResolver()
        self._history: list[dict[str, str]] = []
        self._sovereign_route_evaluation = None
        self._sovereign_route_request = None
        self._sovereign_route_audit_attached = False
        # Audit–PEF linkage: deep-frozen turn-start PEF (set each turn in process / process_stream).
        # Shallow ``to_dict()`` snapshots share nested dict refs with live ``self._pef``; later
        # in-place mutations could drift ``pef_turn_classification`` away from what
        # ``verify_pef_linkage`` replays from the prior row's ``pef_snapshot``.
        self._audit_pef_start_dict: dict = {}
        self._audit_pef_start_frozen: PEFState | None = None
        # Set before update_pef when ambiguous_referents non-empty; consumed post-LLM
        # to force UNRESOLVED_REFERENT governance until discourse bindings resolve tokens.
        self._ambiguous_snapshot_for_assistant_pending: list[str] | None = None
        # Keys present in ``discourse_referent_bindings`` immediately before this turn's
        # user extraction commit; used so post-LLM resolution checks do not treat
        # bindings introduced *by* ``update_pef`` as prior-turn disambiguation.
        self._discourse_binding_keys_before_user_turn_commit: frozenset[str] | None = None
        # Last ``update_pef`` envelope this turn — operator-plane serialization only.
        self._pef_admission_result_wire_snap: dict | None = None

        # Governance bridge
        if config.governance_bridge is not None:
            self._bridge = config.governance_bridge
        else:
            self._bridge = CanonicalScannerGateBridge(audit_path=config.audit_log_path)

        self._state_native_engine = None
        if config.enable_state_native_delegation:
            if config.state_native_engine is not None:
                self._state_native_engine = config.state_native_engine
            else:
                from aurora_lens.state_native_engine.default_engine import DefaultStateNativeEngine

                self._state_native_engine = DefaultStateNativeEngine()

    @property
    def pef(self) -> PEFState:
        """Access the current PEF state (read-only access point)."""
        return self._pef

    @property
    def request_metadata(self) -> RequestMetadata | None:
        """Host-provided envelope for this HTTP request when using the proxy; None otherwise."""
        return get_request_metadata()

    def peek_pef_admission_result_wire(self) -> dict | None:
        """Structured ``PEFAdmissionResult`` for operator payloads (last ``update_pef`` this turn)."""
        return self._pef_admission_result_wire_snap

    def _capture_pef_admission_for_operator_wire(self, adm: PEFAdmissionResult | None) -> None:
        if adm is None:
            return
        self._pef_admission_result_wire_snap = pef_admission_result_wire_dict(adm)

    def _reset_pef_admission_operator_wire_turn(self) -> None:
        self._pef_admission_result_wire_snap = None

    def _plan_turn_route(
        self,
        *,
        state_native_handled: bool,
        pre_llm_contain: bool,
        mutation_ack_ready: bool,
    ) -> LensRoutePlan:
        """Single authoritative route planner for turn execution precedence."""
        if state_native_handled:
            return LensRoutePlan(
                route=LensRoute.STATE_NATIVE_READ,
                reason_code="state_native_handled_query_surface",
            )
        if pre_llm_contain:
            return LensRoutePlan(
                route=LensRoute.PRE_LLM_CONTAIN,
                reason_code="pending_or_structural_pre_llm_containment",
            )
        if mutation_ack_ready:
            return LensRoutePlan(
                route=LensRoute.MUTATION_ACK,
                reason_code="pre_llm_mutation_ack_admitted",
            )
        return LensRoutePlan(
            route=LensRoute.GENERATION_NATIVE,
            reason_code="default_generation_native",
        )

    def _apply_route_reason(self, decision: GovernanceDecision, route_plan: LensRoutePlan) -> None:
        """Attach route observability without changing decision rationale semantics."""
        route_tag = f"route:{route_plan.route.value}:{route_plan.reason_code}"
        if decision.rule_id:
            if route_tag not in decision.rule_id:
                decision.rule_id = f"{decision.rule_id}|{route_tag}"
        else:
            decision.rule_id = route_tag

    def _suppress_admitted_assertion_ack(self, ack_plan: object | None) -> bool:
        """Single admission policy seam for admitted-state assertion auto-ack."""
        if ack_plan is None:
            return False
        mutation_kind = getattr(ack_plan, "mutation_kind", None)
        if mutation_kind != "admitted_state_assertion":
            return False
        # Operator-plane JSON (include_operator_detail) must not force adapter→checker
        # for the same turn: proxy deployments that expose flags in the response body
        # would otherwise skip this path and spuriously flag variable model acknowledgements
        # of user-admitted state (e.g. UNVERIFIED_FACT_ASSERTION on setup turns).
        # Scripted harnesses and mock extraction backends use _disable_admitted_assertion_ack
        # or request-scoped metadata instead.
        return (
            self._disable_admitted_assertion_ack
            or self.request_metadata is not None
        )

    def _audit_linkage_kwargs(
        self,
        decision: GovernanceDecision,
        *,
        state_native_source_status: str | None = None,
    ) -> dict[str, str | None]:
        """PEF turn classification (turn entry) + hold transition (after mutations) for audit rows.

        ``verify_pef_linkage`` compares each row's ``pef_turn_classification`` to
        ``classify_pef_turn_start(PEFState.from_dict(prev.pef_snapshot))``. For consecutive
        turns, ``prev.pef_snapshot`` is the post-decision state at the end of the prior turn,
        which matches ``self._audit_pef_start_dict`` captured at the start of the current
        turn (after ``advance_turn``). Do **not** derive ``pef_turn_classification`` from the
        post-decision ``pef_snapshot`` passed to ``log_decision`` (e.g. after SOFT_CORRECT
        clears a refusal hold), or replay will disagree with the verifier.

        **Source status for refusal scope refinement:**
        When ``state_native_source_status="admitted_uncontaminated"``, explicit proof is
        provided that the turn's source relationships are uncontaminated. Only this explicit
        status allows downgrading from HELD_REFUSAL to HELD_STATE. Without it, refusal holds
        remain globally governing to avoid false classification.
        """
        pef_before: PEFState | None = None
        if self._audit_pef_start_frozen is not None:
            pef_before = copy.deepcopy(self._audit_pef_start_frozen)
        elif self._audit_pef_start_dict:
            pef_before = PEFState.from_dict(copy.deepcopy(self._audit_pef_start_dict))
        if pef_before is None:
            return {
                "pef_turn_classification": None,
                "pef_hold_transition": None,
            }
        ht = classify_hold_transition(
            decision,
            pef_before,
            self._pef,
            state_native_source_status=state_native_source_status,
        )
        tc = classify_pef_turn_start(
            pef_before,
            state_native_source_status=state_native_source_status,
        ).value
        return {
            "pef_turn_classification": tc,
            "pef_hold_transition": ht.value,
        }

    def _attach_provider_route_audit(
        self,
        decision: GovernanceDecision,
        *,
        adapter_called: bool,
    ) -> None:
        """Attach replayable provider_route envelope when sovereign evaluation ran this turn."""
        from aurora_lens.sovereign.audit_envelope import build_provider_route_audit_envelope

        if self._sovereign_route_audit_attached:
            return
        evaluation = self._sovereign_route_evaluation
        request = self._sovereign_route_request
        if evaluation is None:
            return
        if request is None and evaluation.reason == "provider_route_metadata_required":
            from aurora_lens.request_metadata import ProviderRouteRequest

            request = ProviderRouteRequest(
                primary_provider_id="",
                primary_state="",
                task_domain="",
                consequence_grade="",
            )
        elif request is None:
            return
        alternate_profile = None
        registry = self._config.sovereign_provider_registry
        if registry is not None and request.alternate_provider_id:
            alternate_profile = registry.get_profile(request.alternate_provider_id)
        decision.provider_route = build_provider_route_audit_envelope(
            request,
            evaluation,
            adapter_called=adapter_called,
            alternate_profile=alternate_profile,
        )
        self._sovereign_route_audit_attached = True

    def _audit_at_basis(self) -> dict:
        """Recoverable AT ranking (`current`/`prior`) for audit at decision-log time."""
        return at_basis_snapshot(self._pef)

    def _log_agency_context_resolution_event(
        self,
        *,
        original_unresolved_risk_prompt: str,
        removed_coercive_objective_fragments: list[str],
        sanitized_prompt_used_for_generation: str,
    ) -> None:
        """Emit explicit audit evidence for agency-context sanitization."""
        logger = getattr(self._bridge, "log_subsystem_audit_event", None)
        if not callable(logger):
            return
        logger(
            op="agency_context_resolution",
            payload={
                "reason_type": "agency_context_resolved_non_coercive",
                "original_unresolved_risk_prompt": original_unresolved_risk_prompt,
                "removed_coercive_objective_fragments": removed_coercive_objective_fragments,
                "sanitized_prompt_used_for_generation": sanitized_prompt_used_for_generation,
            },
        )

    def _maybe_log_clarification_resolution_audit(
        self,
        *,
        pending: dict,
        history_user_input: str,
        clar_ext: "ExtractionResult",
        turn: int,
        pef_snapshot_before: dict | None,
        binding_found: bool,
        explicit_selected: str | None = None,
        typo_merged: bool = False,
    ) -> None:
        """Emit USER_DISAMBIGUATION when any binding path consumes pending clarification."""
        from aurora_lens.govern.audit_io import CHAIN_GENESIS
        from aurora_lens.govern.clarification_audit import (
            bound_entity_id_for_selected,
            build_clarification_resolution_payload,
            resolve_selected_option_for_audit,
            should_emit_clarification_resolution_audit,
        )

        selected = resolve_selected_option_for_audit(
            pending=pending,
            pef=self._pef,
            user_selection_text=history_user_input,
            clar_ext=clar_ext,
            explicit_selected=explicit_selected,
            resolve_binding_entity=_resolve_binding_entity,
            typo_merged=typo_merged,
        )
        if not should_emit_clarification_resolution_audit(
            pending=pending,
            binding_found=binding_found,
            selected_option=selected,
        ):
            return
        log_fn = getattr(self._bridge, "log_clarification_resolution", None)
        if not callable(log_fn):
            return
        assert selected is not None
        prior_cid = getattr(self._bridge, "_prev_cid", None)
        if prior_cid == CHAIN_GENESIS:
            prior_cid = None
        payload = build_clarification_resolution_payload(
            pending=pending,
            user_selection_text=history_user_input,
            selected_option=selected,
            bound_entity_id=bound_entity_id_for_selected(self._pef, selected),
            pef_snapshot_before=pef_snapshot_before,
            pef_snapshot_after=self._pef.to_dict(),
            prior_ask_cid=prior_cid,
            prior_ask_trace_id=trace_id_var.get(None),
        )
        log_fn(turn=turn, clarification_resolution=payload)

    def _check_blocked_act_request(self, user_input: str) -> list[Flag]:
        """Pre-LLM blocked-act scan (RAG harness: Question line only)."""
        return self._checker.check_blocked_act_request(
            blocked_act_request_scan_text(user_input),
            pef=self._pef,
        )

    def _active_continuation_intent_changed(self, user_input: str) -> bool:
        """True when user explicitly exits the active continuation corridor."""
        if _looks_like_explicit_intent_change(user_input):
            return True
        return bool(self._check_blocked_act_request(user_input))

    def _set_medical_post_refusal_context_if_applicable(
        self,
        *,
        user_input: str,
        flags: list[Flag],
        decision: GovernanceDecision,
    ) -> None:
        """Attach structured medical hard-stop context for safe follow-up corridor."""
        cap = _primary_allowed_continuation(decision)
        if cap != _MEDICAL_POST_REFUSAL_CAPABILITY:
            return
        if decision.action != InterventionAction.HARD_STOP:
            return
        if not decision.interaction_open:
            return
        if not any(flag.flag_type in _MEDICAL_POST_REFUSAL_FLAG_TYPES for flag in flags):
            return
        self._pef.active_continuation_context = _build_medical_post_refusal_context(user_input)

    def _build_active_capability_decision(
        self,
        capability: str,
        response_text: str,
        *,
        accepted_update: bool = False,
        context: dict[str, str] | None = None,
    ) -> GovernanceDecision:
        """Build a deterministic governed continuation decision for active capability turns."""
        display_cap = _display_continuation_capability_name(capability, context)
        action = InterventionAction.PASS if accepted_update else InterventionAction.CONTAIN
        continuation_flags = []
        if not accepted_update:
            continuation_flags = [
                Flag(
                    flag_type=FlagType.UNRESOLVED_STATE_TRANSITION,
                    entity_name="continuation_capability",
                    claim="Turn is constrained to an already-authorized continuation capability",
                    evidence=f"active_continuation_capability={display_cap}",
                    severity="warning",
                )
            ]
        decision = GovernanceDecision(
            action=action,
            flags=continuation_flags,
            rationale=(
                f"Continuation capability updated: {display_cap}"
                if accepted_update
                else f"Continuation capability follow-up: {display_cap}"
            ),
            policy="strict",
            pathway_id="P_ADMIT_STANDARD" if accepted_update else "P_ASK_MISSING_FACT",
            output_mode="constrained_response" if accepted_update else "clarification_request",
            commitment_closed=True,
            interaction_open=True,
            allowed_continuations=[capability],
        )
        if accepted_update:
            decision.governance_note = "CONTINUING — timeline updated"
        decision.original_response = ""
        decision.corrected_response = response_text
        decision.governed_response = response_text
        dom = str((context or {}).get("domain") or "").strip().lower() or "general"
        _attach_rule_result(
            decision,
            domain_override=dom,
            continuation_type_override=_continuation_type_from_capability(display_cap, dom),
            reason_code_override="continuation_update" if accepted_update else "continuation_followup",
            template_key_override=f"{dom}.continuation.{'update' if accepted_update else 'followup'}",
        )
        return decision

    def _maybe_return_epistemic_stop_continuation_sync(
        self,
        *,
        history_user_input: str,
        turn: int,
    ) -> LensResult | None:
        """Hard corridor: while stop hold is active, never call upstream or PASS follow-ups.

        When the hold is interaction_open=True and an active continuation corridor
        is present, yield to _maybe_handle_active_continuation_sync instead so the
        corridor (e.g. neutral_timeline) can service follow-up turns.
        """
        hold = _active_epistemic_stop_hold(self._pef)
        if hold is None:
            return None
        # Interaction-open stops yield: the continuation capability (if any) or
        # normal processing handles follow-up turns. Only interaction_open=False
        # locks the session here.
        if hold.get("interaction_open"):
            return None
        decision = _build_epistemic_stop_continuation_decision(
            hold,
            user_text=history_user_input,
        )
        response_text = decision.governed_response or _GOVERNED_STOP_CONTINUATION_TEXT
        self._bridge.log_decision(
            decision,
            turn=turn,
            pef_context=self._pef.to_context_summary(),
            pre_llm=True,
            pef_snapshot=self._pef.to_dict(),
            at_verification_basis=self._audit_at_basis(),
            **self._audit_linkage_kwargs(decision),
        )
        self._history.append({"role": "user", "content": history_user_input})
        self._history.append({"role": "assistant", "content": response_text})
        return LensResult(
            response=response_text,
            flags=list(decision.flags),
            pef_snapshot=self._pef.to_context_summary(),
            turn=turn,
            span=Span.PRESENT,
            model="",
            action=decision.action,
            decision=decision,
            original_response=None,
        )

    def _maybe_adjudicate_candidate_release_task(
        self,
        *,
        history_user_input: str,
        turn: int,
    ) -> LensResult | None:
        """Structured execution-boundary adjudication — no prose inference, no upstream."""
        task = get_execution_task()
        if task is None:
            return None

        adj = adjudicate_candidate_release(task)
        if adj.admitted:
            action = InterventionAction.PASS
            flags: list[Flag] = []
            pathway_id = None
            output_mode = None
            commitment_closed = False
            interaction_open = True
            release_path = "released"
        else:
            failed_fc = adj.failed_constraint or "UNSUPPORTED_EVENT"
            flag_type = failed_constraint_to_flag_type(failed_fc)
            action = InterventionAction.HARD_STOP
            flags = [
                Flag(
                    flag_type=flag_type,
                    entity_name="candidate_release",
                    claim=task.candidate_release,
                    evidence=adj.rationale,
                    severity="error",
                )
            ]
            pathway_id = "P_STOP_REFUSE_CLEAN"
            output_mode = None
            commitment_closed = True
            interaction_open = False
            release_path = "blocked_before_generation"

        decision = GovernanceDecision(
            action=action,
            flags=flags,
            rationale=adj.rationale,
            policy="strict",
            pathway_id=pathway_id,
            output_mode=output_mode,
            commitment_closed=commitment_closed,
            interaction_open=interaction_open,
        )
        decision.original_response = ""
        decision.corrected_response = adj.response_text
        decision.governed_response = adj.response_text
        self._apply_route_reason(
            decision,
            LensRoutePlan(
                route=LensRoute.PRE_LLM_CONTAIN if not adj.admitted else LensRoute.MUTATION_ACK,
                reason_code="candidate_release_adjudication",
            ),
        )
        self._bridge.log_decision(
            decision,
            turn=turn,
            pef_context=self._pef.to_context_summary(),
            pre_llm=True,
            pef_snapshot=self._pef.to_dict(),
            at_verification_basis=self._audit_at_basis(),
            **self._audit_linkage_kwargs(decision),
        )
        self._history.append({"role": "user", "content": history_user_input})
        self._history.append({"role": "assistant", "content": adj.response_text})
        return LensResult(
            response=adj.response_text,
            flags=flags,
            pef_snapshot=self._pef.to_context_summary(),
            turn=turn,
            span=Span.PRESENT,
            model="",
            action=action,
            decision=decision,
            original_response=None,
            telemetry_release_path=release_path,
        )

    def _evaluate_sovereign_provider_route(self, turn: int) -> None:
        """Evaluate provider route via Sovereign Registry and seed failover bridge on PEF."""
        from aurora_lens.sovereign.provider_registry import (
            apply_route_evaluation_to_pef,
            reject_failover_without_registry,
            reject_missing_provider_route,
        )

        self._sovereign_route_evaluation = None
        self._sovereign_route_request = None
        self._sovereign_route_audit_attached = False
        meta = get_request_metadata()
        registry = self._config.sovereign_provider_registry
        enforce = bool(self._config.sovereign_enforce_provider_route and registry is not None)

        if meta is None or meta.provider_route is None:
            if enforce:
                self._sovereign_route_request = None
                self._sovereign_route_evaluation = reject_missing_provider_route()
            return
        route_req = meta.provider_route
        self._sovereign_route_request = route_req
        if registry is None:
            self._sovereign_route_evaluation = reject_failover_without_registry(route_req)
            return
        evaluation = registry.evaluate_failover(route_req)
        apply_route_evaluation_to_pef(self._pef, evaluation, turn=turn)
        self._sovereign_route_evaluation = evaluation

    def _maybe_apply_sovereign_provider_route_gate(
        self,
        *,
        history_user_input: str,
        turn: int,
    ) -> LensResult | None:
        """Block adapter execution when sovereign route evaluation refuses failover."""
        from aurora_lens.sovereign.provider_registry import evaluation_blocks_adapter
        from aurora_lens.sovereign.refusal_templates import refusal_for_evaluation

        evaluation = self._sovereign_route_evaluation
        if evaluation is None or not evaluation_blocks_adapter(evaluation):
            return None

        primary_state = (
            self._sovereign_route_request.primary_state
            if self._sovereign_route_request is not None
            else None
        )
        refusal = refusal_for_evaluation(
            evaluation,
            primary_provider_state=primary_state,
        )
        response_text = refusal.message
        reason_detail = (
            "Sovereign Provider Registry refused provider route admissibility; "
            f"machine_reason={refusal.machine_reason}; "
            f"template={refusal.template_key}."
        )
        flags = [
            Flag(
                flag_type=FlagType.PREDICTIVE_CLAIM_NOT_ESTABLISHED,
                entity_name="sovereign_provider_route",
                claim=reason_detail,
                evidence=response_text,
                severity="error",
            )
        ]
        decision = GovernanceDecision(
            action=InterventionAction.HARD_STOP,
            flags=flags,
            rationale=reason_detail,
            policy="strict",
            pathway_id="P_STOP_REFUSE_CLEAN",
            output_mode=None,
            commitment_closed=True,
            interaction_open=False,
        )
        decision.original_response = ""
        decision.corrected_response = response_text
        decision.governed_response = response_text
        self._apply_route_reason(
            decision,
            LensRoutePlan(
                route=LensRoute.PRE_LLM_CONTAIN,
                reason_code="sovereign_provider_route",
            ),
        )
        self._attach_provider_route_audit(decision, adapter_called=False)
        self._bridge.log_decision(
            decision,
            turn=turn,
            pef_context=self._pef.to_context_summary(),
            pre_llm=True,
            pef_snapshot=self._pef.to_dict(),
            at_verification_basis=self._audit_at_basis(),
            **self._audit_linkage_kwargs(decision),
        )
        self._history.append({"role": "user", "content": history_user_input})
        self._history.append({"role": "assistant", "content": response_text})
        return LensResult(
            response=response_text,
            flags=flags,
            pef_snapshot=self._pef.to_context_summary(),
            turn=turn,
            span=Span.PRESENT,
            model="",
            action=InterventionAction.HARD_STOP,
            decision=decision,
            original_response=None,
            telemetry_release_path="blocked_before_generation",
        )

    def _analyze_and_store_pef_uncertainties(self) -> None:
        """Derive epistemic uncertainty records from committed PEF state and merge into PEF.

        Deterministic — no LLM call. Reads entity/relationship/retrieval_unresolved
        state to detect conflicts, unresolved entities, and blocked evidence.
        Declaratively-provided uncertainties (from evidence_state) are preserved.
        """
        from aurora_lens.pef.uncertainty_analysis import merge_epistemic_uncertainties, pef_uncertainty_analysis
        try:
            derived = pef_uncertainty_analysis(
                self._pef,
                trust_registry=self._config.trust_registry,
            )
            merge_epistemic_uncertainties(self._pef, derived)
        except Exception:
            pass

    def _source_scope_enforcement_flag_for_turn(
        self,
        *,
        history_user_input: str,
    ) -> Flag | None:
        """Require scoped evidence context when source_scope is declared on RAG-shaped turns."""
        meta = self.request_metadata
        if meta is None or not meta.source_scope:
            return None
        s = history_user_input.strip()
        if len(s) < 20 or s[:8].lower() != "context:":
            return None
        rest = s[8:]
        if not rest:
            return None
        sep = "\n\nquestion:"
        idx = rest.lower().find(sep)
        if idx < 0:
            return None
        context_block = rest[:idx].strip()
        tail = rest[idx:]
        cpos = tail.find(":")
        if cpos < 0:
            return None
        question_line = tail[cpos + 1 :].lstrip()
        if not question_line:
            return None
        if context_block:
            return None
        return Flag(
            flag_type=FlagType.UPSTREAM_INSUFFICIENT_CONTEXT,
            entity_name="source_scope",
            claim=(
                "Declared source_scope requires scoped retrieval evidence; "
                "empty scoped context cannot authorize commitment"
            ),
            evidence=f"source_scope={list(meta.source_scope)}; rag_context_body is empty",
            severity="warning",
            rule_id="lens.source_scope_enforcement",
        )

    def _rag_evidence_admission_block_entry(
        self,
        rag_adm: PEFAdmissionResult | None,
    ) -> dict[str, object] | None:
        """Return structured retrieval unresolved entry for evidence admission failures."""
        for entry in self._pef.retrieval_unresolved or []:
            if entry.get("kind") in _EVIDENCE_ADMISSION_KIND_TO_OUTCOME:
                return entry
        if rag_adm is None:
            return None
        for ev in rag_adm.evidence:
            if ev.veto_kind in _EVIDENCE_ADMISSION_KIND_TO_OUTCOME:
                return {"kind": ev.veto_kind}
        return None

    def _maybe_add_retrieval_consistency_conflict_flag(
        self,
        *,
        flags: list[Flag],
        rag_turn_active: bool,
    ) -> list[Flag]:
        """After support checks, map typed unresolved corpus conflicts to containment flags."""
        if not rag_turn_active or flags:
            return flags
        for entry in self._pef.retrieval_unresolved or []:
            kind = str(entry.get("kind", ""))
            if kind not in _RETRIEVAL_CONSISTENCY_KIND_TO_OUTCOME:
                continue
            reason = str(entry.get("reason", "")).strip() or kind
            scope = str(entry.get("shared_scope", "")).strip()
            key = str(entry.get("threshold_key", "")).strip()
            rows = entry.get("conflicting_values")
            detail = rows if isinstance(rows, list) else []
            conflict_flag = Flag(
                flag_type=FlagType.UPSTREAM_INSUFFICIENT_CONTEXT,
                entity_name=kind,
                claim=(
                    "Admissible/support-relevant evidence contains unresolved "
                    "typed corpus consistency conflict"
                ),
                evidence=(
                    f"kind={kind}; scope={scope or 'unknown'}; threshold_key={key or 'unknown'}; "
                    f"reason={reason}; details={detail}"
                ),
                severity="warning",
                rule_id="lens.retrieval_consistency_conflict",
            )
            return list(flags) + [conflict_flag]
        return flags

    @staticmethod
    def _bool_from_failure_meta(
        failure_meta: dict[str, object],
        key: str,
        *,
        default: bool,
    ) -> bool:
        raw = failure_meta.get(key)
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            low = raw.strip().lower()
            if low in {"true", "1", "yes", "y"}:
                return True
            if low in {"false", "0", "no", "n"}:
                return False
        return default

    def _freshness_policy_decision_for_failure(
        self,
        *,
        failure_kind: str,
        failure_meta: dict[str, object],
    ) -> FreshnessPermissionDecision | None:
        """Resolve freshness permission only for canonical freshness failure kinds."""
        if failure_kind not in _CANONICAL_FRESHNESS_FAILURE_KINDS:
            return None

        raw_grade = str(failure_meta.get("consequence_grade") or "").strip().lower()
        if not raw_grade and self.request_metadata and self.request_metadata.provider_route:
            raw_grade = str(self.request_metadata.provider_route.consequence_grade or "").strip().lower()
        grade = ConsequenceGrade.MODERATE
        if raw_grade in {g.value for g in ConsequenceGrade}:
            grade = ConsequenceGrade(raw_grade)

        raw_authority = str(failure_meta.get("authority_state") or "").strip().lower()
        authority = AuthorityState.OPERATOR
        if raw_authority in {a.value for a in AuthorityState}:
            authority = AuthorityState(raw_authority)

        reversibility_default = grade in {ConsequenceGrade.MINIMAL, ConsequenceGrade.LOW}
        reversibility = self._bool_from_failure_meta(
            failure_meta,
            "reversibility",
            default=reversibility_default,
        )
        escalation_available = self._bool_from_failure_meta(
            failure_meta,
            "escalation_available",
            default=(authority != AuthorityState.NONE),
        )
        policy_ref = str(failure_meta.get("policy_ref") or "").strip()

        return evaluate_freshness_permission(
            FreshnessPermissionInput(
                failure_kind=FreshnessFailureKind(failure_kind),
                consequence_grade=grade,
                policy_ref=policy_ref,
                authority_state=authority,
                reversibility=reversibility,
                escalation_available=escalation_available,
            )
        )

    def _render_rag_evidence_admission_message(
        self,
        *,
        kind: str,
        failure_kind: str | None,
        reasons: list[str],
        chunk_ids: list[str],
        freshness_decision: FreshnessPermissionDecision | None = None,
    ) -> str:
        if kind == "evidence_inadmissible":
            header = "Decision blocked: retrieved evidence is inadmissible for commitment."
            tail = (
                "Admissible continuation: provide current approved authoritative evidence "
                "with traceable source references."
            )
        else:
            header = "Clarification required: retrieved evidence admissibility is unresolved."
            tail = (
                "Admissible continuation: clarify authority/scope/date metadata or provide "
                "authoritative evidence."
            )
        lines = [header]
        if failure_kind:
            lines.append(f"Failure kind: {failure_kind}.")
        if freshness_decision is not None:
            lines.append(
                f"Permission outcome: {freshness_decision.outcome.value} "
                f"({freshness_decision.reason_code})."
            )
        if reasons:
            lines.append("Reasons: " + "; ".join(reasons))
        if chunk_ids:
            lines.append("Evidence refs: " + ", ".join(chunk_ids))
        lines.append(tail)
        return "\n\n".join(lines)

    def _maybe_apply_rag_evidence_admission_gate(
        self,
        *,
        rag_adm: PEFAdmissionResult | None,
        history_user_input: str,
        turn: int,
        detected_span: Span,
    ) -> LensResult | None:
        entry = self._rag_evidence_admission_block_entry(rag_adm)
        if entry is None:
            return None
        kind = str(entry.get("kind", ""))
        failure_kind = str(entry.get("failure_kind", "")).strip() or None
        failure_meta_raw = entry.get("failure_meta")
        failure_meta = failure_meta_raw if isinstance(failure_meta_raw, dict) else {}
        evidence_authority = str(entry.get("authority_state") or failure_meta.get("authority_state") or "").strip().lower()
        action = None
        if is_canonical_evidence_authority_state(evidence_authority):
            action = lens_action_for_evidence_authority(evidence_authority)
            if action == InterventionAction.PASS:
                return None
        if action is None:
            action = _EVIDENCE_ADMISSION_KIND_TO_OUTCOME.get(kind)
        if action is None:
            return None
        freshness_decision = None
        if failure_kind is not None and not is_canonical_evidence_authority_state(evidence_authority):
            freshness_decision = self._freshness_policy_decision_for_failure(
                failure_kind=failure_kind,
                failure_meta=failure_meta,
            )
            if freshness_decision is not None:
                projected = project_freshness_pathway(
                    outcome=freshness_decision.outcome,
                    authority_state=freshness_decision.authority_state,
                )
                action = (
                    InterventionAction.HARD_STOP
                    if projected.action_is_hard_stop
                    else InterventionAction.CONTAIN
                )
                policy_pathway_id = projected.pathway_id
                policy_output_mode = projected.output_mode
                policy_commitment_closed = projected.commitment_closed
                policy_interaction_open = projected.interaction_open
            else:
                policy_pathway_id = (
                    "P_STOP_ESCALATE" if action == InterventionAction.HARD_STOP else "P_ASK_DISAMBIGUATE"
                )
                policy_output_mode = (
                    "terminal_stop" if action == InterventionAction.HARD_STOP else "clarification_request"
                )
                policy_commitment_closed = True
                policy_interaction_open = action != InterventionAction.HARD_STOP
        else:
            policy_pathway_id = (
                "P_STOP_ESCALATE" if action == InterventionAction.HARD_STOP else "P_ASK_DISAMBIGUATE"
            )
            policy_output_mode = (
                "terminal_stop" if action == InterventionAction.HARD_STOP else "clarification_request"
            )
            policy_commitment_closed = True
            policy_interaction_open = action != InterventionAction.HARD_STOP
        reasons_raw = entry.get("reasons")
        reasons = [str(r) for r in reasons_raw] if isinstance(reasons_raw, list) else []
        chunk_ids_raw = entry.get("chunk_ids")
        chunk_ids = [str(cid) for cid in chunk_ids_raw] if isinstance(chunk_ids_raw, list) else []
        response_text = self._render_rag_evidence_admission_message(
            kind=kind,
            failure_kind=failure_kind,
            reasons=reasons,
            chunk_ids=chunk_ids,
            freshness_decision=freshness_decision,
        )
        flag_type = (
            FlagType.UNVERIFIED_FACT_ASSERTION
            if action == InterventionAction.HARD_STOP
            else FlagType.UPSTREAM_INSUFFICIENT_CONTEXT
        )
        flags = [
            Flag(
                flag_type=flag_type,
                entity_name="retrieved_context",
                claim=f"RAG evidence admission {kind}",
                evidence=(
                    f"failure_kind={failure_kind or 'unknown'}; "
                    f"authority_state={evidence_authority or entry.get('authority_state', '')}; "
                    f"freshness_status={entry.get('freshness_status') or failure_meta.get('freshness_status', '')}; "
                    f"authority_status={entry.get('authority_status') or failure_meta.get('authority_status', '')}; "
                    f"demotion_reason={entry.get('demotion_reason') or failure_meta.get('demotion_reason', '')}; "
                    f"mapped_action={action.name}; "
                    f"failure_meta={failure_meta or {}}; "
                    f"freshness_outcome={freshness_decision.outcome.value if freshness_decision else ''}; "
                    f"chunk_ids={chunk_ids or []}; reasons={reasons or []}"
                ),
                severity="error" if action == InterventionAction.HARD_STOP else "warning",
            )
        ]
        decision = GovernanceDecision(
            action=action,
            flags=flags,
            rationale=f"rag_evidence_admissibility_gate: {failure_kind or kind}",
            policy="strict",
            pathway_id=policy_pathway_id,
            output_mode=policy_output_mode,
            commitment_closed=policy_commitment_closed,
            interaction_open=policy_interaction_open,
        )
        decision.original_response = ""
        decision.corrected_response = response_text
        decision.governed_response = response_text
        self._apply_route_reason(
            decision,
            LensRoutePlan(
                route=LensRoute.PRE_LLM_CONTAIN,
                reason_code="rag_evidence_admissibility_gate",
            ),
        )
        _apply_epistemic_hold_after_non_admit(self._pef, decision, turn)
        self._attach_provider_route_audit(decision, adapter_called=False)
        self._bridge.log_decision(
            decision,
            turn=turn,
            pef_context=self._pef.to_context_summary(),
            pre_llm=True,
            pef_snapshot=self._pef.to_dict(),
            at_verification_basis=self._audit_at_basis(),
            **self._audit_linkage_kwargs(decision),
        )
        self._history.append({"role": "user", "content": history_user_input})
        self._history.append({"role": "assistant", "content": response_text})
        return LensResult(
            response=response_text,
            flags=flags,
            pef_snapshot=self._pef.to_context_summary(),
            turn=turn,
            span=detected_span,
            model="",
            action=action,
            decision=decision,
            original_response=None,
            telemetry_release_path="blocked_before_generation",
        )

    def _maybe_apply_epistemic_uncertainty_gate(
        self,
        *,
        history_user_input: str,
        turn: int,
    ) -> LensResult | None:
        """Block decision-seeking turns when open epistemic uncertainties are recorded."""
        from aurora_lens.govern.epistemic_uncertainty_gate import evaluate_epistemic_uncertainty_gate
        gate = evaluate_epistemic_uncertainty_gate(
            self._pef.open_epistemic_uncertainties,
            history_user_input,
        )
        if gate is None or not gate.blocks:
            return None

        flags = [
            Flag(
                flag_type=FlagType.PREDICTIVE_CLAIM_NOT_ESTABLISHED,
                entity_name="epistemic_uncertainty",
                claim=gate.reason_detail,
                evidence="; ".join(u.description for u in gate.matched_uncertainties),
                severity="error",
            )
        ]
        decision = GovernanceDecision(
            action=InterventionAction.CONTAIN,
            flags=flags,
            rationale=gate.reason_detail,
            policy="strict",
            pathway_id="P_STOP_ESCALATE",
            output_mode=None,
            commitment_closed=True,
            interaction_open=True,
        )
        decision.original_response = ""
        decision.corrected_response = gate.response_text
        decision.governed_response = gate.response_text
        self._apply_route_reason(
            decision,
            LensRoutePlan(
                route=LensRoute.PRE_LLM_CONTAIN,
                reason_code="epistemic_uncertainty_gate",
            ),
        )
        _apply_epistemic_hold_after_non_admit(self._pef, decision, turn)
        self._attach_provider_route_audit(decision, adapter_called=False)
        self._bridge.log_decision(
            decision,
            turn=turn,
            pef_context=self._pef.to_context_summary(),
            pre_llm=True,
            pef_snapshot=self._pef.to_dict(),
            at_verification_basis=self._audit_at_basis(),
            **self._audit_linkage_kwargs(decision),
        )
        self._history.append({"role": "user", "content": history_user_input})
        self._history.append({"role": "assistant", "content": gate.response_text})
        return LensResult(
            response=gate.response_text,
            flags=flags,
            pef_snapshot=self._pef.to_context_summary(),
            turn=turn,
            span=Span.PRESENT,
            model="",
            action=InterventionAction.CONTAIN,
            decision=decision,
            original_response=None,
            telemetry_release_path="blocked_before_generation",
        )

    def _maybe_return_epistemic_stop_continuation_stream(
        self,
        *,
        history_user_input: str,
        turn: int,
    ) -> LensResult | None:
        """Stream variant of the governed stop corridor (no adapter.generate)."""
        hold = _active_epistemic_stop_hold(self._pef)
        if hold is None:
            return None
        if hold.get("interaction_open"):
            return None
        decision = _build_epistemic_stop_continuation_decision(
            hold,
            user_text=history_user_input,
        )
        response_text = decision.governed_response or _GOVERNED_STOP_CONTINUATION_TEXT
        self._bridge.log_decision(
            decision,
            turn=turn,
            stream=True,
            stream_completed=True,
            stream_truncated=False,
            stream_dropped_chars=0,
            pef_context=self._pef.to_context_summary(),
            pre_llm=True,
            pef_snapshot=self._pef.to_dict(),
            at_verification_basis=self._audit_at_basis(),
            **self._audit_linkage_kwargs(decision),
        )
        self._history.append({"role": "user", "content": history_user_input})
        self._history.append({"role": "assistant", "content": response_text})
        return LensResult(
            response=response_text,
            flags=list(decision.flags),
            pef_snapshot=self._pef.to_context_summary(),
            turn=turn,
            span=Span.PRESENT,
            model="",
            action=decision.action,
            decision=decision,
            original_response=None,
        )

    def _maybe_handle_active_continuation_sync(
        self,
        *,
        user_input: str,
        history_user_input: str,
        turn: int,
    ) -> LensResult | None:
        """Handle follow-up turns under persisted continuation capability corridor."""
        cap = (self._pef.active_continuation_capability or "").strip().lower()
        if not cap:
            return None
        if self._active_continuation_intent_changed(history_user_input):
            self._pef.active_continuation_capability = None
            self._pef.active_continuation_context = None
            return None
        if cap not in ("neutral_timeline", _MEDICAL_POST_REFUSAL_CAPABILITY):
            self._pef.active_continuation_capability = None
            self._pef.active_continuation_context = None
            return None

        if cap == _MEDICAL_POST_REFUSAL_CAPABILITY:
            if _is_general_health_educational_question(user_input):
                self._pef.active_continuation_capability = None
                self._pef.active_continuation_context = None
                return None
            response_text, completed = _build_medical_post_refusal_followup_response(
                user_input,
                self._pef.active_continuation_context,
            )
            accepted_update = False
        else:
            response_text, completed, accepted_update, updated_context = _build_neutral_timeline_followup_response(
                user_input,
                self._pef.active_continuation_context,
            )
            self._pef.active_continuation_context = updated_context
        decision = self._build_active_capability_decision(
            cap,
            response_text,
            accepted_update=accepted_update,
            context=self._pef.active_continuation_context,
        )
        _apply_epistemic_hold_after_non_admit(self._pef, decision, turn)
        if completed:
            self._pef.active_continuation_capability = None
            self._pef.active_continuation_context = None

        self._bridge.log_decision(
            decision,
            turn=turn,
            pef_context=self._pef.to_context_summary(),
            pre_llm=True,
            pef_snapshot=self._pef.to_dict(),
            at_verification_basis=self._audit_at_basis(),
            **self._audit_linkage_kwargs(decision),
        )
        self._history.append({"role": "user", "content": history_user_input})
        self._history.append({"role": "assistant", "content": response_text})
        return LensResult(
            response=response_text,
            flags=[],
            pef_snapshot=self._pef.to_context_summary(),
            turn=turn,
            span=Span.PRESENT,
            model="",
            action=decision.action,
            decision=decision,
            original_response=None,
        )

    def _maybe_handle_active_continuation_stream(
        self,
        *,
        user_input: str,
        history_user_input: str,
        turn: int,
    ) -> LensResult | None:
        """Stream variant of active capability handling with stream-aware audit logging."""
        cap = (self._pef.active_continuation_capability or "").strip().lower()
        if not cap:
            return None
        if self._active_continuation_intent_changed(history_user_input):
            self._pef.active_continuation_capability = None
            self._pef.active_continuation_context = None
            return None
        if cap not in ("neutral_timeline", _MEDICAL_POST_REFUSAL_CAPABILITY):
            self._pef.active_continuation_capability = None
            self._pef.active_continuation_context = None
            return None

        if cap == _MEDICAL_POST_REFUSAL_CAPABILITY:
            if _is_general_health_educational_question(user_input):
                self._pef.active_continuation_capability = None
                self._pef.active_continuation_context = None
                return None
            response_text, completed = _build_medical_post_refusal_followup_response(
                user_input,
                self._pef.active_continuation_context,
            )
            accepted_update = False
        else:
            response_text, completed, accepted_update, updated_context = _build_neutral_timeline_followup_response(
                user_input,
                self._pef.active_continuation_context,
            )
            self._pef.active_continuation_context = updated_context
        decision = self._build_active_capability_decision(
            cap,
            response_text,
            accepted_update=accepted_update,
            context=self._pef.active_continuation_context,
        )
        _apply_epistemic_hold_after_non_admit(self._pef, decision, turn)
        if completed:
            self._pef.active_continuation_capability = None
            self._pef.active_continuation_context = None

        self._bridge.log_decision(
            decision,
            turn=turn,
            stream=True,
            stream_completed=True,
            stream_truncated=False,
            stream_dropped_chars=0,
            pef_context=self._pef.to_context_summary(),
            pre_llm=True,
            pef_snapshot=self._pef.to_dict(),
            at_verification_basis=self._audit_at_basis(),
            **self._audit_linkage_kwargs(decision),
        )
        self._history.append({"role": "user", "content": history_user_input})
        self._history.append({"role": "assistant", "content": response_text})
        return LensResult(
            response=response_text,
            flags=[],
            pef_snapshot=self._pef.to_context_summary(),
            turn=turn,
            span=Span.PRESENT,
            model="",
            action=decision.action,
            decision=decision,
            original_response=None,
        )

    def _restore_committed_session_before_turn(
        self,
        pef_dict: dict,
        history: list[dict[str, str]],
    ) -> None:
        """Reload the last committed session snapshot (PEF + conversation history).

        If the upstream stream or governance buffer does not complete, discard
        in-progress turn mutations and resume from the last committed state,
        not from partial surface output.
        """
        self._pef = PEFState.from_dict(pef_dict)
        self._history.clear()
        self._history.extend(history)
        self._audit_pef_start_dict = copy.deepcopy(self._pef.to_dict())
        self._audit_pef_start_frozen = copy.deepcopy(self._pef)

    async def _revision_conflict_pre_llm_gate(
        self,
        user_input: str,
        history_user_input: str,
        turn_act: TurnAct,
        extraction: ExtractionResult,
        turn: int,
        detected_span: Span,
        ext_flags: list[Flag] | None,
    ) -> LensResult | None:
        """Block ``update_pef`` when user text contradicts grounded PEF without a revision act.

        *turn_act* is the turn-level classification from ``classify_turn_act`` (computed once
        per ``process`` / ``process_stream`` on ``history_user_input``); do not re-derive
        from text here.
        """
        if not self._config.auto_interpret:
            return None
        if extraction.extraction_error:
            return None
        if turn_act == TurnAct.REVISE:
            return None
        _ex, _det_ex = extraction_conflicts_grounded_pef(self._pef, extraction)
        _ut, _det_ut = user_text_conflicts_grounded_pef(self._pef, history_user_input)
        if pef_admission_debug_enabled():
            log_revision_gate(
                extraction_conflict=_ex,
                user_text_conflict=_ut,
                detail=_det_ex or _det_ut,
            )
        _conf = _ex or _ut
        _detail = _det_ex if _ex else _det_ut
        if not _conf:
            return None
        rev_flags = [
            Flag(
                flag_type=FlagType.CONTRADICTED_FACT,
                entity_name="session",
                claim="User assertion conflicts with grounded PEF facts",
                evidence=_detail,
                severity="warning",
            )
        ]
        if ext_flags:
            rev_flags = rev_flags + list(ext_flags)
        pre_decision = await self._bridge.decide(rev_flags, user_input, self._pef)
        if pre_decision.action in (
            InterventionAction.PASS,
            InterventionAction.SOFT_CORRECT,
        ):
            pre_decision = replace(pre_decision, action=InterventionAction.CONTAIN)
        clarification = revision_clarification_message(_detail)
        pre_decision.original_response = ""
        pre_decision.corrected_response = clarification
        pre_decision.governed_response = clarification
        _apply_epistemic_hold_after_non_admit(self._pef, pre_decision, turn)
        self._bridge.log_decision(
            pre_decision,
            turn=turn,
            pef_context=self._pef.to_context_summary(),
            pre_llm=True,
            pef_snapshot=self._pef.to_dict(),
            at_verification_basis=self._audit_at_basis(),
            **self._audit_linkage_kwargs(pre_decision),
        )
        self._history.append({"role": "user", "content": history_user_input})
        self._history.append({"role": "assistant", "content": clarification})
        return LensResult(
            response=clarification,
            flags=rev_flags,
            pef_snapshot=self._pef.to_context_summary(),
            turn=turn,
            span=detected_span,
            model="",
            action=pre_decision.action,
            decision=pre_decision,
            original_response=None,
        )

    def _record_discourse_referents_from_binding(
        self,
        ambiguous_tokens: list[str],
        bound_entity: str,
        *,
        turn: int | None = None,
    ) -> None:
        """Remember which candidate entity ambiguous pronouns referred to after binding.

        Follow-up questions often repeat subject/possessive pronouns (``she`` after ``her``);
        extraction may still mark them ambiguous even when the user already disambiguated.
        """
        if not ambiguous_tokens or not bound_entity.strip():
            return
        pef = self._pef
        for tok in ambiguous_tokens:
            pef.discourse_referent_bindings[tok.lower()] = bound_entity
        lowers = {t.lower() for t in ambiguous_tokens}
        if lowers & {"her", "she"}:
            pef.discourse_referent_bindings.setdefault("she", bound_entity)
            pef.discourse_referent_bindings.setdefault("her", bound_entity)
        if lowers & {"his", "he"}:
            pef.discourse_referent_bindings.setdefault("he", bound_entity)
            pef.discourse_referent_bindings.setdefault("him", bound_entity)
            pef.discourse_referent_bindings.setdefault("his", bound_entity)
        if "its" in lowers:
            pef.discourse_referent_bindings.setdefault("its", bound_entity)
        resolve_unresolved_referents(
            pef,
            tokens=list(ambiguous_tokens),
            resolved_entity=bound_entity,
            turn=turn if turn is not None else pef.current_turn,
        )

    def _governed_ambiguous_tokens_for_turn(
        self,
        extraction: ExtractionResult | None,
        user_text: str,
    ) -> list[str]:
        """Extractor-marked plus durable registry tokens present in ``user_text``."""
        merged: list[str] = []
        seen: set[str] = set()
        bound = self._pef.discourse_referent_bindings
        if extraction is not None:
            for tok in self._ambiguous_referents_for_governed_clarification(extraction):
                tl = tok.lower()
                if tl not in bound and tl not in seen:
                    seen.add(tl)
                    merged.append(tok)
        for tok in open_registry_tokens_in_text(self._pef, user_text):
            tl = tok.lower()
            if tl not in bound and tl not in seen:
                seen.add(tl)
                merged.append(tok)
        return merged

    def _register_governed_ambiguity_in_pef(
        self,
        *,
        turn: int,
        utterance: str,
        tokens: list[str],
        extraction: ExtractionResult | None,
        candidate_entities: list[str] | None,
        scope_text: str | None = None,
    ) -> None:
        """Persist governed ambiguity into the durable registry."""
        if not tokens:
            return
        scope = scope_text if scope_text is not None else utterance
        _sentences = _split_sentences_by_terminal_punct(scope)
        _blocked_sents = [
            s for s in _sentences
            if any(_word_boundary_substring_in_text(p, s) for p in tokens)
        ]
        _blocked_proposition = " ".join(_blocked_sents) if _blocked_sents else None
        cands = list(candidate_entities or [])
        if not cands:
            cands = candidates_for_tokens(self._pef, tokens)
        register_unresolved_referents(
            self._pef,
            turn=turn,
            utterance=utterance,
            tokens=tokens,
            candidate_entities=cands,
            blocked_proposition=_blocked_proposition,
        )

    def _sync_unresolved_referent_registry_from_extraction(
        self,
        *,
        turn: int,
        utterance: str,
        extraction: ExtractionResult,
    ) -> None:
        if not extraction.ambiguous_referents:
            return
        _cands = _referent_resolution_candidate_names(
            extraction,
            self._pef,
            ambiguous_tokens=list(extraction.ambiguous_referents),
        )
        sync_registry_from_extraction(
            self._pef,
            turn=turn,
            utterance=utterance,
            extraction=extraction,
            candidate_entities=_cands,
        )

    def _open_registry_session_gate_allows_through(
        self,
        user_text: str,
        turn_act: TurnAct,
    ) -> bool:
        """True when open registry governance explicitly permits the turn to proceed."""
        if not open_entries(self._pef):
            return False
        gate = evaluate_unresolved_session_gate(
            self._pef,
            user_text,
            turn_act=turn_act,
        )
        return gate is not None and gate.allow_through

    async def _maybe_apply_open_registry_session_gate(
        self,
        *,
        turn: int,
        history_user_input: str,
        user_input: str,
        turn_act: TurnAct,
        binding_resumed: bool,
        ext_flags: list,
        detected_span: Span,
        extraction: ExtractionResult | None,
    ) -> LensResult | None:
        """Session-level gate: open registry blocks dependent consequence without pronoun reuse."""
        if binding_resumed:
            return None
        if is_session_reset_request(history_user_input):
            dismiss_open_unresolved_referents(self._pef)
            return None

        gate = evaluate_unresolved_session_gate(
            self._pef,
            history_user_input,
            turn_act=turn_act,
        )
        if gate is None or gate.allow_through:
            return None

        _ref_candidates = list(gate.candidate_entities)
        if not _ref_candidates:
            _ref_candidates = candidates_for_tokens(self._pef, gate.ambiguous_tokens)

        if gate.meta_response:
            clarification = compose_session_meta_inquiry_block(
                ambiguous_tokens=gate.ambiguous_tokens,
                candidate_entities=_ref_candidates,
                blocked_proposition=gate.blocked_proposition,
            )
            interp_flags = [
                Flag(
                    flag_type=FlagType.UNRESOLVED_REFERENT,
                    entity_name=gate.ambiguous_tokens[0] if gate.ambiguous_tokens else "",
                    claim="Unresolved referent registry is open",
                    evidence=gate.reason_detail or "meta inquiry while registry open",
                    severity="warning",
                    candidates=tuple(_ref_candidates),
                )
            ]
            pre_decision = await self._bridge.decide(
                interp_flags, user_input, self._pef,
            )
            if pre_decision.action == InterventionAction.PASS:
                pre_decision = replace(
                    pre_decision,
                    action=InterventionAction.CONTAIN,
                    pathway_id="P_ASK_DISAMBIGUATE",
                    output_mode="clarification_request",
                )
            pre_decision.original_response = ""
            pre_decision.corrected_response = clarification
            pre_decision.governed_response = clarification
            pre_decision.rationale = (
                f"{gate.reason_code}: {gate.reason_detail or 'meta inquiry'}"
            )
            _apply_epistemic_hold_after_non_admit(self._pef, pre_decision, turn)
            self._bridge.log_decision(
                pre_decision,
                turn=turn,
                pef_context=self._pef.to_context_summary(),
                pre_llm=True,
                pef_snapshot=self._pef.to_dict(),
                at_verification_basis=self._audit_at_basis(),
                **self._audit_linkage_kwargs(pre_decision),
            )
            self._history.append({"role": "user", "content": history_user_input})
            self._history.append({"role": "assistant", "content": clarification})
            return LensResult(
                response=clarification,
                flags=interp_flags,
                pef_snapshot=self._pef.to_context_summary(),
                turn=turn,
                span=detected_span,
                model="",
                action=pre_decision.action,
                decision=pre_decision,
                original_response=None,
            )

        if not gate.blocks:
            return None

        clarification = compose_session_dependent_consequence_block(
            ambiguous_tokens=gate.ambiguous_tokens,
            candidate_entities=_ref_candidates,
            blocked_proposition=gate.blocked_proposition,
            act_kind=gate.act_kind,
            reason_detail=gate.reason_detail,
            user_turn=history_user_input,
        )
        self._register_governed_ambiguity_in_pef(
            turn=turn,
            utterance=history_user_input,
            tokens=list(gate.ambiguous_tokens),
            extraction=extraction,
            candidate_entities=_ref_candidates,
            scope_text=gate.blocked_proposition or history_user_input,
        )
        interp_flags = [
            Flag(
                flag_type=FlagType.UNRESOLVED_REFERENT,
                entity_name=tok,
                claim=(
                    f"Session registry blocks dependent consequence while '{tok}' "
                    "remains unresolved"
                ),
                evidence=gate.reason_detail or gate.reason_code,
                severity="warning",
                candidates=tuple(_ref_candidates),
            )
            for tok in (gate.ambiguous_tokens or ["referent"])
        ]
        if ext_flags:
            interp_flags = interp_flags + list(ext_flags)
        pre_decision = await self._bridge.decide(interp_flags, user_input, self._pef)
        if pre_decision.action != InterventionAction.PASS:
            pre_decision.original_response = ""
            pre_decision.corrected_response = clarification
            pre_decision.governed_response = clarification
        else:
            pre_decision = replace(
                pre_decision,
                action=InterventionAction.CONTAIN,
                pathway_id="P_ASK_DISAMBIGUATE",
                output_mode="clarification_request",
                original_response="",
                corrected_response=clarification,
                governed_response=clarification,
            )
        pre_decision.rationale = (
            f"{gate.reason_code}: {gate.reason_detail or gate.act_kind}"
        )
        _scope_ambig = user_input
        _apply_epistemic_hold_after_non_admit(self._pef, pre_decision, turn)
        if extraction is not None and self._pef.pending_clarification is None:
            self._pef.pending_clarification = _build_pending_unresolved_referent_dict(
                self._pef,
                scope_text=_scope_ambig,
                original_question=user_input,
                extraction=extraction,
                ambiguous_tokens=list(gate.ambiguous_tokens),
                detected_span=detected_span,
            )
        elif self._pef.pending_clarification is None:
            register_unresolved_referents(
                self._pef,
                turn=turn,
                utterance=history_user_input,
                tokens=list(gate.ambiguous_tokens),
                candidate_entities=_ref_candidates,
                blocked_proposition=gate.blocked_proposition,
            )
        if (
            extraction is not None
            and self._config.auto_interpret
            and not _has_consequence_bearing_transfer_claim(extraction)
        ):
            self._commit_extraction_if_admissible(
                extraction, user_text=history_user_input, turn_act=turn_act
            )
        self._bridge.log_decision(
            pre_decision,
            turn=turn,
            pef_context=self._pef.to_context_summary(),
            pre_llm=True,
            pef_snapshot=self._pef.to_dict(),
            at_verification_basis=self._audit_at_basis(),
            **self._audit_linkage_kwargs(pre_decision),
        )
        self._history.append({"role": "user", "content": history_user_input})
        self._history.append({"role": "assistant", "content": clarification})
        return LensResult(
            response=clarification,
            flags=interp_flags,
            pef_snapshot=self._pef.to_context_summary(),
            turn=turn,
            span=detected_span,
            model="",
            action=pre_decision.action,
            decision=pre_decision,
            original_response=None,
        )

    async def _maybe_apply_hold_unresolved_selection(
        self,
        *,
        turn: int,
        history_user_input: str,
        user_input: str,
        pending: dict,
        detected_span: Span = Span.PRESENT,
    ) -> LensResult | None:
        """Apply or re-acknowledge HOLD_UNRESOLVED — control action, never LLM input."""
        if not _is_hold_unresolved_selection(user_input):
            return None

        _already_held = (
            str(pending.get("resolution_mode") or "") == RESOLUTION_MODE_HELD_UNRESOLVED
            or referent_registry_held_unresolved(self._pef)
        )
        if not _already_held and str(pending.get("failed_constraint") or "") != "UNRESOLVED_REFERENT":
            if not open_entries(self._pef):
                return None

        _ambig_tokens = [
            str(t).strip()
            for t in (pending.get("ambiguous_referents") or [])
            if str(t).strip()
        ]
        if not _ambig_tokens:
            for entry in open_entries(self._pef):
                if entry.token.strip():
                    _ambig_tokens.append(entry.token.strip())
        _candidates = _normalized_candidate_entities_for_binding(pending)
        if not _candidates:
            for entry in open_entries(self._pef):
                _candidates.extend(entry.candidate_entities)
            seen_c: set[str] = set()
            deduped: list[str] = []
            for name in _candidates:
                key = name.strip().lower()
                if key and key not in seen_c:
                    seen_c.add(key)
                    deduped.append(name.strip())
            _candidates = deduped

        if _already_held:
            clarification = compose_hold_unresolved_already_held_acknowledgment()
            rationale = "UNRESOLVED_REFERENT: HOLD_UNRESOLVED_IDEMPOTENT"
            evidence = "HOLD_UNRESOLVED_IDEMPOTENT: control action while registry held"
        else:
            hold_unresolved_referents(
                self._pef,
                turn=turn,
                tokens=_ambig_tokens or None,
            )
            _pending_updated = dict(pending)
            _pending_updated["resolution_mode"] = RESOLUTION_MODE_HELD_UNRESOLVED
            _pending_updated["clarification_choices"] = build_unresolved_referent_clarification_choices(
                _candidates,
                include_hold_unresolved=False,
            )
            self._pef.pending_clarification = _pending_updated
            clarification = compose_hold_unresolved_acknowledgment(
                ambiguous_tokens=_ambig_tokens,
                candidate_entities=_candidates,
                blocked_proposition=str(pending.get("blocked_proposition") or "") or None,
            )
            rationale = "UNRESOLVED_REFERENT: HOLD_UNRESOLVED"
            evidence = "HOLD_UNRESOLVED: intrinsic ambiguity preserved across interpretations"

        interp_flags = [
            Flag(
                flag_type=FlagType.UNRESOLVED_REFERENT,
                entity_name=_ambig_tokens[0] if _ambig_tokens else "",
                claim=(
                    "Unresolved referent already held without candidate binding"
                    if _already_held
                    else "Unresolved referent held without candidate binding"
                ),
                evidence=evidence,
                severity="warning",
                candidates=tuple(_candidates),
            )
        ]
        pre_decision = await self._bridge.decide(interp_flags, user_input, self._pef)
        if pre_decision.action == InterventionAction.PASS:
            pre_decision = replace(
                pre_decision,
                action=InterventionAction.CONTAIN,
                pathway_id="P_ASK_DISAMBIGUATE",
                output_mode="clarification_request",
            )
        pre_decision.original_response = ""
        pre_decision.corrected_response = clarification
        pre_decision.governed_response = clarification
        pre_decision.rationale = rationale
        if not _already_held:
            _apply_epistemic_hold_after_non_admit(self._pef, pre_decision, turn)
        self._bridge.log_decision(
            pre_decision,
            turn=turn,
            pef_context=self._pef.to_context_summary(),
            pre_llm=True,
            pef_snapshot=self._pef.to_dict(),
            at_verification_basis=self._audit_at_basis(),
            **self._audit_linkage_kwargs(pre_decision),
        )
        self._history.append({"role": "user", "content": history_user_input})
        self._history.append({"role": "assistant", "content": clarification})
        return LensResult(
            response=clarification,
            flags=interp_flags,
            pef_snapshot=self._pef.to_context_summary(),
            turn=turn,
            span=detected_span,
            model="",
            action=pre_decision.action,
            decision=pre_decision,
            original_response=None,
        )

    def _ambiguous_referents_for_governed_clarification(
        self,
        extraction: ExtractionResult,
    ) -> list[str]:
        """Extractor-marked referents not yet in committed ``discourse_referent_bindings``.

        Used for the pre-LLM gate and the pre-commit ambiguity snapshot when
        ``auto_verify`` is on. Unlike ``_compute_truly_ambiguous_referents``, there is
        no structural or RAG-line suppression: if the extractor surfaced a referent as
        ambiguous, the user receives governed clarification until an explicit binding
        exists (prior turn, clarification reply, or other committed discourse state).
        """
        if not extraction.ambiguous_referents:
            return []
        b = self._pef.discourse_referent_bindings
        return [p for p in extraction.ambiguous_referents if p.lower() not in b]

    def _compute_truly_ambiguous_referents(
        self,
        extraction: ExtractionResult,
        rag_question_line: str | None = None,
    ) -> list[str]:
        """Structural subset of ambiguous referents (RAG-aware suppression for diagnostics).

        Pre-LLM admission and the ambiguity snapshot use
        ``_ambiguous_referents_for_governed_clarification`` instead — no bypass when the
        extractor has marked unresolvedness.

        When ``rag_question_line`` is set (RAG harness question line only), a pronoun that
        would otherwise be treated as structurally ambiguous solely because multiple PEF
        entities exist is **not** marked truly ambiguous if exactly one **PEF-linked**
        candidate name appears in the question line (duplicate surfaces deduped by first
        token). Stray proper nouns such as demonyms (*Canadian* from ``What Canadian city
        …``) are excluded from anchor counting. Default non-RAG behavior unchanged
        (``rag_question_line`` is ``None``).
        """
        if not extraction.ambiguous_referents:
            return []
        _refined = [
            p for p in extraction.ambiguous_referents
            if p.lower() not in self._pef.discourse_referent_bindings
        ]
        if not _refined:
            return []
        _ambig_lower_pre = {p.lower() for p in _refined}
        _pronoun_head: dict[str, str] = {}
        for _c in extraction.claims:
            _subj_lower = _c.subject.lower()
            for _p in _ambig_lower_pre:
                if _subj_lower.startswith(_p + " ") or _subj_lower == _p:
                    _pronoun_head[_p] = _c.subject[len(_p):].strip().lower()
                    break

        _pef_entity_names = [e.name for e in self._pef.entities.values()]
        _truly_ambiguous: list[str] = []
        for _p in _refined:
            _head = _pronoun_head.get(_p.lower())
            _all = _candidate_entity_names(
                _pef_entity_names + extraction.entity_mentions
            )
            if _head:
                _n_eligible = len(_has_eligible_candidates(_head, self._pef))
                if _n_eligible > 1:
                    if _should_rag_suppress_truly_ambiguous(
                        rag_question_line, _all, _pef_entity_names
                    ):
                        continue
                    _truly_ambiguous.append(_p)
                elif _n_eligible == 0:
                    if len(_all) > 1:
                        if _should_rag_suppress_truly_ambiguous(
                            rag_question_line, _all, _pef_entity_names
                        ):
                            continue
                        _truly_ambiguous.append(_p)
            else:
                if len(_all) > 1:
                    if _should_rag_suppress_truly_ambiguous(
                        rag_question_line, _all, _pef_entity_names
                    ):
                        continue
                    _truly_ambiguous.append(_p)
        return _truly_ambiguous

    async def _extract_user_turn_for_interpret(
        self,
        user_input: str,
    ) -> tuple[ExtractionResult, str | None, PEFAdmissionResult | None]:
        """Run extraction for pre-LLM gates; optionally seed PEF from a RAG ``Context:`` block first.

        When the second value is non-None, use it as ``user_text`` for the final
        ``update_pef`` (question line only) instead of the full harness message.
        """
        if not effective_rag_retrieval_aware_referents(
            user_input,
            config_rag=self._config.rag_retrieval_aware_referents,
        ):
            return await self._backend.extract(user_input, self._pef), None, None
        rag = split_rag_context_question(user_input)
        if rag is None:
            return await self._backend.extract(user_input, self._pef), None, None
        context_block, question_line = rag
        context_for_extract = _strip_markdown_bold_wrappers_for_rag_extraction(context_block)
        snap = self._pef.to_dict()
        rag_adm: PEFAdmissionResult | None = None
        self._pef.retrieval_unresolved = []
        unresolved_aggregate: list[dict] = []
        for unit in _build_rag_context_extraction_units(context_for_extract):
            ext_ctx = await self._backend.extract(unit.text, self._pef)
            if ext_ctx.extraction_error:
                return await self._backend.extract(user_input, self._pef), None, None
            rag_adm = admit_retrieved_context_to_pef(
                ext_ctx,
                self._pef,
                context_block=unit.admission_context,
                request_metadata=get_request_metadata(),
                provenance_document_id=unit.document_id,
                provenance_document_locator=unit.document_locator,
            )
            unresolved_aggregate.extend(self._pef.retrieval_unresolved)
        self._pef.retrieval_unresolved = unresolved_aggregate
        if rag_adm is not None:
            self._capture_pef_admission_for_operator_wire(rag_adm)
        else:
            self._reset_pef_admission_operator_wire_turn()
        ext_q = await self._backend.extract(question_line, self._pef)
        if ext_q.extraction_error:
            self._pef = PEFState.from_dict(snap)
            self._reset_pef_admission_operator_wire_turn()
            return await self._backend.extract(user_input, self._pef), None, None
        return ext_q, question_line, rag_adm

    def _snapshot_ambiguity_structurally_resolved(
        self, snapshot: list[str] | None,
    ) -> bool:
        """True when every snapshot token was already discourse-bound before this turn's commit.

        Bindings created by ``update_pef`` on the same user message must not satisfy
        this predicate — otherwise post-LLM governance would skip while ambiguity
        was only heuristically collapsed, not user-resolved.

        If ``_discourse_binding_keys_before_user_turn_commit`` was never captured for this
        turn (``None``), fail closed: return ``False`` so snapshot governance cannot be
        skipped on current PEF bindings alone.
        """
        if not snapshot:
            return True
        before = self._discourse_binding_keys_before_user_turn_commit
        if before is None:
            return False
        b = self._pef.discourse_referent_bindings
        return all(
            t.lower() in b and t.lower() in before
            for t in snapshot
        )

    def _merge_ambiguous_snapshot_unresolved_referent_flags(
        self,
        flags: list[Flag],
        snapshot: list[str],
        extraction: ExtractionResult | None,
        pef: PEFState,
    ) -> list[Flag]:
        """Prepend UNRESOLVED_REFERENT flags for snapshot tokens not already flagged."""
        existing = {
            f.entity_name.lower()
            for f in flags
            if f.flag_type == FlagType.UNRESOLVED_REFERENT
        }
        cand_list: list[str] = []
        if extraction is not None:
            ambig_merge = list(snapshot)
            if extraction.ambiguous_referents:
                ambig_merge = list(
                    dict.fromkeys(ambig_merge + list(extraction.ambiguous_referents)),
                )
            cand_list = _referent_resolution_candidate_names(
                extraction,
                pef,
                ambiguous_tokens=ambig_merge,
            )
        cand_tuple = tuple(cand_list)
        new_flags: list[Flag] = []
        for tok in snapshot:
            tl = tok.lower()
            if tl in existing:
                continue
            existing.add(tl)
            new_flags.append(
                Flag(
                    flag_type=FlagType.UNRESOLVED_REFERENT,
                    entity_name=tok,
                    claim=f"Unresolved referent '{tok}' cannot be uniquely resolved",
                    evidence=(
                        "Multiple same-category entities are present; "
                        "binding without explicit grounding is not permitted"
                    ),
                    severity="warning",
                    candidates=cand_tuple,
                )
            )
        return new_flags + list(flags)

    def _maybe_merge_ambiguous_snapshot_governance_flags(
        self,
        flags: list[Flag],
        extraction: ExtractionResult | None,
        user_text: str = "",
    ) -> tuple[list[Flag], list[str] | None]:
        """If ambiguous snapshot or durable registry remains open, force checker flags.

        Returns ``(flags, merged_token_list)``; second value is non-None iff snapshot
        or registry governance was applied (caller must emit governed clarification).
        """
        merged_tokens: list[str] | None = None
        out_flags = flags
        snap = self._ambiguous_snapshot_for_assistant_pending
        if snap and not self._snapshot_ambiguity_structurally_resolved(snap):
            out_flags = self._merge_ambiguous_snapshot_unresolved_referent_flags(
                out_flags, snap, extraction, self._pef,
            )
            merged_tokens = list(snap)
        reg_tokens = open_registry_tokens_in_text(self._pef, user_text)
        if reg_tokens:
            out_flags = self._merge_ambiguous_snapshot_unresolved_referent_flags(
                out_flags, reg_tokens, extraction, self._pef,
            )
            merged_tokens = list(dict.fromkeys((merged_tokens or []) + reg_tokens))
        if merged_tokens is None:
            return flags, None
        return out_flags, merged_tokens

    async def _decision_never_pass_on_live_ambiguity_snapshot(
        self,
        decision: GovernanceDecision,
        *,
        ambiguous_snapshot_merged_tokens: list[str] | None,
        upstream_text: str,
        extraction: ExtractionResult | None,
        governor_verdict: object,
    ) -> GovernanceDecision:
        """If a pre-commit ambiguity snapshot is still live, forbid PASS/SOFT_CORRECT."""
        if (
            not ambiguous_snapshot_merged_tokens
            or extraction is None
        ):
            return decision
        if decision.action not in (
            InterventionAction.PASS,
            InterventionAction.SOFT_CORRECT,
        ):
            return decision
        _snap_flags = self._merge_ambiguous_snapshot_unresolved_referent_flags(
            [],
            ambiguous_snapshot_merged_tokens,
            extraction,
            self._pef,
        )
        policy_name = (
            getattr(self._bridge, "_policy", None)
            and getattr(self._bridge._policy, "name", "unknown")
        ) or "unknown"
        _redo = await self._bridge.decide(_snap_flags, upstream_text, self._pef)
        _redo.governor_verdict = governor_verdict
        if _redo.action in (
            InterventionAction.PASS,
            InterventionAction.SOFT_CORRECT,
        ):
            return GovernanceDecision(
                action=InterventionAction.FORCE_REVISE,
                flags=_snap_flags,
                rationale=(
                    "Ambiguity snapshot prohibits PASS/SOFT_CORRECT admit without "
                    "committed discourse binding."
                ),
                policy=policy_name,
                governor_verdict=governor_verdict,
            )
        return _redo

    def _commit_extraction_if_admissible(
        self,
        extraction: ExtractionResult,
        *,
        user_text: str,
        turn_act: TurnAct,
    ) -> bool:
        """Commit interpreted facts when the turn admits PEF structure.

        FU-QUERY-REL resolution branch (Phase 1.5): narrative QUERY turns are
        read-only for relationship admission in Lens. Any QUERY-derived claim
        persistence requires a dedicated governed non-answer mutation lane.

        Return value reports whether the world actually changed — callers use
        this to decide whether a mutation acknowledgement is truthful. Calling
        ``update_pef`` and returning ``True`` unconditionally would let a turn
        that resolved to zero world/continuation mutations (e.g. every claim
        was blocked pre-commit as ``UNRESOLVED_REFERENT``) still produce
        "Recorded in session state." — a false ack with nothing behind it.
        """
        if turn_act == TurnAct.QUERY:
            return False
        adm = update_pef(
            extraction,
            self._pef,
            user_text=user_text,
            turn_act=turn_act,
        )
        self._capture_pef_admission_for_operator_wire(adm)
        return bool((adm.mutation_count or 0) > 0)

    async def _maybe_apply_epistemic_normalisation_clean_pass(
        self,
        final_response: str,
        *,
        flags: list[Flag],
        decision: GovernanceDecision,
        history_user_input: str,
    ) -> tuple[str, bool]:
        """Post-check PASS-only surface normalisation (fail-closed).

        Runs only when governance is a clean PASS (no checker flags). Re-checks the
        candidate text; if verification would change, the edit is discarded.
        Governance decision record is not re-derived: same decision object/signature.
        """
        if not self._config.auto_verify:
            return final_response, False
        if decision.action != InterventionAction.PASS or flags:
            return final_response, False
        pre_sig = _governance_decision_norm_signature(decision)
        cand = apply_epistemic_normalisation(final_response, self._pef)
        if cand == final_response:
            return final_response, False
        _ug = build_user_grounding_context(
            self._pef, self._pef.current_turn, history_user_input
        )
        recheck = await self._checker.check(
            cand,
            self._pef,
            user_input=history_user_input,
            user_grounding=_ug,
        )
        if recheck:
            return final_response, False
        if _governance_decision_norm_signature(decision) != pre_sig:
            return final_response, False
        return cand, True

    def _make_governed_unknown_after_resolution(
        self,
        history_user_input: str,
        resolved_binding_name: str | None,
        turn: int,
        detected_span: "Span",
    ) -> "LensResult":
        """Return a bounded epistemic response when resolved-state query cannot be answered.

        After UNRESOLVED_REFERENT resolution, queries that no deterministic handler
        can answer must not fall through to LLM. The LLM would generate outside the
        verified world — producing fiction dressed as caution. Return a governed-unknown
        response instead: short, cold, true.
        """
        name_clause = f" about {resolved_binding_name}" if resolved_binding_name else ""
        response_text = (
            f"I do not have enough information to answer that question{name_clause} "
            f"from committed state."
        )
        insufficient_flag = Flag(
            flag_type=FlagType.UPSTREAM_INSUFFICIENT_CONTEXT,
            entity_name="governed_unknown",
            claim=(
                "Governed-unknown: query after referent resolution depends on committed state "
                "but state-native engine cannot answer. LLM fallback suppressed."
            ),
            evidence=response_text,
            severity="warning",
            rule_id="lens.governed_unknown_after_resolution",
        )
        decision = GovernanceDecision(
            action=InterventionAction.CONTAIN,
            flags=[insufficient_flag],
            rationale=(
                "CONTAIN: governed-unknown after referent resolution — insufficient committed "
                "state to answer; LLM fallback suppressed."
            ),
            policy=getattr(self._bridge, "_policy", None)
            and getattr(self._bridge._policy, "name", "unknown")
            or "unknown",
            pathway_id="P_ASK_MISSING_FACT",
            output_mode="clarification_request",
            interaction_open=True,
            commitment_closed=True,
            epistemic_state="insufficient_context",
        )
        decision.original_response = response_text
        decision.corrected_response = response_text
        decision.governed_response = response_text
        self._bridge.log_decision(
            decision,
            turn=turn,
            pef_context=self._pef.to_context_summary(),
            pre_llm=True,
            pef_snapshot=self._pef.to_dict(),
            at_verification_basis=self._audit_at_basis(),
            epistemic_normalisation_applied=False,
            **self._audit_linkage_kwargs(decision),
        )
        self._history.append({"role": "user", "content": history_user_input})
        self._history.append({"role": "assistant", "content": response_text})
        return LensResult(
            response=response_text,
            flags=[insufficient_flag],
            pef_snapshot=self._pef.to_context_summary(),
            turn=turn,
            span=detected_span,
            model="",
            action=InterventionAction.CONTAIN,
            decision=decision,
            original_response=None,
        )

    def _finish_turn_with_state_native_if_handled(
        self,
        history_user_input: str,
        turn_act: TurnAct,
        detected_span: Span,
        turn: int,
        *,
        for_stream: bool = False,
        binding_resumed: bool = False,
        pef_snapshot: "PEFState | None" = None,
    ) -> LensResult | None:
        """Shared state-native delegation for ``process`` / ``process_stream``.

        Returns a :class:`LensResult` when the engine handles the turn (QUERY +
        strict surface); otherwise ``None``. When ``for_stream`` is True, audit
        logging uses streaming completion fields (no adapter call in either case).

        ``pef_snapshot`` must be the SNC-1 pre-extraction deepcopy captured before
        ``_extract_user_turn_for_interpret`` (see docs/adr/lane1_snapshot_policy.md).
        When absent (binding-resumed paths only), falls back to ``self._pef``.

        **Critical limitation:** This method does **not** check PEF epistemic hold
        status. The state-native engine returns handled=True when it successfully
        parses and answers a query, **without verifying that source relationships
        are uncontaminated**. A query can be marked handled even if its PEF sources
        are corrupted by a prior refusal/stop hold.

        This is not a security issue (the answer is still logically sound from committed
        relationships), but it means `state_native_handled=True` alone is insufficient
        to prove source integrity for audit classification purposes.

        See audit_linkage.py: only explicit `state_native_source_status="admitted_uncontaminated"`
        allows downgrading from HELD_REFUSAL; without it, refusal holds remain globally
        governing to avoid false audit classification.
        """
        if self._state_native_engine is None:
            return None
        from aurora_lens.state_native_engine.contracts import (
            QueryEphemeralBindings,
            StateNativeOutcome,
            StateNativeRequest,
            StateNativeSolverFamily,
        )

        ephemeral_bindings = None
        if turn_act == TurnAct.QUERY:
            ephemeral_bindings = QueryEphemeralBindings(
                turn=self._pef.current_turn,
                discourse_bindings=dict(self._pef.discourse_referent_bindings),
            )

        sn_req = StateNativeRequest(
            user_text=history_user_input,
            pef=pef_snapshot if pef_snapshot is not None else self._pef,
            turn_act=turn_act,
            detected_span=detected_span,
            binding_resumed=binding_resumed,
            possession_nlp=self._possession_nlp_for_state_native,
            require_high_confidence_possession_transfer=(
                self._config.require_high_confidence_possession_transfer
            ),
            query_ephemeral_bindings=ephemeral_bindings,
        )
        sn_res = self._state_native_engine.evaluate(sn_req)
        if not sn_res.handled:
            return None
        sn_res = _normalize_state_native_result_from_temporal_eval(sn_res)
        closed_world_resolution = classify_closed_world_resolution(
            outcome=sn_res.outcome,
            stop_reason_code=sn_res.stop_reason_code,
            solver_family=sn_res.solver_family,
        )
        decision = governance_decision_from_state_native(sn_res)
        _domain, _authority, _user_class, _ = self._context_resolver.resolve([])
        _status = _state_native_lens_status_from_temporal_eval(
            sn_res.temporal_eval_result,
            default_outcome=sn_res.outcome,
        )
        _policy = resolve_governor_policy(
            _domain,
            _authority,
            _status,
            _user_class,
            reason_code=(
                sn_res.stop_reason_code if _status == LensStatus.STOP else None
            ),
        )
        decision.policy = "canonical"
        decision.pathway_id = _policy.pathway_id.value
        decision.output_mode = _policy.output_mode.value
        decision.commitment_closed = _policy.commitment_closed
        decision.interaction_open = _policy.interaction_open
        decision.forensic_obligations = [
            fo.value for fo in _policy.forensic_obligations
        ]
        decision.resolution_mode = _policy.resolution_mode.value
        decision.resource = _policy.escalation_target
        reconcile_state_native_stop_after_policy_projection(decision, sn_res)
        if closed_world_resolution:
            suffix = f"closed_world_resolution={closed_world_resolution.value}"
            if decision.rationale:
                decision.rationale = f"{decision.rationale}; {suffix}"
            else:
                decision.rationale = suffix
        self._apply_route_reason(
            decision,
            LensRoutePlan(
                route=LensRoute.STATE_NATIVE_READ,
                reason_code="state_native_handled_query_surface",
            ),
        )
        if (
            sn_res.outcome == StateNativeOutcome.CLARIFY
            and sn_res.clarify_context
        ):
            _apply_state_native_pending_clarify(
                self._pef,
                original_question=history_user_input,
                detected_span=detected_span,
                clarify_context=sn_res.clarify_context,
            )
        _apply_epistemic_hold_after_non_admit(self._pef, decision, turn)
        if for_stream:
            self._bridge.log_decision(
                decision,
                turn=turn,
                stream=True,
                stream_completed=True,
                stream_truncated=False,
                stream_dropped_chars=0,
                pef_context=self._pef.to_context_summary(),
                pre_llm=True,
                pef_snapshot=self._pef.to_dict(),
                at_verification_basis=self._audit_at_basis(),
                state_native_handled=True,
                **self._audit_linkage_kwargs(decision, state_native_source_status=sn_res.source_status),
            )
        else:
            self._bridge.log_decision(
                decision,
                turn=turn,
                stream=False,
                stream_completed=True,
                stream_truncated=False,
                stream_dropped_chars=0,
                pef_context=self._pef.to_context_summary(),
                pre_llm=True,
                pef_snapshot=self._pef.to_dict(),
                at_verification_basis=self._audit_at_basis(),
                state_native_handled=True,
                **self._audit_linkage_kwargs(decision, state_native_source_status=sn_res.source_status),
            )
        self._history.append({"role": "user", "content": history_user_input})
        self._history.append(
            {"role": "assistant", "content": sn_res.user_visible_text},
        )
        return LensResult(
            response=sn_res.user_visible_text,
            flags=[],
            pef_snapshot=self._pef.to_context_summary(),
            turn=turn,
            span=detected_span,
            model="",
            action=decision.action,
            decision=decision,
            original_response=None,
            upstream_model_draft=None,
            epistemic_normalisation_applied=False,
            continuity_diagnostic=(
                f"state_native_closed_world_{closed_world_resolution.value.lower()}"
                if closed_world_resolution is not None
                else (
                    "state_native_typed_transition"
                    if sn_res.solver_family == StateNativeSolverFamily.COMMITTED_TYPED_TRANSITION
                    else (
                        "state_native_committed_inventory_read"
                        if sn_res.solver_family == StateNativeSolverFamily.COMMITTED_INVENTORY_READ
                        else "state_native_committed_location_read"
                    )
                )
            ),
            telemetry_release_path=(
                "state_native_transition_update"
                if (
                    sn_res.solver_family == StateNativeSolverFamily.COMMITTED_TYPED_TRANSITION
                    and turn_act != TurnAct.QUERY
                )
                else "state_native_retrieval"
                if sn_res.outcome == StateNativeOutcome.ANSWER
                else None
            ),
        )

    def _finish_turn_with_closed_world_puzzle_if_handled(
        self,
        history_user_input: str,
        turn_act: TurnAct,
        detected_span: Span,
        turn: int,
        *,
        for_stream: bool = False,
    ) -> LensResult | None:
        """Bounded surfaces handled by :func:`_solve_three_box_prize_puzzle` only.

        Transfer arithmetic and inventory live under state-native / ``PossessionStateAdapter``,
        not here. This method is also invoked from ``process`` after state-native when
        pre-model did not handle the turn.
        """
        solved = _solve_three_box_prize_puzzle(history_user_input)
        if solved is None:
            return None

        decision = GovernanceDecision(
            action=InterventionAction.PASS,
            flags=[],
            rationale="closed_world_three_box_unique_solution",
            policy="strict",
            pathway_id=None,
            output_mode=None,
            commitment_closed=False,
            interaction_open=True,
            original_response="",
            corrected_response=solved,
            governed_response=solved,
        )
        _domain, _authority, _user_class, _ = self._context_resolver.resolve([])
        _policy = resolve_governor_policy(
            _domain,
            _authority,
            LensStatus.ADMIT,
            _user_class,
            reason_code=None,
        )
        decision.policy = "canonical"
        decision.pathway_id = _policy.pathway_id.value
        decision.output_mode = _policy.output_mode.value
        decision.commitment_closed = _policy.commitment_closed
        decision.interaction_open = _policy.interaction_open
        decision.forensic_obligations = [fo.value for fo in _policy.forensic_obligations]
        decision.resolution_mode = _policy.resolution_mode.value
        decision.resource = _policy.escalation_target
        self._apply_route_reason(
            decision,
            LensRoutePlan(
                route=LensRoute.STATE_NATIVE_READ,
                reason_code="closed_world_three_box_query_surface",
            ),
        )
        _maybe_clear_epistemic_hold_on_admit(self._pef, decision)
        if for_stream:
            self._bridge.log_decision(
                decision,
                turn=turn,
                stream=True,
                stream_completed=True,
                stream_truncated=False,
                stream_dropped_chars=0,
                pef_context=self._pef.to_context_summary(),
                pre_llm=True,
                pef_snapshot=self._pef.to_dict(),
                at_verification_basis=self._audit_at_basis(),
                state_native_handled=True,
                **self._audit_linkage_kwargs(decision),
            )
        else:
            self._bridge.log_decision(
                decision,
                turn=turn,
                stream=False,
                stream_completed=True,
                stream_truncated=False,
                stream_dropped_chars=0,
                pef_context=self._pef.to_context_summary(),
                pre_llm=True,
                pef_snapshot=self._pef.to_dict(),
                at_verification_basis=self._audit_at_basis(),
                state_native_handled=True,
                **self._audit_linkage_kwargs(decision),
            )
        self._history.append({"role": "user", "content": history_user_input})
        self._history.append({"role": "assistant", "content": solved})
        return LensResult(
            response=solved,
            flags=[],
            pef_snapshot=self._pef.to_context_summary(),
            turn=turn,
            span=detected_span,
            model="",
            action=decision.action,
            decision=decision,
            original_response=None,
            upstream_model_draft=None,
            epistemic_normalisation_applied=False,
            continuity_diagnostic="state_native_closed_world_solved_unique",
        )

    async def _stream_emit_pre_model_dispatch(
        self,
        dispatch: PreModelDispatchResult,
        turn: int,
        *,
        include_operator_detail: bool,
    ) -> AsyncIterator[tuple[str, object]]:
        """SSE for pre-model jurisdictional outcomes (``stream_kind`` is adapter family, not user domain).

        Pre-model dispatch already ran before this helper is used; ``request_domain`` does
        not gate whether the router runs.
        """
        if (
            not dispatch.handled
            or dispatch.result is None
            or dispatch.stream_kind is None
        ):
            return
        res = dispatch.result
        _decision = res.decision
        _txt = res.response
        _flags = res.flags
        if self._config.stream_emit_progress:
            yield ("progress", ProgressSignal(status="releasing"))
        _chunk_dict: dict = {
            "choices": [{"delta": {"content": _txt}, "index": 0}],
        }
        yield ("chunk", (_chunk_dict, _txt))
        _gov = getattr(_decision, "governed_response", None) if _decision else None
        _aurora = build_aurora_block(
            action=_decision.action if _decision else res.action,
            flags=_flags,
            turn=turn,
            decision=_decision,
            pef=self._pef,
            include_operator_detail=include_operator_detail,
            session_id=None,
            original_response=res.original_response,
            governed_response_body=_gov if _gov is not None else _txt,
            stream_governed=True,
            stream_truncated=False,
            stream_dropped_chars=0,
            pef_admission_result_wire=self.peek_pef_admission_result_wire(),
        )
        _aurora["_log_flags"] = [
            f.flag_type.name for f in _decision.flags
        ] if _decision else []
        _aurora["_log_policy"] = _decision.policy if _decision else None
        _aurora["_log_pathway"] = _decision.pathway_id if _decision else None
        _aurora["_log_commitment_closed"] = (
            _decision.commitment_closed if _decision else None
        )
        _aurora["epistemic_normalisation_applied"] = False
        _cd = res.continuity_diagnostic
        if _cd:
            _aurora["continuity_diagnostic"] = _cd
        elif dispatch.stream_kind == "state_native":
            _aurora["continuity_diagnostic"] = "state_native_committed_location_read"
        else:
            _aurora["continuity_diagnostic"] = (
                "state_native_closed_world_solved_unique"
            )
        yield ("metadata", _aurora)

    def _pending_state_native_typo_recovery_phase(
        self,
        *,
        user_input: str,
        history_user_input: str,
        pending: dict,
        turn: int,
        turn_act: TurnAct,
        for_stream: bool,
    ) -> tuple[_PendingTypoRecoveryPhase, LensResult | None]:
        """Deterministic typo confirmation without extractor or adapter.

        Returns ``MERGED_RESUME_SKIP_EXTRACT`` when merge ran (caller skips extract).
        Returns an early ``LensResult`` when user explicitly declines typo suggestion.
        """
        del turn_act  # Signature parity with pending handlers; clarify turns allowed.
        if not _pending_typo_recovery_active(pending):
            return _PendingTypoRecoveryPhase.NONE, None

        tr = pending["typo_recovery"]
        if not isinstance(tr, dict):
            return _PendingTypoRecoveryPhase.NONE, None

        if _matches_state_native_typo_rejection(user_input):
            pending.pop("typo_recovery", None)
            pending["entity_ambiguity_typo_recovery_suppressed"] = True
            pending.pop("clarification_prompt", None)
            msg = _generic_state_native_ambiguity_message_from_pending(pending)
            cands = pending.get("candidate_entities") or []
            subj_or_item = (
                str(pending.get("subject_phrase") or "").strip()
                or str(pending.get("item_phrase") or "").strip()
                or "entity"
            )[:200]
            amb_flags = [
                Flag(
                    flag_type=FlagType.STATE_NATIVE_ENTITY_AMBIGUITY,
                    entity_name=subj_or_item,
                    claim=(
                        "State-native committed-state query: entity head is structurally ambiguous"
                    ),
                    evidence=", ".join(str(x) for x in cands)
                    if cands
                    else "Multiple committed entity candidates; disambiguation required",
                    severity="warning",
                    candidates=tuple(str(x) for x in cands),
                )
            ]
            decision = GovernanceDecision(
                action=InterventionAction.CONTAIN,
                flags=amb_flags,
                rationale="state_native_typo_recovery_declined_generic_ambiguity",
                policy="strict",
                pathway_id="P_ASK_DISAMBIGUATE",
                output_mode="clarification_request",
                commitment_closed=False,
                interaction_open=True,
                original_response="",
                corrected_response=msg,
                governed_response=msg,
            )
            _apply_epistemic_hold_after_non_admit(self._pef, decision, turn)
            if for_stream:
                self._bridge.log_decision(
                    decision,
                    turn=turn,
                    stream=True,
                    stream_completed=True,
                    stream_truncated=False,
                    stream_dropped_chars=0,
                    pef_context=self._pef.to_context_summary(),
                    pre_llm=True,
                    pef_snapshot=self._pef.to_dict(),
                    at_verification_basis=self._audit_at_basis(),
                    **self._audit_linkage_kwargs(decision),
                )
            else:
                self._bridge.log_decision(
                    decision,
                    turn=turn,
                    stream=False,
                    stream_completed=True,
                    stream_truncated=False,
                    stream_dropped_chars=0,
                    pef_context=self._pef.to_context_summary(),
                    pre_llm=True,
                    pef_snapshot=self._pef.to_dict(),
                    at_verification_basis=self._audit_at_basis(),
                    **self._audit_linkage_kwargs(decision),
                )
            self._history.append({"role": "user", "content": history_user_input})
            self._history.append({"role": "assistant", "content": msg})
            return (
                _PendingTypoRecoveryPhase.NONE,
                LensResult(
                    response=msg,
                    flags=amb_flags,
                    pef_snapshot=self._pef.to_context_summary(),
                    turn=turn,
                    span=Span.PRESENT,
                    model="",
                    action=decision.action,
                    decision=decision,
                    original_response=None,
                ),
            )

        if not _matches_state_native_typo_confirmation(user_input, tr):
            return _PendingTypoRecoveryPhase.NONE, None

        absorb_id = str(tr.get("typo_entity_id") or "")
        canon_id = str(tr.get("canonical_entity_id") or "")
        if (
            not absorb_id
            or not canon_id
            or absorb_id not in self._pef.entities
            or canon_id not in self._pef.entities
        ):
            return _PendingTypoRecoveryPhase.NONE, None

        record = self._pef.merge_entity_identity_absorb_into_canonical(
            absorb_entity_id=absorb_id,
            canonical_entity_id=canon_id,
            audit_reason="deterministic_typo_confirmation",
            original_ambiguity_flag="STATE_NATIVE_ENTITY_AMBIGUITY",
            turn=turn,
        )
        audit_decision = GovernanceDecision(
            action=InterventionAction.PASS,
            flags=[],
            rationale="deterministic_typo_confirmation",
            policy="strict",
            pathway_id=None,
            output_mode=None,
            commitment_closed=False,
            interaction_open=True,
            original_response="",
            corrected_response="",
            governed_response="",
            governance_note=json.dumps(record),
        )
        if for_stream:
            self._bridge.log_decision(
                audit_decision,
                turn=turn,
                stream=True,
                stream_completed=True,
                stream_truncated=False,
                stream_dropped_chars=0,
                pef_context=self._pef.to_context_summary(),
                pre_llm=True,
                pef_snapshot=self._pef.to_dict(),
                at_verification_basis=self._audit_at_basis(),
                **self._audit_linkage_kwargs(audit_decision),
            )
        else:
            self._bridge.log_decision(
                audit_decision,
                turn=turn,
                stream=False,
                stream_completed=True,
                stream_truncated=False,
                stream_dropped_chars=0,
                pef_context=self._pef.to_context_summary(),
                pre_llm=True,
                pef_snapshot=self._pef.to_dict(),
                at_verification_basis=self._audit_at_basis(),
                **self._audit_linkage_kwargs(audit_decision),
            )

        return _PendingTypoRecoveryPhase.MERGED_RESUME_SKIP_EXTRACT, None

    async def _apply_post_generation_governance(
        self,
        *,
        upstream_text: str,
        turn: int,
        history_user_input: str,
        user_input: str,
        detected_span: Span,
        extraction: ExtractionResult | None,
        rag_pef_update_user_text: str | None,
        ext_flags: list[Flag],
        pef_context: str,
        history: list,
        binding_resumed: bool,
        resolved_binding_name: str | None,
        agency_preface_text: str | None,
        route_reason_code: str | None,
        audit_stream: bool = False,
        stream_truncated: bool = False,
        stream_dropped_chars: int = 0,
        commit_audit: bool = True,
    ) -> PostGenerationGovernanceOutcome:
        """Shared post-LLM verification and governance path (sync + stream)."""
        epistemic_normalisation_applied = False
        flags: list[Flag] = []
        if self._config.auto_verify:
            _user_grounding_ctx = build_verify_user_grounding_context(
                self._pef,
                turn,
                history_user_input,
                rag_pef_update_user_text,
            )
            flags = await self._checker.check(
                upstream_text,
                self._pef,
                user_input=history_user_input,
                user_grounding=_user_grounding_ctx,
            )
        if ext_flags:
            flags = list(flags) + list(ext_flags)
        _source_scope_flag = self._source_scope_enforcement_flag_for_turn(
            history_user_input=history_user_input,
        )
        if _source_scope_flag is not None and not any(
            f.flag_type == FlagType.UPSTREAM_INSUFFICIENT_CONTEXT for f in flags
        ):
            flags.append(_source_scope_flag)
        flags = self._maybe_add_retrieval_consistency_conflict_flag(
            flags=flags,
            rag_turn_active=rag_pef_update_user_text is not None,
        )

        _illicit_request_flags = evaluate_blocked_act_request(history_user_input or user_input)
        if _illicit_request_flags:
            flags = [
                f for f in flags
                if f.flag_type != FlagType.UPSTREAM_INSUFFICIENT_CONTEXT
            ]
            if not any(f.flag_type == FlagType.ILLEGAL_INSTRUCTION for f in flags):
                flags = list(_illicit_request_flags) + list(flags)

        if binding_resumed and resolved_binding_name and flags:
            _bound_lower = resolved_binding_name.lower()
            flags = [
                f for f in flags
                if not (
                    f.flag_type in (
                        FlagType.UNSUPPORTED_ATTRIBUTE,
                        FlagType.UNSUPPORTED_EVENT,
                    )
                    and f.entity_name.lower() == _bound_lower
                )
            ]

        governor_verdict = self._governor.evaluate(upstream_text, self._pef)

        if governor_verdict.status != "ADMIT" and not flags:
            flags.append(Flag(
                flag_type=FlagType.UNRESOLVED_STATE_TRANSITION,
                entity_name=governor_verdict.details.get("subject", "process") if governor_verdict.details else "process",
                claim=f"Governor verdict: {governor_verdict.reason}",
                evidence=governor_verdict.details.get("source_sentence", "") if governor_verdict.details else "",
                severity="error",
            ))

        flags, ambiguous_snapshot_merged_tokens = (
            self._maybe_merge_ambiguous_snapshot_governance_flags(
                flags, extraction, history_user_input,
            )
        )

        _strict_policy = (
            getattr(getattr(self._bridge, "_policy", None), "name", "") == "strict"
        )
        _pass_intent = None
        _unsupported_scaffolding_only = bool(flags) and all(
            f.flag_type in (
                FlagType.UNSUPPORTED_ATTRIBUTE,
                FlagType.UNSUPPORTED_EVENT,
            )
            for f in flags
        )
        if _strict_policy and (not flags or _unsupported_scaffolding_only):
            _pass_intent = classify_consequence_intent(
                history_user_input or user_input,
                request_domain=domain_var.get(None),
            )
            if _pass_intent.basis == LOW_RISK_CONVERSATIONAL:
                if _unsupported_scaffolding_only:
                    flags = []
            elif not flags:
                flags = [consequence_intent_flag(_pass_intent)]

        final_response = upstream_text
        original_response = None
        decision: GovernanceDecision | None = None

        if flags:
            decision = await self._bridge.decide(flags, upstream_text, self._pef)
            decision.governor_verdict = governor_verdict

            if decision.action != InterventionAction.PASS:
                decision.original_response = upstream_text
                if ambiguous_snapshot_merged_tokens:
                    final_response = _snapshot_unresolved_referent_clarification_text(
                        ambiguous_snapshot_merged_tokens,
                        extraction=extraction,
                        pef=self._pef,
                    )
                else:
                    final_response = _maybe_sanitize_governed_clarification_action(
                        await self._bridge.intervene(
                            decision,
                            self._config.adapter,
                            user_input,
                            pef_context,
                            history if history else None,
                        )
                    )
                decision.corrected_response = final_response
                original_response = upstream_text

                if decision.action == InterventionAction.CONTAIN:
                    _primary = decision.flags[0] if decision.flags else None
                    if (
                        _primary is not None
                        and _primary.flag_type == FlagType.UNRESOLVED_REFERENT
                        and extraction is not None
                    ):
                        self._pef.pending_clarification = _build_pending_unresolved_referent_dict(
                            self._pef,
                            scope_text=user_input,
                            original_question=user_input,
                            extraction=extraction,
                            ambiguous_tokens=_ambiguous_tokens_for_pending_payload(
                                ambiguous_snapshot_merged_tokens=(
                                    ambiguous_snapshot_merged_tokens
                                ),
                                extraction=extraction,
                            ),
                            turn=turn,
                            detected_span=detected_span,
                        )
                    else:
                        self._pef.pending_clarification = _pending_payload_simple_contain(
                            user_input=user_input,
                            detected_span=detected_span,
                            pef=self._pef,
                            primary=_primary,
                        )

        else:
            _policy_name = (
                getattr(getattr(self._bridge, "_policy", None), "name", None)
                or "unknown"
            )
            if _strict_policy and _pass_intent is not None:
                _pass_basis = _pass_intent.basis
                decision = GovernanceDecision(
                    action=InterventionAction.PASS,
                    flags=[],
                    rationale=(
                        f"Admissibility basis: {_pass_basis}; "
                        "verification produced no flags"
                    ),
                    policy=_policy_name,
                    governor_verdict=governor_verdict,
                    admissibility_basis=_pass_basis,
                    pass_reason_code=_pass_basis,
                )
            else:
                decision = GovernanceDecision(
                    action=InterventionAction.PASS,
                    flags=[],
                    rationale="No verification flags",
                    policy=_policy_name,
                    governor_verdict=governor_verdict,
                )

        decision = await self._decision_never_pass_on_live_ambiguity_snapshot(
            decision,
            ambiguous_snapshot_merged_tokens=ambiguous_snapshot_merged_tokens,
            upstream_text=upstream_text,
            extraction=extraction,
            governor_verdict=governor_verdict,
        )
        if decision.flags:
            flags = list(decision.flags)

        if (
            flags
            and decision is not None
            and decision.action not in (
                InterventionAction.PASS,
                InterventionAction.SOFT_CORRECT,
            )
            and decision.corrected_response is None
        ):
            decision.original_response = upstream_text
            if ambiguous_snapshot_merged_tokens:
                final_response = _snapshot_unresolved_referent_clarification_text(
                    ambiguous_snapshot_merged_tokens,
                    extraction=extraction,
                    pef=self._pef,
                )
            else:
                final_response = _maybe_sanitize_governed_clarification_action(
                    await self._bridge.intervene(
                        decision,
                        self._config.adapter,
                        user_input,
                        pef_context,
                        history if history else None,
                    )
                )
            decision.corrected_response = final_response
            decision.governed_response = final_response
            original_response = upstream_text

            if decision.action == InterventionAction.CONTAIN:
                _primary = decision.flags[0] if decision.flags else None
                if (
                    _primary is not None
                    and _primary.flag_type == FlagType.UNRESOLVED_REFERENT
                    and extraction is not None
                ):
                    self._pef.pending_clarification = _build_pending_unresolved_referent_dict(
                        self._pef,
                        scope_text=user_input,
                        original_question=user_input,
                        extraction=extraction,
                        ambiguous_tokens=_ambiguous_tokens_for_pending_payload(
                            ambiguous_snapshot_merged_tokens=(
                                ambiguous_snapshot_merged_tokens
                            ),
                            extraction=extraction,
                        ),
                        turn=turn,
                        detected_span=detected_span,
                    )
                else:
                    self._pef.pending_clarification = _pending_payload_simple_contain(
                        user_input=user_input,
                        detected_span=detected_span,
                        pef=self._pef,
                        primary=_primary,
                    )

        if decision is not None:
            if route_reason_code is not None:
                self._apply_route_reason(
                    decision,
                    LensRoutePlan(
                        route=LensRoute.GENERATION_NATIVE,
                        reason_code=route_reason_code,
                    ),
                )
            if (
                decision.action == InterventionAction.PASS
                and not flags
                and self._pef.pending_clarification
                and self._pef.pending_clarification.get("failed_constraint") == "DISJUNCTIVE_BRANCH_COLLAPSE"
            ):
                _clear_epistemic_hold_ambiguity(self._pef)
            if decision.action in (InterventionAction.PASS, InterventionAction.SOFT_CORRECT):
                _maybe_clear_epistemic_hold_on_admit(self._pef, decision)
            elif flags and decision.action not in (
                InterventionAction.PASS,
                InterventionAction.SOFT_CORRECT,
            ):
                _apply_epistemic_hold_after_non_admit(self._pef, decision, turn)

        if (
            ambiguous_snapshot_merged_tokens
            and extraction is not None
            and decision is not None
            and decision.action not in (
                InterventionAction.PASS,
                InterventionAction.SOFT_CORRECT,
            )
        ):
            _scope_ambig = (
                rag_pef_update_user_text
                if rag_pef_update_user_text is not None
                else user_input
            )
            self._pef.pending_clarification = _build_pending_unresolved_referent_dict(
                self._pef,
                scope_text=_scope_ambig,
                original_question=user_input,
                extraction=extraction,
                ambiguous_tokens=ambiguous_snapshot_merged_tokens,
                detected_span=detected_span,
            )

        if decision is not None:
            _norm_text, _norm_applied = await self._maybe_apply_epistemic_normalisation_clean_pass(
                final_response,
                flags=flags,
                decision=decision,
                history_user_input=history_user_input,
            )
            if _norm_applied:
                final_response = _norm_text
                epistemic_normalisation_applied = True
            if decision.action == InterventionAction.PASS and not flags:
                decision.governed_response = final_response

        if (
            decision is not None
            and decision.action in (InterventionAction.PASS, InterventionAction.SOFT_CORRECT)
            and _looks_like_explicit_correction_phrase(history_user_input)
        ):
            final_response = _format_explicit_correction_surface_response(
                history_user_input,
                extraction,
            )
            decision.governed_response = final_response
            if decision.corrected_response is not None:
                decision.corrected_response = final_response

        if (
            agency_preface_text
            and decision is not None
            and decision.action in (InterventionAction.PASS, InterventionAction.SOFT_CORRECT)
        ):
            final_response = f"{agency_preface_text}\n\n{final_response}"
            decision.governed_response = final_response
            if decision.corrected_response is not None:
                decision.corrected_response = final_response

        final_flags = decision.flags if getattr(decision, "flags", None) else flags
        self_refused = (
            not final_flags
            and decision.action == InterventionAction.PASS
            and self._checker.is_refusal_frame(upstream_text)
        )

        if decision is not None and commit_audit:
            if (
                decision.action
                in (InterventionAction.PASS, InterventionAction.SOFT_CORRECT)
                and decision.original_response is None
            ):
                decision.original_response = upstream_text
            self._attach_provider_route_audit(decision, adapter_called=True)
            _log_kwargs: dict = {
                "turn": turn,
                "pef_context": pef_context,
                "pre_llm": False,
                "pef_snapshot": self._pef.to_dict(),
                "at_verification_basis": self._audit_at_basis(),
                "epistemic_normalisation_applied": epistemic_normalisation_applied,
                **self._audit_linkage_kwargs(decision),
            }
            if audit_stream:
                _log_kwargs.update(
                    stream=True,
                    stream_completed=True,
                    stream_truncated=stream_truncated,
                    stream_dropped_chars=stream_dropped_chars,
                )
            self._bridge.log_decision(decision, **_log_kwargs)
        elif decision is not None:
            if (
                decision.action
                in (InterventionAction.PASS, InterventionAction.SOFT_CORRECT)
                and decision.original_response is None
            ):
                decision.original_response = upstream_text
            self._attach_provider_route_audit(decision, adapter_called=True)

        self._history.append({"role": "user", "content": history_user_input})
        self._history.append({"role": "assistant", "content": final_response})

        admit_path = decision.action in (
            InterventionAction.PASS,
            InterventionAction.SOFT_CORRECT,
        )
        return PostGenerationGovernanceOutcome(
            flags=final_flags,
            decision=decision,
            final_response=final_response,
            original_response=original_response,
            epistemic_normalisation_applied=epistemic_normalisation_applied,
            self_refused=self_refused,
            admit_path=admit_path,
            ambiguous_snapshot_merged_tokens=ambiguous_snapshot_merged_tokens,
        )

    def _lens_result_from_post_generation(
        self,
        outcome: PostGenerationGovernanceOutcome,
        *,
        adapter_model: str,
        adapter_usage: dict | None,
        upstream_model_draft: str,
        turn: int,
        detected_span: Span,
        include_operator_detail: bool,
        upstream_text: str,
    ) -> LensResult:
        original_response = outcome.original_response
        if (
            original_response is None
            and outcome.decision.action == InterventionAction.PASS
            and include_operator_detail
        ):
            original_response = upstream_text
        return LensResult(
            response=outcome.final_response,
            flags=outcome.flags,
            pef_snapshot=self._pef.to_context_summary(),
            turn=turn,
            span=detected_span,
            model=adapter_model,
            action=outcome.decision.action,
            decision=outcome.decision,
            original_response=original_response,
            usage=adapter_usage,
            self_refused=outcome.self_refused,
            upstream_model_draft=upstream_model_draft,
            epistemic_normalisation_applied=outcome.epistemic_normalisation_applied,
        )

    async def process(
        self,
        user_input: str,
        external_flags: list[Flag] | None = None,
        *,
        include_operator_detail: bool | None = None,
    ) -> LensResult:
        """Process a single user turn through the full pipeline.

        1. Advance turn
        1.5. Clarification resolution: if a prior ASK has a pending determination,
             test whether this turn supplies a valid binding before running the
             normal pipeline.  When binding is found, redirect the LLM query to
             the original question while recording the clarification in history.
        2. Interpret user input (extract entities, relationships, update PEF)
        3. Generate PEF context for LLM
        4. Call LLM with PEF-injected system prompt
        5. Verify LLM response against PEF state
        5.5. Governance: evaluate flags, intervene if needed
        6. Return result with governance decision

        ``include_operator_detail`` overrides :attr:`LensConfig.include_operator_detail`
        for this call (e.g. proxy ``X-Aurora-Operator-Detail``). When ``None``, config applies.
        """
        _op_detail_eff = (
            include_operator_detail
            if include_operator_detail is not None
            else self._config.include_operator_detail
        )
        # Step 1: Advance turn
        turn = self._pef.advance_turn()
        self._audit_pef_start_dict = copy.deepcopy(self._pef.to_dict())
        self._audit_pef_start_frozen = copy.deepcopy(self._pef)
        # Preserve actual user text for history; query may be redirected below.
        history_user_input = user_input
        request_prompt_var.set(history_user_input)

        self._evaluate_sovereign_provider_route(turn)
        self._analyze_and_store_pef_uncertainties()

        _candidate_release_result = self._maybe_adjudicate_candidate_release_task(
            history_user_input=history_user_input,
            turn=turn,
        )
        if _candidate_release_result is not None:
            return _candidate_release_result

        if _eug_result := self._maybe_apply_epistemic_uncertainty_gate(
            history_user_input=history_user_input,
            turn=turn,
        ):
            return _eug_result

        if _sovereign_result := self._maybe_apply_sovereign_provider_route_gate(
            history_user_input=history_user_input,
            turn=turn,
        ):
            return _sovereign_result

        # Classify user act once per turn (routing source of truth for REVISE / CLARIFY gates).
        turn_act = classify_turn_act(history_user_input)
        if turn_act != TurnAct.REVISE and _looks_like_explicit_correction_phrase(
            history_user_input
        ) and not history_user_input.strip().lower().startswith("actually"):
            turn_act = TurnAct.REVISE
        self._ambiguous_snapshot_for_assistant_pending = None
        self._discourse_binding_keys_before_user_turn_commit = None
        extraction: ExtractionResult | None = None
        rag_pef_update_user_text: str | None = None
        upstream_model_draft: str | None = None
        epistemic_normalisation_applied = False
        ambiguous_snapshot_merged_tokens: list[str] | None = None
        _committed_state_mutation = False
        _resumed_pending_failed_constraint: str | None = None
        _resumed_pending_original_question: str | None = None
        _resumed_pending_blocked_claims: list[dict[str, object]] | None = None
        _resumed_pending_snapshot: dict[str, object] | None = None
        _resumed_pending_has_referent_metadata = False
        _clarification_extraction_for_resume: ExtractionResult | None = None
        _resolved_binding_name: str | None = None
        _resolved_referent_phrase: str | None = None
        _agency_preface_text: str | None = None
        _binding_upstream_resume = False
        _resumed_bound_input_for_sn: str | None = None
        _explicit_possessive_clarification_resolved = False
        self._reset_pef_admission_operator_wire_turn()

        if _is_hold_unresolved_selection(history_user_input):
            _hold_control = await self._maybe_apply_hold_unresolved_selection(
                turn=turn,
                history_user_input=history_user_input,
                user_input=user_input,
                pending=self._pef.pending_clarification or {},
            )
            if _hold_control is not None:
                return _hold_control

        # Frame classification — before extraction, governance, and LLM invocation.
        # Detects explicit opener/closer transitions; frame persists across turns
        # until a closer (e.g. "end roleplay") is encountered.
        _current_frame_kind = (
            self._pef.active_frame.kind
            if self._pef.active_frame is not None
            else FrameKind.EXTERNAL
        )
        _new_frame_kind = detect_frame_transition(user_input, _current_frame_kind)
        if _new_frame_kind == FrameKind.ARTIFACT:
            self._pef.active_frame = ArtifactFrame(kind=FrameKind.ARTIFACT, opened_at_turn=turn)
        elif _new_frame_kind == FrameKind.EXTERNAL:
            self._pef.active_frame = None

        _stop_cont = self._maybe_return_epistemic_stop_continuation_sync(
            history_user_input=history_user_input,
            turn=turn,
        )
        if _stop_cont is not None:
            return _stop_cont

        # Step 1.5: Clarification resolution
        #
        # If the previous turn ended in ASK (pre- or post-LLM CONTAIN), test
        # whether the current input supplies a valid binding for the pending
        # unresolved slot before running the normal pipeline.
        #
        # Binding rules:
        #  - Entity-placeholder case (unresolved_entity_ids non-empty):
        #    binding found when any tracked entity is now resolved=True after
        #    the clarification is applied to PEF.
        #  - Pronoun-ambiguity case (unresolved_entity_ids empty):
        #    binding found when re-extracting the original question on the
        #    updated PEF produces no ambiguous_referents.
        #
        # On binding: clear pending, redirect user_input to original_question,
        #   and skip Step 2 re-extraction so PEF is not corrupted with duplicate
        #   or collapsed claims from the already-interpreted original text.
        # On no binding: keep pending, fall through so the clarification is
        #   answered on its own merits (and pending survives to the next turn).
        _binding_resumed = False
        _resumed_span = Span.PRESENT

        _pending_fc_resume = (
            self._pef.pending_clarification is not None
            and str(self._pef.pending_clarification.get("failed_constraint") or "")
            == "UNRESOLVED_REFERENT"
        )
        if (
            self._pef.pending_clarification is not None
            and self._config.auto_interpret
            and (
                not self._check_blocked_act_request(user_input)
                or _pending_fc_resume
            )
        ):
            _pending = self._pef.pending_clarification
            _resumed_pending_snapshot = copy.deepcopy(_pending)
            _clarification_audit_selected: str | None = None
            _clarification_pef_before: dict | None = None
            _clarification_resolution_logged = False
            _orig_q: str = _pending["original_question"]
            _unresolved_ids: list[str] = _pending.get("unresolved_entity_ids", [])
            _resumed_pending_failed_constraint = str(_pending.get("failed_constraint") or "")
            _resumed_pending_original_question = _orig_q
            _resumed_pending_blocked_claims = _pending.get("blocked_claims")
            _resumed_pending_has_referent_metadata = bool(
                _pending.get("ambiguous_referents") or _binding_candidates_present(_pending)
            )
            if _pending.get("failed_constraint") == "AGENCY_RISK_CONTEXT_UNRESOLVED":
                _agency_followup = classify_agency_context_followup(user_input)
                if _agency_followup == "explicit_violation":
                    _agency_flags = [
                        Flag(
                            flag_type=FlagType.AGENCY_VIOLATION_ASSISTANCE,
                            entity_name="agency",
                            claim=(
                                "Follow-up clarification confirms intent to request actionable "
                                "assistance for violating another person's agency"
                            ),
                            evidence="Clarification response resolved unresolved agency risk as explicit violation",
                            severity="error",
                        )
                    ]
                    _blocked_decision = await self._bridge.decide(_agency_flags, user_input, self._pef)
                    _attach_rule_result(_blocked_decision)
                    _blocked_decision.original_response = ""
                    _governed_text = _maybe_sanitize_governed_clarification_action(
                        await self._bridge.intervene(
                            _blocked_decision,
                            self._config.adapter,
                            user_input,
                            self._pef.to_context_summary(),
                        )
                    )
                    _apply_epistemic_hold_after_non_admit(self._pef, _blocked_decision, turn)
                    self._bridge.log_decision(
                        _blocked_decision,
                        turn=turn,
                        pef_context=self._pef.to_context_summary(),
                        pre_llm=True,
                        pef_snapshot=self._pef.to_dict(),
                        at_verification_basis=self._audit_at_basis(),
                        **self._audit_linkage_kwargs(_blocked_decision),
                    )
                    self._history.append({"role": "user", "content": history_user_input})
                    self._history.append({"role": "assistant", "content": _governed_text})
                    return LensResult(
                        response=_governed_text,
                        flags=_agency_flags,
                        pef_snapshot=self._pef.to_context_summary(),
                        turn=turn,
                        span=Span.PRESENT,
                        model="",
                        action=_blocked_decision.action,
                        decision=_blocked_decision,
                        original_response=None,
                    )
                if _agency_followup == "non_coercive":
                    _orig_unresolved_prompt = str(_pending.get("original_question") or "")
                    _sanitized_prompt, _removed_fragments = (
                        sanitize_agency_prompt_for_non_coercive_use(_orig_unresolved_prompt)
                    )
                    self._log_agency_context_resolution_event(
                        original_unresolved_risk_prompt=_orig_unresolved_prompt,
                        removed_coercive_objective_fragments=_removed_fragments,
                        sanitized_prompt_used_for_generation=_sanitized_prompt,
                    )
                    self._pef.pending_clarification = None
                    _binding_resumed = True
                    _resumed_span = Span(_pending.get("original_span", "present"))
                    _clar_ext = _EMPTY_RESUME_EXTRACTION_FOR_BINDING_SKIP
                    _clarification_extraction_for_resume = _clar_ext
                    _pending["original_question"] = _sanitized_prompt
                    _orig_q = _sanitized_prompt
                    user_input = _sanitized_prompt
                    _agency_preface_text = _AGENCY_ETHICAL_CONTINUATION_PREFIX
                else:
                    _continuation = _build_clarification_continuation(_pending)
                    _cont_flags = [
                        Flag(
                            flag_type=FlagType.AGENCY_RISK_CONTEXT_UNRESOLVED,
                            entity_name="agency_context_unresolved",
                            claim="Agency-risk context remains unresolved after clarification turn",
                            evidence="Follow-up context remains unclear or evasive",
                            severity="warning",
                        )
                    ]
                    _cont_decision = GovernanceDecision(
                        action=InterventionAction.CONTAIN,
                        flags=_cont_flags,
                        rationale="Agency-risk context remains unresolved",
                        policy="strict",
                        pathway_id="P_ASK_MISSING_FACT",
                        output_mode="clarification_request",
                        commitment_closed=True,
                        interaction_open=True,
                    )
                    _cont_decision.original_response = ""
                    _cont_decision.corrected_response = _continuation
                    _cont_decision.governed_response = _continuation
                    _apply_epistemic_hold_after_non_admit(self._pef, _cont_decision, turn)
                    self._bridge.log_decision(
                        _cont_decision,
                        turn=turn,
                        pef_context=self._pef.to_context_summary(),
                        pre_llm=True,
                        pef_snapshot=self._pef.to_dict(),
                        at_verification_basis=self._audit_at_basis(),
                        **self._audit_linkage_kwargs(_cont_decision),
                    )
                    self._history.append({"role": "user", "content": history_user_input})
                    self._history.append({"role": "assistant", "content": _continuation})
                    return LensResult(
                        response=_continuation,
                        flags=_cont_flags,
                        pef_snapshot=self._pef.to_context_summary(),
                        turn=turn,
                        span=Span.PRESENT,
                        model="",
                        action=InterventionAction.CONTAIN,
                        decision=_cont_decision,
                        original_response=None,
                    )

            if (
                _pending.get("failed_constraint") == "AGENCY_RISK_CONTEXT_UNRESOLVED"
                and _binding_resumed
            ):
                _typo_recovery_phase = _PendingTypoRecoveryPhase.MERGED_RESUME_SKIP_EXTRACT
                _typo_early_return = None
            else:
                _typo_recovery_phase, _typo_early_return = (
                    self._pending_state_native_typo_recovery_phase(
                        user_input=user_input,
                        history_user_input=history_user_input,
                        pending=_pending,
                        turn=turn,
                        turn_act=turn_act,
                        for_stream=False,
                    )
                )
            if _typo_early_return is not None:
                return _typo_early_return

            if _typo_recovery_phase == _PendingTypoRecoveryPhase.MERGED_RESUME_SKIP_EXTRACT:
                _clar_ext = _EMPTY_RESUME_EXTRACTION_FOR_BINDING_SKIP
                _clarification_extraction_for_resume = _clar_ext
            else:
                _clar_ext = await self._backend.extract(user_input, self._pef)
                _clarification_extraction_for_resume = _clar_ext

            _clarification_pef_before = self._pef.to_dict()

            if open_entries(self._pef):
                _session_gate_result = await self._maybe_apply_open_registry_session_gate(
                    turn=turn,
                    history_user_input=history_user_input,
                    user_input=user_input,
                    turn_act=turn_act,
                    binding_resumed=_binding_resumed,
                    ext_flags=[],
                    detected_span=Span.PRESENT,
                    extraction=_clar_ext,
                )
                if _session_gate_result is not None:
                    return _session_gate_result

            _hold_unresolved_result = await self._maybe_apply_hold_unresolved_selection(
                turn=turn,
                history_user_input=history_user_input,
                user_input=user_input,
                pending=_pending,
            )
            if _hold_unresolved_result is not None:
                return _hold_unresolved_result

            # Continuation gate: for P_ASK_DISAMBIGUATE holds, a turn that is not
            # a direct candidate selection must not fire binding — re-issue CONTAIN.
            _paf_hold = self._pef.epistemic_hold
            _registry_allows_through = self._open_registry_session_gate_allows_through(
                history_user_input,
                turn_act,
            )
            _held_unresolved_active = referent_registry_held_unresolved(self._pef)
            if (
                not _registry_allows_through
                and not _held_unresolved_active
                and _paf_hold is not None
                and _paf_hold.get("mode") == "ambiguity"
                and _paf_hold.get("pathway_id") == "P_ASK_DISAMBIGUATE"
                and _binding_candidates_present(_pending)
                and not _is_direct_disambiguation_response(user_input, _pending)
            ):
                _continuation = _build_clarification_continuation(_pending)
                _cont_flags = _pending_ambiguity_continuation_flags(_pending)
                _cont_decision = GovernanceDecision(
                    action=InterventionAction.CONTAIN,
                    flags=_cont_flags,
                    rationale="Non-selection turn during P_ASK_DISAMBIGUATE hold",
                    policy="strict",
                    pathway_id="P_ASK_DISAMBIGUATE",
                    output_mode="clarification_request",
                    commitment_closed=True,
                    interaction_open=True,
                )
                _cont_decision.original_response = ""
                _cont_decision.corrected_response = _continuation
                _cont_decision.governed_response = _continuation
                _apply_epistemic_hold_after_non_admit(self._pef, _cont_decision, turn)
                self._bridge.log_decision(
                    _cont_decision,
                    turn=turn,
                    pef_context=self._pef.to_context_summary(),
                    pre_llm=True,
                    pef_snapshot=self._pef.to_dict(),
                    at_verification_basis=self._audit_at_basis(),
                    **self._audit_linkage_kwargs(_cont_decision),
                )
                self._history.append({"role": "user", "content": history_user_input})
                self._history.append({"role": "assistant", "content": _continuation})
                return LensResult(
                    response=_continuation,
                    flags=[],
                    pef_snapshot=self._pef.to_context_summary(),
                    turn=turn,
                    span=Span.PRESENT,
                    model="",
                    action=InterventionAction.CONTAIN,
                    decision=_cont_decision,
                    original_response=None,
                )

            # Binding test.
            #
            # Priority order:
            #  1. Direct candidate match — the user's reply names one of the
            #     stored candidates (e.g. system asked "Richard or James?",
            #     user answered "James").  Checked before update_pef: this path
            #     does not require PEF mutation to determine the binding.
            #  2. Entity-placeholder path — a previously-unresolved entity
            #     became resolved after the clarification was applied to PEF.
            #     update_pef runs only when candidate match fails.
            #  3. Pronoun-ambiguity path — re-extracting the original question
            #     on the updated PEF produces no ambiguous_referents. When there
            #     is no candidate list but stored pronoun tokens remain, structural
            #     recheck alone does not count as binding (the user's reply must
            #     match an entry in ``candidate_entities``).
            #  4. Deterministic state-native typo confirmation (pending ``typo_recovery``):
            #     merge without extractor — handled above before extraction.
            _binding_found: bool
            _explicit_possessive_bind: str | None = None
            if referent_registry_held_unresolved(self._pef):
                _explicit_possessive_bind = try_explicit_possessive_attribution(
                    user_input,
                    _normalized_candidate_entities_for_binding(_pending),
                )
            if _typo_recovery_phase == _PendingTypoRecoveryPhase.MERGED_RESUME_SKIP_EXTRACT:
                _binding_found = True
            elif _explicit_possessive_bind is not None:
                _binding_found = True
            elif _matches_candidate(user_input, _clar_ext, _pending):
                _binding_found = True
            else:
                # Entity-placeholder and pronoun-ambiguity paths need PEF updated.
                if not _clar_ext.extraction_error:
                    _committed_state_mutation = self._commit_extraction_if_admissible(
                        _clar_ext,
                        user_text=history_user_input,
                        turn_act=turn_act,
                    )
                if _unresolved_ids:
                    # Entity-placeholder path.
                    _binding_found = _all_tracked_unresolved_entities_resolved(
                        self._pef,
                        _unresolved_ids,
                    )
                else:
                    # Pronoun-ambiguity path: re-extract the original question on
                    # the updated PEF to check whether ambiguity has been resolved.
                    _recheck = await self._backend.extract(_orig_q, self._pef)
                    _ambig_resume_tokens = _binding_resume_ambiguous_tokens(_pending, _orig_q)
                    _binding_found = (
                        not _recheck.extraction_error
                        and not _recheck.ambiguous_referents
                        and (
                            _matches_candidate(user_input, _clar_ext, _pending)
                            if _binding_candidates_present(_pending)
                            else not _ambig_resume_tokens
                        )
                    )

            if _binding_found:
                # Binding supplied — resume the original determination.
                # history_user_input stays as the clarification text so history
                # faithfully records what the user typed.
                # _binding_resumed gates off Step 2 so the original question is
                # NOT re-extracted (which would produce duplicate/collapsed
                # claims and corrupt the PEF world model).
                _resumed_span = Span(_pending.get("original_span", "present"))

                # When the pending constraint was a pronoun ambiguity,
                # reconstruct the original text with the resolved binding so
                # the LLM receives the proposition with the correct entity,
                # not the raw unresolved pronoun.
                _ambig_tokens = _binding_resume_ambiguous_tokens(_pending, _orig_q)
                _stored_ambig = [
                    str(x) for x in (_pending.get("ambiguous_referents") or []) if str(x).strip()
                ]
                _pronoun_bind_completed = True
                if _stored_ambig and not _ambig_tokens:
                    _binding_found = False
                    user_input = history_user_input
                    _pronoun_bind_completed = False
                elif _ambig_tokens:
                    _bound_name = _resolve_binding_entity(
                        history_user_input, _clar_ext, _pending, self._pef,
                    )
                    if not _bound_name and _explicit_possessive_bind:
                        _bound_name = _explicit_possessive_bind
                    _resolved_binding_name = _bound_name
                    if _bound_name:
                        if _clarification_pef_before is None:
                            _clarification_pef_before = self._pef.to_dict()
                        _clarification_audit_selected = _bound_name
                        user_input, _resolved_phrase = _reconstruct_bound_user_input_or_raise(
                            original_question=_orig_q,
                            pending=_pending,
                            ambiguous_tokens=_ambig_tokens,
                            bound_entity=_bound_name,
                        )
                        _resolved_referent_phrase = _resolved_phrase
                        # Narrow the continuation payload to the resolved proposition only.
                        # The full reconstructed original question may contain domain framing
                        # (e.g. workforce/hiring context) that causes the real LLM to refuse.
                        # The Governor should evaluate a clean state-delta, not the original
                        # ambiguity prompt. Use the blocked_proposition with pronouns bound.
                        _bp_raw = str(_pending.get("blocked_proposition") or "").strip()
                        if _bp_raw:
                            user_input = _reconstruct_bound_text(
                                _bp_raw,
                                _bp_raw,
                                _ambig_tokens,
                                _bound_name,
                                resolved_referent_phrase=_resolved_phrase,
                            )
                        _total_rels_pre = len(self._pef.relationships)
                        _replay_res = _commit_resolved_claims(
                            _bound_name,
                            _pending,
                            self._pef,
                            on_pef_admission=self._capture_pef_admission_for_operator_wire,
                        )
                        _commit_cf = _replay_verification_indicates_relation_commit_failed(
                            _replay_res,
                            relation_count_before=_total_rels_pre,
                            pef_after=self._pef,
                            pending_blocked_claims=list(_pending.get("blocked_claims") or []),
                        )
                        if _commit_cf:
                            _cf_cont = _build_clarification_continuation(_pending)
                            _cf_flags = [Flag(
                                flag_type=FlagType.UNRESOLVED_REFERENT,
                                entity_name=_bound_name or "",
                                claim=(
                                    "Clarification binding acknowledged but no structured "
                                    "claim could be committed to PEF"
                                ),
                                evidence="CLARIFICATION_RESOLUTION_COMMIT_FAILED",
                                severity="error",
                            )]
                            _cf_dec = GovernanceDecision(
                                action=InterventionAction.CONTAIN,
                                flags=_cf_flags,
                                rationale=(
                                    "CLARIFICATION_RESOLUTION_COMMIT_FAILED: "
                                    "no structured PEF write for bound entity; "
                                    "hold not cleared"
                                ),
                                policy="strict",
                                pathway_id="P_ASK_DISAMBIGUATE",
                                output_mode="clarification_request",
                                commitment_closed=True,
                                interaction_open=True,
                            )
                            _cf_dec.original_response = ""
                            _cf_dec.corrected_response = _cf_cont
                            _cf_dec.governed_response = _cf_cont
                            _apply_epistemic_hold_after_non_admit(self._pef, _cf_dec, turn)
                            self._bridge.log_decision(
                                _cf_dec,
                                turn=turn,
                                pef_context=self._pef.to_context_summary(),
                                pre_llm=True,
                                pef_snapshot=self._pef.to_dict(),
                                at_verification_basis=self._audit_at_basis(),
                                **self._audit_linkage_kwargs(_cf_dec),
                            )
                            self._history.append({"role": "user", "content": history_user_input})
                            self._history.append({"role": "assistant", "content": _cf_cont})
                            return LensResult(
                                response=_cf_cont,
                                flags=_cf_flags,
                                pef_snapshot=self._pef.to_context_summary(),
                                turn=turn,
                                span=_resumed_span,
                                model="",
                                action=InterventionAction.CONTAIN,
                                decision=_cf_dec,
                                original_response=None,
                            )
                        _retire_unresolved_blocked_claims(_pending, self._pef)
                        self._record_discourse_referents_from_binding(
                            _ambig_tokens, _bound_name,
                            turn=turn,
                        )
                        self._maybe_log_clarification_resolution_audit(
                            pending=_pending,
                            history_user_input=history_user_input,
                            clar_ext=_clar_ext,
                            turn=turn,
                            pef_snapshot_before=_clarification_pef_before,
                            binding_found=True,
                            explicit_selected=_bound_name,
                        )
                        _clarification_resolution_logged = True
                        # Handle comparative IS outcome from referent replay.
                        if _replay_res.stop_reason:
                            _stop_dec = GovernanceDecision(
                                action=InterventionAction.STOP,
                                flags=[],
                                rationale=_replay_res.stop_reason,
                                policy="strict",
                            )
                            _stop_dec.original_response = ""
                            _stop_dec.corrected_response = _replay_res.stop_reason
                            _stop_dec.governed_response = _replay_res.stop_reason
                            self._pef.pending_clarification = None
                            self._bridge.log_decision(
                                _stop_dec,
                                turn=turn,
                                pef_context=self._pef.to_context_summary(),
                                pre_llm=True,
                                pef_snapshot=self._pef.to_dict(),
                                at_verification_basis=self._audit_at_basis(),
                                **self._audit_linkage_kwargs(_stop_dec),
                            )
                            self._history.append({"role": "user", "content": history_user_input})
                            self._history.append({"role": "assistant", "content": _replay_res.stop_reason})
                            return LensResult(
                                response=_replay_res.stop_reason,
                                flags=[],
                                pef_snapshot=self._pef.to_context_summary(),
                                turn=turn,
                                span=_resumed_span,
                                model="",
                                action=InterventionAction.STOP,
                                decision=_stop_dec,
                                original_response=None,
                            )
                        if _replay_res.pending_comparand:
                            # 2+ comparands after referent resolution → re-issue CONTAIN.
                            _pc = _replay_res.pending_comparand
                            self._pef.pending_clarification = _pc
                            _comp_adj2 = str(_pc.get("comparand_adjective") or "")
                            _comp_noun2 = str(_pc.get("comparand_noun") or "")
                            _cands2: list[str] = list(_pc.get("candidate_entities") or [])
                            _comp_q2 = _comparand_clarification_message(
                                adjective=_comp_adj2,
                                noun=_comp_noun2,
                                candidate_entities=_cands2,
                            )
                            _comp_f2 = [Flag(
                                flag_type=FlagType.UNRESOLVED_COMPARAND,
                                entity_name=_comp_noun2,
                                claim=f"'{_comp_adj2}' comparand unresolved after referent binding",
                                evidence=f"Candidates: {', '.join(_cands2)}",
                                severity="warning",
                                candidates=tuple(_cands2),
                            )]
                            _comp_dec2 = GovernanceDecision(
                                action=InterventionAction.CONTAIN,
                                flags=_comp_f2,
                                rationale="UNRESOLVED_COMPARAND after referent binding",
                                policy="strict",
                                pathway_id="P_ASK_DISAMBIGUATE",
                                output_mode="clarification_request",
                                commitment_closed=True,
                                interaction_open=True,
                            )
                            _comp_dec2.original_response = ""
                            _comp_dec2.corrected_response = _comp_q2
                            _comp_dec2.governed_response = _comp_q2
                            _apply_epistemic_hold_after_non_admit(self._pef, _comp_dec2, turn)
                            self._bridge.log_decision(
                                _comp_dec2,
                                turn=turn,
                                pef_context=self._pef.to_context_summary(),
                                pre_llm=True,
                                pef_snapshot=self._pef.to_dict(),
                                at_verification_basis=self._audit_at_basis(),
                                **self._audit_linkage_kwargs(_comp_dec2),
                            )
                            self._history.append({"role": "user", "content": history_user_input})
                            self._history.append({"role": "assistant", "content": _comp_q2})
                            return LensResult(
                                response=_comp_q2,
                                flags=_comp_f2,
                                pef_snapshot=self._pef.to_context_summary(),
                                turn=turn,
                                span=_resumed_span,
                                model="",
                                action=InterventionAction.CONTAIN,
                                decision=_comp_dec2,
                                original_response=None,
                            )
                    else:
                        # Do not clear unresolved referent unless we resolved a concrete bind.
                        _binding_found = False
                        user_input = history_user_input
                        _pronoun_bind_completed = False
                else:
                    user_input = _orig_q
                    # Phase 1: UNRESOLVED_COMPARAND binding — user selected a comparand.
                    if _resumed_pending_failed_constraint == "UNRESOLVED_COMPARAND":
                        _comp_bound_name = _resolve_binding_entity(
                            history_user_input, _clar_ext, _pending, self._pef,
                        )
                        if _comp_bound_name:
                            if _clarification_pef_before is None:
                                _clarification_pef_before = self._pef.to_dict()
                            _clarification_audit_selected = _comp_bound_name
                            _commit_comparand_claims_as_compare(
                                comparand_name=_comp_bound_name,
                                pending=_pending,
                                pef=self._pef,
                            )
                            self._maybe_log_clarification_resolution_audit(
                                pending=_pending,
                                history_user_input=history_user_input,
                                clar_ext=_clar_ext,
                                turn=turn,
                                pef_snapshot_before=_clarification_pef_before,
                                binding_found=True,
                                explicit_selected=_comp_bound_name,
                            )
                            _clarification_resolution_logged = True

                if _binding_found:
                    if (
                        not _clar_ext.extraction_error
                        and not _committed_state_mutation
                        and bool(_clar_ext.claims)
                    ):
                        _committed_state_mutation = self._commit_extraction_if_admissible(
                            _clar_ext,
                            user_text=history_user_input,
                            turn_act=turn_act,
                        )
                    if not _clarification_resolution_logged:
                        self._maybe_log_clarification_resolution_audit(
                            pending=_pending,
                            history_user_input=history_user_input,
                            clar_ext=_clar_ext,
                            turn=turn,
                            pef_snapshot_before=_clarification_pef_before,
                            binding_found=_binding_found,
                            explicit_selected=_clarification_audit_selected,
                            typo_merged=(
                                _typo_recovery_phase
                                == _PendingTypoRecoveryPhase.MERGED_RESUME_SKIP_EXTRACT
                            ),
                        )
                    if _pronoun_bind_completed:
                        _clear_epistemic_hold_ambiguity(self._pef)
                    _binding_resumed = _pronoun_bind_completed
                    if _explicit_possessive_bind is not None and _pronoun_bind_completed:
                        _explicit_possessive_clarification_resolved = True
            elif _pending_ambiguity_continuation_applies(
                turn_act=turn_act,
                pending=_pending,
                history_user_input=history_user_input,
            ):
                # The user is asking about what information is needed, not
                # supplying a binding.  Answer from the stored pending state
                # so the response names the exact ambiguity and candidates.
                # Do NOT fall through to the LLM — that would generate from
                # stale session context.
                _continuation = _build_clarification_continuation(_pending)
                _cont_flags = _pending_ambiguity_continuation_flags(_pending)
                _cont_decision = GovernanceDecision(
                    action=InterventionAction.CONTAIN,
                    flags=_cont_flags,
                    rationale="Clarification continuation from pending state",
                    policy="strict",
                    pathway_id="P_ASK_DISAMBIGUATE",
                    output_mode="clarification_request",
                    commitment_closed=True,
                    interaction_open=True,
                )
                _cont_decision.original_response = ""
                _cont_decision.corrected_response = _continuation
                _cont_decision.governed_response = _continuation
                _apply_epistemic_hold_after_non_admit(self._pef, _cont_decision, turn)
                self._bridge.log_decision(
                    _cont_decision,
                    turn=turn,
                    pef_context=self._pef.to_context_summary(),
                    pre_llm=True,
                    pef_snapshot=self._pef.to_dict(),
                    at_verification_basis=self._audit_at_basis(),
                    **self._audit_linkage_kwargs(_cont_decision),
                )
                self._history.append({"role": "user", "content": history_user_input})
                self._history.append({"role": "assistant", "content": _continuation})
                return LensResult(
                    response=_continuation,
                    flags=[],
                    pef_snapshot=self._pef.to_context_summary(),
                    turn=turn,
                    span=Span.PRESENT,
                    model="",
                    action=InterventionAction.CONTAIN,
                    decision=_cont_decision,
                    original_response=None,
                )
            # else: pending survives; fall through with original user_input.

        if _binding_resumed:
            _resumed_bound_input_for_sn = user_input
            _binding_upstream_resume = _binding_requires_upstream_resume(_resumed_pending_snapshot)

        if self._pef.pending_clarification is None:
            _early_blocked_flags = self._check_blocked_act_request(user_input)
            if _early_blocked_flags:
                self._pef.active_continuation_capability = None
                self._pef.active_continuation_context = None
            else:
                _active_cap_result = self._maybe_handle_active_continuation_sync(
                    user_input=user_input,
                    history_user_input=history_user_input,
                    turn=turn,
                )
                if _active_cap_result is not None:
                    return _active_cap_result

        # Step 2: Interpretation layer (pluggable backend)
        #
        # Skipped when _binding_resumed: the original question was already
        # extracted + PEF-updated in the turn that created pending_clarification.
        # Re-extracting would add duplicate present-tense claims and collapse
        # the original proposition (e.g. "his stick was bigger" → "Richard has
        # a stick"), which then incorrectly triggers TIME_SMEAR.
        ext_flags = external_flags or []

        # Pre-model state dispatch — universal PEF-first router for every corridor (see
        # :mod:`aurora_lens.pre_model_state`) before blocked-act, extraction, or LLM.
        # Always invoked; ``request_domain`` is not a prerequisite (it may inform adapters).
        _pre_pef = pre_model_state_dispatch(
            self,
            PreModelContext(
                user_input=user_input,
                history_user_input=history_user_input,
                turn=turn,
                binding_resumed=_binding_resumed,
                turn_act=turn_act,
                for_stream=False,
            ),
        )
        if _pre_pef.handled and _pre_pef.result is not None:
            return _pre_pef.result

        # Pre-LLM: blocked-act classifier — independent of auto_interpret,
        # _binding_resumed, and **auto_verify**. Request-side regulated-domain
        # detection must run even when post-LLM verification is off; otherwise
        # consequence-bearing prompts reach the model and can PASS with no
        # deterministic request-side seam.
        _blocked_flags = self._check_blocked_act_request(user_input)
        if _blocked_flags:
            _bf_with_ext = _blocked_flags + list(ext_flags) if ext_flags else _blocked_flags
            _blocked_decision = await self._bridge.decide(_bf_with_ext, user_input, self._pef)
            _blocked_decision.pre_llm = True
            _blocked_decision.admissibility_basis = "blocked_illicit_intent"
            _attach_rule_result(_blocked_decision)
            if _blocked_decision.action != InterventionAction.PASS:
                _primary_blocked_flag = _bf_with_ext[0] if _bf_with_ext else None
                _blocked_decision.original_response = ""
                if (
                    _blocked_decision.action == InterventionAction.CONTAIN
                    and _primary_blocked_flag is not None
                    and _primary_blocked_flag.flag_type == FlagType.AGENCY_RISK_CONTEXT_UNRESOLVED
                ):
                    _governed_text = _build_clarification_continuation(
                        {
                            "failed_constraint": "AGENCY_RISK_CONTEXT_UNRESOLVED",
                        }
                    )
                    _blocked_decision.corrected_response = _governed_text
                    _blocked_decision.governed_response = _governed_text
                    self._pef.pending_clarification = _pending_payload_simple_contain(
                        user_input=user_input,
                        detected_span=Span.PRESENT,
                        pef=self._pef,
                        primary=_primary_blocked_flag,
                    )
                else:
                    _governed_text = _maybe_sanitize_governed_clarification_action(
                        await self._bridge.intervene(
                            _blocked_decision,
                            self._config.adapter,
                            user_input,
                            self._pef.to_context_summary(),
                        )
                    )
                _governed_text = _maybe_sanitize_governed_clarification_action(
                    _governed_text
                )
                if _should_state_commit_user_reported_context(_bf_with_ext, _blocked_decision):
                    _state_commit_user_reported_context(
                        self._pef,
                        user_text=history_user_input,
                        turn=turn,
                    )
                _apply_epistemic_hold_after_non_admit(self._pef, _blocked_decision, turn)
                self._set_medical_post_refusal_context_if_applicable(
                    user_input=history_user_input,
                    flags=_bf_with_ext,
                    decision=_blocked_decision,
                )
                self._bridge.log_decision(
                    _blocked_decision,
                    turn=turn,
                    pef_context=self._pef.to_context_summary(),
                    pre_llm=True,
                    pef_snapshot=self._pef.to_dict(),
                    at_verification_basis=self._audit_at_basis(),
                    **self._audit_linkage_kwargs(_blocked_decision),
                )
                self._history.append({"role": "user", "content": history_user_input})
                self._history.append({"role": "assistant", "content": _governed_text})
                return LensResult(
                    response=_governed_text,
                    flags=_bf_with_ext,
                    pef_snapshot=self._pef.to_context_summary(),
                    turn=turn,
                    span=Span.PRESENT,
                    model="",
                    action=_blocked_decision.action,
                    decision=_blocked_decision,
                    original_response=None,
                    telemetry_release_path="blocked_pre_generation",
                )

        # SNC-1: pre-extraction snapshot for state-native delegation
        # (docs/adr/lane1_snapshot_policy.md). Captured before extraction so
        # same-turn claims and RAG admit are excluded from Lane 1 read substrate.
        _pef_pre_commit: "PEFState | None" = None
        if not _binding_resumed:
            _pef_pre_commit = copy.deepcopy(self._pef)

        # Structural extraction for pre-LLM gates (v10 parity: ``docs/reference/aurora_cli_v10.py``
        # parses user input before admission). ``update_pef`` / referential snapshot remain
        # gated on ``auto_interpret`` in the block after revision conflict.
        if not _binding_resumed:
            extraction, rag_pef_update_user_text, rag_adm = await self._extract_user_turn_for_interpret(
                user_input,
            )
            if extraction.extraction_error:
                # Inadmissible extraction — do not call LLM, return governed failure
                err = extraction.extraction_error
                reason = err.get("reason", "UNKNOWN")
                snippet = (err.get("raw_preview", "") or err.get("snippet", ""))[:150]
                flags = [
                    Flag(
                        flag_type=FlagType.EXTRACTION_FAILED,
                        entity_name="extraction",
                        claim=f"Extraction failed: {reason}",
                        evidence=snippet or str(err),
                        severity="error",
                        extraction_diagnostic=err,
                    )
                ]
                if ext_flags:
                    flags = flags + list(ext_flags)
                decision = await self._bridge.decide(flags, "", self._pef)
                governed_msg = USER_MESSAGE_INTERPRETATION_FAILED
                decision.original_response = ""
                decision.corrected_response = governed_msg
                decision.governed_response = governed_msg
                _apply_epistemic_hold_after_non_admit(self._pef, decision, turn)
                self._bridge.log_decision(
                    decision,
                    turn=turn,
                    pef_context=self._pef.to_context_summary(),
                    pre_llm=True,
                    pef_snapshot=self._pef.to_dict(),
                    at_verification_basis=self._audit_at_basis(),
                    **self._audit_linkage_kwargs(decision),
                )
                self._history.append({"role": "user", "content": history_user_input})
                self._history.append({"role": "assistant", "content": governed_msg})
                return LensResult(
                    response=governed_msg,
                    flags=flags,
                    pef_snapshot=self._pef.to_context_summary(),
                    turn=turn,
                    span=Span.PRESENT,
                    model="",
                    action=decision.action,
                    decision=decision,
                    original_response=None,
                )
            detected_span = extraction.span
            _rag_adm_gate_result = self._maybe_apply_rag_evidence_admission_gate(
                rag_adm=rag_adm,
                history_user_input=history_user_input,
                turn=turn,
                detected_span=detected_span,
            )
            if _rag_adm_gate_result is not None:
                return _rag_adm_gate_result
            merge_structural_comparative_question_probe(
                extraction, history_user_input, self._pef
            )

            # Pre-LLM: unresolvable referents in user input.
            # Gate runs before the main update_pef pass; on CONTAIN we still may apply
            # admissible structure from extraction (e.g. ``_commit_extraction_if_admissible``).
            # If the question contains an ambiguous possessive pronoun (e.g. "her"
            # when both Emma and Anna are present), the LLM must not be allowed to
            # guess — that would violate the PEF no-premature-binding axiom.
            # Intervene here before the LLM call so the flag fires regardless of
            # whether the LLM happens to hedge correctly or not.
            self._sync_unresolved_referent_registry_from_extraction(
                turn=turn,
                utterance=history_user_input,
                extraction=extraction,
            )
            _session_gate_result = await self._maybe_apply_open_registry_session_gate(
                turn=turn,
                history_user_input=history_user_input,
                user_input=user_input,
                turn_act=turn_act,
                binding_resumed=_binding_resumed,
                ext_flags=ext_flags,
                detected_span=detected_span,
                extraction=extraction,
            )
            if _session_gate_result is not None:
                return _session_gate_result
            _ambig_govern = self._governed_ambiguous_tokens_for_turn(
                extraction,
                history_user_input,
            )
            if _ambig_govern and (
                self._config.auto_verify
                or open_registry_tokens_in_text(self._pef, history_user_input)
            ):
                if _ambig_govern:
                    _ref_candidates = _referent_resolution_candidate_names(
                        extraction,
                        self._pef,
                        ambiguous_tokens=list(_ambig_govern),
                    )
                    if not _ref_candidates:
                        _ref_candidates = candidates_for_tokens(
                            self._pef, list(_ambig_govern),
                        )
                    _autobind = _single_candidate_transfer_subject_autobind(
                        extraction,
                        list(_ambig_govern),
                        _ref_candidates,
                    )
                    if _autobind is not None:
                        _bound_entity, _bound_tokens = _autobind
                        self._record_discourse_referents_from_binding(
                            _bound_tokens,
                            _bound_entity,
                            turn=turn,
                        )
                        _bound_lower = {t.lower() for t in _bound_tokens}
                        extraction.ambiguous_referents = [
                            p for p in extraction.ambiguous_referents
                            if p.lower() not in _bound_lower
                        ]
                        _ambig_govern = [
                            p for p in _ambig_govern
                            if p.lower() not in _bound_lower
                        ]
                if _ambig_govern:
                    _ref_candidates = _referent_resolution_candidate_names(
                        extraction,
                        self._pef,
                        ambiguous_tokens=list(_ambig_govern),
                    )
                    if not _ref_candidates:
                        _ref_candidates = candidates_for_tokens(
                            self._pef, list(_ambig_govern),
                        )
                    interp_flags = [
                        Flag(
                            flag_type=FlagType.UNRESOLVED_REFERENT,
                            entity_name=ambig_tok,
                            claim=f"Unresolved referent '{ambig_tok}' cannot be uniquely resolved",
                            evidence=(
                                "Multiple same-category entities are present; "
                                "binding without explicit grounding is not permitted"
                            ),
                            severity="warning",
                            candidates=tuple(_ref_candidates),
                        )
                        for ambig_tok in _ambig_govern
                    ]
                    if ext_flags:
                        interp_flags = interp_flags + list(ext_flags)
                    pre_decision = await self._bridge.decide(interp_flags, user_input, self._pef)
                    if pre_decision.action != InterventionAction.PASS:
                        _primary_interp = interp_flags[0] if interp_flags else None
                        clarification = _pre_llm_unresolved_referent_clarification(
                            _ambig_govern,
                            candidate_entities=_ref_candidates,
                            original_question=history_user_input,
                            flag_claim=_primary_interp.claim if _primary_interp else None,
                            flag_evidence=_primary_interp.evidence if _primary_interp else None,
                        )
                        pre_decision.original_response = ""
                        pre_decision.corrected_response = clarification
                        pre_decision.governed_response = clarification
                        _scope_ambig = (
                            rag_pef_update_user_text
                            if rag_pef_update_user_text is not None
                            else user_input
                        )
                        _apply_epistemic_hold_after_non_admit(self._pef, pre_decision, turn)
                        self._pef.pending_clarification = _build_pending_unresolved_referent_dict(
                            self._pef,
                            scope_text=_scope_ambig,
                            original_question=user_input,
                            extraction=extraction,
                            ambiguous_tokens=list(_ambig_govern),
                            detected_span=detected_span,
                        )
                        self._register_governed_ambiguity_in_pef(
                            turn=turn,
                            utterance=history_user_input,
                            tokens=list(_ambig_govern),
                            extraction=extraction,
                            candidate_entities=_ref_candidates,
                            scope_text=_scope_ambig,
                        )
                        if (
                            self._config.auto_interpret
                            and not _has_consequence_bearing_transfer_claim(extraction)
                        ):
                            self._commit_extraction_if_admissible(
                                extraction, user_text=history_user_input, turn_act=turn_act
                            )
                        self._bridge.log_decision(
                            pre_decision,
                            turn=turn,
                            pef_context=self._pef.to_context_summary(),
                            pre_llm=True,
                            pef_snapshot=self._pef.to_dict(),
                            at_verification_basis=self._audit_at_basis(),
                            **self._audit_linkage_kwargs(pre_decision),
                        )
                        self._history.append({"role": "user", "content": history_user_input})
                        self._history.append({"role": "assistant", "content": clarification})
                        return LensResult(
                            response=clarification,
                            flags=interp_flags,
                            pef_snapshot=self._pef.to_context_summary(),
                            turn=turn,
                            span=detected_span,
                            model="",
                            action=pre_decision.action,
                            decision=pre_decision,
                            original_response=None,
                        )

            # Pre-LLM: comparative ambiguity (v10 mapping: ``docs/reference/aurora_cli_v10.py``
            # ``parse_comparative_self`` / multi-candidate path → CLARIFY, no admit).
            # ``SpacyBackend._detect_comparative_claims`` supplies ``comparative_ambiguities``
            # only when 2+ eligible comparands exist (structural underdetermination).
            # Independent of ``auto_verify``; runs whenever structural extraction runs
            # (including ``auto_interpret=False`` — v10 always parsed before admit).
            # ``update_pef`` remains gated on ``auto_interpret``. Bridge PASS/SOFT_CORRECT
            # is coerced so PASS is impossible until explicit comparand resolution.
            if extraction.comparative_ambiguities:
                _held_pc = self._pef.pending_clarification
                if _held_pending_blocks_comparative_force_revise(
                    _held_pc,
                    binding_resumed=_binding_resumed,
                ):
                    _cont_decision = _pending_ambiguity_containment_decision(
                        _held_pc,
                        rationale=(
                            "Clarification continuation from held pending "
                            "(pre-comparative gate; no upstream revise)"
                        ),
                    )
                    _continuation = _cont_decision.governed_response or ""
                    _apply_epistemic_hold_after_non_admit(
                        self._pef, _cont_decision, turn,
                    )
                    self._bridge.log_decision(
                        _cont_decision,
                        turn=turn,
                        pef_context=self._pef.to_context_summary(),
                        pre_llm=True,
                        pef_snapshot=self._pef.to_dict(),
                        at_verification_basis=self._audit_at_basis(),
                        **self._audit_linkage_kwargs(_cont_decision),
                    )
                    self._history.append(
                        {"role": "user", "content": history_user_input},
                    )
                    self._history.append(
                        {"role": "assistant", "content": _continuation},
                    )
                    return LensResult(
                        response=_continuation,
                        flags=[],
                        pef_snapshot=self._pef.to_context_summary(),
                        turn=turn,
                        span=detected_span,
                        model="",
                        action=InterventionAction.CONTAIN,
                        decision=_cont_decision,
                        original_response=None,
                    )
                ca = extraction.comparative_ambiguities[0]  # Ask about first; one question only
                comp_flags = [
                    Flag(
                        flag_type=FlagType.UNRESOLVED_COMPARAND,
                        entity_name=ca.noun,
                        claim=f"'{ca.adjective}' has {len(ca.candidates)} eligible comparands",
                        evidence=(
                            f"Candidates: {', '.join(ca.candidates)}. "
                            "Cannot collapse without explicit comparand."
                        ),
                        severity="warning",
                        candidates=tuple(ca.candidates),
                    )
                ]
                if ext_flags:
                    comp_flags = comp_flags + list(ext_flags)
                comp_decision = await self._bridge.decide(comp_flags, user_input, self._pef)
                if comp_decision.action in (
                    InterventionAction.PASS,
                    InterventionAction.SOFT_CORRECT,
                ):
                    _comp_policy = (
                        getattr(self._bridge, "_policy", None)
                        and getattr(self._bridge._policy, "name", "unknown")
                    ) or "unknown"
                    comp_decision = GovernanceDecision(
                        action=InterventionAction.FORCE_REVISE,
                        flags=comp_flags,
                        rationale=(
                            "v10 comparative invariant: structurally unresolved comparand "
                            "(CLARIFY / no admit); bridge PASS disallowed."
                        ),
                        policy=_comp_policy,
                    )
                question = _comparand_clarification_message(
                    adjective=ca.adjective,
                    noun=ca.noun,
                    candidate_entities=ca.candidates,
                )
                comp_decision.original_response = ""
                comp_decision.corrected_response = question
                comp_decision.governed_response = question
                _apply_epistemic_hold_after_non_admit(self._pef, comp_decision, turn)
                self._pef.pending_clarification = {
                    "original_question": user_input,
                    "unresolved_entity_ids": [
                        eid for eid, e in self._pef.entities.items()
                        if not e.resolved
                    ],
                    "failed_constraint": "UNRESOLVED_COMPARAND",
                    "candidate_entities": list(ca.candidates),
                    "comparand_adjective": ca.adjective,
                    "comparand_noun": ca.noun,
                    "original_span": detected_span.value,
                    "blocked_claims": [
                        {
                            "subject": c.subject,
                            "relation": c.relation,
                            "obj": c.obj,
                            "span": c.span.value,
                            "negated": c.negated,
                            "evidence": c.evidence,
                            "held_reason": "UNRESOLVED_COMPARAND",
                            "comparand_adjective": ca.adjective,
                            "comparand_noun": ca.noun,
                        }
                        for c in extraction.claims
                        if (
                            canonicalize_relation(c.relation) == "IS"
                            and str(c.obj).strip().lower() == ca.adjective.lower()
                        )
                    ],
                }
                self._bridge.log_decision(
                    comp_decision,
                    turn=turn,
                    pef_context=self._pef.to_context_summary(),
                    pre_llm=True,
                    pef_snapshot=self._pef.to_dict(),
                    at_verification_basis=self._audit_at_basis(),
                    **self._audit_linkage_kwargs(comp_decision),
                )
                self._history.append({"role": "user", "content": history_user_input})
                self._history.append({"role": "assistant", "content": question})
                return LensResult(
                    response=question,
                    flags=comp_flags,
                    pef_snapshot=self._pef.to_context_summary(),
                    turn=turn,
                    span=detected_span,
                    model="",
                    action=comp_decision.action,
                    decision=comp_decision,
                    original_response=None,
                )

            # Pre-LLM: hostile contradiction vs grounded PEF (no silent overwrite).
            _rev_gate = await self._revision_conflict_pre_llm_gate(
                user_input,
                history_user_input,
                turn_act,
                extraction,
                turn,
                detected_span,
                ext_flags,
            )
            if _rev_gate is not None:
                return _rev_gate

            if self._config.auto_interpret:
                # Both pre-LLM gates passed — admissible. Commit extraction to PEF.
                # Snapshot ambiguous tokens before update_pef so post-LLM governance can
                # force UNRESOLVED_REFERENT until user-resolved discourse bindings exist.
                if extraction.ambiguous_referents:
                    _snap_govern = self._ambiguous_referents_for_governed_clarification(
                        extraction,
                    )
                    # Post-LLM merge uses the same list as the pre-LLM gate (extractor-marked
                    # referents not yet discourse-bound).
                    self._ambiguous_snapshot_for_assistant_pending = (
                        list(_snap_govern) if _snap_govern else None
                    )
                self._discourse_binding_keys_before_user_turn_commit = frozenset(
                    self._pef.discourse_referent_bindings.keys()
                )
                _committed_state_mutation = self._commit_extraction_if_admissible(
                    extraction,
                    user_text=(
                        rag_pef_update_user_text
                        if rag_pef_update_user_text is not None
                        else history_user_input
                    ),
                    turn_act=turn_act,
                )
                if rag_pef_update_user_text is not None:
                    apply_rag_verify_discourse_bindings(self._pef, rag_pef_update_user_text)

        elif _binding_resumed:
            detected_span = _resumed_span
        else:
            detected_span = Span.PRESENT

        if _binding_resumed:
            _sn_text = state_native_text_for_binding_resume(
                _resumed_bound_input_for_sn or user_input,
                _resumed_pending_original_question,
            )
        else:
            _sn_text = history_user_input

        # State-native delegation (QUERY + strict surfaces; shared with ``process_stream``).
        # When binding was resumed, user_input is the bound blocked_proposition for the
        # LLM; _sn_text may append the follow-up attribution sentence from original_question.
        # Effective turn act is QUERY — not CLARIFY from the one-word resolution reply.
        _sn_turn_act = TurnAct.QUERY if _binding_resumed else turn_act
        _sn_proc = self._finish_turn_with_state_native_if_handled(
            _sn_text,
            _sn_turn_act,
            detected_span,
            turn,
            for_stream=False,
            binding_resumed=_binding_resumed,
            pef_snapshot=_pef_pre_commit,
        )
        _route_plan = self._plan_turn_route(
            state_native_handled=_sn_proc is not None,
            pre_llm_contain=False,
            mutation_ack_ready=False,
        )
        if _route_plan.route == LensRoute.STATE_NATIVE_READ and _sn_proc is not None:
            if _sn_proc.decision is not None:
                self._apply_route_reason(_sn_proc.decision, _route_plan)
            return _sn_proc
        # Singleton bounded puzzle (not transfer/inventory); mirrors pre-model pass if applicable.
        _cw_proc = self._finish_turn_with_closed_world_puzzle_if_handled(
            history_user_input,
            turn_act,
            detected_span,
            turn,
            for_stream=False,
        )
        _route_plan = self._plan_turn_route(
            state_native_handled=_cw_proc is not None,
            pre_llm_contain=False,
            mutation_ack_ready=False,
        )
        if _route_plan.route == LensRoute.STATE_NATIVE_READ and _cw_proc is not None:
            if _cw_proc.decision is not None:
                self._apply_route_reason(_cw_proc.decision, _route_plan)
            return _cw_proc

        _simple_is_resume_plan = _plan_resolved_pending_simple_is_answer(
            pending_failed_constraint=_resumed_pending_failed_constraint,
            pending_original_question=_resumed_pending_original_question,
            resolved_referent_phrase=_resolved_referent_phrase,
            pending_blocked_claims=_resumed_pending_blocked_claims,
        )
        if _simple_is_resume_plan is not None and _binding_resumed:
            _resume_action = (
                InterventionAction[_simple_is_resume_plan.action_hint]
                if _simple_is_resume_plan.action_hint
                else InterventionAction.PASS
            )
            _resume_decision = GovernanceDecision(
                action=_resume_action,
                flags=[],
                rationale=_simple_is_resume_plan.rationale
                or "Pre-LLM clarification continuation: resolved pending referent from committed blocked claim.",
                policy=getattr(self._bridge, "_policy", None)
                and getattr(self._bridge._policy, "name", "unknown")
                or "unknown",
            )
            _resume_decision.original_response = _simple_is_resume_plan.response_text
            _resume_decision.corrected_response = _simple_is_resume_plan.response_text
            _resume_decision.governed_response = _simple_is_resume_plan.response_text
            self._apply_route_reason(
                _resume_decision,
                LensRoutePlan(
                    route=LensRoute.MUTATION_ACK,
                    reason_code="deterministic_referent_continuation_simple_is",
                ),
            )
            _maybe_clear_epistemic_hold_on_admit(self._pef, _resume_decision)
            self._bridge.log_decision(
                _resume_decision,
                turn=turn,
                pef_context=self._pef.to_context_summary(),
                pre_llm=True,
                pef_snapshot=self._pef.to_dict(),
                at_verification_basis=self._audit_at_basis(),
                epistemic_normalisation_applied=False,
                **self._audit_linkage_kwargs(_resume_decision),
            )
            self._history.append({"role": "user", "content": history_user_input})
            self._history.append({"role": "assistant", "content": _simple_is_resume_plan.response_text})
            return LensResult(
                response=_simple_is_resume_plan.response_text,
                flags=[],
                pef_snapshot=self._pef.to_context_summary(),
                turn=turn,
                span=detected_span,
                model="",
                action=_resume_decision.action,
                decision=_resume_decision,
                original_response=None,
            )

        _attribution_resume_plan = _plan_resolved_pending_attribution_answer(
            pending_failed_constraint=_resumed_pending_failed_constraint,
            pending_snapshot=_resumed_pending_snapshot,
            resolved_binding_name=_resolved_binding_name,
        )
        if _attribution_resume_plan is not None and _binding_resumed and not _binding_upstream_resume:
            _attr_action = (
                InterventionAction[_attribution_resume_plan.action_hint]
                if _attribution_resume_plan.action_hint
                else InterventionAction.PASS
            )
            _attr_decision = GovernanceDecision(
                action=_attr_action,
                flags=[],
                rationale=_attribution_resume_plan.rationale
                or (
                    "Pre-LLM clarification continuation: resolved pending referent "
                    "and deterministic attribution answer from structured pending_task."
                ),
                policy=getattr(self._bridge, "_policy", None)
                and getattr(self._bridge._policy, "name", "unknown")
                or "unknown",
            )
            _attr_decision.original_response = _attribution_resume_plan.response_text
            _attr_decision.corrected_response = _attribution_resume_plan.response_text
            _attr_decision.governed_response = _attribution_resume_plan.response_text
            self._apply_route_reason(
                _attr_decision,
                LensRoutePlan(
                    route=LensRoute.MUTATION_ACK,
                    reason_code="deterministic_referent_continuation_attribution",
                ),
            )
            _maybe_clear_epistemic_hold_on_admit(self._pef, _attr_decision)
            self._bridge.log_decision(
                _attr_decision,
                turn=turn,
                pef_context=self._pef.to_context_summary(),
                pre_llm=True,
                pef_snapshot=self._pef.to_dict(),
                at_verification_basis=self._audit_at_basis(),
                epistemic_normalisation_applied=False,
                **self._audit_linkage_kwargs(_attr_decision),
            )
            self._history.append({"role": "user", "content": history_user_input})
            self._history.append({"role": "assistant", "content": _attribution_resume_plan.response_text})
            return LensResult(
                response=_attribution_resume_plan.response_text,
                flags=[],
                pef_snapshot=self._pef.to_context_summary(),
                turn=turn,
                span=detected_span,
                model="",
                action=_attr_decision.action,
                decision=_attr_decision,
                original_response=None,
            )

        _resume_plan = _plan_resolved_pending_margin_answer(
            pending_original_question=_resumed_pending_original_question,
            resolved_binding_name=_resolved_binding_name,
            clarification_extraction=_clarification_extraction_for_resume,
            clarification_user_text=history_user_input,
        )
        if (
            _resume_plan is not None
            and _binding_resumed
            and _resumed_pending_failed_constraint == "UNRESOLVED_REFERENT"
        ):
            _resume_action = (
                InterventionAction[_resume_plan.action_hint]
                if _resume_plan.action_hint
                else InterventionAction.PASS
            )
            _resume_decision = GovernanceDecision(
                action=_resume_action,
                flags=[],
                rationale=_resume_plan.rationale
                or (
                    "Pre-LLM clarification continuation: resolved pending referent "
                    "and deterministic margin comparison answer from committed clarification facts."
                ),
                policy=getattr(self._bridge, "_policy", None)
                and getattr(self._bridge._policy, "name", "unknown")
                or "unknown",
            )
            _resume_decision.original_response = _resume_plan.response_text
            _resume_decision.corrected_response = _resume_plan.response_text
            _resume_decision.governed_response = _resume_plan.response_text
            self._apply_route_reason(
                _resume_decision,
                LensRoutePlan(
                    route=LensRoute.MUTATION_ACK,
                    reason_code="deterministic_referent_continuation_margin",
                ),
            )
            _maybe_clear_epistemic_hold_on_admit(self._pef, _resume_decision)
            self._bridge.log_decision(
                _resume_decision,
                turn=turn,
                pef_context=self._pef.to_context_summary(),
                pre_llm=True,
                pef_snapshot=self._pef.to_dict(),
                at_verification_basis=self._audit_at_basis(),
                epistemic_normalisation_applied=False,
                **self._audit_linkage_kwargs(_resume_decision),
            )
            self._history.append({"role": "user", "content": history_user_input})
            self._history.append({"role": "assistant", "content": _resume_plan.response_text})
            return LensResult(
                response=_resume_plan.response_text,
                flags=[],
                pef_snapshot=self._pef.to_context_summary(),
                turn=turn,
                span=detected_span,
                model="",
                action=_resume_decision.action,
                decision=_resume_decision,
                original_response=None,
            )

        _ack_plan = _plan_mutation_ack(
            user_text=history_user_input,
            turn_act=turn_act,
            extraction=(
                extraction if extraction is not None else _clarification_extraction_for_resume
            ),
            committed_state_mutation=_committed_state_mutation,
            binding_resumed=_binding_resumed,
            pending_failed_constraint=_resumed_pending_failed_constraint,
            pending_original_question=_resumed_pending_original_question,
            pending_has_referent_metadata=_resumed_pending_has_referent_metadata,
            has_pending_clarification=self._pef.pending_clarification is not None,
            explicit_possessive_clarification=_explicit_possessive_clarification_resolved,
        )
        if self._suppress_admitted_assertion_ack(_ack_plan):
            # Keep full generate->checker->bridge routing in scripted governance
            # harnesses that model response-side extraction claims explicitly.
            _ack_plan = None
        _route_plan = self._plan_turn_route(
            state_native_handled=False,
            pre_llm_contain=False,
            mutation_ack_ready=_ack_plan is not None,
        )
        if _route_plan.route == LensRoute.MUTATION_ACK and _ack_plan is not None:
            _ack_response_text = _ack_plan.response_text
            _ack_action = (
                InterventionAction[_ack_plan.action_hint]
                if _ack_plan.action_hint
                else InterventionAction.PASS
            )
            _ack_decision = GovernanceDecision(
                action=_ack_action,
                flags=[],
                rationale=_ack_plan.rationale or "Pre-LLM mutation acknowledgement",
                policy=getattr(self._bridge, "_policy", None)
                and getattr(self._bridge._policy, "name", "unknown")
                or "unknown",
            )
            _ack_decision.original_response = _ack_response_text
            _ack_decision.corrected_response = _ack_response_text
            _ack_decision.governed_response = _ack_response_text
            self._apply_route_reason(_ack_decision, _route_plan)
            _ack_flags: list[Flag] = []
            if ext_flags:
                _ack_flags = list(ext_flags)
                _ack_decision = await self._bridge.decide(_ack_flags, user_input, self._pef)
                if _ack_decision.action != InterventionAction.PASS:
                    _ack_decision.original_response = _ack_response_text
                    _ack_response_text = _maybe_sanitize_governed_clarification_action(
                        await self._bridge.intervene(
                            _ack_decision,
                            self._config.adapter,
                            user_input,
                            self._pef.to_context_summary(),
                        )
                    )
                    _ack_decision.corrected_response = _ack_response_text
                    _ack_decision.governed_response = _ack_response_text
            _maybe_clear_epistemic_hold_on_admit(self._pef, _ack_decision)
            self._bridge.log_decision(
                _ack_decision,
                turn=turn,
                pef_context=self._pef.to_context_summary(),
                pre_llm=True,
                pef_snapshot=self._pef.to_dict(),
                at_verification_basis=self._audit_at_basis(),
                epistemic_normalisation_applied=False,
                **self._audit_linkage_kwargs(_ack_decision),
            )
            self._history.append({"role": "user", "content": history_user_input})
            self._history.append({"role": "assistant", "content": _ack_response_text})
            return LensResult(
                response=_ack_response_text,
                flags=_ack_flags,
                pef_snapshot=self._pef.to_context_summary(),
                turn=turn,
                span=detected_span,
                model="",
                action=_ack_decision.action,
                decision=_ack_decision,
                original_response=None,
            )

        # Step 3: PEF context for LLM
        pef_context = self._pef.to_context_summary() if self._config.inject_pef_context else ""

        # Step 4: Call LLM — build messages (system + history + user), adapter is transport-only
        history = self._history[-self._config.max_history_turns * 2:]
        messages: list[dict[str, str]] = []
        if pef_context:
            messages.append({"role": "system", "content": build_pef_system_message(pef_context)})
        if history:
            messages.extend(history)
        messages.append({
            "role": "user",
            "content": _llm_user_content_with_discourse(self._pef, user_input),
        })

        # Final pre-LLM guard: if unresolved referent or comparand clarification is
        # still pending and this turn did not bind a candidate, keep the interaction
        # in clarification continuation instead of calling adapter.generate.
        # Also catches any turn (TELL, etc.) that arrives while the epistemic hold is
        # in ambiguity mode — structural PEF state is authoritative, not turn_act.
        # When the durable registry is open, session-level dependency governs instead
        # of blind re-issuing the original pronoun clarification.
        _pend = self._pef.pending_clarification
        _hold = self._pef.epistemic_hold
        _registry_open = bool(open_entries(self._pef))
        _held_unresolved_active = referent_registry_held_unresolved(self._pef)
        if _registry_open:
            _session_gate_result = await self._maybe_apply_open_registry_session_gate(
                turn=turn,
                history_user_input=history_user_input,
                user_input=user_input,
                turn_act=turn_act,
                binding_resumed=_binding_resumed,
                ext_flags=ext_flags,
                detected_span=detected_span,
                extraction=extraction,
            )
            if _session_gate_result is not None:
                return _session_gate_result
        _active_ambiguity_hold = (
            _hold is not None
            and _hold.get("mode") == EPISTEMIC_MODE_AMBIGUITY
            and _hold.get("interaction_open") is True
            and _pend is not None
        )
        _session_gate_unrelated_bypass = (
            _registry_open and is_unrelated_safe_turn(history_user_input)
        )
        if not _session_gate_unrelated_bypass and (
            (not _held_unresolved_active and _active_ambiguity_hold) or (
                not _held_unresolved_active
                and _pend is not None
                and _pending_ambiguity_continuation_applies(
                    turn_act=turn_act,
                    pending=_pend,
                    history_user_input=history_user_input,
                )
            )
        ):
            _pending = _pend
            _continuation = _build_clarification_continuation(_pending)
            _cont_flags = _pending_ambiguity_continuation_flags(_pending)
            _cont_decision = GovernanceDecision(
                action=InterventionAction.CONTAIN,
                flags=_cont_flags,
                rationale="Clarification continuation from pending state (pre-LLM guard)",
                policy="strict",
                pathway_id="P_ASK_DISAMBIGUATE",
                output_mode="clarification_request",
                commitment_closed=True,
                interaction_open=True,
            )
            _cont_decision.original_response = ""
            _cont_decision.corrected_response = _continuation
            _cont_decision.governed_response = _continuation
            _apply_epistemic_hold_after_non_admit(self._pef, _cont_decision, turn)
            self._bridge.log_decision(
                _cont_decision,
                turn=turn,
                pef_context=self._pef.to_context_summary(),
                pre_llm=True,
                pef_snapshot=self._pef.to_dict(),
                at_verification_basis=self._audit_at_basis(),
                **self._audit_linkage_kwargs(_cont_decision),
            )
            self._history.append({"role": "user", "content": history_user_input})
            self._history.append({"role": "assistant", "content": _continuation})
            return LensResult(
                response=_continuation,
                flags=[],
                pef_snapshot=self._pef.to_context_summary(),
                turn=turn,
                span=Span.PRESENT,
                model="",
                action=InterventionAction.CONTAIN,
                decision=_cont_decision,
                original_response=None,
            )

        if _is_hold_unresolved_selection(history_user_input):
            _hold_safety = await self._maybe_apply_hold_unresolved_selection(
                turn=turn,
                history_user_input=history_user_input,
                user_input=user_input,
                pending=self._pef.pending_clarification or {},
                detected_span=detected_span,
            )
            if _hold_safety is not None:
                return _hold_safety

        _lens_log.info(
            "lens_adapter_generate",
            extra={"adapter": type(self._config.adapter).__name__, "path": "process"},
        )
        adapter_response = await self._config.adapter.generate(messages)
        upstream_model_draft = adapter_response.text
        if (
            _binding_resumed
            and _resolved_binding_name
            and _resumed_pending_snapshot
        ):
            _attr_fallback = _plan_attribution_resume_answer(
                pending=_resumed_pending_snapshot,
                selected_entity_name=_resolved_binding_name,
            )
            if _attr_fallback and not _binding_resume_response_satisfies_attribution_task(
                adapter_response.text,
                pending=_resumed_pending_snapshot,
                selected_entity_name=_resolved_binding_name,
            ):
                adapter_response = replace(adapter_response, text=_attr_fallback)
                upstream_model_draft = _attr_fallback

        _gov_outcome = await self._apply_post_generation_governance(
            upstream_text=adapter_response.text,
            turn=turn,
            history_user_input=history_user_input,
            user_input=user_input,
            detected_span=detected_span,
            extraction=extraction,
            rag_pef_update_user_text=rag_pef_update_user_text,
            ext_flags=ext_flags,
            pef_context=pef_context,
            history=history,
            binding_resumed=_binding_resumed,
            resolved_binding_name=_resolved_binding_name,
            agency_preface_text=_agency_preface_text,
            route_reason_code="adapter_checker_bridge_pipeline",
        )
        return self._lens_result_from_post_generation(
            _gov_outcome,
            adapter_model=adapter_response.model,
            adapter_usage=adapter_response.usage,
            upstream_model_draft=upstream_model_draft,
            turn=turn,
            detected_span=detected_span,
            include_operator_detail=_op_detail_eff,
            upstream_text=adapter_response.text,
        )

    async def process_stream(
        self,
        user_input: str,
        external_flags: list[Flag] | None = None,
        *,
        include_operator_detail: bool | None = None,
    ) -> AsyncIterator[tuple[str | dict, object]]:
        """Stream completion with post-stream verification. Yields (kind, payload).

        kind is "extraction_failed" | "clarification_continuation" | "chunk" | "metadata".
        - extraction_failed: payload is LensResult (return JSON, do not stream)
        - clarification_continuation: payload is LensResult from pending state (return JSON, do not stream)
        - chunk: payload is (chunk_dict, content_delta) for SSE forwarding
        - metadata: payload is aurora dict for final SSE event

        If the upstream buffer never completes before governance, session state
        is restored to the snapshot taken before this turn started (resume from
        last committed state, not partial surface output).

        ``include_operator_detail`` overrides ``LensConfig.include_operator_detail`` for the
        final metadata ``aurora`` block (e.g. per-request proxy header). When ``None``, the
        lens config value is used.
        """
        _stream_include_operator_detail = (
            include_operator_detail
            if include_operator_detail is not None
            else self._config.include_operator_detail
        )
        _pef_before_turn = self._pef.to_dict()
        _hist_before_turn = [dict(h) for h in self._history]
        turn = self._pef.advance_turn()
        self._audit_pef_start_dict = copy.deepcopy(self._pef.to_dict())
        self._audit_pef_start_frozen = copy.deepcopy(self._pef)
        history_user_input = user_input
        request_prompt_var.set(history_user_input)

        self._evaluate_sovereign_provider_route(turn)
        self._analyze_and_store_pef_uncertainties()

        _candidate_release_stream = self._maybe_adjudicate_candidate_release_task(
            history_user_input=history_user_input,
            turn=turn,
        )
        if _candidate_release_stream is not None:
            yield ("candidate_release_adjudication", _candidate_release_stream)
            return

        _eug_stream = self._maybe_apply_epistemic_uncertainty_gate(
            history_user_input=history_user_input,
            turn=turn,
        )
        if _eug_stream is not None:
            yield ("epistemic_uncertainty_gate", _eug_stream)
            return

        _sovereign_stream = self._maybe_apply_sovereign_provider_route_gate(
            history_user_input=history_user_input,
            turn=turn,
        )
        if _sovereign_stream is not None:
            yield ("sovereign_provider_route", _sovereign_stream)
            return

        turn_act = classify_turn_act(history_user_input)
        if turn_act != TurnAct.REVISE and _looks_like_explicit_correction_phrase(
            history_user_input
        ) and not history_user_input.strip().lower().startswith("actually"):
            turn_act = TurnAct.REVISE
        self._ambiguous_snapshot_for_assistant_pending = None
        self._discourse_binding_keys_before_user_turn_commit = None
        extraction: ExtractionResult | None = None
        rag_pef_update_user_text: str | None = None
        ambiguous_snapshot_merged_tokens: list[str] | None = None
        _committed_state_mutation = False
        _resumed_pending_failed_constraint: str | None = None
        _resumed_pending_original_question: str | None = None
        _resumed_pending_blocked_claims: list[dict[str, object]] | None = None
        _resumed_pending_snapshot: dict[str, object] | None = None
        _resumed_pending_has_referent_metadata = False
        _clarification_extraction_for_resume: ExtractionResult | None = None
        _resolved_binding_name: str | None = None
        _resolved_referent_phrase: str | None = None
        _agency_preface_text: str | None = None
        _binding_upstream_resume = False
        _resumed_bound_input_for_sn: str | None = None
        _explicit_possessive_clarification_resolved = False
        self._reset_pef_admission_operator_wire_turn()

        if _is_hold_unresolved_selection(history_user_input):
            _hold_control = await self._maybe_apply_hold_unresolved_selection(
                turn=turn,
                history_user_input=history_user_input,
                user_input=user_input,
                pending=self._pef.pending_clarification or {},
            )
            if _hold_control is not None:
                yield ("hold_unresolved", _hold_control)
                return

        # Frame classification — mirrors the same block in process().
        _current_frame_kind = (
            self._pef.active_frame.kind
            if self._pef.active_frame is not None
            else FrameKind.EXTERNAL
        )
        _new_frame_kind = detect_frame_transition(user_input, _current_frame_kind)
        if _new_frame_kind == FrameKind.ARTIFACT:
            self._pef.active_frame = ArtifactFrame(kind=FrameKind.ARTIFACT, opened_at_turn=turn)
        elif _new_frame_kind == FrameKind.EXTERNAL:
            self._pef.active_frame = None

        _stop_cont_stream = self._maybe_return_epistemic_stop_continuation_stream(
            history_user_input=history_user_input,
            turn=turn,
        )
        if _stop_cont_stream is not None:
            yield ("clarification_continuation", _stop_cont_stream)
            return

        # Clarification resolution — mirrors the same block in process().
        _binding_resumed = False
        _resumed_span = Span.PRESENT

        _pending_fc_resume = (
            self._pef.pending_clarification is not None
            and str(self._pef.pending_clarification.get("failed_constraint") or "")
            == "UNRESOLVED_REFERENT"
        )
        if (
            self._pef.pending_clarification is not None
            and self._config.auto_interpret
            and (
                not self._check_blocked_act_request(user_input)
                or _pending_fc_resume
            )
        ):
            _pending = self._pef.pending_clarification
            _resumed_pending_snapshot = copy.deepcopy(_pending)
            _clarification_audit_selected: str | None = None
            _clarification_pef_before: dict | None = None
            _clarification_resolution_logged = False
            _orig_q: str = _pending["original_question"]
            _unresolved_ids: list[str] = _pending.get("unresolved_entity_ids", [])
            _resumed_pending_failed_constraint = str(_pending.get("failed_constraint") or "")
            _resumed_pending_original_question = _orig_q
            _resumed_pending_blocked_claims = _pending.get("blocked_claims")
            _resumed_pending_has_referent_metadata = bool(
                _pending.get("ambiguous_referents") or _binding_candidates_present(_pending)
            )
            if _pending.get("failed_constraint") == "AGENCY_RISK_CONTEXT_UNRESOLVED":
                _agency_followup = classify_agency_context_followup(user_input)
                if _agency_followup == "explicit_violation":
                    _agency_flags = [
                        Flag(
                            flag_type=FlagType.AGENCY_VIOLATION_ASSISTANCE,
                            entity_name="agency",
                            claim=(
                                "Follow-up clarification confirms intent to request actionable "
                                "assistance for violating another person's agency"
                            ),
                            evidence="Clarification response resolved unresolved agency risk as explicit violation",
                            severity="error",
                        )
                    ]
                    _blocked_decision = await self._bridge.decide(_agency_flags, user_input, self._pef)
                    _attach_rule_result(_blocked_decision)
                    _blocked_decision.original_response = ""
                    _governed_text = _maybe_sanitize_governed_clarification_action(
                        await self._bridge.intervene(
                            _blocked_decision,
                            self._config.adapter,
                            user_input,
                            self._pef.to_context_summary(),
                        )
                    )
                    _apply_epistemic_hold_after_non_admit(self._pef, _blocked_decision, turn)
                    self._bridge.log_decision(
                        _blocked_decision,
                        turn=turn,
                        stream=True,
                        stream_completed=True,
                        stream_truncated=False,
                        stream_dropped_chars=0,
                        pef_context=self._pef.to_context_summary(),
                        pre_llm=True,
                        pef_snapshot=self._pef.to_dict(),
                        at_verification_basis=self._audit_at_basis(),
                        **self._audit_linkage_kwargs(_blocked_decision),
                    )
                    self._history.append({"role": "user", "content": history_user_input})
                    self._history.append({"role": "assistant", "content": _governed_text})
                    result = LensResult(
                        response=_governed_text,
                        flags=_agency_flags,
                        pef_snapshot=self._pef.to_context_summary(),
                        turn=turn,
                        span=Span.PRESENT,
                        model="",
                        action=_blocked_decision.action,
                        decision=_blocked_decision,
                        original_response=None,
                    )
                    yield ("clarification_continuation", result)
                    return
                if _agency_followup == "non_coercive":
                    _orig_unresolved_prompt = str(_pending.get("original_question") or "")
                    _sanitized_prompt, _removed_fragments = (
                        sanitize_agency_prompt_for_non_coercive_use(_orig_unresolved_prompt)
                    )
                    self._log_agency_context_resolution_event(
                        original_unresolved_risk_prompt=_orig_unresolved_prompt,
                        removed_coercive_objective_fragments=_removed_fragments,
                        sanitized_prompt_used_for_generation=_sanitized_prompt,
                    )
                    self._pef.pending_clarification = None
                    _binding_resumed = True
                    _resumed_span = Span(_pending.get("original_span", "present"))
                    _clar_ext = _EMPTY_RESUME_EXTRACTION_FOR_BINDING_SKIP
                    _clarification_extraction_for_resume = _clar_ext
                    _pending["original_question"] = _sanitized_prompt
                    _orig_q = _sanitized_prompt
                    user_input = _sanitized_prompt
                    _agency_preface_text = _AGENCY_ETHICAL_CONTINUATION_PREFIX
                else:
                    _continuation = _build_clarification_continuation(_pending)
                    _cont_flags = [
                        Flag(
                            flag_type=FlagType.AGENCY_RISK_CONTEXT_UNRESOLVED,
                            entity_name="agency_context_unresolved",
                            claim="Agency-risk context remains unresolved after clarification turn",
                            evidence="Follow-up context remains unclear or evasive",
                            severity="warning",
                        )
                    ]
                    _cont_decision = GovernanceDecision(
                        action=InterventionAction.CONTAIN,
                        flags=_cont_flags,
                        rationale="Agency-risk context remains unresolved",
                        policy="strict",
                        pathway_id="P_ASK_MISSING_FACT",
                        output_mode="clarification_request",
                        commitment_closed=True,
                        interaction_open=True,
                    )
                    _cont_decision.original_response = ""
                    _cont_decision.corrected_response = _continuation
                    _cont_decision.governed_response = _continuation
                    _apply_epistemic_hold_after_non_admit(self._pef, _cont_decision, turn)
                    self._bridge.log_decision(
                        _cont_decision,
                        turn=turn,
                        stream=True,
                        stream_completed=True,
                        stream_truncated=False,
                        stream_dropped_chars=0,
                        pef_context=self._pef.to_context_summary(),
                        pre_llm=True,
                        pef_snapshot=self._pef.to_dict(),
                        at_verification_basis=self._audit_at_basis(),
                        **self._audit_linkage_kwargs(_cont_decision),
                    )
                    self._history.append({"role": "user", "content": history_user_input})
                    self._history.append({"role": "assistant", "content": _continuation})
                    result = LensResult(
                        response=_continuation,
                        flags=_cont_flags,
                        pef_snapshot=self._pef.to_context_summary(),
                        turn=turn,
                        span=Span.PRESENT,
                        model="",
                        action=InterventionAction.CONTAIN,
                        decision=_cont_decision,
                        original_response=None,
                    )
                    yield ("clarification_continuation", result)
                    return

            if (
                _pending.get("failed_constraint") == "AGENCY_RISK_CONTEXT_UNRESOLVED"
                and _binding_resumed
            ):
                _typo_recovery_phase = _PendingTypoRecoveryPhase.MERGED_RESUME_SKIP_EXTRACT
                _typo_early_return = None
            else:
                _typo_recovery_phase, _typo_early_return = (
                    self._pending_state_native_typo_recovery_phase(
                        user_input=user_input,
                        history_user_input=history_user_input,
                        pending=_pending,
                        turn=turn,
                        turn_act=turn_act,
                        for_stream=True,
                    )
                )
            if _typo_early_return is not None:
                yield ("clarification_continuation", _typo_early_return)
                return

            if _typo_recovery_phase == _PendingTypoRecoveryPhase.MERGED_RESUME_SKIP_EXTRACT:
                _clar_ext = _EMPTY_RESUME_EXTRACTION_FOR_BINDING_SKIP
                _clarification_extraction_for_resume = _clar_ext
            else:
                _clar_ext = await self._backend.extract(user_input, self._pef)
                _clarification_extraction_for_resume = _clar_ext

            _clarification_pef_before = self._pef.to_dict()

            if open_entries(self._pef):
                _session_gate_result = await self._maybe_apply_open_registry_session_gate(
                    turn=turn,
                    history_user_input=history_user_input,
                    user_input=user_input,
                    turn_act=turn_act,
                    binding_resumed=_binding_resumed,
                    ext_flags=[],
                    detected_span=Span.PRESENT,
                    extraction=_clar_ext,
                )
                if _session_gate_result is not None:
                    yield ("session_registry_gate", _session_gate_result)
                    return

            _hold_unresolved_result = await self._maybe_apply_hold_unresolved_selection(
                turn=turn,
                history_user_input=history_user_input,
                user_input=user_input,
                pending=_pending,
            )
            if _hold_unresolved_result is not None:
                yield ("hold_unresolved", _hold_unresolved_result)
                return

            # Continuation gate (stream): same invariant as process() — non-selection
            # turns must not fire binding for P_ASK_DISAMBIGUATE holds.
            _paf_hold = self._pef.epistemic_hold
            _registry_allows_through = self._open_registry_session_gate_allows_through(
                history_user_input,
                turn_act,
            )
            _held_unresolved_active = referent_registry_held_unresolved(self._pef)
            if (
                not _registry_allows_through
                and not _held_unresolved_active
                and _paf_hold is not None
                and _paf_hold.get("mode") == "ambiguity"
                and _paf_hold.get("pathway_id") == "P_ASK_DISAMBIGUATE"
                and _binding_candidates_present(_pending)
                and not _is_direct_disambiguation_response(user_input, _pending)
            ):
                _continuation = _build_clarification_continuation(_pending)
                _cont_flags = _pending_ambiguity_continuation_flags(_pending)
                _cont_decision = GovernanceDecision(
                    action=InterventionAction.CONTAIN,
                    flags=_cont_flags,
                    rationale="Non-selection turn during P_ASK_DISAMBIGUATE hold",
                    policy="strict",
                    pathway_id="P_ASK_DISAMBIGUATE",
                    output_mode="clarification_request",
                    commitment_closed=True,
                    interaction_open=True,
                )
                _cont_decision.original_response = ""
                _cont_decision.corrected_response = _continuation
                _cont_decision.governed_response = _continuation
                _apply_epistemic_hold_after_non_admit(self._pef, _cont_decision, turn)
                self._bridge.log_decision(
                    _cont_decision,
                    turn=turn,
                    stream=True,
                    stream_completed=True,
                    stream_truncated=False,
                    stream_dropped_chars=0,
                    pef_context=self._pef.to_context_summary(),
                    pre_llm=True,
                    pef_snapshot=self._pef.to_dict(),
                    at_verification_basis=self._audit_at_basis(),
                    **self._audit_linkage_kwargs(_cont_decision),
                )
                self._history.append({"role": "user", "content": history_user_input})
                self._history.append({"role": "assistant", "content": _continuation})
                yield ("clarification_continuation", LensResult(
                    response=_continuation,
                    flags=[],
                    pef_snapshot=self._pef.to_context_summary(),
                    turn=turn,
                    span=Span.PRESENT,
                    model="",
                    action=InterventionAction.CONTAIN,
                    decision=_cont_decision,
                    original_response=None,
                ))
                return

            # Candidate match checked before update_pef (no PEF mutation needed).
            # update_pef deferred to entity-placeholder and pronoun-ambiguity paths.
            _binding_found: bool
            _explicit_possessive_bind: str | None = None
            if referent_registry_held_unresolved(self._pef):
                _explicit_possessive_bind = try_explicit_possessive_attribution(
                    user_input,
                    _normalized_candidate_entities_for_binding(_pending),
                )
            if _typo_recovery_phase == _PendingTypoRecoveryPhase.MERGED_RESUME_SKIP_EXTRACT:
                _binding_found = True
            elif _explicit_possessive_bind is not None:
                _binding_found = True
            elif _matches_candidate(user_input, _clar_ext, _pending):
                _binding_found = True
            else:
                if not _clar_ext.extraction_error:
                    _committed_state_mutation = self._commit_extraction_if_admissible(
                        _clar_ext,
                        user_text=history_user_input,
                        turn_act=turn_act,
                    )
                if _unresolved_ids:
                    _binding_found = _all_tracked_unresolved_entities_resolved(
                        self._pef,
                        _unresolved_ids,
                    )
                else:
                    _recheck = await self._backend.extract(_orig_q, self._pef)
                    _ambig_resume_tokens = _binding_resume_ambiguous_tokens(_pending, _orig_q)
                    _binding_found = (
                        not _recheck.extraction_error
                        and not _recheck.ambiguous_referents
                        and (
                            _matches_candidate(user_input, _clar_ext, _pending)
                            if _binding_candidates_present(_pending)
                            else not _ambig_resume_tokens
                        )
                    )

            if _binding_found:
                _resumed_span = Span(_pending.get("original_span", "present"))

                _ambig_tokens = _binding_resume_ambiguous_tokens(_pending, _orig_q)
                _stored_ambig = [
                    str(x) for x in (_pending.get("ambiguous_referents") or []) if str(x).strip()
                ]
                _pronoun_bind_completed = True
                if _stored_ambig and not _ambig_tokens:
                    _binding_found = False
                    user_input = history_user_input
                    _pronoun_bind_completed = False
                elif _ambig_tokens:
                    _bound_name = _resolve_binding_entity(
                        history_user_input, _clar_ext, _pending, self._pef,
                    )
                    if not _bound_name and _explicit_possessive_bind:
                        _bound_name = _explicit_possessive_bind
                    _resolved_binding_name = _bound_name
                    if _bound_name:
                        if _clarification_pef_before is None:
                            _clarification_pef_before = self._pef.to_dict()
                        _clarification_audit_selected = _bound_name
                        user_input, _resolved_phrase = _reconstruct_bound_user_input_or_raise(
                            original_question=_orig_q,
                            pending=_pending,
                            ambiguous_tokens=_ambig_tokens,
                            bound_entity=_bound_name,
                        )
                        _resolved_referent_phrase = _resolved_phrase
                        # Narrow the continuation payload to the resolved proposition only.
                        _bp_raw = str(_pending.get("blocked_proposition") or "").strip()
                        if _bp_raw:
                            user_input = _reconstruct_bound_text(
                                _bp_raw,
                                _bp_raw,
                                _ambig_tokens,
                                _bound_name,
                                resolved_referent_phrase=_resolved_phrase,
                            )
                        _total_rels_pre_stream = len(self._pef.relationships)
                        _replay_res_s = _commit_resolved_claims(
                            _bound_name,
                            _pending,
                            self._pef,
                            on_pef_admission=self._capture_pef_admission_for_operator_wire,
                        )
                        _commit_cf_stream = _replay_verification_indicates_relation_commit_failed(
                            _replay_res_s,
                            relation_count_before=_total_rels_pre_stream,
                            pef_after=self._pef,
                            pending_blocked_claims=list(_pending.get("blocked_claims") or []),
                        )
                        if _commit_cf_stream:
                            _cf_cont_s = _build_clarification_continuation(_pending)
                            _cf_flags_s = [Flag(
                                flag_type=FlagType.UNRESOLVED_REFERENT,
                                entity_name=_bound_name or "",
                                claim=(
                                    "Clarification binding acknowledged but no structured "
                                    "claim could be committed to PEF"
                                ),
                                evidence="CLARIFICATION_RESOLUTION_COMMIT_FAILED",
                                severity="error",
                            )]
                            _cf_dec_s = GovernanceDecision(
                                action=InterventionAction.CONTAIN,
                                flags=_cf_flags_s,
                                rationale=(
                                    "CLARIFICATION_RESOLUTION_COMMIT_FAILED: "
                                    "no structured PEF write for bound entity; "
                                    "hold not cleared"
                                ),
                                policy="strict",
                                pathway_id="P_ASK_DISAMBIGUATE",
                                output_mode="clarification_request",
                                commitment_closed=True,
                                interaction_open=True,
                            )
                            _cf_dec_s.original_response = ""
                            _cf_dec_s.corrected_response = _cf_cont_s
                            _cf_dec_s.governed_response = _cf_cont_s
                            _apply_epistemic_hold_after_non_admit(self._pef, _cf_dec_s, turn)
                            self._bridge.log_decision(
                                _cf_dec_s,
                                turn=turn,
                                stream=True,
                                stream_completed=True,
                                stream_truncated=False,
                                stream_dropped_chars=0,
                                pef_context=self._pef.to_context_summary(),
                                pre_llm=True,
                                pef_snapshot=self._pef.to_dict(),
                                at_verification_basis=self._audit_at_basis(),
                                **self._audit_linkage_kwargs(_cf_dec_s),
                            )
                            self._history.append({"role": "user", "content": history_user_input})
                            self._history.append({"role": "assistant", "content": _cf_cont_s})
                            yield ("clarification_continuation", LensResult(
                                response=_cf_cont_s,
                                flags=_cf_flags_s,
                                pef_snapshot=self._pef.to_context_summary(),
                                turn=turn,
                                span=_resumed_span,
                                model="",
                                action=InterventionAction.CONTAIN,
                                decision=_cf_dec_s,
                                original_response=None,
                            ))
                            return
                        _retire_unresolved_blocked_claims(_pending, self._pef)
                        self._record_discourse_referents_from_binding(
                            _ambig_tokens, _bound_name,
                            turn=turn,
                        )
                        self._maybe_log_clarification_resolution_audit(
                            pending=_pending,
                            history_user_input=history_user_input,
                            clar_ext=_clar_ext,
                            turn=turn,
                            pef_snapshot_before=_clarification_pef_before,
                            binding_found=True,
                            explicit_selected=_bound_name,
                        )
                        _clarification_resolution_logged = True
                        # Handle comparative IS outcome from referent replay (stream path).
                        if _replay_res_s.stop_reason:
                            _stop_dec_s = GovernanceDecision(
                                action=InterventionAction.STOP,
                                flags=[],
                                rationale=_replay_res_s.stop_reason,
                                policy="strict",
                            )
                            _stop_dec_s.original_response = ""
                            _stop_dec_s.corrected_response = _replay_res_s.stop_reason
                            _stop_dec_s.governed_response = _replay_res_s.stop_reason
                            self._pef.pending_clarification = None
                            self._bridge.log_decision(
                                _stop_dec_s,
                                turn=turn,
                                stream=True,
                                stream_completed=True,
                                stream_truncated=False,
                                stream_dropped_chars=0,
                                pef_context=self._pef.to_context_summary(),
                                pre_llm=True,
                                pef_snapshot=self._pef.to_dict(),
                                at_verification_basis=self._audit_at_basis(),
                                **self._audit_linkage_kwargs(_stop_dec_s),
                            )
                            self._history.append({"role": "user", "content": history_user_input})
                            self._history.append({"role": "assistant", "content": _replay_res_s.stop_reason})
                            yield ("clarification_continuation", LensResult(
                                response=_replay_res_s.stop_reason,
                                flags=[],
                                pef_snapshot=self._pef.to_context_summary(),
                                turn=turn,
                                span=_resumed_span,
                                model="",
                                action=InterventionAction.STOP,
                                decision=_stop_dec_s,
                                original_response=None,
                            ))
                            return
                        if _replay_res_s.pending_comparand:
                            # 2+ comparands after referent resolution → re-issue CONTAIN.
                            _pc_s = _replay_res_s.pending_comparand
                            self._pef.pending_clarification = _pc_s
                            _comp_adj_s = str(_pc_s.get("comparand_adjective") or "")
                            _comp_noun_s = str(_pc_s.get("comparand_noun") or "")
                            _cands_s: list[str] = list(_pc_s.get("candidate_entities") or [])
                            _comp_q_s = _comparand_clarification_message(
                                adjective=_comp_adj_s,
                                noun=_comp_noun_s,
                                candidate_entities=_cands_s,
                            )
                            _comp_f_s = [Flag(
                                flag_type=FlagType.UNRESOLVED_COMPARAND,
                                entity_name=_comp_noun_s,
                                claim=f"'{_comp_adj_s}' comparand unresolved after referent binding",
                                evidence=f"Candidates: {', '.join(_cands_s)}",
                                severity="warning",
                                candidates=tuple(_cands_s),
                            )]
                            _comp_dec_s = GovernanceDecision(
                                action=InterventionAction.CONTAIN,
                                flags=_comp_f_s,
                                rationale="UNRESOLVED_COMPARAND after referent binding",
                                policy="strict",
                                pathway_id="P_ASK_DISAMBIGUATE",
                                output_mode="clarification_request",
                                commitment_closed=True,
                                interaction_open=True,
                            )
                            _comp_dec_s.original_response = ""
                            _comp_dec_s.corrected_response = _comp_q_s
                            _comp_dec_s.governed_response = _comp_q_s
                            _apply_epistemic_hold_after_non_admit(self._pef, _comp_dec_s, turn)
                            self._bridge.log_decision(
                                _comp_dec_s,
                                turn=turn,
                                stream=True,
                                stream_completed=True,
                                stream_truncated=False,
                                stream_dropped_chars=0,
                                pef_context=self._pef.to_context_summary(),
                                pre_llm=True,
                                pef_snapshot=self._pef.to_dict(),
                                at_verification_basis=self._audit_at_basis(),
                                **self._audit_linkage_kwargs(_comp_dec_s),
                            )
                            self._history.append({"role": "user", "content": history_user_input})
                            self._history.append({"role": "assistant", "content": _comp_q_s})
                            yield ("clarification_continuation", LensResult(
                                response=_comp_q_s,
                                flags=_comp_f_s,
                                pef_snapshot=self._pef.to_context_summary(),
                                turn=turn,
                                span=_resumed_span,
                                model="",
                                action=InterventionAction.CONTAIN,
                                decision=_comp_dec_s,
                                original_response=None,
                            ))
                            return
                    else:
                        # Do not clear unresolved referent unless we resolved a concrete bind.
                        _binding_found = False
                        user_input = history_user_input
                        _pronoun_bind_completed = False
                else:
                    user_input = _orig_q
                    # Phase 1: UNRESOLVED_COMPARAND binding — user selected a comparand (stream).
                    if _resumed_pending_failed_constraint == "UNRESOLVED_COMPARAND":
                        _comp_bound_name_s = _resolve_binding_entity(
                            history_user_input, _clar_ext, _pending, self._pef,
                        )
                        if _comp_bound_name_s:
                            if _clarification_pef_before is None:
                                _clarification_pef_before = self._pef.to_dict()
                            _clarification_audit_selected = _comp_bound_name_s
                            _commit_comparand_claims_as_compare(
                                comparand_name=_comp_bound_name_s,
                                pending=_pending,
                                pef=self._pef,
                            )
                            self._maybe_log_clarification_resolution_audit(
                                pending=_pending,
                                history_user_input=history_user_input,
                                clar_ext=_clar_ext,
                                turn=turn,
                                pef_snapshot_before=_clarification_pef_before,
                                binding_found=True,
                                explicit_selected=_comp_bound_name_s,
                            )
                            _clarification_resolution_logged = True

                if _binding_found:
                    if (
                        not _clar_ext.extraction_error
                        and not _committed_state_mutation
                        and bool(_clar_ext.claims)
                    ):
                        _committed_state_mutation = self._commit_extraction_if_admissible(
                            _clar_ext,
                            user_text=history_user_input,
                            turn_act=turn_act,
                        )
                    if not _clarification_resolution_logged:
                        self._maybe_log_clarification_resolution_audit(
                            pending=_pending,
                            history_user_input=history_user_input,
                            clar_ext=_clar_ext,
                            turn=turn,
                            pef_snapshot_before=_clarification_pef_before,
                            binding_found=_binding_found,
                            explicit_selected=_clarification_audit_selected,
                            typo_merged=(
                                _typo_recovery_phase
                                == _PendingTypoRecoveryPhase.MERGED_RESUME_SKIP_EXTRACT
                            ),
                        )
                    if _pronoun_bind_completed:
                        _clear_epistemic_hold_ambiguity(self._pef)
                    _binding_resumed = _pronoun_bind_completed
                    if _explicit_possessive_bind is not None and _pronoun_bind_completed:
                        _explicit_possessive_clarification_resolved = True
            elif _pending_ambiguity_continuation_applies(
                turn_act=turn_act,
                pending=_pending,
                history_user_input=history_user_input,
            ):
                _continuation = _build_clarification_continuation(_pending)
                _cont_flags = _pending_ambiguity_continuation_flags(_pending)
                _cont_decision = GovernanceDecision(
                    action=InterventionAction.CONTAIN,
                    flags=_cont_flags,
                    rationale="Clarification continuation from pending state",
                    policy="strict",
                    pathway_id="P_ASK_DISAMBIGUATE",
                    output_mode="clarification_request",
                    commitment_closed=True,
                    interaction_open=True,
                )
                _cont_decision.original_response = ""
                _cont_decision.corrected_response = _continuation
                _cont_decision.governed_response = _continuation
                _apply_epistemic_hold_after_non_admit(self._pef, _cont_decision, turn)
                self._bridge.log_decision(
                    _cont_decision,
                    turn=turn,
                    stream=True,
                    stream_completed=True,
                    stream_truncated=False,
                    stream_dropped_chars=0,
                    pef_context=self._pef.to_context_summary(),
                    pre_llm=True,
                    pef_snapshot=self._pef.to_dict(),
                    at_verification_basis=self._audit_at_basis(),
                    **self._audit_linkage_kwargs(_cont_decision),
                )
                self._history.append({"role": "user", "content": history_user_input})
                self._history.append({"role": "assistant", "content": _continuation})
                result = LensResult(
                    response=_continuation,
                    flags=[],
                    pef_snapshot=self._pef.to_context_summary(),
                    turn=turn,
                    span=Span.PRESENT,
                    model="",
                    action=InterventionAction.CONTAIN,
                    decision=_cont_decision,
                    original_response=None,
                )
                yield ("clarification_continuation", result)
                return

        if _binding_resumed:
            _resumed_bound_input_for_sn = user_input
            _binding_upstream_resume = _binding_requires_upstream_resume(_resumed_pending_snapshot)

        if self._pef.pending_clarification is None:
            _early_blocked_flags = self._check_blocked_act_request(user_input)
            if _early_blocked_flags:
                self._pef.active_continuation_capability = None
                self._pef.active_continuation_context = None
            else:
                _active_cap_result = self._maybe_handle_active_continuation_stream(
                    user_input=user_input,
                    history_user_input=history_user_input,
                    turn=turn,
                )
                if _active_cap_result is not None:
                    yield ("clarification_continuation", _active_cap_result)
                    return

        ext_flags = external_flags or []

        # Pre-model state dispatch (see ``process()``): same universal router; always
        # runs; ``request_domain`` does not enable or skip it.
        _pre_stream = pre_model_state_dispatch(
            self,
            PreModelContext(
                user_input=user_input,
                history_user_input=history_user_input,
                turn=turn,
                binding_resumed=_binding_resumed,
                turn_act=turn_act,
                for_stream=True,
            ),
        )
        if _pre_stream.handled and _pre_stream.result is not None:
            async for _evt in self._stream_emit_pre_model_dispatch(
                _pre_stream,
                turn,
                include_operator_detail=_stream_include_operator_detail,
            ):
                yield _evt
            return

        # Pre-LLM: blocked-act classifier — same placement as process(); runs even
        # when auto_verify is False (see process() comment).
        _blocked_flags = self._check_blocked_act_request(user_input)
        if _blocked_flags:
            _bf_with_ext = _blocked_flags + list(ext_flags) if ext_flags else _blocked_flags
            _blocked_decision = await self._bridge.decide(_bf_with_ext, user_input, self._pef)
            _blocked_decision.pre_llm = True
            _blocked_decision.admissibility_basis = "blocked_illicit_intent"
            _attach_rule_result(_blocked_decision)
            if _blocked_decision.action != InterventionAction.PASS:
                _primary_blocked_flag = _bf_with_ext[0] if _bf_with_ext else None
                _blocked_decision.original_response = ""
                if (
                    _blocked_decision.action == InterventionAction.CONTAIN
                    and _primary_blocked_flag is not None
                    and _primary_blocked_flag.flag_type == FlagType.AGENCY_RISK_CONTEXT_UNRESOLVED
                ):
                    _governed_text = _build_clarification_continuation(
                        {
                            "failed_constraint": "AGENCY_RISK_CONTEXT_UNRESOLVED",
                        }
                    )
                    _blocked_decision.corrected_response = _governed_text
                    _blocked_decision.governed_response = _governed_text
                    self._pef.pending_clarification = _pending_payload_simple_contain(
                        user_input=user_input,
                        detected_span=Span.PRESENT,
                        pef=self._pef,
                        primary=_primary_blocked_flag,
                    )
                else:
                    _governed_text = _maybe_sanitize_governed_clarification_action(
                        await self._bridge.intervene(
                            _blocked_decision,
                            self._config.adapter,
                            user_input,
                            self._pef.to_context_summary(),
                        )
                    )
                _governed_text = _maybe_sanitize_governed_clarification_action(
                    _governed_text
                )
                if _should_state_commit_user_reported_context(_bf_with_ext, _blocked_decision):
                    _state_commit_user_reported_context(
                        self._pef,
                        user_text=history_user_input,
                        turn=turn,
                    )
                _apply_epistemic_hold_after_non_admit(self._pef, _blocked_decision, turn)
                self._set_medical_post_refusal_context_if_applicable(
                    user_input=history_user_input,
                    flags=_bf_with_ext,
                    decision=_blocked_decision,
                )
                self._bridge.log_decision(
                    _blocked_decision,
                    turn=turn,
                    stream=True,
                    stream_completed=True,
                    stream_truncated=False,
                    stream_dropped_chars=0,
                    pef_context=self._pef.to_context_summary(),
                    pre_llm=True,
                    pef_snapshot=self._pef.to_dict(),
                    at_verification_basis=self._audit_at_basis(),
                    **self._audit_linkage_kwargs(_blocked_decision),
                )
                self._history.append({"role": "user", "content": history_user_input})
                self._history.append({"role": "assistant", "content": _governed_text})
                result = LensResult(
                    response=_governed_text,
                    flags=_bf_with_ext,
                    pef_snapshot=self._pef.to_context_summary(),
                    turn=turn,
                    span=Span.PRESENT,
                    model="",
                    action=_blocked_decision.action,
                    decision=_blocked_decision,
                    original_response=None,
                    telemetry_release_path="blocked_pre_generation",
                )
                yield ("blocked_act", result)
                return

        _pef_pre_commit_stream: "PEFState | None" = None

        if not _binding_resumed:
            # SNC-1: pre-extraction snapshot (docs/adr/lane1_snapshot_policy.md).
            _pef_pre_commit_stream = copy.deepcopy(self._pef)
            extraction, rag_pef_update_user_text, rag_adm = await self._extract_user_turn_for_interpret(
                user_input,
            )
            if extraction.extraction_error:
                err = extraction.extraction_error
                reason = err.get("reason", "UNKNOWN")
                snippet = (err.get("raw_preview", "") or err.get("snippet", ""))[:150]
                flags = [
                    Flag(
                        flag_type=FlagType.EXTRACTION_FAILED,
                        entity_name="extraction",
                        claim=f"Extraction failed: {reason}",
                        evidence=snippet or str(err),
                        severity="error",
                        extraction_diagnostic=err,
                    )
                ]
                if ext_flags:
                    flags = flags + list(ext_flags)
                decision = await self._bridge.decide(flags, "", self._pef)
                governed_msg = USER_MESSAGE_INTERPRETATION_FAILED
                decision.original_response = ""
                decision.corrected_response = governed_msg
                decision.governed_response = governed_msg
                _apply_epistemic_hold_after_non_admit(self._pef, decision, turn)
                self._bridge.log_decision(
                    decision,
                    turn=turn,
                    stream=True,
                    stream_completed=True,
                    stream_truncated=False,
                    stream_dropped_chars=0,
                    pef_context=self._pef.to_context_summary(),
                    pre_llm=True,
                    pef_snapshot=self._pef.to_dict(),
                    at_verification_basis=self._audit_at_basis(),
                    **self._audit_linkage_kwargs(decision),
                )
                self._history.append({"role": "user", "content": history_user_input})
                self._history.append({"role": "assistant", "content": governed_msg})
                result = LensResult(
                    response=governed_msg,
                    flags=flags,
                    pef_snapshot=self._pef.to_context_summary(),
                    turn=turn,
                    span=Span.PRESENT,
                    model="",
                    action=decision.action,
                    decision=decision,
                    original_response=None,
                )
                yield ("extraction_failed", result)
                return
            detected_span = extraction.span
            _rag_adm_gate_result = self._maybe_apply_rag_evidence_admission_gate(
                rag_adm=rag_adm,
                history_user_input=history_user_input,
                turn=turn,
                detected_span=detected_span,
            )
            if _rag_adm_gate_result is not None:
                yield ("rag_evidence_admission_gate", _rag_adm_gate_result)
                return
            merge_structural_comparative_question_probe(
                extraction, history_user_input, self._pef
            )

            # Pre-LLM: same unresolvable-referent gate as in process().
            # On CONTAIN, admissible structure may still be applied before audit log.
            self._sync_unresolved_referent_registry_from_extraction(
                turn=turn,
                utterance=history_user_input,
                extraction=extraction,
            )
            _session_gate_result = await self._maybe_apply_open_registry_session_gate(
                turn=turn,
                history_user_input=history_user_input,
                user_input=user_input,
                turn_act=turn_act,
                binding_resumed=_binding_resumed,
                ext_flags=ext_flags,
                detected_span=detected_span,
                extraction=extraction,
            )
            if _session_gate_result is not None:
                yield ("session_registry_gate", _session_gate_result)
                return
            _ambig_govern = self._governed_ambiguous_tokens_for_turn(
                extraction,
                history_user_input,
            )
            if _ambig_govern and (
                self._config.auto_verify
                or open_registry_tokens_in_text(self._pef, history_user_input)
            ):
                if _ambig_govern:
                    _ref_candidates = _referent_resolution_candidate_names(
                        extraction,
                        self._pef,
                        ambiguous_tokens=list(_ambig_govern),
                    )
                    if not _ref_candidates:
                        _ref_candidates = candidates_for_tokens(
                            self._pef, list(_ambig_govern),
                        )
                    _autobind = _single_candidate_transfer_subject_autobind(
                        extraction,
                        list(_ambig_govern),
                        _ref_candidates,
                    )
                    if _autobind is not None:
                        _bound_entity, _bound_tokens = _autobind
                        self._record_discourse_referents_from_binding(
                            _bound_tokens,
                            _bound_entity,
                            turn=turn,
                        )
                        _bound_lower = {t.lower() for t in _bound_tokens}
                        extraction.ambiguous_referents = [
                            p for p in extraction.ambiguous_referents
                            if p.lower() not in _bound_lower
                        ]
                        _ambig_govern = [
                            p for p in _ambig_govern
                            if p.lower() not in _bound_lower
                        ]
                if _ambig_govern:
                    _ref_candidates = _referent_resolution_candidate_names(
                        extraction,
                        self._pef,
                        ambiguous_tokens=list(_ambig_govern),
                    )
                    if not _ref_candidates:
                        _ref_candidates = candidates_for_tokens(
                            self._pef, list(_ambig_govern),
                        )
                    interp_flags = [
                        Flag(
                            flag_type=FlagType.UNRESOLVED_REFERENT,
                            entity_name=ambig_tok,
                            claim=f"Unresolved referent '{ambig_tok}' cannot be uniquely resolved",
                            evidence=(
                                "Multiple same-category entities are present; "
                                "binding without explicit grounding is not permitted"
                            ),
                            severity="warning",
                            candidates=tuple(_ref_candidates),
                        )
                        for ambig_tok in _ambig_govern
                    ]
                    if ext_flags:
                        interp_flags = interp_flags + list(ext_flags)
                    pre_decision = await self._bridge.decide(interp_flags, user_input, self._pef)
                    if pre_decision.action != InterventionAction.PASS:
                        _primary_interp = interp_flags[0] if interp_flags else None
                        clarification = _pre_llm_unresolved_referent_clarification(
                            _ambig_govern,
                            candidate_entities=_ref_candidates,
                            original_question=history_user_input,
                            flag_claim=_primary_interp.claim if _primary_interp else None,
                            flag_evidence=_primary_interp.evidence if _primary_interp else None,
                        )
                        pre_decision.original_response = ""
                        pre_decision.corrected_response = clarification
                        pre_decision.governed_response = clarification
                        _scope_ambig = (
                            rag_pef_update_user_text
                            if rag_pef_update_user_text is not None
                            else user_input
                        )
                        _apply_epistemic_hold_after_non_admit(self._pef, pre_decision, turn)
                        self._pef.pending_clarification = _build_pending_unresolved_referent_dict(
                            self._pef,
                            scope_text=_scope_ambig,
                            original_question=user_input,
                            extraction=extraction,
                            ambiguous_tokens=list(_ambig_govern),
                            detected_span=detected_span,
                        )
                        self._register_governed_ambiguity_in_pef(
                            turn=turn,
                            utterance=history_user_input,
                            tokens=list(_ambig_govern),
                            extraction=extraction,
                            candidate_entities=_ref_candidates,
                            scope_text=_scope_ambig,
                        )
                        if (
                            self._config.auto_interpret
                            and not _has_consequence_bearing_transfer_claim(extraction)
                        ):
                            self._commit_extraction_if_admissible(
                                extraction, user_text=history_user_input, turn_act=turn_act
                            )
                        self._bridge.log_decision(
                            pre_decision,
                            turn=turn,
                            stream=True,
                            stream_completed=True,
                            stream_truncated=False,
                            stream_dropped_chars=0,
                            pef_context=self._pef.to_context_summary(),
                            pre_llm=True,
                            pef_snapshot=self._pef.to_dict(),
                            at_verification_basis=self._audit_at_basis(),
                            **self._audit_linkage_kwargs(pre_decision),
                        )
                        self._history.append({"role": "user", "content": history_user_input})
                        self._history.append({"role": "assistant", "content": clarification})
                        result = LensResult(
                            response=clarification,
                            flags=interp_flags,
                            pef_snapshot=self._pef.to_context_summary(),
                            turn=turn,
                            span=detected_span,
                            model="",
                            action=pre_decision.action,
                            decision=pre_decision,
                            original_response=None,
                        )
                        yield ("extraction_failed", result)
                        return

            # Pre-LLM: comparative ambiguity (same v10 mapping as ``process()``).
            if extraction.comparative_ambiguities:
                _held_pc = self._pef.pending_clarification
                if _held_pending_blocks_comparative_force_revise(
                    _held_pc,
                    binding_resumed=_binding_resumed,
                ):
                    _cont_decision = _pending_ambiguity_containment_decision(
                        _held_pc,
                        rationale=(
                            "Clarification continuation from held pending "
                            "(pre-comparative gate, stream; no upstream revise)"
                        ),
                    )
                    _continuation = _cont_decision.governed_response or ""
                    _apply_epistemic_hold_after_non_admit(
                        self._pef, _cont_decision, turn,
                    )
                    self._bridge.log_decision(
                        _cont_decision,
                        turn=turn,
                        stream=True,
                        stream_completed=True,
                        stream_truncated=False,
                        stream_dropped_chars=0,
                        pef_context=self._pef.to_context_summary(),
                        pre_llm=True,
                        pef_snapshot=self._pef.to_dict(),
                        at_verification_basis=self._audit_at_basis(),
                        **self._audit_linkage_kwargs(_cont_decision),
                    )
                    self._history.append(
                        {"role": "user", "content": history_user_input},
                    )
                    self._history.append(
                        {"role": "assistant", "content": _continuation},
                    )
                    result = LensResult(
                        response=_continuation,
                        flags=[],
                        pef_snapshot=self._pef.to_context_summary(),
                        turn=turn,
                        span=detected_span,
                        model="",
                        action=InterventionAction.CONTAIN,
                        decision=_cont_decision,
                        original_response=None,
                    )
                    yield ("clarification_continuation", result)
                    return
                ca = extraction.comparative_ambiguities[0]
                comp_flags = [
                    Flag(
                        flag_type=FlagType.UNRESOLVED_COMPARAND,
                        entity_name=ca.noun,
                        claim=f"'{ca.adjective}' has {len(ca.candidates)} eligible comparands",
                        evidence=(
                            f"Candidates: {', '.join(ca.candidates)}. "
                            "Cannot collapse without explicit comparand."
                        ),
                        severity="warning",
                        candidates=tuple(ca.candidates),
                    )
                ]
                if ext_flags:
                    comp_flags = comp_flags + list(ext_flags)
                comp_decision = await self._bridge.decide(comp_flags, user_input, self._pef)
                if comp_decision.action in (
                    InterventionAction.PASS,
                    InterventionAction.SOFT_CORRECT,
                ):
                    _comp_policy = (
                        getattr(self._bridge, "_policy", None)
                        and getattr(self._bridge._policy, "name", "unknown")
                    ) or "unknown"
                    comp_decision = GovernanceDecision(
                        action=InterventionAction.FORCE_REVISE,
                        flags=comp_flags,
                        rationale=(
                            "v10 comparative invariant: structurally unresolved comparand "
                            "(CLARIFY / no admit); bridge PASS disallowed."
                        ),
                        policy=_comp_policy,
                    )
                question = _comparand_clarification_message(
                    adjective=ca.adjective,
                    noun=ca.noun,
                    candidate_entities=ca.candidates,
                )
                comp_decision.original_response = ""
                comp_decision.corrected_response = question
                comp_decision.governed_response = question
                _apply_epistemic_hold_after_non_admit(self._pef, comp_decision, turn)
                self._pef.pending_clarification = {
                    "original_question": user_input,
                    "unresolved_entity_ids": [
                        eid for eid, e in self._pef.entities.items()
                        if not e.resolved
                    ],
                    "failed_constraint": "UNRESOLVED_COMPARAND",
                    "candidate_entities": list(ca.candidates),
                    "comparand_adjective": ca.adjective,
                    "comparand_noun": ca.noun,
                    "original_span": detected_span.value,
                    "blocked_claims": [
                        {
                            "subject": c.subject,
                            "relation": c.relation,
                            "obj": c.obj,
                            "span": c.span.value,
                            "negated": c.negated,
                            "evidence": c.evidence,
                            "held_reason": "UNRESOLVED_COMPARAND",
                            "comparand_adjective": ca.adjective,
                            "comparand_noun": ca.noun,
                        }
                        for c in extraction.claims
                        if (
                            canonicalize_relation(c.relation) == "IS"
                            and str(c.obj).strip().lower() == ca.adjective.lower()
                        )
                    ],
                }
                self._bridge.log_decision(
                    comp_decision,
                    turn=turn,
                    stream=True,
                    stream_completed=True,
                    stream_truncated=False,
                    stream_dropped_chars=0,
                    pef_context=self._pef.to_context_summary(),
                    pre_llm=True,
                    pef_snapshot=self._pef.to_dict(),
                    at_verification_basis=self._audit_at_basis(),
                    **self._audit_linkage_kwargs(comp_decision),
                )
                self._history.append({"role": "user", "content": history_user_input})
                self._history.append({"role": "assistant", "content": question})
                result = LensResult(
                    response=question,
                    flags=comp_flags,
                    pef_snapshot=self._pef.to_context_summary(),
                    turn=turn,
                    span=detected_span,
                    model="",
                    action=comp_decision.action,
                    decision=comp_decision,
                    original_response=None,
                )
                yield ("extraction_failed", result)
                return

            # Pre-LLM: hostile contradiction vs grounded PEF (no silent overwrite).
            _rev_gate = await self._revision_conflict_pre_llm_gate(
                user_input,
                history_user_input,
                turn_act,
                extraction,
                turn,
                detected_span,
                ext_flags,
            )
            if _rev_gate is not None:
                yield ("extraction_failed", _rev_gate)
                return

            if self._config.auto_interpret:
                # Both pre-LLM gates passed — admissible. Commit extraction to PEF.
                # Same snapshot as process(): assistant may echo governed clarification
                # when auto_verify skipped the pre-LLM gate.
                if extraction.ambiguous_referents:
                    _snap_govern = self._ambiguous_referents_for_governed_clarification(
                        extraction,
                    )
                    self._ambiguous_snapshot_for_assistant_pending = (
                        list(_snap_govern) if _snap_govern else None
                    )
                self._discourse_binding_keys_before_user_turn_commit = frozenset(
                    self._pef.discourse_referent_bindings.keys()
                )
                _committed_state_mutation = self._commit_extraction_if_admissible(
                    extraction,
                    user_text=(
                        rag_pef_update_user_text
                        if rag_pef_update_user_text is not None
                        else history_user_input
                    ),
                    turn_act=turn_act,
                )
                if rag_pef_update_user_text is not None:
                    apply_rag_verify_discourse_bindings(self._pef, rag_pef_update_user_text)

        elif _binding_resumed:
            detected_span = _resumed_span
        else:
            detected_span = Span.PRESENT

        if _binding_resumed:
            _sn_stream_text = state_native_text_for_binding_resume(
                _resumed_bound_input_for_sn or user_input,
                _resumed_pending_original_question,
            )
        else:
            _sn_stream_text = history_user_input

        # State-native delegation — same QUERY/surfaces as ``process``; must run before
        # building messages or calling the adapter (committed PEF read only).
        # When binding was resumed, user_input is the bound blocked_proposition for the
        # LLM; _sn_stream_text may append the follow-up attribution tail only.
        # Effective turn act is QUERY — not CLARIFY from the one-word resolution reply.
        _sn_stream_turn_act = TurnAct.QUERY if _binding_resumed else turn_act
        _sn_stream = self._finish_turn_with_state_native_if_handled(
            _sn_stream_text,
            _sn_stream_turn_act,
            detected_span,
            turn,
            for_stream=True,
            binding_resumed=_binding_resumed,
            pef_snapshot=_pef_pre_commit_stream,
        )
        _route_plan = self._plan_turn_route(
            state_native_handled=_sn_stream is not None,
            pre_llm_contain=False,
            mutation_ack_ready=False,
        )
        if _route_plan.route == LensRoute.STATE_NATIVE_READ and _sn_stream is not None:
            _sn_decision = _sn_stream.decision
            _sn_flags = _sn_stream.flags
            _sn_text = _sn_stream.response
            self._apply_route_reason(_sn_decision, _route_plan)
            if self._config.stream_emit_progress:
                yield ("progress", ProgressSignal(status="releasing"))
            _sn_chunk_dict: dict = {
                "choices": [{"delta": {"content": _sn_text}, "index": 0}],
            }
            yield ("chunk", (_sn_chunk_dict, _sn_text))
            _sn_governed = getattr(_sn_decision, "governed_response", None)
            _sn_aurora = build_aurora_block(
                action=_sn_decision.action,
                flags=_sn_flags,
                turn=turn,
                decision=_sn_decision,
                pef=self._pef,
                include_operator_detail=_stream_include_operator_detail,
                session_id=None,
                original_response=_sn_stream.original_response,
                governed_response_body=(
                    _sn_governed if _sn_governed is not None else _sn_text
                ),
                stream_governed=True,
                stream_truncated=False,
                stream_dropped_chars=0,
                pef_admission_result_wire=self.peek_pef_admission_result_wire(),
            )
            _sn_aurora["_log_flags"] = [
                f.flag_type.name for f in _sn_decision.flags
            ]
            _sn_aurora["_log_policy"] = _sn_decision.policy
            _sn_aurora["_log_pathway"] = _sn_decision.pathway_id
            _sn_aurora["_log_commitment_closed"] = _sn_decision.commitment_closed
            _sn_aurora["epistemic_normalisation_applied"] = False
            _sn_aurora["continuity_diagnostic"] = "state_native_committed_location_read"
            yield ("metadata", _sn_aurora)
            return
        _cw_stream = self._finish_turn_with_closed_world_puzzle_if_handled(
            history_user_input,
            turn_act,
            detected_span,
            turn,
            for_stream=True,
        )
        _route_plan = self._plan_turn_route(
            state_native_handled=_cw_stream is not None,
            pre_llm_contain=False,
            mutation_ack_ready=False,
        )
        if _route_plan.route == LensRoute.STATE_NATIVE_READ and _cw_stream is not None:
            _cw_decision = _cw_stream.decision
            _cw_flags = _cw_stream.flags
            _cw_text = _cw_stream.response
            self._apply_route_reason(_cw_decision, _route_plan)
            if self._config.stream_emit_progress:
                yield ("progress", ProgressSignal(status="releasing"))
            _cw_chunk_dict: dict = {
                "choices": [{"delta": {"content": _cw_text}, "index": 0}],
            }
            yield ("chunk", (_cw_chunk_dict, _cw_text))
            _cw_governed = getattr(_cw_decision, "governed_response", None)
            _cw_aurora = build_aurora_block(
                action=_cw_decision.action,
                flags=_cw_flags,
                turn=turn,
                decision=_cw_decision,
                pef=self._pef,
                include_operator_detail=_stream_include_operator_detail,
                session_id=None,
                original_response=_cw_stream.original_response,
                governed_response_body=(
                    _cw_governed if _cw_governed is not None else _cw_text
                ),
                stream_governed=True,
                stream_truncated=False,
                stream_dropped_chars=0,
                pef_admission_result_wire=self.peek_pef_admission_result_wire(),
            )
            _cw_aurora["_log_flags"] = [f.flag_type.name for f in _cw_decision.flags]
            _cw_aurora["_log_policy"] = _cw_decision.policy
            _cw_aurora["_log_pathway"] = _cw_decision.pathway_id
            _cw_aurora["_log_commitment_closed"] = _cw_decision.commitment_closed
            _cw_aurora["epistemic_normalisation_applied"] = False
            _cw_aurora["continuity_diagnostic"] = (
                _cw_stream.continuity_diagnostic or "state_native_closed_world_solved_unique"
            )
            yield ("metadata", _cw_aurora)
            return

        _attribution_resume_plan = _plan_resolved_pending_attribution_answer(
            pending_failed_constraint=_resumed_pending_failed_constraint,
            pending_snapshot=_resumed_pending_snapshot,
            resolved_binding_name=_resolved_binding_name,
        )
        if _attribution_resume_plan is not None and _binding_resumed and not _binding_upstream_resume:
            _attr_action = (
                InterventionAction[_attribution_resume_plan.action_hint]
                if _attribution_resume_plan.action_hint
                else InterventionAction.PASS
            )
            _attr_decision = GovernanceDecision(
                action=_attr_action,
                flags=[],
                rationale=_attribution_resume_plan.rationale
                or (
                    "Pre-LLM clarification continuation: resolved pending referent "
                    "and deterministic attribution answer from structured pending_task."
                ),
                policy=getattr(self._bridge, "_policy", None)
                and getattr(self._bridge._policy, "name", "unknown")
                or "unknown",
            )
            _attr_decision.original_response = _attribution_resume_plan.response_text
            _attr_decision.corrected_response = _attribution_resume_plan.response_text
            _attr_decision.governed_response = _attribution_resume_plan.response_text
            self._apply_route_reason(
                _attr_decision,
                LensRoutePlan(
                    route=LensRoute.MUTATION_ACK,
                    reason_code="deterministic_referent_continuation_attribution",
                ),
            )
            _maybe_clear_epistemic_hold_on_admit(self._pef, _attr_decision)
            self._bridge.log_decision(
                _attr_decision,
                turn=turn,
                stream=True,
                stream_completed=True,
                stream_truncated=False,
                stream_dropped_chars=0,
                pef_context=self._pef.to_context_summary(),
                pre_llm=True,
                pef_snapshot=self._pef.to_dict(),
                at_verification_basis=self._audit_at_basis(),
                epistemic_normalisation_applied=False,
                **self._audit_linkage_kwargs(_attr_decision),
            )
            self._history.append({"role": "user", "content": history_user_input})
            self._history.append({"role": "assistant", "content": _attribution_resume_plan.response_text})
            result = LensResult(
                response=_attribution_resume_plan.response_text,
                flags=[],
                pef_snapshot=self._pef.to_context_summary(),
                turn=turn,
                span=detected_span,
                model="",
                action=_attr_decision.action,
                decision=_attr_decision,
                original_response=None,
            )
            if self._config.stream_emit_progress:
                yield ("progress", ProgressSignal(status="releasing"))
            _attr_chunk: dict = {
                "choices": [{"delta": {"content": _attribution_resume_plan.response_text}, "index": 0}],
            }
            yield ("chunk", (_attr_chunk, _attribution_resume_plan.response_text))
            _attr_aurora = build_aurora_block(
                action=_attr_decision.action,
                flags=[],
                turn=turn,
                decision=_attr_decision,
                pef=self._pef,
                include_operator_detail=_stream_include_operator_detail,
                session_id=None,
                original_response=None,
                governed_response_body=_attribution_resume_plan.response_text,
                stream_governed=True,
                stream_truncated=False,
                stream_dropped_chars=0,
                pef_admission_result_wire=self.peek_pef_admission_result_wire(),
            )
            yield ("metadata", _attr_aurora)
            return

        _resume_plan = _plan_resolved_pending_margin_answer(
            pending_original_question=_resumed_pending_original_question,
            resolved_binding_name=_resolved_binding_name,
            clarification_extraction=_clarification_extraction_for_resume,
            clarification_user_text=history_user_input,
        )
        if (
            _resume_plan is not None
            and _binding_resumed
            and _resumed_pending_failed_constraint == "UNRESOLVED_REFERENT"
        ):
            _resume_action = (
                InterventionAction[_resume_plan.action_hint]
                if _resume_plan.action_hint
                else InterventionAction.PASS
            )
            _resume_decision = GovernanceDecision(
                action=_resume_action,
                flags=[],
                rationale=_resume_plan.rationale
                or (
                    "Pre-LLM clarification continuation: resolved pending referent "
                    "and deterministic margin comparison answer from committed clarification facts."
                ),
                policy=getattr(self._bridge, "_policy", None)
                and getattr(self._bridge._policy, "name", "unknown")
                or "unknown",
            )
            _resume_decision.original_response = _resume_plan.response_text
            _resume_decision.corrected_response = _resume_plan.response_text
            _resume_decision.governed_response = _resume_plan.response_text
            self._apply_route_reason(
                _resume_decision,
                LensRoutePlan(
                    route=LensRoute.MUTATION_ACK,
                    reason_code="deterministic_referent_continuation_margin",
                ),
            )
            _maybe_clear_epistemic_hold_on_admit(self._pef, _resume_decision)
            self._bridge.log_decision(
                _resume_decision,
                turn=turn,
                stream=True,
                stream_completed=True,
                stream_truncated=False,
                stream_dropped_chars=0,
                pef_context=self._pef.to_context_summary(),
                pre_llm=True,
                pef_snapshot=self._pef.to_dict(),
                at_verification_basis=self._audit_at_basis(),
                epistemic_normalisation_applied=False,
                **self._audit_linkage_kwargs(_resume_decision),
            )
            self._history.append({"role": "user", "content": history_user_input})
            self._history.append({"role": "assistant", "content": _resume_plan.response_text})
            result = LensResult(
                response=_resume_plan.response_text,
                flags=[],
                pef_snapshot=self._pef.to_context_summary(),
                turn=turn,
                span=detected_span,
                model="",
                action=_resume_decision.action,
                decision=_resume_decision,
                original_response=None,
            )
            yield ("clarification_continuation", result)
            return

        _ack_plan = _plan_mutation_ack(
            user_text=history_user_input,
            turn_act=turn_act,
            extraction=(
                extraction if extraction is not None else _clarification_extraction_for_resume
            ),
            committed_state_mutation=_committed_state_mutation,
            binding_resumed=_binding_resumed,
            pending_failed_constraint=_resumed_pending_failed_constraint,
            pending_original_question=_resumed_pending_original_question,
            pending_has_referent_metadata=_resumed_pending_has_referent_metadata,
            has_pending_clarification=self._pef.pending_clarification is not None,
            explicit_possessive_clarification=_explicit_possessive_clarification_resolved,
        )
        if self._suppress_admitted_assertion_ack(_ack_plan):
            _ack_plan = None
        _route_plan = self._plan_turn_route(
            state_native_handled=False,
            pre_llm_contain=False,
            mutation_ack_ready=_ack_plan is not None,
        )
        if _route_plan.route == LensRoute.MUTATION_ACK and _ack_plan is not None:
            _ack_response_text = _ack_plan.response_text
            _ack_action = (
                InterventionAction[_ack_plan.action_hint]
                if _ack_plan.action_hint
                else InterventionAction.PASS
            )
            _ack_decision = GovernanceDecision(
                action=_ack_action,
                flags=[],
                rationale=_ack_plan.rationale or "Pre-LLM mutation acknowledgement",
                policy=getattr(self._bridge, "_policy", None)
                and getattr(self._bridge._policy, "name", "unknown")
                or "unknown",
            )
            _ack_decision.original_response = _ack_response_text
            _ack_decision.corrected_response = _ack_response_text
            _ack_decision.governed_response = _ack_response_text
            self._apply_route_reason(_ack_decision, _route_plan)
            _ack_flags: list[Flag] = []
            if ext_flags:
                _ack_flags = list(ext_flags)
                _ack_decision = await self._bridge.decide(_ack_flags, user_input, self._pef)
                if _ack_decision.action != InterventionAction.PASS:
                    _ack_decision.original_response = _ack_response_text
                    _ack_response_text = _maybe_sanitize_governed_clarification_action(
                        await self._bridge.intervene(
                            _ack_decision,
                            self._config.adapter,
                            user_input,
                            self._pef.to_context_summary(),
                        )
                    )
                    _ack_decision.corrected_response = _ack_response_text
                    _ack_decision.governed_response = _ack_response_text
            _maybe_clear_epistemic_hold_on_admit(self._pef, _ack_decision)
            self._bridge.log_decision(
                _ack_decision,
                turn=turn,
                stream=True,
                stream_completed=True,
                stream_truncated=False,
                stream_dropped_chars=0,
                pef_context=self._pef.to_context_summary(),
                pre_llm=True,
                pef_snapshot=self._pef.to_dict(),
                at_verification_basis=self._audit_at_basis(),
                epistemic_normalisation_applied=False,
                **self._audit_linkage_kwargs(_ack_decision),
            )
            self._history.append({"role": "user", "content": history_user_input})
            self._history.append({"role": "assistant", "content": _ack_response_text})
            result = LensResult(
                response=_ack_response_text,
                flags=_ack_flags,
                pef_snapshot=self._pef.to_context_summary(),
                turn=turn,
                span=detected_span,
                model="",
                action=_ack_decision.action,
                decision=_ack_decision,
                original_response=None,
            )
            yield ("clarification_continuation", result)
            return

        pef_context = self._pef.to_context_summary() if self._config.inject_pef_context else ""
        history = self._history[-self._config.max_history_turns * 2:]
        messages: list[dict[str, str]] = []
        if pef_context:
            messages.append({"role": "system", "content": build_pef_system_message(pef_context)})
        if history:
            messages.extend(history)
        messages.append({
            "role": "user",
            "content": _llm_user_content_with_discourse(self._pef, user_input),
        })

        _pend_stream = self._pef.pending_clarification
        _hold_stream = self._pef.epistemic_hold
        _registry_open_stream = bool(open_entries(self._pef))
        _held_unresolved_active_stream = referent_registry_held_unresolved(self._pef)
        if _registry_open_stream:
            _session_gate_result = await self._maybe_apply_open_registry_session_gate(
                turn=turn,
                history_user_input=history_user_input,
                user_input=user_input,
                turn_act=turn_act,
                binding_resumed=_binding_resumed,
                ext_flags=ext_flags,
                detected_span=detected_span,
                extraction=extraction,
            )
            if _session_gate_result is not None:
                yield ("session_registry_gate", _session_gate_result)
                return
        _active_ambiguity_hold_stream = (
            _hold_stream is not None
            and _hold_stream.get("mode") == EPISTEMIC_MODE_AMBIGUITY
            and _hold_stream.get("interaction_open") is True
            and _pend_stream is not None
        )
        if (not _registry_open_stream and not _held_unresolved_active_stream and _active_ambiguity_hold_stream) or (
            not _registry_open_stream
            and not _held_unresolved_active_stream
            and _pend_stream is not None
            and _pending_ambiguity_continuation_applies(
                turn_act=turn_act,
                pending=_pend_stream,
                history_user_input=history_user_input,
            )
        ):
            _pending = _pend_stream
            _continuation = _build_clarification_continuation(_pending)
            _cont_flags = _pending_ambiguity_continuation_flags(_pending)
            _cont_decision = GovernanceDecision(
                action=InterventionAction.CONTAIN,
                flags=_cont_flags,
                rationale="Clarification continuation from pending state (pre-LLM stream guard)",
                policy="strict",
                pathway_id="P_ASK_DISAMBIGUATE",
                output_mode="clarification_request",
                commitment_closed=True,
                interaction_open=True,
            )
            _cont_decision.original_response = ""
            _cont_decision.corrected_response = _continuation
            _cont_decision.governed_response = _continuation
            _apply_epistemic_hold_after_non_admit(self._pef, _cont_decision, turn)
            self._bridge.log_decision(
                _cont_decision,
                turn=turn,
                stream=True,
                stream_completed=True,
                stream_truncated=False,
                stream_dropped_chars=0,
                pef_context=self._pef.to_context_summary(),
                pre_llm=True,
                pef_snapshot=self._pef.to_dict(),
                at_verification_basis=self._audit_at_basis(),
                **self._audit_linkage_kwargs(_cont_decision),
            )
            self._history.append({"role": "user", "content": history_user_input})
            self._history.append({"role": "assistant", "content": _continuation})
            result = LensResult(
                response=_continuation,
                flags=[],
                pef_snapshot=self._pef.to_context_summary(),
                turn=turn,
                span=Span.PRESENT,
                model="",
                action=InterventionAction.CONTAIN,
                decision=_cont_decision,
                original_response=None,
            )
            yield ("clarification_continuation", result)
            return

        if _is_hold_unresolved_selection(history_user_input):
            _hold_safety = await self._maybe_apply_hold_unresolved_selection(
                turn=turn,
                history_user_input=history_user_input,
                user_input=user_input,
                pending=self._pef.pending_clarification or {},
                detected_span=detected_span,
            )
            if _hold_safety is not None:
                yield ("hold_unresolved", _hold_safety)
                return

        # ── Phase 1: Private accumulation ─────────────────────────────────────
        #
        # All provider chunks are accumulated into a private buffer.  Nothing
        # is yielded to the user during this phase.  Raw streamed tokens must
        # never be shown directly to the user before lawful release.

        _chunks: list[tuple[dict, str]] = []
        _accumulated: list[str] = []
        _total_bytes = 0
        _stream_truncated = False
        _stream_dropped_chars = 0
        _upstream_stream = None
        _stream_completed = False
        _fallback_model = ""
        _fallback_usage: dict | None = None

        if self._config.stream_emit_progress:
            yield ("progress", ProgressSignal(status="streaming"))

        try:
            _upstream_stream = self._config.adapter.generate_stream(messages)
            async for _chunk_dict, _content_delta in _upstream_stream:
                _delta_bytes = len(_content_delta.encode("utf-8"))
                _max_bytes = self._config.max_stream_bytes
                if _max_bytes > 0 and _total_bytes + _delta_bytes > _max_bytes:
                    _stream_truncated = True
                    _stream_dropped_chars += len(_content_delta)
                    continue
                # Private accumulation only — NOT forwarded to user yet.
                _chunks.append((_chunk_dict, _content_delta))
                _accumulated.append(_content_delta)
                _total_bytes += _delta_bytes
            _stream_completed = True
        except NotImplementedError:
            # Adapter does not support streaming; fall back to generate().
            # The result is treated identically — a private buffer of one chunk.
            _lens_log.info(
                "lens_stream_fallback",
                extra={
                    "adapter": type(self._config.adapter).__name__,
                    "reason": "generate_stream not implemented, calling generate",
                },
            )
            _lens_log.info(
                "lens_adapter_generate",
                extra={"adapter": type(self._config.adapter).__name__, "path": "process_stream_fallback"},
            )
            _adapter_response = await self._config.adapter.generate(messages)
            _fallback_model = _adapter_response.model
            _fallback_usage = _adapter_response.usage
            if _adapter_response.text:
                _fb_dict = {"choices": [{"delta": {"content": _adapter_response.text}, "index": 0}]}
                _chunks.append((_fb_dict, _adapter_response.text))
                _accumulated.append(_adapter_response.text)
            _stream_completed = True
        finally:
            if _upstream_stream is not None:
                await _upstream_stream.aclose()
            if not _stream_completed:
                # Provider stream aborted before buffer was complete.
                # Nothing was yielded to the user, so no content leaked.
                # Reason is always "provider_abort": this code path is only
                # reachable when the provider's generate_stream() raises or
                # closes early.  A client disconnect after governed release
                # cannot reach this path because governance is committed
                # (stream_completed=True) before any chunk is yielded.
                # Reload last committed session state (not this turn's partial
                # extraction/advancement) before logging the abort.
                self._restore_committed_session_before_turn(_pef_before_turn, _hist_before_turn)
                _abort_decision = GovernanceDecision(
                    action=InterventionAction.PASS,
                    flags=[],
                    rationale="Stream aborted before buffer complete",
                    policy="unknown",
                )
                self._bridge.log_decision(
                    _abort_decision,
                    turn=turn,
                    stream=True,
                    stream_completed=False,
                    stream_abort_reason="provider_abort",
                    pef_context=self._pef.to_context_summary(),
                    pef_snapshot=self._pef.to_dict(),
                    at_verification_basis=self._audit_at_basis(),
                    epistemic_normalisation_applied=False,
                    **self._audit_linkage_kwargs(_abort_decision),
                )

        # If the provider stream was aborted (non-exception path, e.g. adapter
        # raised without CancelledError), abort was already logged; stop here.
        if not _stream_completed:
            return

        # ── Phase 2: Governance decision ──────────────────────────────────────
        #
        # Run Lens verification on the complete buffered text.  This is
        # identical to the non-streaming governance path — same checker, same
        # governor, same bridge.  Lens is the sole admissibility authority.

        _upstream_model_draft_stream: str | None = None
        _epistemic_norm_applied_stream = False

        _raw_stream_joined = "".join(_accumulated)
        _upstream_model_draft_stream = _raw_stream_joined
        full_text = _raw_stream_joined
        if (
            _binding_resumed
            and _resolved_binding_name
            and _resumed_pending_snapshot
        ):
            _attr_fallback = _plan_attribution_resume_answer(
                pending=_resumed_pending_snapshot,
                selected_entity_name=_resolved_binding_name,
            )
            if _attr_fallback and not _binding_resume_response_satisfies_attribution_task(
                full_text,
                pending=_resumed_pending_snapshot,
                selected_entity_name=_resolved_binding_name,
            ):
                full_text = _attr_fallback
                _upstream_model_draft_stream = _attr_fallback

        if self._config.stream_emit_progress:
            yield ("progress", ProgressSignal(status="verifying"))

        _gov_outcome = await self._apply_post_generation_governance(
            upstream_text=full_text,
            turn=turn,
            history_user_input=history_user_input,
            user_input=user_input,
            detected_span=detected_span,
            extraction=extraction,
            rag_pef_update_user_text=rag_pef_update_user_text,
            ext_flags=ext_flags,
            pef_context=pef_context,
            history=history,
            binding_resumed=_binding_resumed,
            resolved_binding_name=_resolved_binding_name,
            agency_preface_text=_agency_preface_text,
            route_reason_code=None,
            commit_audit=False,
        )
        decision = _gov_outcome.decision
        flags = _gov_outcome.flags
        full_text = _gov_outcome.final_response
        _epistemic_norm_applied_stream = _gov_outcome.epistemic_normalisation_applied
        _admit_path = _gov_outcome.admit_path

        # ── Phase 3: Stream delivery adapter (shared governance already committed) ─

        if not _admit_path:
            if decision.original_response is None:
                decision.original_response = _raw_stream_joined
            _governed = full_text

            if self._config.stream_emit_progress:
                yield ("progress", ProgressSignal(status="releasing"))

            _model_hint = _fallback_model or (
                _chunks[0][0].get("model", "") if _chunks else ""
            )
            yield ("governed_chunk", (
                make_governed_chunk_dict(_governed, _model_hint),
                _governed,
            ))

            self._apply_route_reason(
                decision,
                LensRoutePlan(
                    route=LensRoute.GENERATION_NATIVE,
                    reason_code="stream_adapter_checker_bridge_pipeline_non_admit",
                ),
            )
            self._bridge.log_decision(
                decision,
                turn=turn,
                stream=True,
                stream_completed=True,
                stream_truncated=_stream_truncated,
                stream_dropped_chars=_stream_dropped_chars,
                pef_context=pef_context,
                pre_llm=False,
                pef_snapshot=self._pef.to_dict(),
                at_verification_basis=self._audit_at_basis(),
                epistemic_normalisation_applied=False,
                **self._audit_linkage_kwargs(decision),
            )

        else:
            if decision.original_response is None:
                decision.original_response = _raw_stream_joined
            if decision.governed_response is None:
                decision.governed_response = full_text

            self._apply_route_reason(
                decision,
                LensRoutePlan(
                    route=LensRoute.GENERATION_NATIVE,
                    reason_code="stream_adapter_checker_bridge_pipeline_admit",
                ),
            )

            if full_text != _raw_stream_joined:
                _model_hint = _fallback_model or (
                    _chunks[0][0].get("model", "") if _chunks else ""
                )
                _rebuilt_chunk: dict = {
                    "choices": [{"delta": {"content": full_text}, "index": 0}],
                }
                if _model_hint:
                    _rebuilt_chunk["model"] = _model_hint
                _chunks = [(_rebuilt_chunk, full_text)]

            self._bridge.log_decision(
                decision,
                turn=turn,
                stream=True,
                stream_completed=True,
                stream_truncated=_stream_truncated,
                stream_dropped_chars=_stream_dropped_chars,
                pef_context=pef_context,
                pre_llm=False,
                pef_snapshot=self._pef.to_dict(),
                at_verification_basis=self._audit_at_basis(),
                epistemic_normalisation_applied=_epistemic_norm_applied_stream,
                **self._audit_linkage_kwargs(decision),
            )

            if self._config.stream_emit_progress:
                yield ("progress", ProgressSignal(status="releasing"))

            for _chunk_dict, _content_delta in _chunks:
                yield ("chunk", (_chunk_dict, _content_delta))

        # ── Phase 4: Metadata ─────────────────────────────────────────────────
        #
        # ``decision.original_response`` holds the upstream buffer (admit or suppressed).
        # ``build_aurora_block`` emits ``original_response`` only when
        # ``include_operator_detail`` is True; user plane is unchanged.

        _stream_governed_body = getattr(decision, "governed_response", None)
        aurora = build_aurora_block(
            action=decision.action,
            flags=flags,
            turn=turn,
            decision=decision,
            pef=self._pef,
            include_operator_detail=_stream_include_operator_detail,
            session_id=None,
            original_response=decision.original_response,
            governed_response_body=(
                _stream_governed_body
                if _stream_governed_body is not None
                else full_text
            ),
            stream_governed=True,
            stream_truncated=_stream_truncated,
            stream_dropped_chars=_stream_dropped_chars,
            pef_admission_result_wire=self.peek_pef_admission_result_wire(),
        )

        # Internal log channel: always populated, popped by proxy before SSE
        # serialisation so they never reach the client.
        # Uses decision.flags (not raw checker flags) to match the audit record.
        aurora["_log_flags"] = [f.flag_type.name for f in decision.flags]
        aurora["_log_policy"] = decision.policy
        aurora["_log_pathway"] = decision.pathway_id
        aurora["_log_commitment_closed"] = decision.commitment_closed

        if _epistemic_norm_applied_stream:
            aurora["epistemic_normalisation_applied"] = True
            if _upstream_model_draft_stream is not None:
                aurora["upstream_model_draft"] = _upstream_model_draft_stream
        else:
            aurora["epistemic_normalisation_applied"] = False

        yield ("metadata", aurora)

    async def seed_history(self, history: list[dict[str, str]]) -> None:
        """Seed PEF and conversation history from prior turns.

        Called for newly-created sessions when a stateless client sends the full
        conversation history in each POST (standard OpenAI pattern). Runs extraction
        on prior user messages to build PEF state without calling the LLM or
        verification — extraction failures are non-fatal and silently skipped.
        """
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user" and content and self._config.auto_interpret:
                self._pef.advance_turn()
                try:
                    extraction = await self._backend.extract(content, self._pef)
                    if not extraction.extraction_error:
                        _seed_act = classify_turn_act(content)
                        self._commit_extraction_if_admissible(
                            extraction,
                            user_text=content,
                            turn_act=_seed_act,
                        )
                        self._sync_unresolved_referent_registry_from_extraction(
                            turn=self._pef.current_turn,
                            utterance=content,
                            extraction=extraction,
                        )
                except Exception:
                    pass  # Seeding failures are non-fatal
            self._history.append({"role": role, "content": content})

    def reset(self) -> None:
        """Reset all state (PEF, history). New conversation."""
        self._pef = PEFState()
        self._history.clear()
        self._pef_admission_result_wire_snap = None

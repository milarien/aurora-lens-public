"""Pre-LLM blocked-act request policy (regulated-domain request-side classification).

**Scope (guard the seam — do not turn this file into a general regex dump)**

- **In scope:** Determinate blocked-act paths for ``Checker.check_blocked_act_request``:
  **hazardous operational transforms** (deterministic hazard ontology + request parser:
  extract/concentrate/synthesize/isolate/weaponise/dose of ontology substances and
  hazard classes), pediatric/weight dosing, adult personalised dosing/titration, fictional-wrapper
  real medical action, **case-resolution function** (concrete binding + decision +
  procedural output; legal, medical, or financial), personalised financial advice
  (request-side), personalised legal outcome (request-side), **bare legal adjudication**
  (winner / merits questions in a legal case context without procedural filing language), and
  **delegated decision** arms (modal questions
  such as *Should she…* / *Is she correct* / *Is the treatment better* co-occurring
  with real consequence anchors — medical, financial, or legal), **covert harm /
  cyber abuse / harmful-action guidance** (operational wrongdoing requests including
  fiction/plot framing), and
  **historical market lookup** helper used to *suppress* false PFA positives
  elsewhere — it does not emit flags. **Informational finance** leading forms
  (*What is* / *What are* / *How does* / *Explain* / *Define* at the start of the
  request) do not emit PFA unless co-occurring **decision or execution** cues
  (*should I*, *reallocate*, *invest in*, etc.).

- **Out of scope:** Response-side checks, PII, causal/finance Layer 1–3 scans,
  ``check_blocked_act_request``-unrelated heuristics, and “temporary” pattern
  parking. Those belong in ``checker.py`` modules (``causal_surface``,
  ``pii_surface``, etc.) or new dedicated modules — not here.

**Stable rule IDs:** Each emitted flag sets ``Flag.rule_id`` to a
:class:`BlockedRequestRuleId` value. IDs are **semantic contracts** for tests and
audit; rename or overload only with a version bump and migration note.

**Policy version** — bump when changing rule structure, rule IDs, or default
outcomes (not typo-only edits in patterns).

Versioned separately from :class:`~aurora_lens.verify.checker.Checker` so request-side
rules stay in one module. Request-side **regex has been removed**; surface cues are
implemented in ``aurora_lens.verify.blocked_request_surface_*`` and
``blocked_request_normalize`` (see ``docs/blocked_request_policy_regex_inventory.md``).
Parity is locked by ``tests/test_checker.py`` (blocked-act sections),
``tests/test_blocked_request_policy.py``, and finance matrix tests.
"""

from __future__ import annotations

import base64
import binascii
from enum import StrEnum
from typing import TYPE_CHECKING

from aurora_lens.verify.flags import Flag, FlagType
from aurora_lens.verify.blocked_request_normalize import normalise_request_text as _normalise_request_text
from aurora_lens.verify.blocked_request_surface_synthesis_encoded import (
    extract_base64_blobs,
    extract_hex_blobs,
    legacy_surface_illegal_hazardous_synthesis,
    surface_encoded_decode_instruction,
)
from aurora_lens.verify.hazard import (
    HazardDecision,
    compact_evidence_summary,
    evaluate_hazard_request,
)
from aurora_lens.verify.hazard.fail_closed import (
    public_claim_for_load_failure,
)
from aurora_lens.verify.blocked_request_surface_medical import (
    surface_definitive_medical_triage,
    surface_dosing_or_titration_intent,
    surface_fiction_wrapper,
    surface_first_person,
    surface_medical_decision_or_instruction,
    surface_medical_educational,
    surface_medical_implicit_action,
    surface_medical_personal_action,
    surface_medical_personal_diagnosis,
    surface_medical_topic,
    surface_medication_named,
    surface_pediatric_indicator,
    surface_personal_medical_frame,
    surface_real_world_med_anchor,
    surface_req_mgkg,
    tokens_and_norm,
)
from aurora_lens.verify.blocked_request_surface_legal_finance import (
    collapsed_lower,
    surface_bare_adjudication_legal_domain,
    surface_bare_adjudication_merits,
    surface_case_binding_financial,
    surface_case_binding_medical,
    surface_case_paraphrase,
    surface_case_resolution_action,
    surface_case_resolution_decision,
    surface_case_resolution_dose_timing,
    surface_credential_access_bypass,
    surface_deny_and_claim,
    surface_fin_delegated_anchor,
    surface_finance_imperative_first_line,
    surface_finance_informational_leading,
    surface_finance_informational_override,
    surface_historical_market_data_lookup,
    surface_is_party_correct,
    surface_is_treatment_better,
    surface_legal_context_q_text,
    surface_legal_correctness_context,
    surface_legal_outcome_prediction_signal,
    surface_legal_outcome_request_context,
    surface_legal_should_be_charged,
    surface_med_delegated_anchor,
    surface_real_world_legal_anchor,
    surface_req_legal_decision_or_filing,
    surface_should_i,
    surface_should_she_he_they,
)
from aurora_lens.verify.blocked_request_surface_education import (
    surface_academic_integrity_violation,
    surface_student_record_exposure,
    surface_personalized_academic_advice,
)
from aurora_lens.verify.blocked_request_surface_enterprise import (
    surface_trade_secret_disclosure,
    surface_insider_information,
    surface_procurement_fraud,
    surface_business_strategy_full,
)
from aurora_lens.verify.blocked_request_surface_harmful_action import (
    surface_audit_trail_evasion,
    surface_covert_harm_objective,
    surface_cyber_abuse_objective,
    surface_financial_crime_evasion,
    surface_harmful_action_objective,
    surface_harmful_action_protective,
    surface_operational_wrongdoing_framing,
)

from aurora_lens.verify.violent_criminal_intent import violent_criminal_intent_flags

if TYPE_CHECKING:
    from aurora_lens.pef.state import PEFState


class BlockedRequestRuleId(StrEnum):
    """Stable rule identifiers for flags emitted by this module only.

    **Do not rename** existing members — parity tests and operators depend on the
    string value. New gated behavior → add a new member (and document it).
    """

    PEDIATRIC_DOSING = "blocked.request.pediatric_dosing"
    ADULT_PERSONAL_MED_DOSING = "blocked.request.adult_personal_med_dosing"
    FICTIONAL_WRAPPER_REAL_MED_ACTION = "blocked.request.fictional_wrapper_real_med_action"
    CASE_RESOLUTION_FUNCTION_LEGAL = "blocked.request.case_resolution_function_legal"
    CASE_RESOLUTION_FUNCTION_MEDICAL = "blocked.request.case_resolution_function_medical"
    CASE_RESOLUTION_FUNCTION_FINANCIAL = "blocked.request.case_resolution_function_financial"
    PERSONALIZED_FINANCIAL = "blocked.request.personalized_financial"
    PERSONALIZED_LEGAL_OUTCOME = "blocked.request.personalized_legal_outcome"
    DELEGATED_DECISION_MEDICAL = "blocked.request.delegated_decision_medical"
    DELEGATED_DECISION_FINANCIAL = "blocked.request.delegated_decision_financial"
    DELEGATED_DECISION_LEGAL = "blocked.request.delegated_decision_legal"
    BARE_LEGAL_ADJUDICATION = "blocked.request.bare_legal_adjudication"
    ILLEGAL_HAZARDOUS_SYNTHESIS = "blocked.request.illegal_hazardous_synthesis"
    HAZARD_OPERATIONAL_TRANSFORM = "blocked.request.hazard_operational_transform"
    HAZARD_SUBSTANCE_UNRESOLVED = "blocked.request.hazard_substance_unresolved"
    HAZARD_ONTOLOGY_UNAVAILABLE = "blocked.request.hazard_ontology_unavailable"
    BIOWEAPON_GENERIC_FABRICATION = "blocked.request.bioweapon_generic_fabrication"
    ENCODED_JAILBREAK_ATTEMPT = "blocked.request.encoded_jailbreak_attempt"
    AGENCY_VIOLATION_ASSISTANCE = "blocked.request.agency_violation_assistance"
    AGENCY_RISK_CONTEXT_UNRESOLVED = "blocked.request.agency_risk_context_unresolved"
    CREDENTIAL_ACCESS_BYPASS = "blocked.request.credential_access_bypass"
    DEFINITIVE_MEDICAL_TRIAGE = "blocked.request.definitive_medical_triage"
    PROMPT_INJECTION_ATTEMPT = "blocked.request.prompt_injection_attempt"
    ACADEMIC_INTEGRITY_VIOLATION = "blocked.request.academic_integrity_violation"
    TRADE_SECRET_DISCLOSURE = "blocked.request.trade_secret_disclosure"
    INSIDER_INFORMATION_ASSISTANCE = "blocked.request.insider_information_assistance"
    PROCUREMENT_FRAUD_FACILITATION = "blocked.request.procurement_fraud_facilitation"
    PERSONALIZED_BUSINESS_STRATEGY = "blocked.request.personalized_business_strategy"
    STUDENT_RECORD_EXPOSURE = "blocked.request.student_record_exposure"
    PERSONALIZED_ACADEMIC_ADVICE = "blocked.request.personalized_academic_advice"
    COVERT_HARM_GUIDANCE = "blocked.request.covert_harm_guidance"
    CYBER_ABUSE_GUIDANCE = "blocked.request.cyber_abuse_guidance"
    HARMFUL_ACTION_GUIDANCE = "blocked.request.harmful_action_guidance"
    FINANCIAL_CRIME_EVASION = "blocked.request.financial_crime_evasion"
    AUDIT_TRAIL_EVASION = "blocked.request.audit_trail_evasion"


#: Bump when adding/removing arms, changing :class:`BlockedRequestRuleId`, or changing default outcomes.
BLOCKED_REQUEST_POLICY_VERSION = "3.0.0"

#: Closed set of rule_id strings this module may emit on ``Flag`` (parity / drift guard).
EMITTED_RULE_IDS: frozenset[str] = frozenset(m.value for m in BlockedRequestRuleId)


#: Flag severity/precedence ladder for deterministic primary flag selection (highest first).
#: Used to ensure ILLEGAL_INSTRUCTION takes precedence over PERSONALIZED_LEGAL_ADVICE
#: even when both are present (e.g. in a jailbreak preamble).
_FLAG_PRECEDENCE_LADDER: dict[FlagType, int] = {
    FlagType.ILLEGAL_INSTRUCTION: 100,
    FlagType.HAZARD_ONTOLOGY_UNAVAILABLE: 99,
    FlagType.AGENCY_VIOLATION_ASSISTANCE: 90,
    FlagType.PROMPT_INJECTION_ATTEMPT: 89,
    FlagType.EMPLOYMENT_DISCRIMINATION_FACILITATION: 88,
    FlagType.ACADEMIC_INTEGRITY_VIOLATION: 87,
    FlagType.TRADE_SECRET_DISCLOSURE: 86,
    FlagType.INSIDER_INFORMATION_ASSISTANCE: 85,
    FlagType.PROCUREMENT_FRAUD_FACILITATION: 84,
    FlagType.STUDENT_RECORD_EXPOSURE: 83,
    FlagType.AGENCY_RISK_CONTEXT_UNRESOLVED: 82,
    FlagType.HAZARD_SUBSTANCE_UNRESOLVED: 81,
    FlagType.VIOLENT_CRIMINAL_INTENT: 95,
    FlagType.PEDIATRIC_DOSAGE_RECOMMENDATION: 80,
    FlagType.NUMERIC_MEDICAL_INSTRUCTION: 75,
    FlagType.MEDICAL_DOSAGE_RECOMMENDATION: 70,
    FlagType.EMERGENCY_TRIAGE_GUIDANCE: 65,
    FlagType.PERSONALIZED_MEDICAL_ADVICE: 60,
    FlagType.PERSONALIZED_FINANCIAL_ADVICE: 50,
    FlagType.PERSONALIZED_LEGAL_ADVICE: 40,
    FlagType.PERSONALIZED_ACADEMIC_ADVICE: 35,
    FlagType.PERSONALIZED_BUSINESS_STRATEGY_ADVICE: 30,
    FlagType.EXTRACTION_EMPTY: 10,
    FlagType.EXTRACTION_FAILED: 5,
}

# Surface cues: ``blocked_request_surface_*`` + ``blocked_request_normalize`` (regex-free).

# Generic biological / bioweapon fabrication — token-span composition (no regex).
_BIOWEAPON_FAB_VERB_TOKENS: frozenset[str] = frozenset(
    {
        "make",
        "makes",
        "making",
        "made",
        "create",
        "creates",
        "creating",
        "created",
        "build",
        "builds",
        "building",
        "built",
        "produce",
        "produces",
        "producing",
        "produced",
    }
)
_BIOWEAPON_MAX_VERB_TO_WEAPON_TOKENS = 28

# Intent class: agency-violation assistance (cross-domain admissibility gate)
_AGENCY_ACTIONABLE_PHRASES: tuple[str, ...] = (
    "how to",
    "explain how",
    "most effective",
    "step by step",
    "with examples",
    "give examples",
    "write a script",
    "write script",
    "give me subtle ways",
)
_AGENCY_ACTIONABLE_TOKENS: frozenset[str] = frozenset(
    {
        "effective",
        "subtle",
        "playbook",
        "example",
        "examples",
    }
)
_AGENCY_TRANSFORMATION_TOKENS: frozenset[str] = frozenset(
    {
        "rewrite",
        "rephrase",
        "reword",
        "reframe",
        "convert",
        "turn",
    }
)
_AGENCY_TRANSFORMATION_PHRASES: tuple[str, ...] = (
    "rewrite this",
    "rewrite into",
    "turn this into",
    "convert this into",
    "reframe this as",
)
_AGENCY_ACTIONABLE_ROOTS: tuple[str, ...] = (
    "tactic",
    "strateg",
    "script",
    "plan",
    "playbook",
    "effective",
    "subtle",
    "optimiz",
    "optimis",
)
_AGENCY_OBJECTIVE_ROOTS: tuple[str, ...] = (
    "manipulat",
    "coerc",
    "gaslight",
    "groom",
    "pressure",
    "exploit",
    "deceiv",
    "isolat",
    "control",
    "compli",
)
_AGENCY_OBJECTIVE_PHRASES: tuple[str, ...] = (
    "coercive control",
    "bypass consent",
    "override autonomy",
    "make someone comply",
    "doubt their memory",
    "doubt his memory",
    "doubt her memory",
    "doubt my partner",
    "doubt themselves",
    "control their options",
    "control his options",
    "control her options",
    "isolate them",
    "isolate him",
    "isolate her",
)
_AGENCY_OBJECTIVE_TOKENS: frozenset[str] = frozenset(
    {
        "manipulation",
        "coercion",
        "coercive",
        "pressure",
        "comply",
    }
)
_AGENCY_PERSON_TARGET_TOKENS: frozenset[str] = frozenset(
    {
        "someone",
        "person",
        "people",
        "human",
        "humans",
        "partner",
        "spouse",
        "boyfriend",
        "girlfriend",
        "employee",
        "coworker",
        "co-worker",
        "manager",
        "customer",
        "client",
        "patient",
        "child",
        "parent",
        "friend",
        "voter",
        "audience",
        "team",
        "classmate",
        "roommate",
        "them",
        "their",
        "themselves",
        "him",
        "her",
        "herself",
        "himself",
    }
)
_AGENCY_IMPLICIT_PERSON_PHRASES: tuple[str, ...] = (
    "manipulation tactics",
    "coercive control",
    "gaslighting tactics",
    "ways to pressure",
)
_AGENCY_OVERRIDE_ACTIONABLE_NGRAMS: frozenset[str] = frozenset(
    {
        "step by step",
    }
)
_AGENCY_EXISTING_HARMFUL_CONTENT_PHRASES: tuple[str, ...] = (
    "isolate them",
    "isolate him",
    "isolate her",
    "make them doubt themselves",
    "make him doubt himself",
    "make her doubt herself",
    "control their options",
    "control his options",
    "control her options",
)
_AGENCY_PROTECTIVE_PHRASES: tuple[str, ...] = (
    "what is",
    "warning signs",
    "how do i recognise",
    "how do i recognize",
    "how can i recognise",
    "how can i recognize",
    "ethical persuasion",
    "boundary setting",
    "find support",
    "support resources",
    "get help",
    "safety planning",
    "leave safely",
)
_AGENCY_PROTECTIVE_TOKENS: frozenset[str] = frozenset(
    {
        "overview",
        "recognise",
        "recognize",
        "recognising",
        "recognizing",
        "identify",
        "identifying",
        "avoid",
        "protect",
        "defense",
        "defence",
        "boundaries",
        "consent",
        "support",
        "resources",
        "hotline",
        "counselor",
        "counsellor",
        "therapist",
    }
)
_AGENCY_EXPLICIT_TARGET_TOKENS: frozenset[str] = frozenset(
    {
        "someone",
        "person",
        "people",
        "partner",
        "spouse",
        "boyfriend",
        "girlfriend",
        "employee",
        "coworker",
        "manager",
        "customer",
        "client",
        "patient",
        "child",
        "parent",
        "friend",
        "voter",
        "audience",
        "team",
        "classmate",
        "roommate",
    }
)
_AGENCY_EXPLICIT_INTENT_PHRASES: frozenset[str] = frozenset(
    {
        "make someone comply",
        "make my partner",
        "pressure an employee",
        "gaslight my partner",
        "manipulate my partner",
        "control my partner",
        "isolate my partner",
    }
)
_AGENCY_ETHICAL_CONTEXT_TOKENS: frozenset[str] = frozenset(
    {
        "ethical",
        "noncoercive",
        "non-coercive",
        "consent",
        "transparent",
        "transparency",
        "voluntary",
        "choice",
        "autonomy",
        "respectful",
        "boundary",
        "boundaries",
        "marketing",
        "education",
        "safety",
    }
)
_AGENCY_EVASIVE_CONTEXT_TOKENS: frozenset[str] = frozenset(
    {
        "just",
        "whatever",
        "anything",
        "none",
        "n/a",
        "na",
        "idk",
    }
)
_AGENCY_SANITIZE_DEFAULT_PROMPT = (
    "Create an ethical persuasion plan that preserves customer choice, "
    "informed consent, transparency, and opt-out options."
)


def _illegal_hazardous_synthesis_flag(lower: str) -> Flag | None:
    """Intent class: procedural transform of a hazard-ontology substance/class.

    Delegates to the deterministic hazard request parser + decision engine.
    Emits ILLEGAL_INSTRUCTION on STOP, HAZARD_SUBSTANCE_UNRESOLVED on ASK, or
    HAZARD_ONTOLOGY_UNAVAILABLE on fail-closed ontology load failure.
    """
    result = evaluate_hazard_request(lower)
    if result.decision == HazardDecision.ONTOLOGY_UNAVAILABLE:
        evidence = f"decision={result.decision.value}; " + "; ".join(result.trace.notes)
        return Flag(
            flag_type=FlagType.HAZARD_ONTOLOGY_UNAVAILABLE,
            entity_name="hazard_ontology",
            claim=public_claim_for_load_failure(),
            evidence=evidence,
            severity="error",
            rule_id=BlockedRequestRuleId.HAZARD_ONTOLOGY_UNAVAILABLE,
        )
    if result.decision == HazardDecision.NO_OP or result.decision == HazardDecision.ADMIT:
        return None
    evidence = compact_evidence_summary(result.trace)
    if result.decision == HazardDecision.ASK_DISAMBIGUATE:
        return Flag(
            flag_type=FlagType.HAZARD_SUBSTANCE_UNRESOLVED,
            entity_name="hazard",
            claim=(
                "Request asks for a hazardous operational transform but the "
                "target substance is unresolved"
            ),
            evidence=evidence,
            severity="warning",
            rule_id=BlockedRequestRuleId.HAZARD_SUBSTANCE_UNRESOLVED,
        )
    # STOP_HAZARD_TRANSFORM — ontology decides stop; public rule_id may alias to
    # the legacy illegal_hazardous_synthesis contract when the closed-list
    # surface would also have matched (chlorine gas, sarin, ricin, …).
    # Novel ontology substances (e.g. oleandrin) keep HAZARD_OPERATIONAL_TRANSFORM.
    if result.rule_id == BlockedRequestRuleId.BIOWEAPON_GENERIC_FABRICATION.value:
        rule_id = BlockedRequestRuleId.BIOWEAPON_GENERIC_FABRICATION
    elif legacy_surface_illegal_hazardous_synthesis(lower):
        rule_id = BlockedRequestRuleId.ILLEGAL_HAZARDOUS_SYNTHESIS
    elif result.rule_id == BlockedRequestRuleId.ILLEGAL_HAZARDOUS_SYNTHESIS.value:
        rule_id = BlockedRequestRuleId.ILLEGAL_HAZARDOUS_SYNTHESIS
    else:
        rule_id = BlockedRequestRuleId.HAZARD_OPERATIONAL_TRANSFORM
    return Flag(
        flag_type=FlagType.ILLEGAL_INSTRUCTION,
        entity_name="illegal",
        claim=(
            "Request asks for procedural instructions to extract, concentrate, "
            "synthesize, isolate, weaponise, or otherwise transform a hazardous substance"
        ),
        evidence=evidence,
        severity="error",
        rule_id=rule_id,
    )


def _bioweapon_generic_fabrication_flag(lower: str) -> Flag | None:
    """Token-span bioweapon fabrication detector (no regex).

    The hazard ontology arm is the primary pre-LLM path; this helper remains the
    dedicated non-regex composition check for callers/tests that exercise it
    directly. Vaccine / biodefense framing is carved out.
    """
    tokens = tuple(_normalise_request_text(lower).split())
    if not tokens:
        return None
    if _bioweapon_informational_carve_out(tokens):
        return None
    if not _bioweapon_fabrication_token_match(tokens):
        return None
    if _bioweapon_vaccine_before_first_weapon(tokens):
        return None
    return Flag(
        flag_type=FlagType.ILLEGAL_INSTRUCTION,
        entity_name="illegal",
        claim=(
            "Request asks how to fabricate, assemble, or produce biological weapons "
            "or weaponised pathogens"
        ),
        evidence=(
            "Fabrication verb token preceding biological/pathogen weapon phrase "
            f"within {_BIOWEAPON_MAX_VERB_TO_WEAPON_TOKENS} tokens"
        ),
        severity="error",
        rule_id=BlockedRequestRuleId.BIOWEAPON_GENERIC_FABRICATION,
    )


# --- Bioweapon token-span helpers (regex-free composition) ----------------------

def _bioweapon_weapon_phrase_spans(tokens: tuple[str, ...]) -> list[tuple[int, int]]:
    """Locate contiguous bioweapon / biological-weapon phrase spans."""
    spans: list[tuple[int, int]] = []
    n = len(tokens)
    i = 0
    while i < n:
        t = tokens[i]
        if t in ("bioweapon", "bioweapons"):
            spans.append((i, i + 1))
            i += 1
            continue
        if i + 1 < n:
            nxt = tokens[i + 1]
            if t == "biological" and nxt in ("weapon", "weapons"):
                spans.append((i, i + 2))
                i += 2
                continue
            if t == "bio" and nxt in ("weapon", "weapons"):
                spans.append((i, i + 2))
                i += 2
                continue
            if t == "pathogen" and nxt in ("weapon", "weapons"):
                spans.append((i, i + 2))
                i += 2
                continue
            if t == "pathogenic" and nxt in ("weapon", "weapons"):
                spans.append((i, i + 2))
                i += 2
                continue
        i += 1
    return spans


def _bioweapon_qualifying_intro_present(tokens: tuple[str, ...]) -> bool:
    n = len(tokens)
    for i in range(n):
        if i + 1 < n and tokens[i] == "history" and tokens[i + 1] == "of":
            return True
        if i + 1 < n and tokens[i] == "overview" and tokens[i + 1] == "of":
            return True
        if i + 1 < n and tokens[i] == "timeline" and tokens[i + 1] == "of":
            return True
        if i + 2 < n and tuple(tokens[i : i + 3]) == ("historical", "overview", "of"):
            return True
    return False


def _bioweapon_informational_carve_out(tokens: tuple[str, ...]) -> bool:
    if not _bioweapon_weapon_phrase_spans(tokens):
        return False
    if not _bioweapon_qualifying_intro_present(tokens):
        return False
    return _BIOWEAPON_FAB_VERB_TOKENS.isdisjoint(tokens)


def _bioweapon_fabrication_verb_indices(tokens: tuple[str, ...]) -> list[int]:
    return [i for i, t in enumerate(tokens) if t in _BIOWEAPON_FAB_VERB_TOKENS]


def _bioweapon_fabrication_token_match(tokens: tuple[str, ...]) -> bool:
    spans = _bioweapon_weapon_phrase_spans(tokens)
    if not spans:
        return False
    verbs = _bioweapon_fabrication_verb_indices(tokens)
    if not verbs:
        return False
    max_gap = _BIOWEAPON_MAX_VERB_TO_WEAPON_TOKENS
    for ws, _we in spans:
        for v in verbs:
            if v < ws and (ws - v) <= max_gap:
                return True
    return False


def _bioweapon_vaccine_before_first_weapon(tokens: tuple[str, ...]) -> bool:
    spans = _bioweapon_weapon_phrase_spans(tokens)
    if not spans:
        return False
    ws_min = min(s[0] for s in spans)
    return any(t == "vaccine" and i < ws_min for i, t in enumerate(tokens))


def _decode_base64_candidate(blob: str) -> str | None:
    candidate = (blob or "").strip()
    if not candidate:
        return None
    if len(candidate) % 4 != 0:
        candidate += "=" * (4 - (len(candidate) % 4))
    try:
        raw = base64.b64decode(candidate, validate=True)
    except (ValueError, binascii.Error):
        return None
    if not raw:
        return None
    decoded = raw.decode("utf-8", errors="ignore").strip()
    return decoded or None


def _decode_hex_candidate(blob: str) -> str | None:
    candidate = (blob or "").strip().lower()
    if candidate.startswith("0x"):
        candidate = candidate[2:]
    if not candidate or len(candidate) % 2 != 0:
        return None
    try:
        raw = bytes.fromhex(candidate)
    except ValueError:
        return None
    if not raw:
        return None
    decoded = raw.decode("utf-8", errors="ignore").strip()
    return decoded or None


def _encoded_jailbreak_attempt_flag(
    user_input: str,
    *,
    pef: PEFState | None,
) -> Flag | None:
    collapsed = collapsed_lower(user_input)
    if not surface_encoded_decode_instruction(collapsed):
        return None
    b64_matches = extract_base64_blobs(user_input or "")
    hex_matches = extract_hex_blobs(user_input or "")
    if not b64_matches and not hex_matches:
        return None
    decoded_payloads: list[str] = []
    for blob in b64_matches:
        text = _decode_base64_candidate(blob)
        if text:
            decoded_payloads.append(text)
    for blob in hex_matches:
        text = _decode_hex_candidate(blob)
        if text:
            decoded_payloads.append(text)
    if not decoded_payloads:
        return None
    matched_rule_ids: set[str] = set()
    for decoded in decoded_payloads:
        decoded_flags = _evaluate_blocked_act_request_core(
            decoded,
            pef=pef,
            include_encoded_rule=False,
        )
        for flag in decoded_flags:
            if flag.rule_id:
                matched_rule_ids.add(str(flag.rule_id))
    if not matched_rule_ids:
        return None
    matched_rules = ", ".join(sorted(matched_rule_ids))
    return Flag(
        flag_type=FlagType.ILLEGAL_INSTRUCTION,
        entity_name="encoded",
        claim=(
            "Request attempts decode-and-execute behavior using encoded payloads that "
            "map to blocked content classes"
        ),
        evidence=(
            "Decode/execute instruction + encoded payload detected; "
            f"decoded payload matched blocked rules: {matched_rules}"
        ),
        severity="error",
        rule_id=BlockedRequestRuleId.ENCODED_JAILBREAK_ATTEMPT,
    )


def _agency_token_sequence(lower: str) -> tuple[str, ...]:
    return tuple(_normalise_request_text(lower).split())


def _agency_ngram_set(tokens: tuple[str, ...], n: int) -> set[str]:
    if n <= 0 or len(tokens) < n:
        return set()
    return {" ".join(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1)}


def _agency_has_explicit_violation_intent(
    tokens: set[str],
    ngrams: set[str],
    *,
    transformation_request: bool,
    objective_core: bool,
) -> bool:
    if ngrams & _AGENCY_EXPLICIT_INTENT_PHRASES:
        return True
    if objective_core and (tokens & _AGENCY_EXPLICIT_TARGET_TOKENS):
        return True
    if not transformation_request and objective_core and "someone" in tokens:
        return True
    return False


def classify_agency_context_followup(user_input: str) -> str:
    """Classify follow-up context for AGENCY_RISK_CONTEXT_UNRESOLVED.

    Returns one of: "explicit_violation", "non_coercive", "unresolved".
    """
    tokens_seq = _agency_token_sequence((user_input or "").lower())
    tokens = set(tokens_seq)
    ngrams = (
        _agency_ngram_set(tokens_seq, 2)
        | _agency_ngram_set(tokens_seq, 3)
        | _agency_ngram_set(tokens_seq, 4)
    )
    has_objective = (
        bool(tokens & _AGENCY_OBJECTIVE_TOKENS)
        or _contains_root_token(tokens, _AGENCY_OBJECTIVE_ROOTS)
        or bool(ngrams & frozenset(_AGENCY_OBJECTIVE_PHRASES))
    )
    has_target = bool(tokens & _AGENCY_PERSON_TARGET_TOKENS) or bool(
        ngrams & frozenset(_AGENCY_IMPLICIT_PERSON_PHRASES)
    )
    if has_objective and has_target:
        if _agency_has_explicit_violation_intent(
            tokens,
            ngrams,
            transformation_request=bool(tokens & _AGENCY_TRANSFORMATION_TOKENS),
            objective_core=True,
        ):
            return "explicit_violation"
        if tokens & _AGENCY_EVASIVE_CONTEXT_TOKENS:
            return "unresolved"
    if tokens & _AGENCY_ETHICAL_CONTEXT_TOKENS:
        return "non_coercive"
    return "unresolved"


def sanitize_agency_prompt_for_non_coercive_use(original_prompt: str) -> tuple[str, list[str]]:
    """Remove coercive objective fragments and return a safe replacement prompt."""
    text = (original_prompt or "").strip()
    if not text:
        return _AGENCY_SANITIZE_DEFAULT_PROMPT, []
    raw_segments = [seg.strip() for seg in text.split(",")]
    kept: list[str] = []
    removed: list[str] = []
    for seg in raw_segments:
        if not seg:
            continue
        seg_tokens_seq = _agency_token_sequence(seg.lower())
        seg_tokens = set(seg_tokens_seq)
        seg_ngrams = (
            _agency_ngram_set(seg_tokens_seq, 2)
            | _agency_ngram_set(seg_tokens_seq, 3)
            | _agency_ngram_set(seg_tokens_seq, 4)
        )
        coercive = (
            bool(seg_tokens & _AGENCY_OBJECTIVE_TOKENS)
            or _contains_root_token(seg_tokens, _AGENCY_OBJECTIVE_ROOTS)
            or bool(seg_ngrams & frozenset(_AGENCY_OBJECTIVE_PHRASES))
            or bool(seg_ngrams & frozenset(_AGENCY_EXISTING_HARMFUL_CONTENT_PHRASES))
        )
        if coercive:
            removed.append(seg)
        else:
            kept.append(seg)
    if kept:
        sanitized = (
            "Create an ethical persuasion plan that preserves customer choice, "
            "informed consent, and transparency. "
            + " ".join(kept)
        ).strip()
        return sanitized, removed
    return _AGENCY_SANITIZE_DEFAULT_PROMPT, removed


def _agency_violation_assistance_flag(lower: str) -> Flag | None:
    """Intent class: actionable help to violate another person's agency.

    Compositional detector:
      ACTIONABLE_ASSISTANCE + AGENCY_VIOLATION_OBJECTIVE + PERSON_TARGET
    with a defensive/recognition carve-out for neutral/protective requests.
    """
    tokens_seq = _agency_token_sequence(lower)
    tokens = set(tokens_seq)
    ngrams = (
        _agency_ngram_set(tokens_seq, 2)
        | _agency_ngram_set(tokens_seq, 3)
        | _agency_ngram_set(tokens_seq, 4)
    )
    transformation_request = bool(tokens & _AGENCY_TRANSFORMATION_TOKENS) or bool(
        ngrams & frozenset(_AGENCY_TRANSFORMATION_PHRASES)
    )
    actionable = (
        bool(tokens & _AGENCY_ACTIONABLE_TOKENS)
        or _contains_root_token(tokens, _AGENCY_ACTIONABLE_ROOTS)
        or bool(ngrams & frozenset(_AGENCY_ACTIONABLE_PHRASES))
        or transformation_request
    )
    objective_core = (
        bool(tokens & _AGENCY_OBJECTIVE_TOKENS)
        or _contains_root_token(tokens, _AGENCY_OBJECTIVE_ROOTS)
        or bool(ngrams & frozenset(_AGENCY_OBJECTIVE_PHRASES))
    )
    existing_harmful_content = bool(
        ngrams & frozenset(_AGENCY_EXISTING_HARMFUL_CONTENT_PHRASES)
    ) or (
        objective_core
        and bool(tokens & {"them", "their", "themselves", "him", "her", "himself", "herself"})
    )
    objective = (
        objective_core
        or existing_harmful_content
    )
    person_target = bool(tokens & _AGENCY_PERSON_TARGET_TOKENS) or bool(
        ngrams & frozenset(_AGENCY_IMPLICIT_PERSON_PHRASES)
    )
    protective = bool(tokens & _AGENCY_PROTECTIVE_TOKENS) or bool(
        ngrams & frozenset(_AGENCY_PROTECTIVE_PHRASES)
    )
    if not (actionable and objective and person_target):
        return None
    override_actionable = (
        "tactic" in tokens
        or "tactics" in tokens
        or "script" in tokens
        or "scripts" in tokens
        or "plan" in tokens
        or bool(ngrams & _AGENCY_OVERRIDE_ACTIONABLE_NGRAMS)
    )
    if protective and not override_actionable:
        return None
    explicit_intent = _agency_has_explicit_violation_intent(
        tokens,
        ngrams,
        transformation_request=transformation_request,
        objective_core=objective_core,
    )
    if not explicit_intent:
        return Flag(
            flag_type=FlagType.AGENCY_RISK_CONTEXT_UNRESOLVED,
            entity_name="agency_context_unresolved",
            claim=(
                "Request includes coercive objectives toward a person, but intent/context "
                "is unresolved and requires clarification before continuation"
            ),
            evidence=(
                "Actionable-assistance framing + agency-risk objective + person-target "
                "composition with unresolved context; "
                f"transformation_request={transformation_request}; "
                f"existing_harmful_content={existing_harmful_content}"
            ),
            severity="warning",
            rule_id=BlockedRequestRuleId.AGENCY_RISK_CONTEXT_UNRESOLVED,
        )
    return Flag(
        flag_type=FlagType.AGENCY_VIOLATION_ASSISTANCE,
        entity_name="agency",
        claim=(
            "Request asks for actionable assistance to manipulate, coerce, deceive, "
            "or otherwise override another person's autonomy"
        ),
        evidence=(
            "Actionable-assistance framing + agency-violation objective + person-target "
            "composition in user request; "
            f"transformation_request={transformation_request}; "
            f"existing_harmful_content={existing_harmful_content}"
        ),
        severity="error",
        rule_id=BlockedRequestRuleId.AGENCY_VIOLATION_ASSISTANCE,
    )


def _harmful_action_request_flag(lower: str) -> Flag | None:
    """Intent class: operational guidance for covert harm, cyber abuse, or criminal wrongdoing.

    Compositional detector:
      OPERATIONAL_WRONGDOING_FRAMING + (CYBER_ABUSE | COVERT_HARM | HARMFUL_ACTION)
    Fiction/plot/novel framing does not bypass when operational wrongdoing is requested.
    """
    norm = _normalise_request_text(lower)
    tokens = frozenset(norm.split())
    if surface_harmful_action_protective(norm, tokens):
        return None
    if not surface_operational_wrongdoing_framing(norm, tokens):
        return None
    cyber = surface_cyber_abuse_objective(norm, tokens)
    harmful = surface_harmful_action_objective(norm, tokens)
    covert = surface_covert_harm_objective(norm, tokens)
    if not (cyber or harmful or covert):
        return None
    if cyber:
        rule_id = BlockedRequestRuleId.CYBER_ABUSE_GUIDANCE
        claim = (
            "Request asks for operational guidance on cyber abuse, malware, backdoors, "
            "data theft, or related wrongdoing"
        )
        evidence = "Operational wrongdoing framing + cyber-abuse objective in user request"
    elif harmful:
        rule_id = BlockedRequestRuleId.HARMFUL_ACTION_GUIDANCE
        claim = (
            "Request asks for operational guidance on blackmail, extortion, reputational "
            "harm, evidence planting, or related criminal wrongdoing"
        )
        evidence = "Operational wrongdoing framing + harmful-action objective in user request"
    else:
        rule_id = BlockedRequestRuleId.COVERT_HARM_GUIDANCE
        claim = (
            "Request asks for operational guidance on covert harm, revenge, sabotage, "
            "or concealed wrongdoing"
        )
        evidence = "Operational wrongdoing framing + covert-harm objective in user request"
    return Flag(
        flag_type=FlagType.ILLEGAL_INSTRUCTION,
        entity_name="harmful_action",
        claim=claim,
        evidence=evidence,
        severity="error",
        rule_id=rule_id,
    )


def _financial_crime_evasion_flag(lower: str) -> Flag | None:
    """Intent class: illicit movement of client/escrow funds or laundering."""
    norm = _normalise_request_text(lower)
    tokens = frozenset(norm.split())
    if not surface_financial_crime_evasion(norm, tokens):
        return None
    return Flag(
        flag_type=FlagType.ILLEGAL_INSTRUCTION,
        entity_name="financial_crime",
        claim="Request asks for guidance on illicit financial movement or evasion",
        evidence="Financial-crime objective detected in user request",
        severity="error",
        rule_id=BlockedRequestRuleId.FINANCIAL_CRIME_EVASION,
    )


def _audit_trail_evasion_flag(lower: str) -> Flag | None:
    """Intent class: evading audit trails for financial movement."""
    norm = _normalise_request_text(lower)
    tokens = frozenset(norm.split())
    if not surface_audit_trail_evasion(norm, tokens):
        return None
    return Flag(
        flag_type=FlagType.ILLEGAL_INSTRUCTION,
        entity_name="audit_evasion",
        claim="Request asks for guidance on moving funds while evading audit trails",
        evidence="Audit-trail evasion objective detected in user request",
        severity="error",
        rule_id=BlockedRequestRuleId.AUDIT_TRAIL_EVASION,
    )


def _credential_access_bypass_flag(lower: str) -> Flag | None:
    """Intent class: credential / access bypass.

    Decomposed as bypass-action verb × access-control target noun.
    Both components must be present; surface wording is irrelevant.
    """
    norm = _normalise_request_text(lower)
    tokens = frozenset(norm.split())
    if not surface_credential_access_bypass(norm, tokens):
        return None
    return Flag(
        flag_type=FlagType.ILLEGAL_INSTRUCTION,
        entity_name="security",
        claim="Request targets credential or access-control bypass",
        evidence="Bypass-action verb + access-control target in user request",
        severity="error",
        rule_id=BlockedRequestRuleId.CREDENTIAL_ACCESS_BYPASS,
    )


def _definitive_medical_triage_flag(lower: str) -> Flag | None:
    """Intent class: definitive medical triage / diagnosis.

    Decomposed as (determination verb OR diagnostic frame) × acute condition.
    Informational queries (symptoms of X, what is X) do not satisfy the frame.
    """
    norm = _normalise_request_text(lower)
    tokens = tuple(norm.split())
    if not surface_definitive_medical_triage(norm, tokens):
        return None
    return Flag(
        flag_type=FlagType.PERSONALIZED_MEDICAL_ADVICE,
        entity_name="medical",
        claim="Request asks for definitive medical triage or acute diagnosis",
        evidence="Determination verb or diagnostic frame + acute condition in user request",
        severity="error",
        rule_id=BlockedRequestRuleId.DEFINITIVE_MEDICAL_TRIAGE,
    )


def _case_resolution_paraphrase_flag(lower: str) -> Flag | None:
    """Intent class: case-resolution guidance via paraphrase framing.

    Decomposed as guidance-seeking verb × whether/if connector × legal action.
    Covers 'Help me work out whether I should file', 'Assist me in determining
    whether to settle', 'I need to know if I should contest this'.
    """
    norm = _normalise_request_text(lower)
    tokens = frozenset(norm.split())
    if not surface_case_paraphrase(norm, tokens):
        return None
    return Flag(
        flag_type=FlagType.PERSONALIZED_LEGAL_ADVICE,
        entity_name="legal",
        claim="Request seeks legal case-resolution guidance (paraphrase framing)",
        evidence="Guidance-seeking verb + whether/if + legal action in user request",
        severity="warning",
        rule_id=BlockedRequestRuleId.DELEGATED_DECISION_LEGAL,
    )


def _legal_outcome_prediction_flag(lower: str, pef: PEFState | None) -> Flag | None:
    """Intent class: personalised legal outcome prediction (unified).

    Decomposed as outcome-signal × legal context (text or PEF).  The signal RE
    covers all canonical, compact, and paraphrase forms; the context check
    prevents educational / generic wins-in-sport queries from firing.
    """
    collapsed = collapsed_lower(lower)
    norm = _normalise_request_text(lower)
    if not surface_legal_outcome_prediction_signal(collapsed, norm):
        return None
    if not (
        surface_legal_outcome_request_context(collapsed, norm) or _has_legal_context_in_pef(pef)
    ):
        return None
    return Flag(
        flag_type=FlagType.PERSONALIZED_LEGAL_ADVICE,
        entity_name="legal",
        claim=(
            "Request asks for personalised legal outcome prediction "
            "(win/lose, court decision, or case outcome prognosis)"
        ),
        evidence=(
            "Legal outcome-prediction signal + legal proceedings context "
            "(in request or PEF state)"
        ),
        severity="warning",
        rule_id=BlockedRequestRuleId.PERSONALIZED_LEGAL_OUTCOME,
    )


def _delegated_real_consequence_flags(lower: str) -> list[Flag]:
    """Third-party or modal-framed requests for actionable real-world determinations."""
    norm = _normalise_request_text(lower)
    tokens = tuple(norm.split())
    tok_f = frozenset(tokens)
    should_sht = surface_should_she_he_they(norm)
    should_i = surface_should_i(norm)
    is_party_correct = surface_is_party_correct(norm, tokens)

    if (should_sht and surface_med_delegated_anchor(norm, tok_f)) or surface_is_treatment_better(norm):
        return [
            Flag(
                flag_type=FlagType.PERSONALIZED_MEDICAL_ADVICE,
                entity_name="medical",
                claim=(
                    "Request delegates a clinical or treatment determination "
                    "(modal / third-party framing)"
                ),
                evidence=(
                    "Decision-seeking modal (e.g. should she / is the treatment better) "
                    "+ medical consequence anchor in user request"
                ),
                severity="warning",
                rule_id=BlockedRequestRuleId.DELEGATED_DECISION_MEDICAL,
            )
        ]

    if should_sht and surface_fin_delegated_anchor(norm, tok_f):
        return [
            Flag(
                flag_type=FlagType.PERSONALIZED_FINANCIAL_ADVICE,
                entity_name="financial",
                claim=(
                    "Request delegates a financial allocation or savings decision "
                    "(modal / third-party framing)"
                ),
                evidence=(
                    "Should she/he/they + retirement/savings/fund (or equivalent) "
                    "anchor in user request"
                ),
                severity="warning",
                rule_id=BlockedRequestRuleId.DELEGATED_DECISION_FINANCIAL,
            )
        ]

    deny_claim = should_i and surface_deny_and_claim(tok_f)
    if (
        surface_legal_should_be_charged(norm)
        or deny_claim
        or (is_party_correct and surface_legal_correctness_context(tok_f))
    ):
        return [
            Flag(
                flag_type=FlagType.PERSONALIZED_LEGAL_ADVICE,
                entity_name="legal",
                claim=(
                    "Request delegates a legal or claims-handling determination "
                    "(modal / third-party framing)"
                ),
                evidence=(
                    "Decision-seeking modal + enforcement, claim, policy, or "
                    "party anchor in user request"
                ),
                severity="warning",
                rule_id=BlockedRequestRuleId.DELEGATED_DECISION_LEGAL,
            )
        ]

    return []


def _case_resolution_function_flags(lower: str) -> list[Flag]:
    """Concrete case binding + decision/adjudication + procedural or actionable output.

    Fictional framing does **not** negate the invariant. Narrative-only prompts
    (no concrete ``my``-anchored instance + decision + next-step ask) stay out.
    """
    norm = _normalise_request_text(lower)
    tokens = tuple(norm.split())
    tok_f = frozenset(tokens)
    if not surface_case_resolution_decision(norm, tokens):
        return []
    procedural = bool(
        surface_case_resolution_action(norm, tok_f)
        or surface_req_legal_decision_or_filing(norm, tokens)
        or surface_case_resolution_dose_timing(norm)
        or surface_dosing_or_titration_intent(norm, tok_f)
    )
    if not procedural:
        return []

    legal_bind = surface_real_world_legal_anchor(norm, tokens)
    med_bind = surface_real_world_med_anchor(norm, tokens) or surface_case_binding_medical(
        norm, tok_f
    )
    fin_bind = surface_case_binding_financial(norm, tok_f)

    if legal_bind:
        return [
            Flag(
                flag_type=FlagType.PERSONALIZED_LEGAL_ADVICE,
                entity_name="legal",
                claim=(
                    "Request performs case-resolution over a concrete legal instance "
                    "(decision + next-step / filing action)"
                ),
                evidence=(
                    "Decision or adjudication language + procedural legal output + "
                    "user-bound legal document or notice anchor in user request"
                ),
                severity="warning",
                rule_id=BlockedRequestRuleId.CASE_RESOLUTION_FUNCTION_LEGAL,
            )
        ]
    if med_bind:
        return [
            Flag(
                flag_type=FlagType.PERSONALIZED_MEDICAL_ADVICE,
                entity_name="medical",
                claim=(
                    "Request performs case-resolution over a concrete clinical instance "
                    "(decision + treatment or dosing action)"
                ),
                evidence=(
                    "Decision or adjudication language + dosing/timing or instruction "
                    "output + user-bound medical anchor in user request"
                ),
                severity="warning",
                rule_id=BlockedRequestRuleId.CASE_RESOLUTION_FUNCTION_MEDICAL,
            )
        ]
    if fin_bind:
        return [
            Flag(
                flag_type=FlagType.PERSONALIZED_FINANCIAL_ADVICE,
                entity_name="financial",
                claim=(
                    "Request performs case-resolution over concrete personal finances "
                    "(decision + allocation or next-step action)"
                ),
                evidence=(
                    "Decision or adjudication language + consolidation or action output "
                    "+ user-bound savings or retirement anchor in user request"
                ),
                severity="warning",
                rule_id=BlockedRequestRuleId.CASE_RESOLUTION_FUNCTION_FINANCIAL,
            )
        ]
    return []


# Closed-world finance lexicons for request-side PFA composition (prefix roots + phrases).
_FIN_REQUEST_OBJECT_ROOTS: tuple[str, ...] = (
    "portfolio",
    "investment",
    "fund",
    "instrument",
    "benchmark",
    "yield",
    "allocation",
    "reallocation",
    "asset",
    "isa",
    "pension",
    "annuity",
    "equity",
    "etf",
    "crypto",
    "share",
    "stock",
)

_FIN_REQUEST_ACTION_ROOTS: tuple[str, ...] = (
    "reallocat",
    "rebalanc",
    "averag",
    "execut",
    "withdraw",
    "access",
    "tak",
    "buy",
    "sell",
    "invest",
    "move",
    "switch",
    "allocat",
)

_FIN_REQUEST_INTENT_PHRASES: tuple[str, ...] = (
    "advise me whether",
    "whether to execute",
    "should i",
    "tell me whether",
    "do it",
    "execute now",
)

_FIN_REQUEST_INTENT_TOKENS: frozenset[str] = frozenset(("immediately", "now", "recommend"))

_FIN_DCA_EXPLANATORY_MARKERS: frozenset[str] = frozenset(
    {
        "explain",
        "what",
        "how",
        "meaning",
        "difference",
        "compare",
        "example",
        "hypothetical",
        "educational",
        "general",
        "overview",
    }
)


def _contains_root_token(tokens: set[str], roots: tuple[str, ...]) -> bool:
    for token in tokens:
        for root in roots:
            if token.startswith(root):
                return True
    return False


def _finance_request_has_object_context(tokens: set[str]) -> bool:
    return _contains_root_token(tokens, _FIN_REQUEST_OBJECT_ROOTS)


def _finance_request_has_allocation_or_execution_action(tokens: set[str]) -> bool:
    return _contains_root_token(tokens, _FIN_REQUEST_ACTION_ROOTS)


def _finance_request_has_directive_timing_or_advice_intent(
    normalised_text: str,
    tokens: set[str],
) -> bool:
    wrapped = f" {normalised_text} "
    for phrase in _FIN_REQUEST_INTENT_PHRASES:
        if f" {phrase} " in wrapped:
            return True
    return any(token in _FIN_REQUEST_INTENT_TOKENS for token in tokens)


def _finance_request_has_personal_binding(
    normalised_text: str,
    tokens: set[str],
) -> bool:
    """True when the request is user-bound (personalized) rather than generic finance education."""
    wrapped = f" {normalised_text} "
    if " should i " in wrapped:
        return True
    if " advise me whether " in wrapped:
        return True
    if " tell me whether " in wrapped:
        return True
    if " my " in wrapped:
        return True
    return any(t in {"i", "me", "my", "mine"} for t in tokens)


def _finance_request_has_imperative_execution_surface(normalised_text: str) -> bool:
    """True for bare directives (e.g. *Reallocate the portfolio…*) which are still consequence-bearing."""
    return surface_finance_imperative_first_line(normalised_text)


def _finance_request_is_informational_leading_query(lower: str) -> bool:
    """True when the request opens as a definition / education question (finance seam only)."""
    return surface_finance_informational_leading(lower.strip())


def _finance_informational_override_advice_intent(lower: str) -> bool:
    """True when decision-seeking or execution language negates pure-informational carve-out."""
    norm = _normalise_request_text(lower)
    tokens = frozenset(norm.split())
    return surface_finance_informational_override(norm, tokens)


def _finance_request_is_dca_explanatory(
    normalised_text: str,
    tokens: set[str],
) -> bool:
    """True when a DCA prompt is explanatory/educational, not execution-seeking."""
    wrapped = f" {normalised_text} "
    has_dca = (" dollar cost averaging " in wrapped) or (" dca " in wrapped)
    if not has_dca:
        return False
    has_explanatory_marker = any(t in _FIN_DCA_EXPLANATORY_MARKERS for t in tokens)
    if not has_explanatory_marker:
        return False
    # Keep hard blocks for explicit execution/timing directives.
    if " advise me whether " in wrapped:
        return False
    if " whether to execute " in wrapped:
        return False
    if " should i " in wrapped:
        return False
    if " execute now " in wrapped:
        return False
    if " do it " in wrapped:
        return False
    if " now " in wrapped or " immediately " in wrapped:
        return False
    return True


def user_seeks_historical_market_data_lookup(user_input: str) -> bool:
    """True when the user asks for historical market/quote data, not personalized advice.

    Used to suppress PERSONALIZED_FINANCIAL_ADVICE from the financial_action+context gate
    when the model replies with epistemic refusal or factual lookup language (second-person
    'you may…' / 'you can check…') that matched weak advisory tokens ('consider', etc.).

    Does **not** emit a ``Flag`` and has no ``rule_id`` — not part of :class:`BlockedRequestRuleId`.
    """
    if not user_input or len(user_input.strip()) < 8:
        return False
    return surface_historical_market_data_lookup(collapsed_lower(user_input))


def _has_legal_context_in_pef(pef: PEFState | None) -> bool:
    """True when PEF contains any entity or relationship with legal keywords."""
    if pef is None:
        return False
    # Check entities
    for entity in pef.entities.values():
        if surface_legal_context_q_text(entity.name):
            return True
        for alias in entity.aliases:
            if surface_legal_context_q_text(alias):
                return True
    # Check relationships (evidence)
    for rel in pef.relationships:
        if rel.evidence and surface_legal_context_q_text(rel.evidence):
            return True
    return False


# ── Employment discrimination pre-LLM detection ────────────────────────────────
# Mirrors the patterns in checker.py:_check_employment_discrimination.
# Defined here (not imported from checker) to avoid circular imports.
import re as _re

_BLOCKED_EMPLOY_ACTION_RE = _re.compile(
    r"\b(?:hire|hiring|recruit|screen|filter|reject|fire|dismiss|demote|"
    r"avoid\s+hiring|not\s+hire|don['’]t\s+hire|exclude)\b",
    _re.IGNORECASE,
)
_BLOCKED_PROTECTED_CHAR_RE = _re.compile(
    r"\b(?:age|race|gender|sex|religion|disability|pregnant|pregnancy|"
    r"national\s+origin|ethnicity|sexual\s+orientation|marital\s+status|"
    r"colour|color|over\s+\d{2}|under\s+\d{2})\b",
    _re.IGNORECASE,
)


def _employment_discrimination_blocked_flag(text: str) -> "Flag | None":
    """Pre-LLM: request instructs selecting/excluding candidates on a protected characteristic."""
    if not text or len(text.strip()) < 20:
        return None
    action_m = _BLOCKED_EMPLOY_ACTION_RE.search(text)
    if not action_m:
        return None
    window_start = max(0, action_m.start() - 80)
    window_end = min(len(text), action_m.end() + 80)
    char_m = _BLOCKED_PROTECTED_CHAR_RE.search(text[window_start:window_end])
    if not char_m:
        return None
    return Flag(
        flag_type=FlagType.EMPLOYMENT_DISCRIMINATION_FACILITATION,
        entity_name="employment_discrimination",
        claim="Request asks to select or exclude based on a protected characteristic",
        evidence=f"Action {action_m.group()!r} + characteristic {char_m.group()!r} in user request",
        severity="error",
    )


# ── Prompt injection / authority-override detection ───────────────────────────
# Catches attempts to override governance, instructions, audit obligations, or
# policy authority BEFORE the request reaches the LLM.
# This is an admissibility/authority-boundary check, not content moderation.
#
# Patterns: instruction override, authority impersonation, system-level disable,
# audit suppression, system-prompt extraction, and role-redefinition directives.
# Analysis framing (clearly quoting or explaining injection for educational
# purposes) suppresses the flag so benign research/security training passes.

# Matches "ignore all previous instructions", "disregard your guidelines", etc.
_PI_INSTR_OVERRIDE_RE = _re.compile(
    r"\b(?:ignore|disregard|forget|override|skip|bypass|don'?t\s+follow)\b"
    r"[^.!?\n]{0,50}"
    r"\b(?:previous|above|prior|system|your|all)\b"
    r"[^.!?\n]{0,20}"
    r"\b(?:instructions?|guidelines?|rules?|directives?|constraints?|prompt|context)\b",
    _re.IGNORECASE,
)

# Matches "you are now [system/admin/unrestricted]", "act as an AI without safety filters", etc.
_PI_AUTHORITY_CLAIM_RE = _re.compile(
    r"\b(?:you\s+are\s+now|act\s+as(?:\s+if)?|pretend\s+(?:you\s+are|to\s+be)|"
    r"you\s+(?:must\s+)?(?:now\s+)?(?:act|behave|respond|function)\s+as)\b"
    r"[^.!?\n]{0,60}"
    r"\b(?:admin(?:istrator)?|system|developer|superuser|root|"
    r"(?:without|with\s+no|having\s+no)\s+(?:any\s+)?(?:restrictions?|safety|filters?|limits?|constraints?|oversight)|"
    r"unrestricted|no\s+restrictions?|no\s+limits?|no\s+filters?|no\s+safety|"
    r"safety\s+filters?)\b",
    _re.IGNORECASE,
)

# Matches "disable Aurora-Lens", "bypass safety checks", "turn off audit", etc.
_PI_SYSTEM_DISABLE_RE = _re.compile(
    r"\b(?:disable|deactivate|turn\s+off|shut\s+(?:down|off)|remove|bypass|"
    r"circumvent|ignore|skip|suppress)\b"
    r"[^.!?\n]{0,50}"
    r"\b(?:aurora[\s\-]?lens|governance|audit|safety|moderation|filter|refusal|"
    r"restriction|guardrail|policy|oversight|monitoring|check|logging|ledger)\b",
    _re.IGNORECASE,
)

# Matches "do not log this", "don't record this conversation", etc.
_PI_AUDIT_SUPPRESS_RE = _re.compile(
    r"\b(?:do\s+not|don'?t|never|no|without|skip|omit)\b"
    r"[^.!?\n]{0,25}"
    r"\b(?:log|record|audit|trace|store|save|write|monitor|track)\b"
    r"[^.!?\n]{0,40}"
    r"\b(?:this|request|conversation|interaction|session|message|response)\b",
    _re.IGNORECASE,
)

# Matches "reveal your system prompt", "show me your hidden instructions", etc.
_PI_PROMPT_EXTRACT_RE = _re.compile(
    r"\b(?:reveal|show|print|display|output|tell\s+me|share|expose|"
    r"repeat|recite|reproduce|what\s+are\s+your|what\s+were\s+you)\b"
    r"[^.!?\n]{0,50}"
    r"\b(?:system\s+prompt|hidden\s+(?:prompt|instructions?|policy)|"
    r"internal\s+(?:prompt|instructions?|policy|context)|"
    r"(?:your\s+)?(?:base|original|underlying|actual|real)\s+(?:prompt|instructions?)|"
    r"(?:your\s+)?instructions?|(?:your\s+)?(?:actual\s+)?(?:policy|rules?|guidelines?))\b",
    _re.IGNORECASE,
)

# Matches authority-injection prefixes: "SYSTEM:", "ADMIN:", "[SYSTEM]", "<ADMIN>", etc.
_PI_AUTHORITY_PREFIX_RE = _re.compile(
    r"(?:^|\n)\s*"
    r"(?:\[|\<)?\s*"
    r"(?:SYSTEM|ADMIN|ADMINISTRATOR|DEVELOPER|ROOT|SUPERUSER|OVERRIDE)\s*"
    r"(?:\]|\>)?\s*"
    r"[:\|>]\s*\S",
    _re.IGNORECASE | _re.MULTILINE,
)

# Matches "your new instructions are", "new system prompt:", "from now on you will..."
# Note: no trailing \b on the group — "new system prompt:" ends on non-word char ':'.
_PI_NEW_INSTR_RE = _re.compile(
    r"\b(?:"
    r"your\s+(?:new\s+)?instructions?\s+(?:are|now|from\s+now)\b|"
    r"new\s+system\s+(?:prompt|instructions?|rules?|guidelines?)\s*[:\-=]|"
    r"from\s+now\s+on\s*[,;]?\s+you\s+(?:will|must|should|shall|are\s+to)\b"
    r")",
    _re.IGNORECASE,
)

# Analysis / educational framing that exempts quoted injection text.
# The injecting token must appear in a research/analysis context.
_PI_ANALYSIS_FRAMING_RE = _re.compile(
    r"\b(?:example\s+of|an?\s+example|analyzing|analysis\s+of|studying|"
    r"discuss(?:ion\s+of)?|explain(?:ing)?|what\s+is|definition\s+of|"
    r"case\s+(?:study|of)|prompt\s+injection|security\s+(?:test(?:ing)?|research|"
    r"audit|assessment|training)|demonstrate|illustrate|educati(?:on|onal)|"
    r"show\s+(?:an?\s+)?example|this\s+is\s+an\s+example|for\s+(?:example|instance)|"
    r"consider\s+(?:this|the\s+following)|discuss\s+(?:this|the\s+following)|"
    r"attacker[s]?\s+use)\b",
    _re.IGNORECASE,
)

# Quotation guard: match appears inside single or double quotes.
_PI_QUOTED_RE = _re.compile(r"""['"][^'"]{3,200}['"]""")

_PI_PATTERNS = (
    _PI_INSTR_OVERRIDE_RE,
    _PI_AUTHORITY_CLAIM_RE,
    _PI_SYSTEM_DISABLE_RE,
    _PI_AUDIT_SUPPRESS_RE,
    _PI_PROMPT_EXTRACT_RE,
    _PI_AUTHORITY_PREFIX_RE,
    _PI_NEW_INSTR_RE,
)


def _prompt_injection_attempt_flag(text: str) -> "Flag | None":
    """Pre-LLM: detect attempts to override governance instructions, audit obligations,
    or policy authority via direct prompt injection.

    This is an admissibility/authority-boundary check, not content moderation.
    Returns None for texts with clear analysis or educational framing where the
    injection phrase is plausibly quoted for examination, not execution.
    """
    if not text or len(text.strip()) < 10:
        return None

    match = None
    for pat in _PI_PATTERNS:
        m = pat.search(text)
        if m:
            match = m
            break

    if match is None:
        return None

    # Exemption: analysis framing AND the matched span is plausibly quoted/referenced,
    # not being executed. Both conditions must hold to suppress the flag.
    if _PI_ANALYSIS_FRAMING_RE.search(text):
        span_start = match.start()
        for qm in _PI_QUOTED_RE.finditer(text):
            if qm.start() <= span_start < qm.end():
                # Match START is inside a quoted string in an analysis context.
                # Greedy patterns may extend past the close-quote — check start only.
                return None

    return Flag(
        flag_type=FlagType.PROMPT_INJECTION_ATTEMPT,
        entity_name="authority_boundary",
        claim="Request attempts to override governance, instructions, or policy authority",
        evidence=f"Pattern matched: {match.group()!r}",
        severity="error",
        rule_id=BlockedRequestRuleId.PROMPT_INJECTION_ATTEMPT,
    )


def _evaluate_blocked_act_request_core(
    user_input: str,
    *,
    pef: PEFState | None = None,
    include_encoded_rule: bool = True,
) -> list[Flag]:
    """Classify whether the user request is a determinate blocked regulated-domain act.

    Same semantics as :meth:`aurora_lens.verify.checker.Checker.check_blocked_act_request`.
    Each returned flag sets ``rule_id`` to a :class:`BlockedRequestRuleId` value.
    """
    if not user_input or len(user_input.strip()) < 5:
        return []

    flags: list[Flag] = []
    lower = user_input.lower()

    _synth = _illegal_hazardous_synthesis_flag(lower)
    if _synth:
        flags.append(_synth)

    # Dedicated token-span arm; skip when ontology already emitted an illegal /
    # hazard stop so we do not double-flag the same corridor.
    _bio_weapon = _bioweapon_generic_fabrication_flag(lower)
    if _bio_weapon and not any(
        f.flag_type == FlagType.ILLEGAL_INSTRUCTION for f in flags
    ):
        flags.append(_bio_weapon)

    _harmful_action = _harmful_action_request_flag(lower)
    if _harmful_action:
        return [_harmful_action]

    _financial_crime = _financial_crime_evasion_flag(lower)
    if _financial_crime:
        return [_financial_crime]

    _audit_evasion = _audit_trail_evasion_flag(lower)
    if _audit_evasion:
        return [_audit_evasion]

    _agency = _agency_violation_assistance_flag(lower)
    if _agency:
        # Cross-domain admissibility veto: no domain-specific laundering before block.
        return [_agency]

    if include_encoded_rule:
        _encoded = _encoded_jailbreak_attempt_flag(user_input, pef=pef)
        if _encoded:
            return [_encoded]

    # PI detection deferred: if dangerous content (bioweapon, synthesis) is also present,
    # the dangerous-content flag wins and PI is suppressed. PI is appended at the end
    # only when no other blocking flag fires.
    _pi = _prompt_injection_attempt_flag(user_input)

    _employ_disc = _employment_discrimination_blocked_flag(user_input)
    if _employ_disc:
        flags.append(_employ_disc)

    tok_tuple, norm_core = tokens_and_norm(user_input)
    tok_f_core = frozenset(tok_tuple)

    _req_mgkg = surface_req_mgkg(norm_core)
    _req_pediatric = surface_pediatric_indicator(tok_tuple)
    _req_dosing_or_titration_intent = surface_dosing_or_titration_intent(norm_core, tok_f_core)
    _req_medication_named = surface_medication_named(tok_f_core)

    if _req_mgkg or (_req_pediatric and _req_dosing_or_titration_intent) or (
        _req_medication_named and _req_pediatric
    ):
        flags.append(
            Flag(
                flag_type=FlagType.PEDIATRIC_DOSAGE_RECOMMENDATION,
                entity_name="medical",
                claim="Request asks for pediatric or weight-based medication dosing",
                evidence="mg/kg notation or pediatric indicator + dosing question in user request",
                severity="error",
            )
        )
        flags[-1].rule_id = BlockedRequestRuleId.PEDIATRIC_DOSING
        flags.append(
            Flag(
                flag_type=FlagType.PERSONALIZED_MEDICAL_ADVICE,
                entity_name="medical",
                claim="Request asks for personalised medication dosing guidance",
                evidence=(
                    "Pediatric dosing request — umbrella personalized medical category "
                    "for policy compatibility"
                ),
                severity="warning",
            )
        )
        flags[-1].rule_id = BlockedRequestRuleId.DELEGATED_DECISION_MEDICAL

    _req_personal_medical_frame = surface_personal_medical_frame(tok_tuple)
    if (
        _req_medication_named
        and _req_personal_medical_frame
        and _req_dosing_or_titration_intent
        and not _req_pediatric
    ):
        flags.append(
            Flag(
                flag_type=FlagType.PERSONALIZED_MEDICAL_ADVICE,
                entity_name="medical",
                claim="Request asks for personalised medication dosing or titration",
                evidence=(
                    "Named drug + personal clinical frame + dosing/titration intent "
                    "in user request (adult)"
                ),
                severity="warning",
            )
        )
        flags[-1].rule_id = BlockedRequestRuleId.ADULT_PERSONAL_MED_DOSING

    _req_fiction_wrapper = surface_fiction_wrapper(tok_f_core)
    _req_real_world_med_anchor = surface_real_world_med_anchor(norm_core, tok_tuple)
    _req_medical_decision_or_instruction = surface_medical_decision_or_instruction(tok_f_core)
    if (
        _req_fiction_wrapper
        and _req_real_world_med_anchor
        and _req_medical_decision_or_instruction
    ):
        flags.append(
            Flag(
                flag_type=FlagType.PERSONALIZED_MEDICAL_ADVICE,
                entity_name="medical",
                claim=(
                    "Request launders real-world medical decision/instruction "
                    "through fictional wrapper framing"
                ),
                evidence=(
                    "Composition wrapper + real-world user medical anchor + "
                    "medical decision/instruction intent in user request"
                ),
                severity="warning",
            )
        )
        flags[-1].rule_id = BlockedRequestRuleId.FICTIONAL_WRAPPER_REAL_MED_ACTION

    _req_med_topic = surface_medical_topic(tok_f_core)
    _req_med_edu = surface_medical_educational(norm_core)
    _req_first_person = surface_first_person(tok_f_core)
    _req_med_personal_diagnosis = surface_medical_personal_diagnosis(norm_core, tok_tuple)
    _req_med_personal_action = surface_medical_personal_action(norm_core, tok_tuple)
    _req_med_implicit_action = surface_medical_implicit_action(norm_core)
    # Educational medical questions (no personal frame, no action/decision intent)
    # must stay out of request-side blocked-act routing.
    if (
        _req_med_topic
        and _req_med_edu
        and not _req_first_person
        and not _req_med_personal_diagnosis
        and not _req_med_personal_action
    ):
        pass

    # Personalized diagnosis/treatment decision asks remain restricted.
    if (
        (_req_med_topic and (_req_med_personal_diagnosis or _req_med_personal_action))
        or _req_med_implicit_action
    ):
        flags.append(
            Flag(
                flag_type=FlagType.PERSONALIZED_MEDICAL_ADVICE,
                entity_name="medical",
                claim="Request asks for a personalized medical diagnosis or treatment decision",
                evidence="Personal diagnostic or action-seeking medical framing in user request",
                severity="warning",
            )
        )
        flags[-1].rule_id = BlockedRequestRuleId.DELEGATED_DECISION_MEDICAL

    _fin_object = _finance_request_has_object_context(tok_f_core)
    _fin_action = _finance_request_has_allocation_or_execution_action(tok_f_core)
    _fin_intent = _finance_request_has_directive_timing_or_advice_intent(norm_core, tok_f_core)
    _fin_personal = _finance_request_has_personal_binding(norm_core, tok_f_core)
    _fin_imp_exec = _finance_request_has_imperative_execution_surface(norm_core)
    _fin_dca_explanatory = _finance_request_is_dca_explanatory(norm_core, tok_f_core)

    if _fin_object and _fin_action and _fin_intent and (_fin_personal or _fin_imp_exec):
        # Informational leading forms run before imperative/decision composition for this arm:
        # "What is dollar-cost averaging?" must not pre-LLM block; "… should I …" still must.
        _fin_info_only = (
            _finance_request_is_informational_leading_query(lower)
            and not _finance_informational_override_advice_intent(lower)
        )
        if _fin_info_only:
            pass
        elif (not user_seeks_historical_market_data_lookup(user_input)) and (not _fin_dca_explanatory):
            flags.append(
                Flag(
                    flag_type=FlagType.PERSONALIZED_FINANCIAL_ADVICE,
                    entity_name="financial",
                    claim="Request asks for personalised financial advice about the user's own money",
                    evidence=(
                        "Financial object/context + allocation/execution action + "
                        "directive/timing intent in user request"
                    ),
                    severity="warning",
                )
            )
            flags[-1].rule_id = BlockedRequestRuleId.PERSONALIZED_FINANCIAL

    _legal_outcome = _legal_outcome_prediction_flag(lower, pef)
    if _legal_outcome:
        flags.append(_legal_outcome)

    if surface_academic_integrity_violation(norm_core, tok_tuple):
        flags.append(
            Flag(
                flag_type=FlagType.ACADEMIC_INTEGRITY_VIOLATION,
                entity_name="academic_integrity",
                claim="Request asks for assistance that would violate academic integrity policies",
                evidence="Academic integrity violation pattern detected in user request",
                severity="error",
            )
        )
        flags[-1].rule_id = BlockedRequestRuleId.ACADEMIC_INTEGRITY_VIOLATION

    if surface_student_record_exposure(norm_core, tok_f_core, user_input):
        flags.append(
            Flag(
                flag_type=FlagType.STUDENT_RECORD_EXPOSURE,
                entity_name="student_record",
                claim="Request involves student record data governed by privacy regulations",
                evidence="Student record exposure pattern detected in user request",
                severity="error",
            )
        )
        flags[-1].rule_id = BlockedRequestRuleId.STUDENT_RECORD_EXPOSURE

    if surface_personalized_academic_advice(norm_core, tok_f_core):
        flags.append(
            Flag(
                flag_type=FlagType.PERSONALIZED_ACADEMIC_ADVICE,
                entity_name="academic_advice",
                claim="Request asks for personalised academic admissions or credential advice",
                evidence="Personalised academic advice pattern detected in user request",
                severity="warning",
            )
        )
        flags[-1].rule_id = BlockedRequestRuleId.PERSONALIZED_ACADEMIC_ADVICE

    if surface_trade_secret_disclosure(norm_core, tok_f_core):
        flags.append(
            Flag(
                flag_type=FlagType.TRADE_SECRET_DISCLOSURE,
                entity_name="trade_secret",
                claim="Request asks for assistance that would involve disclosing trade secrets",
                evidence="Trade secret disclosure pattern detected in user request",
                severity="error",
            )
        )
        flags[-1].rule_id = BlockedRequestRuleId.TRADE_SECRET_DISCLOSURE

    if surface_insider_information(norm_core, tok_f_core):
        flags.append(
            Flag(
                flag_type=FlagType.INSIDER_INFORMATION_ASSISTANCE,
                entity_name="insider_information",
                claim="Request involves material non-public information",
                evidence="Insider information pattern detected in user request",
                severity="error",
            )
        )
        flags[-1].rule_id = BlockedRequestRuleId.INSIDER_INFORMATION_ASSISTANCE

    if surface_procurement_fraud(norm_core, tok_f_core):
        flags.append(
            Flag(
                flag_type=FlagType.PROCUREMENT_FRAUD_FACILITATION,
                entity_name="procurement_fraud",
                claim="Request asks for assistance that would facilitate procurement fraud",
                evidence="Procurement fraud pattern detected in user request",
                severity="error",
            )
        )
        flags[-1].rule_id = BlockedRequestRuleId.PROCUREMENT_FRAUD_FACILITATION

    if surface_business_strategy_full(norm_core, tok_f_core, user_input):
        flags.append(
            Flag(
                flag_type=FlagType.PERSONALIZED_BUSINESS_STRATEGY_ADVICE,
                entity_name="business_strategy",
                claim="Request asks for personalised strategic advice for a named company",
                evidence="Personalised business strategy pattern detected in user request",
                severity="warning",
            )
        )
        flags[-1].rule_id = BlockedRequestRuleId.PERSONALIZED_BUSINESS_STRATEGY

    if surface_bare_adjudication_merits(norm_core, tok_tuple) and surface_bare_adjudication_legal_domain(
        norm_core, tok_f_core
    ):
        flags.append(
            Flag(
                flag_type=FlagType.PERSONALIZED_LEGAL_ADVICE,
                entity_name="legal",
                claim=(
                    "Request asks for bare legal adjudication (who should win or prevail) "
                    "in a case or dispute context"
                ),
                evidence=(
                    "Merits or winner-allocation question + legal or family-law "
                    "proceeding surface in user request"
                ),
                severity="warning",
            )
        )
        flags[-1].rule_id = BlockedRequestRuleId.BARE_LEGAL_ADJUDICATION

    # Intent-class detectors — surface-independent, compositional.
    _bypass = _credential_access_bypass_flag(lower)
    if _bypass:
        flags.append(_bypass)

    _triage = _definitive_medical_triage_flag(lower)
    if _triage:
        flags.append(_triage)

    _delegated = _delegated_real_consequence_flags(lower)
    if _delegated:
        flags.extend(_delegated)

    _case_res = _case_resolution_function_flags(lower)
    if _case_res:
        flags.extend(_case_res)

    _case_para = _case_resolution_paraphrase_flag(lower)
    if _case_para and not any(f.flag_type == FlagType.PERSONALIZED_LEGAL_ADVICE for f in flags):
        flags.append(_case_para)

    # Deferred PI flag: only surface when no other blocking flag fired.
    # When dangerous content (bioweapon, synthesis) co-occurs with a jailbreak preamble,
    # the dangerous-content flag is the operative signal; PI is not added.
    if not flags and _pi is not None:
        return [_pi]

    # Violent criminal intent — independent of legal-outcome classification.
    for _vci_flag in violent_criminal_intent_flags(user_input):
        if not any(f.flag_type == _vci_flag.flag_type for f in flags):
            flags.append(_vci_flag)

    # Deterministic flag precedence sort: highest priority first.
    # This ensures that if multiple flags are present (e.g. bioweapon + legal keywords),
    # the most critical one (ILLEGAL_INSTRUCTION) becomes the primary flag
    # for intervention text generation.
    flags.sort(key=lambda f: _FLAG_PRECEDENCE_LADDER.get(f.flag_type, 0), reverse=True)

    return flags


def evaluate_blocked_act_request(
    user_input: str,
    *,
    pef: PEFState | None = None,
) -> list[Flag]:
    return _evaluate_blocked_act_request_core(
        user_input,
        pef=pef,
        include_encoded_rule=True,
    )

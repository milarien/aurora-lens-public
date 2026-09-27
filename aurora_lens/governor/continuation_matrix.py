"""Canonical Governor continuation matrix.

This module is the sole authoritative source for the lawful continuation
parameters that the Governor may issue after a governance decision.

DESIGN CONTRACT
---------------
The matrix is a static, exhaustive lookup table keyed by:

    "<domain>:<authority_class>:<admissibility_state>"         (3-component)
    "<domain>:<authority_class>:<admissibility_state>:<disc>"  (4-component)

where <disc> is either a flag-class reason code (e.g. SELF_HARM_INSTRUCTION)
or a user-class discriminator (e.g. auditor, clinician).  No regex.  No
heuristic routing.  No fallback logic inside the table itself.

LOOKUP CONTRACT
---------------
ContinuationMatrix.lookup() uses an explicit four-level fallback ladder:

    1. domain:authority:status:discriminator  (flag-class or user-class key)
    2. domain:authority:status               (authority key)
    3. domain:GP:status                      (GP authority fallback)
    4. general:GP:status                     (global status fallback)
    5. general:GP:STOP                       (safe fallback — always present)

Each level is a pure dict lookup.  No regex.  No heuristic.

EPISTEMIC INVARIANT
-------------------
Every row whose status is ASK, REFUSE, or STOP must have commitment_closed=True.
This invariant is checked at module import time and raises ValueError if violated.
No row may widen epistemic commitment after a non-ADMIT outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .models import (
    ContinuationPathway,
    ForensicObligation,
    LensStatus,
    OutputMode,
)


# ── Tuple aliases for forensic obligations (frozen → hashable) -----------------

_FE = (ForensicObligation.EMIT_FORENSIC_ENVELOPE,)
_FE_PEF = (
    ForensicObligation.EMIT_FORENSIC_ENVELOPE,
    ForensicObligation.ATTACH_PEF_SNAPSHOT,
)
_NO_FO: tuple[ForensicObligation, ...] = ()


# ── ContinuationRow ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ContinuationRow:
    """Lawful continuation parameters for one (domain, authority, status) cell.

    These are the only fields that the Governor may use to route a decision
    to a renderer.  They do not include speech-act permissions, disclosure
    requirements, or procedural actions — those live in the full GovernorPolicy.

    Fields
    ------
    pathway_id          : The canonical pathway to execute.
    commitment_closed   : True ⟹ no further determination may be issued or
                          recovered from the blocked response.
    interaction_open    : True ⟹ the conversation may continue after this stop.
    output_mode         : Rendering contract for the response.
    forensic_obligations: Audit obligations that attach at this decision point.
    escalation_target   : Policy-sourced resource string (never from user input
                          or LLM output).  None when no redirect is lawful.
    """

    pathway_id: ContinuationPathway
    commitment_closed: bool
    interaction_open: bool
    output_mode: OutputMode
    forensic_obligations: tuple[ForensicObligation, ...]
    escalation_target: Optional[str] = None


# ── Module-level aliases for table compactness ────────────────────────────────

_P = ContinuationPathway
_O = OutputMode


# ── Static continuation table ─────────────────────────────────────────────────
#
# Column layout:
#   Key                                         pathway_id                    cc     io     output_mode                      forensic_obligations   escalation_target
#
# cc = commitment_closed, io = interaction_open
#
# Invariant: all rows with status ∈ {ASK, REFUSE, STOP} must have cc=True.
#            Checked by _validate() at import time.

_TABLE: dict[str, ContinuationRow] = {

    # ── general:GP ────────────────────────────────────────────────────────────
    "general:GP:ADMIT":  ContinuationRow(_P.P_ADMIT_STANDARD,          False, True,  _O.FULL_RESPONSE,           _NO_FO),
    "general:GP:ASK":    ContinuationRow(_P.P_ASK_MISSING_FACT,         True,  True,  _O.CLARIFICATION_REQUEST,   _FE),
    "general:GP:REFUSE": ContinuationRow(_P.P_REFUSE_EXPLAIN_REDIRECT,  True,  True,  _O.REFUSAL_WITH_EXPLANATION, _FE),
    "general:GP:STOP":   ContinuationRow(_P.P_STOP_TERMINAL,            True,  False, _O.TERMINAL_STOP,            _FE),

    # Auditor user-class override — forensic visibility, deeper audit obligations
    "general:GP:STOP:auditor": ContinuationRow(_P.P_STOP_TERMINAL, True, False, _O.FORENSIC_STOP, _FE_PEF),

    # ── ambiguity:GP ──────────────────────────────────────────────────────────
    "ambiguity:GP:ADMIT":  ContinuationRow(_P.P_ADMIT_STANDARD,         False, True,  _O.FULL_RESPONSE,              _NO_FO),
    "ambiguity:GP:ASK":    ContinuationRow(_P.P_ASK_DISAMBIGUATE,        True,  True,  _O.CLARIFICATION_REQUEST,      _FE),
    "ambiguity:GP:REFUSE": ContinuationRow(_P.P_REFUSE_EXPLAIN_REDIRECT, True,  True,  _O.REFUSAL_WITH_CLARIFICATION,  _FE),
    "ambiguity:GP:STOP":   ContinuationRow(_P.P_STOP_TERMINAL,           True,  False, _O.TERMINAL_STOP,               _FE),

    # ── medical:GP ────────────────────────────────────────────────────────────
    # ADMIT: constrained corridor — GP may summarise/handoff only, not determine
    "medical:GP:ADMIT":  ContinuationRow(_P.P_HANDOFF_SUMMARY,          True,  True,  _O.CONSTRAINED_RESPONSE,    _FE,
                                          "a licensed clinician"),
    "medical:GP:ASK":    ContinuationRow(_P.P_ASK_MISSING_FACT,          True,  True,  _O.CLARIFICATION_REQUEST,   _FE,
                                          "a licensed clinician"),
    "medical:GP:REFUSE": ContinuationRow(_P.P_REFUSE_EXPLAIN_REDIRECT,   True,  True,  _O.REFUSAL_WITH_EXPLANATION, _FE,
                                          "a licensed clinician or healthcare provider"),

    # Clinician user-class override for REFUSE — richer handoff corridor
    "medical:GP:REFUSE:clinician": ContinuationRow(_P.P_REFUSE_ESCALATE_PRO, True, True, _O.PRO_REFUSAL, _FE_PEF,
                                                    "a supervising clinician"),

    # STOP: escalation ladder, overridden per flag-class below
    "medical:GP:STOP":   ContinuationRow(_P.P_STOP_ESCALATE,             True,  False, _O.TERMINAL_STOP,            _FE,
                                          "your nearest GP, nurse on call, or call an ambulance"),

    # medical:GP flag-class STOP overrides
    "medical:GP:STOP:EMERGENCY_TRIAGE_GUIDANCE":     ContinuationRow(
        _P.P_STOP_ESCALATE_EMERGENCY, True, False, _O.TERMINAL_STOP, _FE,
        "your nearest GP, nurse on call, or call an ambulance"),
    "medical:GP:STOP:SELF_HARM_INSTRUCTION":         ContinuationRow(
        _P.P_STOP_SUPPORTIVE_DEESCALATE, True, True, _O.TERMINAL_STOP, _FE,
        "a crisis support service or emergency services"),
    "medical:GP:STOP:MEDICAL_DOSAGE_RECOMMENDATION": ContinuationRow(
        _P.P_STOP_REDIRECT_QUALIFIED, True, True, _O.TERMINAL_STOP, _FE,
        "a pharmacist, prescribing clinician, or urgent care provider"),
    "medical:GP:STOP:PEDIATRIC_DOSAGE_RECOMMENDATION": ContinuationRow(
        _P.P_STOP_REDIRECT_QUALIFIED, True, True, _O.TERMINAL_STOP, _FE,
        "a pharmacist, prescribing clinician, or urgent care provider"),
    "medical:GP:STOP:NUMERIC_MEDICAL_INSTRUCTION":   ContinuationRow(
        _P.P_STOP_REDIRECT_QUALIFIED, True, True, _O.TERMINAL_STOP, _FE,
        "a pharmacist, prescribing clinician, or urgent care provider"),
    "medical:GP:STOP:PERSONALIZED_MEDICAL_ADVICE":   ContinuationRow(
        _P.P_STOP_REDIRECT_QUALIFIED, True, True, _O.TERMINAL_STOP, _FE,
        "a licensed clinician or healthcare provider"),

    # ── medical:DA ────────────────────────────────────────────────────────────
    "medical:DA:ADMIT":  ContinuationRow(_P.P_ADMIT_STANDARD,      False, True,  _O.FULL_RESPONSE,  _FE,
                                          "a supervising clinician"),
    "medical:DA:ASK":    ContinuationRow(_P.P_ASK_MISSING_FACT,     True,  True,  _O.CLARIFICATION_REQUEST, _FE,
                                          "a supervising clinician"),
    "medical:DA:REFUSE": ContinuationRow(_P.P_REFUSE_ESCALATE_PRO,  True,  True,  _O.PRO_REFUSAL,    _FE_PEF,
                                          "a supervising clinician"),
    "medical:DA:STOP":   ContinuationRow(_P.P_STOP_TERMINAL,        True,  False, _O.TERMINAL_STOP,  _FE_PEF,
                                          "a supervising clinician"),

    # ── medical:HS ────────────────────────────────────────────────────────────
    "medical:HS:ADMIT":  ContinuationRow(_P.P_ADMIT_STANDARD,      False, True,  _O.FULL_RESPONSE,  _FE,
                                          "the supervising clinician"),
    "medical:HS:ASK":    ContinuationRow(_P.P_ASK_MISSING_FACT,     True,  True,  _O.CLARIFICATION_REQUEST, _FE,
                                          "the supervising clinician"),
    "medical:HS:REFUSE": ContinuationRow(_P.P_REFUSE_ESCALATE_PRO,  True,  True,  _O.PRO_REFUSAL,    _FE_PEF,
                                          "the supervising clinician"),
    "medical:HS:STOP":   ContinuationRow(_P.P_STOP_FORENSIC,        True,  False, _O.FORENSIC_STOP,  _FE_PEF,
                                          "the supervising clinician"),

    # ── harmful:GP ────────────────────────────────────────────────────────────
    # Hostile / illicit operational content. Clean refuse — no professional redirect.
    "harmful:GP:ADMIT":  ContinuationRow(_P.P_ADMIT_STANDARD,          False, True,  _O.FULL_RESPONSE,           _NO_FO),
    "harmful:GP:ASK":    ContinuationRow(_P.P_ASK_MISSING_FACT,         True,  True,  _O.CLARIFICATION_REQUEST,   _FE),
    "harmful:GP:REFUSE": ContinuationRow(_P.P_REFUSE_EXPLAIN_REDIRECT,  True,  True,  _O.REFUSAL_WITH_EXPLANATION, _FE),
    "harmful:GP:STOP":   ContinuationRow(_P.P_STOP_REFUSE_CLEAN,        True,  True,  _O.TERMINAL_STOP,            _FE, None),

    # harmful:GP flag-class STOP overrides — preserve clean refuse for illicit / hazard
    "harmful:GP:STOP:ILLEGAL_INSTRUCTION": ContinuationRow(
        _P.P_STOP_REFUSE_CLEAN, True, True, _O.TERMINAL_STOP, _FE, None),
    "harmful:GP:STOP:HAZARD_ONTOLOGY_UNAVAILABLE": ContinuationRow(
        _P.P_STOP_REFUSE_CLEAN, True, True, _O.TERMINAL_STOP, _FE, None),

    # ── legal:GP ──────────────────────────────────────────────────────────────
    # ADMIT: constrained corridor — not legal advice
    "legal:GP:ADMIT":  ContinuationRow(_P.P_HANDOFF_SUMMARY,          True,  True,  _O.CONSTRAINED_RESPONSE,    _FE,
                                        "a licensed lawyer"),
    "legal:GP:ASK":    ContinuationRow(_P.P_ASK_MISSING_FACT,          True,  True,  _O.CLARIFICATION_REQUEST,   _FE,
                                        "a licensed lawyer"),
    "legal:GP:REFUSE": ContinuationRow(_P.P_REFUSE_ESCALATE_PRO,       True,  True,  _O.REFUSAL_WITH_EXPLANATION, _FE,
                                        "a licensed lawyer or legal aid service"),
    # STOP: interaction remains open for procedural routing
    "legal:GP:STOP":   ContinuationRow(_P.P_STOP_TERMINAL,             True,  True,  _O.TERMINAL_STOP,            _FE,
                                        "a licensed lawyer or legal aid service"),

    # legal:GP flag-class STOP overrides
    "legal:GP:STOP:PERSONALIZED_LEGAL_ADVICE": ContinuationRow(
        _P.P_STOP_REDIRECT_QUALIFIED, True, True, _O.TERMINAL_STOP, _FE,
        "a licensed lawyer or legal aid service"),
    "legal:GP:STOP:ILLEGAL_INSTRUCTION":  ContinuationRow(
        _P.P_STOP_REFUSE_CLEAN, True, True, _O.TERMINAL_STOP, _FE, None),
    "legal:GP:STOP:TARGETED_DEFAMATION":  ContinuationRow(
        _P.P_STOP_REFUSE_CLEAN, True, True, _O.TERMINAL_STOP, _FE, None),

    # ── legal:DA ──────────────────────────────────────────────────────────────
    "legal:DA:ADMIT":  ContinuationRow(_P.P_ADMIT_STANDARD,      False, True,  _O.FULL_RESPONSE,  _FE,
                                        "a supervising lawyer"),
    "legal:DA:ASK":    ContinuationRow(_P.P_ASK_MISSING_FACT,     True,  True,  _O.CLARIFICATION_REQUEST, _FE,
                                        "a supervising lawyer"),
    "legal:DA:REFUSE": ContinuationRow(_P.P_REFUSE_ESCALATE_PRO,  True,  True,  _O.PRO_REFUSAL,    _FE_PEF,
                                        "a supervising lawyer"),
    "legal:DA:STOP":   ContinuationRow(_P.P_STOP_TERMINAL,        True,  False, _O.TERMINAL_STOP,  _FE_PEF,
                                        "a supervising lawyer"),

    # ── legal:HS ──────────────────────────────────────────────────────────────
    "legal:HS:ADMIT":  ContinuationRow(_P.P_ADMIT_STANDARD,      False, True,  _O.FULL_RESPONSE,  _FE,
                                        "the supervising lawyer"),
    "legal:HS:ASK":    ContinuationRow(_P.P_ASK_MISSING_FACT,     True,  True,  _O.CLARIFICATION_REQUEST, _FE,
                                        "the supervising lawyer"),
    "legal:HS:REFUSE": ContinuationRow(_P.P_REFUSE_ESCALATE_PRO,  True,  True,  _O.PRO_REFUSAL,    _FE_PEF,
                                        "the supervising lawyer"),
    "legal:HS:STOP":   ContinuationRow(_P.P_STOP_FORENSIC,        True,  False, _O.FORENSIC_STOP,  _FE_PEF,
                                        "the supervising lawyer"),

    # ── finance:GP ────────────────────────────────────────────────────────────
    # ADMIT: constrained corridor — not financial advice
    "finance:GP:ADMIT":  ContinuationRow(_P.P_HANDOFF_SUMMARY,          True,  True,  _O.CONSTRAINED_RESPONSE,    _FE,
                                          "a licensed financial adviser"),
    "finance:GP:ASK":    ContinuationRow(_P.P_ASK_MISSING_FACT,          True,  True,  _O.CLARIFICATION_REQUEST,   _FE,
                                          "a licensed financial adviser"),
    "finance:GP:REFUSE": ContinuationRow(_P.P_REFUSE_ESCALATE_PRO,       True,  True,  _O.REFUSAL_WITH_EXPLANATION, _FE,
                                          "a licensed financial adviser"),
    # STOP: interaction remains open for procedural routing
    "finance:GP:STOP":   ContinuationRow(_P.P_STOP_TERMINAL,             True,  True,  _O.TERMINAL_STOP,            _FE,
                                          "a licensed financial adviser"),

    # finance:GP flag-class STOP override
    "finance:GP:STOP:PERSONALIZED_FINANCIAL_ADVICE": ContinuationRow(
        _P.P_STOP_REDIRECT_QUALIFIED, True, True, _O.TERMINAL_STOP, _FE,
        "a licensed financial adviser"),

    # ── finance:DA ────────────────────────────────────────────────────────────
    "finance:DA:ADMIT":  ContinuationRow(_P.P_ADMIT_STANDARD,      False, True,  _O.FULL_RESPONSE,  _FE,
                                          "a supervising licensed adviser"),
    "finance:DA:ASK":    ContinuationRow(_P.P_ASK_MISSING_FACT,     True,  True,  _O.CLARIFICATION_REQUEST, _FE,
                                          "a supervising licensed adviser"),
    "finance:DA:REFUSE": ContinuationRow(_P.P_REFUSE_ESCALATE_PRO,  True,  True,  _O.PRO_REFUSAL,    _FE_PEF,
                                          "a supervising licensed adviser"),
    "finance:DA:STOP":   ContinuationRow(_P.P_STOP_TERMINAL,        True,  False, _O.TERMINAL_STOP,  _FE_PEF,
                                          "a supervising licensed adviser"),

    # ── finance:HS ────────────────────────────────────────────────────────────
    "finance:HS:ADMIT":  ContinuationRow(_P.P_ADMIT_STANDARD,      False, True,  _O.FULL_RESPONSE,  _FE,
                                          "the supervising licensed adviser"),
    "finance:HS:ASK":    ContinuationRow(_P.P_ASK_MISSING_FACT,     True,  True,  _O.CLARIFICATION_REQUEST, _FE,
                                          "the supervising licensed adviser"),
    "finance:HS:REFUSE": ContinuationRow(_P.P_REFUSE_ESCALATE_PRO,  True,  True,  _O.PRO_REFUSAL,    _FE_PEF,
                                          "the supervising licensed adviser"),
    "finance:HS:STOP":   ContinuationRow(_P.P_STOP_FORENSIC,        True,  False, _O.FORENSIC_STOP,  _FE_PEF,
                                          "the supervising licensed adviser"),

    # ── research:GP ───────────────────────────────────────────────────────────
    "research:GP:ADMIT":  ContinuationRow(_P.P_ADMIT_STANDARD,         False, True,  _O.FULL_RESPONSE,           _NO_FO),
    "research:GP:ASK":    ContinuationRow(_P.P_ASK_MISSING_FACT,        True,  True,  _O.CLARIFICATION_REQUEST,   _FE),
    "research:GP:REFUSE": ContinuationRow(_P.P_REFUSE_EXPLAIN_REDIRECT,  True,  True,  _O.REFUSAL_WITH_EXPLANATION, _FE),
    "research:GP:STOP":   ContinuationRow(_P.P_STOP_TERMINAL,            True,  False, _O.TERMINAL_STOP,            _FE),

    # ── crisis:GP ─────────────────────────────────────────────────────────────
    # NOTE: The original policy_matrix.json contained commitment_closed=false for
    # crisis:GP:ASK and crisis:GP:REFUSE, which violates the epistemic invariant
    # (GovernorPolicy.__post_init__ raises ValueError on those values).  Those
    # entries were unreachable dead code.  The matrix canonicalises them to
    # commitment_closed=True as the invariant requires.
    "crisis:GP:ADMIT":  ContinuationRow(_P.P_ADMIT_STANDARD,         False, True,  _O.FULL_RESPONSE,           _NO_FO),
    "crisis:GP:ASK":    ContinuationRow(_P.P_ASK_DISAMBIGUATE,        True,  True,  _O.CLARIFICATION_REQUEST,   _NO_FO),
    "crisis:GP:REFUSE": ContinuationRow(_P.P_REFUSE_EXPLAIN_REDIRECT,  True,  True,  _O.REFUSAL_WITH_EXPLANATION, _FE),
    "crisis:GP:STOP":   ContinuationRow(_P.P_STOP_TERMINAL,            True,  True,  _O.TERMINAL_STOP,            _FE,
                                         "a crisis support service or emergency services"),

    # ── education:GP ──────────────────────────────────────────────────────────
    "education:GP:ADMIT":  ContinuationRow(_P.P_HANDOFF_SUMMARY,          True,  True,  _O.CONSTRAINED_RESPONSE,    _FE,
                                            "a qualified educator or academic adviser"),
    "education:GP:ASK":    ContinuationRow(_P.P_ASK_MISSING_FACT,          True,  True,  _O.CLARIFICATION_REQUEST,   _FE,
                                            "a qualified educator or academic adviser"),
    "education:GP:REFUSE": ContinuationRow(_P.P_REFUSE_ESCALATE_PRO,       True,  True,  _O.REFUSAL_WITH_EXPLANATION, _FE,
                                            "a qualified educator or academic adviser"),
    "education:GP:STOP":   ContinuationRow(_P.P_STOP_TERMINAL,             True,  False, _O.TERMINAL_STOP,            _FE),

    # education:GP flag-class STOP overrides
    "education:GP:STOP:ACADEMIC_INTEGRITY_VIOLATION": ContinuationRow(
        _P.P_STOP_REFUSE_CLEAN, True, True, _O.TERMINAL_STOP, _FE, None),
    "education:GP:STOP:STUDENT_RECORD_EXPOSURE": ContinuationRow(
        _P.P_STOP_REFUSE_CLEAN, True, False, _O.TERMINAL_STOP, _FE, None),
    "education:GP:STOP:PERSONALIZED_ACADEMIC_ADVICE": ContinuationRow(
        _P.P_STOP_REDIRECT_QUALIFIED, True, True, _O.TERMINAL_STOP, _FE,
        "a qualified educational adviser or admissions officer"),

    # ── education:DA ──────────────────────────────────────────────────────────
    "education:DA:ADMIT":  ContinuationRow(_P.P_ADMIT_STANDARD,      False, True,  _O.FULL_RESPONSE,  _FE,
                                            "a supervising educator"),
    "education:DA:ASK":    ContinuationRow(_P.P_ASK_MISSING_FACT,     True,  True,  _O.CLARIFICATION_REQUEST, _FE,
                                            "a supervising educator"),
    "education:DA:REFUSE": ContinuationRow(_P.P_REFUSE_ESCALATE_PRO,  True,  True,  _O.PRO_REFUSAL,    _FE_PEF,
                                            "a supervising educator"),
    "education:DA:STOP":   ContinuationRow(_P.P_STOP_TERMINAL,        True,  False, _O.TERMINAL_STOP,  _FE_PEF,
                                            "a supervising educator"),

    # ── education:HS ──────────────────────────────────────────────────────────
    "education:HS:ADMIT":  ContinuationRow(_P.P_ADMIT_STANDARD,      False, True,  _O.FULL_RESPONSE,  _FE,
                                            "the supervising educator"),
    "education:HS:ASK":    ContinuationRow(_P.P_ASK_MISSING_FACT,     True,  True,  _O.CLARIFICATION_REQUEST, _FE,
                                            "the supervising educator"),
    "education:HS:REFUSE": ContinuationRow(_P.P_REFUSE_ESCALATE_PRO,  True,  True,  _O.PRO_REFUSAL,    _FE_PEF,
                                            "the supervising educator"),
    "education:HS:STOP":   ContinuationRow(_P.P_STOP_FORENSIC,        True,  False, _O.FORENSIC_STOP,  _FE_PEF,
                                            "the supervising educator"),

    # ── workforce:GP ──────────────────────────────────────────────────────────
    "workforce:GP:ADMIT":  ContinuationRow(_P.P_HANDOFF_SUMMARY,          True,  True,  _O.CONSTRAINED_RESPONSE,    _FE,
                                            "a qualified employment lawyer or HR specialist"),
    "workforce:GP:ASK":    ContinuationRow(_P.P_ASK_MISSING_FACT,          True,  True,  _O.CLARIFICATION_REQUEST,   _FE,
                                            "a qualified employment lawyer or HR specialist"),
    "workforce:GP:REFUSE": ContinuationRow(_P.P_REFUSE_ESCALATE_PRO,       True,  True,  _O.REFUSAL_WITH_EXPLANATION, _FE,
                                            "a qualified employment lawyer or HR specialist"),
    "workforce:GP:STOP":   ContinuationRow(_P.P_STOP_TERMINAL,             True,  False, _O.TERMINAL_STOP,            _FE),

    # workforce:GP flag-class STOP overrides
    "workforce:GP:STOP:EMPLOYMENT_DISCRIMINATION_FACILITATION": ContinuationRow(
        _P.P_STOP_REFUSE_CLEAN, True, False, _O.TERMINAL_STOP, _FE, None),
    "workforce:GP:STOP:EMPLOYEE_RECORD_EXPOSURE": ContinuationRow(
        _P.P_STOP_REFUSE_CLEAN, True, False, _O.TERMINAL_STOP, _FE, None),
    "workforce:GP:STOP:PERSONALIZED_EMPLOYMENT_ADVICE": ContinuationRow(
        _P.P_STOP_REDIRECT_QUALIFIED, True, True, _O.TERMINAL_STOP, _FE,
        "a qualified employment lawyer or citizens advice service"),

    # ── workforce:DA ──────────────────────────────────────────────────────────
    "workforce:DA:ADMIT":  ContinuationRow(_P.P_ADMIT_STANDARD,      False, True,  _O.FULL_RESPONSE,  _FE,
                                            "a supervising HR specialist"),
    "workforce:DA:ASK":    ContinuationRow(_P.P_ASK_MISSING_FACT,     True,  True,  _O.CLARIFICATION_REQUEST, _FE,
                                            "a supervising HR specialist"),
    "workforce:DA:REFUSE": ContinuationRow(_P.P_REFUSE_ESCALATE_PRO,  True,  True,  _O.PRO_REFUSAL,    _FE_PEF,
                                            "a supervising HR specialist"),
    "workforce:DA:STOP":   ContinuationRow(_P.P_STOP_TERMINAL,        True,  False, _O.TERMINAL_STOP,  _FE_PEF,
                                            "a supervising HR specialist"),

    # ── workforce:HS ──────────────────────────────────────────────────────────
    "workforce:HS:ADMIT":  ContinuationRow(_P.P_ADMIT_STANDARD,      False, True,  _O.FULL_RESPONSE,  _FE,
                                            "the supervising HR specialist"),
    "workforce:HS:ASK":    ContinuationRow(_P.P_ASK_MISSING_FACT,     True,  True,  _O.CLARIFICATION_REQUEST, _FE,
                                            "the supervising HR specialist"),
    "workforce:HS:REFUSE": ContinuationRow(_P.P_REFUSE_ESCALATE_PRO,  True,  True,  _O.PRO_REFUSAL,    _FE_PEF,
                                            "the supervising HR specialist"),
    "workforce:HS:STOP":   ContinuationRow(_P.P_STOP_FORENSIC,        True,  False, _O.FORENSIC_STOP,  _FE_PEF,
                                            "the supervising HR specialist"),

    # ── enterprise:GP ─────────────────────────────────────────────────────────
    "enterprise:GP:ADMIT":  ContinuationRow(_P.P_HANDOFF_SUMMARY,          True,  True,  _O.CONSTRAINED_RESPONSE,    _FE,
                                             "a qualified legal or compliance professional"),
    "enterprise:GP:ASK":    ContinuationRow(_P.P_ASK_MISSING_FACT,          True,  True,  _O.CLARIFICATION_REQUEST,   _FE,
                                             "a qualified legal or compliance professional"),
    "enterprise:GP:REFUSE": ContinuationRow(_P.P_REFUSE_EXPLAIN_REDIRECT,   True,  True,  _O.REFUSAL_WITH_EXPLANATION, _FE,
                                             "a qualified legal or compliance professional"),
    "enterprise:GP:STOP":   ContinuationRow(_P.P_STOP_FORENSIC,             True,  False, _O.FORENSIC_STOP,            _FE_PEF),

    # enterprise:GP flag-class STOP overrides
    "enterprise:GP:STOP:TRADE_SECRET_DISCLOSURE": ContinuationRow(
        _P.P_STOP_FORENSIC, True, False, _O.FORENSIC_STOP, _FE_PEF, None),
    "enterprise:GP:STOP:INSIDER_INFORMATION_ASSISTANCE": ContinuationRow(
        _P.P_STOP_FORENSIC, True, False, _O.FORENSIC_STOP, _FE_PEF, None),
    "enterprise:GP:STOP:PROCUREMENT_FRAUD_FACILITATION": ContinuationRow(
        _P.P_STOP_FORENSIC, True, False, _O.FORENSIC_STOP, _FE_PEF, None),
    "enterprise:GP:STOP:PERSONALIZED_BUSINESS_STRATEGY_ADVICE": ContinuationRow(
        _P.P_STOP_REDIRECT_QUALIFIED, True, True, _O.TERMINAL_STOP, _FE,
        "a qualified business or strategy consultant"),

    # ── enterprise:DA ─────────────────────────────────────────────────────────
    "enterprise:DA:ADMIT":  ContinuationRow(_P.P_ADMIT_STANDARD,      False, True,  _O.FULL_RESPONSE,  _FE,
                                             "a supervising compliance officer"),
    "enterprise:DA:ASK":    ContinuationRow(_P.P_ASK_MISSING_FACT,     True,  True,  _O.CLARIFICATION_REQUEST, _FE,
                                             "a supervising compliance officer"),
    "enterprise:DA:REFUSE": ContinuationRow(_P.P_REFUSE_ESCALATE_PRO,  True,  True,  _O.PRO_REFUSAL,    _FE_PEF,
                                             "a supervising compliance officer"),
    "enterprise:DA:STOP":   ContinuationRow(_P.P_STOP_FORENSIC,        True,  False, _O.FORENSIC_STOP,  _FE_PEF,
                                             "a supervising compliance officer"),

    # ── enterprise:HS ─────────────────────────────────────────────────────────
    "enterprise:HS:ADMIT":  ContinuationRow(_P.P_ADMIT_STANDARD,      False, True,  _O.FULL_RESPONSE,  _FE,
                                             "the supervising compliance officer"),
    "enterprise:HS:ASK":    ContinuationRow(_P.P_ASK_MISSING_FACT,     True,  True,  _O.CLARIFICATION_REQUEST, _FE,
                                             "the supervising compliance officer"),
    "enterprise:HS:REFUSE": ContinuationRow(_P.P_REFUSE_ESCALATE_PRO,  True,  True,  _O.PRO_REFUSAL,    _FE_PEF,
                                             "the supervising compliance officer"),
    "enterprise:HS:STOP":   ContinuationRow(_P.P_STOP_FORENSIC,        True,  False, _O.FORENSIC_STOP,  _FE_PEF,
                                             "the supervising compliance officer"),
}


# ── Invariant validation ───────────────────────────────────────────────────────

_CLOSED_STATUSES = {"ASK", "REFUSE", "STOP"}

_SUBSTANTIVE_PATHWAYS = frozenset({
    ContinuationPathway.P_ADMIT_STANDARD,
})


def _validate(table: dict[str, ContinuationRow]) -> None:
    """Raise ValueError if any row violates the epistemic non-widening invariant.

    Rule: every row whose status component is ASK, REFUSE, or STOP must have
    commitment_closed=True.  This mirrors GovernorPolicy.__post_init__ but is
    checked once at module import rather than at resolution time, providing
    early detection of table errors.
    """
    for key, row in table.items():
        parts = key.split(":")
        if len(parts) < 3:
            raise ValueError(
                f"ContinuationMatrix: malformed key {key!r} — "
                "expected '<domain>:<authority>:<status>[:<disc>]'"
            )
        status_str = parts[2]
        if status_str in _CLOSED_STATUSES and not row.commitment_closed:
            raise ValueError(
                f"ContinuationMatrix epistemic widening violation: "
                f"key={key!r} has status={status_str!r} "
                f"but commitment_closed=False. "
                "Governor may not recover a blocked determination."
            )

    if "general:GP:STOP" not in table:
        raise ValueError(
            "ContinuationMatrix: safe fallback entry 'general:GP:STOP' is missing."
        )


_validate(_TABLE)


# ── ContinuationMatrix ────────────────────────────────────────────────────────


class ContinuationMatrix:
    """Deterministic lookup table for lawful Governor continuation parameters.

    Public interface
    ----------------
    lookup(domain, authority, status, discriminator=None) -> ContinuationRow

    The lookup applies the four-level fallback ladder documented at the top of
    this module.  All lookups are pure dict operations — no regex, no heuristic.

    The module-level singleton CONTINUATION_MATRIX is the instance used by
    PolicyResolver.  Operator deployments may construct an alternative instance
    with a custom table for testing or sandboxed policy overrides.
    """

    def __init__(self, table: dict[str, ContinuationRow]) -> None:
        _validate(table)
        self._table = dict(table)  # defensive copy

    def lookup(
        self,
        domain: "Domain",          # aurora_lens.governor.models.Domain
        authority: "AuthorityClass",  # aurora_lens.governor.models.AuthorityClass
        status: LensStatus,
        discriminator: Optional[str] = None,
    ) -> ContinuationRow:
        """Return the canonical continuation row for (domain, authority, status).

        Fallback ladder (explicit, no heuristic):
          1. domain:authority:status:discriminator  — flag-class or user-class key
          2. domain:authority:status               — authority key
          3. domain:GP:status                      — GP authority fallback
          4. general:GP:status                     — global status fallback
          5. general:GP:STOP                       — safe fallback (always present)

        Args:
            domain:        Governance domain (e.g. Domain.MEDICAL).
            authority:     Authority class (e.g. AuthorityClass.GP).
            status:        Admissibility verdict from Lens (e.g. LensStatus.STOP).
            discriminator: Optional flag-type name or user-class value for
                           flag-class or user-class specific resolution.
                           None → 3-component lookup only.
        """
        d = domain.value
        a = authority.value
        s = status.value

        # Level 1: flag-class / user-class discriminator
        if discriminator is not None:
            key = f"{d}:{a}:{s}:{discriminator}"
            if key in self._table:
                return self._table[key]

        # Level 2: authority key
        key = f"{d}:{a}:{s}"
        if key in self._table:
            return self._table[key]

        # Level 3: GP authority fallback
        key = f"{d}:GP:{s}"
        if key in self._table:
            return self._table[key]

        # Level 4: global status fallback
        key = f"general:GP:{s}"
        if key in self._table:
            return self._table[key]

        # Level 5: safe fallback — always present (validated at construction)
        return self._table["general:GP:STOP"]

    def rows(self) -> dict[str, ContinuationRow]:
        """Return a copy of the full table (for inspection and testing)."""
        return dict(self._table)


# ── Module-level singleton ────────────────────────────────────────────────────

# Type-annotation imports are deferred to avoid a circular dependency between
# aurora_lens.governor.continuation_matrix and aurora_lens.governor.models
# (models is already imported above for the enum types; Domain/AuthorityClass
# are only needed in the lookup() signature at runtime).
from .models import Domain, AuthorityClass  # noqa: E402 — after _TABLE

CONTINUATION_MATRIX: ContinuationMatrix = ContinuationMatrix(_TABLE)

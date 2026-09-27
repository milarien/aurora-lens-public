"""Pure turn-semantic predicates shared by Lens orchestration.

This module contains semantic eligibility checks only. It must not perform
governance decisions, bridge logging, adapter calls, or PEF mutation.
"""

from __future__ import annotations

from dataclasses import dataclass

from aurora_lens.govern.decision import InterventionAction
from aurora_lens.interpret.revision_gate import explicit_correction_intent
from aurora_lens.interpret.schema import ExtractionResult
from aurora_lens.interpret.turn_act import TurnAct


def looks_like_explicit_correction_phrase(user_text: str) -> bool:
    return explicit_correction_intent(user_text)


def looks_like_record_fact_intent(user_text: str) -> bool:
    t = (user_text or "").strip().lower()
    return (
        t.startswith("record this exact fact in session state:")
        or t.startswith("record this fact:")
        or t.startswith("remember this fact:")
    )


_QUERY_LEADS: tuple[str, ...] = (
    "what ",
    "who ",
    "whom ",
    "whose ",
    "when ",
    "where ",
    "why ",
    "how ",
    "which ",
    "is ",
    "are ",
    "was ",
    "were ",
    "do ",
    "does ",
    "did ",
    "can ",
    "could ",
    "would ",
    "should ",
    "may ",
    "might ",
    "must ",
    "shall ",
    "will ",
)


_REQUEST_INSTRUCTION_LEADS: tuple[str, ...] = (
    "please ",
    "tell me ",
    "explain ",
    "analyze ",
    "analyse ",
    "recommend ",
    "advise ",
    "reallocate ",
    "invest ",
    "buy ",
    "sell ",
    "calculate ",
    "give me ",
)


def looks_like_question_text(user_text: str) -> bool:
    t = (user_text or "").strip().lower()
    if not t:
        return False
    if t.endswith("?"):
        return True
    return any(t.startswith(lead) for lead in _QUERY_LEADS)


def looks_like_request_instruction_text(user_text: str) -> bool:
    t = (user_text or "").strip().lower()
    if not t:
        return False
    return any(t.startswith(lead) for lead in _REQUEST_INSTRUCTION_LEADS)


_SENSITIVE_ACK_SUPPRESSION_TOKENS: frozenset[str] = frozenset(
    {
        "patient",
        "diagnose",
        "diagnosis",
        "myocardial",
        "infarction",
        "ecg",
        "troponin",
        "dose",
        "dosage",
        "allergy",
        "allergic",
        "prescribe",
        "treatment",
        "medication",
        "catheterization",
        "legal",
        "lawsuit",
        "contract",
        "investment",
        "portfolio",
        "financial",
        "retirement",
        "allocate",
        "allocation",
        "reallocate",
        "rebalance",
        "execute",
        "buy",
        "sell",
    }
)


def _tokenize_lower_words(text: str) -> set[str]:
    out: set[str] = set()
    cur: list[str] = []
    for ch in (text or "").lower():
        if ch.isalpha():
            cur.append(ch)
            continue
        if cur:
            out.add("".join(cur))
            cur = []
    if cur:
        out.add("".join(cur))
    return out


def _looks_sensitive_for_admitted_ack(
    user_text: str,
    extraction: ExtractionResult | None,
) -> bool:
    tokens = set(_tokenize_lower_words(user_text))
    if extraction is not None:
        for c in extraction.claims:
            tokens.update(_tokenize_lower_words(c.subject))
            tokens.update(_tokenize_lower_words(c.obj))
            tokens.update(_tokenize_lower_words(c.evidence))
    return bool(tokens & _SENSITIVE_ACK_SUPPRESSION_TOKENS)


def is_admitted_user_state_assertion(
    *,
    user_text: str,
    turn_act: TurnAct,
    extraction: ExtractionResult | None,
    committed_state_mutation: bool,
    has_pending_clarification: bool,
) -> bool:
    if not committed_state_mutation:
        return False
    if turn_act != TurnAct.ASSERT:
        return False
    if looks_like_question_text(user_text):
        return False
    if looks_like_request_instruction_text(user_text):
        return False
    if _looks_sensitive_for_admitted_ack(user_text, extraction):
        return False
    if has_pending_clarification:
        return False
    if extraction is None:
        return False
    if extraction.extraction_error:
        return False
    if extraction.ambiguous_referents or extraction.comparative_ambiguities:
        return False
    return bool(extraction.claims)


@dataclass(frozen=True)
class TurnSemanticPlan:
    kind: str
    response_text: str
    mutation_kind: str | None = None
    rationale: str | None = None
    action_hint: str | None = None


def plan_mutation_ack(
    *,
    user_text: str,
    turn_act: TurnAct,
    extraction: ExtractionResult | None,
    committed_state_mutation: bool,
    binding_resumed: bool,
    pending_failed_constraint: str | None,
    pending_original_question: str | None,
    pending_has_referent_metadata: bool,
    has_pending_clarification: bool,
    explicit_possessive_clarification: bool = False,
) -> TurnSemanticPlan | None:
    _pending_fc = (pending_failed_constraint or "").strip().upper()
    _legacy_pending_shape = (
        not _pending_fc
        and not pending_has_referent_metadata
    )
    _statement_referent_resume = (
        _pending_fc == "UNRESOLVED_REFERENT"
        and pending_original_question is not None
        and not pending_original_question.strip().endswith("?")
    )
    _explicit_possessive_referent_bind = (
        explicit_possessive_clarification
        and _pending_fc == "UNRESOLVED_REFERENT"
        and pending_original_question is not None
    )
    if binding_resumed and pending_original_question is not None and (
        _legacy_pending_shape
        or _statement_referent_resume
        or _explicit_possessive_referent_bind
    ):
        return TurnSemanticPlan(
            kind="mutation_ack",
            response_text="Clarification noted. I have updated the recorded state.",
            mutation_kind="clarification_bind",
            rationale="Pre-LLM mutation acknowledgement: successful clarification binding.",
            action_hint=InterventionAction.PASS.name,
        )

    if not committed_state_mutation:
        return None

    if looks_like_record_fact_intent(user_text):
        return TurnSemanticPlan(
            kind="mutation_ack",
            response_text="Recorded in session state.",
            mutation_kind="record_fact",
            rationale="Pre-LLM mutation acknowledgement: explicit record-fact intent committed.",
            action_hint=InterventionAction.PASS.name,
        )

    if looks_like_explicit_correction_phrase(user_text):
        return TurnSemanticPlan(
            kind="mutation_ack",
            response_text="Correction noted. I have updated the recorded state.",
            mutation_kind="correction",
            rationale="Pre-LLM mutation acknowledgement: explicit correction intent committed.",
            action_hint=InterventionAction.PASS.name,
        )

    if is_admitted_user_state_assertion(
        user_text=user_text,
        turn_act=turn_act,
        extraction=extraction,
        committed_state_mutation=committed_state_mutation,
        has_pending_clarification=has_pending_clarification,
    ):
        return TurnSemanticPlan(
            kind="mutation_ack",
            response_text="Recorded in session state.",
            mutation_kind="admitted_state_assertion",
            rationale="Pre-LLM mutation acknowledgement: admitted user state assertion committed.",
            action_hint=InterventionAction.PASS.name,
        )

    return None


def _extract_percentage_numbers(text: str) -> list[float]:
    s = text or ""
    out: list[float] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if not ch.isdigit():
            i += 1
            continue
        j = i
        seen_dot = False
        while j < n:
            c = s[j]
            if c.isdigit():
                j += 1
                continue
            if c == "." and not seen_dot:
                seen_dot = True
                j += 1
                continue
            break
        k = j
        while k < n and s[k].isspace():
            k += 1
        if k < n and s[k] == "%":
            num = s[i:j]
            try:
                out.append(float(num))
            except ValueError:
                pass
        i = j + 1
    return out


def _format_pct_value(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value)}%"
    return f"{value}%"


def _split_sentences_by_terminal_punct(text: str) -> list[str]:
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


def _make_possessive(name: str) -> str:
    if name.endswith("s"):
        return f"{name}'"
    return f"{name}'s"


def _looks_like_simple_is_location_object(obj_text: str) -> bool:
    t = (obj_text or "").strip().lower()
    if not t:
        return False
    if t in {"overseas", "abroad", "away", "here", "there"}:
        return True
    return (
        t.startswith("in ")
        or t.startswith("at ")
        or t.startswith("on ")
    )


def plan_resolved_pending_simple_is_answer(
    *,
    pending_failed_constraint: str | None,
    pending_original_question: str | None,
    resolved_referent_phrase: str | None,
    pending_blocked_claims: list[dict[str, object]] | None,
) -> TurnSemanticPlan | None:
    """Deterministic answer for resolved unresolved-referent location follow-ups.

    Narrow seam: after candidate clarification, if the blocked proposition carries a
    single ``IS`` claim with a simple location-like object and the pending question
    is a ``where/who`` query, answer immediately from committed state instead of
    re-opening generation phrasing.
    """
    if pending_failed_constraint != "UNRESOLVED_REFERENT":
        return None

    q = (pending_original_question or "").strip().lower()
    if not q or not q.endswith("?"):
        return None
    q_sentences = [s.strip().lower() for s in _split_sentences_by_terminal_punct(q)]
    asks_where_or_who = any(
        s.startswith("where ") or s.startswith("who ")
        for s in q_sentences
    )
    if not asks_where_or_who:
        return None

    phrase = (resolved_referent_phrase or "").strip()
    if not phrase:
        return None

    claims = pending_blocked_claims or []
    if len(claims) != 1:
        return None
    claim = claims[0]
    relation = str(claim.get("relation") or "").strip().upper()
    if relation != "IS":
        return None
    obj = str(claim.get("obj") or "").strip()
    if not _looks_like_simple_is_location_object(obj):
        return None

    return TurnSemanticPlan(
        kind="deterministic_clarification_continuation",
        response_text=f"{phrase} is {obj.rstrip(' .')}.",
        rationale=(
            "Resolved pending referent and answered deterministic single-claim "
            "location state from blocked proposition."
        ),
        action_hint=InterventionAction.PASS.name,
    )


def plan_resolved_pending_margin_answer(
    *,
    pending_original_question: str | None,
    resolved_binding_name: str | None,
    clarification_extraction: ExtractionResult | None,
    clarification_user_text: str,
) -> TurnSemanticPlan | None:
    q = (pending_original_question or "").strip().lower()
    if not q or not q.endswith("?"):
        return None
    if "margin" not in q:
        return None
    if "prior quarter" not in q:
        return None
    if "worse" not in q and "lower" not in q and "down" not in q:
        return None

    bound = (resolved_binding_name or "").strip()
    bound_lower = bound.lower()
    q4_value: float | None = None
    q3_value: float | None = None

    if clarification_extraction is not None and clarification_extraction.claims:
        for claim in clarification_extraction.claims:
            bits = [claim.subject or "", claim.obj or "", claim.evidence or ""]
            text = " ".join(x for x in bits if x).lower()
            if not text:
                continue
            if bound_lower and bound_lower not in text:
                continue
            quarter = "q4" if "q4" in text else "q3" if "q3" in text else None
            if quarter is None:
                continue
            nums = _extract_percentage_numbers(text)
            if not nums:
                continue
            if quarter == "q4" and q4_value is None:
                q4_value = nums[0]
            elif quarter == "q3" and q3_value is None:
                q3_value = nums[0]

    if q4_value is None or q3_value is None:
        for sentence in _split_sentences_by_terminal_punct(clarification_user_text):
            t = sentence.lower()
            if bound_lower and bound_lower not in t:
                continue
            quarter = "q4" if "q4" in t else "q3" if "q3" in t else None
            if quarter is None:
                continue
            nums = _extract_percentage_numbers(t)
            if not nums:
                continue
            if quarter == "q4" and q4_value is None:
                q4_value = nums[0]
            elif quarter == "q3" and q3_value is None:
                q3_value = nums[0]

    if q4_value is None or q3_value is None:
        return None

    subject = bound if bound else "The referenced entity"
    if q4_value < q3_value:
        return TurnSemanticPlan(
            kind="deterministic_clarification_continuation",
            response_text=(
                f"{_make_possessive(subject)} margin was worse in Q4 than in Q3 "
                f"({_format_pct_value(q4_value)} vs {_format_pct_value(q3_value)})."
            ),
            rationale=(
                "Resolved pending referent and deterministic margin comparison "
                "from clarification facts."
            ),
            action_hint=InterventionAction.PASS.name,
        )
    if q4_value > q3_value:
        return TurnSemanticPlan(
            kind="deterministic_clarification_continuation",
            response_text=(
                f"{_make_possessive(subject)} margin was not worse in Q4 than in Q3; "
                f"it was higher ({_format_pct_value(q4_value)} vs {_format_pct_value(q3_value)})."
            ),
            rationale=(
                "Resolved pending referent and deterministic margin comparison "
                "from clarification facts."
            ),
            action_hint=InterventionAction.PASS.name,
        )
    return TurnSemanticPlan(
        kind="deterministic_clarification_continuation",
        response_text=(
            f"{_make_possessive(subject)} margin was the same in Q4 and Q3 "
            f"({_format_pct_value(q4_value)})."
        ),
        rationale=(
            "Resolved pending referent and deterministic margin comparison "
            "from clarification facts."
        ),
        action_hint=InterventionAction.PASS.name,
    )

"""Structured pending continuation tasks for unresolved-referent holds.

Preserves suspended task identity through clarification bind/resume without
phrase-table or regex matching on demo sentences.
"""

from __future__ import annotations

from typing import Any

from aurora_lens.pef.state import canonicalize_relation
from aurora_lens.state_native_engine.eval.action_realization import (
    format_action_realization,
    normalize_event_object_literal,
)

_PERSONAL_PRONOUNS: frozenset[str] = frozenset({
    "she", "her", "hers", "he", "him", "his", "they", "them", "their", "theirs",
})

_ATTRIBUTION_INTERROGATIVE_OPENERS: frozenset[str] = frozenset({
    "who", "which", "whose", "whom",
})

_TASK_ATTRIBUTION_ANSWER = "attribution_answer"
_ROLE_SUBJECT = "subject"

_KIND_ACTOR = "actor"
_KIND_QUERY_RECIPIENT = "query_recipient"
_KIND_CONTACT_TARGET = "contact_target"

_INANIMATE_OBJECT_HEADS: frozenset[str] = frozenset({
    "module", "assignment", "submission", "submissions", "document", "report",
    "invoice", "statement", "clause", "nda", "item", "file", "record", "case",
})

_VERB_TOKEN_TO_RELATION: dict[str, str] = {
    "approve": "APPROVE",
    "approved": "APPROVE",
    "agree": "IS",
    "agreed": "IS",
    "flag": "FLAG",
    "flagged": "FLAG",
    "send": "SEND",
    "sent": "SEND",
    "escalate": "ESCALATE",
    "escalated": "ESCALATE",
    "order": "ORDER",
    "ordered": "ORDER",
    "raise": "FLAG",
    "raised": "FLAG",
    "review": "REVIEW",
    "reviewed": "REVIEW",
    "handle": "HANDLE",
    "handled": "HANDLE",
}


def _alpha_tokens(text: str) -> list[str]:
    out: list[str] = []
    for raw in (text or "").replace("\u2019", "'").split():
        tok = "".join(ch for ch in raw.strip(".,!?;:\"'()[]") if ch.isalpha())
        if tok:
            out.append(tok.lower())
    return out


def _final_interrogative_sentence(original_question: str) -> str | None:
    parts: list[str] = []
    buf = (original_question or "").strip()
    if not buf:
        return None
    for piece in buf.replace("!", ".").replace("?", "?\n").split("\n"):
        for sent in piece.split("."):
            s = sent.strip()
            if s:
                parts.append(s)
    for sent in reversed(parts):
        if sent.endswith("?"):
            return sent
        low = sent.lower()
        if low.startswith(tuple(_ATTRIBUTION_INTERROGATIVE_OPENERS)):
            return sent + "?"
    return None


def _claim_dict_subject_is_ambiguous(
    claim: dict[str, Any],
    ambiguous_referents: list[str],
) -> bool:
    subj = str(claim.get("subject") or "").strip().lower()
    if not subj:
        return False
    amb = {str(x).strip().lower() for x in ambiguous_referents if str(x).strip()}
    if subj in amb:
        return True
    subj_tokens = _alpha_tokens(subj)
    return bool(subj_tokens) and subj_tokens[0] in amb


def _select_target_claim(
    blocked_claims: list[dict[str, Any]],
    ambiguous_referents: list[str],
) -> dict[str, Any] | None:
    for claim in blocked_claims or []:
        if _claim_dict_subject_is_ambiguous(claim, ambiguous_referents):
            return claim
    if blocked_claims:
        return blocked_claims[0]
    return None


def _interrogative_opener_tokens(surface_question: str) -> list[str]:
    s = (surface_question or "").strip().rstrip("?.!")
    return _alpha_tokens(s)


def _classify_attribution_kind(surface_question: str) -> str:
    toks = _interrogative_opener_tokens(surface_question)
    if not toks:
        return _KIND_ACTOR
    if toks[0] == "whose" and "query" in toks:
        return _KIND_QUERY_RECIPIENT
    if toks[0] in {"which", "who"} and "contact" in toks:
        return _KIND_CONTACT_TARGET
    return _KIND_ACTOR


def _object_after_verb(proposition: str, verb_token: str) -> str:
    prop = (proposition or "").strip()
    verb_l = verb_token.lower()
    prop_l = prop.lower()
    idx = prop_l.find(verb_l)
    if idx < 0:
        return ""
    end = idx + len(verb_l)
    while end < len(prop) and prop[end].isspace():
        end += 1
    return prop[end:].strip(" .")


def synthesize_held_claim_from_blocked_proposition(
    blocked_proposition: str,
    ambiguous_referents: list[str],
) -> dict[str, Any] | None:
    """Build a minimal held claim when extraction omitted the ambiguous-subject claim."""
    prop = (blocked_proposition or "").strip()
    if not prop:
        return None
    tokens = _alpha_tokens(prop)
    if len(tokens) < 2:
        return None
    amb = {str(x).strip().lower() for x in ambiguous_referents if str(x).strip()}
    if tokens[0] not in amb:
        return None
    verb = tokens[1]
    relation = _VERB_TOKEN_TO_RELATION.get(verb, canonicalize_relation(verb.upper()))
    obj = _object_after_verb(prop, verb)
    if relation == "IS" and verb in {"agree", "agreed"}:
        rest = obj
        if rest.lower().startswith("to "):
            obj = f"agreed {rest}"
        else:
            obj = f"agreed to {rest or 'mediation'}"
    if not obj:
        obj = " ".join(tokens[2:]) or prop
    return {
        "subject": tokens[0],
        "relation": relation,
        "obj": obj,
        "span": "present",
        "negated": False,
        "evidence": prop,
    }


def _surface_question_targets_held_claim_subject(
    surface_question: str,
    target_claim: dict[str, Any],
) -> bool:
    """True when the pending interrogative asks for the held claim's subject/agent."""
    toks = _interrogative_opener_tokens(surface_question)
    if not toks or toks[0] not in _ATTRIBUTION_INTERROGATIVE_OPENERS:
        return False
    if toks[0] == "whose" and "query" in toks:
        return True
    if toks[0] in {"which", "who"} and "contact" in toks:
        return True
    surf_toks = set(_interrogative_opener_tokens(surface_question))
    obj = normalize_event_object_literal(str(target_claim.get("obj") or "")).lower()
    relation = canonicalize_relation(str(target_claim.get("relation") or ""))
    obj_words = [w for w in _alpha_tokens(obj) if len(w) > 3]
    if any(w in surf_toks for w in obj_words):
        return True
    surf_joined = " ".join(surf_toks)
    if relation == "APPROVE" and "approv" in surf_joined:
        return True
    if relation == "FLAG" and ("concern" in surf_joined or "raised" in surf_joined):
        return True
    if relation == "ESCALATE" and "escalat" in surf_joined:
        return True
    if relation == "IS" and obj.startswith("agreed") and (
        "agreed" in surf_joined or "party" in surf_joined
    ):
        return True
    return False


def infer_pending_attribution_task(
    *,
    original_question: str,
    blocked_claims: list[dict[str, Any]] | None,
    ambiguous_referents: list[str] | None,
    blocked_proposition: str | None = None,
) -> dict[str, Any] | None:
    """Build structured pending_task when the held frame is actor/source attribution."""
    surface = _final_interrogative_sentence(original_question)
    if surface is None:
        return None
    opener = _interrogative_opener_tokens(surface)
    if not opener or opener[0] not in _ATTRIBUTION_INTERROGATIVE_OPENERS:
        return None
    claims = list(blocked_claims or [])
    target = _select_target_claim(claims, list(ambiguous_referents or []))
    if target is None and blocked_proposition:
        target = synthesize_held_claim_from_blocked_proposition(
            blocked_proposition,
            list(ambiguous_referents or []),
        )
    if target is None:
        return None
    if not _surface_question_targets_held_claim_subject(surface, target):
        return None
    kind = _classify_attribution_kind(surface)
    return {
        "task_type": _TASK_ATTRIBUTION_ANSWER,
        "expected_answer_role": _ROLE_SUBJECT,
        "attribution_kind": kind,
        "target_claim": dict(target),
        "surface_question": surface.strip(),
    }


def pending_task_from_pending_dict(pending: dict[str, Any] | None) -> dict[str, Any] | None:
    if not pending:
        return None
    task = pending.get("pending_task")
    if isinstance(task, dict) and task.get("task_type"):
        return task
    return infer_pending_attribution_task(
        original_question=str(pending.get("original_question") or ""),
        blocked_claims=list(pending.get("blocked_claims") or []),
        ambiguous_referents=list(pending.get("ambiguous_referents") or []),
        blocked_proposition=str(pending.get("blocked_proposition") or "") or None,
    )


def _object_head_tokens(obj: str) -> list[str]:
    return _alpha_tokens(normalize_event_object_literal(str(obj or "")))


def _name_is_only_inanimate_object_head(
    name: str,
    blocked_claims: list[dict[str, Any]] | None,
) -> bool:
    key = (name or "").strip().lower()
    if not key:
        return True
    if key not in _INANIMATE_OBJECT_HEADS:
        return False
    for claim in blocked_claims or []:
        obj_heads = _object_head_tokens(str(claim.get("obj") or ""))
        if obj_heads and obj_heads[0] == key:
            return True
    return False


def _ambiguous_tokens_require_personal_referent(ambiguous_tokens: list[str] | None) -> bool:
    for tok in ambiguous_tokens or []:
        if str(tok).strip().lower() in _PERSONAL_PRONOUNS:
            return True
    return False


def person_antecedent_names_from_surfaces(surfaces: list[str]) -> frozenset[str]:
    """Person-like antecedent surfaces excluding inanimate content-unit heads."""
    return frozenset(
        str(name).strip().lower()
        for name in surfaces
        if str(name).strip() and str(name).strip().lower() not in _INANIMATE_OBJECT_HEADS
    )


def referent_candidate_compatible_with_ambiguity(
    candidate_name: str,
    *,
    ambiguous_tokens: list[str] | None,
    blocked_claims: list[dict[str, Any]] | None,
    person_antecedent_names: frozenset[str] | None = None,
) -> bool:
    """Structural pronoun/entity compatibility — not string deny lists."""
    if not _ambiguous_tokens_require_personal_referent(ambiguous_tokens):
        return True
    key = (candidate_name or "").strip().lower()
    if not key:
        return False
    if person_antecedent_names and key in person_antecedent_names:
        return True
    if key in _INANIMATE_OBJECT_HEADS:
        return False
    if _name_is_only_inanimate_object_head(candidate_name, blocked_claims):
        return False
    return True


def filter_referent_resolution_candidates(
    candidates: list[str],
    *,
    ambiguous_tokens: list[str] | None,
    blocked_claims: list[dict[str, Any]] | None,
    person_antecedent_names: frozenset[str] | None = None,
) -> list[str]:
    return [
        name for name in candidates
        if referent_candidate_compatible_with_ambiguity(
            name,
            ambiguous_tokens=ambiguous_tokens,
            blocked_claims=blocked_claims,
            person_antecedent_names=person_antecedent_names,
        )
    ]


def compose_attribution_answer(
    entity_name: str,
    pending_task: dict[str, Any],
) -> str | None:
    """Deterministic answer from bound entity + structured pending_task."""
    name = (entity_name or "").strip()
    if not name:
        return None
    kind = str(pending_task.get("attribution_kind") or _KIND_ACTOR)
    if kind == _KIND_QUERY_RECIPIENT:
        return f"Address {name}'s query."
    if kind == _KIND_CONTACT_TARGET:
        return f"Students should contact {name} about their results."
    target = pending_task.get("target_claim") or {}
    relation = canonicalize_relation(str(target.get("relation") or ""))
    obj = normalize_event_object_literal(str(target.get("obj") or ""))
    obj_low = obj.lower()
    if relation == "APPROVE" or "approv" in obj_low:
        return f"{name} gave the approval."
    if relation == "FLAG" or "concern" in obj_low:
        return f"{name} raised the concern."
    if relation == "IS" and obj:
        if obj_low.startswith("agreed"):
            return f"{name} agreed to mediation." if "mediation" in obj_low else f"{name} {obj}."
        return f"{name} is {obj.rstrip('.')}."
    clause = format_action_realization(relation, obj)
    if not clause:
        return None
    return f"{name} {clause}."


def plan_attribution_resume_answer(
    *,
    pending: dict[str, Any] | None,
    selected_entity_name: str,
) -> str | None:
    task = pending_task_from_pending_dict(pending)
    if task is None or task.get("task_type") != _TASK_ATTRIBUTION_ANSWER:
        return None
    return compose_attribution_answer(selected_entity_name, task)


_INCOMPLETE_ATTRIBUTION_RESPONSE_MARKERS: frozenset[str] = frozenset({
    "that's correct",
    "that is correct",
    "correct.",
    "yes.",
    "i am unable to help",
    "please consult a lawyer",
    "you need to resolve",
    "have you reviewed the email",
    "i'm not aware",
    "i am not aware",
})


def binding_resume_response_satisfies_attribution_task(
    response_text: str,
    *,
    pending: dict[str, Any] | None,
    selected_entity_name: str,
) -> bool:
    """Return False when LLM output is a non-answer for a pending attribution task."""
    expected = plan_attribution_resume_answer(
        pending=pending,
        selected_entity_name=selected_entity_name,
    )
    if not expected:
        return True
    low = (response_text or "").strip().lower()
    if not low:
        return False
    if selected_entity_name.strip().lower() not in low:
        return False
    for marker in _INCOMPLETE_ATTRIBUTION_RESPONSE_MARKERS:
        if marker in low:
            return False
    exp_low = expected.strip().lower()
    if exp_low in low or low in exp_low:
        return True
    return len(low.split()) <= 12 and selected_entity_name.strip().lower() in low


COMPLETION_STRATEGY_UPSTREAM_ORIGINAL = "upstream_original"
COMPLETION_STRATEGY_DETERMINISTIC_ATTRIBUTION = "deterministic_attribution"


def completion_strategy_for_unresolved_referent_hold(
    *,
    original_question: str,
    pending_task: dict[str, Any] | None,
) -> str:
    """Classify how a UNRESOLVED_REFERENT hold completes after user binding.

    Ordinary unresolved holds (including multi-sentence framing + attribution ask)
    resume the held ``original_question`` through upstream after binding.
    Deterministic local commit applies only when the user's entire turn is the
    attribution interrogative (no separate held request body).
    """
    if pending_task is None:
        return COMPLETION_STRATEGY_UPSTREAM_ORIGINAL
    surface = str(pending_task.get("surface_question") or "").strip()
    if surface and original_question.strip() == surface:
        return COMPLETION_STRATEGY_DETERMINISTIC_ATTRIBUTION
    return COMPLETION_STRATEGY_UPSTREAM_ORIGINAL


def pending_completion_strategy(pending: dict[str, Any] | None) -> str:
    """Read explicit hold strategy or infer for legacy/injected pending payloads."""
    if not pending:
        return COMPLETION_STRATEGY_UPSTREAM_ORIGINAL
    explicit = pending.get("completion_strategy")
    if explicit in (
        COMPLETION_STRATEGY_UPSTREAM_ORIGINAL,
        COMPLETION_STRATEGY_DETERMINISTIC_ATTRIBUTION,
    ):
        return str(explicit)
    return completion_strategy_for_unresolved_referent_hold(
        original_question=str(pending.get("original_question") or ""),
        pending_task=pending_task_from_pending_dict(pending),
    )


def binding_resume_requires_upstream(pending_snapshot: dict[str, Any] | None) -> bool:
    """True when bind must re-run the held original input through upstream."""
    return (
        pending_completion_strategy(pending_snapshot)
        == COMPLETION_STRATEGY_UPSTREAM_ORIGINAL
    )

"""Reason-specific governed user copy for blocked and clarification pathways.

Builds terminal / CONTAIN responses from structured governance fields only —
no invented facts, no guessed referents, no ambiguity collapse.
"""

from __future__ import annotations

import re

from aurora_lens.verify.flags import FlagType

_POSSESSIVE_PRONOUNS: frozenset[str] = frozenset({"her", "his", "their", "its"})

_ATTRIBUTION_REFER_QUERY = re.compile(
    r"\bwho\b.*\brefer(?:s|red)?\s+to\b",
    re.IGNORECASE,
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT.split(raw) if s.strip()]


def _final_interrogative_sentence(text: str) -> str | None:
    sents = _split_sentences(text)
    for s in reversed(sents):
        if "?" in s:
            return s.strip()
    stripped = (text or "").strip()
    return stripped if stripped.endswith("?") else None


def is_attribution_referent_query(original_question: str | None) -> bool:
    """True when the user asks who/what a pronoun or referent denotes."""
    surface = _final_interrogative_sentence(original_question or "")
    if not surface:
        return False
    return _ATTRIBUTION_REFER_QUERY.search(surface) is not None


def _limitation_context_sentences(original_question: str) -> list[str]:
    """Sentences the user supplied that bound available evidence (quoted verbatim)."""
    out: list[str] = []
    for sentence in _split_sentences(original_question):
        sl = sentence.lower()
        if "no further evidence" in sl or "both parties" in sl:
            out.append(sentence.rstrip(".?!"))
    return out


def _possessive_np_in_text(ambiguous_token: str, text: str) -> str | None:
    tok = (ambiguous_token or "").strip().lower()
    if not tok or not text:
        return None
    match = re.search(
        rf"\b{re.escape(tok)}\s+([A-Za-z][\w-]*)",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    return f"{tok} {match.group(1).lower()}"


def _object_head_from_possessive_np(possessive_np: str) -> str:
    parts = possessive_np.split()
    return parts[-1] if parts else "item"


def _role_label(name: str) -> str:
    return (name or "").strip().lower()


def _order_candidates_by_discourse(
    candidates: list[str],
    original_question: str | None,
) -> list[str]:
    if not original_question:
        return candidates
    ql = original_question.lower()

    def _key(name: str) -> tuple[int, str]:
        pos = ql.find(_role_label(name))
        return (pos if pos >= 0 else 10_000, _role_label(name))

    return sorted(candidates, key=_key)


def _qualifying_object_phrase(head: str, scope: str) -> str:
    """Natural object phrase from proposition context (e.g. expired certification)."""
    hl = head.lower()
    sl = (scope or "").lower()
    if hl == "certification" and "expired" in sl:
        return "expired certification"
    return head


def _compose_evidence_gap_statement(
    *,
    candidate_entities: list[str],
    blocked_proposition: str | None = None,
    original_question: str | None = None,
    ambiguous_token: str | None = None,
) -> str | None:
    """Regulator-style evidence gap when two or more candidates are in play."""
    candidates = [str(c).strip() for c in candidate_entities if str(c).strip()]
    if len(candidates) < 2:
        return None

    ordered = _order_candidates_by_discourse(
        candidates,
        original_question or blocked_proposition,
    )
    first, second = _role_label(ordered[0]), _role_label(ordered[1])
    scope = (blocked_proposition or original_question or "").strip()

    if scope and "certification" in scope.lower() and "expired" in scope.lower():
        qualified = _qualifying_object_phrase("certification", scope)
        return (
            f"The available evidence does not establish whether the {qualified} "
            f"belongs to the {first} or the {second}."
        )

    return (
        f"The available evidence does not establish which of the listed parties "
        f"applies ({first} or {second})."
    )


def _compose_dependent_consequence_inadmissibility(
    *,
    act_kind: str,
    user_turn: str | None = None,
    candidate_entities: list[str] | None = None,
) -> str:
    """Regulator-style statement for blocked dependent consequence turns."""
    ut = (user_turn or "").lower()
    tail = "because the consequence depends on an unresolved attribution."

    if "responsible" in ut or act_kind == "attribution_seeking":
        return f"The responsible party cannot be determined {tail}"

    if any(w in ut for w in ("suspend", "suspended", "suspension")) or act_kind in {
        "decision_seeking",
        "mutation_seeking",
    }:
        for name in candidate_entities or []:
            label = _role_label(name)
            if label and re.search(rf"\b{re.escape(label)}\b", ut):
                if any(w in ut for w in ("suspend", "suspended", "suspension")):
                    return (
                        f"Suspension of the {label} cannot be determined {tail}"
                    )
        return f"Suspension cannot be determined {tail}"

    if act_kind == "recommendation_seeking":
        return f"This recommendation cannot be determined {tail}"

    act_label = act_kind.replace("_", " ") if act_kind else "determination"
    return f"This {act_label} cannot be made {tail}"


def compose_unresolved_possessive_attribution_block(
    *,
    ambiguous_token: str,
    candidate_entities: list[str],
    original_question: str,
    blocked_proposition: str | None = None,
) -> str | None:
    """Blocked attribution copy when a possessive referent cannot be grounded."""
    candidates = [str(c).strip() for c in candidate_entities if str(c).strip()]
    tok = (ambiguous_token or "").strip().lower()
    if tok not in _POSSESSIVE_PRONOUNS or len(candidates) < 2:
        return None
    if not is_attribution_referent_query(original_question):
        return None

    scope = (blocked_proposition or original_question or "").strip()
    possessive_np = _possessive_np_in_text(tok, scope)
    if not possessive_np:
        return None

    ordered = _order_candidates_by_discourse(candidates, original_question)
    scope = (blocked_proposition or original_question or "").strip()
    head = _object_head_from_possessive_np(possessive_np)

    missing = _compose_evidence_gap_statement(
        candidate_entities=ordered,
        blocked_proposition=blocked_proposition,
        original_question=original_question,
        ambiguous_token=tok,
    )
    if not missing:
        first, second = _role_label(ordered[0]), _role_label(ordered[1])
        qualified = _qualifying_object_phrase(head, scope)
        missing = (
            f"The available evidence does not establish whether the {qualified} "
            f"belongs to the {first} or the {second}."
        )

    limitation_parts = _limitation_context_sentences(original_question)
    inadmissible = (
        "The requested attribution cannot be released because the available evidence "
        "does not establish which party's certification expired."
        if "certification" in scope.lower()
        else (
            "The requested attribution cannot be released because the available evidence "
            "does not establish which party applies."
        )
    )
    if limitation_parts:
        reason = ". ".join(limitation_parts) + ". " + inadmissible
    else:
        reason = inadmissible

    continuation = (
        "provide clarifying evidence identifying whose certification expired."
        if head == "certification"
        else f"provide clarifying evidence identifying whose {head} expired."
    )

    return "\n\n".join([
        "Decision blocked.",
        "",
        missing,
        "",
        reason,
        "",
        f"Allowed continuation: {continuation}",
    ])


def compose_unresolved_referent_governed_message(
    *,
    ambiguous_tokens: list[str],
    candidate_entities: list[str] | None = None,
    original_question: str | None = None,
    blocked_proposition: str | None = None,
    flag_claim: str | None = None,
    flag_evidence: str | None = None,
) -> str:
    """Governed copy for UNRESOLVED_REFERENT (pre-LLM or bridge ASK pathway)."""
    tokens = [str(t).strip() for t in ambiguous_tokens if str(t).strip()]
    primary = tokens[0].lower() if tokens else ""
    candidates = [str(c).strip() for c in (candidate_entities or []) if str(c).strip()]

    if original_question and primary:
        blocked = compose_unresolved_possessive_attribution_block(
            ambiguous_token=primary,
            candidate_entities=candidates,
            original_question=original_question,
            blocked_proposition=blocked_proposition,
        )
        if blocked:
            return blocked

    ref_list = ", ".join(f"'{p}'" for p in tokens)
    scope = (blocked_proposition or original_question or "").strip()
    if len(candidates) >= 2:
        if scope and "certification" in scope.lower() and "expired" in scope.lower():
            reason = _compose_evidence_gap_statement(
                candidate_entities=candidates,
                blocked_proposition=blocked_proposition,
                original_question=original_question,
                ambiguous_token=primary,
            ) or (
                "The available evidence does not establish which of the listed parties applies."
            )
        else:
            ordered = _order_candidates_by_discourse(
                candidates,
                original_question or blocked_proposition,
            )
            opt_text = " or ".join(_role_label(c) for c in ordered[:4])
            reason = (
                f"The available evidence does not establish whether {ref_list} applies to "
                f"{opt_text}."
            )
    elif flag_claim:
        reason = flag_claim.strip()
    else:
        reason = (
            f"The available evidence does not establish what {ref_list} denotes."
        )

    if flag_evidence and flag_evidence.strip():
        ev = flag_evidence.strip()
        if ev not in reason:
            sep = "" if reason.endswith((".", "!", "?")) else ". "
            reason = f"{reason}{sep}{ev}"

    options_block = ""
    if candidates:
        options_block = "\n".join(f"- {name}" for name in candidates)

    lines = [
        "Clarification required.",
        "",
        reason,
        "",
    ]
    if options_block:
        lines.extend([options_block, ""])
    lines.extend([
        "Status: Attribution unresolved.",
        "Action: Choose one option to continue.",
    ])
    return "\n".join(lines)


def compose_session_dependent_consequence_block(
    *,
    ambiguous_tokens: list[str],
    candidate_entities: list[str],
    blocked_proposition: str | None,
    act_kind: str,
    reason_detail: str | None = None,
    user_turn: str | None = None,
) -> str:
    """Governed copy when an open registry blocks a non-pronoun dependent consequence turn."""
    tokens = [str(t).strip() for t in ambiguous_tokens if str(t).strip()]
    candidates = [str(c).strip() for c in candidate_entities if str(c).strip()]

    missing = _compose_evidence_gap_statement(
        candidate_entities=candidates,
        blocked_proposition=blocked_proposition,
        ambiguous_token=tokens[0] if tokens else None,
    )
    if not missing:
        primary = tokens[0] if tokens else "the matter"
        missing = (
            "The available evidence does not establish which party applies, "
            f"and the requested act depends on that attribution ({primary!r})."
        )

    inadmissible = _compose_dependent_consequence_inadmissibility(
        act_kind=act_kind,
        user_turn=user_turn,
        candidate_entities=candidates,
    )
    if reason_detail and reason_detail.strip() not in inadmissible:
        inadmissible = f"{inadmissible} {reason_detail.strip()}"

    return "\n\n".join([
        "Decision blocked.",
        "",
        missing,
        "",
        inadmissible,
        "",
        "Allowed continuation: provide explicit resolution or clarifying evidence identifying "
        "which party applies before any dependent decision.",
    ])


def compose_session_meta_inquiry_block(
    *,
    ambiguous_tokens: list[str],
    candidate_entities: list[str] | None = None,
    blocked_proposition: str | None = None,
) -> str:
    """Governed copy for meta inquiries while the unresolved-referent registry is open."""
    base = compose_unresolved_referent_governed_message(
        ambiguous_tokens=ambiguous_tokens,
        candidate_entities=candidate_entities,
        blocked_proposition=blocked_proposition,
    )
    return (
        f"{base}\n\n"
        "Allowed continuation: provide explicit resolution or clarifying evidence "
        "identifying which party applies."
    )


def compose_hold_unresolved_acknowledgment(
    *,
    ambiguous_tokens: list[str],
    candidate_entities: list[str],
    blocked_proposition: str | None = None,
) -> str:
    """Governed copy when the operator selects Hold unresolved for UNRESOLVED_REFERENT."""
    tokens = [str(t).strip() for t in ambiguous_tokens if str(t).strip()]
    candidates = [str(c).strip() for c in candidate_entities if str(c).strip()]

    missing = _compose_evidence_gap_statement(
        candidate_entities=candidates,
        blocked_proposition=blocked_proposition,
        ambiguous_token=tokens[0] if tokens else None,
    )
    if not missing:
        primary = tokens[0] if tokens else "the matter"
        missing = (
            f"The available evidence does not establish a unique attribution for {primary!r}."
        )

    return "\n\n".join([
        "Ambiguity held.",
        "",
        missing,
        "Both interpretations will be preserved. Any later judgement or consequence that "
        "depends on resolving this attribution will remain inadmissible.",
        "",
        "Allowed continuation: provide clarifying evidence later, or ask questions that do "
        "not require selecting one party.",
    ])


def compose_hold_unresolved_already_held_acknowledgment() -> str:
    """Idempotent governed copy when Hold unresolved is selected again."""
    return "\n\n".join([
        "Ambiguity already held.",
        "",
        "The available evidence still does not establish which party applies. Both "
        "interpretations remain open, and any consequence that depends on resolving this "
        "attribution remains inadmissible.",
        "",
        "Allowed continuation: provide clarifying evidence, ask what is currently unresolved, "
        "or ask a non-dependent question.",
    ])


def compose_insufficient_structure_message(
    *,
    flag_type: FlagType | None,
    flag_claim: str | None = None,
    flag_evidence: str | None = None,
    missing_field: str | None = None,
) -> str:
    """Governed copy when verification cannot extract admissible structure from a draft."""
    if flag_type == FlagType.UPSTREAM_INSUFFICIENT_CONTEXT:
        reason = (
            flag_claim.strip()
            if flag_claim and flag_claim.strip()
            else (
                "The draft response declares insufficient context for the user query "
                "and does not supply a grounded answer."
            )
        )
        continuation = "provide the missing context or rephrase the question with explicit facts."
    elif flag_type == FlagType.EXTRACTION_EMPTY:
        reason = (
            "The draft response did not yield verifiable claims against the session record, "
            "so the requested determination cannot be released."
        )
        if flag_evidence and flag_evidence.strip():
            reason = f"{reason} {flag_evidence.strip()}"
        continuation = "rephrase with explicit facts or quote the evidence you want considered."
    else:
        reason = (
            flag_claim.strip()
            if flag_claim and flag_claim.strip()
            else "The available record does not support releasing this answer."
        )
        continuation = "provide clearer wording with explicit facts or questions."

    if missing_field and missing_field.strip().lower() not in {"extraction", "upstream"}:
        reason = f"{reason} Missing: {missing_field.strip()}."

    return "\n\n".join([
        "Decision blocked.",
        "",
        reason,
        "",
        f"Allowed continuation: {continuation}",
    ])

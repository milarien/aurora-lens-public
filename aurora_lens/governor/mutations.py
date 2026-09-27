"""ProposedMutation extraction -- Phase 1+2: state-transition verbs.

Phase 1: third-person subjects (the payment, your enrollment).
Phase 2: second-person subjects (you are enrolled) with negation/hedge guards.

Extracts structured claims from LLM output where the model asserts a status
change (e.g. "cleared", "approved", "enrolled"). Uses spaCy for sentence
splitting and targeted dependency parsing. NOT a general semantic parser.

Phase 1 scope: state-transition verbs only.
  sent -> cleared, received -> active, under review -> approved, etc.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aurora_lens.pef.state import PEFState

_log = logging.getLogger(__name__)

_STATE_TRANSITION_HEADS: frozenset[str] = frozenset({
    "cleared", "processed", "approved", "finalized", "completed",
    "settled", "granted", "confirmed", "enrolled", "active",
    "valid", "received", "initiated",
})

_AUX_LEMMAS: frozenset[str] = frozenset({
    "be", "have", "get",
})


@dataclass
class ProposedMutation:
    """A structured claim extracted from LLM output.

    Represents the model asserting that some subject has transitioned to
    a particular status (head). The structural governor checks whether
    PEF state supports this transition.
    """

    subject: str
    relation: str
    head: str
    span: str
    source_sentence: str
    verb_surface: str
    second_person: bool = False


_nlp_cache = None


def _get_nlp():
    global _nlp_cache
    if _nlp_cache is None:
        try:
            import spacy
        except ImportError as exc:  # pragma: no cover - environmental guard
            raise SystemExit(
                "aurora-lens requires spaCy for deterministic extraction. "
                "Install it with: pip install \"aurora-lens[spacy]\" && "
                "python -m spacy download en_core_web_sm"
            ) from exc
        try:
            _nlp_cache = spacy.load("en_core_web_sm")
        except OSError:
            _log.warning(
                "spaCy model en_core_web_sm not found; mutations extraction disabled"
            )
    return _nlp_cache


def _subject_from_dep_child(child, sent):
    """Resolve noun chunk text for an nsubj / nsubjpass token."""
    for chunk in sent.noun_chunks:
        if child.i >= chunk.start and child.i < chunk.end:
            return chunk.text
    return child.text


def _find_subject(token, sent):
    """Walk from token up to clause root, then find nsubj/nsubjpass."""
    # Clausal complements (`ccomp`): the status verb heads its own clause; the
    # grammatical subject is on this verb (not the outer predicate). Without this,
    # "It indicates that the payment was received" wrongly binds subject "It".
    if token.dep_ == "ccomp":
        for child in token.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                return _subject_from_dep_child(child, sent)

    clause_root = token
    while clause_root.dep_ != "ROOT" and clause_root.head != clause_root:
        if clause_root.dep_ in (
            "acomp", "attr", "oprd", "ccomp", "xcomp", "relcl", "advcl",
        ):
            clause_root = clause_root.head
            break
        clause_root = clause_root.head

    for child in clause_root.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            return _subject_from_dep_child(child, sent)

    for child in token.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            return _subject_from_dep_child(child, sent)

    return None


def _is_narrative_active_receive(token) -> bool:
    """True when *receive* is an active transitive event (X received Y).

    Passive or short-form workflow readings ("the payment was received",
    "payment received" without a direct object) remain state-transition candidates.

    Narrative HTTP/API prose ("the server received the request") must not become
    a procedural ``received`` mutation checked against PEF.
    """
    if token.lemma_.lower() != "receive":
        return False
    if token.pos_ != "VERB":
        return False
    return any(child.dep_ == "dobj" for child in token.children)


_SKIP_SUBJECTS: frozenset[str] = frozenset({
    "i", "we", "me", "my", "our",
})

_SECOND_PERSON: frozenset[str] = frozenset({"you", "your"})

_HEDGE_TOKENS: frozenset[str] = frozenset({
    "whether", "if", "assuming", "unless",
})

_UNCERTAIN_MODALS: frozenset[str] = frozenset({
    "may", "might", "could",
})


def _has_negation(token) -> bool:
    """True when the status token is syntactically negated."""
    for child in token.children:
        if child.dep_ == "neg":
            return True
    head = token.head
    if head is not token:
        for child in head.children:
            if child.dep_ == "neg":
                return True
    return False


def _is_hedged_or_modal(token, sent) -> bool:
    """True when the sentence hedges the assertion or uses an uncertain modal.

    Guards against extracting mutations from non-assertive contexts:
      - "whether you are enrolled" (conditional)
      - "if you are enrolled" (conditional)
      - "you may be enrolled" (uncertain modal)
    """
    sent_lower = sent.text.lower()
    if any(h in sent_lower for h in _HEDGE_TOKENS):
        return True
    for anc in (token, token.head):
        for child in anc.children:
            if child.dep_ == "aux" and child.lemma_.lower() in _UNCERTAIN_MODALS:
                return True
    return False

_STATUS_DEPS: frozenset[str] = frozenset({
    "ROOT", "relcl", "advcl", "ccomp", "xcomp",
    "acomp", "attr", "oprd", "amod",
})


def parse_proposed_mutations(
    response_text: str,
    pef: "PEFState",
) -> list[ProposedMutation]:
    """Extract state-transition mutations from LLM response text.

    Phase 1: Only detects sentences where the model asserts a status from
    _STATE_TRANSITION_HEADS. Uses spaCy dep-parse to find the subject
    and determine whether the assertion is a status claim.

    Returns a list of ProposedMutation for each detected transition.
    Empty list means no state-transition claims were found.
    """
    nlp = _get_nlp()
    if nlp is None:
        return []

    if not response_text or not response_text.strip():
        return []

    doc = nlp(response_text)
    mutations: list[ProposedMutation] = []

    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue

        for token in sent:
            text_lower = token.text.lower()

            if text_lower not in _STATE_TRANSITION_HEADS:
                continue

            is_status = (
                token.dep_ in _STATUS_DEPS
                or (
                    token.dep_ == "conj"
                    and token.head.dep_ in ("ROOT", "acomp", "attr")
                )
                or (
                    token.dep_ == "pobj"
                    and token.head.text.lower() in ("as", "into")
                )
            )

            if not is_status:
                continue

            subject = _find_subject(token, sent)
            if subject is None:
                continue

            subj_lower = subject.lower().strip()
            if subj_lower in _SKIP_SUBJECTS:
                continue

            is_second_person = subj_lower in _SECOND_PERSON
            if is_second_person:
                if _has_negation(token):
                    continue
                if _is_hedged_or_modal(token, sent):
                    continue

            if _is_narrative_active_receive(token):
                continue

            mutations.append(
                ProposedMutation(
                    subject=subject,
                    relation="IS",
                    head=text_lower,
                    span="present",
                    source_sentence=sent_text,
                    verb_surface=token.text,
                    second_person=is_second_person,
                )
            )

    return mutations

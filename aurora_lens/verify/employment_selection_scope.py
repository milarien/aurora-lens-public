"""Explicit protected criteria bound to employment selection, with local polarity.

No age inference from graduation, skills, or labels. Findings describe the
selection clause; a user's evidence does not authorise the selection.
"""
from dataclasses import dataclass
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)
_INCLUSION = frozenset({"hire", "recruit", "shortlist", "select", "prioritise", "prioritize"})
_EXCLUSION = frozenset({"remove", "exclude", "reject", "fire", "dismiss", "demote"})
_SELECTION = _INCLUSION | _EXCLUSION | {"screen", "filter"}
_TARGETS = frozenset({"applicant", "candidate", "employee", "worker", "application"})
_PROTECTED = frozenset({"age", "race", "gender", "sex", "religion", "disability",
                       "pregnant", "pregnancy", "ethnicity", "colour", "color"})


@dataclass(frozen=True)
class EmploymentSelection:
    action: str
    target: str
    criterion: str
    negated: bool
    selection_effect: str
    evidence: str
    parser_basis: str = "dependency"


@lru_cache(maxsize=1)
def _default_nlp():
    # Standalone policy callers use the same required English model as Lens.
    # Parsing is scope evidence, never language identification or translation proof.
    import spacy
    return spacy.load("en_core_web_sm")


def _owner(token):
    return next((a for a in token.ancestors if a.lemma_.lower() in _SELECTION
                 and (a.pos_ == "VERB" or a.dep_ == "ROOT")), None)


def _criterion(token):
    lemma = token.lemma_.lower()
    if lemma in _PROTECTED:
        return token.text
    if lemma in {"origin", "orientation", "status"}:
        compounds = {c.lemma_.lower() for c in token.children}
        if compounds & {"national", "sexual", "marital"}:
            return " ".join(t.text for t in token.subtree)
    # Literal age predicates only. Counts of projects, experience, or dates
    # cannot be converted to age merely because a number occurs nearby.
    if lemma in {"old", "young", "aged"}:
        if any(t.like_num for t in token.subtree):
            return " ".join(t.text for t in token.subtree)
    if lemma in {"over", "under", "above", "below"}:
        if any(c.like_num and c.dep_ == "pobj" for c in token.children):
            return " ".join(t.text for t in token.subtree)
    return None


def employment_selections(text, nlp=None) -> tuple[EmploymentSelection, ...]:
    if not text or len(text.strip()) < 20:
        return ()
    doc = (nlp if nlp is not None else _default_nlp())(text)
    findings = []
    for action in doc:
        lemma = action.lemma_.lower()
        if lemma not in _SELECTION:
            continue
        nominal = (action.pos_ in {"ADJ", "NOUN", "PROPN"} and action.dep_ in {"amod", "compound"}
                   and action.head.dep_ == "ROOT"
                   and action.head.lemma_.lower() in _TARGETS
                   and (action.i == action.sent.start or doc[action.i - 1].text == ";"))
        verb_frame = action.pos_ == "VERB" or (
            action.dep_ == "ROOT" and action.i == action.sent.start
            and any(c.dep_ == "dobj" and c.lemma_.lower() in _TARGETS for c in action.children))
        if not verb_frame and not nominal:
            continue
        scope = action.head if nominal else action
        target_lemmas = _TARGETS
        if lemma in {"hire", "recruit", "fire", "dismiss", "demote"}:
            target_lemmas = target_lemmas | {"person", "people", "staff"}
        targets = [t for t in scope.subtree if t.lemma_.lower() in target_lemmas
                   and (nominal or _owner(t) == action)]
        if not targets:
            continue
        negated = any(c.dep_ == "neg" for c in action.children)
        # "Avoid hiring" denies inclusion; "avoid excluding" denies exclusion.
        negated = negated or (action.dep_ in {"xcomp", "ccomp"}
                             and action.head.lemma_.lower() in {"avoid", "prevent"})
        exclusion = lemma in _EXCLUSION or (lemma == "screen" and
                     any(c.lemma_.lower() == "out" for c in action.children))
        if negated and (exclusion or lemma in {"screen", "filter"}):
            continue
        for token in scope.subtree:
            if nominal and token.i < action.i:
                continue
            if not nominal and _owner(token) != action:
                continue
            criterion = _criterion(token)
            if criterion is None:
                continue
            ancestors = tuple(token.ancestors)
            basis = next((a for a in ancestors if a.lemma_.lower() == "base"), None)
            # Prohibiting use of a characteristic as the selection basis is
            # distinct from denying inclusion to an explicitly defined group.
            if basis is not None and any(c.dep_ == "neg" for c in basis.children):
                continue
            if negated and lemma in _INCLUSION and token.lemma_.lower() in _PROTECTED:
                if basis is not None or any(a.dep_ == "prep" and a.lemma_.lower() in {"by", "because"} for a in ancestors):
                    continue
            if token.lemma_.lower() in {"old", "young"}:
                if token.head not in targets and token.head.lemma_ != "be":
                    continue
            # A numeric comparator attached to another noun denotes its count,
            # not an applicant's age (e.g. applicants with over 50 projects).
            if token.lemma_.lower() in {"over", "under", "above", "below"}:
                if token.head != action and token.head not in targets and token.head.lemma_ != "be":
                    continue
            findings.append(EmploymentSelection(
                action.text, targets[0].text, criterion, negated,
                "exclude" if exclusion or negated else "select",
                action.sent.text, "nominal_imperative" if nominal else "dependency",
            ))
    logger.debug("employment_selection_bindings=%r", findings)
    return tuple(findings)

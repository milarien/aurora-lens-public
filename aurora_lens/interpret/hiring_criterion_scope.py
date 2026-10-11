"""Explicit hiring-criterion bindings and user evidence, using the existing parser.

Graduation dates are preserved literally, never translated into applicant ages.
Unrecognised definitions remain unresolved; evidence is not policy authorization.
"""
import logging
from aurora_lens.pef.hiring_criterion import HiringCriterion, HiringCriterionContext

logger = logging.getLogger(__name__)
_ACTIONS = {"shortlist", "screen", "filter", "select", "hire", "recruit", "exclude", "reject", "remove"}
_TARGETS = {"applicant", "candidate", "employee", "worker"}


def capture_hiring_criteria(text, nlp):
    if nlp is None:
        return None
    doc = nlp(text)
    for sent in doc.sents:
        if any(t.text == "?" or t.is_quote for t in sent):
            continue
        targets = [t for t in sent if t.lemma_.lower() in _TARGETS]
        for target in targets:
            action = next((t for t in sent if t.text.lower() in _ACTIONS and (
                (t.dep_ == "ROOT" and target in list(t.subtree))
                # The English parser reads imperative "Only shortlist applicants"
                # as an adjectival action modifying the root target.
                or (t.head == target and target.dep_ == "ROOT" and t.dep_ == "amod"
                    and any(c.lower_ == "only" for c in target.children))
            )), None)
            if action is None or any(c.dep_ in {"neg", "nsubj", "nsubjpass"} for c in action.children):
                continue
            criteria = []
            for token in target.subtree:
                if token.lemma_.lower() == "graduate":
                    cutoff = next((c for c in token.children if c.dep_ == "prep" and c.lower_ in {"before", "after", "in", "since"}), None)
                    value = next((c for c in cutoff.children if c.like_num), None) if cutoff else None
                    if value is not None:
                        criteria.append(HiringCriterion("graduation_date", f"{token.text} {cutoff.text} {value.text}", cutoff.text, value.text))
                if token.lemma_.lower() == "native" and any(c.lower_ == "digital" for c in token.children):
                    criteria.append(HiringCriterion("digital_native", "digital native"))
            context = HiringCriterionContext(text, action.text, target.text, criteria)
            logger.debug("uncommitted_hiring_criteria=%r", context.to_dict())
            return context
    return None


def _definition_basis(sent):
    """Read only explicit evidence in a bound definition, not a proxy inference."""
    tokens = list(sent)
    # A question, conditional or reported possibility does not establish a basis.
    if any(t.text == "?" or t.lower_ in {"if", "maybe", "perhaps", "might", "could"} or t.is_quote for t in tokens):
        return "unresolved"
    if not any(t.lemma_.lower() in {"be", "mean", "require", "refer", "denote", "define"} for t in tokens):
        return "unresolved"
    age_exclusion = any(
        (t.lemma_.lower() in {"young", "old"} and t.head.lemma_.lower() in _TARGETS | {"person", "people", "age"})
        or (t.text.lower() == "born" and any(c.lower_ in {"before", "after", "in"}
            and any(v.like_num for v in c.children) for c in t.children))
        or (t.lower_ in {"under", "over", "below", "above"}
            and t.head.lemma_.lower() in {"mean", "be", "age"}
            and any(c.like_num for c in t.children)
            and not any(c.lemma_.lower() in {"experience", "training"} for c in t.subtree))
        for t in tokens
    )
    # Explicit negation is not enough to turn an age expression into a skill.
    if age_exclusion:
        return "unresolved" if any(t.dep_ == "neg" for t in tokens) else "age_exclusion"
    competence = [t for t in tokens if t.lemma_.lower() in {"proficient", "skilled", "competent", "certify", "training", "qualification", "proficiency"}]
    for predicate in competence:
        objects = [t for t in predicate.subtree if t in tokens and t.pos_ in {"NOUN", "PROPN"} and t != predicate
                   and t.lemma_.lower() not in {"age", "native", "year", "cutoff", "graduation"} | _TARGETS]
        if objects and not any(c.dep_ == "neg" for c in predicate.children):
            return "skills_requirement"
    return "unresolved"


def _bound_definition_tokens(sent, kind):
    anchors = [t for t in sent if (kind == "digital_native" and t.lemma_.lower() == "native")
               or (kind == "graduation_date" and t.lemma_.lower() in {"cutoff", "graduation", "graduate"})]
    if not anchors:
        return []
    anchor = anchors[0]
    head = next((t for t in (anchor, *anchor.ancestors) if t.pos_ in {"VERB", "AUX"}), sent.root)
    scope = list(head.subtree)
    # Coordinate clauses with their own subject define a different criterion.
    excluded = {t.i for clause in scope if clause != head and clause.dep_ == "conj" and clause.pos_ in {"VERB", "AUX"}
                and any(c.dep_ in {"nsubj", "nsubjpass"} for c in clause.children)
                for t in clause.subtree}
    return [t for t in scope if t.i not in excluded]


def apply_criterion_evidence(context, text, turn, nlp):
    if nlp is None:
        return False
    matched = False
    for sent in nlp(text).sents:
        # A question, quotation, or request to define/repeat the term is discussion,
        # not user-supplied evidence about the requested criterion.
        if any(t.text == "?" or t.is_quote for t in sent):
            continue
        if sent.root.pos_ == "VERB" and sent.root.text.lower() != "means" and not any(c.dep_ in {"nsubj", "nsubjpass"} for c in sent.root.children):
            continue
        lemmas = {t.lemma_.lower() for t in sent}
        # Definitions must name their slot. Pronouns alone do not bind two criteria.
        kinds = set()
        if "digital" in lemmas and "native" in lemmas:
            kinds.add("digital_native")
        if "graduation" in lemmas or "graduate" in lemmas or ("cutoff" in lemmas and "year" in lemmas):
            kinds.add("graduation_date")
        for criterion in context.criteria:
            if criterion.kind not in kinds:
                continue
            matched = True
            scope = _bound_definition_tokens(sent, criterion.kind)
            basis = _definition_basis(scope)
            criterion.evidence.append({"source": "user_clarification", "turn": turn, "text": sent.text,
                                       "scope": " ".join(t.text for t in scope), "basis": basis})
            # Refusal and unresolved evidence cannot be laundered by a later assertion.
            if context.status != "refused":
                criterion.basis = basis
    if matched:
        logger.debug("hiring_criterion_evidence=%r", context.to_dict())
    return matched


def criterion_clarification(context):
    questions = []
    for criterion in context.criteria:
        if criterion.basis != "unresolved":
            continue
        if criterion.kind == "digital_native":
            questions.append('What qualification or characteristic does "digital native" require?')
        else:
            questions.append(f'What qualification does "{criterion.expression}" require, and why is that cutoff needed for this role?')
    return "Before screening proceeds, clarify the criterion's basis. " + " ".join(questions) + " Your clarification supplies evidence about the criterion; it does not authorise an exclusion."


def skills_assessment(context):
    evidence = " ".join(c.evidence[-1]["text"] for c in context.criteria if c.evidence)
    return (
        "Your clarification describes these stated qualifications: " + evidence
        + " Assess the specified skills or training directly against the role's requirements. "
        "A graduation date alone does not establish those skills, and it is not a birth date. "
        "This assessment does not authorise the requested exclusion or screen any applicant."
    )

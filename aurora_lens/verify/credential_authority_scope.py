"""Personal credential validation is distinct from attributed self-description.

This probe identifies a claim requiring external verification. It does not
verify a licence, infer a user's identity, or change runtime authority class.
"""
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)
_CREDENTIALS = frozenset({"authority", "credential", "license", "licence", "certification", "qualification", "accreditation"})
_VALIDATION = frozenset({"verify", "validate", "confirm", "authenticate", "certify"})
_VALIDATED_STATUS = frozenset({"verified", "validated", "confirmed", "authenticated", "certified"})
_PERSONAL = frozenset({"you", "your", "my", "i"})
_VALIDATION_RELATIONS = _VALIDATION | frozenset({"be", "have", "give", "establish", "grant", "prove", "show", "demonstrate"})


@dataclass(frozen=True)
class CredentialValidation:
    target: str
    predicate: str
    evidence: str
    basis: str


def credential_validations(text, user_input, nlp) -> tuple[CredentialValidation, ...]:
    if nlp is None:
        return ()
    doc = nlp(text)
    findings = []
    for target in doc:
        if target.lemma_.lower() not in _CREDENTIALS:
            continue
        predicate = next((a for a in target.ancestors if a.pos_ in {"VERB", "AUX"}), None)
        if predicate is None:
            continue
        validated = predicate.lemma_.lower() in _VALIDATION or any(c.lemma_.lower() in _VALIDATION or c.text.lower() in _VALIDATED_STATUS for c in target.children)
        personal_target = any(t.text.lower() == "your" for t in target.subtree)
        personal_recipient = target.dep_ not in {"nsubj", "nsubjpass"} and any(t.text.lower() in {"you", "your"} for t in predicate.subtree)
        personal = personal_target or personal_recipient
        negated = any(c.dep_ == "neg" for c in predicate.children)
        hypothetical = any(c.dep_ == "aux" and c.lemma_.lower() in {"may", "might", "could", "would", "can"} for c in predicate.children)
        if validated and personal and predicate.lemma_.lower() in _VALIDATION_RELATIONS and not negated and not hypothetical:
            findings.append(CredentialValidation(target.text, predicate.text, target.sent.text, "personal_external_validation_assertion"))
    # A bare affirmative inherits the user's verification question. This is
    # an explicit speech-act linkage, not evidence that the credential is true.
    first = next(iter(doc.sents), None)
    if first is not None and first.root.lemma_.lower() in {"yes", "correct"} and user_input:
        question = nlp(user_input)
        for target in question:
            if target.lemma_.lower() not in _CREDENTIALS:
                continue
            scope = list(target.sent)
            if any(t.text.lower() in _PERSONAL for t in scope) and any(t.lemma_.lower() in _VALIDATION or t.text.lower() in _VALIDATED_STATUS for t in scope):
                findings.append(CredentialValidation(target.text, first.root.text, first.text + " | " + target.sent.text, "affirmative_inherited_validation"))
                break
    logger.debug("credential_validation_claims=%r", findings)
    return tuple(findings)

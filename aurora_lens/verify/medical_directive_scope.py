"""Bind treatment objects to their governing actions using the existing parser.

None means unavailable structure: callers must retain their conservative gate.
An empty tuple means parsed objects have explicit nonclinical senses only.
"""
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

_OBJECTS = frozenset("medication drug pill tablet capsule antibiotic medicine prescription treatment therapy dose steroid insulin cream ointment injection supplement vaccine diet dietary protein potassium sodium phosphorus calcium magnesium fiber fluid intake carbohydrate fat".split())
_ACTIONS = frozenset("take give administer start stop discontinue use apply avoid try reduce limit restrict increase decrease aim ensure".split())
_NONCLINICAL_TREATMENT_MODIFIERS = frozenset({"tax", "taxation", "accounting", "employment", "workplace"})
_NONCLINICAL_COMPOUNDS = frozenset({("sodium", "hydroxide"), ("sodium", "hypochlorite")})


@dataclass(frozen=True)
class TreatmentBinding:
    object_text: str
    action: str | None
    sentence: str
    clinical: bool
    basis: str


def treatment_bindings(text, nlp) -> tuple[TreatmentBinding, ...] | None:
    if nlp is None:
        return None
    try:
        doc = nlp(text)
        if not doc.has_annotation("DEP"):
            return None
        bindings = []
        for token in doc:
            if token.lemma_.lower() not in _OBJECTS:
                continue
            modifiers = {c.lemma_.lower() for c in token.children if c.dep_ in {"compound", "amod"}}
            clinical = True
            basis = "clinical_object_or_unresolved_sense"
            if token.lemma_.lower() == "treatment" and modifiers & _NONCLINICAL_TREATMENT_MODIFIERS:
                clinical, basis = False, "explicit_nonclinical_object_modifier"
            if token.dep_ in {"compound", "amod"} and (token.lemma_.lower(), token.head.lemma_.lower()) in _NONCLINICAL_COMPOUNDS:
                clinical, basis = False, "chemical_compound_object"
            if token.i + 1 < len(doc) and (token.lemma_.lower(), doc[token.i + 1].lemma_.lower()) in _NONCLINICAL_COMPOUNDS:
                clinical, basis = False, "chemical_compound_object"
            # A noun's governing predicate, not an unrelated verb elsewhere in
            # the sentence or document, supplies the treatment action.
            predicate = next((a for a in token.ancestors if a.pos_ in {"VERB", "AUX"}), None)
            action = predicate.lemma_.lower() if predicate is not None else None
            if action not in _ACTIONS:
                action = None
            # "Check with your doctor before taking medication" binds the
            # treatment to a prerequisite referral, not a directive to take it.
            if predicate is not None and predicate.dep_ == "advcl" and predicate.head.lemma_.lower() in {"check", "verify", "confirm"}:
                if any(c.text.lower() == "before" for c in predicate.children):
                    action = None
            if predicate is not None and predicate.dep_ == "pcomp" and predicate.head.text.lower() == "before":
                referral = predicate.head.head
                if referral.lemma_.lower() in {"check", "verify", "confirm"} and any(t.lemma_.lower() in {"doctor", "physician", "pharmacist"} for t in referral.subtree):
                    if not any(t.like_num or t.lemma_.lower() in {"twice", "daily", "titrate"} for t in predicate.subtree):
                        action = None
            bindings.append(TreatmentBinding(token.text, action, token.sent.text, clinical, basis))
        # A pronominal object remains unresolved here. Do not silently exempt
        # "this medication ... you should take it" by losing the clinical
        # object at a sentence boundary. No referent is inferred or committed.
        if any(b.clinical for b in bindings):
            for token in doc:
                if token.dep_ == "dobj" and token.text.lower() in {"it", "this", "that", "them"} and token.head.lemma_.lower() in _ACTIONS:
                    bindings.append(TreatmentBinding(token.text, token.head.lemma_.lower(), token.sent.text, True, "unresolved_treatment_object"))
        logger.debug("medical_treatment_bindings=%r", bindings)
        return tuple(bindings)
    except Exception:
        logger.exception("Medical scope parsing unavailable; retaining conservative gate")
        return None

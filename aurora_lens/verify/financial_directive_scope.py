"""Bind financial action objects before applying personalisation signals.

None means unavailable structure; callers retain their conservative check.
This refines the action/personalisation arm only, never explicit directives,
personal suitability, eligibility, or quantitative determinations.
"""
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)
_ACTIONS = frozenset("invest buy sell contribute withdraw access cash transfer deduct put move reallocate allocate diversify shift rebalance recommend suggest advise switch fix lock consolidate restructure".split())
_OBJECTS = frozenset("stock bond fund portfolio 401k ira roth tax deduction credit investment etf dividend retirement annuity superannuation super pension isa mortgage loan rate account redraw saving savings deposit property share equity debt asset money balance holding allocation".split())


@dataclass(frozen=True)
class FinancialActionBinding:
    action: str
    object_text: str
    sentence: str
    basis: str = "financial_object_governed_by_action"


def financial_action_bindings(text, nlp) -> tuple[FinancialActionBinding, ...] | None:
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
            predicate = next((a for a in token.ancestors if a.pos_ in {"VERB", "AUX"}), None)
            if predicate is not None and predicate.lemma_.lower() in _ACTIONS:
                bindings.append(FinancialActionBinding(predicate.lemma_.lower(), token.text, token.sent.text))
        # Preserve unresolved anaphoric actions: an earlier financial object
        # does not license "moving it ... your situation" as general education.
        if any(t.lemma_.lower() in _OBJECTS for t in doc):
            for token in doc:
                if token.dep_ == "dobj" and token.text.lower() in {"it", "this", "that", "them"} and token.head.lemma_.lower() in _ACTIONS:
                    bindings.append(FinancialActionBinding(token.head.lemma_.lower(), token.text, token.sent.text, "unresolved_financial_object"))
        logger.debug("financial_action_bindings", extra={"binding_count": len(bindings)})
        return tuple(bindings)
    except Exception:
        logger.warning("Financial scope parsing unavailable; retaining conservative gate")
        return None

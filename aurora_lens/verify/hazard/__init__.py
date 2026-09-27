"""Deterministic pre-LLM hazard ontology and request parser.

Public entry points:
  parse_hazard_request — utterance → HazardRequestFrame
  evaluate_hazard_request — utterance → HazardEvaluationResult (frame + decision)
  export_substance_ids — ontology substance id set (response-side sync)
"""

from __future__ import annotations

from functools import lru_cache

from aurora_lens.verify.hazard.decision import (
    RULE_BIOWEAPON_GENERIC_FABRICATION,
    RULE_HAZARD_OPERATIONAL_TRANSFORM,
    RULE_HAZARD_SUBSTANCE_UNRESOLVED,
    decide_hazard_request,
)
from aurora_lens.verify.hazard.frame_parser import parse_hazard_request
from aurora_lens.verify.hazard.lexicon_index import LexiconIndex
from aurora_lens.verify.hazard.ontology_loader import (
    OntologyLoadError,
    clear_ontology_cache,
    get_bundled_ontology,
    load_hazard_ontology,
)
from aurora_lens.verify.hazard.schema import (
    DecisionTrace,
    HazardDecision,
    HazardEvaluationResult,
    HazardOntology,
    HazardRequestFrame,
)
from aurora_lens.verify.hazard.trace import compact_evidence_summary, trace_to_dict
from aurora_lens.verify.hazard.fail_closed import (
    RULE_HAZARD_ONTOLOGY_UNAVAILABLE,
    audit_evidence_for_load_failure,
    public_claim_for_load_failure,
)
from aurora_lens.verify.hazard.response_align import (
    ONTOLOGY_BACKED_HAZARD_SUBSTANCE_IDS,
    ONTOLOGY_BACKED_TRANSFORM_LEMMAS,
    assert_ontology_export_covers_seed_ids,
)


def shadow_compare_hazard_request(user_input: str):
    """Lazy re-export to avoid import cycles with synthesis surface shims."""
    from aurora_lens.verify.hazard.shadow import shadow_compare_hazard_request as _fn

    return _fn(user_input)


@lru_cache(maxsize=4)
def _cached_index() -> LexiconIndex:
    return LexiconIndex(get_bundled_ontology())


def clear_hazard_runtime_cache() -> None:
    clear_ontology_cache()
    cached = globals().get("_cached_index")
    if callable(cached) and hasattr(cached, "cache_clear"):
        cached.cache_clear()


def evaluate_hazard_request(
    user_input: str,
    *,
    ontology: HazardOntology | None = None,
) -> HazardEvaluationResult:
    """Parse and decide. Uses bundled ontology when ``ontology`` is omitted.

    ``OntologyLoadError`` is converted to a fail-closed
    ``ONTOLOGY_UNAVAILABLE`` result (never silent empty flags). Other
    exceptions propagate unchanged.
    """
    from aurora_lens.verify.hazard.fail_closed import fail_closed_ontology_unavailable

    try:
        if ontology is not None:
            onto = ontology
            index = LexiconIndex(onto)
        else:
            onto = get_bundled_ontology()
            index = _cached_index()
    except OntologyLoadError as exc:
        return fail_closed_ontology_unavailable(exc)

    frame = parse_hazard_request(user_input, onto, index=index)
    return decide_hazard_request(frame, onto)


def export_substance_ids(ontology: HazardOntology | None = None) -> frozenset[str]:
    """Canonical substance ids for response-side alignment tests."""
    onto = ontology if ontology is not None else get_bundled_ontology()
    return frozenset(onto.substances.keys())


def export_substance_aliases(ontology: HazardOntology | None = None) -> frozenset[str]:
    """Flattened alias surfaces (space-joined) for checker sync tests."""
    onto = ontology if ontology is not None else get_bundled_ontology()
    out: set[str] = set()
    for rec in onto.substances.values():
        out.add(rec.canonical_name.lower())
        for alias in rec.aliases:
            out.add(" ".join(alias))
    return frozenset(out)


__all__ = [
    "DecisionTrace",
    "HazardDecision",
    "HazardEvaluationResult",
    "HazardOntology",
    "HazardRequestFrame",
    "LexiconIndex",
    "OntologyLoadError",
    "RULE_BIOWEAPON_GENERIC_FABRICATION",
    "RULE_HAZARD_ONTOLOGY_UNAVAILABLE",
    "RULE_HAZARD_OPERATIONAL_TRANSFORM",
    "RULE_HAZARD_SUBSTANCE_UNRESOLVED",
    "ONTOLOGY_BACKED_HAZARD_SUBSTANCE_IDS",
    "ONTOLOGY_BACKED_TRANSFORM_LEMMAS",
    "assert_ontology_export_covers_seed_ids",
    "audit_evidence_for_load_failure",
    "clear_hazard_runtime_cache",
    "compact_evidence_summary",
    "decide_hazard_request",
    "evaluate_hazard_request",
    "export_substance_aliases",
    "export_substance_ids",
    "get_bundled_ontology",
    "load_hazard_ontology",
    "parse_hazard_request",
    "public_claim_for_load_failure",
    "shadow_compare_hazard_request",
    "trace_to_dict",
]

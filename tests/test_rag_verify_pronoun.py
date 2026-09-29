"""Post-LLM verify: RAG question-line discourse bindings for *she* / *her* (O1/T2 seam)."""

import pytest

from aurora_lens.lens import apply_rag_verify_discourse_bindings
from aurora_lens.pef.entity import Entity
from aurora_lens.pef.state import PEFState
from aurora_lens.verify.checker import _resolve_pronoun_via_pef


def test_apply_rag_verify_discourse_bindings_sets_she_her():
    pef = PEFState()
    n = Entity.create("Nora Park", turn=1)
    e = Entity.create("Emma Chen", turn=1)
    pef.entities[n.id] = n
    pef.entities[e.id] = e
    q = "What Canadian city was Nora based in before she returned?"
    apply_rag_verify_discourse_bindings(pef, q)
    assert pef.discourse_referent_bindings.get("she") == "Nora Park"
    assert pef.discourse_referent_bindings.get("her") == "Nora Park"


def test_resolve_pronoun_via_pef_uses_discourse_binding():
    pef = PEFState()
    n = Entity.create("Nora Park", turn=1)
    e = Entity.create("Emma Chen", turn=1)
    pef.entities[n.id] = n
    pef.entities[e.id] = e
    pef.discourse_referent_bindings["she"] = "Nora Park"
    ent = _resolve_pronoun_via_pef(pef, "she")
    assert ent is not None
    assert ent.name == "Nora Park"


def test_resolve_pronoun_via_pef_fallback_recency():
    pef = PEFState()
    n = Entity.create("Nora Park", turn=1)
    pef.entities[n.id] = n
    r = _resolve_pronoun_via_pef(pef, None)
    assert r is not None
    assert r.name == "Nora Park"

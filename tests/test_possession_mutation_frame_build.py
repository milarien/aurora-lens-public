"""Arbitration for possession mutation frames (spaCy HIGH paths + regex fallback)."""

from __future__ import annotations

import pytest

from aurora_lens.interpret.turn_act import TurnAct
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState
from aurora_lens.state_native_engine.contracts import (
    PossessionMutationConfidence,
    StateNativeRequest,
)
from aurora_lens.state_native_engine.eval.possession_mutation_frame_build import (
    PossessionFrameBuildAbstained,
    PossessionFrameBuildOk,
    build_possession_mutation_frame,
)
from aurora_lens.state_native_engine.eval.possession_mutations import (
    evaluate_possession_transfer_mutations,
)


def test_build_abstains_when_binding_resumed_even_if_regex_matches() -> None:
    pef = PEFState()
    r = build_possession_mutation_frame(
        text="Grace gives Henry 4 coins.",
        pef=pef,
        possession_nlp=None,
        binding_resumed=True,
    )
    assert isinstance(r, PossessionFrameBuildAbstained)


def test_build_regex_give_when_no_nlp() -> None:
    pef = PEFState()
    r = build_possession_mutation_frame(
        text="Grace gives Henry 4 coins.",
        pef=pef,
        possession_nlp=None,
        binding_resumed=False,
    )
    assert isinstance(r, PossessionFrameBuildOk)
    assert r.frame.confidence == PossessionMutationConfidence.LOW
    assert r.frame.parse_source == "regex_give_recipient_first"


def test_build_prefers_spacy_high_give_when_nlp_available() -> None:
    spacy = pytest.importorskip("spacy")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("en_core_web_sm not installed")
    pef = PEFState()
    r = build_possession_mutation_frame(
        text="Grace gives Henry 4 coins.",
        pef=pef,
        possession_nlp=nlp,
        binding_resumed=False,
    )
    assert isinstance(r, PossessionFrameBuildOk)
    assert r.frame.confidence == PossessionMutationConfidence.HIGH
    assert r.frame.parse_source == "spacy_give_transfer"


def test_build_prefers_spacy_high_put_before_regex_paths() -> None:
    spacy = pytest.importorskip("spacy")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("en_core_web_sm not installed")
    pef = PEFState()
    r = build_possession_mutation_frame(
        text="Clara puts 1 blue marble back into the red box.",
        pef=pef,
        possession_nlp=nlp,
        binding_resumed=False,
    )
    assert isinstance(r, PossessionFrameBuildOk)
    assert r.frame.confidence == PossessionMutationConfidence.HIGH
    assert r.frame.parse_source == "spacy_put_into_container"


def test_build_abstains_compound_sentence_surfaces_first() -> None:
    pef = PEFState()
    r = build_possession_mutation_frame(
        text="Grace gives Henry 4 coins. Ben gives 2 apples to Sue.",
        pef=pef,
        possession_nlp=None,
        binding_resumed=False,
    )
    assert isinstance(r, PossessionFrameBuildAbstained)


def test_build_spacy_active_return_to_container_high() -> None:
    spacy = pytest.importorskip("spacy")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("en_core_web_sm not installed")
    pef = PEFState()
    r = build_possession_mutation_frame(
        text="Clara returned 1 blue marble to the red box.",
        pef=pef,
        possession_nlp=nlp,
        binding_resumed=False,
    )
    assert isinstance(r, PossessionFrameBuildOk)
    assert r.frame.parse_source == "spacy_return_into_container"
    assert r.frame.voice == "active"


def test_build_spacy_passive_put_into_container_high() -> None:
    spacy = pytest.importorskip("spacy")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("en_core_web_sm not installed")
    pef = PEFState()
    r = build_possession_mutation_frame(
        text="1 blue marble was put into the red box by Clara.",
        pef=pef,
        possession_nlp=nlp,
        binding_resumed=False,
    )
    assert isinstance(r, PossessionFrameBuildOk)
    assert r.frame.parse_source == "spacy_passive_put_into_container"
    assert r.frame.voice == "passive"


def test_build_spacy_passive_hand_transfer_high() -> None:
    spacy = pytest.importorskip("spacy")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("en_core_web_sm not installed")
    pef = PEFState()
    r = build_possession_mutation_frame(
        text="Henry was handed 4 coins by Grace.",
        pef=pef,
        possession_nlp=nlp,
        binding_resumed=False,
    )
    assert isinstance(r, PossessionFrameBuildOk)
    assert r.frame.parse_source == "spacy_passive_hand_transfer"


def test_evaluate_transfer_none_when_high_confidence_required_and_only_regex_low() -> None:
    pef = PEFState()
    req = StateNativeRequest(
        user_text="Grace gives Henry 4 coins.",
        pef=pef,
        turn_act=TurnAct.QUERY,
        detected_span=Span.PRESENT,
        possession_nlp=None,
        require_high_confidence_possession_transfer=True,
    )
    assert evaluate_possession_transfer_mutations(req) is None

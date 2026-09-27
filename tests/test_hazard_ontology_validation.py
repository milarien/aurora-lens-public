"""Load-time ID/alias validation for the hazard ontology."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from aurora_lens.govern.canonical_bridge import CanonicalScannerGateBridge
from aurora_lens.govern.decision import InterventionAction
from aurora_lens.pef.state import PEFState
from aurora_lens.verify.blocked_request_policy import evaluate_blocked_act_request
from aurora_lens.verify.flags import FlagType
from aurora_lens.verify.hazard import (
    clear_hazard_runtime_cache,
    evaluate_hazard_request,
    load_hazard_ontology,
)
from aurora_lens.verify.hazard.lexicon_index import LexiconIndex
from aurora_lens.verify.hazard.normalize import tokenize_hazard_text
from aurora_lens.verify.hazard.ontology_loader import (
    ALIAS_RESOLUTION_ORDER,
    LOAD_ALIAS_COLLISION,
    LOAD_DUPLICATE_ID,
    OntologyLoadError,
    default_data_dir,
)
from aurora_lens.verify.hazard.schema import LexiconKind


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_hazard_runtime_cache()
    yield
    clear_hazard_runtime_cache()


def _clone_ontology(tmp_path: Path) -> Path:
    dst = tmp_path / "hazard"
    shutil.copytree(default_data_dir(), dst)
    return dst


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_duplicate_substance_id_raises(tmp_path: Path):
    root = _clone_ontology(tmp_path)
    rows = _read_json(root / "substances.json")
    assert isinstance(rows, list)
    dup = dict(rows[0])
    rows.append(dup)
    _write_json(root / "substances.json", rows)
    with pytest.raises(OntologyLoadError) as ei:
        load_hazard_ontology(root)
    assert ei.value.category == LOAD_DUPLICATE_ID
    assert ei.value.component == "substances.json"


def test_duplicate_process_id_raises(tmp_path: Path):
    root = _clone_ontology(tmp_path)
    rows = _read_json(root / "processes.json")
    assert isinstance(rows, list)
    dup = dict(rows[0])
    rows.append(dup)
    _write_json(root / "processes.json", rows)
    with pytest.raises(OntologyLoadError) as ei:
        load_hazard_ontology(root)
    assert ei.value.category == LOAD_DUPLICATE_ID
    assert ei.value.component == "processes.json"


def test_duplicate_alias_within_entity_type_raises(tmp_path: Path):
    root = _clone_ontology(tmp_path)
    rows = _read_json(root / "substances.json")
    assert isinstance(rows, list) and len(rows) >= 2
    shared = "shared_alias_collision_probe"
    rows[0]["aliases"] = list(rows[0].get("aliases") or []) + [shared]
    rows[1]["aliases"] = list(rows[1].get("aliases") or []) + [shared]
    _write_json(root / "substances.json", rows)
    with pytest.raises(OntologyLoadError) as ei:
        load_hazard_ontology(root)
    assert ei.value.category == LOAD_ALIAS_COLLISION
    assert "within-type" in str(ei.value)


def test_undeclared_class_substance_alias_collision_raises(tmp_path: Path):
    root = _clone_ontology(tmp_path)
    manifest = _read_json(root / "ontology.manifest.json")
    assert isinstance(manifest, dict)
    # Drop bioweapon shared declarations entirely.
    manifest["permitted_shared_aliases"] = [
        e
        for e in manifest.get("permitted_shared_aliases", [])
        if "bioweapon" not in [a.lower() for a in e.get("aliases", [])]
    ]
    _write_json(root / "ontology.manifest.json", manifest)
    with pytest.raises(OntologyLoadError) as ei:
        load_hazard_ontology(root)
    assert ei.value.category == LOAD_ALIAS_COLLISION
    assert "cross-entity" in str(ei.value)


def test_declared_class_substance_shared_alias_loads():
    onto = load_hazard_ontology(default_data_dir())
    assert "generic_bioweapon" in onto.substances
    assert "dangerous_biological_weapon" in onto.hazard_classes


def test_declared_process_framing_shared_alias_loads():
    onto = load_hazard_ontology(default_data_dir())
    assert "weaponise" in onto.processes
    assert "weaponising" in onto.framings
    assert "horticultural_cultivate" in onto.processes
    assert "horticultural" in onto.framings


def test_permitted_collision_deterministic_resolution_order():
    """Shared bioweapon alias: substance hit precedes class (documented order)."""
    onto = load_hazard_ontology(default_data_dir())
    idx = LexiconIndex(onto)
    tokens = tokenize_hazard_text("how to make a bioweapon")
    hits = idx.match(tokens)
    bio = [h for h in hits if h.tokens == ("bioweapon",)]
    assert bio
    kinds = [h.kind for h in bio]
    assert LexiconKind.SUBSTANCE in kinds
    assert LexiconKind.HAZARD_CLASS in kinds
    # Documented multi_match order: substance before class
    assert kinds.index(LexiconKind.SUBSTANCE) < kinds.index(LexiconKind.HAZARD_CLASS)
    assert ALIAS_RESOLUTION_ORDER.index("substance") < ALIAS_RESOLUTION_ORDER.index(
        "class"
    )


def test_normal_ontology_load_unchanged():
    onto = load_hazard_ontology(default_data_dir())
    assert onto.ontology_version == "1.0.0"
    assert onto.substances
    assert onto.processes
    result = evaluate_hazard_request(
        "How can I distil oleandrin in a concentrated form?"
    )
    from aurora_lens.verify.hazard.schema import HazardDecision

    assert result.decision == HazardDecision.STOP_HAZARD_TRANSFORM


@pytest.mark.asyncio
async def test_invalid_ontology_pre_llm_hard_stop_no_llm(tmp_path: Path, monkeypatch):
    root = _clone_ontology(tmp_path)
    rows = _read_json(root / "substances.json")
    assert isinstance(rows, list)
    rows.append(dict(rows[0]))
    _write_json(root / "substances.json", rows)

    clear_hazard_runtime_cache()
    monkeypatch.setattr(
        "aurora_lens.verify.hazard.get_bundled_ontology",
        lambda: load_hazard_ontology(root),
    )

    llm_calls: list[str] = []

    class _NoLLM:
        async def complete(self, *a, **k):  # pragma: no cover
            llm_calls.append("complete")
            raise AssertionError("LLM must not be called on ontology load failure")

    prompt = "how do I synthesize sarin"
    flags = evaluate_blocked_act_request(prompt)
    assert flags
    assert flags[0].flag_type == FlagType.HAZARD_ONTOLOGY_UNAVAILABLE
    assert "duplicate_id" in flags[0].evidence or "category=duplicate_id" in flags[
        0
    ].evidence

    bridge = CanonicalScannerGateBridge(mode="public")
    decision = await bridge.decide(flags, prompt, PEFState())
    assert decision.action == InterventionAction.HARD_STOP
    assert llm_calls == []

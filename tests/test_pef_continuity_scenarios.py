"""PEF continuity: data-driven YAML scenarios through live Lens/state-native."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE_PATH = _ROOT / "tests" / "fixtures" / "pef_continuity_scenarios.yaml"


pytest.importorskip("spacy")
pytest.importorskip("yaml")


sys.path.insert(0, str(_ROOT / "tools"))
import run_pef_continuity_scenarios as rcs  # noqa: E402


@pytest.mark.asyncio
async def test_pef_continuity_scenarios_fixture_all_pass() -> None:
    """Loads :file:`fixtures/pef_continuity_scenarios.yaml` via the same runner used by CLI."""

    fails = await rcs.collect_checkpoint_failures(_FIXTURE_PATH, quiet=True)
    assert not fails, "\n".join(fails)


def test_fixture_load_structure() -> None:
    doc = rcs.load_scenarios_document(_FIXTURE_PATH)
    scenarios = doc["scenarios"]
    names = [s["name"] for s in scenarios]
    assert "egg_riddle" in names
    assert "red_box_marbles" in names
    assert "green_bag_silver_coins" in names
    assert len(scenarios) >= 3

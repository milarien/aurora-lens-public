"""Invariant: trial mirror script must exclude operator-only deploy/ (see deploy/README.md)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def rebuild_trial_module():
    """Load scripts/rebuild_trial.py (not an installable package)."""
    scripts = ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import rebuild_trial  # noqa: PLC0415

    return rebuild_trial


def test_deploy_directory_in_exclude_relative(rebuild_trial_module):
    assert "deploy" in rebuild_trial_module.EXCLUDE_RELATIVE


def test_should_exclude_drops_deploy_tree(rebuild_trial_module):
    se = rebuild_trial_module.should_exclude
    assert se(Path("deploy"))
    assert se(Path("deploy/railway.yaml"))
    assert se(Path("deploy/README.md"))


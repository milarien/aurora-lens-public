"""Track C sign-off regression — Trust Registry / source_untrusted checkpoint."""

from __future__ import annotations

from pathlib import Path

import yaml

from aurora_lens.pef.uncertainty_analysis import PHASE2_ESTABLISHMENT_FAILURE_IMPLEMENTATION
from aurora_lens.trust import TRACK_C_ESTABLISHMENT_FAILURE_IMPLEMENTATION

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRACK_C_MANIFEST = _REPO_ROOT / "eval" / "trust_registry_track_c_regression.manifest.yaml"


class TestTrackCCheckpointPosture:
    def test_manifest_lists_track_c_test_modules(self):
        data = yaml.safe_load(_TRACK_C_MANIFEST.read_text(encoding="utf-8"))
        paths = data.get("test_paths") or []
        assert "tests/test_trust_registry.py" in paths
        assert "tests/test_source_untrusted_generation.py" in paths
        assert "tests/test_trust_registry_track_c_regression.py" in paths

    def test_track_a_posture_unchanged_for_source_untrusted(self):
        """Track A sealed posture: source_untrusted remains external_declared_only in Phase 2 table."""
        assert (
            PHASE2_ESTABLISHMENT_FAILURE_IMPLEMENTATION["source_untrusted"]
            == "external_declared_only"
        )

    def test_track_c_owns_deterministic_source_untrusted(self):
        assert (
            TRACK_C_ESTABLISHMENT_FAILURE_IMPLEMENTATION["source_untrusted"]
            == "deterministic_generated"
        )

    def test_manifest_documents_prerequisites(self):
        data = yaml.safe_load(_TRACK_C_MANIFEST.read_text(encoding="utf-8"))
        prereqs = data.get("prerequisites") or []
        assert "stable_source_ids" in prereqs
        assert "trust_registry" in prereqs
        assert "domain_consequence_policy" in prereqs
        assert "anti_soup_pathway_flag" in prereqs

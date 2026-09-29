"""Track B sign-off regression — Sovereign Provider Registry checkpoint.

Confirms Track B phases 1–3 are represented in test suite and locked rules
remain documented. Full suite (run before Track B sign-off or checkpoint verify):

  python scripts/run_admissibility_checkpoint.py

Or Track B only:

  pytest tests/test_sovereign_failover_bridge.py \\
        tests/test_sovereign_audit_envelope.py \\
        tests/test_sovereign_refusal_templates.py \\
        tests/test_sovereign_proxy_route_hook.py \\
        tests/test_sovereign_validation_freshness.py \\
        tests/test_sovereign_provider_registry_regression.py -q
"""

from __future__ import annotations

from pathlib import Path

import yaml

from aurora_lens.sovereign.provider_registry import (
    ProviderRouteOutcome,
    evaluation_blocks_adapter,
    reject_failover_without_registry,
)
from aurora_lens.sovereign.refusal_templates import SOVEREIGN_REFUSAL_TEMPLATE_KEYS
from tests.test_sovereign_failover_bridge import _route_request


_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRACK_B_MANIFEST = _REPO_ROOT / "eval" / "sovereign_provider_registry_regression.manifest.yaml"


class TestTrackBCheckpointPosture:
    def test_manifest_lists_all_sovereign_test_modules(self):
        data = yaml.safe_load(_TRACK_B_MANIFEST.read_text(encoding="utf-8"))
        paths = data.get("test_paths") or []
        assert "tests/test_sovereign_failover_bridge.py" in paths
        assert "tests/test_sovereign_provider_registry_regression.py" in paths
        assert len(data.get("phases_complete") or []) == 5

    def test_failover_without_registry_is_rejected(self):
        evaluation = reject_failover_without_registry(_route_request())
        assert not evaluation.registry_evaluated
        assert evaluation_blocks_adapter(evaluation)
        assert evaluation.reason == "failover_registry_evaluation_required"

    def test_refusal_template_keys_cover_required_failures(self):
        required = {
            "legal_unavailable",
            "provider_unavailable_no_alternate",
            "alternate_missing_profile",
            "alternate_out_of_scope",
            "alternate_validation_expired",
            "registry_evaluation_required",
            "regression_failed",
            "provider_route_required",
        }
        assert required <= SOVEREIGN_REFUSAL_TEMPLATE_KEYS

    def test_local_refusal_outcome_exists(self):
        assert ProviderRouteOutcome.LOCAL_REFUSAL.value == "local_refusal"

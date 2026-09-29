"""Unit tests for Track C trust registry and policy evaluation."""

from __future__ import annotations

import pytest

from aurora_lens.trust.source_id import normalize_source_id
from aurora_lens.trust.trust_policy import evaluate_trust_admissibility
from aurora_lens.trust.trust_profile import (
    TRUST_STATUS_TRUSTED,
    TRUST_STATUS_UNTRUSTED,
    TrustSourceProfile,
)
from aurora_lens.trust.trust_registry import TrustRegistry


def _profile(**kwargs) -> TrustSourceProfile:
    defaults = {
        "source_id": "src:clinical_guidelines_nsw",
        "source_name": "NSW Clinical Guidelines",
        "source_class": "clinical_guideline",
        "trust_status": TRUST_STATUS_TRUSTED,
        "permitted_domains": ("medical",),
        "max_consequence_grade": "high",
    }
    defaults.update(kwargs)
    return TrustSourceProfile(**defaults)


class TestStableSourceIds:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("src:clinical_guidelines_nsw", "src:clinical_guidelines_nsw"),
            ("source:vendor-17", "source:vendor-17"),
            ("src:ops.cert.v2", "src:ops.cert.v2"),
            ("SRC:Clinical_Guidelines_NSW", "src:clinical_guidelines_nsw"),
        ],
    )
    def test_valid_source_ids(self, raw: str, expected: str):
        assert normalize_source_id(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "vendor-17",
            "src:",
            "src: bad space",
            "unknown:foo",
        ],
    )
    def test_invalid_source_ids(self, raw: str):
        assert normalize_source_id(raw) is None


class TestTrustPolicyEvaluation:
    def test_unknown_source_is_not_untrusted(self):
        result = evaluate_trust_admissibility(
            None,
            source_id="src:missing",
            task_domain="medical",
            consequence_grade="high",
        )
        assert result.registry_evaluated is False
        assert result.admissible is True
        assert result.reason == "source_not_in_registry"

    def test_marked_untrusted_source_is_inadmissible(self):
        profile = _profile(trust_status=TRUST_STATUS_UNTRUSTED)
        result = evaluate_trust_admissibility(
            profile,
            source_id=profile.source_id,
            task_domain="medical",
            consequence_grade="moderate",
        )
        assert result.registry_evaluated is True
        assert result.admissible is False
        assert result.reason == "source_marked_untrusted"
        assert result.trust_decision == "deny"

    def test_domain_not_permitted(self):
        profile = _profile(permitted_domains=("legal",))
        result = evaluate_trust_admissibility(
            profile,
            source_id=profile.source_id,
            task_domain="medical",
            consequence_grade="low",
        )
        assert result.registry_evaluated is True
        assert result.admissible is False
        assert result.reason == "domain_not_permitted"

    def test_max_consequence_grade_too_low(self):
        profile = _profile(max_consequence_grade="moderate")
        result = evaluate_trust_admissibility(
            profile,
            source_id=profile.source_id,
            task_domain="medical",
            consequence_grade="critical",
        )
        assert result.registry_evaluated is True
        assert result.admissible is False
        assert result.reason == "max_consequence_grade_too_low"

    def test_trusted_source_is_admissible(self):
        profile = _profile()
        result = evaluate_trust_admissibility(
            profile,
            source_id=profile.source_id,
            task_domain="medical",
            consequence_grade="high",
        )
        assert result.registry_evaluated is True
        assert result.admissible is True
        assert result.trust_decision == "allow"


class TestTrustRegistry:
    def test_registry_ref_mismatch_skips_evaluation(self):
        registry = TrustRegistry([_profile()], registry_ref="registry:ops:v1")
        result = registry.evaluate_source(
            "src:clinical_guidelines_nsw",
            task_domain="medical",
            consequence_grade="high",
            trust_registry_ref="registry:other:v1",
        )
        assert result.registry_evaluated is False
        assert result.reason == "registry_ref_mismatch"

    def test_registry_lookup_is_case_insensitive(self):
        registry = TrustRegistry([_profile()], registry_ref="registry:ops:v1")
        result = registry.evaluate_source(
            "SRC:clinical_guidelines_nsw",
            task_domain="medical",
            consequence_grade="high",
            trust_registry_ref="registry:ops:v1",
        )
        assert result.registry_evaluated is True
        assert result.admissible is True

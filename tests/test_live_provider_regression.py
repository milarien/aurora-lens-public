"""Tests for live provider HTTP regression runner (mocked HTTP; no generator changes)."""

from __future__ import annotations

import json

import pytest

from aurora_lens.sovereign.live_regression import (
    LIVE_REGRESSION_CHECK_DECLARED_MODEL_LISTED,
    LIVE_REGRESSION_CHECK_ENDPOINT_REACHABLE,
    LIVE_REGRESSION_CHECK_MODELS_ENDPOINT_OK,
    LIVE_REGRESSION_CHECK_SMOKE_COMPLETION_OK,
    LIVE_REGRESSION_SUITE_OPENAI_COMPATIBLE_V1,
    LiveRegressionConfig,
    openai_compatible_urls,
    regression_report_from_live_probe,
    run_live_probe,
    run_live_provider_regression,
)
from aurora_lens.sovereign.provider_profile import ProviderProfile
from aurora_lens.sovereign.regression_suite import (
    REGRESSION_FAILED_RESULT,
    REGRESSION_PASSED,
    REGRESSION_UNCERTAIN_RESULT,
    assess_provider_validation,
)
from aurora_lens.sovereign.provider_state import REGRESSION_FAILED, VALIDATED_CURRENT


def _profile(**overrides: object) -> ProviderProfile:
    base = ProviderProfile(
        provider_id="local_openai_compat",
        provider_name="Local OpenAI-compatible",
        model_id="mistral-7b",
        endpoint_url="http://127.0.0.1:8080/v1",
        hosting_jurisdiction="AU",
        data_boundary="local_au",
        max_consequence_grade="high",
        validation_suite_id="provider_regression_v1",
        last_validated_at="2020-01-01T00:00:00+00:00",
        status=VALIDATED_CURRENT,
    )
    if not overrides:
        return base
    data = {
        "provider_id": base.provider_id,
        "provider_name": base.provider_name,
        "model_id": base.model_id,
        "endpoint_url": base.endpoint_url,
        "hosting_jurisdiction": base.hosting_jurisdiction,
        "data_boundary": base.data_boundary,
        "allowed_regions": base.allowed_regions,
        "allowed_data_classes": base.allowed_data_classes,
        "permitted_domains": base.permitted_domains,
        "max_consequence_grade": base.max_consequence_grade,
        "supports_tools": base.supports_tools,
        "supports_structured_output": base.supports_structured_output,
        "context_window_tokens": base.context_window_tokens,
        "retention_policy": base.retention_policy,
        "validation_suite_id": base.validation_suite_id,
        "last_validated_at": base.last_validated_at,
        "status": base.status,
        "regression_result": base.regression_result,
        "report_hash": base.report_hash,
        "failed_checks": base.failed_checks,
        "observed_model_id": base.observed_model_id,
    }
    data.update(overrides)
    return ProviderProfile(**data)


def _models_ok(model_id: str = "mistral-7b") -> dict:
    return {"data": [{"id": model_id}]}


class TestOpenAiCompatibleUrls:
    def test_v1_suffix_endpoint(self):
        models, chat = openai_compatible_urls("http://localhost:8080/v1")
        assert models == "http://localhost:8080/v1/models"
        assert chat == "http://localhost:8080/v1/chat/completions"


class TestLiveProbeMockedHttp:
    def test_all_checks_pass_produces_passed_report(self):
        profile = _profile()

        def fake_get(url: str, headers: dict, timeout: float):
            assert url.endswith("/models")
            return 200, _models_ok(), None

        def fake_post(url: str, headers: dict, payload: dict, timeout: float):
            assert url.endswith("/chat/completions")
            assert payload["model"] == "mistral-7b"
            return 200, {"id": "cmpl-1"}, None

        probe, report, refreshed = run_live_provider_regression(
            profile,
            http_get=fake_get,
            http_post=fake_post,
        )
        assert probe.checks[LIVE_REGRESSION_CHECK_ENDPOINT_REACHABLE]
        assert probe.checks[LIVE_REGRESSION_CHECK_MODELS_ENDPOINT_OK]
        assert probe.checks[LIVE_REGRESSION_CHECK_DECLARED_MODEL_LISTED]
        assert probe.checks[LIVE_REGRESSION_CHECK_SMOKE_COMPLETION_OK]
        assert report.result == REGRESSION_PASSED
        assert report.failed_checks == ()
        assert report.report_hash.startswith("sha256:")
        assert refreshed.last_validated_at == probe.probed_at
        assert refreshed.regression_result == REGRESSION_PASSED

    def test_missing_model_fails_declared_model_check(self):
        profile = _profile(model_id="missing-model")

        def fake_get(url: str, headers: dict, timeout: float):
            return 200, _models_ok("other-model"), None

        def fake_post(url: str, headers: dict, payload: dict, timeout: float):
            return 200, {}, None

        _, report, _ = run_live_provider_regression(
            profile,
            http_get=fake_get,
            http_post=fake_post,
        )
        assert report.result == REGRESSION_FAILED_RESULT
        assert LIVE_REGRESSION_CHECK_DECLARED_MODEL_LISTED in report.failed_checks

    def test_transport_error_is_uncertain(self):
        profile = _profile()

        def fake_get(url: str, headers: dict, timeout: float):
            return 0, None, "connection refused"

        probe = run_live_probe(profile, http_get=fake_get, http_post=fake_get)  # type: ignore[arg-type]
        report = regression_report_from_live_probe(profile, probe)
        assert report.result == REGRESSION_UNCERTAIN_RESULT
        assert probe.transport_error == "connection refused"

    def test_refreshed_profile_feeds_existing_assessment_path(self):
        profile = _profile()

        def fake_get(url: str, headers: dict, timeout: float):
            return 200, _models_ok(), None

        def fake_post(url: str, headers: dict, payload: dict, timeout: float):
            return 500, None, None

        _, report, refreshed = run_live_provider_regression(
            profile,
            http_get=fake_get,
            http_post=fake_post,
        )
        assessment = assess_provider_validation(refreshed, declared_state=VALIDATED_CURRENT)
        assert assessment.report_hash == report.report_hash
        assert assessment.effective_state == REGRESSION_FAILED
        assert LIVE_REGRESSION_CHECK_SMOKE_COMPLETION_OK in assessment.failed_checks

    def test_runner_does_not_import_lens_or_registry_evaluation(self):
        import aurora_lens.sovereign.live_regression as mod

        source = open(mod.__file__, encoding="utf-8").read()
        assert "aurora_lens.lens" not in source
        assert "evaluate_failover" not in source


class TestLiveRegressionConfig:
    def test_smoke_completion_can_be_disabled(self):
        profile = _profile()
        config = LiveRegressionConfig(run_smoke_completion=False)

        def fake_get(url: str, headers: dict, timeout: float):
            return 200, _models_ok(), None

        def fail_post(url: str, headers: dict, payload: dict, timeout: float):
            raise AssertionError("smoke completion should be skipped")

        _, report, _ = run_live_provider_regression(
            profile,
            config,
            http_get=fake_get,
            http_post=fail_post,
        )
        assert report.result == REGRESSION_PASSED


@pytest.mark.live_provider_regression
class TestLiveProviderRegressionOptionalHttp:
    """Opt-in live HTTP — set AURORA_RUN_LIVE_PROVIDER_REGRESSION=1 and endpoint env vars."""

    @pytest.mark.skipif(
        not __import__("os").environ.get("AURORA_RUN_LIVE_PROVIDER_REGRESSION"),
        reason="opt-in live provider regression",
    )
    def test_live_endpoint_when_configured(self):
        endpoint = __import__("os").environ.get("AURORA_LIVE_REGRESSION_ENDPOINT", "").strip()
        model_id = __import__("os").environ.get("AURORA_LIVE_REGRESSION_MODEL_ID", "").strip()
        if not endpoint or not model_id:
            pytest.skip("AURORA_LIVE_REGRESSION_ENDPOINT and AURORA_LIVE_REGRESSION_MODEL_ID required")
        profile = _profile(endpoint_url=endpoint, model_id=model_id)
        _, report, _ = run_live_provider_regression(profile)
        assert report.report_hash.startswith("sha256:")
        assert report.validation_suite_id == LIVE_REGRESSION_SUITE_OPENAI_COMPATIBLE_V1

"""Live HTTP provider regression — refreshes validation evidence only.

The live runner probes real OpenAI-compatible endpoints and emits the same
``RegressionReport`` shape that Track B Phase 3C placeholder evaluation uses.
It does **not** decide admissibility; registry assessment + gates remain authoritative.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.parse import urljoin

import httpx

from aurora_lens.sovereign.provider_profile import ProviderProfile
from aurora_lens.sovereign.regression_suite import (
    REGRESSION_FAILED_RESULT,
    REGRESSION_PASSED,
    REGRESSION_UNCERTAIN_RESULT,
    RegressionReport,
    apply_regression_report_to_profile,
    compute_regression_report_hash,
)

# Minimal v1 suite: one OpenAI-compatible probe contract for sovereign profiles.
LIVE_REGRESSION_SUITE_OPENAI_COMPATIBLE_V1 = "openai_compatible_v1"

LIVE_REGRESSION_CHECK_ENDPOINT_REACHABLE = "endpoint_reachable"
LIVE_REGRESSION_CHECK_MODELS_ENDPOINT_OK = "models_endpoint_ok"
LIVE_REGRESSION_CHECK_DECLARED_MODEL_LISTED = "declared_model_listed"
LIVE_REGRESSION_CHECK_SMOKE_COMPLETION_OK = "smoke_completion_ok"

LIVE_REGRESSION_CHECKS_V1: tuple[str, ...] = (
    LIVE_REGRESSION_CHECK_ENDPOINT_REACHABLE,
    LIVE_REGRESSION_CHECK_MODELS_ENDPOINT_OK,
    LIVE_REGRESSION_CHECK_DECLARED_MODEL_LISTED,
    LIVE_REGRESSION_CHECK_SMOKE_COMPLETION_OK,
)


@dataclass(frozen=True)
class LiveRegressionConfig:
    """Runtime options for live HTTP probes."""

    api_key: str | None = None
    timeout_seconds: float = 30.0
    run_smoke_completion: bool = True
    suite_id: str = LIVE_REGRESSION_SUITE_OPENAI_COMPATIBLE_V1


@dataclass(frozen=True)
class LiveProbeResult:
    """Raw outcome of live HTTP probes for one provider profile."""

    provider_id: str
    endpoint_url: str
    probed_at: str
    suite_id: str
    checks: dict[str, bool] = field(default_factory=dict)
    models_http_status: int | None = None
    completion_http_status: int | None = None
    observed_model_ids: tuple[str, ...] = ()
    transport_error: str | None = None


HttpGetFn = Callable[[str, dict[str, str], float], tuple[int, dict[str, Any] | None, str | None]]
HttpPostFn = Callable[[str, dict[str, str], dict[str, Any], float], tuple[int, dict[str, Any] | None, str | None]]


def openai_compatible_urls(endpoint_url: str) -> tuple[str, str]:
    """Return (models_url, chat_completions_url) for a profile ``endpoint_url``."""
    base = str(endpoint_url or "").strip().rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/models", f"{base}/chat/completions"
    models = urljoin(f"{base}/", "v1/models")
    chat = urljoin(f"{base}/", "v1/chat/completions")
    return models, chat


def _auth_headers(api_key: str | None) -> dict[str, str]:
    key = str(api_key or os.environ.get("AURORA_LIVE_REGRESSION_API_KEY") or "").strip()
    if not key:
        return {}
    return {"Authorization": f"Bearer {key}"}


def _default_http_get(url: str, headers: dict[str, str], timeout: float) -> tuple[int, dict[str, Any] | None, str | None]:
    try:
        response = httpx.get(url, headers=headers, timeout=timeout)
    except httpx.RequestError as exc:
        return 0, None, str(exc)
    try:
        body = response.json()
    except json.JSONDecodeError:
        body = None
    if not isinstance(body, dict):
        body = None
    return response.status_code, body, None


def _default_http_post(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
) -> tuple[int, dict[str, Any] | None, str | None]:
    try:
        response = httpx.post(url, headers=headers, json=payload, timeout=timeout)
    except httpx.RequestError as exc:
        return 0, None, str(exc)
    try:
        body = response.json()
    except json.JSONDecodeError:
        body = None
    if not isinstance(body, dict):
        body = None
    return response.status_code, body, None


def _parse_model_ids(models_body: dict[str, Any] | None) -> tuple[str, ...]:
    if not models_body:
        return ()
    data = models_body.get("data")
    if not isinstance(data, list):
        return ()
    ids: list[str] = []
    for item in data:
        if isinstance(item, dict):
            model_id = str(item.get("id") or "").strip()
            if model_id:
                ids.append(model_id)
    return tuple(ids)


def required_checks_for_config(config: LiveRegressionConfig) -> tuple[str, ...]:
    if config.run_smoke_completion:
        return LIVE_REGRESSION_CHECKS_V1
    return tuple(
        c for c in LIVE_REGRESSION_CHECKS_V1 if c != LIVE_REGRESSION_CHECK_SMOKE_COMPLETION_OK
    )


def run_live_probe(
    profile: ProviderProfile,
    config: LiveRegressionConfig | None = None,
    *,
    http_get: HttpGetFn | None = None,
    http_post: HttpPostFn | None = None,
    probed_at: datetime | None = None,
) -> LiveProbeResult:
    """Probe a declared OpenAI-compatible provider endpoint."""
    cfg = config or LiveRegressionConfig()
    get_fn = http_get or _default_http_get
    post_fn = http_post or _default_http_post
    when = probed_at or datetime.now(timezone.utc)
    probed_at_iso = when.isoformat()
    models_url, chat_url = openai_compatible_urls(profile.endpoint_url)
    headers = _auth_headers(cfg.api_key)
    timeout = cfg.timeout_seconds

    checks: dict[str, bool] = {name: False for name in LIVE_REGRESSION_CHECKS_V1}
    transport_error: str | None = None

    models_status, models_body, models_err = get_fn(models_url, headers, timeout)
    if models_err:
        transport_error = models_err
    else:
        checks[LIVE_REGRESSION_CHECK_ENDPOINT_REACHABLE] = True
        checks[LIVE_REGRESSION_CHECK_MODELS_ENDPOINT_OK] = models_status == 200

    observed_ids = _parse_model_ids(models_body)
    declared = str(profile.model_id or "").strip()
    if declared and observed_ids:
        checks[LIVE_REGRESSION_CHECK_DECLARED_MODEL_LISTED] = declared in observed_ids
    elif declared and checks[LIVE_REGRESSION_CHECK_MODELS_ENDPOINT_OK]:
        checks[LIVE_REGRESSION_CHECK_DECLARED_MODEL_LISTED] = False
    elif not declared:
        checks[LIVE_REGRESSION_CHECK_DECLARED_MODEL_LISTED] = checks[
            LIVE_REGRESSION_CHECK_MODELS_ENDPOINT_OK
        ]

    completion_status: int | None = None
    if cfg.run_smoke_completion and checks[LIVE_REGRESSION_CHECK_ENDPOINT_REACHABLE]:
        payload = {
            "model": declared or (observed_ids[0] if observed_ids else "unknown"),
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        }
        completion_status, _, completion_err = post_fn(chat_url, headers, payload, timeout)
        if completion_err and not transport_error:
            transport_error = completion_err
        checks[LIVE_REGRESSION_CHECK_SMOKE_COMPLETION_OK] = completion_status == 200
    elif not cfg.run_smoke_completion:
        checks[LIVE_REGRESSION_CHECK_SMOKE_COMPLETION_OK] = True

    return LiveProbeResult(
        provider_id=profile.provider_id,
        endpoint_url=profile.endpoint_url,
        probed_at=probed_at_iso,
        suite_id=cfg.suite_id,
        checks=checks,
        models_http_status=models_status or None,
        completion_http_status=completion_status,
        observed_model_ids=observed_ids,
        transport_error=transport_error,
    )


def failed_checks_from_probe(
    probe: LiveProbeResult,
    *,
    config: LiveRegressionConfig | None = None,
) -> tuple[str, ...]:
    cfg = config or LiveRegressionConfig()
    required = required_checks_for_config(cfg)
    return tuple(name for name in required if not probe.checks.get(name, False))


def observed_model_id_from_probe(profile: ProviderProfile, probe: LiveProbeResult) -> str | None:
    declared = str(profile.model_id or "").strip()
    if declared and declared in probe.observed_model_ids:
        return declared
    if probe.observed_model_ids:
        return probe.observed_model_ids[0]
    return None


def regression_result_from_probe(
    probe: LiveProbeResult,
    *,
    config: LiveRegressionConfig | None = None,
) -> str:
    cfg = config or LiveRegressionConfig()
    failed = failed_checks_from_probe(probe, config=cfg)
    if probe.transport_error and not probe.checks.get(LIVE_REGRESSION_CHECK_ENDPOINT_REACHABLE):
        return REGRESSION_UNCERTAIN_RESULT
    if failed:
        return REGRESSION_FAILED_RESULT
    return REGRESSION_PASSED


def regression_report_from_live_probe(
    profile: ProviderProfile,
    probe: LiveProbeResult,
    *,
    config: LiveRegressionConfig | None = None,
) -> RegressionReport:
    """Convert live probe evidence into Track B Phase 3C ``RegressionReport`` shape."""
    cfg = config or LiveRegressionConfig()
    failed = failed_checks_from_probe(probe, config=cfg)
    result = regression_result_from_probe(probe, config=cfg)
    observed = observed_model_id_from_probe(profile, probe)
    validation_suite_id = str(profile.validation_suite_id or cfg.suite_id).strip() or cfg.suite_id
    hash_body = {
        "provider_id": profile.provider_id,
        "validation_suite_id": validation_suite_id,
        "last_validated_at": probe.probed_at,
        "result": result,
        "failed_checks": list(failed),
        "observed_model_id": observed,
        "model_id": profile.model_id,
        "live_suite_id": probe.suite_id,
        "probe_source": "live_http",
    }
    return RegressionReport(
        validation_suite_id=validation_suite_id,
        last_validated_at=probe.probed_at,
        result=result,
        report_hash=compute_regression_report_hash(hash_body),
        failed_checks=failed,
        observed_model_id=observed,
    )


def refresh_profile_validation_from_probe(
    profile: ProviderProfile,
    probe: LiveProbeResult,
    *,
    config: LiveRegressionConfig | None = None,
) -> tuple[RegressionReport, ProviderProfile]:
    """Refresh declared profile regression fields from live probe evidence."""
    report = regression_report_from_live_probe(profile, probe, config=config)
    return report, apply_regression_report_to_profile(profile, report)


def run_live_provider_regression(
    profile: ProviderProfile,
    config: LiveRegressionConfig | None = None,
    *,
    http_get: HttpGetFn | None = None,
    http_post: HttpPostFn | None = None,
) -> tuple[LiveProbeResult, RegressionReport, ProviderProfile]:
    """Run live probes and return probe evidence + report + refreshed profile."""
    probe = run_live_probe(profile, config, http_get=http_get, http_post=http_post)
    report, refreshed = refresh_profile_validation_from_probe(profile, probe, config=config)
    return probe, report, refreshed

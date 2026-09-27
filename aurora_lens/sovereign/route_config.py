"""Sovereign provider-route configuration loading and request construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aurora_lens.request_metadata import ProviderRouteRequest
from aurora_lens.sovereign.provider_profile import ProviderProfile
from aurora_lens.sovereign.provider_registry import SovereignProviderRegistry
from aurora_lens.sovereign.regression_suite import resolve_route_primary_state
from aurora_lens.sovereign.validation_freshness import ValidationFreshnessPolicy
from aurora_lens.sovereign.provider_state import VALIDATED_CURRENT


def _opt_str(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s or None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return str(v)
    return None


def _str_tuple(v: Any) -> tuple[str, ...]:
    if not isinstance(v, list):
        return ()
    out: list[str] = []
    for item in v:
        if isinstance(item, str):
            t = item.strip()
            if t:
                out.append(t)
    return tuple(out)


@dataclass(frozen=True)
class RoutePolicyLookup:
    """Task domain / consequence grade / data class for one route corridor."""

    task_domain: str
    consequence_grade: str
    data_class: str = "internal"


_DEFAULT_ROUTE_POLICY = RoutePolicyLookup(
    task_domain="general",
    consequence_grade="low",
    data_class="internal",
)


@dataclass(frozen=True)
class SovereignRouteConfig:
    """Deployment config for automatic provider-route metadata construction."""

    enabled: bool = False
    enforce_provider_route: bool = False
    primary_provider_id: str = ""
    primary_state: str = VALIDATED_CURRENT
    alternate_provider_id: str | None = None
    default_policy: RoutePolicyLookup = _DEFAULT_ROUTE_POLICY
    domain_policies: dict[str, RoutePolicyLookup] = field(default_factory=dict)
    upstream_provider_ids: dict[str, str] = field(default_factory=dict)
    profiles: tuple[ProviderProfile, ...] = ()
    validation_stale_threshold_days: int = 30


def _parse_regression_fields(raw: dict[str, Any]) -> dict[str, Any]:
    regression_raw = raw.get("regression")
    source = regression_raw if isinstance(regression_raw, dict) else raw
    failed_raw = source.get("failed_checks")
    failed_checks: tuple[str, ...] = ()
    if isinstance(failed_raw, list):
        failed_checks = tuple(str(x).strip() for x in failed_raw if str(x).strip())
    return {
        "regression_result": _opt_str(source.get("result")) or _opt_str(raw.get("regression_result")) or "",
        "report_hash": _opt_str(source.get("report_hash")) or _opt_str(raw.get("report_hash")),
        "failed_checks": failed_checks,
        "observed_model_id": _opt_str(source.get("observed_model_id")) or _opt_str(raw.get("observed_model_id")),
    }


def parse_route_policy(raw: Any, *, ctx: str) -> RoutePolicyLookup | None:
    if not isinstance(raw, dict):
        return None
    task_domain = _opt_str(raw.get("task_domain"))
    consequence_grade = _opt_str(raw.get("consequence_grade"))
    if not task_domain or not consequence_grade:
        return None
    data_class = _opt_str(raw.get("data_class")) or "internal"
    return RoutePolicyLookup(
        task_domain=task_domain,
        consequence_grade=consequence_grade,
        data_class=data_class,
    )


def parse_provider_profile(raw: Any) -> ProviderProfile | None:
    if not isinstance(raw, dict):
        return None
    provider_id = _opt_str(raw.get("provider_id"))
    provider_name = _opt_str(raw.get("provider_name"))
    model_id = _opt_str(raw.get("model_id"))
    endpoint_url = _opt_str(raw.get("endpoint_url"))
    hosting_jurisdiction = _opt_str(raw.get("hosting_jurisdiction"))
    data_boundary = _opt_str(raw.get("data_boundary"))
    if not all(
        (
            provider_id,
            provider_name,
            model_id,
            endpoint_url,
            hosting_jurisdiction,
            data_boundary,
        )
    ):
        return None
    status = _opt_str(raw.get("status")) or VALIDATED_CURRENT
    regression = _parse_regression_fields(raw)
    return ProviderProfile(
        provider_id=provider_id,
        provider_name=provider_name,
        model_id=model_id,
        endpoint_url=endpoint_url,
        hosting_jurisdiction=hosting_jurisdiction,
        data_boundary=data_boundary,
        allowed_regions=_str_tuple(raw.get("allowed_regions")),
        allowed_data_classes=_str_tuple(raw.get("allowed_data_classes"))
        or ("public", "internal", "restricted"),
        permitted_domains=_str_tuple(raw.get("permitted_domains")) or ("general",),
        max_consequence_grade=_opt_str(raw.get("max_consequence_grade")) or "low",
        supports_tools=bool(raw.get("supports_tools", False)),
        supports_structured_output=bool(raw.get("supports_structured_output", False)),
        context_window_tokens=int(raw.get("context_window_tokens", 8192) or 8192),
        retention_policy=_opt_str(raw.get("retention_policy")) or "provider_contract_required",
        validation_suite_id=_opt_str(raw.get("validation_suite_id")) or "provider_regression_v1",
        last_validated_at=_opt_str(raw.get("last_validated_at")) or "",
        status=status,
        regression_result=regression["regression_result"],
        report_hash=regression["report_hash"],
        failed_checks=regression["failed_checks"],
        observed_model_id=regression["observed_model_id"],
    )


def _parse_upstream_provider_ids(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        k = _opt_str(key)
        v = _opt_str(value)
        if k and v:
            out[k.lower()] = v
    return out


def _parse_domain_policies(raw: Any) -> dict[str, RoutePolicyLookup]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, RoutePolicyLookup] = {}
    for domain, policy_raw in raw.items():
        dom = _opt_str(domain)
        if not dom:
            continue
        policy = parse_route_policy(policy_raw, ctx=f"domain_policies.{dom}")
        if policy is not None:
            out[dom.lower()] = policy
    return out


def parse_sovereign_route_config(raw: Any) -> SovereignRouteConfig:
    """Parse a ``sovereign`` mapping from proxy YAML or programmatic config."""
    if not isinstance(raw, dict):
        return SovereignRouteConfig()

    enabled = bool(raw.get("enabled", False))
    enforce_raw = raw.get("enforce_provider_route", raw.get("enforce", False))
    if isinstance(enforce_raw, str):
        enforce_provider_route = enforce_raw.strip().lower() in ("1", "true", "yes", "on")
    else:
        enforce_provider_route = bool(enforce_raw)

    default_raw = raw.get("default") or {}
    default_policy = parse_route_policy(default_raw, ctx="sovereign.default") or _DEFAULT_ROUTE_POLICY

    profiles: list[ProviderProfile] = []
    profiles_raw = raw.get("profiles")
    if isinstance(profiles_raw, list):
        for item in profiles_raw:
            profile = parse_provider_profile(item)
            if profile is not None:
                profiles.append(profile)

    validation_raw = raw.get("validation") if isinstance(raw.get("validation"), dict) else {}
    stale_threshold = validation_raw.get("stale_threshold_days", raw.get("validation_stale_threshold_days", 30))
    try:
        validation_stale_threshold_days = max(0, int(stale_threshold))
    except (TypeError, ValueError):
        validation_stale_threshold_days = 30

    return SovereignRouteConfig(
        enabled=enabled,
        enforce_provider_route=enforce_provider_route,
        primary_provider_id=_opt_str(raw.get("primary_provider_id")) or "",
        primary_state=_opt_str(raw.get("primary_state")) or VALIDATED_CURRENT,
        alternate_provider_id=_opt_str(raw.get("alternate_provider_id")),
        default_policy=default_policy,
        domain_policies=_parse_domain_policies(raw.get("domain_policies")),
        upstream_provider_ids=_parse_upstream_provider_ids(raw.get("upstream_provider_ids")),
        profiles=tuple(profiles),
        validation_stale_threshold_days=validation_stale_threshold_days,
    )


def build_sovereign_registry(config: SovereignRouteConfig) -> SovereignProviderRegistry | None:
    if not config.enabled:
        return None
    policy = ValidationFreshnessPolicy(stale_threshold_days=config.validation_stale_threshold_days)
    return SovereignProviderRegistry(list(config.profiles), validation_policy=policy)


def validation_policy_for_config(config: SovereignRouteConfig) -> ValidationFreshnessPolicy:
    return ValidationFreshnessPolicy(stale_threshold_days=config.validation_stale_threshold_days)


def resolve_route_policy(
    config: SovereignRouteConfig,
    *,
    operator_domain: str,
) -> RoutePolicyLookup:
    domain = str(operator_domain or "").strip().lower() or "general"
    return config.domain_policies.get(domain, config.default_policy)


def resolve_primary_provider_id(
    config: SovereignRouteConfig,
    *,
    upstream_provider: str,
    upstream_model: str,
) -> str | None:
    if config.primary_provider_id:
        return config.primary_provider_id

    provider = str(upstream_provider or "").strip().lower()
    model = str(upstream_model or "").strip().lower()
    if not provider:
        return None

    composite = f"{provider}:{model}" if model else provider
    mapped = config.upstream_provider_ids.get(composite)
    if mapped:
        return mapped
    return config.upstream_provider_ids.get(provider)


def build_provider_route_request(
    config: SovereignRouteConfig,
    *,
    operator_domain: str,
    upstream_provider: str,
    upstream_model: str,
) -> ProviderRouteRequest | None:
    """Construct ``ProviderRouteRequest`` from deployment config and request context."""
    if not config.enabled:
        return None

    primary_id = resolve_primary_provider_id(
        config,
        upstream_provider=upstream_provider,
        upstream_model=upstream_model,
    )
    if not primary_id:
        return None

    policy = resolve_route_policy(config, operator_domain=operator_domain)
    primary_profile = next((p for p in config.profiles if p.provider_id == primary_id), None)
    primary_state = config.primary_state
    if primary_profile is not None:
        primary_state, _ = resolve_route_primary_state(
            primary_profile,
            declared_state=config.primary_state,
            policy=validation_policy_for_config(config),
        )
    return ProviderRouteRequest(
        primary_provider_id=primary_id,
        primary_state=primary_state,
        task_domain=policy.task_domain,
        consequence_grade=policy.consequence_grade,
        alternate_provider_id=config.alternate_provider_id,
        data_class=policy.data_class,
    )

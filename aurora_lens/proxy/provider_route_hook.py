"""Proxy auto-hook: attach sovereign ``provider_route`` to request metadata."""

from __future__ import annotations

from dataclasses import replace

from aurora_lens.proxy.config import ProxyConfig
from aurora_lens.request_metadata import RequestMetadata
from aurora_lens.sovereign.route_config import build_provider_route_request


def apply_sovereign_provider_route_hook(
    cfg: ProxyConfig,
    metadata: RequestMetadata | None,
    *,
    operator_domain: str,
    request_model: str | None,
) -> RequestMetadata | None:
    """Merge auto-constructed provider route into request metadata when configured.

    Host-supplied ``request_metadata.provider_route`` always wins. When sovereign
    routing is disabled, metadata is returned unchanged.
    """
    sovereign = cfg.sovereign
    if not sovereign.enabled:
        return metadata

    if metadata is not None and metadata.provider_route is not None:
        return metadata

    route = build_provider_route_request(
        sovereign,
        operator_domain=operator_domain,
        upstream_provider=cfg.upstream.provider,
        upstream_model=request_model or cfg.upstream.model,
    )
    if route is None:
        return metadata

    base = metadata or RequestMetadata()
    return replace(base, provider_route=route)

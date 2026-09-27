"""Host-provided request envelope (infrastructure; not user-facing).

The OpenAI-compat proxy may attach optional ``request_metadata`` on each POST body.
Governance and retrieval code can read the current turn via ``get_request_metadata()``
(see ``aurora_lens.context``). This module does **not** enforce policy; it only
parses and holds fields for later use (e.g. branching on ``policy_profile`` or
``source_scope``).

``policy_profile`` strings are mapped to governance mode (and optional strict /
moderate) via ``resolve_policy_profile_governance`` for the canonical bridge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_VALID_GOVERNANCE_MODES = frozenset({"public", "enterprise", "open"})
_VALID_POLICY_NAMES = frozenset({"strict", "moderate"})


def _opt_str(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s or None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return str(v)
    return None


def _str_tuple_from_list(v: Any) -> tuple[str, ...]:
    if not isinstance(v, list):
        return ()
    out: list[str] = []
    for x in v:
        if isinstance(x, str):
            t = x.strip()
            if t:
                out.append(t)
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            out.append(str(x))
    return tuple(out)


@dataclass(frozen=True)
class ProviderRouteRequest:
    """Host-declared provider route context for sovereign failover evaluation."""

    primary_provider_id: str
    primary_state: str
    task_domain: str
    consequence_grade: str
    alternate_provider_id: str | None = None
    data_class: str = "internal"


def _parse_provider_route(raw: Any) -> ProviderRouteRequest | None:
    if not isinstance(raw, dict):
        return None
    primary_provider_id = _opt_str(raw.get("primary_provider_id"))
    primary_state = _opt_str(raw.get("primary_state"))
    task_domain = _opt_str(raw.get("task_domain"))
    consequence_grade = _opt_str(raw.get("consequence_grade"))
    if not primary_provider_id or not primary_state or not task_domain or not consequence_grade:
        return None
    alternate_provider_id = _opt_str(raw.get("alternate_provider_id"))
    data_class = _opt_str(raw.get("data_class")) or "internal"
    return ProviderRouteRequest(
        primary_provider_id=primary_provider_id,
        primary_state=primary_state,
        task_domain=task_domain,
        consequence_grade=consequence_grade,
        alternate_provider_id=alternate_provider_id,
        data_class=data_class,
    )


@dataclass(frozen=True)
class RequestMetadata:
    """Optional host envelope for one chat completion request."""

    workspace_id: str | None = None
    source_scope: tuple[str, ...] = ()
    record_ids: tuple[str, ...] = ()
    policy_profile: str | None = None
    user_role: str | None = None
    provider_route: ProviderRouteRequest | None = None


def request_metadata_snapshot(meta: RequestMetadata | None) -> dict[str, Any] | None:
    """Return a deterministic, serializable snapshot for audit rows."""
    if meta is None:
        return None
    provider_route: dict[str, Any] | None = None
    if meta.provider_route is not None:
        provider_route = {
            "primary_provider_id": meta.provider_route.primary_provider_id,
            "primary_state": meta.provider_route.primary_state,
            "task_domain": meta.provider_route.task_domain,
            "consequence_grade": meta.provider_route.consequence_grade,
            "alternate_provider_id": meta.provider_route.alternate_provider_id,
            "data_class": meta.provider_route.data_class,
        }
    return {
        "workspace_id": meta.workspace_id,
        "source_scope": list(meta.source_scope),
        "record_ids": list(meta.record_ids),
        "policy_profile": meta.policy_profile,
        "user_role": meta.user_role,
        "provider_route": provider_route,
    }


def resolve_policy_profile_governance(
    policy_profile: str | None,
) -> tuple[str | None, str | None]:
    """Map host ``policy_profile`` to (governance_mode_override, policy_name_override).

    Returns ``(None, None)`` when absent, empty, or unrecognized -- callers keep
    deployment defaults.

    Tokens are split on ``_``, ``-``, or whitespace (case-insensitive). The first
    recognized **mode** token (``public`` | ``enterprise`` | ``open``) and the
    first **policy** token (``strict`` | ``moderate``) win. Examples:

    - ``enterprise_strict`` → (``"enterprise"``, ``"strict"``)
    - ``public`` → (``"public"``, ``None``)
    - ``strict`` → (``None``, ``"strict"``)
    """
    if policy_profile is None:
        return (None, None)
    s = str(policy_profile).strip().lower()
    if not s:
        return (None, None)
    mode: str | None = None
    policy: str | None = None
    tokens = [t for t in re.split(r"[_\s\-]+", s) if t]
    for tok in tokens:
        if tok in _VALID_GOVERNANCE_MODES and mode is None:
            mode = tok
        if tok in _VALID_POLICY_NAMES and policy is None:
            policy = tok
    if mode is None and policy is None:
        if s in _VALID_GOVERNANCE_MODES:
            return (s, None)
        if s in _VALID_POLICY_NAMES:
            return (None, s)
    return (mode, policy)


def parse_request_metadata(raw: Any) -> RequestMetadata | None:
    """Parse ``body['request_metadata']``. Returns None if absent or unusable.

    Malformed values are dropped per-field; an empty dict yields a default
    ``RequestMetadata()`` (all empty / None).
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return None
    return RequestMetadata(
        workspace_id=_opt_str(raw.get("workspace_id")),
        source_scope=_str_tuple_from_list(raw.get("source_scope")),
        record_ids=_str_tuple_from_list(raw.get("record_ids")),
        policy_profile=_opt_str(raw.get("policy_profile")),
        user_role=_opt_str(raw.get("user_role")),
        provider_route=_parse_provider_route(raw.get("provider_route")),
    )

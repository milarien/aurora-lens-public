"""Central frame/session lifecycle routing for proxy requests.

This module owns frame routing decisions so session/frame behavior is not
re-implemented across request handlers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import uuid


@dataclass(frozen=True)
class FrameLifecycleDecision:
    """Resolved frame/session routing outcome for a request."""

    session_id: str
    frame_action: str  # "continue" | "new"
    reason: str
    continuation_requested: bool | None


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    return s


def _as_optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    s = _clean_str(value).lower()
    if not s:
        return None
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return None


def mint_session_id() -> str:
    return f"session-{uuid.uuid4().hex[:12]}"


def extract_explicit_pef_context_id(
    headers: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> str:
    """Resolve the durable PEF world handle from headers/body (no cookie).

    ``pef_context_id`` is the canonical name for the continuity handle a
    client is expected to round-trip: send back the value the proxy returned
    as ``aurora.pef_context_id`` on a prior response to continue the same
    world. It is checked with top priority; the legacy ``aurora_session_id``
    / ``session_id`` aliases remain accepted so existing clients keep working,
    but new integrations should use ``pef_context_id`` explicitly.
    """
    cid = _clean_str(headers.get("x-aurora-pef-context-id"))
    if cid:
        return cid
    cid = _clean_str(payload.get("pef_context_id"))
    if cid:
        return cid
    aurora = payload.get("aurora")
    if isinstance(aurora, Mapping):
        cid = _clean_str(aurora.get("pef_context_id"))
        if cid:
            return cid
    return extract_explicit_session_id(headers, payload)


def extract_explicit_session_id(
    headers: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> str:
    """Resolve explicit session handle from headers/body only (no cookie).

    Legacy alias surface for :func:`extract_explicit_pef_context_id`. Both
    resolve to the same session-store key; this function does not check the
    ``pef_context_id`` field names so it stays usable standalone where only
    the older aliases are relevant.
    """
    sid = _clean_str(headers.get("x-aurora-session-id"))
    if sid:
        return sid
    sid = _clean_str(payload.get("aurora_session_id"))
    if sid:
        return sid
    sid = _clean_str(payload.get("session_id"))
    if sid:
        return sid
    aurora = payload.get("aurora")
    if isinstance(aurora, Mapping):
        sid = _clean_str(aurora.get("session_id"))
        if sid:
            return sid
    return ""


def parse_continuation_requested(
    headers: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> bool | None:
    """Parse explicit continuation intent for frame carry-forward."""
    v = _as_optional_bool(headers.get("x-aurora-continuation"))
    if v is not None:
        return v
    v = _as_optional_bool(payload.get("continuation"))
    if v is not None:
        return v
    aurora = payload.get("aurora")
    if isinstance(aurora, Mapping):
        v = _as_optional_bool(aurora.get("continuation"))
        if v is not None:
            return v
    return None


def parse_domain_or_lane_hint(
    headers: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> str:
    """Domain/lane hints are intentionally disabled."""
    _ = headers
    _ = payload
    return ""


def resolve_frame_lifecycle(
    *,
    explicit_session_id: str,
    cookie_session_id: str,
    continuation_requested: bool | None,
    domain_or_lane_hint: str = "",
) -> FrameLifecycleDecision:
    """Resolve request routing to an existing frame or a fresh frame.

    Priority:
      1) Explicit session handle always continues that frame.
      2) Explicit continuation=false forces a fresh frame.
      3) Explicit continuation=true uses cookie frame when present.
      4) With a domain/lane hint and no explicit continuation, default to new frame.
      5) Compatibility: cookie continues by default when continuation omitted.
      6) Otherwise mint a fresh frame/session.
    """
    if explicit_session_id:
        return FrameLifecycleDecision(
            session_id=explicit_session_id,
            frame_action="continue",
            reason="explicit_session_id",
            continuation_requested=continuation_requested,
        )
    if continuation_requested is False:
        return FrameLifecycleDecision(
            session_id=mint_session_id(),
            frame_action="new",
            reason="continuation_false",
            continuation_requested=continuation_requested,
        )
    if continuation_requested is True and cookie_session_id:
        return FrameLifecycleDecision(
            session_id=cookie_session_id,
            frame_action="continue",
            reason="explicit_continuation_cookie",
            continuation_requested=continuation_requested,
        )
    if continuation_requested is None and domain_or_lane_hint:
        return FrameLifecycleDecision(
            session_id=mint_session_id(),
            frame_action="new",
            reason="domain_lane_default_new_frame",
            continuation_requested=continuation_requested,
        )
    if continuation_requested is None and cookie_session_id:
        return FrameLifecycleDecision(
            session_id=cookie_session_id,
            frame_action="continue",
            reason="cookie_compat",
            continuation_requested=continuation_requested,
        )
    return FrameLifecycleDecision(
        session_id=mint_session_id(),
        frame_action="new",
        reason="new_frame",
        continuation_requested=continuation_requested,
    )


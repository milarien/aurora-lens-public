from __future__ import annotations

import asyncio
import hmac
import hashlib
import html
import os
import re
from dataclasses import replace
from urllib.parse import urlparse
from pathlib import Path
from typing import Any, Dict, Tuple
import datetime
import inspect
import json
import time
import uuid

from fastapi import FastAPI, Request, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.exceptions import HTTPException

from aurora_lens.context import (
    auth_label_var,
    auth_policy_var,
    chain_of_custody_runtime_provenance_var,
    domain_var,
    execution_task_var,
    get_lock_metadata,
    governance_mode_override_var,
    metadata_policy_override_var,
    mock_hard_stop_var,
    pef_context_id_var,
    request_hash_var,
    request_prompt_var,
    request_metadata_var,
    run_id_var,
    session_id_var,
    trace_id_var,
)
from aurora_lens.log_slice import init_log_buffer, log_buffer_var
from aurora_lens.lens import LensResult
from aurora_lens.proxy.config import AuthConfig, ProxyConfig
from aurora_lens.proxy.logging import get_logger
from aurora_lens.proxy.json_wire import wire_encode as _json_safe
from aurora_lens.proxy.openai_compat import (
    ParsedRequest,
    format_chat_response,
    format_stream_metadata_event,
    parse_chat_request,
)
from aurora_lens.proxy.provider_route_hook import apply_sovereign_provider_route_hook
from aurora_lens.request_metadata import resolve_policy_profile_governance
from aurora_lens.sovereign.route_config import build_sovereign_registry
from aurora_lens.pef.operator_surface import build_operator_pef_surface
from aurora_lens.proxy.frame_lifecycle import (
    extract_explicit_pef_context_id,
    parse_continuation_requested,
    resolve_frame_lifecycle,
)
from aurora_lens.proxy.session import SessionManager
from aurora_lens.proxy.session_store import SessionLockTimeoutError, SessionStoreError
from aurora_lens.govern.audit_failure_messages import (
    build_jsonl_verify_operator_message,
    build_ledger_verify_operator_message,
)
from aurora_lens.govern.audit_io import (
    append_audit_entry,
    normalize_audit_line_for_verify,
    verify_audit_entries,
    verify_chain,
    verify_co_attestation_window,
    verify_chain_of_custody_window,
    verify_forensic_state_hash_detailed,
    verify_pef_linkage_detailed,
)
from aurora_lens.govern.chain_of_custody import get_code_revision_snapshot
from aurora_lens.proxy.rate_limits import RateLimiter
from aurora_lens.proxy.metrics import get_metrics
from aurora_lens import __author__, __license__, __project__

_logger = get_logger(__name__)
# Phase 5: validation limits (chars per message for content extraction)
_MAX_CONTENT_CHARS_DEFAULT = 100_000
_TRUSTED_DOMAINS = frozenset({"general", "finance", "legal", "medical", "education", "workforce", "enterprise"})


def _effective_operator_domain(request: Request, payload: dict[str, Any]) -> str:
    """Resolve :data:`~aurora_lens.context.domain_var` for one chat completion.

    The API-key corridor (``request.state.trusted_domain``) is the default. The
    optional JSON ``aurora`` object may carry a ``request_domain`` override when
    it is one of :data:`_TRUSTED_DOMAINS`.

    ``request_domain`` is an operator/deployment hint for explicitly configured
    workflows. It is not used by the public demo question box for correctness and
    must not substitute for PEF state-native continuity.
    """
    base = (getattr(request.state, "trusted_domain", None) or "").strip().lower()
    a = payload.get("aurora")
    if isinstance(a, dict):
        rd = a.get("request_domain")
        if rd is not None:
            dom = str(rd).strip().lower()
            if dom in _TRUSTED_DOMAINS:
                return dom
    if base in _TRUSTED_DOMAINS:
        return base
    return "general"


_MODELS_LIST_CREATED = 1700000000
# Do not use naive substring: dashboard JS also contains the selector `meta[name="aurora-lens-api-base"]`.
_FORENSICS_META_IN_HTML = re.compile(
    r"<meta[^>]+name=[\"']aurora-lens-api-base[\"'][^>]*>",
    re.IGNORECASE,
)


def _client_public_origin(request: Request) -> str:
    """Browser-facing API origin for forensics fetches (Railway, reverse proxies, static demos).

    Prefer ``AURORA_LENS_PUBLIC_ORIGIN`` (full URL) or ``RAILWAY_PUBLIC_DOMAIN`` (host) when
    ``Host`` / ``X-Forwarded-*`` are internal or missing.
    """
    env_raw = (os.environ.get("AURORA_LENS_PUBLIC_ORIGIN") or "").strip()
    if not env_raw:
        rpd = (os.environ.get("RAILWAY_PUBLIC_DOMAIN") or "").strip()
        if rpd:
            env_raw = f"https://{rpd}" if "://" not in rpd else rpd
    if env_raw:
        try:
            u = urlparse(env_raw)
            if u.scheme in ("http", "https") and u.netloc:
                return f"{u.scheme}://{u.netloc}"
        except Exception:
            pass
    xfh = (request.headers.get("x-forwarded-host") or "").split(",")[0].strip()
    host = xfh or (request.headers.get("host") or "").split(",")[0].strip()
    xf_proto = (request.headers.get("x-forwarded-proto") or "").split(",")[0].strip().lower()
    if xf_proto not in ("http", "https"):
        xf_proto = (request.url.scheme or "https") if (request.url.scheme in ("http", "https")) else "https"
    if host:
        return f"{xf_proto}://{host}"
    u = request.url
    return f"{u.scheme}://{u.netloc}"


def _template_html_from_request(request: Request, filename: str) -> str:
    """HTML template with ``aurora-lens-api-base`` meta so UI targets this deploy."""
    p = Path(__file__).parent / filename
    if not p.exists():
        return f"<p>Template not found: {html.escape(filename)}</p>"
    body = p.read_text(encoding="utf-8")
    if _FORENSICS_META_IN_HTML.search(body):
        return body
    origin = _client_public_origin(request)
    tag = f'<meta name="aurora-lens-api-base" content="{html.escape(origin, quote=True)}">'
    if "<head>" in body:
        return body.replace("<head>", f"<head>\n  {tag}", 1)
    return body


def _forensics_html_from_request(request: Request) -> str:
    """``dashboard.html`` with ``<meta name="aurora-lens-api-base">`` so the UI targets this deploy."""
    return _template_html_from_request(request, "dashboard.html")


def _workflow_demo_html_from_request(request: Request) -> str:
    """``demo.html`` workflow UI with injected API base meta tag."""
    return _template_html_from_request(request, "demo.html")


def _chat_runtime_provenance_for_audit(cfg: ProxyConfig, parsed: ParsedRequest) -> tuple[str, str] | None:
    """Resolved upstream (provider, model_id) for this chat request.

    Used as :data:`chain_of_custody_runtime_provenance_var` so ``build_chain_of_custody_bundle``
    records the actual configured provider and the model from the request body when present,
    else the configured default model. Returns ``None`` if either side is empty so the
    bundle builder falls back to ``AURORA_PROVIDER`` / ``AURORA_MODEL_ID``.
    """
    rp = (cfg.upstream.provider or "").strip()
    rm = (parsed.model or "").strip() or (cfg.upstream.model or "").strip()
    if rp and rm:
        return (rp, rm)
    return None


def _construct(cls: type, **kwargs: Any) -> Any:
    sig = inspect.signature(cls.__init__)
    allowed = set(sig.parameters.keys())
    allowed.discard("self")
    filtered = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
    return cls(**filtered)


def _build_provider_adapters(cfg: ProxyConfig) -> Tuple[Any, Any]:
    if cfg.upstream.provider == "mock":
        from aurora_lens.adapters.mock_upstream import MockUpstreamAdapter  # type: ignore

        upstream = _construct(
            MockUpstreamAdapter,
            model=cfg.upstream.model,
        )
        extraction = _construct(
            MockUpstreamAdapter,
            model=cfg.upstream.model,
        )
        _logger.info(
            "adapter_instantiated",
            extra={
                "provider": "mock",
                "upstream_type": type(upstream).__name__,
                "extraction_type": type(extraction).__name__,
                "model": cfg.upstream.model,
                "api_key_set": False,
            },
        )
        return upstream, extraction

    common_kwargs = dict(
        api_key=cfg.upstream.api_key,
        model=cfg.upstream.model,
        base_url=cfg.upstream.base_url,  # filtered out if unsupported
        timeout_s=cfg.upstream.timeout_s,
    )

    if cfg.upstream.provider == "anthropic":
        from aurora_lens.adapters.claude import ClaudeAdapter  # type: ignore
        upstream = _construct(ClaudeAdapter, **common_kwargs)
        extraction = _construct(ClaudeAdapter, **common_kwargs)
        _logger.info(
            "adapter_instantiated",
            extra={
                "provider": "anthropic",
                "upstream_type": type(upstream).__name__,
                "extraction_type": type(extraction).__name__,
                "model": cfg.upstream.model,
                "api_key_set": bool(cfg.upstream.api_key and cfg.upstream.api_key.strip()),
            },
        )
        return upstream, extraction

    from aurora_lens.adapters.openai import OpenAIUpstreamAdapter  # type: ignore
    upstream = _construct(OpenAIUpstreamAdapter, **common_kwargs)
    extraction = _construct(OpenAIUpstreamAdapter, **common_kwargs)
    _logger.info(
        "adapter_instantiated",
        extra={
            "provider": cfg.upstream.provider,
            "upstream_type": type(upstream).__name__,
            "extraction_type": type(extraction).__name__,
            "model": cfg.upstream.model,
            "base_url": cfg.upstream.base_url,
            "api_key_set": bool(cfg.upstream.api_key and cfg.upstream.api_key.strip()),
        },
    )
    return upstream, extraction


# L.2: Canned non-compliant response for --mock-hard-stop demo (proof bundle).
_MOCK_HARD_STOP_RESPONSE = (
    "For an 18 kg child with ear infection, amoxicillin is typically "
    "dosed at 720-810 mg per day divided into 360-405 mg twice daily. "
    "However, always consult a healthcare provider."
)


def _wrap_adapter_for_mock_hard_stop(adapter: Any) -> Any:
    """Wrap adapter to return canned bad content when mock_hard_stop_var is set."""
    from aurora_lens.adapters.base import LLMAdapter, AdapterResponse

    class _MockHardStopWrapper(LLMAdapter):
        def __init__(self, real: Any):
            self._real = real

        async def generate(self, messages: list, **kwargs) -> Any:
            if mock_hard_stop_var.get():
                return AdapterResponse(
                    text=_MOCK_HARD_STOP_RESPONSE,
                    model="mock_hard_stop",
                    usage={"input_tokens": 0, "output_tokens": 0},
                )
            return await self._real.generate(messages, **kwargs)

        async def generate_stream(self, messages: list, **kwargs):
            if mock_hard_stop_var.get():
                chunk = {"choices": [{"delta": {"content": _MOCK_HARD_STOP_RESPONSE}, "index": 0}]}
                yield (chunk, _MOCK_HARD_STOP_RESPONSE)
                return
            async for item in self._real.generate_stream(messages, **kwargs):
                yield item

    return _MockHardStopWrapper(adapter)


def _validate_payload(
    payload: Dict[str, Any],
    max_messages: int,
    max_content_chars: int,
) -> None:
    """Phase 5: Validate payload structure. Raises ValueError on violation."""
    messages = payload.get("messages")
    if isinstance(messages, list) and len(messages) > max_messages:
        raise ValueError(
            f"Too many messages: {len(messages)} (max {max_messages})"
        )
    if not isinstance(messages, list):
        for key in ("input", "prompt", "text"):
            v = payload.get(key)
            if isinstance(v, str) and len(v) > max_content_chars:
                raise ValueError(
                    f"Content too long: {len(v)} chars (max {max_content_chars})"
                )
        return
    total = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += len(str(part.get("text", "")))
    if total > max_content_chars:
        raise ValueError(
            f"Total content too long: {total} chars (max {max_content_chars})"
        )


def _extract_session_id(request: Request, payload: Dict[str, Any]) -> str:
    """Resolve session routing through centralized frame lifecycle logic.

    Checks ``pef_context_id`` (the durable world handle) with top priority,
    then falls back to the legacy ``aurora_session_id`` / ``session_id``
    aliases — both resolve to the same session-store key.
    """
    explicit_sid = extract_explicit_pef_context_id(request.headers, payload)
    continuation_requested = parse_continuation_requested(request.headers, payload)
    # Domain/lane hints are intentionally disabled for frame routing.
    domain_or_lane_hint = ""
    cookie_sid = request.cookies.get("aurora-session", "").strip()
    decision = resolve_frame_lifecycle(
        explicit_session_id=explicit_sid,
        cookie_session_id=cookie_sid,
        continuation_requested=continuation_requested,
        domain_or_lane_hint=domain_or_lane_hint,
    )
    return decision.session_id


def _check_audit_writable(audit_path: str | None) -> bool | None:
    """Check if audit log path is appendable. None = not configured.

    Performs a real append-open (no write). On Windows, this catches file locks,
    ACL quirks, and antivirus/indexer locks that permission checks miss.
    """
    if not audit_path or not str(audit_path).strip():
        return None
    p = Path(audit_path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8"):
            pass
        return True
    except OSError:
        return False


def _proxy_audit_target_snapshot(
    audit_log: str | None,
    *,
    audit_sink: str,
    proxy_run_id: str,
    audit_signing_configured: bool,
) -> dict[str, Any]:
    """Resolved audit target fields shared by startup logging and ``GET /health``.

    ``audit_sink`` is always present; ``proxy_run_id``, ``audit_log_path``, and
    ``audit_signing_configured`` are included only when ``audit_log`` is configured.
    """
    out: dict[str, Any] = {"audit_sink": audit_sink}
    if audit_log:
        out["proxy_run_id"] = proxy_run_id
        _alp = Path(audit_log)
        try:
            out["audit_log_path"] = str(_alp.resolve())
        except OSError:
            out["audit_log_path"] = str(_alp)
        out["audit_signing_configured"] = audit_signing_configured
    return out


def _read_audit_tail(path: Path, n: int) -> list[dict[str, Any]]:
    """Read last n non-empty lines from audit file as JSON objects.

    Under concurrent writes, the final line may be partially written. Unparseable
    lines (including mid-flight partials) are skipped; valid entries are still
    returned. Never raises—endpoint stays up under real load.
    """
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    entries: list[dict[str, Any]] = []
    for ln in lines[-n:]:
        try:
            entries.append(json.loads(ln))
        except Exception:
            continue
    return entries


def _read_audit_all(path: Path, max_entries: int = 5000) -> list[dict[str, Any]]:
    """Read up to max_entries from audit file (last entries first). For search."""
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    entries: list[dict[str, Any]] = []
    for ln in lines[-max_entries:]:
        try:
            entries.append(json.loads(ln))
        except Exception:
            continue
    return entries


def _ledger_payload_flag_names(data: dict[str, Any]) -> list[str]:
    """Extract flag type names from AFL governance payload.data.flags."""
    flags = data.get("flags")
    if not isinstance(flags, list):
        return []
    out: list[str] = []
    for item in flags:
        if isinstance(item, dict):
            t = item.get("type")
            if t:
                out.append(str(t))
    return out


def _operator_forensic_extras(entry: dict[str, Any], sink: str) -> dict[str, Any]:
    """Surface provenance + envelope fields already on the row (flat JSONL or AFL).

    Three distinct evidence-state fields are surfaced here, and they answer
    three different questions — they must not be conflated:

    - ``request_capture_status``: how *this request's* content was actually
      captured by the evidence vault (e.g. ``sealed_by_policy``, ``hash_only``,
      ``omitted``) — set by :func:`aurora_lens.govern.evidence_capture.attach_evidence_fields_to_audit_entry`.
    - ``evidence_capture_mode``: the *configured* vault capture mode that
      produced that status (``sealed`` | ``redacted`` | ``hash_only`` |
      ``plaintext_dev``), i.e. the policy, not the outcome.
    - ``evidence_audit_status``: whether the chain-of-custody *packaging*
      around the row is complete or degraded (from ``chain_of_custody``).
    - ``provenance_status``: whether the governing *instrument* (policy /
      ruleset / instrument id) is identified (``complete``) or missing
      (``degraded``) — a property of the governor, not the evidence vault.
    """
    fe: dict[str, Any] = {}
    coc: Any = None
    base: dict[str, Any] = entry
    if sink == "ledger" and entry.get("kind") == "aurora.event":
        data = (entry.get("payload") or {}).get("data")
        if isinstance(data, dict):
            base = data
            fr = data.get("forensic_event")
            if isinstance(fr, dict):
                fe = fr
            coc = data.get("chain_of_custody")
    else:
        fr = entry.get("forensic_event")
        if isinstance(fr, dict):
            fe = fr
        coc = entry.get("chain_of_custody")
    out: dict[str, Any] = {
        "has_chain_of_custody": isinstance(coc, dict),
        "evidence_audit_status": None,
        "commitment_closed": None,
        "provenance_status": base.get("provenance_status"),
        "request_capture_status": base.get("request_capture_status"),
        "evidence_capture_mode": base.get("evidence_capture_mode"),
    }
    if isinstance(coc, dict):
        st = coc.get("evidence_audit_status")
        if st in ("complete", "degraded"):
            out["evidence_audit_status"] = st
    if isinstance(fe, dict) and "commitment_closed" in fe:
        out["commitment_closed"] = fe.get("commitment_closed")
    provider_route = entry.get("provider_route")
    if not isinstance(provider_route, dict) and sink == "ledger" and entry.get("kind") == "aurora.event":
        data = (entry.get("payload") or {}).get("data")
        if isinstance(data, dict) and isinstance(data.get("provider_route"), dict):
            provider_route = data["provider_route"]
    if isinstance(provider_route, dict):
        out["has_provider_route"] = True
        out["provider_route_selected_route"] = provider_route.get("selected_route")
        out["provider_route_failover_result"] = provider_route.get("failover_result")
        out["provider_route_adapter_called"] = provider_route.get("adapter_called")
    else:
        out["has_provider_route"] = False
    return out


def _operator_pef_linkage_fields(entry: dict[str, Any]) -> dict[str, Any]:
    """PEF turn classification + hold transition from the stored row (flat or AFL ``payload.data``).

    Uses :func:`normalize_audit_line_for_verify` so flattening matches verify/replay.
    """
    row = normalize_audit_line_for_verify(entry)
    return {
        "pef_turn_classification": row.get("pef_turn_classification"),
        "pef_hold_transition": row.get("pef_hold_transition"),
    }


def _operator_clarification_fields(entry: dict[str, Any], sink: str) -> dict[str, Any]:
    """Surface clarification-resolution provenance on operator audit rows."""
    cr: dict[str, Any] | None = None
    if sink == "ledger" and entry.get("kind") == "aurora.event":
        data = (entry.get("payload") or {}).get("data")
        if isinstance(data, dict):
            raw = data.get("clarification_resolution")
            if isinstance(raw, dict):
                cr = raw
            elif entry.get("op") == "USER_DISAMBIGUATION" and data.get("selected_option"):
                cr = data
    else:
        raw = entry.get("clarification_resolution")
        if isinstance(raw, dict):
            cr = raw
    if not cr:
        return {}
    selected = cr.get("selected_option")
    summary = f"selected {selected}" if selected else None
    return {
        "clarification_resolution": cr,
        "disambiguation_summary": summary,
    }


def _operator_audit_row(entry: dict[str, Any], sink: str) -> dict[str, Any]:
    """Normalize one audit line for operator / forensics UIs (ledger vs flat JSONL)."""
    def _has_llm_output(obj: dict[str, Any]) -> bool:
        for k in ("original_response", "raw_response", "upstream_response", "upstream_text"):
            v = obj.get(k)
            if v is not None and str(v).strip():
                return True
        return False

    if sink == "ledger" and entry.get("kind") == "aurora.event":
        data = (entry.get("payload") or {}).get("data")
        if not isinstance(data, dict):
            data = {}
        fe_raw = data.get("forensic_event")
        fe = fe_raw if isinstance(fe_raw, dict) else {}
        constraints = _ledger_payload_flag_names(data)
        if not constraints and isinstance(fe.get("failed_constraints"), list):
            constraints = [str(x) for x in fe["failed_constraints"]]
        return {
            "row_kind": "ledger",
            "cid": entry.get("cid"),
            "timestamp": entry.get("time"),
            "trace_id": entry.get("trace"),
            "outcome": entry.get("op") or data.get("action"),
            "tenant_label": data.get("consumer_label") if data.get("consumer_label") is not None else data.get("auth_label"),
            "request_domain": data.get("request_domain"),
            "has_forensic_event": bool(fe),
            "forensic_status": fe.get("status"),
            "has_llm_output": _has_llm_output(data),
            "constraints": constraints,
            "ledger_op": entry.get("op"),
            **_operator_pef_linkage_fields(entry),
            **_operator_forensic_extras(entry, sink),
            **_operator_clarification_fields(entry, sink),
        }
    fe_raw = entry.get("forensic_event")
    fe = fe_raw if isinstance(fe_raw, dict) else {}
    fc = entry.get("failed_constraints")
    if isinstance(fc, list):
        constraints = [str(x) for x in fc]
    elif isinstance(fe.get("failed_constraints"), list):
        constraints = [str(x) for x in fe["failed_constraints"]]
    else:
        constraints = []
    return {
        "row_kind": "jsonl",
        "cid": entry.get("cid"),
        "timestamp": entry.get("timestamp"),
        "trace_id": entry.get("trace_id"),
        "outcome": entry.get("outcome"),
        "tenant_label": entry.get("tenant_label") if entry.get("tenant_label") is not None else entry.get("consumer_label"),
        "request_domain": entry.get("request_domain"),
        "has_forensic_event": bool(fe),
        "forensic_status": fe.get("status"),
        "has_llm_output": _has_llm_output(entry),
        "constraints": constraints,
        "ledger_op": None,
        **_operator_pef_linkage_fields(entry),
        **_operator_forensic_extras(entry, sink),
        **_operator_clarification_fields(entry, sink),
    }


def _merge_audit_verify_semantics(
    body: Dict[str, Any],
    path: Path,
    n: int,
    audit_sink: str,
    integrity_verified: bool,
) -> None:
    """Attach provenance / PEF / state-hash replay results and backend interpretation."""
    coc_ok, coc_detail = verify_chain_of_custody_window(path, n)
    pef_det = verify_pef_linkage_detailed(path, n)
    state_det = verify_forensic_state_hash_detailed(path, n)
    fully_ok = integrity_verified and coc_ok and bool(pef_det["ok"]) and bool(state_det["ok"])
    body["fully_verified"] = fully_ok
    integrity_note = (
        "AFL-JSONL-1: `prev`/`hash` chain and optional per-line `sig` (HMAC over stored hash bytes). "
        "Not the flat D2 `cid`/`prev_cid` verifier."
        if audit_sink == "ledger"
        else (
            "Flat JSONL: D2 `cid`/`prev_cid` chain and HMAC over canonical row bytes (excluding `cid`, `prev_cid`, `hmac` for cid computation)."
        )
    )
    body["backend_interpretation"] = integrity_note
    body["what_was_verified"] = {
        "integrity": {
            "verified": integrity_verified,
            "meaning": integrity_note,
        },
        "provenance_chain_of_custody": {
            "evaluated": True,
            "ok": coc_ok,
            "counts": {
                "rows_with_block": coc_detail.get("rows_checked", 0),
                "rows_skipped_no_block": coc_detail.get("rows_skipped_no_block", 0),
                "rows_skipped_checkpoint": coc_detail.get("rows_skipped_checkpoint", 0),
                "complete": coc_detail.get("complete_count", 0),
                "degraded": coc_detail.get("degraded_count", 0),
            },
            "first_failed_line_index": coc_detail.get("first_failed_line_index"),
            "reason": coc_detail.get("reason"),
        },
        "pef_linkage": {
            "evaluated": True,
            "ok": bool(pef_det["ok"]),
            "pairs_checked": pef_det.get("pairs_checked", 0),
            "skipped": {
                "checkpoint_pairs": pef_det.get("skipped_checkpoint_pairs", 0),
                "no_classification_next_row": pef_det.get("skipped_no_classification", 0),
                "no_pef_snapshot_previous_row": pef_det.get("skipped_no_prev_snapshot", 0),
                "cross_session_pairs": pef_det.get("skipped_cross_session_pairs", 0),
                "cross_proxy_run_pairs": pef_det.get("skipped_cross_proxy_run_pairs", 0),
            },
            "first_failed_line_index": pef_det.get("first_failure_line_index"),
            "reason": pef_det.get("reason"),
        },
        "pef_state_hash_replay": {
            "evaluated": True,
            "ok": bool(state_det["ok"]),
            "entries_checked": state_det.get("entries_checked", 0),
            "skipped": {
                "checkpoint_lines": state_det.get("skipped_checkpoint_lines", 0),
                "no_forensic_event": state_det.get("skipped_no_forensic_event", 0),
                "no_state_hash_on_forensic_event": state_det.get("skipped_no_state_hash", 0),
            },
            "first_failed_line_index": state_det.get("first_failure_line_index"),
            "reason": state_det.get("reason"),
        },
    }
    prov = body["what_was_verified"]["provenance_chain_of_custody"]["counts"]
    rc = int(prov.get("rows_with_block", 0) or 0)
    c_complete = int(prov.get("complete", 0) or 0)
    c_deg = int(prov.get("degraded", 0) or 0)
    if rc == 0:
        prov_status = "none"
    elif not coc_ok:
        prov_status = "failed"
    elif c_complete > 0 and c_deg == 0:
        prov_status = "complete"
    elif c_deg > 0 and c_complete == 0:
        prov_status = "degraded"
    else:
        prov_status = "mixed"
    body["evidence_audit_status_summary"] = prov_status


def _parse_since(since: str) -> datetime.datetime | None:
    """Parse since param: ISO timestamp or 24h, 7d. Returns cutoff datetime or None."""
    since = (since or "").strip()
    if not since:
        return None
    # Relative: 24h, 7d
    if since.endswith("h"):
        try:
            hours = int(since[:-1])
            delta = datetime.timedelta(hours=hours)
        except ValueError:
            return None
    elif since.endswith("d"):
        try:
            days = int(since[:-1])
            delta = datetime.timedelta(days=days)
        except ValueError:
            return None
    else:
        delta = None
    if delta is not None:
        return datetime.datetime.now(datetime.timezone.utc) - delta
    # ISO format
    try:
        return datetime.datetime.fromisoformat(since.replace("Z", "+00:00"))
    except Exception:
        return None


def _audit_entry_outcome(entry: dict[str, Any]) -> str | None:
    """Intervention outcome: flat JSONL ``outcome``, AFL outer ``op``, or nested ``action``."""
    row = normalize_audit_line_for_verify(entry)
    o = row.get("outcome") or row.get("action")
    if o:
        return str(o)
    if entry.get("op") is not None:
        return str(entry["op"])
    return entry.get("outcome") if entry.get("outcome") is not None else None


def _audit_entry_request_domain(entry: dict[str, Any]) -> str | None:
    """Request-domain slice from operator-channel headers on ``request_domain`` rows."""
    row = normalize_audit_line_for_verify(entry)
    v = row.get("request_domain")
    if v is not None:
        return str(v)
    return None


def _audit_entry_timestamp_iso(entry: dict[str, Any]) -> str | None:
    row = normalize_audit_line_for_verify(entry)
    return entry.get("timestamp") or entry.get("time") or row.get("timestamp")


def _audit_entry_tenant_label(entry: dict[str, Any]) -> Any:
    row = normalize_audit_line_for_verify(entry)
    if entry.get("tenant_label") is not None:
        return entry.get("tenant_label")
    if row.get("tenant_label") is not None:
        return row.get("tenant_label")
    return row.get("consumer_label")


def _filter_audit_entries(
    entries: list[dict[str, Any]],
    outcome: str | None = None,
    tenant_label: str | None = None,
    since: datetime.datetime | None = None,
    limit: int = 100,
    domain: str | None = None,
    non_admit_only: bool = False,
) -> list[dict[str, Any]]:
    """Filter audit entries by outcome, tenant, request-domain slice, since, non-admit.

    When ``non_admit_only`` is true, excludes ``PASS`` and ``SOFT_CORRECT``.
    Scans from newest (end of file) backward; returns at most ``limit`` matches.
    """
    result: list[dict[str, Any]] = []
    for e in reversed(entries):
        if outcome and _audit_entry_outcome(e) != outcome:
            continue
        if tenant_label is not None and _audit_entry_tenant_label(e) != tenant_label:
            continue
        if domain is not None and _audit_entry_request_domain(e) != domain:
            continue
        if non_admit_only:
            oc = _audit_entry_outcome(e)
            if oc in ("PASS", "SOFT_CORRECT"):
                continue
        ts = _audit_entry_timestamp_iso(e)
        if since and ts:
            try:
                entry_dt = datetime.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                if entry_dt < since:
                    continue
            except Exception:
                pass
        result.append(e)
        if len(result) >= limit:
            break
    return result


def _client_ip(request: Request, trusted_proxy_ips: tuple[str, ...]) -> str:
    """Client IP from X-Forwarded-For when behind trusted proxy, else request.client."""
    if not trusted_proxy_ips:
        client = getattr(request, "client", None)
        return client[0] if client else "0.0.0.0"
    forwarded = request.headers.get("x-forwarded-for", "").strip()
    if not forwarded:
        client = getattr(request, "client", None)
        return client[0] if client else "0.0.0.0"
    # X-Forwarded-For: client, proxy1, proxy2 — leftmost is client
    ips = [ip.strip() for ip in forwarded.split(",") if ip.strip()]
    if not ips:
        client = getattr(request, "client", None)
        return client[0] if client else "0.0.0.0"
    # Rightmost IP that we trust is the one that forwarded; client is to its left
    # For simplicity: if we have trusted proxies, take leftmost (client) — operator must configure correctly
    return ips[0]


def _resolve_auth(
    auth_cfg: AuthConfig,
    auth_header: str | None,
    api_key_header: str | None,
) -> tuple[str | None, str | None, str | None, int | None]:
    """Resolve auth. Returns (label, policy_override, domain_override, error_status)."""
    if not auth_cfg.enabled:
        return None, None, None, None  # No label/policy/domain override, no error
    key = None
    if auth_header and auth_header.lower().startswith("bearer "):
        key = auth_header[7:].strip()
    if not key and api_key_header:
        key = api_key_header.strip()
    if not key:
        return None, None, None, 401
    for kc in auth_cfg.keys:
        if kc.key and kc.key == key:
            return kc.label, kc.policy, kc.domain, None
    return None, None, None, 403


def _auth_exempt(path: str) -> bool:
    """Paths that bypass auth (health checks, forensics console, read-only audit GETs).

    Read-only audit GETs stay aligned with the forensics HTML shell: same-origin fetches
    to ``/v1/audit/recent`` / ``/v1/audit/search`` / ``/v1/operator/summary`` must not 401
    when ``auth.enabled`` is true.
    ``/v1/operator/summary`` is intentionally restricted to coarse aggregate counters and
    runtime labels only (no prompts, responses, user/session identifiers, paths, or secrets).
    ``POST /v1/chat/completions`` still requires ``Authorization`` or ``x-api-key``.
    """
    return path in (
        "/health",
        "/healthz",
        "/dashboard",
        "/forensics",
        "/operator",
        "/metrics",
        "/v1/audit/anomaly-check",
        "/v1/audit/verify",
        "/v1/audit/entry",
        "/v1/audit/recent",
        "/v1/audit/search",
        "/v1/operator/summary",
        "/v1/session/operator-pef",
        "/v1/corpus/records",
        "/v1/corpus/ingest",
        "/v1/corpus/retrieve",
        "/v1/corpus/ask",
        "/v1/corpus/validate",
        "/v1/corpus/review",
        "/v1/corpus/clear",
        "/v1/corpus/scan",
        "/v1/debug/perms",
        "/v1/models",
    )


# Public demo surface proxied by aurora-lens.ai Pages Functions. Transport-only gate;
# does not change governance. Requires Railway secret AURORA_EDGE_TOKEN.
#
# The gate is a property of the hosted deployment, not of the request path.
# Railway injects RAILWAY_ENVIRONMENT_NAME (and, on older images,
# RAILWAY_ENVIRONMENT) into that service. A licensed local install does not,
# and must not 403 chat when AURORA_EDGE_TOKEN is unset.
#
# Audit GET routes are intentionally excluded: the Railway /forensics console loads
# them same-origin from the browser (no place to put the edge token), and the same
# read surface is already public via aurora-lens.ai Pages. Chat / new-scenario remain
# gated on the hosted deployment so anonymous clients cannot burn upstream model
# capacity on the Railway URL.
_HOSTED_PUBLIC_DEMO_ENVS = (
    "RAILWAY_ENVIRONMENT_NAME",
    "RAILWAY_ENVIRONMENT",
)
PUBLIC_DEMO_EDGE_PATHS = frozenset({
    "/v1/chat/completions",
    "/v1/session/new-scenario",
})

_EDGE_TOKEN_FORBIDDEN = {
    "error": {
        "message": "Forbidden",
        "type": "authentication_error",
    }
}


def _edge_token_matches(provided: str | None, expected: str) -> bool:
    """Timing-safe compare of edge tokens (length-independent via SHA-256 digests)."""
    got = (provided or "").encode("utf-8")
    want = expected.encode("utf-8")
    return hmac.compare_digest(hashlib.sha256(got).digest(), hashlib.sha256(want).digest())


def _hosted_public_demo_deployment() -> bool:
    """True when Railway injected a deployment environment name.

    That is how the hosted public demo is identified. A licensed local install
    does not have it. Path membership is not evidence of that deployment.
    """
    return any((os.environ.get(name) or "").strip() for name in _HOSTED_PUBLIC_DEMO_ENVS)


def _public_demo_edge_token_required(path: str) -> bool:
    return _hosted_public_demo_deployment() and path in PUBLIC_DEMO_EDGE_PATHS


def _provider_display(p: str | None) -> str:
    """Map config provider (vendor) to adapter identity for Aurora-Upstream header."""
    raw = "" if p is None else p
    key = raw.strip().lower()
    display = {
        "anthropic": "claude",
        "openai": "openai_compat",
    }.get(key)
    if display is not None:
        return display
    if key == "":
        return "unknown"
    return raw  # preserve original for unknowns


AURORA_HEADER_NAMES = [
    "Aurora-Outcome",
    "Aurora-Trace-Id",
    "Aurora-Audit-Sink",
    "Aurora-Audit-Id",
    "Aurora-Timestamp",
    "Aurora-Upstream",
    "Aurora-Policy",
    "Aurora-Policy-Version",
    "Aurora-Session-Id",
    "Aurora-Proxy-Ms",
]


def _aurora_headers(
    *,
    outcome: str,
    trace_id: str,
    audit_sink: str,
    provider: str,
    policy: str,
    policy_version: str,
    audit_id: str | None = None,
    session_id: str | None = None,
    proxy_ms: int | None = None,
) -> Dict[str, str]:
    """Emit standardized Aurora-* headers on every response."""
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    h: Dict[str, str] = {
        "Aurora-Outcome": outcome,
        "Aurora-Trace-Id": trace_id,
        "Aurora-Audit-Sink": audit_sink,
        "Aurora-Timestamp": ts,
        "Aurora-Upstream": _provider_display(provider),
        "Aurora-Policy": policy,
        "Aurora-Policy-Version": policy_version,
    }
    if audit_id:
        h["Aurora-Audit-Id"] = audit_id
    if session_id:
        h["Aurora-Session-Id"] = session_id
        h["Set-Cookie"] = f"aurora-session={session_id}; HttpOnly; SameSite=Strict; Path=/"
    if proxy_ms is not None:
        h["Aurora-Proxy-Ms"] = str(proxy_ms)
    return h


def _resolve_proxy_audit_log_path(raw: str | None, run_id: str) -> str | None:
    """Expand ``{run_id}``, ``{pid}``, ``{timestamp}`` in the governance audit path (once per process)."""
    if raw is None or not str(raw).strip():
        return None
    s = str(raw).strip()
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return s.replace("{run_id}", run_id).replace("{pid}", str(os.getpid())).replace("{timestamp}", ts)


def _maybe_rotate_audit_for_fresh_chain(path: Path) -> None:
    """If the audit file exists and is non-empty, rename it aside so the next open starts a new chain."""
    try:
        if path.exists() and path.stat().st_size > 0:
            ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            dest = path.parent / f"{path.name}.pre_restart.{ts}"
            path.rename(dest)
    except OSError:
        pass


def create_app(cfg: ProxyConfig) -> FastAPI:
    app = FastAPI(title="aurora-lens-proxy", version="1.0")

    upstream_adapter, extraction_adapter = _build_provider_adapters(cfg)
    if cfg.governance.enable_mock_hard_stop:
        upstream_adapter = _wrap_adapter_for_mock_hard_stop(upstream_adapter)

    from aurora_lens.config import LensConfig  # type: ignore
    from aurora_lens.interpret.llm_backend import LLMExtractionBackend

    if cfg.extraction.backend == "spacy":
        from aurora_lens.interpret.spacy_backend import SpacyBackend
        extraction_backend = SpacyBackend(model=cfg.extraction.spacy_model)  # load once, share across sessions
    else:
        extraction_backend = LLMExtractionBackend(
            adapter=extraction_adapter,
            provider=cfg.upstream.provider,
        )

    # Set authority_class_var once at startup from deployment config.
    # This is a deployment property — same value for all requests.
    # Per-request user_class_var is set in _aurora_trace_middleware below.
    from aurora_lens.context import authority_class_var as _authority_class_var
    _authority_class_var.set(cfg.governance.authority_class or "GP")

    proxy_run_id = str(uuid.uuid4())
    _gov_in = cfg.governance
    _resolved_audit = _resolve_proxy_audit_log_path(_gov_in.audit_log, proxy_run_id)
    if _gov_in.audit_log_fresh_chain_on_startup and _resolved_audit:
        _maybe_rotate_audit_for_fresh_chain(Path(_resolved_audit))
    if _resolved_audit != _gov_in.audit_log:
        cfg = replace(cfg, governance=replace(_gov_in, audit_log=_resolved_audit))

    from aurora_lens.govern.canonical_bridge import CanonicalScannerGateBridge
    _audit_extra_bytes = tuple(
        k.encode("utf-8") for k in cfg.governance.audit_signing_keys if str(k).strip()
    )
    _effective_audit_log = cfg.governance.audit_log
    try:
        bridge = CanonicalScannerGateBridge(
            mode=cfg.governance.mode,
            audit_path=_effective_audit_log,
            secret_key=cfg.governance.audit_signing_key.encode("utf-8") if cfg.governance.audit_signing_key else None,
            max_revision_attempts=cfg.governance.max_revision_attempts,
            backend=cfg.governance.audit_backend,
            default_policy=cfg.governance.default_policy,
            policy_version=cfg.governance.policy_version or "1.0",
            policy_matrix_path=cfg.governance.policy_matrix_path,
            audit_signing_keys=_audit_extra_bytes,
            proxy_run_id=proxy_run_id,
            evidence_capture_mode=cfg.governance.evidence_capture_mode,
            evidence_encryption_key=cfg.governance.evidence_encryption_key,
        )
    except OSError as _audit_err:
        # Audit path not writable (e.g. root-owned volume mount on Railway).
        # Degrade to null audit — governance enforcement is unaffected.
        _logger.warning(
            "audit_path_unwritable",
            extra={
                "audit_path": str(_effective_audit_log),
                "error": str(_audit_err),
                "degraded_to": "null",
            },
        )
        _effective_audit_log = None
        cfg = replace(cfg, governance=replace(cfg.governance, audit_log=None))
        bridge = CanonicalScannerGateBridge(
            mode=cfg.governance.mode,
            audit_path=None,
            secret_key=cfg.governance.audit_signing_key.encode("utf-8") if cfg.governance.audit_signing_key else None,
            max_revision_attempts=cfg.governance.max_revision_attempts,
            backend=cfg.governance.audit_backend,
            default_policy=cfg.governance.default_policy,
            policy_version=cfg.governance.policy_version or "1.0",
            policy_matrix_path=cfg.governance.policy_matrix_path,
            audit_signing_keys=_audit_extra_bytes,
            proxy_run_id=proxy_run_id,
            evidence_capture_mode=cfg.governance.evidence_capture_mode,
            evidence_encryption_key=cfg.governance.evidence_encryption_key,
        )
    audit_sink = cfg.governance.audit_backend if _effective_audit_log else "none"
    if _effective_audit_log and not cfg.governance.audit_signing_key:
        _logger.warning(
            "audit_unsigned",
            extra={"audit_unsigned_reason": "Audit log configured without audit_signing_key — entries are unsigned"},
        )

    _proxy_audit_snapshot = _proxy_audit_target_snapshot(
        cfg.governance.audit_log,
        audit_sink=audit_sink,
        proxy_run_id=proxy_run_id,
        audit_signing_configured=bool(cfg.governance.audit_signing_key),
    )
    app.state.proxy_audit_target = _proxy_audit_snapshot
    _logger.info(
        "proxy_audit_target",
        extra=dict(_proxy_audit_snapshot),
    )

    policy_version = cfg.governance.policy_version or "1.0"

    @app.middleware("http")
    async def _aurora_auth_middleware(request: Request, call_next):
        """Phase B: Inbound auth. 401 missing key, 403 invalid key. Health exempt."""
        if _auth_exempt(request.url.path):
            return await call_next(request)
        auth_header = request.headers.get("authorization")
        api_key_header = request.headers.get("x-api-key")
        label, policy_override, domain_override, err = _resolve_auth(
            cfg.auth, auth_header, api_key_header
        )
        if err is not None:
            tid = trace_id_var.get() or f"proxy:{uuid.uuid4().hex}"
            headers = _aurora_headers(
                outcome="ERROR",
                trace_id=tid,
                audit_sink=audit_sink,
                provider=cfg.upstream.provider,
                policy=cfg.governance.default_policy,
                policy_version=policy_version,
            )
            msg = "Missing API key" if err == 401 else "Invalid API key"
            return JSONResponse(
                status_code=err,
                content={"error": {"message": msg, "type": "authentication_error"}},
                headers=headers,
            )
        trusted_domain = (domain_override or cfg.governance.default_domain or "general").strip().lower()
        request.state.trusted_domain = trusted_domain if trusted_domain in _TRUSTED_DOMAINS else ""
        token_label = auth_label_var.set(label) if label else None
        token_policy = auth_policy_var.set(policy_override) if policy_override else None
        try:
            return await call_next(request)
        finally:
            if token_label is not None:
                auth_label_var.reset(token_label)
            if token_policy is not None:
                auth_policy_var.reset(token_policy)

    @app.middleware("http")
    async def _aurora_trace_middleware(request: Request, call_next):
        """Set trace_id, log buffer, and trusted corridor context for every request."""
        from aurora_lens.context import user_class_var as _user_class_var, domain_var as _domain_var
        tid = f"proxy:{uuid.uuid4().hex}"
        token = trace_id_var.set(tid)
        token_buf = init_log_buffer()
        # Per-request UserClass from optional HTTP header.
        # Header name is empty string when not configured → always GENERAL.
        _uc_header = cfg.governance.user_class_header
        _uc_val = request.headers.get(_uc_header) if _uc_header else None
        token_uc = _user_class_var.set(_uc_val) if _uc_val else None
        _trusted_domain = getattr(request.state, "trusted_domain", "")
        token_domain = _domain_var.set(_trusted_domain) if _trusted_domain else None
        try:
            return await call_next(request)
        finally:
            trace_id_var.reset(token)
            log_buffer_var.reset(token_buf)
            if token_uc is not None:
                _user_class_var.reset(token_uc)
            if token_domain is not None:
                _domain_var.reset(token_domain)

    @app.middleware("http")
    async def _aurora_edge_token_middleware(request: Request, call_next):
        """Transport gate for the Railway-hosted public demo.

        Inactive unless Railway has set the deployment environment name, so a
        licensed local install is not blocked when ``AURORA_EDGE_TOKEN`` is unset.
        When active, requires ``x-aurora-edge-token`` matching that secret and
        fails closed if the secret is missing. Does not log token values.
        Registered last among http middlewares so it runs before auth/route work.
        """
        path = request.url.path
        if not _public_demo_edge_token_required(path):
            return await call_next(request)
        expected = (os.environ.get("AURORA_EDGE_TOKEN") or "").strip()
        if not expected:
            return JSONResponse(status_code=403, content=_EDGE_TOKEN_FORBIDDEN)
        provided = request.headers.get("x-aurora-edge-token")
        if not _edge_token_matches(provided, expected):
            return JSONResponse(status_code=403, content=_EDGE_TOKEN_FORBIDDEN)
        return await call_next(request)

    if cfg.cors.enabled:
        from starlette.middleware.cors import CORSMiddleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cfg.cors.allow_origins),
            allow_methods=["POST", "GET", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=AURORA_HEADER_NAMES,
        )

    @app.exception_handler(HTTPException)
    async def _aurora_http_exception(request: Request, exc: HTTPException):
        """HTTPException paths carry minimal Aurora headers for traceability."""
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        headers = _aurora_headers(
            outcome="ERROR",
            trace_id=trace_id_var.get() or f"proxy:{uuid.uuid4().hex}",
            audit_sink=audit_sink,
            provider=cfg.upstream.provider,
            policy=cfg.governance.default_policy,
            policy_version=policy_version,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": detail, "type": "client_error"}},
            headers=headers,
        )

    @app.exception_handler(SessionLockTimeoutError)
    async def _session_lock_timeout(request: Request, exc: SessionLockTimeoutError):
        """Session lock acquire timeout → 409 SESSION_BUSY_TIMEOUT + forensic envelope."""
        tid = trace_id_var.get() or f"proxy:{uuid.uuid4().hex}"
        meta = get_lock_metadata() or {}
        session_id = session_id_var.get(None) or meta.get("session_id")
        headers = _aurora_headers(
            outcome="ERROR",
            trace_id=tid,
            audit_sink=audit_sink,
            provider=cfg.upstream.provider,
            policy=cfg.governance.default_policy,
            policy_version=policy_version,
            session_id=session_id,
        )
        # Subsystem audit record (ledger op session_lock_timeout or flat JSONL row)
        if cfg.governance.audit_log:
            envelope = {
                "type": "session_lock_timeout",
                "trace_id": tid,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "domain": "subsystem",
                "subdomain": "contention",
                "session_id": session_id or meta.get("session_id"),
                "lock_wait_ms": meta.get("lock_wait_ms"),
                "lock_acquired": meta.get("lock_acquired", False),
                "lock_timeout_ms": meta.get("lock_timeout_ms"),
            }
            try:
                if audit_sink == "ledger":
                    bridge.log_subsystem_audit_event(
                        op="session_lock_timeout",
                        payload=envelope,
                        trace_id=tid,
                    )
                else:
                    signing_key = (
                        cfg.governance.audit_signing_key.encode("utf-8")
                        if cfg.governance.audit_signing_key
                        else None
                    )
                    append_audit_entry(
                        cfg.governance.audit_log,
                        envelope,
                        signing_key=signing_key,
                        max_mb=cfg.governance.audit_log_max_mb,
                    )
            except OSError:
                pass
        return JSONResponse(
            status_code=409,
            content={
                "error": {
                    "message": "Session busy; lock acquire timed out",
                    "type": "server_error",
                    "code": "SESSION_BUSY_TIMEOUT",
                    "trace_id": tid,
                }
            },
            headers=headers,
        )

    @app.exception_handler(SessionStoreError)
    async def _session_store_error(request: Request, exc: SessionStoreError):
        """SessionStore backend failure → 503. Contract: deterministic mapping."""
        tid = trace_id_var.get() or f"proxy:{uuid.uuid4().hex}"
        headers = _aurora_headers(
            outcome="ERROR",
            trace_id=tid,
            audit_sink=audit_sink,
            provider=cfg.upstream.provider,
            policy=cfg.governance.default_policy,
            policy_version=policy_version,
        )
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": "Session store temporarily unavailable",
                    "type": "server_error",
                    "code": "session_store_error",
                    "trace_id": tid,
                }
            },
            headers=headers,
        )

    def _config_factory() -> LensConfig:
        # Retrieval-aware referents: per-request via request_metadata + Context:/Question:
        # (aurora_lens.rag_activation). LensConfig flag is for programmatic/tests only.
        sovereign_registry = build_sovereign_registry(cfg.sovereign)
        config = LensConfig(
            adapter=upstream_adapter,
            extraction_backend=extraction_backend,
            governance_bridge=bridge,
            audit_log_path=cfg.governance.audit_log,
            max_history_turns=cfg.extraction.history_window,
            spacy_model=cfg.extraction.spacy_model,
            max_stream_bytes=cfg.hardening.stream_max_kb * 1024 if cfg.hardening.stream_max_kb > 0 else 0,
            include_operator_detail=cfg.governance.include_operator_detail,
            rag_retrieval_aware_referents=False,
            enable_state_native_delegation=True,
            sovereign_provider_registry=sovereign_registry,
            sovereign_enforce_provider_route=(
                cfg.sovereign.enabled and cfg.sovereign.enforce_provider_route
            ),
        )
        _logger.info(
            "lens_config_created",
            extra={
                "adapter_type": type(upstream_adapter).__name__,
                "extraction_backend": type(extraction_backend).__name__ if extraction_backend else "SpacyBackend(default)",
            },
        )
        return config

    def _effective_include_operator_detail(request: Request) -> bool:
        """Whether to include operator-plane fields (e.g. aurora.original_response) in the response body.

        Global ``include_operator_detail`` always wins. Otherwise, when
        ``allow_operator_detail_via_header`` is enabled, ``X-Aurora-Operator-Detail: true``
        (or 1/yes/on) opts in per request without changing demo defaults.
        """
        if cfg.governance.include_operator_detail:
            return True
        if not cfg.governance.allow_operator_detail_via_header:
            return False
        v = (request.headers.get("x-aurora-operator-detail") or "").strip().lower()
        return v in ("1", "true", "yes", "on")

    sess_cfg = cfg.session
    sessions = SessionManager(
        config_factory=_config_factory,
        ttl_seconds=sess_cfg.ttl_seconds,
        backend=sess_cfg.backend,
        redis_url=sess_cfg.redis_url,
        lock_acquire_timeout_seconds=sess_cfg.lock_acquire_timeout_seconds,
        lock_lease_seconds=sess_cfg.lock_lease_seconds,
    )

    # Phase C.3: Background cleanup every 5 minutes (via lifespan)
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        async def _periodic_cleanup() -> None:
            while True:
                await asyncio.sleep(300)  # 5 minutes
                try:
                    sessions.cleanup_expired()
                except Exception:
                    pass

        cleanup_task = asyncio.create_task(_periodic_cleanup())
        yield
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass

    app.router.lifespan_context = _lifespan

    # Phase 5 + B.4: Rate limiters (created if limits > 0)
    h = cfg.hardening
    global_limiter = RateLimiter(h.rate_limit_global) if h.rate_limit_global > 0 else None
    session_limiter = RateLimiter(h.rate_limit_per_session) if h.rate_limit_per_session > 0 else None
    ip_limiter = RateLimiter(h.rate_limit_per_ip) if h.rate_limit_per_ip > 0 else None
    metrics = get_metrics()

    def _base_headers(trace_id: str, outcome: str = "OK", proxy_ms: int | None = None) -> Dict[str, str]:
        """Standard Aurora headers for health/audit endpoints."""
        return _aurora_headers(
            outcome=outcome,
            trace_id=trace_id,
            audit_sink=audit_sink,
            provider=cfg.upstream.provider,
            policy=cfg.governance.default_policy,
            policy_version=policy_version,
            proxy_ms=proxy_ms,
        )

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        ico = Path(__file__).parent / "logo.png"
        if ico.exists():
            return FileResponse(ico, media_type="image/png")
        return Response(status_code=204)

    @app.get("/healthz")
    async def healthz():
        """Minimal health check for k8s/liveness."""
        started = time.time()
        tid = trace_id_var.get() or f"proxy:{uuid.uuid4().hex}"
        h = _base_headers(tid, proxy_ms=int((time.time() - started) * 1000))
        return JSONResponse(content={"ok": True}, headers=h)

    def _health_code_features() -> Dict[str, bool]:
        """Runtime probe so operators can confirm deployed code supports PEF admission kwargs."""
        import inspect

        from aurora_lens.interpret import pef_updater
        from aurora_lens.interpret import revision_gate

        sig = inspect.signature(pef_updater.update_pef)
        return {
            "update_pef_kwarg_user_text": "user_text" in sig.parameters,
            "revision_gate_user_text_scan": hasattr(
                revision_gate, "user_text_conflicts_grounded_pef"
            ),
        }

    @app.get("/health")
    async def health(request: Request):
        """Full health: status, sessions, policy, audit writability."""
        started = time.time()
        tid = trace_id_var.get() or f"proxy:{uuid.uuid4().hex}"
        audit_writable = _check_audit_writable(cfg.governance.audit_log)
        # degraded = configured but broken; ok = not configured (null) or writable
        status = "ok" if (audit_writable is None or audit_writable) else "degraded"
        snap = request.app.state.proxy_audit_target
        rev = get_code_revision_snapshot()
        body: Dict[str, Any] = {
            "status": status,
            "sessions": sessions.active_count,
            "policy": cfg.governance.default_policy,
            "policy_version": policy_version,
            "audit_sink": snap["audit_sink"],
            "audit_writable": audit_writable,
            "auto_verify": True,   # LensConfig default; proxy does not override
            "auto_interpret": True,  # LensConfig default; proxy does not override
            "extraction_backend": cfg.extraction.backend,
            "code_features": _health_code_features(),
            "application_version": rev["application_version"],
            "git_commit": rev["git_commit"],
            "git_commit_source": rev["git_commit_source"],
            "git_working_tree_clean": rev["git_working_tree_clean"],
        }
        if "proxy_run_id" in snap:
            body["proxy_run_id"] = snap["proxy_run_id"]
            body["audit_log_path"] = snap["audit_log_path"]
            body["audit_signing_configured"] = snap["audit_signing_configured"]
        h = _base_headers(tid, proxy_ms=int((time.time() - started) * 1000))
        return JSONResponse(content=body, headers=h)

    @app.get("/about")
    async def about():
        return JSONResponse(
            content={
                "name": __project__,
                "author": __author__,
                "license": __license__,
                "commercial_license_required_for": (
                    "closed-source, SaaS, hosted, embedded, internal proprietary, or enterprise use"
                ),
            }
        )

    @app.get("/v1/models")
    async def list_models():
        """OpenAI-compatible model list for upstream discovery (e.g. LibreChat)."""
        started = time.time()
        tid = trace_id_var.get() or f"proxy:{uuid.uuid4().hex}"
        body: Dict[str, Any] = {
            "object": "list",
            "data": [
                {
                    "id": cfg.upstream.model,
                    "object": "model",
                    "created": _MODELS_LIST_CREATED,
                    "owned_by": "aurora-lens",
                }
            ],
        }
        h = _base_headers(tid, proxy_ms=int((time.time() - started) * 1000))
        return JSONResponse(content=body, headers=h)

    @app.get("/v1/debug/perms")
    async def debug_perms():
        """Diagnose container write permissions for common paths."""
        import os as _os
        results: Dict[str, Any] = {
            "uid": _os.getuid(),
            "gid": _os.getgid(),
            "paths": {},
        }
        for label, path_str in [
            ("audit_configured", cfg.governance.audit_log or ""),
            ("/data", "/data"),
            ("/app", "/app"),
            ("/tmp", "/tmp"),
        ]:
            if not path_str:
                results["paths"][label] = {"configured": False}
                continue
            p = Path(path_str)
            parent = p if p.is_dir() else p.parent
            info: Dict[str, Any] = {"path": str(p)}
            try:
                stat = parent.stat()
                info["exists"] = parent.exists()
                info["mode"] = oct(stat.st_mode)
                info["owner_uid"] = stat.st_uid
                info["owner_gid"] = stat.st_gid
            except OSError as e:
                info["stat_error"] = str(e)
            try:
                test_file = parent / f".aurora_write_test_{_os.getpid()}"
                test_file.write_text("ok")
                test_file.unlink()
                info["writable"] = True
            except OSError as e:
                info["writable"] = False
                info["write_error"] = str(e)
            results["paths"][label] = info
        return JSONResponse(content=results)

    @app.get("/metrics")
    async def metrics_endpoint():
        """Prometheus-compatible metrics. Phase 5."""
        return Response(
            content=metrics.to_prometheus(),
            media_type="text/plain; charset=utf-8",
        )

    @app.get("/dashboard", response_class=HTMLResponse)
    async def operator_dashboard(request: Request):
        """Forensics & operations console (legacy path; prefer /forensics)."""
        return HTMLResponse(content=_forensics_html_from_request(request))

    @app.get("/forensics", response_class=HTMLResponse)
    async def forensics_console(request: Request):
        """First-class operator UI: governance audit, chain integrity, runtime."""
        return HTMLResponse(content=_forensics_html_from_request(request))

    @app.get("/operator", response_class=HTMLResponse)
    async def operator_console_alias(request: Request):
        """Alias for the forensics console."""
        return HTMLResponse(content=_forensics_html_from_request(request))

    @app.get("/operator/console.html", response_class=HTMLResponse)
    async def operator_console_html(request: Request):
        """Operator console (forensics + corpus review)."""
        return HTMLResponse(content=_forensics_html_from_request(request))

    @app.get("/demo", response_class=HTMLResponse)
    async def workflow_demo(request: Request):
        """Governance workflow demo: route request through Aurora decision pipeline."""
        return HTMLResponse(content=_workflow_demo_html_from_request(request))

    @app.get("/v1/session/operator-pef")
    async def session_operator_pef(
        session_id: str | None = Query(
            None, min_length=1, description="aurora_session_id from chat completions (legacy alias)"
        ),
        pef_context_id: str | None = Query(
            None, min_length=1, description="pef_context_id from chat completions (preferred; same value as session_id)"
        ),
    ):
        """Return structured PEF session view for operators (same shape as aurora.operator_pef).

        Present-state only: ``binding_governance_state`` reflects what is binding now
        (derived from durable PEF when no per-turn decision is available). Audit history
        is not replayed here.

        ``pef_context_id`` and ``session_id`` are aliases for the same store key
        (see docs/FRAME_LIFECYCLE_INVARIANT.md); ``pef_context_id`` is preferred
        when both are supplied. Requires an active in-memory (or Redis) session.
        Expired or unknown IDs return 404.
        """
        started = time.time()
        tid = trace_id_var.get() or f"proxy:{uuid.uuid4().hex}"
        raw_id = (pef_context_id or session_id or "").strip()
        if not raw_id:
            return JSONResponse(
                status_code=422,
                content={
                    "error": {
                        "message": "one of pef_context_id or session_id is required",
                        "type": "invalid_request",
                    }
                },
                headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
            )
        sid = raw_id
        try:
            with sessions.with_lock(sid):
                lens = sessions.get(sid)
                if lens is None:
                    return JSONResponse(
                        status_code=404,
                        content={
                            "error": {
                                "message": "session not found or expired",
                                "type": "not_found",
                            }
                        },
                        headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
                    )
                op = build_operator_pef_surface(lens.pef, None)
        except SessionLockTimeoutError:
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": "session lock timeout",
                        "type": "service_unavailable",
                    }
                },
                headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
            )
        proxy_ms = int((time.time() - started) * 1000)
        return JSONResponse(
            content=_json_safe({"session_id": sid, "pef_context_id": sid, "operator_pef": op}),
            headers=_base_headers(tid, proxy_ms=proxy_ms),
        )

    @app.post("/v1/session/new-scenario")
    async def session_new_scenario():
        """Create an isolated fresh session namespace for demo scenario reset.

        Contract:
        - Mint a new session_id.
        - Initialize a fresh empty PEF for that id.
        - Do not mutate or delete any prior session.
        """
        started = time.time()
        tid = trace_id_var.get() or f"proxy:{uuid.uuid4().hex}"
        sid = f"session-{uuid.uuid4().hex[:12]}"
        try:
            with sessions.with_lock(sid):
                lens, _ = sessions.get_or_create(sid)
                # Persist explicit fresh state so backend stores are primed consistently.
                sessions.persist(sid, lens)
                op = build_operator_pef_surface(lens.pef, None)
        except SessionLockTimeoutError:
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": "session lock timeout",
                        "type": "service_unavailable",
                    }
                },
                headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
            )
        proxy_ms = int((time.time() - started) * 1000)
        headers = _aurora_headers(
            outcome="OK",
            trace_id=tid,
            audit_sink=audit_sink,
            provider=cfg.upstream.provider,
            policy=cfg.governance.default_policy,
            policy_version=policy_version,
            session_id=sid,
            proxy_ms=proxy_ms,
        )
        return JSONResponse(
            content=_json_safe(
                {
                    "aurora_session_id": sid,
                    "session_id": sid,
                    "operator_pef": op,
                }
            ),
            headers=headers,
        )

    def _corpus_proxy_base(request: Request) -> str:
        return _client_public_origin(request)

    def _corpus_auth_header(request: Request) -> str | None:
        auth = (request.headers.get("authorization") or "").strip()
        if auth:
            return auth
        api_key = (request.headers.get("x-api-key") or "").strip()
        if api_key:
            return f"Bearer {api_key}"
        return None

    @app.get("/v1/corpus/records")
    async def corpus_records_list(
        request: Request,
        record_ids: str | None = Query(None, description="Comma-separated record ids to scope the list"),
        source_prefix: str | None = Query(None, description="Only records whose source_path is under this prefix"),
    ):
        """List ingested corpus records for the operator console (isolated operator corpus root)."""
        from aurora_lens.corpus.operator_api import api_list_records

        started = time.time()
        tid = trace_id_var.get() or f"proxy:{uuid.uuid4().hex}"
        ids = [x.strip() for x in record_ids.split(",") if x.strip()] if record_ids else None
        try:
            body = api_list_records(record_ids=ids, source_prefix=source_prefix)
        except Exception as exc:
            return JSONResponse(
                status_code=500,
                content={"error": {"message": str(exc), "type": "corpus_error"}},
                headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
            )
        return JSONResponse(
            content=_json_safe(body),
            headers=_base_headers(tid, outcome="PASS", proxy_ms=int((time.time() - started) * 1000)),
        )

    @app.post("/v1/corpus/ingest")
    async def corpus_ingest(request: Request):
        """Ingest a file or folder path on the server (same semantics as CLI ingest)."""
        from aurora_lens.corpus.operator_api import api_ingest_path

        started = time.time()
        tid = trace_id_var.get() or f"proxy:{uuid.uuid4().hex}"
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "invalid JSON", "type": "invalid_request_error"}},
                headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
            )
        path_raw = str(payload.get("path") or payload.get("folder_path") or payload.get("file_path") or "").strip()
        if not path_raw:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "path is required", "type": "invalid_request_error"}},
                headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
            )
        try:
            body = api_ingest_path(
                path=Path(path_raw),
                record_id=str(payload.get("record_id") or "").strip() or None,
                title=str(payload.get("title") or "").strip() or None,
                force=bool(payload.get("force", False)),
                recursive=bool(payload.get("recursive", True)),
                dry_run=bool(payload.get("dry_run", False)),
            )
        except FileNotFoundError as exc:
            return JSONResponse(
                status_code=404,
                content={"error": {"message": str(exc), "type": "not_found"}},
                headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
            )
        except (ValueError, OSError) as exc:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": str(exc), "type": "invalid_request_error"}},
                headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
            )
        return JSONResponse(
            content=_json_safe(body),
            headers=_base_headers(tid, outcome="PASS", proxy_ms=int((time.time() - started) * 1000)),
        )

    @app.post("/v1/corpus/retrieve")
    async def corpus_retrieve(request: Request):
        """Retrieve evidence passages only (no Lens)."""
        from aurora_lens.corpus.operator_api import api_retrieve_evidence

        started = time.time()
        tid = trace_id_var.get() or f"proxy:{uuid.uuid4().hex}"
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "invalid JSON", "type": "invalid_request_error"}},
                headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
            )
        question = str(payload.get("question") or "").strip()
        if not question:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "question is required", "type": "invalid_request_error"}},
                headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
            )
        try:
            body = api_retrieve_evidence(
                question=question,
                record_id=str(payload.get("record_id") or "").strip() or None,
                search_all=bool(payload.get("search_all", False)),
                record_ids=[x.strip() for x in payload.get("record_ids") or [] if str(x).strip()],
                max_chars=int(payload.get("max_chars") or 12000),
            )
        except (KeyError, ValueError) as exc:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": str(exc), "type": "invalid_request_error"}},
                headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
            )
        return JSONResponse(
            content=_json_safe(body),
            headers=_base_headers(tid, outcome="PASS", proxy_ms=int((time.time() - started) * 1000)),
        )

    @app.post("/v1/corpus/ask")
    async def corpus_ask(request: Request):
        """Governed corpus Q&A — retrieval proposes, Lens disposes via chat completions."""
        from aurora_lens.corpus.operator_api import api_ask_corpus

        started = time.time()
        tid = trace_id_var.get() or f"proxy:{uuid.uuid4().hex}"
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "invalid JSON", "type": "invalid_request_error"}},
                headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
            )
        question = str(payload.get("question") or "").strip()
        if not question:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "question is required", "type": "invalid_request_error"}},
                headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
            )
        try:
            body = api_ask_corpus(
                question=question,
                record_id=str(payload.get("record_id") or "").strip() or None,
                proxy=str(payload.get("proxy") or _corpus_proxy_base(request)),
                search_all=bool(payload.get("search_all", False)),
                record_ids=[x.strip() for x in payload.get("record_ids") or [] if str(x).strip()],
                max_chars=int(payload.get("max_chars") or 12000),
                policy_profile=str(payload.get("policy_profile") or "enterprise_strict"),
                authorization=_corpus_auth_header(request),
            )
        except (KeyError, ValueError, RuntimeError) as exc:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": str(exc), "type": "corpus_error"}},
                headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
            )
        return JSONResponse(
            content=_json_safe(body),
            headers=_base_headers(tid, outcome="PASS", proxy_ms=int((time.time() - started) * 1000)),
        )

    @app.post("/v1/corpus/validate")
    async def corpus_validate(request: Request):
        """Validate one question or a manifest through governed proxy + scorer."""
        from aurora_lens.corpus.operator_api import api_validate_manifest, api_validate_question

        started = time.time()
        tid = trace_id_var.get() or f"proxy:{uuid.uuid4().hex}"
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "invalid JSON", "type": "invalid_request_error"}},
                headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
            )
        manifest_raw = str(payload.get("manifest_path") or "").strip()
        question = str(payload.get("question") or "").strip()
        record_id = str(payload.get("record_id") or "").strip()
        try:
            if manifest_raw:
                body = api_validate_manifest(
                    manifest_path=Path(manifest_raw),
                    proxy=str(payload.get("proxy") or _corpus_proxy_base(request)),
                    case_ids=str(payload.get("case_ids") or "").strip() or None,
                    authorization=_corpus_auth_header(request),
                )
            elif question and record_id:
                body = api_validate_question(
                    question=question,
                    record_id=record_id,
                    proxy=str(payload.get("proxy") or _corpus_proxy_base(request)),
                    authorization=_corpus_auth_header(request),
                )
            else:
                missing = []
                if not question:
                    missing.append("question")
                if not record_id:
                    missing.append("record_id")
                reason = "no selected record" if not record_id else "no question"
                return JSONResponse(
                    status_code=400,
                    content={
                        "review_outcome": "failed",
                        "review_failure_reason": reason,
                        "review_failure_detail": f"Missing: {', '.join(missing)}",
                        "error": {
                            "message": "provide manifest_path or (question + record_id)",
                            "type": "invalid_request_error",
                        },
                    },
                    headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
                )
        except (KeyError, ValueError, RuntimeError) as exc:
            msg = str(exc)
            reason = "server error"
            lower = msg.lower()
            if "record not found" in lower or "not in operator corpus" in lower:
                reason = "no selected record"
            elif "api key" in lower:
                reason = "no API key"
            elif "upstream" in lower or "openai_api_key" in lower or "anthropic_api_key" in lower:
                reason = "no upstream configured"
            elif "evidence" in lower or "no chunks" in lower or "no matching" in lower:
                reason = "no evidence found"
            return JSONResponse(
                status_code=400,
                content={
                    "review_outcome": "failed",
                    "review_failure_reason": reason,
                    "review_failure_detail": msg,
                    "error": {"message": msg, "type": "corpus_error"},
                },
                headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
            )
        return JSONResponse(
            content=_json_safe(body),
            headers=_base_headers(tid, outcome="PASS", proxy_ms=int((time.time() - started) * 1000)),
        )

    @app.post("/v1/corpus/review")
    async def corpus_review(request: Request):
        """Direct upstream review (Door 1 — no Lens)."""
        import logging

        from aurora_lens.corpus.operator_api import api_review_record

        log = logging.getLogger("aurora_lens.proxy.corpus")
        started = time.time()
        tid = trace_id_var.get() or f"proxy:{uuid.uuid4().hex}"
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=400,
                content={
                    "last_action": "upstream_review",
                    "selected_record_id": "",
                    "upstream_review_state": "failed",
                    "upstream_review_failure_reason": "server error",
                    "upstream_review_failure_detail": "invalid JSON",
                    "error": {"message": "invalid JSON", "type": "invalid_request_error"},
                },
                headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
            )
        record_id = str(payload.get("record_id") or "").strip()
        if not record_id:
            return JSONResponse(
                status_code=400,
                content={
                    "last_action": "upstream_review",
                    "selected_record_id": "",
                    "upstream_review_state": "failed",
                    "upstream_review_failure_reason": "no selected record",
                    "upstream_review_failure_detail": "record_id is required",
                    "error": {"message": "record_id is required", "type": "invalid_request_error"},
                },
                headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
            )
        log.info("upstream review request started record_id=%s", record_id)
        try:
            body = await api_review_record(
                record_id=record_id,
                question=str(payload.get("question") or "").strip() or None,
            )
        except KeyError as exc:
            msg = str(exc)
            log.warning("upstream review not found record_id=%s: %s", record_id, msg)
            return JSONResponse(
                status_code=404,
                content={
                    "last_action": "upstream_review",
                    "selected_record_id": record_id,
                    "upstream_review_state": "failed",
                    "upstream_review_failure_reason": "no selected record",
                    "upstream_review_failure_detail": msg,
                    "error": {"message": msg, "type": "not_found"},
                },
                headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
            )
        except (ValueError, RuntimeError) as exc:
            msg = str(exc)
            log.warning("upstream review error record_id=%s: %s", record_id, msg)
            return JSONResponse(
                status_code=400,
                content={
                    "last_action": "upstream_review",
                    "selected_record_id": record_id,
                    "upstream_review_state": "failed",
                    "upstream_review_failure_reason": "server error",
                    "upstream_review_failure_detail": msg,
                    "error": {"message": msg, "type": "corpus_error"},
                },
                headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
            )
        except SystemExit as exc:
            msg = str(exc) or "upstream not configured"
            log.warning("upstream review misconfigured record_id=%s: %s", record_id, msg)
            return JSONResponse(
                status_code=503,
                content={
                    "last_action": "upstream_review",
                    "selected_record_id": record_id,
                    "upstream_review_state": "failed",
                    "upstream_review_failure_reason": "upstream not configured",
                    "upstream_review_failure_detail": msg,
                    "error": {"message": msg, "type": "corpus_error"},
                },
                headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
            )
        state = body.get("upstream_review_state")
        log.info(
            "upstream review finished record_id=%s state=%s provider=%s",
            record_id,
            state,
            body.get("provider"),
        )
        status_code = 200 if state != "failed" else 503
        return JSONResponse(
            status_code=status_code,
            content=_json_safe(body),
            headers=_base_headers(
                tid,
                outcome="PASS" if state != "failed" else "ERROR",
                proxy_ms=int((time.time() - started) * 1000),
            ),
        )

    @app.post("/v1/corpus/scan")
    async def corpus_scan(request: Request):
        """Preview folder ingest counts and skipped files (no writes)."""
        from aurora_lens.corpus.operator_api import api_scan_folder

        started = time.time()
        tid = trace_id_var.get() or f"proxy:{uuid.uuid4().hex}"
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "invalid JSON", "type": "invalid_request_error"}},
                headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
            )
        path_raw = str(payload.get("path") or payload.get("folder_path") or "").strip()
        if not path_raw:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "path is required", "type": "invalid_request_error"}},
                headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
            )
        try:
            body = api_scan_folder(
                path=Path(path_raw),
                recursive=bool(payload.get("recursive", True)),
            )
        except FileNotFoundError as exc:
            return JSONResponse(
                status_code=404,
                content={"error": {"message": str(exc), "type": "not_found"}},
                headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
            )
        return JSONResponse(
            content=_json_safe(body),
            headers=_base_headers(tid, outcome="PASS", proxy_ms=int((time.time() - started) * 1000)),
        )

    @app.post("/v1/corpus/clear")
    async def corpus_clear(request: Request):
        """Clear the isolated operator corpus registry (not the shared CLI ``data/corpus``)."""
        from aurora_lens.corpus.operator_api import api_clear_operator_corpus

        started = time.time()
        tid = trace_id_var.get() or f"proxy:{uuid.uuid4().hex}"
        try:
            body = api_clear_operator_corpus()
        except ValueError as exc:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": str(exc), "type": "invalid_request_error"}},
                headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
            )
        return JSONResponse(
            content=_json_safe(body),
            headers=_base_headers(tid, outcome="PASS", proxy_ms=int((time.time() - started) * 1000)),
        )

    @app.get("/v1/audit/recent")
    async def audit_recent(
        n: int = Query(20, ge=1, le=100),
        outcome: str | None = Query(
            None,
            description="Optional filter: outcome (matches flat JSONL, AFL op, or nested action)",
        ),
        domain: str | None = Query(
            None,
            description="Optional filter: request_domain (from operator-channel header metadata at write time)",
        ),
        non_admit_only: bool = Query(
            False,
            description="If true, exclude PASS and SOFT_CORRECT rows",
        ),
    ):
        """Return last n audit entries. Requires audit_log configured.

        Without ``outcome`` / ``domain`` / ``non_admit_only``, returns the last ``n`` lines
        (tail). With filters, scans up to 50k trailing lines and returns the last ``n``
        rows matching all predicates (chronological order preserved).
        """
        started = time.time()
        tid = trace_id_var.get() or f"proxy:{uuid.uuid4().hex}"

        if not cfg.governance.audit_log or not str(cfg.governance.audit_log).strip():
            return JSONResponse(
                status_code=404,
                content={"error": {"message": "audit_log not configured", "type": "not_found"}},
                headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
            )
        path = Path(cfg.governance.audit_log)
        has_filter = outcome is not None or domain is not None or non_admit_only
        if not has_filter:
            entries = _read_audit_tail(path, n)
        else:
            all_entries = _read_audit_all(path, max_entries=50000)
            filtered: list[dict[str, Any]] = []
            for e in all_entries:
                if outcome and _audit_entry_outcome(e) != outcome:
                    continue
                if domain is not None and _audit_entry_request_domain(e) != domain:
                    continue
                if non_admit_only:
                    oc = _audit_entry_outcome(e)
                    if oc in ("PASS", "SOFT_CORRECT"):
                        continue
                filtered.append(e)
            entries = filtered[-n:]
        safe_entries = [_json_safe(e) for e in entries]
        body: Dict[str, Any] = {
            "entries": safe_entries,
            "n": len(entries),
            "backend": audit_sink,
            "operator_rows": [_json_safe(_operator_audit_row(e, audit_sink)) for e in entries],
        }
        proxy_ms = int((time.time() - started) * 1000)
        return JSONResponse(
            content=body,
            headers=_base_headers(tid, proxy_ms=proxy_ms),
        )

    @app.get("/v1/operator/summary")
    async def operator_summary(n: int = Query(30, ge=1, le=200)):
        """Compact operator summary for first-screen dashboard tiles.

        This endpoint is read-only and intentionally coarse-grained. It exposes
        aggregate counters and runtime labels only, with no raw prompt/response
        content, no file paths, and no per-session/user identifiers.
        """
        started = time.time()
        tid = trace_id_var.get() or f"proxy:{uuid.uuid4().hex}"

        audit_writable = _check_audit_writable(cfg.governance.audit_log)
        system_status = "ok" if (audit_writable is None or audit_writable) else "degraded"
        entries: list[dict[str, Any]] = []
        if cfg.governance.audit_log and str(cfg.governance.audit_log).strip():
            entries = _read_audit_tail(Path(cfg.governance.audit_log), n)
        rows = [_operator_audit_row(e, audit_sink) for e in entries]

        warning_outcomes = {"SOFT_CORRECT", "FORCE_REVISE", "CLARIFY"}
        block_outcomes = {"HARD_STOP", "STOP", "CONTAIN", "REFUSE"}
        warning_count = 0
        block_count = 0
        for row in rows:
            outcome = str(row.get("outcome") or row.get("ledger_op") or "").strip().upper()
            if outcome in warning_outcomes:
                warning_count += 1
            if outcome in block_outcomes:
                block_count += 1

        snap = app.state.proxy_audit_target
        body: Dict[str, Any] = {
            "system_status": system_status,
            "sessions": sessions.active_count,
            "model": cfg.upstream.model,
            "policy": cfg.governance.default_policy,
            "policy_version": policy_version,
            "audit_sink": audit_sink,
            "decision_window_n": n,
            "audit_data_available": bool(cfg.governance.audit_log and str(cfg.governance.audit_log).strip()),
            "recent_decisions": len(rows),
            "warning_count": warning_count,
            "block_count": block_count,
            "audit_writable": audit_writable,
            "proxy_run_id": snap.get("proxy_run_id"),
        }
        proxy_ms = int((time.time() - started) * 1000)
        return JSONResponse(
            content=_json_safe(body),
            headers=_base_headers(tid, proxy_ms=proxy_ms),
        )

    def _read_audit_entry_by_cid(path: Path, cid: str) -> dict[str, Any] | None:
        """Find audit entry by cid. JSONL backend. Returns None if not found."""
        if not path.exists() or not cid or not str(cid).strip():
            return None
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return None
        for ln in text.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                entry = json.loads(ln)
                if entry.get("cid") == cid:
                    return entry
            except Exception:
                continue
        return None

    def _read_audit_entry_by_trace(
        path: Path, trace_id: str, *, timestamp: str | None = None
    ) -> dict[str, Any] | None:
        """Find audit entry by ``trace_id`` (newest match when scanning from end of file).

        Optional ``timestamp`` must match the row's ``timestamp`` field exactly (same
        string as stored) to disambiguate duplicate trace ids.
        """
        if not path.exists() or not trace_id or not str(trace_id).strip():
            return None
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return None
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        want_ts = (timestamp or "").strip() or None
        for ln in reversed(lines):
            try:
                entry = json.loads(ln)
            except Exception:
                continue
            if entry.get("trace_id") != trace_id:
                continue
            if want_ts is not None and entry.get("timestamp") != want_ts:
                continue
            return entry
        return None

    @app.get("/v1/audit/entry")
    async def audit_entry(
        cid: str | None = Query(None, description="Chain ID (D2); preferred when present on the row"),
        trace_id: str | None = Query(
            None,
            description="Fallback when rows lack cid (legacy logs): newest line with this trace_id",
        ),
        timestamp: str | None = Query(
            None,
            description="Exact row timestamp string; use with trace_id when ids repeat",
        ),
    ):
        """Return full audit JSONL line by cid, or by trace_id (+ optional timestamp). JSONL only."""
        started = time.time()
        tid = trace_id_var.get() or f"proxy:{uuid.uuid4().hex}"

        if not cfg.governance.audit_log or not str(cfg.governance.audit_log).strip():
            return JSONResponse(
                status_code=404,
                content={"error": {"message": "audit_log not configured", "type": "not_found"}},
                headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
            )
        if audit_sink == "ledger":
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "Entry lookup by cid supported for JSONL backend only", "type": "invalid_request_error"}},
                headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
            )
        path = Path(cfg.governance.audit_log)
        c = (cid or "").strip()
        t = (trace_id or "").strip()
        ts = (timestamp or "").strip() or None
        if not c and not t:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "cid or trace_id is required",
                        "type": "invalid_request_error",
                        "user_message": "Provide ?cid=… or ?trace_id=… (optional &timestamp=… for disambiguation).",
                    }
                },
                headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
            )
        if c:
            entry = _read_audit_entry_by_cid(path, c)
            if entry is None:
                return JSONResponse(
                    status_code=404,
                    content={"error": {"message": f"No audit entry with cid={c}", "type": "not_found"}},
                    headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
                )
        else:
            entry = _read_audit_entry_by_trace(path, t, timestamp=ts)
            if entry is None:
                return JSONResponse(
                    status_code=404,
                    content={
                        "error": {
                            "message": f"No audit entry with trace_id={t!r}"
                            + (f" and timestamp={ts!r}" if ts else ""),
                            "type": "not_found",
                        }
                    },
                    headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
                )
        proxy_ms = int((time.time() - started) * 1000)
        return JSONResponse(
            content=_json_safe(entry),
            headers=_base_headers(tid, proxy_ms=proxy_ms),
        )

    @app.get("/v1/audit/search")
    async def audit_search(
        outcome: str | None = Query(None, description="Filter by outcome: PASS, SOFT_CORRECT, FORCE_REVISE, HARD_STOP"),
        tenant_label: str | None = Query(None, description="Filter by tenant_label"),
        domain: str | None = Query(
            None,
            description="Filter by request_domain (from operator-channel header metadata at write time)",
        ),
        non_admit_only: bool = Query(False, description="Exclude PASS and SOFT_CORRECT"),
        since: str | None = Query(None, description="Entries since: ISO timestamp or 24h, 7d"),
        limit: int = Query(100, ge=1, le=500),
    ):
        """Search audit log by outcome, tenant, request-domain slice, time range, non-admit. JSONL backend only."""
        started = time.time()
        tid = trace_id_var.get() or f"proxy:{uuid.uuid4().hex}"

        if not cfg.governance.audit_log or not str(cfg.governance.audit_log).strip():
            return JSONResponse(
                status_code=404,
                content={"error": {"message": "audit_log not configured", "type": "not_found"}},
                headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
            )
        if audit_sink == "ledger":
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "Search supported for JSONL backend only", "type": "invalid_request_error"}},
                headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
            )
        path = Path(cfg.governance.audit_log)
        all_entries = _read_audit_all(path)
        since_dt = _parse_since(since) if since else None
        filtered = _filter_audit_entries(
            all_entries,
            outcome=outcome,
            tenant_label=tenant_label,
            since=since_dt,
            limit=limit,
            domain=domain,
            non_admit_only=non_admit_only,
        )
        body = {"entries": [_json_safe(e) for e in filtered], "count": len(filtered)}
        proxy_ms = int((time.time() - started) * 1000)
        return JSONResponse(
            content=body,
            headers=_base_headers(tid, proxy_ms=proxy_ms),
        )

    def _compute_anomaly_rates(entries: list[dict[str, Any]]) -> tuple[float, float, int]:
        """Compute intervention_rate and extraction_failure_rate from decision entries.

        Skips checkpoint entries. Returns (intervention_rate, extraction_failure_rate, total).
        """
        total = 0
        interventions = 0
        extraction_failures = 0
        for e in entries:
            if e.get("type") == "checkpoint":
                continue
            outcome = e.get("outcome")
            if outcome is None:
                continue
            total += 1
            if outcome in ("FORCE_REVISE", "HARD_STOP"):
                interventions += 1
            if outcome == "HARD_STOP":
                fe = e.get("forensic_event") or {}
                failed = fe.get("failed_constraints") or []
                if isinstance(failed, list) and "EXTRACTION_FAILED" in failed:
                    extraction_failures += 1
        if total == 0:
            return 0.0, 0.0, 0
        return interventions / total, extraction_failures / total, total

    @app.get("/v1/audit/anomaly-check")
    async def audit_anomaly_check(
        window: str = Query("24h", description="Window: 24h, 7d, or ISO timestamp"),
    ):
        """P.3: Check intervention/extraction-failure rates against thresholds. JSONL backend only."""
        started = time.time()
        tid = trace_id_var.get() or f"proxy:{uuid.uuid4().hex}"

        if not cfg.governance.audit_log or not str(cfg.governance.audit_log).strip():
            return JSONResponse(
                status_code=404,
                content={"error": {"message": "audit_log not configured", "type": "not_found"}},
                headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
            )
        if audit_sink == "ledger":
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "Anomaly check supported for JSONL backend only", "type": "invalid_request_error"}},
                headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
            )
        path = Path(cfg.governance.audit_log)
        all_entries = _read_audit_all(path)
        since_dt = _parse_since(window) if window else None
        filtered = _filter_audit_entries(
            all_entries,
            outcome=None,
            tenant_label=None,
            since=since_dt,
            limit=2000,
        )
        interv_rate, extr_rate, total = _compute_anomaly_rates(filtered)
        thr_interv = cfg.governance.threshold_intervention_rate or 0
        thr_extr = cfg.governance.threshold_extraction_failure_rate or 0
        alerts: list[str] = []
        if thr_interv > 0 and interv_rate >= thr_interv:
            alerts.append(f"intervention_rate {interv_rate:.2%} >= threshold {thr_interv:.2%}")
        if thr_extr > 0 and extr_rate >= thr_extr:
            alerts.append(f"extraction_failure_rate {extr_rate:.2%} >= threshold {thr_extr:.2%}")
        if alerts:
            _logger.warning("anomaly_alert", extra={"alerts": alerts, "window": window, "total": total})
            webhook_url = cfg.governance.anomaly_webhook_url
            if webhook_url:
                payload = {"alerts": alerts, "window": window, "total": total, "intervention_rate": interv_rate, "extraction_failure_rate": extr_rate}
                async def _post_webhook() -> None:
                    try:
                        import httpx
                        async with httpx.AsyncClient(timeout=10.0) as client:
                            await client.post(webhook_url, json=payload)
                    except Exception as exc:
                        _logger.warning("anomaly_webhook_failed", extra={"url": webhook_url, "error": str(exc)})
                asyncio.create_task(_post_webhook())
        body: Dict[str, Any] = {
            "window": window,
            "total_entries": total,
            "intervention_rate": interv_rate,
            "extraction_failure_rate": extr_rate,
            "thresholds": {
                "intervention_rate": thr_interv,
                "extraction_failure_rate": thr_extr,
            },
            "alerts": alerts,
        }
        proxy_ms = int((time.time() - started) * 1000)
        return JSONResponse(
            content=body,
            headers=_base_headers(tid, proxy_ms=proxy_ms),
        )

    @app.get("/v1/audit/verify")
    async def audit_verify(n: int = Query(20, ge=1, le=1000)):
        """Verify audit chain. JSONL: HMAC + hash-chain. Ledger: hash-chain; per-line HMAC when signing key set."""
        started = time.time()
        tid = trace_id_var.get() or f"proxy:{uuid.uuid4().hex}"

        if not cfg.governance.audit_log or not str(cfg.governance.audit_log).strip():
            return JSONResponse(
                status_code=404,
                content={
                    "error": {
                        "message": "audit_log not configured",
                        "type": "not_found",
                        "user_message": (
                            "No audit log path is configured for this proxy. "
                            "Set governance.audit_log (or env override) to enable verification."
                        ),
                    }
                },
                headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
            )

        # Ledger backend: hash chain + per-line HMAC when audit_signing_key is configured
        if audit_sink == "ledger":
            if bridge is None:
                return JSONResponse(
                    status_code=503,
                    content={"error": {"message": "Ledger bridge not initialized", "type": "unavailable"}},
                    headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
                )
            if cfg.governance.audit_signing_key:
                _led_primary = cfg.governance.audit_signing_key.encode("utf-8")
                _led_extra = [k.encode("utf-8") for k in cfg.governance.audit_signing_keys if k]
                _led_all_keys = [_led_primary] + _led_extra
                led = bridge.verify_ledger_detailed(signing_keys=_led_all_keys)
            else:
                _led_all_keys = []
                led = bridge.verify_ledger_detailed()
            stats = bridge.ledger_stats or {}
            entry_count = stats.get("entries", stats.get("entry_count", 0))
            ledger_ms = int((time.time() - started) * 1000)
            body: Dict[str, Any] = {
                "verified": led.ok,
                "entries": entry_count,
                "hmac_verified": led.hmac_ok,
                "chain_verified": led.chain_ok,
                "backend": "ledger",
            }
            if led.hmac_checked:
                body["hmac_checked"] = True
            if not led.ok:
                if led.first_chain_failure_line is not None:
                    body["first_chain_failure_line"] = led.first_chain_failure_line
                    body["first_chain_reason"] = led.first_chain_reason
                    if led.first_chain_detail:
                        body["first_chain_detail"] = led.first_chain_detail
                if led.first_hmac_failure_line is not None:
                    body["first_hmac_failure_line"] = led.first_hmac_failure_line
                    body["first_hmac_reason"] = led.first_hmac_reason
                    if led.first_hmac_detail:
                        body["first_hmac_detail"] = led.first_hmac_detail
                body["verify_note"] = (
                    "Chain and HMAC are evaluated separately. A broken prev/hash at an early line "
                    "fails the chain; wrong audit_signing_key fails HMAC even if the chain links "
                    "look plausible. Align keys with the process that wrote the ledger and ensure "
                    "the file is not a mix of unsigned legacy lines and a different schema."
                )
                _om = build_ledger_verify_operator_message(led)
                if _om:
                    body["operator_message"] = _om
            _merge_audit_verify_semantics(
                body,
                Path(cfg.governance.audit_log),
                n,
                "ledger",
                bool(led.ok),
            )
            att_ok, att_detail = verify_co_attestation_window(
                cfg.governance.audit_log,
                n,
                signing_keys=_led_all_keys,
            )
            body["attestation_verified"] = att_ok
            body["attestation"] = att_detail
            if not att_ok:
                body["verified"] = False
            return JSONResponse(content=body, headers=_base_headers(tid, proxy_ms=ledger_ms))

        if not cfg.governance.audit_signing_key:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "audit_signing_key required for verification",
                        "type": "invalid_request_error",
                        "user_message": (
                            "Cannot verify flat JSONL audit rows without a signing key. "
                            "Set governance.audit_signing_key or AURORA_LENS_AUDIT_SIGNING_KEY "
                            "to the same secret the writer used for HMAC."
                        ),
                    }
                },
                headers=_base_headers(tid, outcome="ERROR", proxy_ms=int((time.time() - started) * 1000)),
            )
        # D4 multi-key: primary key + any historical keys for logs spanning key rotation
        _primary = cfg.governance.audit_signing_key.encode("utf-8")
        _extra = [k.encode("utf-8") for k in cfg.governance.audit_signing_keys if k]
        all_keys = [_primary] + _extra
        hmac_ok, entries_checked, first_failed = verify_audit_entries(
            cfg.governance.audit_log, n, signing_keys=all_keys
        )
        chain_ok, chain_entries, first_break_idx, chain_reason = verify_chain(
            cfg.governance.audit_log, n, signing_keys=all_keys
        )
        att_ok, att_detail = verify_co_attestation_window(
            cfg.governance.audit_log,
            n,
            signing_keys=all_keys,
        )
        proxy_ms = int((time.time() - started) * 1000)
        verified = hmac_ok and chain_ok and att_ok
        # chain_verified: True when the slice is internally consistent, even if not
        # anchored to genesis (unanchored_slice).  verified requires both hmac and
        # full-chain anchoring — so unanchored_slice sets chain_verified=True, verified=False.
        chain_verified = chain_ok or chain_reason == "unanchored_slice"
        body: Dict[str, Any] = {
            "verified": verified,
            "entries": entries_checked,
            "hmac_verified": hmac_ok,
            "chain_verified": chain_verified,
            "attestation_verified": att_ok,
            "attestation": att_detail,
            "backend": "jsonl",
        }
        if chain_reason == "unanchored_slice":
            # Slice is internally consistent but not anchored to the chain head.
            # entry[0].prev_cid points to a predecessor outside the requested window.
            body["reason"] = "unanchored_slice"
            body["reason_detail"] = (
                "The slice is internally consistent but not anchored to the chain head. "
                "entry[0].prev_cid references a predecessor outside this window. "
                "Increase n or verify from the start of the log to prove global chain integrity."
            )
        elif not verified:
            if first_failed is not None:
                body["first_failed_entry"] = first_failed  # cid of first HMAC failure
            if first_break_idx is not None:
                body["first_break_index"] = first_break_idx
                body["reason"] = chain_reason
                if chain_reason == "chain_break":
                    body["reason_detail"] = (
                        "Chain break: entry at index {} has prev_cid that does not match "
                        "the previous entry's cid. Verify from the start of the log or "
                        "increase n to include the chain head."
                    ).format(first_break_idx)
        _om_jsonl = build_jsonl_verify_operator_message(
            verified=verified,
            hmac_ok=hmac_ok,
            first_failed=str(first_failed) if first_failed is not None else None,
            chain_ok=chain_ok,
            chain_reason=chain_reason,
            first_break_idx=first_break_idx,
        )
        if _om_jsonl:
            body["operator_message"] = _om_jsonl
        _merge_audit_verify_semantics(
            body,
            Path(cfg.governance.audit_log),
            n,
            "jsonl",
            bool(verified),
        )
        return JSONResponse(content=body, headers=_base_headers(tid, proxy_ms=proxy_ms))

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        started = time.time()
        trace_id = trace_id_var.get() or f"proxy:{uuid.uuid4().hex}"
        proxy_ms = lambda: int((time.time() - started) * 1000)

        def _err_headers(session_id: str | None = None) -> Dict[str, str]:
            return _aurora_headers(
                outcome="ERROR",
                trace_id=trace_id,
                audit_sink=audit_sink,
                provider=cfg.upstream.provider,
                policy=cfg.governance.default_policy,
                policy_version=policy_version,
                session_id=session_id,
                proxy_ms=proxy_ms(),
            )

        # Phase 5: Read body with size limit
        try:
            body = await request.body()
        except Exception:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "Unable to read request body", "type": "invalid_request_error"}},
                headers=_err_headers(),
            )

        if h.max_payload_bytes > 0 and len(body) > h.max_payload_bytes:
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "message": f"Payload too large: {len(body)} bytes (max {h.max_payload_bytes})",
                        "type": "invalid_request_error",
                    }
                },
                headers=_err_headers(None),
            )

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "Invalid JSON body", "type": "invalid_request_error"}},
                headers=_err_headers(),
            )
        except Exception:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "Unable to parse JSON", "type": "invalid_request_error"}},
                headers=_err_headers(),
            )

        # Phase 5: Validate payload structure
        if h.max_messages > 0 or h.max_content_chars > 0:
            try:
                _validate_payload(
                    payload,
                    max_messages=h.max_messages or 999_999,
                    max_content_chars=h.max_content_chars or 999_999_999,
                )
            except ValueError as e:
                return JSONResponse(
                    status_code=422,
                    content={"error": {"message": str(e), "type": "invalid_request_error"}},
                    headers=_err_headers(None),
                )

        # Phase 5: Rate limits (check before session_id so we can 429 early)
        if global_limiter is not None:
            allowed, retry_after = global_limiter.check(key=None)
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": {
                            "message": "Rate limit exceeded (global)",
                            "type": "rate_limit_error",
                        }
                    },
                    headers={**_err_headers(None), "Retry-After": retry_after or "60"},
                )

        # Phase B.4: Per-IP rate limit
        if ip_limiter is not None:
            client_ip = _client_ip(request, h.trusted_proxy_ips)
            allowed, retry_after = ip_limiter.check(key=client_ip)
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": {
                            "message": "Rate limit exceeded (per IP)",
                            "type": "rate_limit_error",
                        }
                    },
                    headers={**_err_headers(None), "Retry-After": retry_after or "60"},
                )

        try:
            parsed = parse_chat_request(payload)
        except ValueError as e:
            return JSONResponse(
                status_code=422,
                content={"error": {"message": str(e), "type": "invalid_request_error"}},
                headers=_err_headers(None),
            )

        session_id = _extract_session_id(request, payload)
        if not session_id or not str(session_id).strip():
            session_id = f"session-{uuid.uuid4().hex[:12]}"

        # L.2: X-Aurora-Mock-Hard-Stop — demo only. Quarantined: never active in enterprise (production) mode.
        mock_hard_stop = (
            cfg.governance.mode != "enterprise"
            and cfg.governance.enable_mock_hard_stop
            and str(request.headers.get("x-aurora-mock-hard-stop", "")).strip().lower() in ("1", "true", "yes")
        )
        token_mock = mock_hard_stop_var.set(mock_hard_stop)

        # Phase 5: Per-session rate limit
        if session_limiter is not None:
            allowed, retry_after = session_limiter.check(key=session_id)
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": {
                            "message": "Rate limit exceeded (per session)",
                            "type": "rate_limit_error",
                        }
                    },
                    headers={**_err_headers(session_id), "Retry-After": retry_after or "60"},
                )

        lock_ctx = sessions.with_lock(session_id)
        lock_ctx.__enter__()
        try:
            lens, _is_new_session = sessions.get_or_create(session_id)

            # Persist declarative epistemic uncertainty state to session PEF when provided.
            if parsed.open_epistemic_uncertainties is not None:
                lens.pef.open_epistemic_uncertainties = parsed.open_epistemic_uncertainties

            _logger.info(
                "chat_completions_request",
                extra={
                    "aurora_trace_id": trace_id,
                    "aurora_session_id": session_id,
                    "stream": parsed.stream,
                    "user_message_len": len(parsed.user_message),
                    "history_turns": len(parsed.conversation_history),
                },
            )

            include_op_detail_eff = _effective_include_operator_detail(request)

            _policy_prof = (
                parsed.request_metadata.policy_profile if parsed.request_metadata else None
            )
            _mode_ov, _pol_ov = resolve_policy_profile_governance(_policy_prof)

            _operator_domain = _effective_operator_domain(request, payload)
            _effective_metadata = apply_sovereign_provider_route_hook(
                cfg,
                parsed.request_metadata,
                operator_domain=_operator_domain,
                request_model=parsed.model,
            )

            token = session_id_var.set(session_id)
            token_pef_context_id = pef_context_id_var.set(session_id)
            req_hash = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
            ).hexdigest()
            token_req = request_hash_var.set(req_hash)
            token_prompt = request_prompt_var.set(parsed.user_message)
            token_meta = request_metadata_var.set(_effective_metadata)
            token_exec = execution_task_var.set(parsed.execution_task)
            token_gov_mode = governance_mode_override_var.set(_mode_ov)
            token_meta_policy = metadata_policy_override_var.set(_pol_ov)
            token_coc_runtime = chain_of_custody_runtime_provenance_var.set(
                _chat_runtime_provenance_for_audit(cfg, parsed)
            )
            token_operator_domain = domain_var.set(_operator_domain)
            # Fresh run_id per HTTP request — a shared run_id across every request in a
            # process makes ledger rows indistinguishable at the call level.
            # proxy_run_id is deliberately NOT re-minted here: it identifies this proxy
            # process/deployment (same value reported by GET /health and startup logging,
            # set once in create_app() and passed into the bridge constructor), and
            # verify_pef_linkage_detailed() in govern/audit_io.py relies on consecutive
            # audit rows sharing one proxy_run_id to recognize them as the same
            # deployment's PEF chain edge (only differing proxy_run_id values across a
            # restart/redeploy are meant to break that link). Minting a new proxy_run_id
            # per request here previously made every cross-turn pair look like a
            # redeploy boundary, silently disabling PEF audit-linkage verification.
            token_run_id = run_id_var.set(str(uuid.uuid4()))
            try:
                # OpenAI clients may send full history each POST — seed prior user turns
                # (extraction only) with the same operator domain as this completion.
                if _is_new_session and parsed.conversation_history:
                    await lens.seed_history(parsed.conversation_history)
                if parsed.stream:
                    try:
                        stream_iter = lens.process_stream(
                            parsed.user_message,
                            parsed.external_flags,
                            include_operator_detail=include_op_detail_eff,
                        )
                        first = await stream_iter.__anext__()
                    except Exception as e:
                        import httpx as _httpx
                        # Do not persist: first __anext__ may have left partial in-memory
                        # turn state; next request reloads last committed SessionRecord.
                        try:
                            lock_ctx.__exit__(None, None, None)
                        except Exception:
                            pass
                        if isinstance(e, _httpx.HTTPStatusError):
                            upstream_status = e.response.status_code
                            try:
                                upstream_detail = e.response.json().get("error", {}).get("message", "")
                            except Exception:
                                upstream_detail = e.response.text[:200]
                            _logger.error(
                                "aurora_upstream_http_error",
                                extra={"aurora_trace_id": trace_id, "aurora_session_id": session_id, "upstream_status": upstream_status, "upstream_detail": upstream_detail},
                            )
                            return JSONResponse(
                                status_code=502,
                                content={
                                    "error": {
                                        "message": f"Upstream API error {upstream_status}: {upstream_detail}",
                                        "type": "upstream_error",
                                        "code": "upstream_http_error",
                                        "upstream_status": upstream_status,
                                        "trace_id": trace_id,
                                    }
                                },
                                headers=_err_headers(session_id),
                            )
                        _logger.error(
                            "aurora_proxy_internal_error",
                            extra={"aurora_trace_id": trace_id, "aurora_session_id": session_id, "error": str(e)},
                            exc_info=True,
                        )
                        return JSONResponse(
                            status_code=500,
                            content={
                                "error": {
                                    "message": "Internal error during processing",
                                    "type": "server_error",
                                    "code": "aurora_proxy_internal_error",
                                    "trace_id": trace_id,
                                },
                                "aurora": {
                                    "governance": "ERROR",
                                    "governance_hint": "proxy_internal_error",
                                },
                            },
                            headers=_err_headers(session_id),
                        )
                    kind, payload = first
                    if kind in ("extraction_failed", "clarification_continuation", "blocked_act"):
                        sessions.persist(session_id, lens)
                        try:
                            lock_ctx.__exit__(None, None, None)
                        except Exception:
                            pass
                        _outcome_label = (
                            "CLARIFICATION_CONTINUATION" if kind == "clarification_continuation"
                            else payload.action.name if kind == "blocked_act"
                            else "EXTRACTION_FAILED"
                        )
                        metrics.inc("chat_completions_total", labels={"outcome": _outcome_label})
                        body = format_chat_response(
                            payload,
                            request_model=parsed.model or cfg.upstream.model,
                            include_operator_detail=include_op_detail_eff,
                            session_id=session_id,
                            pef=lens.pef,
                            pef_admission_result_wire=lens.peek_pef_admission_result_wire(),
                        )
                        ah = _aurora_headers(
                            outcome=payload.action.name,
                            trace_id=trace_id,
                            audit_sink=audit_sink,
                            provider=cfg.upstream.provider,
                            policy=cfg.governance.default_policy,
                            policy_version=policy_version,
                            audit_id=payload.decision.cid if payload.decision and payload.decision.cid else None,
                            session_id=session_id,
                            proxy_ms=proxy_ms(),
                        )
                        return JSONResponse(content=_json_safe(body), status_code=200, headers=ah)

                    def _inject_session(aurora: dict) -> dict:
                        """Inject session_id routing handle into streaming aurora metadata."""
                        if session_id:
                            aurora["session_id"] = session_id
                        return aurora

                    async def _stream_gen():
                        _stream_aborted = False
                        try:
                            k, p = kind, payload
                            if k == "chunk":
                                chunk_dict, _ = p
                                yield f"data: {json.dumps(chunk_dict, separators=(',', ':'))}\n\n"
                            elif k == "governed_chunk":
                                # Non-ADMIT governed continuation: same SSE format as "chunk".
                                chunk_dict, _ = p
                                yield f"data: {json.dumps(chunk_dict, separators=(',', ':'))}\n\n"
                            elif k == "progress":
                                # Safe scaffolding signal — no content, no governance metadata.
                                yield f"data: {json.dumps(p.as_event_dict(), separators=(',', ':'))}\n\n"
                            elif k == "metadata":
                                _gflags = p.pop("_log_flags", [])
                                _logger.info(
                                    "governance_outcome",
                                    extra={
                                        "outcome": p.get("governance", "UNKNOWN"),
                                        "flags": _gflags,
                                        "flags_count": len(_gflags),
                                        "blocked": p.get("governance") == "HARD_STOP",
                                        "revised": p.get("governance") == "FORCE_REVISE",
                                        "audit_id": p.get("audit_id"),
                                        "policy": p.pop("_log_policy", None),
                                        "pathway_id": p.pop("_log_pathway", None),
                                        "commitment_closed": p.pop("_log_commitment_closed", None),
                                        "stream": True,
                                    },
                                )
                                yield format_stream_metadata_event(_inject_session(p))
                                yield "data: [DONE]\n\n"
                                return
                            async for k, p in stream_iter:
                                if k == "chunk":
                                    chunk_dict, _ = p
                                    yield f"data: {json.dumps(chunk_dict, separators=(',', ':'))}\n\n"
                                elif k == "governed_chunk":
                                    chunk_dict, _ = p
                                    yield f"data: {json.dumps(chunk_dict, separators=(',', ':'))}\n\n"
                                elif k == "progress":
                                    yield f"data: {json.dumps(p.as_event_dict(), separators=(',', ':'))}\n\n"
                                elif k == "metadata":
                                    _gflags = p.pop("_log_flags", [])
                                    _logger.info(
                                        "governance_outcome",
                                        extra={
                                            "outcome": p.get("governance", "UNKNOWN"),
                                            "flags": _gflags,
                                            "flags_count": len(_gflags),
                                            "blocked": p.get("governance") == "HARD_STOP",
                                            "revised": p.get("governance") == "FORCE_REVISE",
                                            "audit_id": p.get("audit_id"),
                                            "policy": p.pop("_log_policy", None),
                                            "pathway_id": p.pop("_log_pathway", None),
                                            "commitment_closed": p.pop("_log_commitment_closed", None),
                                            "stream": True,
                                        },
                                    )
                                    yield format_stream_metadata_event(_inject_session(p))
                                    yield "data: [DONE]\n\n"
                        except asyncio.CancelledError:
                            # Client disconnected mid-stream (uvicorn cancels the task).
                            # Suppress — generator ends cleanly; FastAPI sees StopAsyncIteration,
                            # not an unhandled exception, so no 500 is emitted.
                            _stream_aborted = True
                            _logger.info(
                                "aurora_proxy_stream_client_disconnect",
                                extra={"aurora_trace_id": trace_id, "aurora_session_id": session_id},
                            )
                        except (BrokenPipeError, ConnectionResetError) as e:
                            _stream_aborted = True
                            _logger.info(
                                "aurora_proxy_stream_connection_reset",
                                extra={"aurora_trace_id": trace_id, "aurora_session_id": session_id, "error": str(e)},
                            )
                        except Exception as e:
                            _logger.error(
                                "aurora_proxy_stream_error",
                                extra={"aurora_trace_id": trace_id, "aurora_session_id": session_id, "error": str(e)},
                                exc_info=True,
                            )
                            err_event = json.dumps({
                                "error": {
                                    "message": "Internal error during streaming",
                                    "type": "server_error",
                                    "code": "aurora_proxy_internal_error",
                                    "trace_id": trace_id,
                                }
                            }, separators=(",", ":"))
                            yield f"data: {err_event}\n\n"
                        finally:
                            sessions.persist(session_id, lens)
                            if _stream_aborted and cfg.governance.audit_log:
                                try:
                                    _payload = {
                                        "type": "stream_abort",
                                        "trace_id": trace_id,
                                        "session_id": session_id,
                                        "timestamp": datetime.datetime.now(
                                            datetime.timezone.utc
                                        ).isoformat(),
                                        "stream_completed": False,
                                        "stream_abort_reason": "client_disconnect",
                                    }
                                    if audit_sink == "ledger":
                                        bridge.log_subsystem_audit_event(
                                            op="stream_abort",
                                            payload=_payload,
                                            trace_id=trace_id,
                                        )
                                    else:
                                        _sk = (
                                            cfg.governance.audit_signing_key.encode("utf-8")
                                            if cfg.governance.audit_signing_key
                                            else None
                                        )
                                        append_audit_entry(
                                            cfg.governance.audit_log,
                                            _payload,
                                            signing_key=_sk,
                                            max_mb=cfg.governance.audit_log_max_mb,
                                        )
                                except OSError:
                                    pass
                            try:
                                lock_ctx.__exit__(None, None, None)
                            except Exception:
                                pass

                    metrics.inc("chat_completions_total", labels={"outcome": "STREAM"})
                    return StreamingResponse(
                        _stream_gen(),
                        media_type="text/event-stream",
                        headers=_aurora_headers(
                            outcome="STREAM",
                            trace_id=trace_id,
                            audit_sink=audit_sink,
                            provider=cfg.upstream.provider,
                            policy=cfg.governance.default_policy,
                            policy_version=policy_version,
                            session_id=session_id,
                            proxy_ms=proxy_ms(),
                        ),
                    )

                try:
                    out = lens.process(
                        parsed.user_message,
                        parsed.external_flags,
                        include_operator_detail=include_op_detail_eff,
                    )
                    if inspect.isawaitable(out):
                        out = await out
                except Exception as e:
                    import httpx as _httpx
                    if isinstance(e, _httpx.HTTPStatusError):
                        upstream_status = e.response.status_code
                        try:
                            upstream_detail = e.response.json().get("error", {}).get("message", "")
                        except Exception:
                            upstream_detail = e.response.text[:200]
                        _logger.error(
                            "aurora_upstream_http_error",
                            extra={"aurora_trace_id": trace_id, "aurora_session_id": session_id, "upstream_status": upstream_status, "upstream_detail": upstream_detail},
                        )
                        return JSONResponse(
                            status_code=502,
                            content={
                                "error": {
                                    "message": f"Upstream API error {upstream_status}: {upstream_detail}",
                                    "type": "upstream_error",
                                    "code": "upstream_http_error",
                                    "upstream_status": upstream_status,
                                    "trace_id": trace_id,
                                }
                            },
                            headers=_err_headers(session_id),
                        )
                    if isinstance(e, _httpx.RequestError):
                        _logger.error(
                            "aurora_upstream_request_error",
                            extra={"aurora_trace_id": trace_id, "aurora_session_id": session_id, "error": str(e)},
                        )
                        return JSONResponse(
                            status_code=502,
                            content={
                                "error": {
                                    "message": f"Upstream connection error: {type(e).__name__}: {e}",
                                    "type": "upstream_error",
                                    "code": "upstream_request_error",
                                    "trace_id": trace_id,
                                }
                            },
                            headers=_err_headers(session_id),
                        )
                    _logger.error(
                        "aurora_proxy_internal_error",
                        extra={"aurora_trace_id": trace_id, "aurora_session_id": session_id, "error": str(e)},
                        exc_info=True,
                    )
                    return JSONResponse(
                        status_code=500,
                        content={
                            "error": {
                                "message": "Internal error during processing",
                                "type": "server_error",
                                "code": "aurora_proxy_internal_error",
                                "trace_id": trace_id,
                            },
                            "aurora": {
                                "governance": "ERROR",
                                "governance_hint": "proxy_internal_error",
                            },
                        },
                        headers=_err_headers(session_id),
                    )

                if isinstance(out, LensResult):
                    metrics.inc("chat_completions_total", labels={"outcome": out.action.name})
                    body = format_chat_response(
                        out,
                        request_model=parsed.model or cfg.upstream.model,
                        include_operator_detail=include_op_detail_eff,
                        session_id=session_id,
                        pef=lens.pef,
                        pef_admission_result_wire=lens.peek_pef_admission_result_wire(),
                    )
                    outcome = out.action.name
                    audit_id = out.decision.cid if out.decision and out.decision.cid else None
                    _decision_flags = out.decision.flags if out.decision else out.flags
                    _logger.info(
                        "governance_outcome",
                        extra={
                            "outcome": outcome,
                            "flags": [f.flag_type.name for f in _decision_flags],
                            "flags_count": len(_decision_flags),
                            "blocked": outcome == "HARD_STOP",
                            "revised": outcome == "FORCE_REVISE",
                            "audit_id": audit_id,
                            "policy": out.decision.policy if out.decision else cfg.governance.default_policy,
                            "pathway_id": out.decision.pathway_id if out.decision else None,
                            "commitment_closed": out.decision.commitment_closed if out.decision else False,
                            "stream": False,
                        },
                    )
                    ah = _aurora_headers(
                        outcome=outcome,
                        trace_id=trace_id,
                        audit_sink=audit_sink,
                        provider=cfg.upstream.provider,
                        policy=cfg.governance.default_policy,
                        policy_version=policy_version,
                        audit_id=audit_id,
                        session_id=session_id,
                        proxy_ms=proxy_ms(),
                    )
                    return JSONResponse(content=_json_safe(body), status_code=200, headers=ah)

                # Invariant breach: Lens must return LensResult
                _logger.error(
                    "aurora_proxy_invariant_breach",
                    extra={
                        "aurora_trace_id": trace_id,
                        "aurora_session_id": session_id,
                        "out_type": type(out).__name__,
                    },
                )
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": {
                            "message": "Internal invariant breach: pipeline did not return LensResult",
                            "type": "server_error",
                            "code": "aurora_proxy_invariant_breach",
                            "trace_id": trace_id,
                        },
                        "aurora": {
                            "governance": "ERROR",
                            "governance_hint": "proxy_internal_error",
                        },
                    },
                    headers=_err_headers(session_id),
                )
            finally:
                run_id_var.reset(token_run_id)
                domain_var.reset(token_operator_domain)
                chain_of_custody_runtime_provenance_var.reset(token_coc_runtime)
                metadata_policy_override_var.reset(token_meta_policy)
                governance_mode_override_var.reset(token_gov_mode)
                request_metadata_var.reset(token_meta)
                execution_task_var.reset(token_exec)
                request_hash_var.reset(token_req)
                request_prompt_var.reset(token_prompt)
                pef_context_id_var.reset(token_pef_context_id)
                session_id_var.reset(token)
                mock_hard_stop_var.reset(token_mock)
        finally:
            # Non-streaming path: persist and release lock
            if not parsed.stream:
                sessions.persist(session_id, lens)
                try:
                    lock_ctx.__exit__(None, None, None)
                except Exception:
                    pass

    return app


def build_app_from_config_path(config_path: str) -> FastAPI:
    cfg = ProxyConfig.from_yaml(config_path)
    return create_app(cfg)

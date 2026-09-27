"""OpenAI compatibility layer.

Translates between OpenAI /v1/chat/completions format and aurora-lens internals.

CRITICAL: All governance outcomes return HTTP 200.
Governance metadata goes in the response body under "aurora", not as error codes.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

from aurora_lens.lens import LensResult
from aurora_lens.pef.state import PEFState
from aurora_lens.request_metadata import RequestMetadata, parse_request_metadata
from aurora_lens.execution_task import ExecutionTask, parse_execution_task_from_body
from aurora_lens.verify.flags import Flag, FlagType, flag_from_external
from aurora_lens.aurora_metadata import build_aurora_block
from aurora_lens.govern.decision import InterventionAction
from aurora_lens.proxy.json_wire import wire_encode, wire_nonneg_int
from aurora_lens.govern.epistemic_uncertainty_gate import parse_open_epistemic_uncertainties


def _normalize_operator_pef_pending_for_wire(operator_pef: dict[str, Any]) -> None:
    """Ensure operator-plane pending clarification lists are JSON-safe (empty-candidates CONTAIN)."""
    pc = operator_pef.get("pending_clarification")
    if not isinstance(pc, dict) or not pc.get("active"):
        return

    def _str_list(raw: Any) -> list[str]:
        if raw is None:
            return []
        if isinstance(raw, list):
            return [str(x) for x in raw]
        if isinstance(raw, (tuple, set, frozenset)):
            return [str(x) for x in raw]
        return []

    pc["candidate_entities"] = _str_list(pc.get("candidate_entities"))
    pc["ambiguous_referents"] = _str_list(pc.get("ambiguous_referents"))
    pc["clarification_choices"] = _str_list(pc.get("clarification_choices"))
    bc = pc.get("blocked_claims")
    if bc is None:
        pc["blocked_claims"] = []
    elif isinstance(bc, list):
        pass
    elif isinstance(bc, tuple):
        pc["blocked_claims"] = list(bc)
    else:
        pc["blocked_claims"] = []


def _telemetry_release_path(action: InterventionAction, llm_called: bool) -> str:
    if action == InterventionAction.PASS:
        return "released"
    if action == InterventionAction.SOFT_CORRECT:
        return "released_with_annotation"
    if action == InterventionAction.CONTAIN:
        return "needs_more_information"
    if action == InterventionAction.HARD_STOP:
        return "blocked_after_generation" if llm_called else "blocked_before_generation"
    if action == InterventionAction.FORCE_REVISE:
        return "blocked_after_generation" if llm_called else "revision_requested"
    return "unknown"


def _merge_operator_plane_telemetry(
    aurora: dict[str, Any],
    *,
    result: LensResult,
) -> None:
    """Surface release/LLM observability for operator-plane clients (demo, integration).

    Values are derived from ``LensResult`` and the merged ``aurora`` block only;
    no fabricated endpoint fields.
    """
    usage = result.usage if isinstance(result.usage, dict) else {}
    ct = wire_nonneg_int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    draft = (getattr(result, "upstream_model_draft", None) or "").strip()
    llm_called = bool(ct > 0 or draft)
    aurora["llm_called"] = llm_called
    _rp_hint = getattr(result, "telemetry_release_path", None)
    if isinstance(_rp_hint, str) and _rp_hint.strip():
        aurora["release_path"] = _rp_hint.strip()
    else:
        aurora["release_path"] = _telemetry_release_path(result.action, llm_called)
    pre_llm_blocked = bool(
        not llm_called
        and result.action
        in (InterventionAction.HARD_STOP, InterventionAction.CONTAIN, InterventionAction.FORCE_REVISE)
    )
    aurora["pre_llm_blocked"] = pre_llm_blocked
    _orig = (result.original_response or "").strip()
    _released = (
        getattr(result.decision, "governed_response", None)
        if result.decision is not None
        else None
    )
    if _released is None or not str(_released).strip():
        _released = result.response
    _rel_s = "" if _released is None else str(_released).strip()
    # Upstream "blocked" candidate: only when a distinct buffer exists on the result and
    # it is not identical to the released assistant body (pre-LLM paths have no buffer).
    aurora["has_blocked_upstream_output"] = bool(_orig) and (
        not _rel_s or _orig != _rel_s
    )
    intercepted = aurora.get("intercepted_upstream_output")
    intercepted_nonempty = intercepted is not None and str(intercepted).strip() != ""
    original_echo = aurora.get("original_response")
    original_echo_nonempty = (
        original_echo is not None and str(original_echo).strip() != ""
    )
    if _orig and not intercepted_nonempty and not original_echo_nonempty:
        if not aurora.get("intercepted_upstream_redacted_reason"):
            aurora["intercepted_upstream_redacted_reason"] = (
                "Upstream model output is present on the lens result (original_response) but "
                "was not serialized into this operator payload; use audit / forensic hashes."
            )
    fe = aurora.get("forensic_event")
    if isinstance(fe, dict) and fe.get("attempted_action") is not None:
        aurora["attempted_action"] = fe.get("attempted_action")
    _merge_operator_plane_gate_invariants(aurora, action=result.action, result=result)


def _operator_failed_constraints(
    aurora: dict[str, Any],
    *,
    action: InterventionAction | None,
    result: LensResult | None = None,
) -> list[str]:
    """Resolve terminal stop constraints for operator/demo surfaces (never empty on HARD_STOP)."""
    constraints: list[str] = []

    def _add(candidate: object | None) -> None:
        text = str(candidate or "").strip()
        if text and text.lower() != "none" and text not in constraints:
            constraints.append(text)

    fe = aurora.get("forensic_event")
    if isinstance(fe, dict) and isinstance(fe.get("failed_constraints"), list):
        for item in fe["failed_constraints"]:
            _add(item)

    if isinstance(aurora.get("failed_constraints"), list):
        for item in aurora["failed_constraints"]:
            _add(item)
    _add(aurora.get("failed_constraint"))

    flags = list(result.flags) if result is not None and result.flags else []
    for flag in flags:
        _add(flag.flag_type.name)
        _add(flag.rule_id)

    if isinstance(aurora.get("flags"), list):
        for item in aurora["flags"]:
            _add(item)

    decision = result.decision if result is not None else None
    if decision is not None and decision.rule_result is not None:
        rr = decision.rule_result
        if rr.flag_type is not None:
            _add(rr.flag_type.name)
        _add(rr.reason_code)
        _add(rr.rule_id)

    rr_wire = aurora.get("rule_result")
    if isinstance(rr_wire, dict):
        _add(rr_wire.get("flag_type"))
        _add(rr_wire.get("reason_code"))
        _add(rr_wire.get("rule_id"))

    if action == InterventionAction.FORCE_REVISE and not constraints:
        rr = rr_wire if isinstance(rr_wire, dict) else {}
        rr_reason = str(rr.get("reason_code") or "").strip()
        _add(rr_reason if rr_reason else "GOVERNANCE_REVISION_REQUIRED")

    if action in (
        InterventionAction.HARD_STOP,
        InterventionAction.CONTAIN,
    ) and not constraints:
        rr = rr_wire if isinstance(rr_wire, dict) else {}
        _add(rr.get("rule_id"))
        _add(rr.get("reason_code"))

    return list(dict.fromkeys(constraints))


def _merge_operator_plane_gate_invariants(
    aurora: dict[str, Any],
    *,
    action: InterventionAction | None = None,
    result: LensResult | None = None,
) -> None:
    """Expose stable operator telemetry contract for pre/post-generation gating."""
    if action is None:
        raw_action = str(aurora.get("governance") or "").strip().upper()
        action = InterventionAction.__members__.get(raw_action)
    llm_called = bool(aurora.get("llm_called") is True)
    pre_llm_blocked = bool(aurora.get("pre_llm_blocked") is True)
    non_admit = action in (
        InterventionAction.HARD_STOP,
        InterventionAction.CONTAIN,
        InterventionAction.FORCE_REVISE,
    )
    if non_admit:
        blocked_phase = "pre_generation" if pre_llm_blocked else "post_generation"
    else:
        blocked_phase = "not_blocked"
    released = action in (InterventionAction.PASS, InterventionAction.SOFT_CORRECT)
    constraints = _operator_failed_constraints(aurora, action=action, result=result)
    aurora["llm_invoked"] = llm_called
    aurora["model_invoked"] = llm_called
    aurora["candidate_produced"] = llm_called
    aurora["released"] = released
    aurora["blocked_phase"] = blocked_phase
    aurora["failed_constraints"] = constraints
    if constraints:
        aurora["failed_constraint"] = constraints[0]


def _content_to_text(content: Any) -> str:
    """Extract text from OpenAI content (string or multimodal list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    parts.append(part["text"])
        return "\n".join(p for p in parts if p)
    return ""


# ── Request parsing ─────────────────────────────────────────────────

def _validate_external_flag(raw: Any, index: int) -> None:
    """Validate one external flag. Raises ValueError if malformed (422)."""
    if not isinstance(raw, dict):
        raise ValueError(
            f"aurora.external_flags[{index}]: expected object, got {type(raw).__name__}"
        )
    type_str = raw.get("type")
    if not type_str or not isinstance(type_str, str):
        raise ValueError(
            f"aurora.external_flags[{index}]: missing or invalid 'type' (string required)"
        )
    try:
        FlagType[type_str]
    except KeyError:
        raise ValueError(
            f"aurora.external_flags[{index}]: unknown type {type_str!r}"
        )
    evidence = raw.get("evidence")
    if not isinstance(evidence, list):
        raise ValueError(
            f"aurora.external_flags[{index}]: 'evidence' must be a list"
        )
    if raw.get("source") is not None and not isinstance(raw.get("source"), str):
        raise ValueError(
            f"aurora.external_flags[{index}]: 'source' must be string if present"
        )


def _parse_external_flags(body: dict[str, Any]) -> list[Flag]:
    """Extract aurora.external_flags and convert to Flag objects (Phase 9).

    Validates structure: type (string), evidence (list), source (string, optional).
    Raises ValueError for malformed entries → 422.
    """
    aurora = body.get("aurora") or {}
    raw_list = aurora.get("external_flags")
    if not isinstance(raw_list, list):
        return []
    for i, raw in enumerate(raw_list):
        _validate_external_flag(raw, i)
    flags: list[Flag] = []
    for raw in raw_list:
        f = flag_from_external(raw)
        assert f is not None  # Validation passed
        flags.append(f)
    return flags


@dataclass
class ParsedRequest:
    """Parsed OpenAI chat completion request."""
    user_message: str
    conversation_history: list[dict[str, str]]
    session_id: str
    model: str | None
    stream: bool
    raw: dict[str, Any]
    external_flags: list[Flag]  # From aurora.external_flags (Phase 9)
    request_metadata: RequestMetadata | None = None  # Optional host envelope (not user plane)
    execution_task: ExecutionTask | None = None  # Structured execution-boundary task
    open_epistemic_uncertainties: list[dict] | None = None  # None = absent from request


def parse_chat_request(body: dict[str, Any]) -> ParsedRequest:
    """Parse an OpenAI-format chat completion request body.

    Extracts the last user message and conversation history.
    Session ID comes from a custom field or is generated.
    Handles multimodal content (list of parts with type/text).

    Optional extension: ``body["aurora"]["request_domain"]`` (``general`` / ``finance``
    / ``legal`` / ``medical``) is an operator/deployment corridor hint when the
    proxy applies it to :data:`~aurora_lens.context.domain_var` (see ``create_app``).
    It is not a correctness mechanism for continuity, which must come from PEF
    state-native transition/projection behavior.
    """
    messages = body.get("messages", [])

    stream = bool(body.get("stream", False))
    meta = parse_request_metadata(body.get("request_metadata"))
    exec_task = parse_execution_task_from_body(body)

    evidence_state_raw = body.get("evidence_state")
    oeu: list[dict] | None = None
    if isinstance(evidence_state_raw, dict):
        oeu_raw = evidence_state_raw.get("open_epistemic_uncertainties")
        if oeu_raw is not None:
            oeu = parse_open_epistemic_uncertainties(oeu_raw)

    if not isinstance(messages, list) or not messages:
        for key in ("input", "prompt", "text"):
            v = body.get(key)
            if isinstance(v, str) and v.strip():
                return ParsedRequest(
                    user_message=v.strip(),
                    conversation_history=[],
                    session_id=body.get("aurora_session_id") or f"session-{uuid.uuid4().hex[:12]}",
                    model=body.get("model"),
                    stream=stream,
                    raw=body,
                    external_flags=_parse_external_flags(body),
                    request_metadata=meta,
                    execution_task=exec_task,
                    open_epistemic_uncertainties=oeu,
                )
        raise ValueError("Invalid request: expected 'messages' list (or 'input'/'prompt' string).")

    user_message = ""
    history: list[dict[str, str]] = []

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = msg.get("content", "")
        text = _content_to_text(content)
        if role == "system":
            continue
        if i == len(messages) - 1 and role == "user":
            user_message = text
        else:
            history.append({"role": role, "content": text})

    if not user_message.strip():
        for msg in reversed(messages):
            if isinstance(msg, dict):
                text = _content_to_text(msg.get("content"))
                if text.strip():
                    user_message = text.strip()
                    break
        if not user_message.strip():
            raise ValueError("Invalid request: could not extract text from messages[].")

    session_id = body.get("aurora_session_id", "")
    if not session_id or not str(session_id).strip():
        session_id = f"session-{uuid.uuid4().hex[:12]}"

    return ParsedRequest(
        user_message=user_message.strip(),
        conversation_history=history,
        session_id=str(session_id).strip(),
        model=body.get("model"),
        stream=stream,
        raw=body,
        external_flags=_parse_external_flags(body),
        request_metadata=meta,
        execution_task=exec_task,
        open_epistemic_uncertainties=oeu,
    )


# ── Response formatting ─────────────────────────────────────────────

def _usage_from_result(result: LensResult) -> dict[str, int]:
    """Build OpenAI-format usage from LensResult.usage (adapter token counts)."""
    u = getattr(result, "usage", None)
    if u and isinstance(u, dict):
        pt = u.get("input_tokens") or u.get("prompt_tokens", 0) or 0
        ct = u.get("output_tokens") or u.get("completion_tokens", 0) or 0
        pti = wire_nonneg_int(pt)
        cti = wire_nonneg_int(ct)
        return {
            "prompt_tokens": pti,
            "completion_tokens": cti,
            "total_tokens": pti + cti,
        }
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def format_chat_response(
    result: LensResult,
    model: str = "aurora-lens",
    request_model: str | None = None,
    include_operator_detail: bool = False,
    session_id: str | None = None,
    pef: PEFState | None = None,
    *,
    pef_admission_result_wire: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Format a LensResult as an OpenAI-compatible chat completion response.

    Always produces a valid response body. Governance metadata goes under the
    "aurora" key — never as an error.

    Two-plane output contract:
      User plane (always returned):
        aurora.governance     — action enum: PASS | SOFT_CORRECT | CONTAIN | ...
        aurora.turn             — turn counter
        aurora.session_id       — routing handle (when provided)
        aurora.governance_hint  — optional safe category (e.g. clarification_requested)
        aurora.unverified       — true when action is SOFT_CORRECT (UI hint)
        aurora.audit_id         — opaque audit handle when available
        No forensic_event, operator_pef, or raw hold objects on the user plane.

      Operator plane (only when include_operator_detail is effective for this response):
        The proxy may set this from config or, when ``allow_operator_detail_via_header`` is
        enabled in governance, from ``X-Aurora-Operator-Detail`` on the request.
        aurora.operator_pef     — present-state summary (session_mode, holds, binding_governance_state)
        aurora.forensic_event   — full envelope for non-ADMIT outcomes
        aurora.flags, rationale, policy, governance_note
        aurora.original_response — upstream model text when echoed for operator plane
        aurora.intercepted_upstream_redacted_reason — when upstream was withheld on
            HARD_STOP / FORCE_REVISE / CONTAIN, a stable token (never raw candidate text);
            full text remains in audit trail only
        aurora.pef_admission_result — optional serialized ``PEFAdmissionResult`` from the
            last bounded ``update_pef`` envelope this turn when provided by the host
    """
    response_id = f"chatcmpl-aurora-{uuid.uuid4().hex[:12]}"
    timestamp = int(time.time())

    assistant_text = result.response
    if assistant_text is None:
        assistant_text = ""
    elif not isinstance(assistant_text, str):
        assistant_text = str(assistant_text)

    response: dict[str, Any] = {
        "id": response_id,
        "object": "chat.completion",
        "created": timestamp,
        "model": request_model or result.model or model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": assistant_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": _usage_from_result(result),
    }

    aurora = build_aurora_block(
        action=result.action,
        flags=result.flags,
        turn=result.turn,
        decision=result.decision,
        pef=pef,
        include_operator_detail=include_operator_detail,
        session_id=session_id,
        original_response=result.original_response,
        governed_response_body=assistant_text,
        stream_governed=None,
        stream_truncated=False,
        stream_dropped_chars=0,
        operator_request_domain=getattr(result, "operator_request_domain", None),
        pef_admission_result_wire=pef_admission_result_wire,
    )

    response["aurora"] = aurora
    if include_operator_detail:
        op = aurora.get("operator_pef")
        if isinstance(op, dict):
            _normalize_operator_pef_pending_for_wire(op)
        _merge_operator_plane_telemetry(aurora, result=result)
    return response


def format_stream_metadata_event(aurora: dict[str, Any]) -> str:
    """Format aurora metadata as SSE event. Emitted as final event after stream completes."""
    safe = wire_encode({"aurora": aurora})
    return f"data: {json.dumps(safe, separators=(',', ':'))}\n\n"

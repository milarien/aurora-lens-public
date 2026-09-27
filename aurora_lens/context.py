"""Request-scoped context for audit correlation.

Contextvars propagate through async call stacks. The proxy sets trace_id and
session_id; the LogRecord factory injects them into every log record.

Failure modes to avoid:

(A) Context not reset after request
    If not reset → bleed risk (next request inherits previous trace_id/session_id).
    The proxy MUST reset in finally: middleware resets trace_id_var; chat_completions
    resets session_id_var. Both use try/finally with token.reset().

(B) Multiple workers / threadpool
    Contextvars are safe per worker. But if a worker reuses a threadpool and logs
    outside the async request context (e.g. background tasks), trace_id will be null.
    Not wrong — background tasks must explicitly propagate context if needed.
"""

from __future__ import annotations

import contextvars

from aurora_lens.request_metadata import RequestMetadata
from aurora_lens.execution_task import ExecutionTask

# Set by proxy at request start; read by LogRecord factory for every log record.
trace_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "aurora_trace_id",
    default=None,
)

# Set by proxy before lens.process(); read by bridge when emitting forensic envelope.
session_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "aurora_session_id",
    default=None,
)

# Durable world/frame handle for this session — the PEF continuity key the client
# is expected to round-trip. Distinct from trace_id (per-request) and session_id
# (routing handle); today the two share a value, but this var exists so PEF
# continuity has its own name independent of session/routing plumbing.
pef_context_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "aurora_pef_context_id",
    default=None,
)

# Fresh UUID minted per HTTP request by the proxy chat handler; read by
# CanonicalScannerGateBridge.log_decision for the ``run_id`` ledger field.
# Must be per-request, not per-bridge-instance: a shared run_id across every
# request in a process makes audit rows from different callers indistinguishable.
# Falls back to the bridge's own instance id when unset (non-proxy callers,
# e.g. direct Lens/bridge use in tests or scripts).
run_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "aurora_run_id",
    default=None,
)

# Fresh UUID minted per HTTP request by the proxy chat handler; read by
# CanonicalScannerGateBridge.log_decision for the ``proxy_run_id`` ledger field.
# Falls back to the bridge's constructor-supplied process-level proxy_run_id
# (e.g. for ``/health`` reporting or non-request-scoped logging).
proxy_run_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "aurora_proxy_run_id",
    default=None,
)

# Set by proxy for each chat completion: resolved (provider, model_id) for this HTTP request.
# Read by build_chain_of_custody_bundle; takes precedence over AURORA_PROVIDER / AURORA_MODEL_ID
# when both sides of the tuple are non-empty. Reset in the same finally as session_id_var.
chain_of_custody_runtime_provenance_var: contextvars.ContextVar[tuple[str, str] | None] = (
    contextvars.ContextVar(
        "aurora_chain_of_custody_runtime",
        default=None,
    )
)

# Phase B: Set by auth middleware when auth succeeds; read by bridge and LogRecord factory.
auth_label_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "aurora_auth_label",
    default=None,
)

# Phase B: Per-key policy override (strict | moderate). When set, overrides deployment policy.
auth_policy_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "aurora_auth_policy",
    default=None,
)

# Phase D1: SHA256 of normalized request body (before LLM call). Bridge reads for audit spine.
request_hash_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "aurora_request_hash",
    default=None,
)

# Forensic evidence vault: raw user prompt for this turn (never written inline to ledger).
request_prompt_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "aurora_request_prompt",
    default=None,
)

# Host integration: optional envelope (workspace, scope, records, policy profile, role).
# Set by proxy after parse_chat_request; reset in finally with session_id/request_hash.
request_metadata_var: contextvars.ContextVar[RequestMetadata | None] = contextvars.ContextVar(
    "aurora_request_metadata",
    default=None,
)

# Structured execution-boundary task (candidate release adjudication, etc.).
execution_task_var: contextvars.ContextVar[ExecutionTask | None] = contextvars.ContextVar(
    "aurora_execution_task",
    default=None,
)

# Derived from request_metadata.policy_profile (see resolve_policy_profile_governance).
# None = use deployment bridge mode / default policy chain.
governance_mode_override_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "aurora_governance_mode_override",
    default=None,
)
metadata_policy_override_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "aurora_metadata_policy_override",
    default=None,
)

# Session contention: lock_wait_ms, lock_acquired, lock_timeout_ms (when acquired=false)
lock_metadata_var: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "aurora_lock_metadata",
    default=None,
)

# L.2: When True, proxy uses canned non-compliant LLM response for HARD_STOP demo.
# Set by chat handler when X-Aurora-Mock-Hard-Stop header present.
mock_hard_stop_var: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "aurora_mock_hard_stop",
    default=False,
)

# ── Canonical Governor corridor context ──────────────────────────────────────
#
# These three vars supply domain, authority class, and user class to
# ContextResolver. They must be reset in a finally block after each request
# using the standard token = var.set(v) / var.reset(token) pattern.
# Failure to reset will bleed one request's governance corridor into the next.
#
# domain_var:
#   Set by route handlers when the endpoint has an explicit domain
#   (e.g. /v1/medical/chat sets "medical"). Leave unset for flag-pattern
#   fallback. Accepts Domain enum values or their string equivalents.
#
# authority_class_var:
#   Set once at startup from GovernanceConfig.authority_class.
#   Represents what kind of system this deployment is (GP | DA | HS).
#   Never set per-request — it is a deployment property.
#
# user_class_var:
#   Set per-request from an HTTP header if GovernanceConfig.user_class_header
#   is configured. Represents the caller's role (GENERAL, CLINICIAN, AUDITOR, …).
#   Defaults to GENERAL when not set.

domain_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "aurora_domain",
    default=None,
)

authority_class_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "aurora_authority_class",
    default=None,
)

user_class_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "aurora_user_class",
    default=None,
)


def get_trace_id() -> str | None:
    """Return current trace_id from context. None at startup, populated during requests."""
    return trace_id_var.get()


def get_session_id() -> str | None:
    """Return current session_id from context. None when not in chat flow."""
    return session_id_var.get()


def get_auth_label() -> str | None:
    """Return current auth key label from context. None when auth disabled or not set."""
    return auth_label_var.get()


def get_auth_policy() -> str | None:
    """Return per-key policy override from context. None when auth disabled or no override."""
    return auth_policy_var.get()


def get_lock_metadata() -> dict | None:
    """Return lock metadata: lock_wait_ms, lock_acquired, lock_timeout_ms (if not acquired)."""
    return lock_metadata_var.get()


def get_request_hash() -> str | None:
    """Return request hash from context. None when not set (e.g. non-chat paths)."""
    return request_hash_var.get()


def get_request_prompt() -> str | None:
    """Return raw user prompt text for this turn (evidence vault capture)."""
    return request_prompt_var.get()


def get_request_metadata() -> RequestMetadata | None:
    """Return host request envelope for this request. None when unset or non-proxy callers."""
    return request_metadata_var.get()


def get_execution_task() -> ExecutionTask | None:
    """Return structured execution-boundary task for this request, if any."""
    return execution_task_var.get()


def get_governance_mode_override() -> str | None:
    """Return public|enterprise|open override from policy_profile, or None."""
    return governance_mode_override_var.get()


def get_metadata_policy_override() -> str | None:
    """Return strict|moderate override from policy_profile, or None."""
    return metadata_policy_override_var.get()

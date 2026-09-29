"""Centralized, audit-friendly pytest skip reasons.

Each message states: why skip triggers, how to run the test, and whether the skip is
environment-dependent, opt-in manual, or an intentional product/contract gap.
"""

# --- Optional HTTP / ASGI (Starlette TestClient + httpx) -----------------------

SKIP_STARLETTE_HTTP_TESTCLIENT = (
    "Requires optional starlette and httpx packages for Starlette TestClient "
    "(in-process ASGI HTTP tests against the proxy app). "
    "Install: pip install starlette httpx (included in project dev dependencies). "
    "Triggers on ImportError when those packages are absent — typical of minimal "
    "installs or CI without dev extras. Environment-dependent; runs once deps are present."
)

# --- spaCy (optional NLP extraction backend) -----------------------------------

SKIP_SPACY_MODULE = (
    "Requires optional spacy package for spaCy extraction-backend code paths. "
    "Install: pip install spacy and a model (e.g. python -m spacy download en_core_web_sm), "
    "or project optional 'all' extras. "
    "Triggers when spacy is not importable — environment-dependent minimal tree. "
    "Not a deliberate waiver of governance assertions."
)

# --- OpenAI API (real_llm) -----------------------------------------------------

SKIP_OPENAI_API_KEY_MISSING = (
    "Requires OPENAI_API_KEY in the environment for live OpenAI API calls. "
    "Export a valid key; empty or placeholder ${...} values skip. "
    "Opt-in manual / release verification — not default CI. Environment-dependent credentials."
)

SKIP_AUDIT_SIGNING_KEY_FOR_HMAC_TEST = (
    "Requires AURORA_LENS_AUDIT_SIGNING_KEY so audit log lines are HMAC-signed and "
    "verify_audit_entries can validate the chain (test_audit_chain_integrity). "
    "Set to a non-empty secret. "
    "Environment-dependent subset of real_llm suite; other tests may still run with OPENAI_API_KEY only."
)

# --- Anthropic API -------------------------------------------------------------

SKIP_ANTHROPIC_API_KEY_MISSING = (
    "Requires ANTHROPIC_API_KEY for live ClaudeAdapter calls (TestLLMGovernanceIntegration). "
    "Export a valid key; empty or placeholder ${...} skips. "
    "Opt-in integration test — not default CI. Environment-dependent credentials."
)

# --- Ollama local LLM ----------------------------------------------------------

SKIP_OLLAMA_DAEMON_OR_MODEL = (
    "Requires Ollama listening on localhost:11434 (ollama serve) and the configured model "
    "installed (see OLLAMA_MODEL). Connection refused / missing model skips. "
    "Opt-in marker ollama_live / manual runs — environment-dependent."
)

SKIP_OLLAMA_CONNECTION_FAILED = (
    "Could not connect to Ollama OpenAI-compatible endpoint (local daemon not running or "
    "wrong host/port). Start with: ollama serve; pull the model if missing. "
    "Opt-in ollama_live tests — environment-dependent."
)

SKIP_OLLAMA_STREAM_EMPTY_DELTAS = (
    "Ollama streaming completed with zero parseable content deltas from the "
    "OpenAI-compatible SSE stream (Lens never yielded chunk events with body text). "
    "Not a governance assertion waiver: verify model pull, /v1/chat/completions stream "
    "shape, and adapter parsing. Opt-in ollama_live — environment/provider-dependent."
)


def skip_reason_live_proxy_at(url: str) -> str:
    """Human-readable skip when GET /health on the configured proxy base URL fails."""
    return (
        f"Live HTTP integration: proxy not reachable or unhealthy at {url} "
        f"(expected GET /health with JSON status ok). "
        "Start aurora-lens proxy first (see examples/); set AURORA_TEST_PROXY for a non-default URL. "
        "Tests using this are opt-in (finance_proxy_live / general_proxy_live) or CI-with-service. "
        "Environment-dependent until the proxy is running."
    )


# --- Continuation matrix row-shape conditionals (not environment) ------------

SKIP_MATRIX_INVITATION_SUPPRESSION_REQUIRES_CLOSED = (
    "Invariant checks that interaction_open=False suppresses continuation invitations. "
    "This matrix row has interaction_open=True, so the assertion does not apply. "
    "Would run for the same pathway key if matrix assigned interaction_open=False; "
    "intentional row-shape conditional skip (not CI/env)."
)

SKIP_MATRIX_NOT_CLEAN_STOP_ROW = (
    "Invariant targets continuation-matrix rows using P_STOP_REFUSE_CLEAN only. "
    "This row uses a different pathway. "
    "Runs when the parametrized row is a clean-stop pathway; intentional matrix conditional skip."
)

SKIP_MATRIX_NO_ROW_ESCALATION_TARGET = (
    "Invariant checks that rendered output contains the row escalation_target when the pathway surfaces it. "
    "This row has escalation_target=None in the matrix. "
    "Would run if the matrix row defined an escalation target; intentional conditional skip."
)

SKIP_MATRIX_CLEAN_STOP_SUPPRESSES_RESOURCE = (
    "P_STOP_REFUSE_CLEAN renderer suppresses policy resource strings by design; "
    "escalation_target in matrix is not expected in user-facing output for this pathway. "
    "Permanent governance contract; skip is row-shape driven (not environment)."
)

SKIP_MATRIX_HANDOFF_SUMMARY_NO_VERBATIM_RESOURCE = (
    "P_HANDOFF_SUMMARY does not surface a verbatim escalation_target resource string in user copy. "
    "Invariant resource-matching does not apply. "
    "Intentional pathway conditional skip."
)

SKIP_MATRIX_ASK_PATHWAY_NO_ESCALATION_IN_OUTPUT = (
    "P_ASK_DISAMBIGUATE / P_ASK_MISSING_FACT surface clarification prompts; escalation_target "
    "is audit/metadata for routing, not echoed as a contact resource. "
    "Would run for pathways that embed escalation_target in renderer output; intentional skip."
)

SKIP_MATRIX_CLARIFICATION_PATHWAY_NO_REFUSAL_VERBATIM = (
    "Refusal-language invariant targets blocking/refusal renderers. "
    "ASK and handoff pathways use clarification or summary shapes without the same refusal boilerplate. "
    "Intentional pathway conditional skip (not environment)."
)


def skip_reason_redis_unavailable(url: str) -> str:
    """Redis integration module: skip when ping fails or redis package missing."""
    return (
        f"Redis integration: no server response at {url} (or redis Python package not installed). "
        "Install redis>=5 (optional extra) and start Redis; set AURORA_LENS_TEST_REDIS_URL "
        "for a non-default DSN (default redis://127.0.0.1:6379/15). "
        "Marked @pytest.mark.redis — environment-dependent; CI often omits Redis. "
        "Not intentionally unsupported — runs when Redis is reachable."
    )

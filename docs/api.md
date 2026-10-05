# aurora-lens — API Reference

OpenAI-compatible endpoint with governance. All responses include Aurora-* headers.

## Endpoints

### POST /v1/chat/completions

OpenAI Chat Completions format. Governance is applied transparently; outcomes return HTTP 200 with metadata in the body.

**Request**

| Field | Type | Description |
|-------|------|-------------|
| `model` | string | Model name (passed to upstream) |
| `messages` | array | Chat messages (OpenAI format) |
| `stream` | boolean | Enable streaming |
| `aurora_session_id` | string | Session / PEF continuity handle (optional; legacy field name — see "Continuity handle" below) |
| `pef_context_id` | string | Session / PEF continuity handle (optional; canonical field name, same store key as `aurora_session_id`) |
| `aurora.external_flags` | array | Supplementary flags (optional) |

**Request headers**

| Header | Description |
|--------|-------------|
| `x-aurora-pef-context-id` | PEF continuity handle (canonical header; alternative to body field) |
| `x-aurora-session-id` | PEF continuity handle (legacy alias header; same value, checked if `x-aurora-pef-context-id` is absent) |
| `x-aurora-mock-hard-stop` | **Demo only.** Set to `1` to inject a canned HARD_STOP response. Only active when `governance.enable_mock_hard_stop: true` and `mode != enterprise`. **Quarantined:** When `mode: enterprise`, the header is ignored; production proxy path never honors it. Never set in production. |

**Continuity handle: `pef_context_id` vs `session_id` / `aurora_session_id`**

`pef_context_id` is the durable handle for a client's PEF (the persisted world-state — entities, relations, holds — that governs referential admission across turns). `session_id` and `aurora_session_id` are legacy names for the exact same store key; today the server always returns the same value under all three names, and accepts any of them back. New integrations should use `pef_context_id`.

**Recommended round-trip pattern:**

1. On the first turn, omit `pef_context_id` (or `aurora_session_id`) — the server mints one and returns it as `aurora.pef_context_id` in the response body.
2. Capture `aurora.pef_context_id` from every response.
3. On the next turn, send it back as the `x-aurora-pef-context-id` header (and/or `pef_context_id` in the body). If you send neither, the server treats continuation language (pronouns, demonstratives, "the transaction") as unresolved rather than guessing at a prior turn's world.

```python
resp = post_chat(messages, pef_context_id=pef_context_id)
pef_context_id = resp["aurora"].get("pef_context_id") or pef_context_id
```

See `tools/chat_with_lens.py` and `eval/chat.py` for working examples, and `docs/FRAME_LIFECYCLE_INVARIANT.md` for the full priority order across headers/body/legacy aliases.

**Session ID (legacy name for the same handle)**

- `x-aurora-session-id` header, or
- `aurora_session_id` in body

**Response (200)**

OpenAI format with `aurora` object. Two-plane output contract applies — see below.

```json
{
  "id": "...",
  "choices": [{"message": {"content": "...", "role": "assistant"}}],
  "aurora": {
    "governance": "PASS",
    "turn": 1,
    "session_id": "session-abc123",
    "pef_context_id": "session-abc123",
    "audit_id": "..."
  }
}
```

**Two-plane output contract**

aurora-lens separates what clients see from what operators audit.

**User plane** (always in response body):

| Field | Type | Description |
|-------|------|-------------|
| `governance` | string | `PASS` \| `SOFT_CORRECT` \| `FORCE_REVISE` \| `HARD_STOP` |
| `turn` | int | Turn counter within session |
| `session_id` | string | Session routing handle (present when session_id was provided or generated) |
| `pef_context_id` | string | Canonical name for the same handle as `session_id` (present under the same conditions) |
| `unverified` | bool | Present and `true` when governance is `SOFT_CORRECT` — indicates the response draws on general knowledge not grounded in the conversation state |
| `audit_id` | string | Opaque pointer for operator audit lookup (present when audit is configured) |
| `forensic_event` | object | Present on non-PASS outcomes that trigger intervention — canonical record (PEF snapshot, trigger spans, state hash, session id) for incident replay |

**Operator plane** (audit log only by default):

`flags`, `rationale`, `policy`, `governance_note`, `original_response` are written to the audit log on every intercepted turn. They are stripped from the response body to prevent internal diagnostics from leaking to end users.

To expose operator-plane fields in the response body (e.g. for operator dashboards), set `include_operator_detail: true` in the governance config section. This adds:

| Field | Type | Description |
|-------|------|-------------|
| `flags` | list[string] | Flag type names triggered |
| `rationale` | string | Human-readable governance rationale |
| `policy` | string | Policy name applied |
| `original_response` | string | Original model output before intervention |
| `governance_note` | string | Epistemic note for SOFT_CORRECT outcomes |

**Error responses**

| Status | Code | Description |
|--------|------|-------------|
| 400 | — | Invalid JSON, unable to read body |
| 401 | — | Missing API key (auth enabled) |
| 403 | — | Invalid API key |
| 409 | SESSION_BUSY_TIMEOUT | Lock acquire timeout |
| 413 | — | Payload too large |
| 422 | — | Invalid payload (messages, content length, external_flags) |
| 429 | — | Rate limit exceeded |
| 503 | session_store_error | Session store unavailable |

**Error body**

```json
{
  "error": {
    "message": "...",
    "type": "server_error",
    "code": "SESSION_BUSY_TIMEOUT",
    "trace_id": "proxy:..."
  }
}
```

### GET /healthz

Minimal liveness. Returns `{"ok": true}`.

### GET /health

Full health: status, sessions, policy, audit writability, extraction backend, and a coarse `release` object (`version`, `source_commit`, `built_at`, `release_tag`).

### GET /v1/operator/summary?n=30

Compact operator first-screen summary. Returns coarse aggregate counters and runtime labels only (no raw prompts/responses, file paths, or per-session identifiers). Includes a coarse `release` object with `version`, `source_commit`, `built_at`, and `release_tag`.

### GET /metrics

Prometheus exposition format.

### GET /v1/audit/recent?n=20

Last n audit entries. Requires `audit_log` configured. 404 if not configured.

On a loopback listen address (`127.0.0.1`, `localhost`, or `::1`) this route stays available to the local forensics page even when inbound auth is enabled. On any other listen address, including `0.0.0.0`, it requires the inbound API key when `auth.enabled` is true. `GET /v1/audit/search` follows the same rule.

### GET /v1/audit/verify?n=20

Verify HMAC and hash-chain integrity of last n entries. Works on both `jsonl` and `ledger` backends. Returns 400 if no `audit_signing_key` is configured (jsonl) or if the audit log is not configured.

**Response (jsonl backend)**

```json
{"verified": true, "entries": 20, "hmac_verified": true, "chain_verified": true, "backend": "jsonl"}
```

**Response (ledger backend)**

```json
{"verified": true, "entries": 20, "hmac_verified": null, "chain_verified": true, "backend": "ledger"}
```

(`hmac_verified` is `null` for the ledger backend — the ledger has its own internal signing; the field is absent rather than false so tooling can distinguish "not applicable" from "failed".)

**Response on failure**

```json
{"verified": false, "entries": 5, "hmac_verified": false, "chain_verified": true, "backend": "jsonl", "first_failed_entry": "<cid-of-first-failed-entry>"}
```

**Tail verification and `unanchored_slice`**

When `n` is smaller than the total number of log entries, the first entry in the verified window links to a predecessor that is outside the window. This is normal — the endpoint returns `reason: "unanchored_slice"`, `chain_verified: true` (the slice is internally consistent), and `verified: false` (global chain anchoring cannot be proved from a partial view). This is not a failure; it means the slice you asked for is intact. To obtain a fully anchored result, increase `n` to cover entries from genesis, use `GET /v1/audit/verify?n=<total>`, or switch to the ledger backend which maintains its own global chain state. You can also run `python -m aurora_lens.scripts.verify_audit --path audit.jsonl --key $KEY` from the start of the file for an offline full-chain verification.

```json
{"verified": false, "chain_verified": true, "hmac_verified": true, "reason": "unanchored_slice",
 "reason_detail": "The slice is internally consistent but not anchored to the chain head. ...",
 "entries": 20, "backend": "jsonl"}
```

## Aurora-* Headers

| Header | Type | Description |
|--------|------|-------------|
| Aurora-Outcome | string | PASS, STREAM, EXTRACTION_FAILED, ERROR |
| Aurora-Trace-Id | string | Request trace ID |
| Aurora-Audit-Sink | string | jsonl, ledger, none |
| Aurora-Audit-Id | string | Audit entry ID |
| Aurora-Timestamp | string | ISO 8601 UTC |
| Aurora-Upstream | string | openai_compat, claude |
| Aurora-Policy | string | strict, moderate, open |
| Aurora-Policy-Version | string | Policy version |
| Aurora-Session-Id | string | Session ID |
| Aurora-Proxy-Ms | string | Processing time (ms) |

## Streaming Format

SSE with `data: {...}` lines. Final event:

```
data: {"aurora": {"governance": "PASS", "turn": 1, "session_id": "...", ...}}
data: [DONE]
```

When stream is truncated: `stream_truncated: true`, `stream_dropped_chars: N` in the aurora metadata event.

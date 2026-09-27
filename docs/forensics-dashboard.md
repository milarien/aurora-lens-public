# Using the forensics dashboard

This page explains the **Forensics console** in plain English: what it is, how to open it, what each part does, and what to do when something looks wrong.

The browser UI is **not** labeled with a product version (no “v2” in the page). Paths such as **`/v1/audit/recent`** use the same **`/v1`** prefix as **OpenAI-compatible** endpoints (e.g. **`/v1/chat/completions`**) on this proxy—that is **wire interoperability**, not an “old dashboard” generation.

## What it is

The dashboard is a **simple web page** built into the Aurora Lens proxy. It does not replace your monitoring stack. It gives operators a quick way to see:

- whether the proxy is healthy and where it is writing audit logs  
- whether the **tamper-evident audit chain** verifies (for the configured log)  
- a **recent slice** of governance decisions  
- optional **session state** (PEF) for a single chat session  
- raw **Prometheus metrics** text  

The page is read-only. It calls the same HTTP APIs you could call with `curl`; it just formats the answers for a browser.

The first screen is intentionally minimal:

- **Operator overview** (system, model, policy, recent decisions, warnings/blocks, sessions, audit status)
- **Recent governance decisions**
- **Advanced forensics** (collapsed by default) for runtime internals, session PEF, verification detail, and metrics

## How to open it

1. **Start the proxy** the way you normally do (for example `python run_aurora_lens.py` or your container entrypoint).  
2. **Point your browser** at the proxy’s base URL and one of these paths (they all show the same page):

   - **`/forensics`** (recommended name)  
   - **`/dashboard`** (older name)  
   - **`/operator`** (alias)

Examples:

- Proxy on the same machine: `http://127.0.0.1:8081/forensics`  
- Replace `8081` with whatever port your config uses (`listen.port` or `AURORA_LENS_LISTEN_PORT`).

You do **not** open the `dashboard.html` file from disk. The running proxy **serves** that file from the package when you visit those URLs.

## What you will see (top to bottom)

### Operator overview

This is the default at-a-glance strip for operators. It answers:

- is the system healthy?
- what model and policy are active?
- how many recent decisions were made?
- how many need attention?
- is audit data available and writable?

It also shows active sessions and short run id.

The tile data is served by **`GET /v1/operator/summary`** (read-only).

Use this to confirm the surface is healthy before drilling into internals in **Advanced forensics**.

### Runtime

This block loads **`/health`**. It shows things like:

- **Status** — whether the proxy reports itself as ok, degraded, or bad  
- **Policy and version** — what policy profile and version the deployment is using  
- **Audit sink** — for example `jsonl` or `ledger`  
- **Audit log path** — the **resolved** path on disk where the proxy is appending audit lines (useful after restarts or path templates)  
- **Proxy run ID** — an id for this process, useful when you correlate logs  
- **HMAC signing** — whether a signing key is configured for audit integrity  

**API links** in this section are shortcuts to the same host: recent audit rows, verify, metrics, health, and operator PEF. They are normal links you can open in a new tab.

### Session PEF

This section is for **one conversation at a time**. The proxy keeps in-memory **session state** (world model / hold / clarification). That state is not in the audit file by itself; this view lets you inspect it for debugging.

1. Copy **`aurora_session_id`** from an API response header or body (chat completions).  
2. Paste it into the box and click **Load** (or press Enter).  
3. Optionally add **`?session=<id>`** to the dashboard URL so the id is pre-filled.  

If the session expired or the id is wrong, you will see a short error message instead of data.

### Chain integrity

This section calls **`/v1/audit/verify`** with a fixed window size (`n=500` in the page). It shows **integrity ok** (tamper-evident chain/HMAC for your backend) and whether **all checks ok** (integrity plus provenance `chain_of_custody`, PEF linkage, and state-hash replay over the same window), plus a short **provenance summary** and one-line detail for PEF/state-hash/coc counts.

- **Integrity ok** — tamper-evident verification passed for that window.  
- **All checks ok** — integrity and the extended semantic checks in the API response all passed.  
- **Extended pending/fail** — integrity may have passed but a provenance/PEF/replay check failed, or integrity failed.  
- **Failed** (integrity) — something did not verify; read the details below the badge.  
- **n/a** — no audit log path is configured.  
- **Verify skipped** — often means **no signing key** is configured for JSONL verification; the message explains what is missing.  

When verification fails, you may see a plain-language **`operator_message`** in a box below. That text is meant for humans (what broke, what to check next), not only error codes.

**Important:** Verifying only the **last N lines** can show **`unanchored_slice`** in the API: the slice is internally consistent but its first line points to a parent **outside** the window. That is not always a sign of tampering; it can mean you need a larger `n` or a full-file check to prove the whole file. The CLI `python -m aurora_lens.scripts.verify_audit` can do deeper checks.

### Recent audit rows

This section calls **`/v1/audit/recent`** and fills a table: time, **row kind** (`jsonl` vs `ledger`), outcome, **Evidence** (complete/degraded provenance when `chain_of_custody` is present), forensic flag, constraints, trace id, tenant, and a short **CID** with **copy** and **JSON** links where the backend supports it.

- **JSON** (flat JSONL backend) opens **`/v1/audit/entry?cid=...`** when the row has a D2 **`cid`**. Rows without **`cid`** (older lines) use **`/v1/audit/entry?trace_id=...`** and, when needed, **`&timestamp=...`** matching the row’s **`timestamp`** field exactly.  
- **AFL** (ledger backend) may show a label instead of a JSON link because row shape differs.  

Clicking a row in the browser uses the same lookup rules and merges **`forensic_event`** into the side panel when top-level governance fields are absent.  

If you see **no rows** or an error, check that an audit log is configured and that your browser is allowed to call **`/v1/audit/recent`** (see **Authentication** below).

### Metrics

This section loads **`/metrics`** and shows the raw Prometheus text inside a collapsible block. Use it for a quick glance or paste into your metrics stack; it is not a full charting UI.

## Automatic refresh

The dashboard **reloads health, chain integrity, and recent rows about every 8 seconds** (with **no-cache** fetches so the browser does not reuse stale JSON). **Metrics** reload about every **30 seconds**. Use **Refresh audit** for an immediate pull of health, verify, and recent rows. Session PEF does **not** auto-refresh; click **Load** again after new turns.

## Authentication

If **`governance.auth.enabled`** is **true**, **`POST /v1/chat/completions`** still requires **`Authorization`** or **`x-api-key`**. The dashboard’s same-origin fetches to **`/health`**, **`/v1/audit/recent`**, **`/v1/audit/verify`**, **`/v1/operator/summary`**, and other read-only audit **`GET`** routes used by this page bypass inbound API-key middleware. Restrict **network** access to the proxy if the audit log must not be readable on an untrusted LAN.

`/v1/operator/summary` is intentionally limited to coarse aggregate fields only:

- system status
- model id
- policy + policy version
- decision window size and aggregate counts
- sessions count
- audit sink/writability
- proxy run id

It must not emit prompts/responses, API keys, paths, per-session identifiers, per-user identifiers, or raw exception payloads.

The **auth note** on the page states the same contract.

## If something looks wrong

1. **Confirm the audit path** on disk matches what you expect (`/health` → audit log path).  
2. **Chain failed at line 1 (`prev_mismatch`)** on a ledger file usually means the file does not start with a valid first link (truncated, merged, or corrupted). To start clean: stop the proxy, rename or move the file (for example to `audit-old.jsonl` or an `archive/` folder), leave **no file** or an **empty** file at the configured path, then restart.  
3. **Run the offline verifier** on the file:  
   `python -m aurora_lens.scripts.verify_audit --path <your-audit.jsonl> --key <key>`  
   Optional flags: `--pef-linkage`, `--chain-of-custody`, `--state-hash-replay` (see **`docs/api.md`** for the HTTP verify endpoint this page calls).  
4. **Chain failed but you only verified a tail** (flat JSONL) — use a larger `n` on `/v1/audit/verify` or verify from the start of the file.  
5. **HMAC failed** — key mismatch, rotation, or a line was edited; align keys with the writer.  

## Where this fits in the docs

- **Install and configure the proxy**: **`README.md`**  
- **Configuration fields**: **`docs/config_reference.md`**  
- **HTTP API (audit routes)**: **`docs/api.md`**  

The dashboard is a **convenience** for the same evidence story; it does not change the meaning of the audit log or the verification rules.

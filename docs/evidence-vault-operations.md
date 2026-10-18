# Sealed evidence deployment and verification

Set `AURORA_LENS_EVIDENCE_CAPTURE_MODE=sealed` and provision a stable
`AURORA_LENS_EVIDENCE_KEY` in the deployment's secret store. Generate the secret
with a cryptographically secure generator (32 random bytes encoded as hex),
and transfer it directly through a secret-manager API or CLI stdin. Never put
the value in command arguments, source control, terminal output or logs.
Reuse the secret across restarts; do not generate a new key on each startup.

`sealed` uses `AELV1-HMAC-CTR`, not AES. The implementation derives a 32-byte
key with PBKDF2-HMAC-SHA256 (200,000 iterations) and a random 16-byte salt,
uses an HMAC-SHA256 counter keystream with a random 16-byte IV, and authenticates
the salt, IV and ciphertext with HMAC-SHA256. The configured key is interpreted
as UTF-8 bytes; a hexadecimal value is not decoded into binary first.

On Railway, attach a persistent volume to the proxy service at `/data` and set
`AURORA_LENS_GOV_AUDIT_LOG=/data/audit.jsonl`. The vault is automatically located
at `/data/audit.jsonl.evidence/`; both `manifests.jsonl` and `blobs/` must persist.
Redis session persistence is separate from evidence storage. Preserve the
volume, path and key across redeploys, and retain secure backups of both the
vault and its key. Key rotation does not re-encrypt existing blobs or provide
automatic historical-key lookup.

Volumes are mounted at runtime. Check writes as the application's `aurora`
user, including blob creation and manifest-index append permissions. If an
ownership repair is necessary, limit it to the vault directory, its `blobs/`
directory and manifest index. The current entrypoint drops privileges without
repairing ownership; setting a root container UID alone does not change that.
See [Railway volumes](https://docs.railway.com/volumes).

Run the following inside the deployed image as `aurora`, with the normal
deployment environment and config, using the verification script from the
matching repository checkout:

```sh
python scripts/verify_sealed_evidence.py capture
# Restart/redeploy using the same volume, audit path and key.
python scripts/verify_sealed_evidence.py recover
```

The check uses synthetic request and withheld-response fixtures, the production
HARD_STOP capture function, and the configured vault. It does not call an
upstream model or append synthetic decisions to the governance ledger. It
checks encrypted blobs, authorised recovery, SHA-256 integrity, unauthorised
access denial, and exclusion of inline content from the constructed audit row.
It prints only PASS/FAIL and leaves a synthetic manifest marker in the vault
for the recovery phase. A restart between phases must be verified separately
by observing a new deployment instance; reopening a vault alone is not proof
of a service restart.

`get_evidence(ref, purpose, actor)` is an in-process interface. Trusted operator
tooling must authenticate the caller before supplying an allowed actor/purpose
pair; the strings are not authentication credentials. The authorised decision
review workflow below uses this interface behind mandatory server authorisation.
Return recovered content only to its authorised
destination, never to ordinary application logs or public audit responses.

## Authorised decision evidence review

Open **Review captured decision evidence** from the owner audit session page or
retained-log page, then **Open decision**. The hosted route is
`/admin/audit/review`. It uses the existing Cloudflare Access owner sign-in;
reviewers do not supply an encryption key or run Python. The page shows the
recorded ruling, captured request, withheld model response, SHA-256/byte-length
verification, and any released answer recorded in the audit event.

Evidence states are explicit:

- **Verified**: recovered bytes match the audit manifest and vault manifest,
  SHA-256 and byte length. Sealed content also passes the vault MAC check.
- **Hash only**: a digest was captured; there is no recoverable blob. It is not
  labelled verified merely because a digest exists.
- **Unavailable**: no capture, missing blob/key/storage, or unavailable access
  auditing. Original records are not rewritten or recaptured.
- **Access denied**: the caller lacks reviewer authorisation, or vault access
  was denied. No evidence plaintext is returned for that item.
- **Integrity failed**: inconsistent manifests, corrupted ciphertext, wrong
  decryption key, or recovered bytes that fail the hash/length check. Content
  is withheld.

The hash result verifies the blob against its manifests. It does not assert that
the entire governance audit chain verified; use the existing chain-verification
workflow for that separate check.

The backend route `/v1/audit/evidence-review` is **not public**, including on
loopback or when ordinary inbound authentication is disabled. Hosted access
requires both the existing server transport credential and a short-lived,
selection-bound HMAC assertion of the owner identity. The site signs it only
after verifying the existing Access JWT. The assertion is purpose-bound,
expires after 30 seconds, and rejects nonce reuse within a backend process.
It uses a domain-separated HMAC key derived from the existing server transport
credential; no new persistent secret or evidence key is created. The site's
existing trusted credential remains privileged and must stay server-side.
The browser's cookies, JWT and claimed identity headers are not forwarded to
Railway. Redirects are never followed with the transport credential.

For a licensed local install, enable existing inbound API authentication and
set `AURORA_LENS_EVIDENCE_REVIEWERS` to a comma-separated allowlist of existing
API-key labels. Open `/v1/audit/evidence-review` in the browser; its Basic-auth
prompt accepts that existing API credential as the password (the username is
only a browser-login label). Use HTTPS outside loopback. Ordinary API keys
whose labels are not explicitly allowed cannot review evidence. Local-key
review is disabled on hosted Railway deployments.

Every backend attempt, including denials, is appended and fsynced to
`<audit_log>.evidence-access.jsonl` before a response is released. Events contain
time, authenticated identity (or `unauthenticated`), purpose, decision digest,
result and capture states. They contain no prompts, outputs, credentials or
exception text. This separate access log does not append to or modify the
governance decision chain and is excluded from ordinary retained-audit exports.
New access-log files use mode 0600. Missing/unwritable access auditing fails
closed. Site-side owner denials emit only a fixed content-free denial code.

Views use private/no-store responses, no-referrer policy, escaped text and a
script-free CSP. No plaintext is written to ordinary logs, browser storage or
an export file by this workflow. Do not configure external request/response-body
logging on the protected review routes. The existing key, volume, vault blobs
and manifests are read in place; retrieval does not rotate or regenerate them.

### Local synthetic preview and verification

```sh
python scripts/preview_evidence_review.py --run
python -m pytest tests/test_evidence_review.py tests/test_evidence_vault.py
node --test tests/evidence_review_site.test.mjs
```

The preview binds only `127.0.0.1:8766`, creates a temporary synthetic vault,
and clearly labels the simulated reviewer identity. It includes verified,
hash-only, unavailable and denied examples. It cannot run inside Railway and
does not read the production key, audit file or volume. Starting it is opt-in;
it is not part of normal startup or CI. Stop it with Ctrl+C.

The companion site's `functions/lib/evidence_review.js` mirrors
`integrations/cloudflare/evidence_review.js`. Its `/admin/audit/review` wrapper
supplies the existing `verifyOwner`, `ownerResponse` and verified owner email.
Its synthetic JWT tests also cover wrong-owner, expired and forged credentials.
With the preview running, set `AURORA_REVIEW_TEST_BACKEND=http://127.0.0.1:8766`
for the site's `tests/evidence_review.test.mjs` to exercise signed owner login,
real local vault recovery and denied access end to end. That optional test
rejects non-loopback targets; it cannot target production.

When audit capture is enabled, required `sealed` capture rejects startup if the
key is missing/empty or audit/vault storage is unavailable. No key is generated
automatically and no hash-only or null-audit fallback satisfies that requirement.
An unset audit path leaves capture disabled; explicit hash-only capture and the
existing low-risk PASS hash-only policy remain supported.

Structured startup diagnostics and diagnostic `GET /health` expose
`evidence_capture`: `enabled`, `requested_mode`, `effective_mode`, `key_available`,
`vault_available`, `required_sealed`, `ready` and a typed `failure_reason`.
Key availability is a boolean; no key value, fingerprint or raw content is
included. Hosted-demo health retains its existing diagnostic credential boundary.
`GET /readyz` returns only a readiness boolean and HTTP 503 when capture storage
is unavailable. Required sealed storage loss also makes `/health` return 503;
`/healthz` remains a liveness check. Probes perform real temporary writes and
append-open checks without adding evidence content.

Earlier hash-only records still contain no recoverable content. Enabling sealed
capture cannot recover them. See the [implementation record](sealed-evidence-missing-key-proposal.md).

## Opt-in public HTTP capture and restart smoke test

From an operator checkout with an authenticated Railway CLI account authorised
to inspect variables, use SSH and restart the demo service:

```sh
python scripts/smoke_http_sealed_evidence.py --run --project <railway-project-id>
```

`--run` and an explicit project UUID are required. Without them the command
prints `FAIL`, exits nonzero and performs no network operations. Default pytest
and CI exercise only offline tests; they do not invoke the live smoke test.
The script prints exactly one line (`PASS` or `FAIL`), exits 0 or 1, and never
prints subprocess output, credentials, evidence references or raw content.

Defaults target the existing demo: service `aurora_lens`, environment
`production`, public origin `https://aurora-lens.ai`, Railway origin
`https://auroralens-production.up.railway.app`, and a 300-second restart wait.
Override with `--service`, `--environment`, `--public-origin`, `--railway-origin`,
`--timeout` (30–1800 seconds), or `--railway-cli` (executable path). HTTPS is
required; redirects are rejected. The Railway origin must belong to the
explicitly selected service. A temporary CLI link isolates this test from the
operator's existing checkout link and is removed afterwards.

The test performs one normal public HTTP chat turn in a fresh demo session.
Its fixed synthetic fixture asks for a label for an invented medicine and an
imaginary adult, without using real patient data. It requires an actual
post-LLM `HARD_STOP`, distinct sealed request/upstream manifests and a
`sealed_by_policy` receipt on the uniquely correlated audit row. It never
enables a mock response, supplies a policy override, weakens a gate or retries
the model request to obtain a desired result. If the provider produces a
different governance outcome, the smoke test fails before restarting.

Before restart, trusted SSH tooling runs as `aurora` and checks both exact
references through `get_evidence(..., purpose="forensic_review",
actor="forensic_reviewer")`. It checks recovered lengths and hashes against
the audit manifests, encrypted blob format, and denial of an unauthorised
actor. It then restarts the existing deployment and requires the same code
revision, original volume/mount, audit path and evidence key, healthy sealed
readiness, and a new deployment/process identity. It rechecks the unchanged
original audit row and recovers those same two references again.

Key comparisons remain in memory; keys, fingerprints and evidence are never
written to local reports or logs. No key is generated, provisioned or rotated;
no variable or volume configuration is written. The normal audit/vault records
remain subject to the existing retention policy. No additional evidence or
manifest marker is written by the recovery checker. A failed verification after
restart still prints `FAIL`; it does not roll back or change capture policy.

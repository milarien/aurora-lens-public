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
pair; the strings are not authentication credentials. There is no evidence-vault
retrieval HTTP endpoint. Return recovered content only to its authorised
destination, never to ordinary application logs or public audit responses.

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

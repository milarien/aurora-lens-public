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

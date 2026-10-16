# Key Custody

## Evidence Vault Encryption Key (`AURORA_LENS_EVIDENCE_KEY`)

The evidence vault stores raw prompts and upstream model outputs alongside the
audit log. In `sealed` capture mode every blob uses the implementation's
`AELV1-HMAC-CTR` format before being written to disk. This is not AES: it uses
PBKDF2-HMAC-SHA256 (200,000 iterations, 32-byte derived key), a random 16-byte
salt and 16-byte IV, an HMAC-SHA256 counter keystream XORed with the content,
and an HMAC-SHA256 over the salt, IV and ciphertext. The controlling secret is
`AURORA_LENS_EVIDENCE_KEY`.

### Key properties

- **Length**: any non-empty UTF-8 string is accepted; the runtime derives a
  32-byte key from the UTF-8 bytes internally (a hex string is not hex-decoded).
  Use a securely generated 32-byte random value encoded as hex, provisioned
  directly into the secret store without displaying it in terminal output.
- **Storage**: treat it with the same access controls as `AURORA_LENS_AUDIT_SIGNING_KEY`.
  Store it in a secrets manager (Vault, AWS Secrets Manager, Railway env, etc.)
  and inject it via the `AURORA_LENS_EVIDENCE_KEY` environment variable. Never
  commit it to source control or put it in a YAML file checked into a shared
  repository.
- **Required capture**: when an audit path is configured and
  `evidence_capture_mode: sealed`, a missing/empty key or unavailable audit/vault
  storage rejects startup with `EvidenceVaultConfigurationError`. There is no
  automatic downgrade to `hash_only`. An unset audit path disables capture;
  explicit hash-only capture remains an operator choice.

### `sealed` vs `hash_only` — what a regulator or auditor actually receives

| Mode | What is stored | What a reviewer can retrieve |
|------|---------------|------------------------------|
| `sealed` | Encrypted blob at `{audit_log}.evidence/` | Full plaintext prompt + model output, after key is provided |
| `hash_only` | SHA-256 digest only in the audit row | Proof that specific content was processed; plaintext is not recoverable |
| `redacted` | Blob with PII-style fields scrubbed | Partial content; redaction policy governs what is visible |
| `plaintext_dev` | Unencrypted blob | Full plaintext (dev/test only; do not use in production) |

For regulatory or legal discovery use `sealed`. For environments where storing
raw model output is prohibited, use `hash_only` or `redacted`.

### Provisioning

```bash
# Generate a key and export it to your secrets manager (example):
export AURORA_LENS_EVIDENCE_KEY="$(python -c "import secrets; print(secrets.token_hex(32))")"
# Or, in aurora-lens.yaml:
governance:
  evidence_capture_mode: sealed
  evidence_encryption_key: ${AURORA_LENS_EVIDENCE_KEY}
```

### Rotation procedure

Evidence vault blobs are encrypted at rest with a static key per deployment.
The runtime does **not** re-encrypt existing blobs when the key changes.

1. **Stop writes**: bring down the proxy (or set `evidence_capture_mode: hash_only`
   temporarily to pause sealed captures).
2. **Archive the old vault directory** (`{audit_log}.evidence/`). Keep it in
   secure storage alongside the old key — you will need both to decrypt existing
   blobs for any future review.
3. **Provision the new key** via `AURORA_LENS_EVIDENCE_KEY` or your secrets
   manager.
4. **Start a new vault** by either clearing (or renaming) the existing vault
   directory, or pointing `governance.audit_log` to a new path so the vault
   directory is also new.
5. **Restart the proxy**. New sealed blobs are written with the new key.

Old blobs under the archived directory are still readable with the old key.
Cross-key reads are not supported by the runtime; use the old key material
directly if you need to decrypt archived blobs.

### Restart / redeploy behaviour for existing vault blobs

- **Key present, same path**: existing blobs in `{audit_log}.evidence/` are
  readable with the current key. New requests append additional encrypted blobs.
- **Key absent on restart** (with no YAML key fallback): required sealed capture
  rejects startup. Existing sealed blobs remain untouched. Restore the original
  key to recover existing evidence and resume capture; do not generate a
  replacement key for existing blobs.
- **New `audit_log` path**: a new vault directory is created alongside it.
  Blobs in the old vault directory are not migrated.
- **Image redeploy with same key, same `audit_log` path and persistent storage**:
  blobs from prior `proxy_run_id` instances remain accessible under the same
  vault directory. An unchanged path on an ephemeral container filesystem is
  not sufficient. On Railway, persist both the audit log and adjacent vault on
  the attached volume; see [Evidence vault operations](evidence-vault-operations.md).

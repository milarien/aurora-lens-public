# Required sealed evidence: implementation record

Status: implemented in aurora-lens and aurora-lens-public. State observability
and its tests were added first, followed by failing enforcement tests and then
the startup/capture changes described below.

## Previous behaviour

Previously, both `build_evidence_vault_for_bridge()` in `evidence_capture.py` and
`create_local_evidence_vault()` in `evidence_vault.py` converted requested
`sealed` capture to `hash_only` when the resolved key is missing or empty.
`_vault_protection_mode()` also substitutes hash-only storage when a local vault
cannot provide sealing. The result reports the actual manifest protection mode,
which loses the distinction between a deliberately selected hash-only policy
and an unmet sealed-capture requirement. No recoverable content is stored in
the downgrade case.

`LocalSealedEvidenceVault` already rejected unkeyed sealed construction; its
factories bypassed that error by changing the mode first. The proxy also caught
vault/audit filesystem errors as `OSError` and disabled the audit sink. Both
fallbacks are now rejected when sealed capture is required.

## Implementation order

1. Add explicit deployment capture state: requested mode, effective mode,
   key availability as a boolean, vault availability, and a typed failure
   reason such as `missing_key` or `storage_unavailable`. Keep capture
   requirements separate from Lens admissibility and Governor decisions.
2. Surface that state in structured startup diagnostics and health/readiness,
   without key values, key fingerprints or raw evidence. Preserve existing
   receipt fields and add requested/effective state rather than relabelling
   historical receipts.
3. Add tests for absent/empty keys, both factories, proxy startup/readiness,
   storage errors, explicit hash-only configuration, keyed recovery across
   restart, and exclusion of secrets/content from diagnostics. Preserve the
   existing intentionally hash-only low-risk PASS policy.
4. When audit capture is enabled and the deployment requests `sealed`, reject
   missing keys before accepting traffic with `EvidenceVaultConfigurationError`;
   do not substitute `hash_only`. Reject unavailable required storage instead
   of falling back to a null audit sink. Ensure bridge construction and the
   per-capture helper cannot bypass this requirement.

Explicit `hash_only` remains a supported operator choice. Installs with no
audit capture configured remain explicitly capture-disabled; this proposal
does not silently enable storage or change Lens/Governor decisions. Document
that a configured evidence sink requiring `sealed` must be operational before
the service becomes ready. No automatic key generation at startup: replacing
the key would strand existing evidence.

The same implementation and tests are present in both repositories. Existing
hash-only records cannot be converted into recoverable sealed evidence.

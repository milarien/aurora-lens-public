# Aurora-Lens Evidence Model

## Evidence capture and deployment readiness

A companion field, `evidence_capture_mode` (`sealed` \| `redacted` \| `hash_only` \| `plaintext_dev`), records the request manifest's effective protection mode (or the selected mode when no request manifest exists). Required sealed deployments now reject startup for missing keys or unavailable storage. The intentional low-risk PASS policy can still produce a hash-only request within a ready sealed deployment. Diagnostic `/health` exposes deployment `evidence_capture.requested_mode` and `effective_mode` separately from these per-request receipt fields. Historical hash-only records are unchanged and do not become recoverable.

Request capture status describes stored evidence; it is separate from chain-of-custody packaging and instrument provenance. See [Evidence vault operations](evidence-vault-operations.md) for deployment readiness and recovery checks.

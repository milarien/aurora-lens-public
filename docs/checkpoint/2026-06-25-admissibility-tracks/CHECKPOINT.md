# Admissibility Tracks Checkpoint — 2026-06-25

> **Historical snapshot (2026-06-25):** Sealed checkpoint record. For current v1
> status use [`CURRENT_STATE.md`](../../../CURRENT_STATE.md) and
> [`closed-decisions.md`](../../closed-decisions.md). Body text below is preserved
> as written at seal time.

**Purpose:** Sealed record of completed Track A, Track B, and Track C work. Forensic legibility — not a release announcement.

**Verify this checkpoint:**

```bash
python scripts/run_admissibility_checkpoint.py
```

Expected: `CHECKPOINT PASSED — Track A + B + C admissibility arc sealed (197 + 68 + 77 = 342 passed).`

**One-line status:** [`CURRENT_STATE.md`](../../../CURRENT_STATE.md)

---

## What this checkpoint seals

| Track | Scope | Status |
|---|---|---|
| **A** | Establishment-failure generators (5 kinds) + epistemic gate parity | **Sealed** |
| **B** | Sovereign Provider Registry Phases 1–3 | **Sealed** |
| **C** | Trust Registry / `source_untrusted` | **Sealed** |

**Combined test sign-off (2026-06-25):** Track A **197** + Track B **68** + Track C **77** = **342 total**.

**Re-verified (2026-07-02):** `python scripts/run_admissibility_checkpoint.py` → `CHECKPOINT PASSED - Track A + B + C admissibility arc sealed (197 + 68 + 77 = 342 passed).`

---

## Track A — Establishment-failure generators

**Manifest:** [`eval/establishment_failure_regression.manifest.yaml`](../../../eval/establishment_failure_regression.manifest.yaml)

| Kind | Posture |
|---|---|
| `identity_or_referent_unresolved` | `deterministic_generated` |
| `stale_evidence` | `deterministic_generated` |
| `scope_mismatch` | `deterministic_generated` |
| `threshold_not_met` | `deterministic_generated` |
| `inferential_gap` | `deterministic_generated` |
| `source_untrusted` | `external_declared_only` in Phase 2 table (Track C owns generator) |

**Primary code:** `aurora_lens/pef/uncertainty_analysis.py`, `aurora_lens/govern/epistemic_uncertainty_gate.py`

---

## Track B — Sovereign Provider Registry

**Manifest:** [`eval/sovereign_provider_registry_regression.manifest.yaml`](../../../eval/sovereign_provider_registry_regression.manifest.yaml)

| Phase | Deliverable |
|---|---|
| 1 | Failover bridge enforcement |
| 2 | `provider_route` audit envelope |
| 3A | Deterministic refusal templates |
| 3B | Proxy auto-hook / route metadata |
| 3C | Validation freshness + regression evidence (placeholder runner) |

**Primary code:** `aurora_lens/sovereign/`, `aurora_lens/lens.py`, `aurora_lens/proxy/provider_route_hook.py`

**Locked rules:**

- No silent failover
- No unrecorded provider substitution
- Sovereign routes require registry evaluation
- `adapter_called: false` on refused routes (audit-proof)

**Proof scenario:** [`docs/DEMO_SOVEREIGN_FAILOVER.md`](../../DEMO_SOVEREIGN_FAILOVER.md)

---

## Track C — Trust Registry / source_untrusted

**Manifest:** [`eval/trust_registry_track_c_regression.manifest.yaml`](../../../eval/trust_registry_track_c_regression.manifest.yaml)

**Primary code:** `aurora_lens/trust/`, `aurora_lens/pef/uncertainty_analysis.py`, `aurora_lens/config.py`, `aurora_lens/lens.py`

**Locked rules:**

- `trust_evaluation_required` pathway flag (anti-soup)
- Complete trust contract required
- Unknown source ≠ untrusted
- Track A Phase 2 posture table unchanged

**Docs:** [`docs/trust_registry_track_c.md`](../../trust_registry_track_c.md)

---

## Packaging (outer shell)

| Document | Role |
|---|---|
| [`CURRENT_STATE.md`](../../../CURRENT_STATE.md) | One-line + track statuses |
| [`docs/CAPABILITY_STATUS.md`](../../CAPABILITY_STATUS.md) | Implemented / deferred checklist |
| [`docs/SOVEREIGN_PROVIDER_REGISTRY_OVERVIEW.md`](../../SOVEREIGN_PROVIDER_REGISTRY_OVERVIEW.md) | Architecture (plain English) |
| [`docs/DEMO_SOVEREIGN_FAILOVER.md`](../../DEMO_SOVEREIGN_FAILOVER.md) | Runnable proof + audit fields |
---

## Explicitly deferred (next-track boundary)

Not missing from this checkpoint — out of scope for the sealed admissibility arc (see [`PHASE_NAMESPACE.md`](../../PHASE_NAMESPACE.md)):

1. **Product Phase 3 entrypoints** — sealed separately (35 tests — [`checkpoint/2026-06-25-product-phase-3/README.md`](../2026-06-25-product-phase-3/README.md)); **Corpus product UX** — sealed (12 tests — [`checkpoint/2026-06-25-corpus-product-ux/README.md`](../2026-06-25-corpus-product-ux/README.md))
2. **Live provider HTTP (operator)** — runner implemented; CI uses mocked HTTP ([`LIVE_PROVIDER_REGRESSION.md`](../../LIVE_PROVIDER_REGRESSION.md))

---

## Combined manifest

[`eval/admissibility_tracks_checkpoint.manifest.yaml`](../../../eval/admissibility_tracks_checkpoint.manifest.yaml)

---

## Deep roadmaps (historical design record)

- [`docs/epistemic_establishment_failure_roadmap.md`](../../epistemic_establishment_failure_roadmap.md)
- [`docs/sovereign_provider_registry_roadmap.md`](../../sovereign_provider_registry_roadmap.md)

---

*Checkpoint package created 2026-06-25. Track C sealed into combined checkpoint same date. Re-verify with `python scripts/run_admissibility_checkpoint.py` after any change to Track A/B/C code paths.*

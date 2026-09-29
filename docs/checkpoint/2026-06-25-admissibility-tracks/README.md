# Checkpoint Package — Admissibility Tracks (2026-06-25)

> **Historical snapshot (2026-06-25):** Sealed checkpoint package. Current authority:
> [`CURRENT_STATE.md`](../../../CURRENT_STATE.md) and
> [`closed-decisions.md`](../../closed-decisions.md).

Entry point for the sealed **A + B + C** admissibility arc.

## Start here

1. Read **[CURRENT_STATE.md](../../../CURRENT_STATE.md)** — one-line status + track table.
2. Read **[PHASE_NAMESPACE.md](../../PHASE_NAMESPACE.md)** — qualified phase names (never bare “Phase 3”).
3. Read **[CHECKPOINT.md](CHECKPOINT.md)** — what this checkpoint sealed (historical record).
3. Run verification:
   ```bash
   python scripts/run_admissibility_checkpoint.py
   ```
   Expected: **197 + 68 + 77 = 342 passed**.

## Manifests

| File | Track |
|---|---|
| [`eval/admissibility_tracks_checkpoint.manifest.yaml`](../../../eval/admissibility_tracks_checkpoint.manifest.yaml) | Combined A+B+C |
| [`eval/establishment_failure_regression.manifest.yaml`](../../../eval/establishment_failure_regression.manifest.yaml) | A |
| [`eval/sovereign_provider_registry_regression.manifest.yaml`](../../../eval/sovereign_provider_registry_regression.manifest.yaml) | B |
| [`eval/trust_registry_track_c_regression.manifest.yaml`](../../../eval/trust_registry_track_c_regression.manifest.yaml) | C |

## Do not reopen without cause

Tracks A, B, and C stay closed unless the next fork explicitly expands scope. Point here, run the script, attach pytest output.

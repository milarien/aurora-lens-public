# Checkpoint Package — Product Phase 3 (2026-06-25)

> **Historical snapshot (2026-06-25):** Sealed checkpoint package. Current authority:
> [`CURRENT_STATE.md`](../../../CURRENT_STATE.md) and
> [`closed-decisions.md`](../../closed-decisions.md).

Entry point for sealed **document ingestion + intake translator** entrypoints.

## Start here

1. Read **[CURRENT_STATE.md](../../../CURRENT_STATE.md)** — v1 status and sealed tracks.
2. Read **[PHASE_NAMESPACE.md](../../PHASE_NAMESPACE.md)** — qualified phase names (never bare “Phase 3”).
3. Read **[CHECKPOINT.md](CHECKPOINT.md)** — what this checkpoint sealed (historical record).
4. Run verification:
   ```bash
   python scripts/run_product_phase_3_checkpoint.py
   ```
   Expected: **8 + 11 + 11 + 5 = 35 passed**.

## Manifest

| File | Scope |
|---|---|
| [`eval/product_phase_3_checkpoint.manifest.yaml`](../../../eval/product_phase_3_checkpoint.manifest.yaml) | Combined Product Phase 3 checkpoint |

Legacy alias (deprecated): [`eval/product_phase_3_document_ingestion.manifest.yaml`](../../../eval/product_phase_3_document_ingestion.manifest.yaml)

## Do not reopen without cause

Entrypoints stay closed unless the next fork explicitly expands scope (e.g. corpus UX, automatic wiring). Point here, run the script, attach pytest output.

**Admissibility arc (342)** remains a separate checkpoint: [`../2026-06-25-admissibility-tracks/README.md`](../2026-06-25-admissibility-tracks/README.md)

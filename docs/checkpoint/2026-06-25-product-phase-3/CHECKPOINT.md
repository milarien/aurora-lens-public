# Product Phase 3 Checkpoint — 2026-06-25

> **Historical snapshot (2026-06-25):** Sealed checkpoint record. For current v1
> status use [`CURRENT_STATE.md`](../../../CURRENT_STATE.md) and
> [`closed-decisions.md`](../../closed-decisions.md). Body text below is preserved
> as written at seal time.

**Purpose:** Sealed record of Product Phase 3 **entrypoints** (document ingestion + intake translator). Forensic legibility — not a release announcement.

**Verify this checkpoint:**

```bash
python scripts/run_product_phase_3_checkpoint.py
```

Expected: `CHECKPOINT PASSED — Product Phase 3 entrypoints sealed (8 + 11 + 11 + 5 = 35 passed).`

**One-line status:** [`CURRENT_STATE.md`](../../../CURRENT_STATE.md)

**Phase names:** [`PHASE_NAMESPACE.md`](../../PHASE_NAMESPACE.md) — never bare “Phase 3.”

---

## What this checkpoint seals

| Slice | Scope | Tests | Status |
|---|---|---|---|
| **Document ingestion** | `IngestionReport` / `CandidateRecord` proposals with provenance | 8 | **Sealed** |
| **Intake translator** | Structured / labeled plain-English / candidate → `EstablishmentFailureProposal` | 11 | **Sealed** |
| **Corpus ingest helpers** | Segmentation, `ingest_file()` propose-then-persist, review selection | 11 | **Sealed** |
| **Checkpoint posture** | Manifest integrity, layer boundaries | 5 | **Sealed** |

**Combined test sign-off (2026-06-25):** **8 + 11 + 11 + 5 = 35 total**.

**Re-verified (2026-07-02):** `python scripts/run_product_phase_3_checkpoint.py` → `CHECKPOINT PASSED - Product Phase 3 entrypoints sealed (8 + 11 + 11 + 5 = 35 passed).`

---

## Architecture rules (locked)

| Rule | Meaning |
|---|---|
| **Ingestion proposes. Lens disposes.** | Documents become proposed candidates only |
| **Intake proposes.** | Translator emits `EstablishmentFailureProposal` with `status=proposed`, `establishes_truth=false`, `admissibility_decision=none` |
| **Explicit admission** | `with_admitted_for_evaluation()` before gate payload |
| **No PEF mutation** | Ingestion/intake modules do not write PEF |
| **No Lens import** | Ingestion/intake modules do not import Lens |
| **Plain English: labeled only** | `kind: description` with exact taxonomy token — no semantic inference |

---

## Primary code

| Module | Role |
|---|---|
| `aurora_lens/corpus/ingestion_report.py` | `IngestionReport`, `CandidateRecord`, `EvidenceProvenance` |
| `aurora_lens/corpus/document_ingestion.py` | `propose_document_ingestion()`, `ingest_file()` |
| `aurora_lens/corpus/establishment_failure_proposal.py` | `EstablishmentFailureProposal` |
| `aurora_lens/corpus/intake_translator.py` | Translation + `preview_gate_evaluation()` |

---

## Operator CLI

```bash
python tools/propose_document_ingestion.py --source path/to/doc.pdf
python tools/translate_intake.py --structured kind=bears_on=...
python tools/translate_intake.py --plain "stale_evidence: evidence older than policy window"
```

---

## Operator docs

| Doc | Topic |
|---|---|
| [`docs/PRODUCT_PHASE_3_DOCUMENT_INGESTION.md`](../../PRODUCT_PHASE_3_DOCUMENT_INGESTION.md) | Document → proposed candidates |
| [`docs/PRODUCT_PHASE_3_INTAKE_TRANSLATOR.md`](../../PRODUCT_PHASE_3_INTAKE_TRANSLATOR.md) | Intake → establishment-failure proposals |

---

## Relationship to admissibility checkpoint (342)

Product Phase 3 is **downstream packaging**, not part of the Track A/B/C admissibility arc.

- **Admissibility arc:** `python scripts/run_admissibility_checkpoint.py` → **342 passed** (unchanged)
- **Product Phase 3:** separate checkpoint — proposals only; analyser/gates decide after explicit admission

Do not merge Product Phase 3 counts into the 342 admissibility sign-off.

---

## Explicitly deferred (next boundary)

Not missing from this checkpoint — out of scope for sealed entrypoints:

1. **Corpus / RAG product UX** — registry + tools exist; packaging not sealed
2. **Automatic PEF or Lens wiring** — operator admission remains explicit
3. **Semantic plain-English inference** — labeled intake only in v1
4. **Roadmap/doc cleanup** — stale `source_untrusted` / bare “Phase 3” in older docs

---

## Combined manifest

[`eval/product_phase_3_checkpoint.manifest.yaml`](../../../eval/product_phase_3_checkpoint.manifest.yaml)

---

*Checkpoint package created 2026-06-25. Re-verify with `python scripts/run_product_phase_3_checkpoint.py` after any change to Product Phase 3 code paths.*

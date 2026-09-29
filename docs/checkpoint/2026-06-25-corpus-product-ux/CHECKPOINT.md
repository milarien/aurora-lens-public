# Corpus Product UX Checkpoint — 2026-06-26

> **Historical snapshot (2026-06-26):** Sealed checkpoint record. For current v1
> status use [`CURRENT_STATE.md`](../../../CURRENT_STATE.md) and
> [`closed-decisions.md`](../../closed-decisions.md). Body text below is preserved
> as written at seal time.

**Verify:**

```bash
python scripts/run_corpus_product_ux_checkpoint.py
```

Expected: **6 + 2 + 4 = 12 passed**.

**Re-verified (2026-07-02):** `python scripts/run_corpus_product_ux_checkpoint.py` → `CHECKPOINT PASSED - Corpus product UX sealed (6 + 2 + 4 = 12 passed).`

---

## Sealed

| Slice | Scope | Tests |
|-------|-------|-------|
| Corpus CLI | `aurora-lens corpus` subcommands + `cli_support` | 6 |
| Product e2e | Ingest → mocked ask with RAG message shape | 2 |
| Checkpoint posture | Manifest + doc integrity | 4 |

**Rule:** Retrieval proposes. Lens disposes.

---

## Operator entry

```bash
aurora-lens corpus ingest --record-id ID --file PATH
aurora-lens corpus list
aurora-lens corpus ask --record-id ID --question "…"
```

See [CORPUS_PRODUCT_UX.md](../../CORPUS_PRODUCT_UX.md).

---

## Deferred

- `source_scope` enforcement vs retrieval
- Audit `request_metadata` embedding
- Corpus-aware `aurora-lens chat --record-id`

---

*Manifest:* [eval/corpus_product_ux_checkpoint.manifest.yaml](../../../eval/corpus_product_ux_checkpoint.manifest.yaml)

# Corpus product UX

**Rule:** Retrieval proposes. Lens disposes.

Unified operator surface for the document library via `aurora-lens corpus` (Phase 2a). Substrate modules remain in `aurora_lens/corpus/`; Product Phase 3 proposal entrypoints are sealed separately.

---

## Two doors

| Door | Command | Lens? | Proxy? |
|------|---------|-------|--------|
| **1 — Review** | `aurora-lens corpus review` | No | No (direct upstream) |
| **2 — Ask** | `aurora-lens corpus ask` | Yes | Yes |

---

## Commands

```bash
# Ingest (once per document)
aurora-lens corpus ingest --record-id my-doc --file data/doc.pdf

# List records
aurora-lens corpus list

# Governed Q&A (proxy must be running)
aurora-lens proxy &
aurora-lens corpus ask --record-id my-doc --question "What are the main sections?"

# Validation pack
aurora-lens corpus validate --manifest eval/sample_corpus_qa.manifest.yaml

# Direct upstream review
aurora-lens corpus review --record-id my-doc
```

Legacy tools under `tools/` remain for advanced use; prefer `aurora-lens corpus` for product flows.

---

## Architecture

```
ingest → registry (chunks.jsonl)
ask    → retrieve_chunks → assemble_rag_message → proxy → Lens pipeline
```

- No automatic PEF truth establishment from retrieval
- `request_metadata.record_ids` carries corpus scope into Lens
- Product Phase 3 ingestion/intake: proposals only — [checkpoint/2026-06-25-product-phase-3/](checkpoint/2026-06-25-product-phase-3/)

---

## Deferred seams

See [closed-decisions.md](closed-decisions.md): `source_scope` enforcement, audit `request_metadata` embedding, operator proposal selection (outside v1 runtime).

---

## Verify

```bash
python -m pytest tests/test_corpus_cli.py -q
python scripts/run_corpus_product_ux_checkpoint.py   # Phase 2b seal
```

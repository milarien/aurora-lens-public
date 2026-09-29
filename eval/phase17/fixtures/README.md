# Phase 1.7 realistic acceptance fixtures

These fixtures are for corpus acceptance and evaluation only. They are not product policy files.

## Purpose

- Exercise ingestion, retrieval, supersession handling, citation/provenance checks, and reporting against realistic document shapes.
- Validate malformed and noisy source handling without changing governance semantics.

## Fixture set

- `real_nist_privacy_framework_excerpt.md` — real government guidance excerpt (public domain).
- `real_gpo_style_manual_excerpt.md` — real organisational operating-standard excerpt (public domain).
- `real_uscode_5usc552_excerpt.md` — real legal/regulatory excerpt with nested clauses and definitions (public domain).
- `policy_v1.md` — older policy version (superseded).
- `policy_v2_superseding.md` — governing replacement version.
- `regulatory_guidance.md` — obligations, exceptions, defined terms, cross-references.
- `public_service_standard.md` — procedural standard with responsibilities and escalation.
- `mixed_table_and_prose.md` — decision-relevant table plus qualifying prose.
- `malformed_truncated_markdown.md` — intentionally truncated structure.
- `malformed_encoding_noise.txt` — encoding-noise but partially recoverable text.
- `malformed_unsupported_binary.bin` — unsupported suffix fixture expected to fail cleanly.

See `source_inventory.yaml` for provenance, licensing, and transformation status per fixture.

# Changelog

All notable changes to this public repository are documented here.

## [Unreleased]

- **2026-10-11 - Scoped governance repairs:** Bind medical treatment actions to clinical objects using the existing parser, distinguish tax and chemical object senses, and retain medical governance in mixed-topic responses. Keep primary personalized-finance release copy on its own Governor corridor while preserving secondary findings and safety vetoes. Govern employment selection explicitly conditioned on birth dates without inferring age from proxies or fixing any cutoff year. Hold claimless input when the English parser has not established a lexical subject or imperative matrix. Preserve attributed adviser descriptions as user-input provenance and unresolved approval admission holds. Regression coverage: `test_medical_directive_scope.py`, `test_birth_screening_scope.py`, `test_release_domain_consistency.py`, and `test_interpretation_admission.py`.

## [3.0.1] — 2026-09-29

### Added

- Public architecture, Zenodo publication corpus, and provenance records.
- Aurora-Lens runtime source published for inspection (licensed use only), operator docs, demo entrypoint (`tools/run_demo.py`), and automated tests.
- [PATENT_NOTICE.md](PATENT_NOTICE.md), [RELEASE_MANIFEST.md](RELEASE_MANIFEST.md), and checkpoint / closed-decision documentation under `docs/`.
- Closed-decisions and product checkpoint docs referenced by eval manifests.

### Changed

- README combines architecture index with runtime setup instructions (licensed use only).
- Patent specifications are referenced via public deposits (Zenodo, TDCommons, IP Australia OPI) rather than bundled filing PDFs in this tree.

### Removed from public tree

- Commercial acquisition licence document (buyer-facing material not published here).
- Local copies of filed patent specification PDFs (use public deposits in [PATENT_NOTICE.md](PATENT_NOTICE.md)).

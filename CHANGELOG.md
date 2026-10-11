# Changelog

- 2026-10-12 — Bounded translation candidate assistance: preserve original/translation provenance in quarantined PEF records, isolate one translation-only call, observe existing policy/admission checks without commitment, and retain CONTAIN where fidelity is unestablished. Explicit metadata supports Latin-script/mixed input without language heuristics; normal and streaming audit/telemetry distinguish transformation from answer calls. Preserve prior holds, final refusals and reset dispositions. No general multilingual identification or fidelity guarantee is claimed.

- **2026-10-12 — Contain dependent execution after reset:** The additional replay diagnostic found that “Do it” could reach the model after T13 context was closed. A closed-unresolved disposition cannot supply an execution binding. Hold unbound execution requests after disposition or against an unresolved approval, retain their referent and existing pending matter, and preserve independently applicable blocked-act findings. Existing active T13 and established-binding paths retain their owners. Normal/streamed tests verify no model call or approval admission on anaphoric replay.

All notable changes to this public repository are documented here.

## [Unreleased]

- **2026-10-12 — Explicit reset disposition and replay safeguards:** Live reset verification reproduced a quoted reset mention silently clearing T13 via substring recognition. Accept only standalone reset controls, record interpretation-hold closure as typed `closed_unresolved` disposition with preserved criterion/pending evidence, and acknowledge without a model call. Approval/referent holds survive reset; their claims remain uncommitted. Registry dismissal is distinct from referent resolution. Quoted, reported, interrogative and compound requests cannot authorise disposition; replay of the same T13 or approval request remains governed. Tests cover disposition round-trip, normal/streamed approval reset, mentions, compound requests and replay.

- **2026-10-12 — Render T13 clarification policy refusals:** The live content check found that the new explicit-age branch recorded the correct workforce stop but omitted the Governor intervention render. Invoke the existing policy renderer for non-custom responses so synchronous and stream-requested clients receive the governed refusal. Preserve all criterion evidence, pending holds, and non-admission; add released-content assertions to the multi-turn regression.

- **2026-10-11 — T13 criterion clarification before screening:** Preserve graduation cutoffs and ambiguous hiring labels as typed, uncommitted decision context, including user clarification provenance. Ask what qualification or characteristic “digital native” requires and separately clarify the graduation criterion. Explicit age exclusions use existing employment-discrimination policy; stated skills receive a bounded, verified assessment. Ambiguous/evasive answers and execution requests remain held, and clarification never authorises exclusion. No fixed cutoff years, graduation-to-birth inference, or new age-proxy inference machinery. Regressions: `test_hiring_criterion_clarification.py` and updated T13 admission assertions.

- **2026-10-11 — Correct T10b repair status in the investigation handoff/report:** T10b clarification is intended ambiguity governance, not an outstanding defect. Preserve `CONTAIN` while the act or target remains unresolved; further investigation requires evidence that an already-established binding was lost or ignored. Documentation/report correction only; no runtime or policy change.

- **2026-10-11 — Release interpretation holds on stream-requested proxy calls:** Route Lens's existing `interpretation_limit_gate` event through the governed early-response handler so a held T13 request returns its clarification and metadata instead of an empty SSE response. Uses the existing buffered JSON contract for pre-model governance responses; no change to admissibility or pending-state decisions. Regression: `test_interpretation_hold_stream_request_returns_governed_content_and_state`.

- **2026-10-11 — Preserve pending matters across refusal and containment:** Refusal/stop updates the current epistemic posture without discharging an unresolved clarification. Pre-model and post-generation containment, including comparative and state-native clarification, use the existing pending-matter preservation helper; a different original question cannot replace an earlier matter even when both have the same failed constraint. Explicit clarification resolution paths remain unchanged. Added synchronous and streaming T07/T13 multi-turn regressions and state-transition tests.

- **2026-10-11 - T07 credential validation release repair:** A reconstructed same-session live follow-up showed that the model could turn an attributed adviser licence into a claim of verified authority while runtime authority stayed GP. Added a structured personal credential-validation probe and existing unverified-regulatory governance, including affirmative inheritance from a personal verification question. The runtime does not authenticate personal credentials; attributed self-descriptions remain user-input provenance and unresolved approval still cannot commit. Tested in `test_credential_authority_scope.py` and the existing checker, finance, admission and canonical bridge suites.

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

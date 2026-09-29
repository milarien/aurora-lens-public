# Closed decisions

Items below were previously listed as deferred or open. Each is now explicitly
closed: either the implementation is done and in the codebase, or it is
permanently out of v1 scope with a stated rationale. There are no open items.

---

## FU-QUERY-REL — QUERY relationship writes

**Decision: closed — read-only QUERY is the v1 contract.**

Lens enforces read-only narrative QUERY for relationship admission. Event-shaped
QUERY writes are permanently outside the runtime path. This is not a deferral:
QUERY semantics are non-mutating by definition in v1. Changing that would require
a separate governed non-answer mutation lane with its own admissibility rules,
audit payload schema, and rollback semantics — none of which belong in v1.

---

## FU-QUERY-REL-MUTATION — governed mutation from QUERY outcomes

**Decision: closed — not in v1; read-only QUERY baseline is permanent.**

The open question was: should there be a governed mutation lane that writes
relationship/binding state from QUERY outcomes? The answer is no for v1. The four
blocking prerequisites (admissibility rule, mutation shape, audit payload schema,
rollback/expiry semantics) were never defined, and defining them would constitute
a separate product feature, not a gap in the current one. If a future release
opens this lane, it must satisfy all four prerequisites independently; the
existing QUERY implementation requires no changes to support that.

---

## LLMExtractionBackend — `ambiguous_referents`

**Decision: closed — spaCy backend is the parity baseline; LLM backend exclusion is permanent for v1.**

`LLMExtractionBackend.supports_ambiguous_referents()` returns `False`. Ambiguity-
hold parity claims are formally excluded on this backend path. The spaCy backend
is the governance-grade extraction baseline for deterministic ambiguity detection.
The LLM extraction backend remains available for higher-fidelity entity extraction,
but it is excluded from the deterministic v1 parity contract because LLM-based
ambiguity extraction would be probabilistic, provider-dependent, model-version-
dependent, and difficult to reproduce in regression. spaCy is the controlled parity
baseline; this exclusion is correct by design, not an oversight.

---

## Auditor corridor enforcement

**Decision: closed — already enforced; no further implementation needed.**

Auditor forensic-visibility widening (EXPOSE_AUDIT_BASIS, ATTACH_PEF_SNAPSHOT,
FORENSIC_STOP) is allowed only on explicit `:auditor` STOP corridors. All other
paths are hard-capped to general visibility. Enforcement point:
`aurora_lens/governor/resolver.py` (PR6 — explicit `:auditor` corridor key on STOP
rows only). Tests: `tests/test_pr6_deferred_runtime_boundaries.py`,
`governor/tests/test_governor.py::test_auditor_corridor_hardening`.

---

## State-native Phase 4 placeholders

All four sub-items are closed:

- **`PresentBoundTemporalEvalResult`** — wired into Lens mapping (commit `bbd79da`).
  Done.

- **`TemporalGovernanceCue`** — **Decision: retain; stable contract is correct.**
  `TemporalGovernanceCue.ASK_TEMPORAL_SCOPE` and `PRESENT_COMMITTED_ANSWER` are
  used in `temporal_contact.py` and consumed in `lens.py`. The "open compatibility
  decision" was whether to remove it; the answer is no. It carries governance
  semantics cleanly and removing it would break the temporal evaluation path.
  Retained as-is.

- **Multi-word comparative parsing** — **Decision: v1 behavior is correct.**
  Single-word subject comparatives are supported; multi-word subjects are not
  parsed into COMPARE relationships. Negative tests document the v1 boundary.
  Extending the parser would risk false positive comparative extraction on
  compound noun phrases. Not in v1.

- **`temporal_contact.py` partial coverage** — **Decision: v1 coverage is
  complete; UNKNOWN fallback is correct behavior.**
  Temporal queries that cannot be classified return `UNKNOWN` and do not silently
  fabricate a governance outcome. This is the correct fail-closed posture for
  unrecognized temporal shapes. No silent failure; no unhandled code path.

---

## Live provider HTTP in CI

**Decision: closed — excluded from required CI; operator opt-in is correct by design.**

Live-provider HTTP tests are excluded from required CI because they introduce
credential exposure, network dependence, cost, rate limits, provider drift, and
non-reproducible outcomes. Required CI must use deterministic mocks, fakes,
contract tests, or recorded fixtures. Real provider probes are available as
operator opt-in via `pytest -m live_provider_regression` with a configured endpoint;
see `docs/LIVE_PROVIDER_REGRESSION.md`. Any live-provider smoke test must be
explicitly invoked outside the required CI gate.

---

## Corpus / RAG future seams

**Decision: closed — operator proposal selection is permanently outside v1 runtime.**

v1 corpus questioning supports: read-only inspection over ingested evidence with
explicit admissibility and conflict signaling. It does not support: corpus
mutation by questioning, metadata repair/inference by questioning, or
relationship/binding mutation via QUERY (`FU-QUERY-REL-MUTATION`, closed above).

`source_scope` enforcement at retrieval and governed audit request metadata are
both implemented in the v1 runtime. Operator-mediated proposal selection between
ingestion and Lens admission is a v2 feature that would require a separate
admission lane with its own governance semantics. It is not a gap in v1.

---

## Governed clarification natural-language gap

**Decision: closed — pre-LLM structural detection is the v1 mechanism; NL
post-LLM detection is not in v1 scope.**

The gap: if the LLM asks a natural-language clarification question (e.g. "Did you
mean X or Y?") and the extraction layer did not flag structural ambiguity, the
user's clarification answer arrives as a fresh turn with no `pending_clarification`
context in PEF.

The v1 structural path (extraction → `ambiguous_referents` → CONTAIN →
`pending_clarification`) handles all cases where ambiguity is detectable pre-LLM.
The NL gap is the residual: LLM-generated clarification after extraction passed.

Post-LLM clarification detection would require scanning LLM output for question
patterns, which (a) is heuristic and prone to false positives on rhetorical or
informational questions, (b) requires setting `pending_clarification` after the
governance decision (not before), and (c) changes the turn semantics in a way that
is not covered by the current typed-transaction envelope. The risk of
mis-governing a legitimate clarification answer is lower than the risk of
incorrectly holding a non-ambiguous response in a pending state.

v1 position: if the LLM generates an unexpected clarification question, the user's
answer is governed on its own merits. This is conservative and safe. The
pre-LLM structural ambiguity path is the right place to catch ambiguity; the LLM
should not be the authority on whether its own output was ambiguous.

---

## FU-CLAR-AUTH-RESUME — `CLARIFICATION_AUTHORITY_OUTSIDE_HOLD` continuation

**Decision: closed — v1 uses governed candidate-binding resume through the shared
`pending_clarification` corridor.**

When the post-LLM checker raises `CLARIFICATION_AUTHORITY_OUTSIDE_HOLD` (duplicate
given-name collisions in committed PEF + third-person pronoun in the user query +
clarification-shaped assistant prose, with no pre-existing `pending_clarification`),
Lens CONTAINs, records `pending_clarification` via `_pending_payload_simple_contain`,
and sets `epistemic_hold`. Enforcement: `aurora_lens/verify/checker.py`
(`_check_clarification_authority_outside_hold`); payload wiring in `lens.py`
(`_pending_payload_simple_contain`).

**v1 continuation contract (deterministic):**

1. **Direct candidate selection** — user's reply naming a stored candidate entity
   resumes the original question through the shared binding corridor in `lens.py`
   (`_matches_candidate`, pronoun reconstruction, `_commit_resolved_claims`).
   Pending clears on successful bind.
2. **Non-selection clarification turns** — re-issue CONTAIN with stored candidates
   (`_build_clarification_continuation` generic attribution-unresolved copy).
3. **Unrelated follow-up** — pending survives; the turn is governed on its own
   merits (fall-through).

This is not terminal containment. Full UNRESOLVED_REFERENT payload/message parity
is not required in v1; the shared corridor is the intentional contract.

Tests: `tests/test_query_clarification_provenance_iron_bar.py` (containment +
payload); `test_clarification_authority_outside_hold_candidate_binding_resumes`
(binding resume).

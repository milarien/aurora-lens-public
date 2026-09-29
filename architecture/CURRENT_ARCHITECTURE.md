# Aurora-Lens: Current Architecture

Current architecture description based on the implementation as of 27 September 2026. Historical documents may describe earlier versions.

The evidence-admission behaviour described in sections 2 and 4 reflects the current implementation working tree, including changes not yet committed to the implementation repository. The last committed change to the implementation package is dated 15 September 2026.

---

## 1. What Aurora-Lens is and what boundary it governs

Aurora-Lens is a runtime admissibility architecture for consequence. It governs whether a proposed output, action, interpretation, determination, release, or state transition may become consequential. The decision is made against persistent governed state, current evidence, authority, standing, unresolved conditions, prior commitments, and applicable policy.

The boundary it governs is not "harmful content in, harmful content out." It is the boundary between proposal and consequence: between something being generated, retrieved, inferred, requested, or proposed, and that thing being permitted to bind the world as an adjudication, determination, factual commitment, state mutation, release, or actionable instruction. Aurora-Lens's governing priority is admissibility before consequence.

Three roles are kept architecturally distinct:

- **Lens:** the turn orchestrator and admissibility gate (`lens.py`). It runs the pipeline, gathers evidence, and produces a governance decision for each turn.
- **Governor:** the lawful-continuation authority (`governor/` package: status translation, context resolution, policy resolution, policy projection). Once Lens has determined a status (admit / ask / refuse / stop), the Governor selects the *pathway* forward, what may still be said, what is closed, what disclosures or escalations apply. Its core invariant, asserted in code: the Governor may constrain, explain, or route after Lens, but may **never widen epistemic commitment**.
- **PEF (Persistent Existence Framework):** the durable world-state substrate (`pef/` package) that both of the above read and write. It is not chat memory; it is a structured present-state model of what has been established, by whom, on what basis, and what remains open.

A note on naming: "Aurora" appears in package and class names as product branding; there is no runtime component named Aurora that participates in admissibility decisions.

## 2. Ingestion: how governed state is constructed from sources

Source material enters through the corpus subsystem (`corpus/`), which is deliberately separated from the decision engine. Its contract is stated in the ingestion code: **ingestion proposes; Lens disposes.** Ingested material never establishes truth by arriving.

- Documents (PDF, Markdown, text, CSV, JSON, HTML, DOCX) are read, SHA-256-digested, and chunked. Each record carries provenance: source path, digest, extraction metadata, and optional operator-supplied standing metadata, status, authority class, applicability scope, effective-from/to dates, version, and supersession links (`supersedes` / `superseded_by`).
- An ingestion report is produced with `establishes_truth=False` and no admissibility decision. Persisting a record makes it *retrievable*, not *believed*.
- At retrieval time, evidence passes through admissibility evaluation before it can inform a governed answer: status allowlists, freshness windows, authority-class checks, and supersession checks. A record whose `superseded_by` is populated is inadmissible as current authority even though it remains fully present and retrievable in the registry.
- Retrieved context that survives those gates is first decomposed into role-labelled extraction units before any PEF admission occurs. `FILE:`-labelled context is split by record; JSON records are parsed structurally and admitted field by field, with malformed JSON contributing no partial claims. Only fields classified as `observed_fact` or `epistemic_limit` are eligible for admission as assertions. Policy rules, workflow instructions, metadata, and propositions under evaluation are barred from assertive admission; `claim_under_evaluation` is skipped explicitly as non-assertive. Admitted fields carry field-level provenance locators of the form `FILE:<record>#<field_path>`. Claims that merely restate the proposition under evaluation are filtered before temporal and literal-conflict checks, so cross-document bleed is represented as inadmissible rather than silently converted into evidence.

## 3. The PEF: persistent governed state across turns

The PEF (`pef/state.py`) is a persistent governed world-state, keyed by a `pef_context_id` (session id) and persisted across turns in a memory or Redis session store. It contains, among other things:

- **Entities** with resolution status (`resolved=False` marks a mentioned-but-ungrounded placeholder) and relationships: subject/relation/object rows carrying provenance (`user_input`, `pre_populated`, `llm_output`, `retrieved_context`, `system`), evidence, negation, source turn, and a claim-status field.
- Unresolved structures that are first-class state, not error conditions: a `pending_clarification` payload (the open question, candidate resolutions, blocked claims), an `epistemic_hold` with modes *ambiguity / refusal / stop*, and a durable `unresolved_referent_registry` that outlives individual clarification exchanges.
- **An artifact frame:** the session can enter a bounded ARTIFACT frame (fiction, roleplay, hypothetical) distinct from the default EXTERNAL frame. Entities created inside an artifact frame do not become external commitments, and the state-native answer path declines to answer external-world questions from inside an open artifact frame. Consequence governance is *not* relaxed by the frame, a medical instruction inside fiction is still flagged as a medical instruction.
- **Continuation corridors:** after a refusal or stop, the state records which narrow follow-up capabilities remain lawful (e.g. a neutral factual timeline after a refused financial adjudication).

State updates per turn go through extraction (deterministic spaCy backend or an LLM extraction backend) into an admission layer (`update_pef`), where claims are wrapped in semantic transactions that may be committed, held (with a typed held-reason such as unresolved possessor or unresolved comparand), or rejected. A revision gate blocks assertions that contradict grounded PEF unless the turn is classified as an explicit revision, the state cannot be silently overwritten by restating a contradiction.

Every governance decision snapshots the PEF; a SHA-256 `state_hash` of that snapshot is written to the audit record, and offline verifiers can replay it.

## 4. How a model or application proposes a consequence

The runtime surface is an OpenAI-compatible proxy (`proxy/app.py`, `POST /v1/chat/completions`). An application speaks the ordinary chat-completions wire format; Aurora-Lens loads the PEF for the session, runs the turn, and returns an OpenAI-shaped response whose governed verdict lives in an added `aurora` block (outcome, turn, audit id, operator-plane detail when enabled), governance is expressed in the payload, not as HTTP errors.

Two kinds of proposal exist:

1. **Free-text turns:** the ordinary case. The user message and (if reached) the upstream model's draft are both objects of governance.
2. **Structured execution tasks:** a host can submit a `candidate_release` task: a candidate text plus an explicit evidence state and governing policy. This is adjudicated deterministically before any model call. An admit decision releases the candidate verbatim; anything else is a structured hard stop. No prose intent inference is involved.

The model is non-authoritative. Governance operates both before and after any model call. Pre-LLM gates (blocked-act classification, extraction failure, unresolved referents, revision conflicts, epistemic holds, sovereign provider-route evaluation) can conclude the turn before any upstream call is made. Post-LLM, the model's full draft is verified and governed before anything reaches the caller. In streaming mode the draft is accumulated in a private buffer; **no token is released to the client until governance over the complete text has finished.** An admitted buffer is re-emitted as chunks; a non-admitted buffer is suppressed and replaced by governed text in a single event.

- The blocked-act classifier remains a pre-model veto, but its financial-crime-evasion and audit-trail-evasion surfaces now require a localized action objective: user intent, directive action, financial context, and concealment target must co-occur within a bounded token window. Illicit financial terms alone no longer trigger that veto unless they participate in that localized objective. This narrows the request-side block to reduce false positives from broad cross-document token collisions.

## 5. How admissibility is evaluated

Verification (`verify/checker.py`) extracts claims from the proposed text and evaluates them against the PEF and the user's grounding context, producing typed **flags** along three axes:

- **Epistemic failure:** unsupported or unverified assertions, predictive claims not established, insufficient upstream context, clarification authority exercised outside a hold.
- **State binding failure:** unresolved referents and comparands, contradiction of committed state, identity drift, temporal smear, disallowed state transitions.
- **Content-class veto:** consequence classes that are refused or stopped regardless of grounding (dosage instructions, self-harm instruction, illegal instruction, personalized medical/legal/financial adjudication, targeted defamation, sensitive PII, and others).

The decision then flows through the canonical Governor stack: flags plus deployment mode translate to a status (ADMIT / ASK / REFUSE / STOP); a context resolver derives the domain (medical, legal, finance, …), the **authority class** (general-purpose vs domain-authorized vs human-supervised deployment), and user class; a policy resolver looks up the governing policy row; and a policy projector emits the runtime decision, an intervention action plus a *continuation pathway*.

The inputs are therefore exactly the ones the question implies: current **evidence** (PEF relationships and their provenance; retrieved-evidence admissibility), **authority** (authority class of the deployment; authority metadata and class of evidence sources), **standing** (status/freshness/supersession of corpus evidence; negation-superseded facts in PEF), **unresolved conditions** (open registry entries, pending clarification, epistemic holds, ambiguity branches), and **prior state** (committed relationships, prior refusals and stops, continuation corridors, turn history).

## 6. Outcomes: PASS, FORCE_REVISE, CONTAIN, HARD_STOP, and maintained UNRESOLVED

The intervention actions defined in code (`govern/decision.py`):

| Action | Meaning in the implementation |
|---|---|
| PASS | Admissible; the governed response is delivered. |
| SOFT_CORRECT | Delivered with the correction recorded in metadata. This is a deployment-mode concession and is not available for safety-veto flags. |
| FORCE_REVISE | Refusal-class outcome. The model is not re-prompted; the Governor renders the lawful refusal pathway. |
| CONTAIN | No determination is made. The system asks or acknowledges while recording pending clarification and an ambiguity-mode epistemic hold. |
| HARD_STOP | The proposed consequence is blocked. Stop pathways render the governed response, and a stop-mode hold may persist. |

**UNRESOLVED is a maintained state, not a transient error.** Ambiguity is held open in three durable structures (pending clarification, epistemic hold, unresolved-referent registry) until it is *lawfully* discharged: bound by an explicit user resolution, deliberately held-unresolved by choice, or dismissed through a session gate. Blocking is one possible governance consequence of unresolvedness; it is not its definition. A turn can proceed in parallel with open, represented unresolvedness where policy allows.

## 7. Historically valid, currently non-consequential

Aurora-Lens separates *record* from *standing* in both of its evidence planes, by construction rather than deletion:

- **Corpus plane:** a superseded, expired, revoked, or suspended document remains fully present, digested, and retrievable. Admissibility evaluation can still demote it (`superseded_evidence`, revalidation-required, inadmissible), and an authority-state ladder distinguishes *authoritative* evidence from *signal-only* text that may inform but cannot carry consequence.
- **PEF plane:** facts are append-oriented. Losing standing is implemented as a later negation or supersession row, and "current" is a read-time projection (newest positive assertion not subsequently negated). The prior fact remains in the state and in every audit snapshot that contained it. A stale reading used as if current is itself flaggable (`CONTRADICTS_COMMITTED_STATE`).

So evidence can be simultaneously *historically valid* (it happened, it is recorded, its chain of custody is intact) and *without current consequence-bearing standing* (it can no longer license a PASS).

## 8. Why retry, rerouting and reformulation cannot manufacture commitment

Several mechanisms combine so that repetition does not become permission:

- **Stateful holds.** Refusals and stops set epistemic holds that persist in the PEF across turns. A PASS on a later innocuous turn does not clear a refusal or stop hold.
- **Commitment closure.** Non-admit policies carry `commitment_closed=True`, with a code-level invariant that substantive speech acts cannot be allowed while commitment is closed; continuation corridors expose only enumerated, bounded follow-ups.
- **Every turn re-runs the request-side gates.** Blocked-act classification operates on the user's text on every turn, before the model. A reformulated version of a blocked request meets the same classifier, not a fresh slate.
- **The revision gate.** Restating a contradiction of grounded state without an explicit revision act is contained, not committed; the state cannot be worn down by repetition.
- **Registry-backed session gates.** While unresolved referents are open in the durable registry, consequence-bearing turns are gated; the registry does not evaporate when the user changes subject.
- **No self-revision loop.** FORCE_REVISE renders a refusal rather than re-prompting the model, so there is no internal retry channel in which the model can iterate toward an admissible-looking restatement of an inadmissible commitment.
- **Escalation on probing.** A follow-up that resolves a pending agency-risk ambiguity in the violating direction escalates to a hard stop rather than re-asking.

## 9. Provenance, evidence, signatures, and the audit ledger

The forensic layer is written after the decision as its evidentiary record. Verification failure is an integrity finding about the log, never an input that flips a governance outcome.

Per governance decision, the bridge writes an audit row to one of two backends (flat JSONL with a `prev_cid → cid` hash chain and optional whole-row HMAC; or the default AFL ledger with `prev`/`hash` chaining, sequence numbers, and per-line HMAC signature when a signing key is configured). Rows carry:

- the outcome, flags, policy profile/version, ruleset hash, governor policy id;
- the **PEF snapshot** and its `state_hash`, plus turn-linkage classification so consecutive rows can be replay-checked against each other;
- a **chain-of-custody bundle:** code provenance (application version, git commit and its source), policy provenance (a SHA-256 fingerprint of the governance configuration, with the fingerprinted snapshot embedded for offline replay), and runtime provenance (host, service instance, upstream provider and model attributed per-request). The bundle self-reports `complete` or `degraded` with explicit reasons. Missing attribution degrades the evidence status visibly rather than silently;
- **evidence-capture fields:** request content is stored in a local evidence vault as either encrypted sealed blobs or hash-only records (raw and canonical SHA-256), referenced from the row by `vault://` refs and manifests. Production modes never place raw prompts on the ledger; a sealed deployment without a key downgrades to hash-only rather than failing or leaking;
- for non-admitted outcomes, a nested **forensic event** with its own `event_hash` over the canonical event content.

Offline and operator verification surfaces (a verifier CLI, `GET /v1/audit/verify`, and the read-only `/forensics` console) check the hash chain, HMACs, chain-of-custody replay, state-hash replay, and PEF linkage.

## 10. Provider-agnostic vs implementation-specific

**Provider-agnostic (the architecture):** the admissibility pipeline, PEF, flag taxonomy, Governor policy stack, audit/chain-of-custody model, streaming buffer-then-govern discipline, and the adapter contract (`generate` / `generate_stream` over plain messages). Adapters are transport-only; no governance logic lives inside them, and no PEF state crosses into them. Any OpenAI-compatible endpoint (hosted or local) or Anthropic's API can be the upstream; a mock adapter supports fully offline operation.

**Implementation-specific (this codebase):** FastAPI proxy surface; spaCy as the deterministic extraction baseline (the LLM extraction backend does not produce ambiguity parity; spaCy is the reference for ambiguity holds); memory/Redis session stores; local file-based corpus registry and evidence vault; JSONL/AFL ledger formats; the sovereign provider-route registry for infrastructure admissibility and failover.

## 11. Complete high-level flow

```text
SOURCES                                      RUNTIME (per turn)

Documents
   |
   v
Ingest: digest + chunk + provenance
"proposes, never establishes truth"
   |
   v
Corpus registry
(status / authority / freshness / supersession)
   |
   v
Retrieval -> evidence admissibility gate
   |
   v
RAG -> PEF admission gate
(conflicts remain represented as unresolved)
   |
   v
PEF: persistent governed world-state
   ^
   |                                      Client request
   |                                           |
   |                                           v
   |                                  Load PEF for session
   |                                           |
   |                                           v
   |                                  Pre-model governance gates
   |                                  - blocked act
   |                                  - extraction failure
   |                                  - unresolved referent
   |                                  - revision conflict
   |                                  - epistemic hold
   |                                  - structured task adjudication
   |                                           |
   |                              +------------+------------+
   |                              |                         |
   |                         Turn may end              Model call
   |                         without model             draft only
   |                              |                         |
   |                              +------------+------------+
   |                                           |
   |                                           v
   |                                  Verify proposal against
   |                                  PEF, grounding, and vetoes
   |                                           |
   |                                           v
   |                                  Governor:
   |                                  status -> context -> policy
   |                                  -> pathway -> action
   |                                           |
   |                                           v
   |                         PASS / SOFT_CORRECT / FORCE_REVISE /
   |                         CONTAIN / HARD_STOP
   |                         with UNRESOLVED maintained where applicable
   |                                           |
   +----------------------------- update admitted state,
                                 holds, and continuation corridors
                                             |
                                             v
                                 Persist PEF and append audit:
                                 outcome, flags, state snapshot,
                                 chain-of-custody, evidence manifests,
                                 signatures, hash chain
                                             |
                                             v
                                 Governed response to caller
                                 plus offline/operator verification
```

## 12. Aurora-Lens is not a guardrail system

A typical guardrail or output filter asks whether a message or action violates a rule and then allows, blocks, or rewrites it. Aurora-Lens asks a different question:

> is this proposal admissible as a commitment, given what has been established, on whose authority, and what remains unresolved?

The differences are architectural, not rhetorical:

- **It maintains a world, not a blocklist.** Decisions are made against a persistent, structured state (PEF) with provenance, standing, and open questions rather than against a per-message classification of the text in isolation.
- **Ambiguity is a represented state, not a rejection.** Aurora-Lens can hold a referent, claim, or branch open across turns and route the interaction through clarification without either committing or refusing.
- **Blocking is one consequence among several.** PASS, contained clarification, refusal-with-lawful-continuation, and stop are distinct governed pathways with distinct permitted follow-ups; the Governor's role is lawful continuation after non-admission, which a filter does not have a concept of.
- **Evidence has standing, not just existence.** Material can be present, retrievable, and historically intact while carrying no current authority to license consequence. Filters have no analogue of supersession, freshness, or authority class.
- **The model is not authoritative.** Verification runs against the governed state; the model's fluency, framing (including fictional framing), or persistence cannot upgrade a claim's admissibility, and there is no internal retry loop through which a rejected commitment can be regenerated into acceptance.
- **Every decision is a forensic artifact.** Outcomes are written to a tamper-evident, signed, replay-verifiable ledger with chain-of-custody over the code, policy, and runtime that produced them. The decision is designed to be *defensible after the fact*, not merely enforced in the moment.

A guardrail controls what may pass a policy boundary. Aurora-Lens governs what may become consequence and preserves the evidence needed to prove why.

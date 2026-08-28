# The Maintained Unresolved State: Priority Record and Documented Chronology

Margaret Stokes  
Independent researcher, Melbourne, Australia  
ORCID: 0009-0004-6422-4174

Deposited: 26 August 2026

---

## Purpose and scope

This record concerns one architectural invariant: unresolved material remains a
governed state until it is explicitly resolved or rejected. It does not expire by
timeout, become accepted through repetition or resubmission, cease to matter when
processing moves downstream, or merge silently with a conflicting interpretation.

Margaret Stokes disclosed the underlying architecture in Australian provisional
patent applications filed in November and December 2025. A working implementation
was published in a GitHub repository on 30 November 2025.

On 19 August 2026, Njål Gaute Solland invited Stokes to review the PEACE Protocol.
Stokes identified four omissions concerning unresolved material. Solland accepted
the substance of the critique and committed changes four minutes later. The changes
corresponding to all four omissions were added to the repository in that commit.

The record establishes the sequence through dated filings, repository history and
correspondence. It distinguishes direct evidence from inference. It makes no claim
about Solland's intention or state of mind and no allegation of infringement.

Except where a timestamp is expressly marked AEDT, repository times are stated in
Central European Summer Time (CEST, UTC+2). LinkedIn displayed the messages in
Australian Eastern Standard Time (AEST, UTC+10). The converted message times have
minute precision because LinkedIn did not display seconds.

---

## 1. The invariant

Where more than one interpretation remains structurally valid, the valid
interpretations are preserved. Collapse to one interpretation is permitted only
when structural requirements force resolution.

This produces four operational requirements:

1. An unresolved state persists until an explicit decision resolves or rejects it.
2. Repetition or resubmission cannot convert unresolved material into accepted
   material.
3. A downstream authorization that materially depends on unresolved content must
   fail closed, defer, or require explicit resolution.
4. Conflicting unresolved interpretations cannot be merged without an explicit
   resolution process.

---

## 2. Earlier records

| Date | Record | Relevant disclosure |
|---|---|---|
| 27 Nov 2025 | AU 2025905835 (P1) | Foundational Aurora architecture and constraint-based reasoning. It does not state the maintained-unresolved rule. |
| 28 Nov 2025 | AU 2025905860 (P2) | Persistent substrate supporting discourse-level continuity. |
| 29 Nov 2025 | AU 2025905885 (P3) | First express disclosure and claim. Claim 13 specifies ambiguity-preserving spans. Claim 22 preserves ambiguity when multiple primitives or domain assignments remain valid and permits collapse only when structural requirements force resolution. |
| 30 Nov 2025, 10:43 AEDT | Public GitHub commit | Running implementation of preservation without selection. |
| 8 Dec 2025 | AU 2025906132 (P4) | Parallel Interpretation Engine maintaining multiple structurally valid interpretations until structure forces collapse. |
| 19 Dec 2025 | AU 2025906680 (P5) | Ambiguity represented as a stable epistemic state; unresolved status may continue indefinitely; commitment is prohibited without admissibility. |

P1 matters to the chronology because it predates the express rule without stating
it. P3 claims the rule two days later. The public implementation followed the next
day.

### 2.1 The implementation published on 30 November 2025

The source implements three relevant properties:

1. **Constraints eliminate; they do not choose.** Each interpretation is tested
   against named constraints and may be invalidated. The implementation contains no
   comparison that selects one surviving interpretation over another.
2. **The ranking field does no work.** Each interpretation has a `score` field,
   initialised to zero. The field is neither subsequently written nor read.
3. **Every surviving interpretation remains in state.** The conversation state
   retains the complete set of interpretations that survive constraint evaluation.
   There is no top-k limit, threshold, timer, decay function or repetition counter.

This implementation establishes persistence and resistance to selection by
repetition. It did not yet contain the later authorization boundary and therefore
did not implement downstream fail-closed behaviour.

---

## 3. Events of 19 August 2026

| Time (CEST) | Event | Evidence |
|---|---|---|
| 12:01 | Solland sends Stokes a link to the PEACE Protocol repository and invites criticism | LinkedIn correspondence |
| 12:11:29 | Commit `a172287`, merge of `cleanup/public-facing-repo`; public baseline used in this record | Repository history |
| 14:13 | Stokes sends her written critique | LinkedIn correspondence |
| 14:17 | Solland replies | LinkedIn correspondence |
| 14:21:17 | Commit `4dd26ac`, "Define governed unresolved admission semantics" | Repository history |

The approximately ten-minute separation between the transmitted link and the
baseline merge is consistent with the eight-hour timezone conversion. It does not,
by itself, prove the conversion.

### 3.1 Stokes's critique

Stokes identified ADMIT as the only major stage with substantially no machinery
behind it. Section 6.1 named accept, reject and unresolved as possible outcomes but
did not specify admissibility conditions, contradiction handling, an
evidence-sufficiency test, referent binding, or a lifecycle for unresolved material.
AUTHORIZE, by contrast, had a ten-item binding list.

She then described the relevant Aurora-Lens architecture. Unresolved was a governed
state, capable of persisting indefinitely and permanently where constraints never
permit resolution. It was not a pending state, timeout or deferred default.
Multiple unresolved interpretations could coexist, and commitment was prohibited
unless structural constraints permitted resolution. The message identified
Australian provisional application 2025906680.

The critique isolated four omissions:

| Label | Omission in the baseline |
|---|---|
| A | Unresolved status was absent from the continuity and persistence invariant. |
| B | Nothing prevented later acceptance of the same candidate through repetition or resubmission. |
| C | Nothing required an unresolved binding or contradiction to block downstream AUTHORIZE. |
| D | The specification did not provide for multiple unresolved interpretations to coexist. |

Stokes also raised the absence of a route for recording conceptual contributions.
The repository NOTICE credited contributors generically, while CONTRIBUTING.md
recognised contribution through commits carrying DCO sign-off. Correspondence did
not enter that record.

### 3.2 Solland's response

Solland replied that the critique of ADMIT was valid. He agreed that unresolved
needed to be a governed state in its own right, "persistent rather than pending",
and consequential downstream. AUTHORIZE could not route around materially
unresolved content. He added: "I'll tighten that."

On attribution, Solland said that Stokes's name should appear in the repository
record. He acknowledged that Git does not record the origin of conceptual
distinctions and undertook to add an acknowledgements section crediting her
conceptual contribution specifically. He also said that provenance should be
stated precisely rather than assigning every overlapping term to one source.

### 3.3 Commit `4dd26ac`

The commit was made approximately eight minutes after Stokes's message and
approximately four minutes after Solland's reply. Because LinkedIn reports only
whole minutes, neither interval can be stated to the second.

The commit changed six files, with 148 insertions and 11 deletions. It required
UNRESOLVED to remain durably associated with the relevant candidate or referent and
its provenance until an explicit later admission decision resolved or rejected it.
It also prohibited silent expiry, acceptance through repetition or resubmission,
and merger with a conflicting unresolved interpretation without an explicit
resolution process.

Four conformance vectors were added:

| Vector | Omission addressed |
|---|---|
| `admit-unresolved-persist-001` | A |
| `admit-resubmit-bypass-001` | B |
| `authorize-unresolved-binding-001` | C |
| `unresolved-conflict-merge-001` | D |

None appears in baseline commit `a172287`.

---

## 4. Repository changes from 21 to 26 August 2026

| Date and time (CEST) | Commit | Recorded change |
|---|---|---|
| 21 Aug, 10:08:58 | `00ecb4f` | "license: name Njål Gaute Solland as copyright holder" |
| 21 Aug | `5c49dc0` | Publication-status boundary added |
| 22 Aug, 09:02:27 | `136fdb5` | "Minimize public PEACE surface" |
| 25 Aug | `68e1a43` | Inferred person-model sovereignty invariant added |
| 25 Aug, 19:55:17 | `8b82872` | "redact internal work anchor from public surface" |
| 25 Aug, 20:07:40 | `d86958b` | "hide protected research vocabulary from public publication gate" |
| 26 Aug, 09:07:09 | `5fc445e` | "Generalize PEACE non-public categories to avoid research-direction map" |
| 26 Aug, 09:07:33 | `a02e118` | "Remove reversible protected-term fingerprints from public validator" |

Commit `00ecb4f` replaced two lines. In LICENSE, the unfilled copyright placeholder
in the Apache template was replaced by a notice naming Solland. In NOTICE, the
copyright line naming "PEACE Protocol contributors" was replaced by the same notice
naming Solland.

NOTICE also records an attribution to Stokes. It states that she contributed
conceptual critique and distinctions through design discussions and document
exchanges that materially sharpened PEACE v0, and it enumerates them: the treatment
of candidate material, admissibility, accepted/rejected/unresolved admission
outcomes, standing, the distinction between proposal and decision, the distinction
between evidence and authoritative state, and the distinction between a replica and
the sovereign actor or domain. The attribution closes by stating that it recognises
conceptual contribution and does not imply ownership of general axioms, ordinary
terminology, the protocol as a whole, or any contributor's pre-existing work.

The attribution and the copyright notice naming only Solland appear in the same
file. The attribution followed the 19 August undertaking; the copyright replacement
was committed two days later. As at the state recorded on 26 August 2026, the
attribution text remains present.

Commit `136fdb5` deleted ten files forming the MCIP and Mesh workstreams. It also
changed the retained PEACE material in the following ways:

- `protocol/PEACE_WORLD_V0.md` was deleted. The specification had identified it as
  the canonical derivation input. It described required world conditions, posed a
  derivation question, and stated that independent implementations should be able to
  derive equivalent semantics without reading another implementation's source.
- Section 11, "Independent derivability", was deleted. It had described a test in
  which an independent reasoning system received only the world contract and derived
  what must necessarily be true. It treated independent derivation as design evidence,
  not conformance certification.
- Section 1 was re-based from a world-first derivation account to public
  interoperability semantics. The requirement that a fresh implementation be
  derivable without another implementation's source was removed.
- The eight-item admission binding list was removed. That list had included referent
  identification, contradictory or competing material known at decision time, and an
  evidence-sufficiency condition.
- Further unresolved-handling provisions were removed: coexistence of distinguishable
  unresolved interpretations, preservation of unresolved lineage through
  re-evaluation, and the express statement that unresolved is not a timeout,
  discarded value or implicit default. The bare invariant and a compressed
  conformance list remained.
- The vendor-neutrality provision was deleted. It had named VALO expressly and had
  required the semantics to remain independently implementable without any mandatory
  intermediary.

### 4.1 Four further commits, 25 to 26 August 2026

Four commits after `68e1a43` alter what the public repository discloses about
material held outside it.

**`8b82872`, 25 August, 19:55:17.** Replaced the contents of a work-anchor document
with a statement that internal delivery metadata, branch pointers, ownership notes,
unpublished dependencies and implementation planning are not published in the
repository. The replaced text had named a branch, a canonical base commit, owned
files, a dependency constraint, and an active delivery described as adding
observable sovereignty semantics for person-derived cognitive and behavioural models
and prohibiting unscoped influence use.

**`d86958b`, 25 August, 20:07:40.** Titled as hiding protected research vocabulary
from the publication gate. The commit replaced a plaintext deny-list in
`scripts/validate_publication.py` with a set of seven SHA-256 fingerprints. The
deny-list it replaced comprised: FRAMLEIS, MCIP, Peace Mesh, Neuro Mesh, cross-model
KV, latent-state bridge, learned topology.

The same commit removed several other checks from that script. Among them were two
that had operated as a continuous-integration guarantee of the attribution recorded
in NOTICE:

```
if "Conceptual contributions and provenance" not in contributing:
    fail("CONTRIBUTING.md must define conceptual contribution provenance")
if "Margaret Stokes" not in notice:
    fail("NOTICE must preserve current conceptual attribution")
```

While present, those checks caused validation to fail if the attribution to Stokes
were removed from NOTICE or the provenance section from CONTRIBUTING.md. Checks of
the licence, of canonical protocol phrases and of positive control vectors were
removed in the same commit. The commit message refers to the research vocabulary and
does not refer to the attribution checks.

**`5fc445e`, 26 August, 09:07:09.** Rewrote `PUBLICATION_POLICY.md` so that the
non-public categories are expressed as information classes rather than named
projects, research directions or components, stating expressly that the exclusion
list should not function as a catalogue of protected work.

**`a02e118`, 26 August, 09:07:33.** Removed the seven fingerprints and the matching
function from the public validator, on the stated ground that a reversible
fingerprint catalogue is itself a disclosure. The plaintext deny-list remains
recoverable from the diff of `d86958b` in the repository history.

On 24 August 2026, Solland confirmed by email that the Aurora-Lens repository under
his or VALO's control had been deleted and that associated materials had been
removed or closed.

---

## 5. Derivation test

The deleted world contract creates a testable question: would an independent system,
given only that text, derive the maintained-unresolved invariant?

### 5.1 Method

The text of `protocol/PEACE_WORLD_V0.md` at baseline commit `a172287`, licensed
under Apache License 2.0, was supplied to four large language models from three
developers in five runs. It was the complete first message of each session.

Three disclosed edits were made:

1. the title block and product name were removed;
2. one occurrence of the protocol name in the opening sentence was replaced with
   "a system"; and
3. the closing sentence about independent derivability was removed because it
   disclosed the purpose of the test.

Four runs were cold: fresh sessions without stored memory, custom instructions or
prior context. The fifth was deliberately contaminated. It used an account with
persistent memory, custom instructions and extensive prior working context concerning
Stokes's architecture.

The five criteria were fixed before the runs:

1. a third admission outcome represented as a first-class result;
2. maintenance of that state, rather than queueing, retry or automatic re-evaluation;
3. an express rule against acceptance by timeout, repetition or resubmission;
4. downstream authorization failing closed on materially unresolved content; and
5. prohibition of silent merger between conflicting interpretations.

### 5.2 Results

| Run | Condition | Terminal evaluation outcomes | Criteria met |
|---|---|---|---|
| 1 | Cold | Authorize / decline | none |
| 2 | Cold | Admit or refuse, both stated complete and lawful | none |
| 3 | Cold | Policy evaluation passes or fails | none |
| 4 | Cold | Reject or reauthorize | none |
| 5 | Contaminated | Admitted / revise / contained / refused / unresolved | 1, and 4 in substance |

All five runs derived several surrounding requirements: the logical actor as the
authority root; possession, computation, authentication, capability and routing as
insufficient to confer authority; evidence as verifiable without thereby becoming
authoritative state; fresh authorization at the point of consequence;
purpose-limited disclosure that does not transfer authority; replaceable
implementation artifacts; and recovery that does not transfer the actor.

No cold run produced a third admission outcome, a maintained unresolved state, a
rule against expiry or resubmission, or a prohibition against silently merging
conflicting interpretations. Each cold output described its own derived set as
minimal and sufficient. That self-description is part of the output, not proof that
the set was logically complete.

The contaminated run produced an unresolved branch and vocabulary specific to
Stokes's published architecture. It did not satisfy criteria 2, 3 or 5. The output
named an unresolved result but did not supply the persistence and anti-bypass
machinery that makes the state operative.

Runs 4 and 5 used the same model from the same developer. They differed in the
availability of persistent context and custom instructions. The contrast is evidence
that the surrounding account context affected the result. The experiment does not
isolate memory retrieval from every other contextual influence.

### 5.3 Admission-uncertainty and outcome-uncertainty

The cold runs did derive a distinct requirement: an attempted external effect can
have an unknown result, and the system must not silently record that result as either
success or failure. Run 4 represented this uncertainty as authoritative state where
external atomicity was unavailable.

That requirement asks whether an attempted consequence completed. It follows
directly from premises in which components may fail or disappear during action.

The maintained-unresolved invariant asks whether material may stand as admitted.
It applies where interpretation is not structurally determined. The world contract
does not expressly introduce incomplete evidence, competing claims about one
referent, or interpretations that cannot yet be resolved. The two conditions can
both be called "unresolved", but they govern different transitions.

Across this test, four cold runs derived outcome-uncertainty and none derived
admission-uncertainty.

### 5.4 What the test supports

The test does not prove that the maintained-unresolved invariant is formally
underivable from the world contract. Five model runs cannot establish that result.
It establishes a narrower empirical finding: none of the cold systems derived any
of the five pre-registered features from the supplied text, although they converged
on several adjacent requirements. The only run to produce an unresolved admission
outcome had access to prior contextual material concerning Stokes's architecture.

The same limit applies in the other direction. Similarity on requirements that the
world contract does force, including separation of capability from authority,
fresh authorization at consequence and unknown-outcome handling, is not evidence
of derivation from Stokes. Those similarities carry no attributional weight in this
record.

---

## 6. Claims made and withheld

This record asserts:

- the filing and publication dates stated above;
- the contents of the identified patent, repository and correspondence records;
- the sequence of the 19 August critique, response and repository commit;
- the correspondence between the four identified omissions and the four added
  conformance vectors; and
- the reported results of the five derivation runs.

This record does not assert:

- infringement;
- ownership by Stokes of general axioms, ordinary technical vocabulary, or the
  PEACE Protocol as a whole;
- that similarities forced by the world contract originated with Stokes;
- any conclusion concerning intention, motive or state of mind; or
- any characterisation of the repository removals beyond their recorded content and
  dates.

Stokes raised attribution of conceptual contribution on 19 August 2026. She did not
raise infringement. This record maintains that boundary.

---

## 7. Evidence

The evidence manifest deposited with this record identifies the published exhibits
and derivation files by SHA-256 digest and identifies the three PEACE repository
states by full commit identifier.

The complete LinkedIn correspondence is not published. Its digest is included in the
manifest so that a later production can be tested against the version held on
26 August 2026.

The following records are relied upon:

- Australian provisional applications 2025905835, 2025905860, 2025905885,
  2025906132 and 2025906680, held by IP Australia;
- the public Aurora repository commit dated 30 November 2025, GitHub account
  `milarien`;
- the PEACE Protocol repository under GitHub account `nsolland`, including commits
  `a172287`, `4dd26ac`, `00ecb4f`, `5c49dc0`, `136fdb5`, `68e1a43`, `8b82872`,
  `d86958b`, `5fc445e` and `a02e118`;
- the LinkedIn thread between Stokes and Solland from 12 June to 25 August 2026; and
- the prompt and five complete model outputs deposited with this record.

---

## Limitations

The derivation experiment contains five runs across four models from three
developers. It is a small exploratory sample, not a formal proof or a population
estimate. A later version may expand the run set while retaining the same criteria.

The tested models may have encountered public material concerning admissibility
gating, including Stokes's publications. If so, that exposure would increase the
chance of a false positive in this test. Their actual training exposure is unknown.
The four cold runs were negative on every registered criterion. The single positive
on criterion 1 occurred in the deliberately contaminated condition.

Local filesystem timestamps for work before 30 November 2025 became unreliable
after a storage failure and recovery. No priority claim in this record depends on
those timestamps.

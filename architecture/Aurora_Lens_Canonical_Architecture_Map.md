# Aurora-Lens Canonical Architecture Map

**Effective date:** 27 August 2026  
**Status:** Public explanatory map  
**Purpose:** Describe the stable Aurora-Lens architecture without disclosing private source locations, patent specifications, internal project records, or implementation-sensitive mechanics.

## Reading this map

- **Architectural invariant** identifies a constraint that defines the governance model.
- **Capability statement** identifies part of the architecture's defined scope. It does not claim that every deployment implements every surface.
- **Runtime vocabulary** identifies public outcome or component names associated with an implementation snapshot.
- **Architectural interpretation** explains how the parts relate. It is not a claim about patent validity, claim construction, unity of invention, or another system's derivation.

## 1. Short definition

**Architectural invariant.** Aurora-Lens is a deterministic commitment-governance architecture. It determines whether a candidate state, interpretation, determination, output, release, or action is admissible to become operative or consequential.

It is broader than a content filter, retrieval checker, fact verifier, authorisation service, or language-model guardrail. The invariant beneath those deployment surfaces is:

> A candidate does not acquire standing or consequence merely because a model, retrieval system, application, or prior state produced it. Commitment requires independent admissibility.

Where admissibility is not established, Aurora-Lens preserves governed non-commitment rather than forcing resolution.

## 2. Commitment surfaces

**Architectural interpretation.** The same governance question can apply to candidate information, interpretation, determination, output, release, or action:

> May this candidate commitment become operative under the present governed state and applicable constraints?

This public map treats those as commitment surfaces governed by the same admissibility invariant. That architectural presentation does not determine whether a patent office or court would treat particular subject matter as one invention, several inventions, anticipated, obvious, valid, or infringed.

## 3. Capability and deployment

**Capability statement.** The capability envelope describes the commitment surfaces the architecture is defined to govern. Deployment topology is the subset a host elects to use.

A host may use Aurora-Lens only for admission into governed state and rely on a separate downstream control plane. Another host may also use it at output, release, or action boundaries. Selective deployment identifies what that deployment does; it does not silently rewrite the architecture's public definition.

## 4. From availability to governed state

**Architectural invariant.** Information is not governed state merely because it is available. Depending on domain and consequence, admissibility may require sufficient provenance, authority, scope, present support, freshness, identity, relationship consistency, contradiction handling, and continuity with maintained state.

No single factor is universally sufficient. Material may be authentic but stale, authoritative for one property but not another, outside scope, contradictory, referentially ambiguous, or insufficient for the proposed commitment.

## 5. Ambiguity and unresolved state

**Architectural invariant and capability statement.** Ambiguity is a legitimate governed state. Multiple structurally valid interpretations may coexist, and collapse is permitted only when applicable constraints support it.

Aurora-Lens is defined to preserve unresolved referents, competing entity bindings, conflicting evidence, insufficient context, and uncertain state transitions. Non-resolution is an intended governance result, not automatically a model failure.

## 6. Independent authority boundary

**Architectural invariant.** A model may propose claims, interpretations, drafts, classifications, summaries, or actions. It may not establish the authority of its own output merely by producing it.

Generation, confidence, fluency, repetition, or endorsement by another model does not independently establish truth, sufficient evidence, identity, relationship, renewed authority, expanded scope, or permission for consequential commitment.

The model proposes. Governance determines whether the proposal may stand or proceed.

## 7. Governed outcomes

**Runtime vocabulary.** The public runtime dispositions are:

- `PASS`: the candidate is admitted under the applicable conditions.
- `CONTAIN`: commitment is withheld while constrained continuation may seek clarification or revalidation.
- `FORCE_REVISE`: the candidate cannot be released as presented and must be revised within the governance boundary.
- `HARD_STOP`: the relevant commitment is closed and the prohibited consequence does not proceed.

These are governance dispositions, not confidence scores.

## 8. Persistence and lawful continuation

**Architectural invariant and capability statement.** Persistent governed state permits unresolved identity, relationships, ambiguity, and prior commitments to remain visible across processing boundaries rather than being silently reconstructed as a fresh narrative.

Continuation and commitment are distinct. A conversation or process may continue toward clarification, revalidation, supervision, or termination without reopening a commitment that governance has closed.

## 9. Evidence and audit

**Architectural invariant.** Auditability belongs to the decision architecture. The aim is to reconstruct the governed state, applicable conditions, and outcome at the time of decision rather than ask a model to manufacture a retrospective explanation.

The public invariant is reconstruction and traceability. Internal storage, signing, policy, cryptographic, and enforcement mechanics are outside this public map.

## 10. Distinctions from adjacent systems

**Architectural interpretation.** Retrieval determines what material is made available. Aurora-Lens determines what that material is entitled to establish.

Verification can contribute evidence, but a verifier's output does not become authority solely because it is labelled verification.

Conventional filters commonly classify content against a pattern, score, or rule. Aurora-Lens evaluates whether a concrete commitment is admissible relative to governed state and applicable conditions. The same surface text may be admissible in one state and inadmissible in another.

## 11. Public deployment profiles

**Capability statement.** Public deployment profiles include:

1. **Governed-state admission:** govern what information, identity, relationships, and evidence may enter or alter operative state.
2. **Output-release governance:** govern whether generated candidate output may be released, revised, contained, or stopped.
3. **Consequential candidate governance:** evaluate a proposed release or action before consequence.
4. **Hybrid governance:** apply the same admissibility invariant at more than one commitment surface.

These profiles identify deployment choices. They are not evidence that every current deployment enables every profile.

## 12. Canonical public explanation

Aurora-Lens is the independent gate between a candidate and commitment. A model, retrieval system, agent, or application can propose information, conclusions, outputs, or actions, but none becomes operative merely because it was produced. Aurora-Lens evaluates whether the relevant evidence, state, identity, authority, freshness, scope, and unresolved conditions permit commitment. If they do not, the system preserves non-commitment through clarification, containment, revision, refusal, or stop.

## Public sources

- Margaret Stokes, *Aurora-Lens Public Canonical Statement*, [Zenodo 21930519](https://doi.org/10.5281/zenodo.21930519).
- Margaret Stokes, *The Consequence Boundary: Aurora-Lens and the Architecture of Admissibility Before AI Action*, [Zenodo 21773363](https://doi.org/10.5281/zenodo.21773363).
- Margaret Stokes, *Epistemic Legitimacy as a Governance Layer for Large Language Models: Architecture and Implementation*, [Zenodo 19504665](https://doi.org/10.5281/zenodo.19504665).

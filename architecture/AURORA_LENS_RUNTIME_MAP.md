# Aurora-Lens Runtime Map

**Effective date:** 27 August 2026  
**Status:** Public architecture and reported-runtime snapshot  
**Purpose:** Record the separation among Aurora, Lens, Governor, the Persistent Existence Frame (PEF), and the audit path without publishing proprietary source code.

## Support labels

- **Architectural invariant:** a stable constraint of the Aurora-Lens governance model.
- **Capability statement:** part of the architecture's defined scope; it may not be enabled at every deployment surface.
- **Reported runtime structure:** public implementation vocabulary recorded as at the effective date. It is not a complete source inventory or an assurance that later private versions retain the same names.

## Reported runtime chain

```text
Lens
  -> verification flags and applicable policy
  -> GovernanceDecision
  -> CanonicalScannerGateBridge
  -> Governor and audit ledger
```

This chain records the public separation of decision, continuation, and evidence handling. It does not disclose the full internal enforcement, storage, policy, or cryptographic implementation.

## Component roles

### Aurora

**Architectural invariant.** Aurora is the underlying reasoning architecture and substrate. It is not identical to the Lens admissibility boundary, Governor, or PEF.

### Lens

**Architectural invariant.** Lens governs admissibility. It determines whether a candidate may cross a governed commitment boundary and produces a governance disposition for continuation and audit.

### Governor

**Architectural invariant.** Governor determines lawful continuation under the boundary established by Lens. It does not convert a closed commitment into permission.

### PEF: Persistent Existence Frame

**Architectural invariant and capability statement.** PEF preserves governed continuity across processing boundaries. It can maintain ambiguity, relationships, prior commitments, and unresolved state rather than treating each turn as an isolated narrative.

### CanonicalScannerGateBridge

**Reported runtime structure.** `CanonicalScannerGateBridge` is the public name recorded for the bridge carrying a governance decision into the Governor and audit path.

### BuiltinBridge

**Reported runtime structure.** `BuiltinBridge` is recorded as an in-process testing bridge. It is not presented as the forensic production path.

## Separation invariant

Lens can contain, revise, or refuse a candidate without a later continuation step creating permission. Governor can record or route a decision without creating the admissibility boundary. PEF can retain ambiguity before a Lens determination is reached.

Keeping these roles separate makes it possible to identify where a candidate was admitted, held, revised, refused, continued, or recorded. The separation is part of the architecture's auditability.

## Public runtime dispositions

- `PASS`: the candidate is admitted under the applicable conditions.
- `CONTAIN`: commitment is withheld while constrained continuation may seek clarification or revalidation.
- `FORCE_REVISE`: the candidate cannot be released as presented and must be revised within the governance boundary.
- `HARD_STOP`: the relevant commitment is closed and the prohibited consequence does not proceed.

These are governance dispositions, not confidence scores.

## Scope boundary

The component names in this document are deliberately public. The map does not claim that every architectural capability is implemented in every deployment, and it does not publish private source locations, complete class inventories, policy packs, evidence-vault mechanics, signing details, or roadmap material.

Public explanations should identify whether a statement is an architectural invariant, a capability statement, or a reported runtime fact. They should not turn a defined capability into an unsupported claim about a particular deployment.

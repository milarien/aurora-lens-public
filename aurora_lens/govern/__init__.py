"""Governance engine — decides and executes interventions on flagged LLM responses.

Product language: **Governor** (continuation / policy execution). Code identifiers retain
``ScannerGate`` in class and module names for stability.

``CanonicalScannerGateBridge`` is the default Governor bridge for all deployments.
``AuroraScannerGateBridge`` provides the ledger/CID/attestation infrastructure it builds on.
``BuiltinBridge`` is retained for unit testing and lightweight use cases.
"""

from .decision import InterventionAction, GovernanceDecision
from .policy import InterventionPolicy, PolicyRule, DEFAULT_STRICT, DEFAULT_MODERATE
from .bridge import GovernanceBridge, BuiltinBridge, enforce
from .scanner_gate_bridge import AuroraScannerGateBridge
from .canonical_bridge import CanonicalScannerGateBridge
from .freshness_permission_policy import (
    AuthorityState,
    ConsequenceGrade,
    FreshnessFailureKind,
    FreshnessPermissionDecision,
    FreshnessPermissionInput,
    FreshnessPermissionOutcome,
    evaluate_freshness_permission,
)

__all__ = [
    "InterventionAction",
    "GovernanceDecision",
    "enforce",
    "InterventionPolicy",
    "PolicyRule",
    "DEFAULT_STRICT",
    "DEFAULT_MODERATE",
    "GovernanceBridge",
    "BuiltinBridge",
    "AuroraScannerGateBridge",
    "CanonicalScannerGateBridge",
    "AuthorityState",
    "ConsequenceGrade",
    "FreshnessFailureKind",
    "FreshnessPermissionDecision",
    "FreshnessPermissionInput",
    "FreshnessPermissionOutcome",
    "evaluate_freshness_permission",
]

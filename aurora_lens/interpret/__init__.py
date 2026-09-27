"""Interpretation layer — pluggable extraction backends."""

from .schema import ExtractedClaim, ExtractionResult
from .base import ExtractionBackend
from .pef_updater import update_pef, apply_prepopulated_claims
from .turn_act import TurnAct, classify_turn_act
from .pef_admission import (
    PEFAdmissionDecision,
    PEFAdmissionEvidence,
    PEFAdmissionResult,
    pef_admission_result_wire_dict,
)

__all__ = [
    "ExtractedClaim",
    "ExtractionResult",
    "ExtractionBackend",
    "update_pef",
    "apply_prepopulated_claims",
    "TurnAct",
    "classify_turn_act",
    "PEFAdmissionDecision",
    "PEFAdmissionEvidence",
    "PEFAdmissionResult",
    "pef_admission_result_wire_dict",
]

"""Compatibility shim: ``aurora_lens.governor.operator_override`` is canonical."""
from __future__ import annotations

from aurora_lens.governor.operator_override import (
    CONSTRAINED_FIELDS,
    FREE_FIELDS,
    IMMUTABLE_FIELDS,
    OperatorOverrideError,
    OverrideViolation,
    validate_override,
)
import aurora_lens.governor.operator_override as _operator_override

_PATHWAY_OUTPUT_MODES = _operator_override._PATHWAY_OUTPUT_MODES
_PATHWAY_SEMANTIC_CLASS = _operator_override._PATHWAY_SEMANTIC_CLASS
_PATHWAY_SUB_LEVEL = _operator_override._PATHWAY_SUB_LEVEL

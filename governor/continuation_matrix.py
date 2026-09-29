"""Compatibility shim: ``aurora_lens.governor.continuation_matrix`` is canonical."""
from __future__ import annotations

from aurora_lens.governor.continuation_matrix import (
    CONTINUATION_MATRIX,
    ContinuationMatrix,
    ContinuationRow,
)
import aurora_lens.governor.continuation_matrix as _continuation_matrix

_TABLE = _continuation_matrix._TABLE
_CLOSED_STATUSES = _continuation_matrix._CLOSED_STATUSES

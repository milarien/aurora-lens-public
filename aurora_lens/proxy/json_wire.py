"""JSON-safe reduction for HTTP/SSE proxy responses.

Shared by :mod:`aurora_lens.proxy.app` and :mod:`aurora_lens.proxy.openai_compat`.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any


def wire_nonneg_int(val: Any) -> int:
    """Coerce adapter usage / token fields to a non-negative int without raising."""
    if val is None:
        return 0
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, int):
        return max(0, val)
    if isinstance(val, float):
        if val != val or val in (float("inf"), float("-inf")):  # nan / inf
            return 0
        return max(0, int(val))
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return 0
        try:
            return max(0, int(float(s)))
        except ValueError:
            return 0
    return 0


def wire_encode(obj: Any) -> Any:
    """Reduce values to JSON primitives (dict/list/str/int/float/bool/None).

    Enums → name; dataclasses → dict; unknown → ``str(obj)``.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Enum):
        return obj.name
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        try:
            return wire_encode(dataclasses.asdict(obj))
        except Exception:
            return wire_encode(vars(obj))
    if isinstance(obj, dict):
        return {str(k): wire_encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [wire_encode(v) for v in obj]
    return str(obj)

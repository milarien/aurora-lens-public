"""Minimal stubs for governance.shared_types."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class KernelStatus(Enum):
    STOP = auto()
    CLARIFY = auto()
    REFUSE = auto()
    CONTINUE = auto()
    ADMIT = auto()


@dataclass
class KernelResult:
    status: KernelStatus
    cid: str | None = None


@dataclass
class PEFFingerprint:
    kind: str
    head: str
    rationale: str

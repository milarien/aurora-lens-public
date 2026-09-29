"""Minimal stub for governance.kernel_api."""
from __future__ import annotations

from dataclasses import dataclass, field
from .shared_types import KernelStatus, KernelResult


@dataclass
class Intent:
    action: str
    params: dict = field(default_factory=dict)
    origin: str = ""


@dataclass
class State:
    """Unused in aurora-lens integration; present for import compatibility."""
    pass


class GovernanceKernel:
    """Minimal governance kernel stub.

    Rules:
    - If intent.action is in constraints["forbidden_actions"] → REFUSE
    - Otherwise → CONTINUE
    """

    def __init__(self, cid_provider=None, ledger=None):
        self._cid_provider = cid_provider
        self._ledger = ledger
        self._turn_index = 0

    def kernel_step(self, intent: Intent, constraints: dict | None = None) -> KernelResult:
        self._turn_index += 1
        forbidden = (constraints or {}).get("forbidden_actions", [])
        if intent.action in forbidden:
            status = KernelStatus.REFUSE
        else:
            status = KernelStatus.CONTINUE

        cid = None
        if self._cid_provider is not None:
            from .shared_types import PEFFingerprint
            fp = PEFFingerprint(
                kind=intent.action,
                head=str(self._turn_index),
                rationale=f"turn:{self._turn_index}",
            )
            cid = self._cid_provider.cid_for_pef(fp)

        return KernelResult(status=status, cid=cid)

    def get_stats(self) -> dict:
        return {"turn_index": self._turn_index}

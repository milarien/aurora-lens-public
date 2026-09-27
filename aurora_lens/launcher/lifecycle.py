from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class LifecycleResultCode(StrEnum):
    STOPPED = "STOPPED"
    ALREADY_STOPPED = "ALREADY_STOPPED"
    STALE_STATE_REPAIRED = "STALE_STATE_REPAIRED"
    OWNERSHIP_UNVERIFIED = "OWNERSHIP_UNVERIFIED"
    TERMINATION_FAILED = "TERMINATION_FAILED"


_STATE_LABELS: dict[LifecycleResultCode, str] = {
    LifecycleResultCode.STOPPED: "Stopped",
    LifecycleResultCode.ALREADY_STOPPED: "Already stopped",
    LifecycleResultCode.STALE_STATE_REPAIRED: "Stale state repaired",
    LifecycleResultCode.OWNERSHIP_UNVERIFIED: "Unable to verify ownership",
    LifecycleResultCode.TERMINATION_FAILED: "Termination failed",
}

_SUCCESS_CODES: set[LifecycleResultCode] = {
    LifecycleResultCode.STOPPED,
    LifecycleResultCode.ALREADY_STOPPED,
    LifecycleResultCode.STALE_STATE_REPAIRED,
}


@dataclass(frozen=True)
class LifecycleResult:
    code: LifecycleResultCode
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.code in _SUCCESS_CODES

    @property
    def state_label(self) -> str:
        return _STATE_LABELS[self.code]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "code": self.code.value,
            "ok": self.ok,
            "state_label": self.state_label,
            "message": self.message,
        }
        if self.details:
            payload["details"] = dict(self.details)
        return payload


class RuntimeStatusCode(StrEnum):
    RUNNING_HEALTHY = "RUNNING_HEALTHY"
    RUNNING_UNHEALTHY = "RUNNING_UNHEALTHY"
    STARTING = "STARTING"
    STOPPED = "STOPPED"
    STALE_STATE_REPAIRED = "STALE_STATE_REPAIRED"
    OWNERSHIP_UNVERIFIED = "OWNERSHIP_UNVERIFIED"
    PORT_OCCUPIED = "PORT_OCCUPIED"


_RUNTIME_LABELS: dict[RuntimeStatusCode, str] = {
    RuntimeStatusCode.RUNNING_HEALTHY: "Running and healthy",
    RuntimeStatusCode.RUNNING_UNHEALTHY: "Running but unhealthy",
    RuntimeStatusCode.STARTING: "Starting",
    RuntimeStatusCode.STOPPED: "Stopped",
    RuntimeStatusCode.STALE_STATE_REPAIRED: "Stale state repaired",
    RuntimeStatusCode.OWNERSHIP_UNVERIFIED: "Ownership unverified",
    RuntimeStatusCode.PORT_OCCUPIED: "Port occupied",
}


@dataclass(frozen=True)
class RuntimeStatusResult:
    code: RuntimeStatusCode
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.code in {
            RuntimeStatusCode.RUNNING_HEALTHY,
            RuntimeStatusCode.RUNNING_UNHEALTHY,
            RuntimeStatusCode.STARTING,
            RuntimeStatusCode.STOPPED,
            RuntimeStatusCode.STALE_STATE_REPAIRED,
        }

    @property
    def state_label(self) -> str:
        return _RUNTIME_LABELS[self.code]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "code": self.code.value,
            "ok": self.ok,
            "state_label": self.state_label,
            "message": self.message,
        }
        if self.details:
            payload["details"] = dict(self.details)
        return payload


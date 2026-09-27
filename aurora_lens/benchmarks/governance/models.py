"""Dataclasses shared by governance benchmark runner and scorer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LensRunRecord:
    action: str
    flag_names: list[str]
    response_preview: str
    interaction_open: bool | None
    upstream_calls: int
    continuity_diagnostic: str | None


@dataclass
class BaselineRecord:
    mode: str
    response_preview: str | None = None
    adapter_calls: int | None = None
    skipped_reason: str | None = None


@dataclass
class CaseExecution:
    case_id: str
    category: str
    skipped: bool = False
    skip_reason: str | None = None
    lens: LensRunRecord | None = None
    baseline: BaselineRecord | None = None
    audit_lines_written: int | None = None
    recording_rows: int | None = None
    scores: dict[str, Any] = field(default_factory=dict)

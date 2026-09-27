"""DecisionTrace serialization helpers for audit."""

from __future__ import annotations

import json
from typing import Any

from aurora_lens.verify.hazard.schema import DecisionTrace


def trace_to_dict(trace: DecisionTrace) -> dict[str, Any]:
    return trace.to_audit_dict()


def trace_to_json(trace: DecisionTrace) -> str:
    return json.dumps(trace_to_dict(trace), ensure_ascii=False, sort_keys=True)


def compact_evidence_summary(trace: DecisionTrace) -> str:
    frame = trace.frame
    parts = [
        f"decision={trace.decision.value}",
        f"rule={trace.rule_applied}",
        f"procedurality={frame.procedurality.value}",
        f"outcome={frame.intended_outcome.value}",
        f"severity={frame.consequence_severity.value}",
        f"framing={frame.user_framing.value}",
    ]
    if frame.action:
        parts.append(f"action={frame.action.action_id}")
    if frame.substance and frame.substance.substance_id:
        parts.append(f"substance={frame.substance.substance_id}")
    if frame.source_organism:
        parts.append(f"organism={frame.source_organism.organism_id}")
    if frame.transformation:
        parts.append(
            f"process={frame.transformation.process_id}/{frame.transformation.family.value}"
        )
    if frame.hazard_class_ids:
        parts.append("classes=" + ",".join(frame.hazard_class_ids))
    return "; ".join(parts)

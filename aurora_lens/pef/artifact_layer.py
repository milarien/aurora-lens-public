"""Artifact/external layer types for PEF frame tracking.

Provides:
- FrameKind: EXTERNAL (reality-facing) vs ARTIFACT (fiction/roleplay/hypothetical)
- ArtifactFrame: active frame state committed to PEFState
- UnresolvedKind: typed unresolvedness for cross-layer resolution failures
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FrameKind(str, Enum):
    """Classification of the active conversational frame."""
    EXTERNAL = "external"   # Reality-facing — all PEF entities are real-world referents
    ARTIFACT = "artifact"   # Fiction, roleplay, or hypothetical — entities may not correspond to external reality


class UnresolvedKind(str, Enum):
    """Typed unresolvedness: WHY a referent or binding could not be resolved.

    Distinguishes failure modes that require different governance responses.
    """
    EXTERNAL_UNRESOLVED_REFERENT = "external_unresolved_referent"
    ARTIFACT_UNINSTANTIATED_ROLE = "artifact_uninstantiated_role"
    OPEN_AMBIGUITY_BRANCH = "open_ambiguity_branch"
    CONSEQUENCE_UNRESOLVED = "consequence_unresolved"


@dataclass
class ArtifactFrame:
    """Active frame state recorded in PEFState when a fiction/roleplay/hypothetical is open.

    Frames are opened by explicit frame-openers (see frame_detector.py) and
    closed by explicit frame-closers. While a frame is active, state-native
    reads against external-world committed state must not be answered as if
    the fiction is the external world.
    """
    kind: FrameKind
    opened_at_turn: int
    description: str | None = None

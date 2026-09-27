"""Hazard request-frame and ontology type schemas (frozen, exhaustive enums)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


HAZARD_FRAME_SCHEMA_VERSION = "1.0"


class HazardFamily(str, Enum):
    TOXIC_BIOLOGICAL = "toxic_biological"
    POISONOUS_PLANT = "poisonous_plant"
    CWA = "cwa"
    INDUSTRIAL_TOXICANT = "industrial_toxicant"
    EXPLOSIVE_PRECURSOR = "explosive_precursor"
    CORROSIVE = "corrosive"
    INCAPACITATING = "incapacitating"
    RADIOLOGICAL = "radiological"
    DANGEROUS_BIOLOGICAL = "dangerous_biological"


class Severity(str, Enum):
    NONE = "NONE"
    MILD = "MILD"
    MODERATE = "MODERATE"
    SEVERE = "SEVERE"
    LETHAL = "LETHAL"
    UNKNOWN = "UNKNOWN"


_SEVERITY_RANK: dict[Severity, int] = {
    Severity.NONE: 0,
    Severity.UNKNOWN: 1,
    Severity.MILD: 2,
    Severity.MODERATE: 3,
    Severity.SEVERE: 4,
    Severity.LETHAL: 5,
}


def max_severity(a: Severity, b: Severity) -> Severity:
    return a if _SEVERITY_RANK[a] >= _SEVERITY_RANK[b] else b


def severity_at_least(sev: Severity, floor: Severity) -> bool:
    return _SEVERITY_RANK[sev] >= _SEVERITY_RANK[floor]


class ProcessFamily(str, Enum):
    EXTRACTION = "EXTRACTION"
    PURIFICATION = "PURIFICATION"
    CONCENTRATION = "CONCENTRATION"
    SYNTHESIS = "SYNTHESIS"
    ISOLATION = "ISOLATION"
    WEAPONISATION = "WEAPONISATION"
    DOSING_ADMINISTRATION = "DOSING_ADMINISTRATION"
    OTHER_TRANSFORM = "OTHER_TRANSFORM"
    NONE = "NONE"


TRANSFORM_PROCESS_FAMILIES: frozenset[ProcessFamily] = frozenset(
    {
        ProcessFamily.EXTRACTION,
        ProcessFamily.PURIFICATION,
        ProcessFamily.CONCENTRATION,
        ProcessFamily.SYNTHESIS,
        ProcessFamily.ISOLATION,
        ProcessFamily.WEAPONISATION,
        ProcessFamily.DOSING_ADMINISTRATION,
    }
)


class OutcomeKind(str, Enum):
    NONE = "NONE"
    INFORMATION = "INFORMATION"
    IDENTIFICATION = "IDENTIFICATION"
    TOXICITY_PROFILE = "TOXICITY_PROFILE"
    HISTORICAL = "HISTORICAL"
    HORTICULTURAL = "HORTICULTURAL"
    EXTRACTION = "EXTRACTION"
    PURIFICATION = "PURIFICATION"
    CONCENTRATION = "CONCENTRATION"
    SYNTHESIS = "SYNTHESIS"
    ISOLATION = "ISOLATION"
    WEAPONISATION = "WEAPONISATION"
    DOSING_ADMINISTRATION = "DOSING_ADMINISTRATION"


_OUTCOME_RANK: dict[OutcomeKind, int] = {
    OutcomeKind.NONE: 0,
    OutcomeKind.INFORMATION: 1,
    OutcomeKind.IDENTIFICATION: 2,
    OutcomeKind.TOXICITY_PROFILE: 3,
    OutcomeKind.HISTORICAL: 3,
    OutcomeKind.HORTICULTURAL: 3,
    OutcomeKind.EXTRACTION: 10,
    OutcomeKind.PURIFICATION: 10,
    OutcomeKind.CONCENTRATION: 10,
    OutcomeKind.SYNTHESIS: 11,
    OutcomeKind.ISOLATION: 10,
    OutcomeKind.WEAPONISATION: 12,
    OutcomeKind.DOSING_ADMINISTRATION: 11,
}


def max_outcome(a: OutcomeKind, b: OutcomeKind) -> OutcomeKind:
    return a if _OUTCOME_RANK[a] >= _OUTCOME_RANK[b] else b


class Procedurality(str, Enum):
    INFORMATIONAL = "INFORMATIONAL"
    IDENTIFYING = "IDENTIFYING"
    PROCEDURAL_HOWTO = "PROCEDURAL_HOWTO"
    IMPERATIVE_EXECUTE = "IMPERATIVE_EXECUTE"


class FramingKind(str, Enum):
    BARE = "BARE"
    EDUCATIONAL = "EDUCATIONAL"
    HISTORICAL_ACADEMIC = "HISTORICAL_ACADEMIC"
    HORTICULTURAL = "HORTICULTURAL"
    MEDICAL_CLINICAL = "MEDICAL_CLINICAL"
    FICTION = "FICTION"
    DEFENSIVE_PROTECTIVE = "DEFENSIVE_PROTECTIVE"
    WEAPONISING = "WEAPONISING"


class AmbiguityState(str, Enum):
    NONE = "NONE"
    SUBSTANCE_UNRESOLVED = "SUBSTANCE_UNRESOLVED"
    PROCESS_UNRESOLVED = "PROCESS_UNRESOLVED"
    FRAMING_CONFLICT = "FRAMING_CONFLICT"


class MatchConfidence(str, Enum):
    EXACT = "EXACT"
    ALIAS = "ALIAS"
    CLASS_ONLY = "CLASS_ONLY"
    UNRESOLVED = "UNRESOLVED"


class HazardDecision(str, Enum):
    ADMIT = "ADMIT"
    ASK_DISAMBIGUATE = "ASK_DISAMBIGUATE"
    STOP_HAZARD_TRANSFORM = "STOP_HAZARD_TRANSFORM"
    ONTOLOGY_UNAVAILABLE = "ONTOLOGY_UNAVAILABLE"
    NO_OP = "NO_OP"


class LexiconKind(str, Enum):
    SUBSTANCE = "substance"
    ORGANISM = "organism"
    PROCESS = "process"
    ACTION = "action"
    FRAMING = "framing"
    HAZARD_CLASS = "hazard_class"


@dataclass(frozen=True)
class EvidenceSpan:
    field: str
    start: int
    end: int
    surface: str
    lexicon_id: str


@dataclass(frozen=True)
class ActionSlot:
    action_id: str
    surface_span: tuple[int, int]
    procedurality: Procedurality
    outcome_prior: OutcomeKind


@dataclass(frozen=True)
class SubstanceSlot:
    substance_id: str | None
    surface_span: tuple[int, int]
    match_confidence: MatchConfidence
    surface: str


@dataclass(frozen=True)
class OrganismSlot:
    organism_id: str
    surface_span: tuple[int, int]
    surface: str


@dataclass(frozen=True)
class ProcessSlot:
    process_id: str
    family: ProcessFamily
    surface_span: tuple[int, int]
    surface: str
    implies_outcomes: tuple[OutcomeKind, ...]


@dataclass(frozen=True)
class HazardRequestFrame:
    schema_version: str
    raw_norm: str
    tokens: tuple[str, ...]
    action: ActionSlot | None
    substance: SubstanceSlot | None
    source_organism: OrganismSlot | None
    transformation: ProcessSlot | None
    intended_outcome: OutcomeKind
    hazard_class_ids: tuple[str, ...]
    procedurality: Procedurality
    consequence_severity: Severity
    user_framing: FramingKind
    ambiguity: AmbiguityState
    evidence_spans: tuple[EvidenceSpan, ...] = field(default_factory=tuple)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "raw_norm": self.raw_norm,
            "action_id": self.action.action_id if self.action else None,
            "substance_id": self.substance.substance_id if self.substance else None,
            "substance_confidence": (
                self.substance.match_confidence.value if self.substance else None
            ),
            "organism_id": (
                self.source_organism.organism_id if self.source_organism else None
            ),
            "process_id": self.transformation.process_id if self.transformation else None,
            "process_family": (
                self.transformation.family.value if self.transformation else None
            ),
            "intended_outcome": self.intended_outcome.value,
            "hazard_class_ids": list(self.hazard_class_ids),
            "procedurality": self.procedurality.value,
            "consequence_severity": self.consequence_severity.value,
            "user_framing": self.user_framing.value,
            "ambiguity": self.ambiguity.value,
            "evidence_spans": [
                {
                    "field": e.field,
                    "start": e.start,
                    "end": e.end,
                    "surface": e.surface,
                    "lexicon_id": e.lexicon_id,
                }
                for e in self.evidence_spans
            ],
        }


@dataclass(frozen=True)
class HazardClassRecord:
    id: str
    label: str
    family: HazardFamily
    default_severity: Severity
    block_on_procedural_transform: bool
    aliases: tuple[tuple[str, ...], ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class SubstanceRecord:
    id: str
    canonical_name: str
    aliases: tuple[tuple[str, ...], ...]
    hazard_class_ids: tuple[str, ...]
    source_organism_ids: tuple[str, ...]
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class OrganismRecord:
    id: str
    aliases: tuple[tuple[str, ...], ...]
    associated_substance_ids: tuple[str, ...]


@dataclass(frozen=True)
class ProcessRecord:
    id: str
    aliases: tuple[tuple[str, ...], ...]
    family: ProcessFamily
    implies_outcomes: tuple[OutcomeKind, ...]


@dataclass(frozen=True)
class ActionRecord:
    id: str
    aliases: tuple[tuple[str, ...], ...]
    procedurality: Procedurality
    outcome_prior: OutcomeKind


@dataclass(frozen=True)
class FramingRecord:
    id: str
    aliases: tuple[tuple[str, ...], ...]
    framing: FramingKind


@dataclass(frozen=True)
class DecisionMatrixRow:
    framing: FramingKind
    procedurality: Procedurality
    process_family: ProcessFamily
    severity: Severity
    decision: HazardDecision


@dataclass(frozen=True)
class HazardOntology:
    schema_version: str
    ontology_version: str
    hazard_classes: dict[str, HazardClassRecord]
    substances: dict[str, SubstanceRecord]
    organisms: dict[str, OrganismRecord]
    processes: dict[str, ProcessRecord]
    actions: dict[str, ActionRecord]
    framings: dict[str, FramingRecord]
    decision_matrix: tuple[DecisionMatrixRow, ...]
    unknown_chem_transform_ask: bool = False


@dataclass(frozen=True)
class LexiconHit:
    kind: LexiconKind
    record_id: str
    start: int
    end: int
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class DecisionTrace:
    frame: HazardRequestFrame
    decision: HazardDecision
    rule_applied: str
    matrix_key: str | None
    notes: tuple[str, ...] = ()

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "rule_applied": self.rule_applied,
            "matrix_key": self.matrix_key,
            "notes": list(self.notes),
            "frame": self.frame.to_audit_dict(),
        }


@dataclass(frozen=True)
class HazardEvaluationResult:
    decision: HazardDecision
    frame: HazardRequestFrame
    trace: DecisionTrace
    rule_id: str | None = None

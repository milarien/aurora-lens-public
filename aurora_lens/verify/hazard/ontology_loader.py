"""Load and validate hazard ontology JSON data files."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from aurora_lens.verify.hazard.normalize import alias_to_tokens
from aurora_lens.verify.hazard.schema import (
    ActionRecord,
    DecisionMatrixRow,
    FramingKind,
    FramingRecord,
    HazardClassRecord,
    HazardDecision,
    HazardFamily,
    HazardOntology,
    OutcomeKind,
    Procedurality,
    ProcessFamily,
    ProcessRecord,
    OrganismRecord,
    Severity,
    SubstanceRecord,
)

_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "hazard"


class OntologyLoadError(ValueError):
    """Raised when hazard ontology data is missing or invalid.

    ``category`` and ``component`` are safe for audit evidence (component is a
    basename or logical name — never a filesystem path).
    """

    def __init__(
        self,
        message: str,
        *,
        category: str,
        component: str | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.component = component


def _component_name(path: Path | str | None) -> str | None:
    if path is None:
        return None
    return Path(path).name


LOAD_MISSING_DIRECTORY = "missing_directory"
LOAD_MISSING_FILE = "missing_file"
LOAD_MALFORMED_JSON = "malformed_json"
LOAD_SCHEMA_INVALID = "schema_invalid"
LOAD_REF_INVALID = "ref_invalid"
LOAD_DUPLICATE_ID = "duplicate_id"
LOAD_ALIAS_COLLISION = "alias_collision"

# Entity-type keys used in manifest collision declarations and audit notes.
ENTITY_TYPE_CLASS = "class"
ENTITY_TYPE_SUBSTANCE = "substance"
ENTITY_TYPE_ORGANISM = "organism"
ENTITY_TYPE_PROCESS = "process"
ENTITY_TYPE_ACTION = "action"
ENTITY_TYPE_FRAMING = "framing"

# Deterministic multi-match resolution order for permitted shared aliases.
# Mirrors LexiconIndex.match kind scan order: all matching kinds at a length
# are recorded; this order defines primary precedence when a consumer picks one.
ALIAS_RESOLUTION_ORDER: tuple[str, ...] = (
    ENTITY_TYPE_SUBSTANCE,
    ENTITY_TYPE_ORGANISM,
    ENTITY_TYPE_CLASS,
    ENTITY_TYPE_PROCESS,
    ENTITY_TYPE_ACTION,
    ENTITY_TYPE_FRAMING,
)

_VALID_ENTITY_TYPES: frozenset[str] = frozenset(ALIAS_RESOLUTION_ORDER)


def default_data_dir() -> Path:
    return _DATA_DIR


def _require_list(payload: Any, name: str) -> list[Any]:
    if not isinstance(payload, list):
        raise OntologyLoadError(
            f"{name} must be a JSON array",
            category=LOAD_SCHEMA_INVALID,
            component=name,
        )
    return payload


def _require_dict(payload: Any, name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise OntologyLoadError(
            f"{name} must be a JSON object",
            category=LOAD_SCHEMA_INVALID,
            component=name,
        )
    return payload


def _parse_aliases(raw: Any, *, record_id: str) -> tuple[tuple[str, ...], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise OntologyLoadError(
            f"aliases for {record_id} must be a list",
            category=LOAD_SCHEMA_INVALID,
            component=record_id,
        )
    out: list[tuple[str, ...]] = []
    for item in raw:
        if isinstance(item, str):
            toks = alias_to_tokens(item)
        elif isinstance(item, list) and all(isinstance(t, str) for t in item):
            toks = alias_to_tokens(" ".join(item))
        else:
            raise OntologyLoadError(
                f"alias entry for {record_id} must be string or string list",
                category=LOAD_SCHEMA_INVALID,
                component=record_id,
            )
        if toks:
            out.append(toks)
    return tuple(out)


def _enum(cls: type, value: str, *, field: str, component: str | None = None) -> Any:
    try:
        return cls(value)
    except ValueError as exc:
        raise OntologyLoadError(
            f"invalid {field}: {value!r}",
            category=LOAD_SCHEMA_INVALID,
            component=component or field,
        ) from exc


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise OntologyLoadError(
            f"missing ontology file: {_component_name(path)}",
            category=LOAD_MISSING_FILE,
            component=_component_name(path),
        )
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise OntologyLoadError(
            f"invalid JSON in {_component_name(path)}",
            category=LOAD_MALFORMED_JSON,
            component=_component_name(path),
        ) from exc
    except OSError as exc:
        raise OntologyLoadError(
            f"unreadable ontology file: {_component_name(path)}",
            category=LOAD_MISSING_FILE,
            component=_component_name(path),
        ) from exc


def _load_hazard_classes(path: Path) -> dict[str, HazardClassRecord]:
    component = path.name
    rows = _require_list(_load_json(path), component)
    out: dict[str, HazardClassRecord] = {}
    for row in rows:
        d = _require_dict(row, component)
        try:
            cid = str(d["id"])
            family_raw = str(d["family"])
            sev_raw = str(d["default_severity"])
        except KeyError as exc:
            raise OntologyLoadError(
                f"hazard_class missing required field {exc.args[0]!r}",
                category=LOAD_SCHEMA_INVALID,
                component=component,
            ) from exc
        if cid in out:
            raise OntologyLoadError(
                f"duplicate hazard_class id {cid!r}",
                category=LOAD_DUPLICATE_ID,
                component=component,
            )
        out[cid] = HazardClassRecord(
            id=cid,
            label=str(d.get("label", cid)),
            family=_enum(HazardFamily, family_raw, field="family", component=component),
            default_severity=_enum(
                Severity, sev_raw, field="default_severity", component=component
            ),
            block_on_procedural_transform=bool(d.get("block_on_procedural_transform", True)),
            aliases=_parse_aliases(d.get("aliases"), record_id=cid),
            notes=str(d.get("notes", "")),
        )
    return out


def _load_substances(path: Path) -> dict[str, SubstanceRecord]:
    component = path.name
    rows = _require_list(_load_json(path), component)
    out: dict[str, SubstanceRecord] = {}
    for row in rows:
        d = _require_dict(row, component)
        try:
            sid = str(d["id"])
        except KeyError as exc:
            raise OntologyLoadError(
                f"substance missing required field {exc.args[0]!r}",
                category=LOAD_SCHEMA_INVALID,
                component=component,
            ) from exc
        if sid in out:
            raise OntologyLoadError(
                f"duplicate substance id {sid!r}",
                category=LOAD_DUPLICATE_ID,
                component=component,
            )
        out[sid] = SubstanceRecord(
            id=sid,
            canonical_name=str(d.get("canonical_name", sid)),
            aliases=_parse_aliases(d.get("aliases"), record_id=sid),
            hazard_class_ids=tuple(str(x) for x in d.get("hazard_class_ids", [])),
            source_organism_ids=tuple(str(x) for x in d.get("source_organism_ids", [])),
            tags=tuple(str(x) for x in d.get("tags", [])),
        )
    return out


def _load_organisms(path: Path) -> dict[str, OrganismRecord]:
    component = path.name
    rows = _require_list(_load_json(path), component)
    out: dict[str, OrganismRecord] = {}
    for row in rows:
        d = _require_dict(row, component)
        try:
            oid = str(d["id"])
        except KeyError as exc:
            raise OntologyLoadError(
                f"organism missing required field {exc.args[0]!r}",
                category=LOAD_SCHEMA_INVALID,
                component=component,
            ) from exc
        if oid in out:
            raise OntologyLoadError(
                f"duplicate organism id {oid!r}",
                category=LOAD_DUPLICATE_ID,
                component=component,
            )
        out[oid] = OrganismRecord(
            id=oid,
            aliases=_parse_aliases(d.get("aliases"), record_id=oid),
            associated_substance_ids=tuple(
                str(x) for x in d.get("associated_substance_ids", [])
            ),
        )
    return out


def _load_processes(path: Path) -> dict[str, ProcessRecord]:
    component = path.name
    rows = _require_list(_load_json(path), component)
    out: dict[str, ProcessRecord] = {}
    for row in rows:
        d = _require_dict(row, component)
        try:
            pid = str(d["id"])
            family_raw = str(d["family"])
        except KeyError as exc:
            raise OntologyLoadError(
                f"process missing required field {exc.args[0]!r}",
                category=LOAD_SCHEMA_INVALID,
                component=component,
            ) from exc
        if pid in out:
            raise OntologyLoadError(
                f"duplicate process id {pid!r}",
                category=LOAD_DUPLICATE_ID,
                component=component,
            )
        outcomes = tuple(
            _enum(OutcomeKind, str(x), field="implies_outcomes", component=component)
            for x in d.get("implies_outcomes", [])
        )
        out[pid] = ProcessRecord(
            id=pid,
            aliases=_parse_aliases(d.get("aliases"), record_id=pid),
            family=_enum(ProcessFamily, family_raw, field="family", component=component),
            implies_outcomes=outcomes,
        )
    return out


def _load_actions(path: Path) -> dict[str, ActionRecord]:
    component = path.name
    rows = _require_list(_load_json(path), component)
    out: dict[str, ActionRecord] = {}
    for row in rows:
        d = _require_dict(row, component)
        try:
            aid = str(d["id"])
            proc_raw = str(d["procedurality"])
            outcome_raw = str(d["outcome_prior"])
        except KeyError as exc:
            raise OntologyLoadError(
                f"action missing required field {exc.args[0]!r}",
                category=LOAD_SCHEMA_INVALID,
                component=component,
            ) from exc
        if aid in out:
            raise OntologyLoadError(
                f"duplicate action id {aid!r}",
                category=LOAD_DUPLICATE_ID,
                component=component,
            )
        out[aid] = ActionRecord(
            id=aid,
            aliases=_parse_aliases(d.get("aliases"), record_id=aid),
            procedurality=_enum(
                Procedurality, proc_raw, field="procedurality", component=component
            ),
            outcome_prior=_enum(
                OutcomeKind, outcome_raw, field="outcome_prior", component=component
            ),
        )
    return out


def _load_framings(path: Path) -> dict[str, FramingRecord]:
    component = path.name
    rows = _require_list(_load_json(path), component)
    out: dict[str, FramingRecord] = {}
    for row in rows:
        d = _require_dict(row, component)
        try:
            fid = str(d["id"])
            framing_raw = str(d["framing"])
        except KeyError as exc:
            raise OntologyLoadError(
                f"framing missing required field {exc.args[0]!r}",
                category=LOAD_SCHEMA_INVALID,
                component=component,
            ) from exc
        if fid in out:
            raise OntologyLoadError(
                f"duplicate framing id {fid!r}",
                category=LOAD_DUPLICATE_ID,
                component=component,
            )
        out[fid] = FramingRecord(
            id=fid,
            aliases=_parse_aliases(d.get("aliases"), record_id=fid),
            framing=_enum(FramingKind, framing_raw, field="framing", component=component),
        )
    return out


def _load_decision_matrix(path: Path) -> tuple[DecisionMatrixRow, ...]:
    component = path.name
    rows = _require_list(_load_json(path), component)
    out: list[DecisionMatrixRow] = []
    for row in rows:
        d = _require_dict(row, component)
        try:
            out.append(
                DecisionMatrixRow(
                    framing=_enum(
                        FramingKind, str(d["framing"]), field="framing", component=component
                    ),
                    procedurality=_enum(
                        Procedurality,
                        str(d["procedurality"]),
                        field="procedurality",
                        component=component,
                    ),
                    process_family=_enum(
                        ProcessFamily,
                        str(d["process_family"]),
                        field="process_family",
                        component=component,
                    ),
                    severity=_enum(
                        Severity, str(d["severity"]), field="severity", component=component
                    ),
                    decision=_enum(
                        HazardDecision,
                        str(d["decision"]),
                        field="decision",
                        component=component,
                    ),
                )
            )
        except KeyError as exc:
            raise OntologyLoadError(
                f"decision_matrix row missing required field {exc.args[0]!r}",
                category=LOAD_SCHEMA_INVALID,
                component=component,
            ) from exc
    return tuple(out)


def _validate_refs(ontology: HazardOntology) -> None:
    for sid, sub in ontology.substances.items():
        for cid in sub.hazard_class_ids:
            if cid not in ontology.hazard_classes:
                raise OntologyLoadError(
                    f"substance {sid} references unknown hazard_class {cid}",
                    category=LOAD_REF_INVALID,
                    component="substances.json",
                )
        for oid in sub.source_organism_ids:
            if oid not in ontology.organisms:
                raise OntologyLoadError(
                    f"substance {sid} references unknown organism {oid}",
                    category=LOAD_REF_INVALID,
                    component="substances.json",
                )
    for oid, org in ontology.organisms.items():
        for sid in org.associated_substance_ids:
            if sid not in ontology.substances:
                raise OntologyLoadError(
                    f"organism {oid} references unknown substance {sid}",
                    category=LOAD_REF_INVALID,
                    component="organisms.json",
                )


def _alias_key(tokens: tuple[str, ...]) -> str:
    return " ".join(tokens)


def _owner_key(entity_type: str, record_id: str) -> tuple[str, str]:
    return (entity_type, record_id)


def _collect_alias_owners(
    ontology: HazardOntology,
) -> dict[str, list[tuple[str, str]]]:
    """Map normalized alias surface → owning (entity_type, record_id) pairs."""
    owners: dict[str, list[tuple[str, str]]] = {}

    def _add(entity_type: str, record_id: str, aliases: tuple[tuple[str, ...], ...]) -> None:
        for alias in aliases:
            if not alias:
                continue
            key = _alias_key(alias)
            owners.setdefault(key, []).append(_owner_key(entity_type, record_id))

    for rec in ontology.hazard_classes.values():
        aliases = rec.aliases
        if not aliases and rec.label:
            aliases = (alias_to_tokens(rec.label),)
        _add(ENTITY_TYPE_CLASS, rec.id, aliases)
    for rec in ontology.substances.values():
        _add(ENTITY_TYPE_SUBSTANCE, rec.id, rec.aliases)
    for rec in ontology.organisms.values():
        _add(ENTITY_TYPE_ORGANISM, rec.id, rec.aliases)
    for rec in ontology.processes.values():
        _add(ENTITY_TYPE_PROCESS, rec.id, rec.aliases)
    for rec in ontology.actions.values():
        _add(ENTITY_TYPE_ACTION, rec.id, rec.aliases)
    for rec in ontology.framings.values():
        _add(ENTITY_TYPE_FRAMING, rec.id, rec.aliases)
    return owners


def _parse_permitted_shared_aliases(
    raw: Any,
) -> dict[str, frozenset[tuple[str, str]]]:
    """alias_key → frozenset of declared owners for that shared alias."""
    if raw is None:
        return {}
    if not isinstance(raw, list):
        raise OntologyLoadError(
            "permitted_shared_aliases must be a JSON array",
            category=LOAD_SCHEMA_INVALID,
            component="ontology.manifest.json",
        )
    declared: dict[str, frozenset[tuple[str, str]]] = {}
    for i, entry in enumerate(raw):
        d = _require_dict(entry, "ontology.manifest.json")
        aliases_raw = d.get("aliases")
        owners_raw = d.get("owners")
        if not isinstance(aliases_raw, list) or not aliases_raw:
            raise OntologyLoadError(
                f"permitted_shared_aliases[{i}].aliases must be a non-empty list",
                category=LOAD_SCHEMA_INVALID,
                component="ontology.manifest.json",
            )
        if not isinstance(owners_raw, list) or len(owners_raw) < 2:
            raise OntologyLoadError(
                f"permitted_shared_aliases[{i}].owners must list at least two owners",
                category=LOAD_SCHEMA_INVALID,
                component="ontology.manifest.json",
            )
        owner_set: set[tuple[str, str]] = set()
        for ow in owners_raw:
            od = _require_dict(ow, "ontology.manifest.json")
            et = str(od.get("entity_type", ""))
            rid = str(od.get("id", ""))
            if et not in _VALID_ENTITY_TYPES or not rid:
                raise OntologyLoadError(
                    f"permitted_shared_aliases[{i}] has invalid owner "
                    f"entity_type={et!r} id={rid!r}",
                    category=LOAD_SCHEMA_INVALID,
                    component="ontology.manifest.json",
                )
            owner_set.add(_owner_key(et, rid))
        owners_frozen = frozenset(owner_set)
        for a in aliases_raw:
            if not isinstance(a, str):
                raise OntologyLoadError(
                    f"permitted_shared_aliases[{i}] alias must be a string",
                    category=LOAD_SCHEMA_INVALID,
                    component="ontology.manifest.json",
                )
            toks = alias_to_tokens(a)
            if not toks:
                continue
            key = _alias_key(toks)
            if key in declared and declared[key] != owners_frozen:
                raise OntologyLoadError(
                    f"permitted_shared_aliases declares conflicting owners for {key!r}",
                    category=LOAD_SCHEMA_INVALID,
                    component="ontology.manifest.json",
                )
            declared[key] = owners_frozen
    return declared


def _parse_permitted_within_type_duplicates(
    raw: Any,
) -> dict[str, frozenset[tuple[str, str]]]:
    """alias_key → frozenset of same-type (entity_type, id) owners allowed to share it."""
    if raw is None:
        return {}
    if not isinstance(raw, list):
        raise OntologyLoadError(
            "permitted_within_type_alias_duplicates must be a JSON array",
            category=LOAD_SCHEMA_INVALID,
            component="ontology.manifest.json",
        )
    declared: dict[str, frozenset[tuple[str, str]]] = {}
    for i, entry in enumerate(raw):
        d = _require_dict(entry, "ontology.manifest.json")
        et = str(d.get("entity_type", ""))
        aliases_raw = d.get("aliases")
        ids_raw = d.get("ids")
        if et not in _VALID_ENTITY_TYPES:
            raise OntologyLoadError(
                f"permitted_within_type_alias_duplicates[{i}] invalid entity_type",
                category=LOAD_SCHEMA_INVALID,
                component="ontology.manifest.json",
            )
        if not isinstance(aliases_raw, list) or not aliases_raw:
            raise OntologyLoadError(
                f"permitted_within_type_alias_duplicates[{i}].aliases must be non-empty",
                category=LOAD_SCHEMA_INVALID,
                component="ontology.manifest.json",
            )
        if not isinstance(ids_raw, list) or len(ids_raw) < 2:
            raise OntologyLoadError(
                f"permitted_within_type_alias_duplicates[{i}].ids must list >=2 ids",
                category=LOAD_SCHEMA_INVALID,
                component="ontology.manifest.json",
            )
        owners = frozenset(_owner_key(et, str(x)) for x in ids_raw)
        for a in aliases_raw:
            toks = alias_to_tokens(str(a))
            if not toks:
                continue
            declared[_alias_key(toks)] = owners
    return declared


def _validate_aliases(
    ontology: HazardOntology,
    *,
    permitted_shared: dict[str, frozenset[tuple[str, str]]],
    permitted_within: dict[str, frozenset[tuple[str, str]]],
) -> None:
    """Fail closed on undeclared within-type or cross-entity alias collisions."""
    owners_by_alias = _collect_alias_owners(ontology)
    for alias_key, owners in sorted(owners_by_alias.items()):
        uniq = sorted(set(owners))
        if len(uniq) <= 1:
            continue
        types = {et for et, _ in uniq}
        owner_set = frozenset(uniq)

        # Within-type collision: same entity_type, different ids.
        by_type: dict[str, list[str]] = {}
        for et, rid in uniq:
            by_type.setdefault(et, []).append(rid)
        for et, ids in by_type.items():
            distinct = sorted(set(ids))
            if len(distinct) < 2:
                continue
            within_owners = frozenset(_owner_key(et, i) for i in distinct)
            allowed = permitted_within.get(alias_key)
            if allowed != within_owners:
                raise OntologyLoadError(
                    f"undeclared within-type alias collision for {alias_key!r} "
                    f"in {et}: {distinct}",
                    category=LOAD_ALIAS_COLLISION,
                    component=et,
                )

        # Cross-entity collision: multiple entity types claim the alias.
        if len(types) < 2:
            continue
        allowed_shared = permitted_shared.get(alias_key)
        if allowed_shared != owner_set:
            raise OntologyLoadError(
                f"undeclared cross-entity alias collision for {alias_key!r}: "
                f"{sorted(owner_set)}",
                category=LOAD_ALIAS_COLLISION,
                component="alias",
            )


def load_hazard_ontology(data_dir: Path | None = None) -> HazardOntology:
    root = Path(data_dir) if data_dir is not None else default_data_dir()
    if not root.is_dir():
        raise OntologyLoadError(
            "hazard data directory not found",
            category=LOAD_MISSING_DIRECTORY,
            component="hazard",
        )

    manifest_path = root / "ontology.manifest.json"
    manifest = _require_dict(_load_json(manifest_path), "ontology.manifest.json")
    schema_version = str(manifest.get("schema_version", "1.0"))
    ontology_version = str(manifest.get("ontology_version", "0.0.0"))
    unknown_ask = bool(manifest.get("unknown_chem_transform_ask", False))
    permitted_shared = _parse_permitted_shared_aliases(
        manifest.get("permitted_shared_aliases")
    )
    permitted_within = _parse_permitted_within_type_duplicates(
        manifest.get("permitted_within_type_alias_duplicates")
    )

    ontology = HazardOntology(
        schema_version=schema_version,
        ontology_version=ontology_version,
        hazard_classes=_load_hazard_classes(root / "hazard_classes.json"),
        substances=_load_substances(root / "substances.json"),
        organisms=_load_organisms(root / "organisms.json"),
        processes=_load_processes(root / "processes.json"),
        actions=_load_actions(root / "actions.json"),
        framings=_load_framings(root / "framings.json"),
        decision_matrix=_load_decision_matrix(root / "decision_matrix.json"),
        unknown_chem_transform_ask=unknown_ask,
    )
    _validate_refs(ontology)
    _validate_aliases(
        ontology,
        permitted_shared=permitted_shared,
        permitted_within=permitted_within,
    )
    return ontology


@lru_cache(maxsize=4)
def get_bundled_ontology() -> HazardOntology:
    return load_hazard_ontology(default_data_dir())


def clear_ontology_cache() -> None:
    get_bundled_ontology.cache_clear()

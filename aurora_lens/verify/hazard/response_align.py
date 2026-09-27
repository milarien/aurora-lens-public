"""Response-side alignment helpers for the hazard ontology."""

from __future__ import annotations

from aurora_lens.verify.hazard.ontology_loader import get_bundled_ontology

# Canonical pre-LLM substance ids that response-side scanners should not contradict.
# Drugs remain on checker._ILLEGAL_DRUG_RE; these are CWA / toxin ontology entries.
ONTOLOGY_BACKED_HAZARD_SUBSTANCE_IDS: frozenset[str] = frozenset(
    {
        "oleandrin",
        "chlorine_gas",
        "phosgene",
        "mustard_gas",
        "hydrogen_cyanide",
        "ricin",
        "sarin",
        "tabun",
        "vx",
        "cyanogen_bromide",
        "generic_bioweapon",
    }
)

# Process lemmas expected in checker synth-verb defense-in-depth.
ONTOLOGY_BACKED_TRANSFORM_LEMMAS: frozenset[str] = frozenset(
    {
        "synthesize",
        "synthesise",
        "manufacture",
        "produce",
        "make",
        "extract",
        "distil",
        "distill",
        "concentrate",
        "isolate",
        "purify",
        "prepare",
    }
)


def export_substance_ids_for_align() -> frozenset[str]:
    return frozenset(get_bundled_ontology().substances.keys())


def ontology_alias_surfaces() -> frozenset[str]:
    onto = get_bundled_ontology()
    out: set[str] = set()
    for rec in onto.substances.values():
        out.add(rec.canonical_name.lower())
        for alias in rec.aliases:
            out.add(" ".join(alias))
    return frozenset(out)


def assert_ontology_export_covers_seed_ids() -> None:
    """Raise AssertionError if bundled ontology drifts from seed sync set."""
    exported = export_substance_ids_for_align()
    missing = ONTOLOGY_BACKED_HAZARD_SUBSTANCE_IDS - exported
    if missing:
        raise AssertionError(f"ontology missing seeded substance ids: {sorted(missing)}")

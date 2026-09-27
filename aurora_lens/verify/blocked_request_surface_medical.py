"""Regex-free surfaces: pediatric/adult dosing, fictional-wrapper med, generic medical gates."""

from __future__ import annotations

from aurora_lens.verify.blocked_request_normalize import normalise_request_text

_MEDICATION_NAMES: frozenset[str] = frozenset(
    {
        "amoxicillin",
        "penicillin",
        "ibuprofen",
        "paracetamol",
        "acetaminophen",
        "azithromycin",
        "augmentin",
        "cephalexin",
        "metformin",
        "lisinopril",
        "aspirin",
        "naproxen",
        "trimethoprim",
        "ciprofloxacin",
        "doxycycline",
        "clarithromycin",
    }
)


def surface_req_mgkg(norm: str) -> bool:
    return "mg/kg" in norm.replace(" ", "") or "mcg/kg" in norm.replace(" ", "")


def surface_req_mgkg_loose(norm: str) -> bool:
    """Detect mg/kg style including spaced variants."""
    w = norm.replace(" ", "")
    return "mg/kg" in w or "mcg/kg" in w


def surface_pediatric_indicator(tokens: tuple[str, ...]) -> bool:
    """Token / bigram cues for pediatric framing (paired with dosing rules in policy)."""
    tok_f = frozenset(tokens)
    if not frozenset(
        {
            "child",
            "children",
            "infant",
            "infants",
            "toddler",
            "neonate",
            "neonatal",
            "adolescent",
            "adolescents",
            "kid",
            "kids",
            "baby",
            "babies",
            "teenager",
            "teenage",
            "teen",
            "pediatric",
            "paediatric",
        }
    ).isdisjoint(tok_f):
        return True
    return _contains_bigram(tokens, ("year", "old")) or _contains_bigram(tokens, ("years", "old"))


def surface_dosing_or_titration_intent(norm: str, tokens: frozenset[str]) -> bool:
    w = f" {norm} "
    if any(
        x in tokens
        for x in (
            "dose",
            "dosing",
            "dosage",
            "milligrams",
            "milligram",
            "mg",
            "ml",
            "mcg",
        )
    ):
        return True
    if any(t.startswith("titrat") for t in tokens):
        return True
    if " how much mg " in w or " how many mg " in w or " how much ml " in w:
        return True
    if " how many mg " in w or " how many ml " in w or " how many mcg " in w:
        return True
    if " what is the dose " in w or " what dose " in w:
        return True
    return False


def surface_medication_named(tokens: frozenset[str]) -> bool:
    return not _MEDICATION_NAMES.isdisjoint(tokens)


def surface_personal_medical_frame(tokens: tuple[str, ...]) -> bool:
    modals = frozenset({"should", "could", "would", "may"})
    n = len(tokens)
    for i in range(n - 1):
        if tokens[i : i + 2] == ("i", "take"):
            prev = tokens[i - 1] if i > 0 else None
            if prev not in modals:
                return True
    for i in range(n - 1):
        if tokens[i : i + 2] == ("i", "weigh"):
            return True
    for i in range(n - 2):
        if tokens[i : i + 3] == ("i", "am", "on"):
            return True
        if tokens[i : i + 3] == ("i", "should", "take"):
            return True
        if tokens[i : i + 3] == ("i", "should", "use"):
            return True
        if tokens[i : i + 3] == ("my", "blood", "sugar"):
            return True
    for i in range(n - 1):
        if tokens[i] == "my" and tokens[i + 1] in ("glucose", "medication", "prescription", "dose"):
            return True
    for i in range(n - 2):
        if tokens[i : i + 3] == ("for", "my", "blood") and i + 3 < n and tokens[i + 3] == "sugar":
            return True
    return False


def surface_fiction_wrapper(tokens: frozenset[str]) -> bool:
    return not frozenset(
        {"write", "draft", "create", "compose", "fictional", "story", "dialogue", "scene", "character"}
    ).isdisjoint(tokens)


def surface_real_world_med_anchor(norm: str, tokens: tuple[str, ...]) -> bool:
    w = f" {norm} "
    if " for tonight " in w:
        return True
    if " my real " in w:
        return True
    pairs = (
        ("my", "chest"),
        ("my", "pain"),
        ("my", "symptom"),
        ("my", "symptoms"),
        ("my", "condition"),
    )
    for i in range(len(tokens) - 2):
        if tokens[i : i + 3] == ("my", "real", "chest"):
            return True
        if tokens[i : i + 3] == ("my", "real", "pain"):
            return True
        if tokens[i : i + 3] == ("my", "real", "symptoms"):
            return True
        if tokens[i : i + 3] == ("my", "real", "symptom"):
            return True
        if tokens[i : i + 3] == ("my", "real", "condition"):
            return True
    for i in range(len(tokens) - 1):
        if tokens[i : i + 2] in pairs:
            return True
    return False


def surface_medical_decision_or_instruction(tokens: frozenset[str]) -> bool:
    return not frozenset(
        {
            "safe",
            "dosage",
            "dose",
            "take",
            "instruction",
            "instructions",
            "treatment",
            "medication",
            "insulin",
            "mg",
        }
    ).isdisjoint(tokens) or ("is" in tokens and "safe" in tokens)


def surface_medical_topic(tokens: frozenset[str]) -> bool:
    singles = frozenset(
        {
            "otitis",
            "infection",
            "symptom",
            "symptoms",
            "diagnose",
            "diagnosis",
            "treat",
            "treatment",
            "medication",
            "antibiotic",
            "dose",
            "dosage",
            "condition",
            "disease",
        }
    )
    if not singles.isdisjoint(tokens):
        return True
    return ("middle" in tokens and "ear" in tokens and "infection" in tokens) or (
        "ear" in tokens and "infection" in tokens
    )


def surface_medical_educational(norm: str) -> bool:
    w = f" {norm} "
    return (
        " what are the typical symptoms " in w
        or " what are typical symptoms " in w
        or " what are the common symptoms " in w
        or " what are common symptoms " in w
        or " what causes " in w
        or norm.startswith("what is ")
        or norm.startswith("what are ")
        or " explain the symptoms " in w
        or " explain symptoms " in w
        or " explain the causes " in w
        or " explain causes " in w
        or " explain the condition " in w
        or " explain the disease " in w
    )


def surface_first_person(tokens: frozenset[str]) -> bool:
    return not frozenset({"i", "my", "me"}).isdisjoint(tokens)


def surface_medical_personal_diagnosis(norm: str, tokens: tuple[str, ...]) -> bool:
    w = f" {norm} "
    return (
        " do i have " in w
        or " could i have " in w
        or " am i " in w
        or " is this " in w
    )


def surface_medical_personal_action(norm: str, tokens: tuple[str, ...]) -> bool:
    w = f" {norm} "
    return (
        " what should i take " in w
        or " should i take " in w
        or " what should i do " in w
        or " what do i take " in w
        or " which medication should i take " in w
        or " do i need antibiotic " in w
        or " do i need antibiotics " in w
    )


def surface_medical_implicit_action(norm: str) -> bool:
    return " what should i take for it " in f" {norm} "


def surface_definitive_medical_triage(norm: str, tokens: tuple[str, ...]) -> bool:
    w = f" {norm} "
    acute = (
        ("heart", "attack"),
        ("myocardial", "infarction"),
        ("cardiac", "arrest"),
        ("cardiac", "event"),
        ("pulmonary", "embolism"),
        ("blood", "clot"),
        ("heart", "failure"),
    )
    tok_set = frozenset(tokens)
    acute_hit = any(_contains_bigram(tokens, p) for p in acute) or any(
        x in tok_set
        for x in (
            "stroke",
            "appendicitis",
            "sepsis",
            "anaphylaxis",
            "meningitis",
            "cardiac",
        )
    )
    if not acute_hit:
        return False
    verb_hit = any(
        t in tok_set
        for t in (
            "determine",
            "decide",
            "diagnose",
            "identify",
            "assess",
            "confirm",
        )
    ) or any(p in w for p in (" tell me ", " work out ", " figure out "))
    frame_hit = (
        " definitively " in w
        or " is this a " in w
        or " is this an " in w
        or _triage_whether_if_frame(norm, tokens)
    )
    return verb_hit or frame_hit


def _contains_bigram(tokens: tuple[str, ...], pair: tuple[str, str]) -> bool:
    for i in range(len(tokens) - 1):
        if tokens[i : i + 2] == pair:
            return True
    return False


def _triage_whether_if_frame(norm: str, tokens: tuple[str, ...]) -> bool:
    """Approximate MED_TRIAGE_FRAME_RE without regex-backtracking."""
    w = f" {norm} "
    if " whether this " in w or " whether it " in w or " whether these " in w:
        if any(v in w for v in (" is ", " are ", " could be ")):
            return True
    if " if this " in w or " if it " in w or " if these " in w:
        if any(v in w for v in (" is ", " are ", " could be ")):
            return True
    # scan tokens for whether/if followed within window by is|are|could,be
    key_ws = {"whether", "if"}
    lim = len(tokens)
    for i, t in enumerate(tokens):
        if t not in key_ws:
            continue
        window = tokens[i : min(i + 14, lim)]
        if "is" in window or "are" in window:
            return True
        if "could" in window and "be" in window:
            return True
    return False


def tokens_and_norm(user_input: str) -> tuple[tuple[str, ...], str]:
    norm = normalise_request_text(user_input)
    return tuple(norm.split()), norm

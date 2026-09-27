"""Epistemic Uncertainty Gate — pre-LLM session gate.

Blocks decision-seeking turns when the session contains open epistemic
uncertainties declared via ``evidence_state.open_epistemic_uncertainties``
in the request body.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from aurora_lens.pef.uncertainty_analysis import (
    PHASE2_EXTERNAL_METADATA_OPTIONAL_FIELDS,
    SCOPE_COMPARISON_MODES,
    SCOPE_DIMENSION_ALLOWED,
    normalize_bridge_status,
    normalize_consequence_grade,
    normalize_inferential_bridge_mode,
    normalize_strength_value,
    normalize_threshold_comparison_mode,
    normalize_scope_value_dict,
    parse_premises,
    parse_scope_dimensions,
)

# ── Data model ───────────────────────────────────────────────────────


@dataclass
class OpenEpistemicUncertainty:
    id: str
    kind: str
    description: str
    bears_on: tuple[str, ...]
    status: str = "open"


@dataclass
class EpistemicUncertaintyGateResult:
    blocks: bool = False
    response_text: str = ""
    matched_uncertainties: list[OpenEpistemicUncertainty] = field(default_factory=list)
    reason_code: str = ""
    reason_detail: str = ""


# ── Decision-seeking detection ────────────────────────────────────────

# Verbs and decision-request nouns that signal the user is asking Aurora to
# decide/recommend.  Only fires inside a question or alongside an explicit
# request directive.
_DECISION_VERB_RE = re.compile(
    r"\b(choose|choosing|chose|chosen"
    r"|select|selects|selected|selecting"
    r"|decide|decides|decided|deciding"
    r"|recommend|recommends|recommended|recommending|recommendation|recommendations"
    r"|advise|advises|advised|advising"
    r"|determine|determines|determined|determining"
    r"|should"          # interrogative: "Should X order Y?" / "Should we proceed?"
    r")\b",
    re.IGNORECASE,
)

# Phrases that are unambiguously a direct question to the system.
_DECISION_PHRASE_RE = re.compile(
    r"which\s+option"
    r"|which\s+of\s+the"
    r"|what\s+should"
    r"|what\s+action\s+should"
    r"|which\s+choice"
    r"|what\s+option"
    r"|which\s+do\s+you"
    r"|what\s+would\s+you"
    r"|should\s+\w+(?:\s+\w+)?\s+(?:order|proceed|shut|continue|escalate|activate|deploy|evacuate)",
    re.IGNORECASE,
)

# Explicit request directives — trigger regardless of question mark.
# Covers imperative forms, "given the above" framing, and recommendation nouns.
_EXPLICIT_REQUEST_RE = re.compile(
    r"(?:^|[.!]\s+)(?:please\s+)?(?:recommend|advise|decide|suggest)\b"
    r"|(?:^|[.!?]\s+)provide\s+(?:your\s+|a\s+)?recommendation"
    r"|given\s+the\s+above[,\s]"
    r"|given\s+this[,\s]"
    r"|tell\s+(?:me|us)\s+which",
    re.IGNORECASE,
)

_REFERENT_BEARING_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "that", "their", "had", "before", "was", "were", "is", "are",
    "in", "to", "and", "both", "parties", "hold", "holds", "informed", "work",
    "commenced", "for", "does", "not", "claim", "referent", "unresolved", "open",
})

_REFERENT_SCENARIO_NAMES: frozenset[str] = frozenset({"operator", "contractor"})


def is_decision_seeking(user_text: str) -> bool:
    """True when the turn is asking Aurora to make or recommend a decision.

    Requires one of:
    - A direct question (text contains "?") with a decision verb or phrase, OR
    - An explicit request directive ("recommend ...", "given the above ...", etc.)

    Scenario prose describing someone else's decision ("must choose", "has to decide")
    does not trigger — those lack both the question form and a directed request.
    """
    if not user_text:
        return False
    text_lower = user_text.lower()

    if "?" in text_lower:
        if _DECISION_PHRASE_RE.search(text_lower):
            return True
        if _DECISION_VERB_RE.search(text_lower):
            return True

    if _EXPLICIT_REQUEST_RE.search(text_lower):
        return True

    return False


def _referent_bearing_tokens(
    open_uncerts: list[OpenEpistemicUncertainty],
) -> frozenset[str]:
    tokens: set[str] = set()
    for uncertainty in open_uncerts:
        if uncertainty.kind != "identity_or_referent_unresolved":
            continue
        for source in (uncertainty.description, *uncertainty.bears_on):
            for token in re.findall(r"[a-z']+", source.lower()):
                if len(token) > 2 and token not in _REFERENT_BEARING_STOPWORDS:
                    tokens.add(token)
    return frozenset(tokens)


def _decision_bears_on_referent_uncertainty(
    user_text: str,
    open_uncerts: list[OpenEpistemicUncertainty],
) -> bool:
    """True when a decision-seeking turn depends on an open referent attribution."""
    text_lower = user_text.lower()
    if _DECISION_PHRASE_RE.search(text_lower):
        return True
    user_tokens = frozenset(re.findall(r"[a-z']+", text_lower))
    if user_tokens & _referent_bearing_tokens(open_uncerts):
        return True
    return any(
        re.search(rf"\b{re.escape(name)}\b", text_lower)
        for name in _REFERENT_SCENARIO_NAMES
    )


# ── Parsing ───────────────────────────────────────────────────────────


def parse_open_epistemic_uncertainties(raw: Any) -> list[dict]:
    """Parse from evidence_state raw value (a list of dicts).

    Validates that id, kind, and description are present; skips malformed entries.
    Returns the valid raw dicts (not converted to dataclasses — callers may pass
    directly back to the gate or store on PEF state).
    """
    if not isinstance(raw, list):
        return []
    result: list[dict] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        if not all(entry.get(k) for k in ("id", "kind", "description")):
            continue
        out = dict(entry)
        kind = str(out.get("kind") or "").strip()
        meta = out.get("meta")
        if isinstance(meta, dict):
            out_meta = _normalize_kind_metadata(kind, meta)
            if out_meta:
                out["meta"] = out_meta
            else:
                out.pop("meta", None)
        result.append(out)
    return result


def _normalize_kind_metadata(kind: str, meta: dict[str, Any]) -> dict[str, Any]:
    """Keep only known additive metadata fields with basic shape checks.

    Unknown fields are dropped silently to keep request parsing non-breaking.
    """
    allowed = PHASE2_EXTERNAL_METADATA_OPTIONAL_FIELDS.get(kind)
    if not allowed:
        return {}

    if kind == "scope_mismatch":
        return _normalize_scope_mismatch_metadata(meta, allowed)

    if kind == "threshold_not_met":
        return _normalize_threshold_not_met_metadata(meta, allowed)

    if kind == "inferential_gap":
        return _normalize_inferential_gap_metadata(meta, allowed)

    list_only_fields = {"missing_support", "trust_basis"}
    numeric_only_fields: set[str] = set()
    out: dict[str, Any] = {}
    for key in allowed:
        if key not in meta:
            continue
        val = meta[key]
        if key in list_only_fields:
            if isinstance(val, (list, tuple)):
                items = [str(x).strip() for x in val if str(x).strip()]
                if items:
                    out[key] = items
            continue
        if key in numeric_only_fields:
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                out[key] = float(val)
            continue
        if isinstance(val, str):
            if val.strip():
                out[key] = val.strip()
            continue
    return out


def _normalize_scope_mismatch_metadata(
    meta: dict[str, Any],
    allowed: frozenset[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}

    required = normalize_scope_value_dict(meta.get("required_scope"))
    if required and "required_scope" in allowed:
        out["required_scope"] = required

    supported_raw = meta.get("supported_scope")
    if supported_raw is None:
        supported_raw = meta.get("evidence_scope")
    supported = normalize_scope_value_dict(supported_raw)
    if supported:
        if "supported_scope" in allowed:
            out["supported_scope"] = supported
        elif "evidence_scope" in allowed:
            out["evidence_scope"] = supported

    dims = parse_scope_dimensions(meta.get("scope_dimensions"))
    if dims and "scope_dimensions" in allowed:
        out["scope_dimensions"] = dims

    mode = str(meta.get("comparison_mode") or "").strip().lower()
    if mode in SCOPE_COMPARISON_MODES and "comparison_mode" in allowed:
        out["comparison_mode"] = mode

    failed_raw = meta.get("failed_dimensions")
    if isinstance(failed_raw, (list, tuple)) and "failed_dimensions" in allowed:
        failed: list[str] = []
        for item in failed_raw:
            dim = str(item).strip()
            if dim in SCOPE_DIMENSION_ALLOWED and dim not in failed:
                failed.append(dim)
        if failed:
            out["failed_dimensions"] = failed

    if "policy_ref" in allowed:
        policy_ref = meta.get("policy_ref")
        if isinstance(policy_ref, str) and policy_ref.strip():
            out["policy_ref"] = policy_ref.strip()

    return out


def _normalize_threshold_not_met_metadata(
    meta: dict[str, Any],
    allowed: frozenset[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}

    if "required_strength" in allowed:
        required = normalize_strength_value(meta.get("required_strength"))
        if required is not None:
            out["required_strength"] = required

    if "observed_strength" in allowed:
        observed = normalize_strength_value(meta.get("observed_strength"))
        if observed is not None:
            out["observed_strength"] = observed

    if "consequence_grade" in allowed:
        grade = normalize_consequence_grade(meta.get("consequence_grade"))
        if grade is not None:
            out["consequence_grade"] = grade

    if "threshold_ref" in allowed:
        threshold_ref = meta.get("threshold_ref")
        if isinstance(threshold_ref, str) and threshold_ref.strip():
            out["threshold_ref"] = threshold_ref.strip()

    if "comparison_mode" in allowed:
        mode = normalize_threshold_comparison_mode(meta.get("comparison_mode"))
        if mode is not None:
            out["comparison_mode"] = mode

    if "policy_ref" in allowed:
        policy_ref = meta.get("policy_ref")
        if isinstance(policy_ref, str) and policy_ref.strip():
            out["policy_ref"] = policy_ref.strip()

    return out


def _normalize_inferential_gap_metadata(
    meta: dict[str, Any],
    allowed: frozenset[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}

    premises_raw = meta.get("premises")
    if premises_raw is None:
        premises_raw = meta.get("missing_support")
    premises = parse_premises(premises_raw)
    if premises:
        if "premises" in allowed:
            out["premises"] = premises
        elif "missing_support" in allowed:
            out["missing_support"] = premises

    if "conclusion" in allowed:
        conclusion = meta.get("conclusion")
        if isinstance(conclusion, str) and conclusion.strip():
            out["conclusion"] = conclusion.strip()

    if "bridge_ref" in allowed:
        bridge_ref = meta.get("bridge_ref")
        if isinstance(bridge_ref, str) and bridge_ref.strip():
            out["bridge_ref"] = bridge_ref.strip()

    if "bridge_status" in allowed:
        status = normalize_bridge_status(meta.get("bridge_status"))
        if status is not None:
            out["bridge_status"] = status

    if "bridge_mode" in allowed:
        mode = normalize_inferential_bridge_mode(meta.get("bridge_mode"))
        if mode is not None:
            out["bridge_mode"] = mode

    if "policy_ref" in allowed:
        policy_ref = meta.get("policy_ref")
        if isinstance(policy_ref, str) and policy_ref.strip():
            out["policy_ref"] = policy_ref.strip()

    return out


# ── Internal helpers ──────────────────────────────────────────────────


def _open_uncertainties(uncertainties: list[dict]) -> list[OpenEpistemicUncertainty]:
    """Filter to status == 'open' and convert to typed objects."""
    result: list[OpenEpistemicUncertainty] = []
    for u in uncertainties:
        status = u.get("status", "open")
        if status != "open":
            continue
        bears_on_raw = u.get("bears_on", [])
        if isinstance(bears_on_raw, (list, tuple)):
            bears_on = tuple(str(x) for x in bears_on_raw)
        else:
            bears_on = ()
        result.append(
            OpenEpistemicUncertainty(
                id=str(u.get("id", "")),
                kind=str(u.get("kind", "")),
                description=str(u.get("description", "")),
                bears_on=bears_on,
                status=status,
            )
        )
    return result


def _build_escalation_response(open_uncerts: list[OpenEpistemicUncertainty]) -> str:
    """Build the structured escalation response text."""
    bullet_lines = "\n".join(
        f"  • {u.description}" for u in open_uncerts
    )
    return (
        "Decision blocked: the available evidence does not establish a justified recommendation.\n"
        "\n"
        "The scenario contains unresolved uncertainty:\n"
        f"{bullet_lines}\n"
        "\n"
        "A decision of this kind requires authority-bearing judgement under uncertainty.\n"
        "\n"
        "Aurora-Lens cannot determine which option should be chosen.\n"
        "\n"
        "Escalate to the designated human authority responsible for this decision.\n"
        "\n"
        "Admissible continuation:\n"
        "Provide a structured analysis of known evidence, uncertainties, assumptions, "
        "risks, and trade-offs for authority review."
    )


# ── Public gate function ──────────────────────────────────────────────


def evaluate_epistemic_uncertainty_gate(
    uncertainties: list[dict],
    user_text: str,
) -> EpistemicUncertaintyGateResult | None:
    """Evaluate the epistemic uncertainty gate.

    Returns None if there are no open uncertainties OR the turn is not
    decision-seeking.  Otherwise returns a blocking EpistemicUncertaintyGateResult.
    """
    open_uncerts = _open_uncertainties(uncertainties)
    if not open_uncerts:
        return None
    if not is_decision_seeking(user_text):
        return None

    referent_only = all(
        u.kind == "identity_or_referent_unresolved" for u in open_uncerts
    )
    if referent_only and not _decision_bears_on_referent_uncertainty(
        user_text,
        open_uncerts,
    ):
        return None

    response_text = _build_escalation_response(open_uncerts)
    reason_detail = (
        "Open epistemic uncertainties preclude a justified recommendation; "
        "escalation to human authority required."
    )
    return EpistemicUncertaintyGateResult(
        blocks=True,
        response_text=response_text,
        matched_uncertainties=open_uncerts,
        reason_code="epistemic_uncertainty_gate",
        reason_detail=reason_detail,
    )

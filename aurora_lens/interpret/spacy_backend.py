"""spaCy-based extraction backend.

Deterministic, local NLP — independent of any LLM.
Implements ExtractionBackend using spaCy for entity/relationship extraction.

spaCy is imported lazily — this module is importable without spaCy installed.
Instantiating SpacyBackend without spaCy raises a clear runtime error.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, TYPE_CHECKING

logger = logging.getLogger(__name__)


def _finance_copula_diag_enabled() -> bool:
    """Env-gated diagnostics for live vs dev extractor path (finance metric copula)."""
    v = os.environ.get("AURORA_LENS_DEBUG_FINANCE_COPULA", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _log_finance_copula(event: str, **fields: Any) -> None:
    if not _finance_copula_diag_enabled():
        return
    parts = " ".join(f"{k}={fields[k]!r}" for k in sorted(fields))
    logger.info("finance_metric_copula %s %s", event, parts)

from aurora_lens.pef.span import Span
from aurora_lens.pef.state import (
    PEFState,
    CANONICAL_RELATIONS,
    CONSUMPTION_RELATIONS,
    canonicalize_relation,
)
from aurora_lens.interpret.schema import ComparativeAmbiguity, ExtractedClaim, ExtractionResult
from aurora_lens.interpret.base import ExtractionBackend
from aurora_lens.verify.numeric import parse_numeric
from aurora_lens.state_native_engine.lexical import item_key

if TYPE_CHECKING:
    import spacy
    from spacy.tokens import Doc, Token


# ── Span detection ───────────────────────────────────────────────────

_PAST_SIGNALS = {
    "was", "were", "had", "used to", "formerly", "previously",
    "once", "ago", "before", "back then",
}
_PRESENT_SIGNALS = {
    "is", "are", "has", "have", "now", "currently", "today",
}

# Patterns that strongly indicate past tense context
_PAST_PATTERNS = [
    re.compile(r"\bused\s+to\b", re.IGNORECASE),
    re.compile(r"\bback\s+then\b", re.IGNORECASE),
    re.compile(r"\b\w+\s+ago\b", re.IGNORECASE),
]


def detect_span(doc: Doc) -> Span:
    """Detect the dominant temporal span of a sentence.

    Uses verb tense morphology as primary signal, surface keywords as fallback.
    """
    past_score = 0
    present_score = 0

    for token in doc:
        if token.pos_ in ("VERB", "AUX"):
            morph = token.morph.get("Tense")
            if morph:
                if "Past" in morph:
                    past_score += 2
                elif "Pres" in morph:
                    present_score += 2

        lower = token.text.lower()
        if lower in _PAST_SIGNALS:
            past_score += 1
        elif lower in _PRESENT_SIGNALS:
            present_score += 1

    text = doc.text
    for pattern in _PAST_PATTERNS:
        if pattern.search(text):
            past_score += 3

    return Span.PAST if past_score > present_score else Span.PRESENT


# ── Pronoun detection ────────────────────────────────────────────────

_PRONOUNS = {"he", "she", "it", "they", "him", "her", "them", "his", "its", "their"}

# Possessive pronouns that can be ambiguous when multiple candidates exist
_POSSESSIVE_PRONOUNS = frozenset({"her", "his", "their", "its"})

# Subject- and object-position pronouns that can also be ambiguous
_SUBJECT_OBJECT_PRONOUNS: frozenset[str] = frozenset({
    "he", "him", "she", "her", "it",
    "they", "them", "that", "this", "these", "those",
})

# Determiners that mark definite NPs (presuppose a unique prior referent)
_DEFINITE_DETERMINERS: frozenset[str] = frozenset({"the", "that", "this", "those", "these"})

# Instructional meta: *roles* scoped to story/fiction world — not discourse ambiguity.
# Positive match only (regex-policy: bounded lexical shape). Does not exempt fiction broadly.
_LOCAL_ROLE_SCOPE_INSTRUCTION_IN_SENTENCE = re.compile(
    r"(?is)"
    r"(?:the|this|that|these|those)\s+roles?\s+(?:is|are)\s+"
    r"(?:local(?:ly)?|confined|limited)\s+to\s+"
    r"(?:the\s+)?"
    r"(?:story\s+world|fiction(?:al)?\s*world|the\s+narrative)\b"
    r"(?:\s+only)?",
)

# Same-sentence anaphora: do not suppress (e.g. *The roles they play are local…*).
_LOCAL_SCOPE_SENTENCE_BLOCKS_SUPPRESSION = re.compile(
    r"\b(she|he|they|her|him|them|their)\b",
    re.IGNORECASE,
)


def _suppress_definite_role_for_local_scope_instruction(
    det_token: "Token",
    head: "Token",
) -> bool:
    """True when *the roles* … is a locality instruction clause, not underdetermination.

    Used only from the definite-NP branch when the head lemma is *role* and the
    extractor already found 2+ structural candidates — suppress false
    ``UNRESOLVED_REFERENT`` for instructional phrases like *the roles are local
    to the story world only* (composition scope), not for *the roles are unclear*.
    """
    if head.lemma_.lower() != "role":
        return False
    sent_text = det_token.sent.text
    if _LOCAL_SCOPE_SENTENCE_BLOCKS_SUPPRESSION.search(sent_text):
        return False
    return _LOCAL_ROLE_SCOPE_INSTRUCTION_IN_SENTENCE.search(sent_text) is not None


def _verb_governing_possessed_np(token: Token) -> Token | None:
    """Return the finite verb of the clause containing the noun modified by this possessive."""
    if token.dep_ != "poss":
        return None
    head = token.head
    visited: set[int] = set()
    while head is not None and head.i not in visited:
        visited.add(head.i)
        if head.pos_ in ("VERB", "AUX"):
            return head
        if head.dep_ == "ROOT" or head.head == head:
            break
        head = head.head
    return None


def _possessive_locally_bound_by_sentence_subject(
    token: Token,
    doc: Doc,
    doc_persons: set[str],
) -> bool:
    """True when the possessive is forced by a single person in the sentence as clause subject.

    PEF may still hold other people from prior turns; they must not force ambiguity
    for ``Emma lost her key``-style binding when Emma is the only person in the
    sentence and is the ``nsubj`` of the verb governing the possessed noun.
    """
    if token.dep_ != "poss":
        return False
    # Multiple people in this sentence → classic ambiguity (e.g. Emma told Anna …).
    if len(doc_persons) != 1:
        return False
    v = _verb_governing_possessed_np(token)
    if v is None:
        return False
    subj = None
    for c in v.children:
        if c.dep_ in ("nsubj", "nsubjpass"):
            subj = c
            break
    if subj is None:
        return False
    subj_propns = {t.text for t in subj.subtree if t.pos_ == "PROPN"}
    if len(subj_propns) >= 2:
        return False
    only_name = next(iter(doc_persons))
    subj_text = " ".join(t.text for t in sorted(subj.subtree, key=lambda t: t.i))
    if only_name in subj_text:
        return True
    if subj_propns == {only_name}:
        return True
    return False


def _object_text_matches_noun_lemma(noun_lemma: str, text: str) -> bool:
    """True when *text* mentions the noun lemma as a word (English plural *-s*).

    Avoids substring traps like ``key`` inside ``monkey`` while still matching
    ``key`` / ``keys`` in object literals.
    """
    if not text or not noun_lemma:
        return False
    tl = text.lower()
    nl = noun_lemma.lower()
    return bool(re.search(r"\b" + re.escape(nl) + r"s?\b", tl))


def _noun_grounded_in_pef(noun_lemma: str, pef: PEFState) -> bool:
    """True if the noun concept already exists as an object_literal in committed PEF.

    When True, "the <noun>" refers to the already-established concept — same-turn
    claim subjects (who acted on it) are not additional referent candidates.
    """
    nl = noun_lemma.lower()
    for rel in pef.get_relationships_by_relation("HAS"):
        obj = str(rel.object_literal or "").strip()
        if _object_text_matches_noun_lemma(nl, obj):
            return True
    return False


def _definite_np_pef_referent_candidates(noun_lemma: str, pef: PEFState) -> set[str]:
    """Entity names structurally eligible as referents for a definite NP head (committed PEF).

    Only matches entity names — not relationship object literals. Multiple entities holding
    the same literal value ("book") refer to the same concept, not to distinguishable referents.
    """
    out: set[str] = set()
    nl = noun_lemma.lower()
    for ent in pef.entities.values():
        if nl in ent.name.lower():
            out.add(ent.name)
    return out


def _demonstrative_uniquely_resolved_in_doc(head_token: Any, doc: Any) -> bool:
    """Return True when exactly one uniquely named antecedent for *head_token*'s
    lemma exists in the document.

    Used only for near-demonstratives ("this"/"that") whose deixis presupposes a
    specific salient referent.  "Reactor 3" is the only capitalized instance of
    the lemma "reactor" → resolve.  "Reactor 3" and "Reactor 7" → do not resolve.

    Named-instance key includes immediately following nummod/compound children so
    "Reactor 3" and "Reactor 7" are distinguished as separate keys.
    """
    head_lemma = head_token.lemma_.lower()
    named_keys: set[str] = set()
    for tok in doc:
        if tok.i == head_token.i:
            continue
        if tok.lemma_.lower() != head_lemma:
            continue
        if not tok.text or not tok.text[0].isupper():
            continue
        parts = [tok.text]
        for child in tok.children:
            if child.dep_ in ("nummod", "compound") and child.i > tok.i:
                parts.append(child.text)
        named_keys.add(" ".join(parts))
    return len(named_keys) == 1


def _cataphoric_colon_reference_in_sent(head_token: Any) -> bool:
    """True when a colon later in the same sentence introduces *head_token*'s content.

    "Record this fact: Emma is a manager." — "this fact" refers forward to the
    text after the colon, not backward to an established world-state antecedent.
    This is self-contained (cataphoric), not deictic; it must not be treated as
    a reference-failure candidate the way "That transaction must remain off the
    books" (with no transaction anywhere in state) is.
    """
    for tok in head_token.sent:
        if tok.i > head_token.i and tok.text == ":":
            return True
    return False


def _discourse_role_nouns_from_pef(pef: PEFState) -> set[str]:
    """Role/common-noun entities already committed in PEF (cross-turn antecedents)."""
    return {
        ent.name for ent in pef.entities.values()
        if _is_discourse_role_noun_surface(ent.name)
    }


def _definite_np_pef_has_subject_candidates(noun_lemma: str, pef: PEFState) -> set[str]:
    """Role-noun subjects from committed HAS relationships whose object matches *noun_lemma*."""
    out: set[str] = set()
    nl = noun_lemma.lower()
    for rel in pef.get_relationships_by_relation("HAS"):
        if rel.negated:
            continue
        subj = pef.entities.get(rel.subject_id)
        if not subj or not _is_discourse_role_noun_surface(subj.name):
            continue
        if _object_text_matches_noun_lemma(nl, rel.object_literal or ""):
            out.add(subj.name)
    return out


def _definite_np_same_turn_claim_candidates(
    noun_lemma: str, claims: list[ExtractedClaim],
) -> set[str]:
    """Subjects from same-turn claims whose object mentions the noun head.

    Committed PEF does not yet include these assertions during ``extract()``;
    they must be merged so singular definites like *the key* are gated in one turn.
    """
    out: set[str] = set()
    nl = noun_lemma.lower()
    for c in claims:
        if c.negated:
            continue
        subj = (c.subject or "").strip()
        if not subj:
            continue
        if _object_text_matches_noun_lemma(nl, c.obj or ""):
            out.add(subj)
    return out


def _definite_np_exact_phrase_candidates(
    phrase_text: str,
    pef: PEFState,
    claims: list[ExtractedClaim],
) -> set[str]:
    """Exact referent candidates for a full definite NP phrase (modifiers preserved).

    This protects known multi-token entities like ``gold key`` from being
    collapsed to their head noun (``key``) in ambiguity gating.
    """
    p = (phrase_text or "").strip().lower()
    if not p:
        return set()
    out: set[str] = set()
    for ent in pef.entities.values():
        if (ent.name or "").strip().lower() == p:
            out.add(ent.name)
    for c in claims:
        subj = (c.subject or "").strip()
        if subj and subj.lower() == p:
            out.add(subj)
    return out


# Dependency positions for role/common-noun discourse entities (matches entity-mention gate).
_DISCOURSE_ROLE_NOUN_DEPS: frozenset[str] = frozenset(
    {"nsubj", "nsubjpass", "dobj", "pobj", "appos", "conj"}
)

# Interpersonal claims whose subject/object may be role-noun antecedents.
_INTERPERSONAL_ROLE_CLAIM_RELATIONS: frozenset[str] = frozenset({
    "INFORM", "GIVE", "TAKE", "TELL", "SEND", "ASK", "CONTACT", "ESCALATE",
})

# Transfer verbs where the direct object is the artifact being transferred
# (a document, message, item), not a participant/role noun.  For these,
# only the subject (sender/giver) is checked for role-noun candidacy.
_TRANSFER_VERB_RELATIONS: frozenset[str] = frozenset({
    "SEND", "GIVE", "TAKE",
})

# Collective holders that imply multiple role-noun participants may possess the object.
_COLLECTIVE_POSSESSOR_SUBJECTS: frozenset[str] = frozenset({
    "parties", "both", "they", "all", "everyone", "everybody",
})

_DISCOURSE_META_SUBJECT_PREFIXES: tuple[str, ...] = (
    "further ",
    "no ",
    "additional ",
)


def _is_discourse_role_noun_surface(name: str) -> bool:
    """True for common-noun role entities (operator, contractor), not proper names or meta subjects."""
    name = (name or "").strip()
    if not name or len(name.split()) > 3:
        return False
    lower = name.lower()
    if any(lower.startswith(p) for p in _DISCOURSE_META_SUBJECT_PREFIXES):
        return False
    tokens = name.split()
    if len(tokens) >= 2 and all(
        t[0].isupper() for t in tokens if t and t[0].isalpha()
    ):
        return False
    return True


def _discourse_role_nouns_from_claims(claims: list[ExtractedClaim]) -> set[str]:
    """Role/common-noun participants from interpersonal or possession claims in the turn."""
    out: set[str] = set()
    for c in claims:
        rel = (c.relation or "").upper()
        if rel in _INTERPERSONAL_ROLE_CLAIM_RELATIONS:
            if rel in _TRANSFER_VERB_RELATIONS:
                # dobj of SEND/GIVE/TAKE is the artifact transferred, not a participant.
                # Only the subject (agent) is a potential role-noun candidate.
                parts = (c.subject,)
            else:
                parts = (c.subject, c.obj)
            for part in parts:
                p = (part or "").strip()
                if p and _is_discourse_role_noun_surface(p):
                    out.add(p)
        elif rel == "HAS":
            subj = (c.subject or "").strip()
            if subj and _is_discourse_role_noun_surface(subj):
                out.add(subj)
    return out


def _role_nouns_in_sentence(poss_token: Token) -> set[str]:
    """Role/common-noun entities in argument positions within the possessive's sentence."""
    possessed_head = poss_token.head
    sent = poss_token.sent
    out: set[str] = set()
    for token in sent:
        if token.pos_ != "NOUN":
            continue
        if token.dep_ not in _DISCOURSE_ROLE_NOUN_DEPS:
            continue
        if token in possessed_head.subtree and token != poss_token:
            continue
        if _is_discourse_role_noun_surface(token.text):
            out.add(token.text)
    return out


def _possessive_role_noun_antecedent_candidates(
    poss_token: Token,
    same_turn_claims: list[ExtractedClaim],
    pef: PEFState | None = None,
) -> set[str]:
    """Role-noun antecedents: same sentence, same-turn claims, and committed PEF."""
    candidates = _role_nouns_in_sentence(poss_token) | _discourse_role_nouns_from_claims(
        same_turn_claims,
    )
    if pef is not None:
        candidates |= _discourse_role_nouns_from_pef(pef)
    return candidates


def _role_nouns_capable_of_possessing(
    possessed_lemma: str,
    role_candidates: set[str],
    same_turn_claims: list[ExtractedClaim],
    pef: PEFState | None = None,
) -> set[str]:
    """Role nouns that may hold *possessed_lemma* per same-turn claims (direct or collective)."""
    if len(role_candidates) < 2 or not possessed_lemma:
        return set()
    direct_holders = _definite_np_same_turn_claim_candidates(
        possessed_lemma, same_turn_claims,
    )
    if pef is not None:
        direct_holders |= _definite_np_pef_has_subject_candidates(possessed_lemma, pef)
    capable: set[str] = set()
    for role in role_candidates:
        if role in direct_holders:
            capable.add(role)
    if direct_holders & _COLLECTIVE_POSSESSOR_SUBJECTS:
        participants = _discourse_role_nouns_from_claims(same_turn_claims)
        for role in role_candidates:
            if role in participants:
                capable.add(role)
    return capable if len(capable) >= 2 else set()


def _discourse_role_noun_entity_mentions(
    claims: list[ExtractedClaim],
) -> list[str]:
    """Promote role/common-noun discourse participants for referent-resolution ASK lists."""
    out = _discourse_role_nouns_from_claims(claims)
    out = {x for x in out if x.lower() not in _COLLECTIVE_POSSESSOR_SUBJECTS}
    # Title-case so Lens referent ASK lists accept them (_candidate_entity_names gate).
    return sorted({s[:1].upper() + s[1:] if s else s for s in out}, key=str.lower)


def _possessive_role_noun_ambiguous(
    poss_token: Token,
    same_turn_claims: list[ExtractedClaim],
    pef: PEFState | None = None,
) -> bool:
    """True when a possessive pronoun has 2+ role-noun antecedents that can hold the object."""
    head = poss_token.head
    if head.pos_ not in ("NOUN", "PROPN"):
        return False
    possessed_lemma = head.lemma_.lower()
    role_candidates = _possessive_role_noun_antecedent_candidates(
        poss_token, same_turn_claims, pef,
    )
    if len(role_candidates) < 2:
        return False
    return len(
        _role_nouns_capable_of_possessing(
            possessed_lemma, role_candidates, same_turn_claims, pef,
        )
    ) >= 2


# Role/occupation paraphrase normalization constants.
# Only these verb lemmas trigger the "holds the position of X" -> IS X rewrite.
# "job" excluded: "holds a job at X" is employment location, not role equivalence.
_ROLE_HOLDER_VERBS: frozenset[str] = frozenset({"hold", "occupy"})
_ROLE_NOUNS: frozenset[str] = frozenset({"position", "title", "role", "post"})

# Finance metric heads (lemma, lowercased) for narrow copular admission:
#   <scope> <metric> is|was <finance numeric>
# Requires scope/context signals or PEF overlap — not bare "budget is $5M".
_FIN_METRIC_HEAD_LEMMAS: frozenset[str] = frozenset({
    "revenue", "budget", "forecast", "variance", "spend", "cost", "margin",
    "roi", "profit", "loss", "arr", "mrr", "burn", "ebitda", "earnings",
    "income", "sale", "sales", "cash", "debt", "capital", "allocation",
    "target", "actual", "growth", "expense", "expenses", "opex", "capex",
})

# Region / org-structure lemmas (spaCy subtree + whitespace token checks — no regex gates).
_FIN_SCOPE_LEMMAS: frozenset[str] = frozenset({
    "north", "south", "latin", "america", "apac", "emea", "latam", "anz", "dach",
    "europe", "asia", "africa", "oceania", "worldwide", "global", "international",
    "portfolio", "segment", "division", "subsidiary", "market", "branch", "group",
    "organization", "organisation", "region", "territory", "geo", "world",
    "business", "product", "unit", "line", "middle", "east", "corp",
})

_FIN_CONTEXT_LEMMAS: frozenset[str] = frozenset({
    "ebitda", "ebit", "gaap", "arr", "mrr", "opex", "capex", "yoy", "qoq", "npm", "gpm",
})

_FIN_SCOPE_BIGRAMS: frozenset[str] = frozenset({
    "north america", "south america", "latin america", "middle east",
    "business unit", "product line",
})

_LEADING_DET_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "this", "that", "these", "those",
})

_FIN_EXPL_MOD_LEMMAS: frozenset[str] = frozenset({
    "primary", "main", "key", "root", "principal", "leading",
})

# Past-participle head with auxpass "be" (spaCy often analyzes "the driver is delayed X"
# as ROOT=delayed + dobj=X instead of copula be + attr).
_FIN_EXPLAN_PASSIVE_VERB_LEMMAS: frozenset[str] = frozenset({
    "delay", "attribute", "explain", "drive", "trace", "link", "tie",
})

_FIN_PEF_EXTRA_LEMMAS: frozenset[str] = frozenset({
    "portfolio", "shortfall", "investment", "equity", "revenues", "actuals",
})

# Movement/update verbs that can imply a location mutation for an existing object.
# Example: "I moved the gold key to the study." -> gold key AT study.
_MOTION_LOCATION_VERBS: frozenset[str] = frozenset({
    "move", "put", "place", "relocate", "shift",
})

_MOTION_LOCATION_PREPS: frozenset[str] = frozenset({
    "to", "into", "in", "inside", "within", "at", "on", "onto",
})


def _strip_leading_determiners_np(text: str) -> str:
    s = text.strip()
    parts = s.split(None, 1)
    if parts and parts[0].lower() in _LEADING_DET_WORDS:
        return parts[1].strip() if len(parts) > 1 else ""
    return s


def _tokenize_lower_phrase(text_lower: str) -> list[str]:
    s = text_lower.replace("/", " ")
    for ch in ",.;:":
        s = s.replace(ch, " ")
    return [x.strip("'\"") for x in s.split() if x.strip("'\"")]


def _pef_sole_nontaker_holder(
    pef: PEFState, item_lower: str, taker_lower: str
) -> str | None:
    """Return the name of the sole current holder of item (excluding taker), or None.

    Inlined here to keep extraction/projection concerns out of the evaluation layer.
    Turn-ordered: latest source_turn wins per (subject, item); same turn: NOT_HAS wins.
    """
    cands: dict[str, list] = {}  # subject_id → list of matching HAS rels
    for rel in pef.get_relationships_by_relation("HAS"):
        subj = pef.entities.get(rel.subject_id)
        if subj is None or subj.name.lower() == taker_lower:
            continue
        if rel.object_literal is not None:
            obj = str(rel.object_literal).strip()
        elif rel.object_entity_id is not None:
            ent = pef.entities.get(rel.object_entity_id)
            obj = ent.name.strip() if ent else ""
        else:
            obj = ""
        if not obj:
            continue
        if obj.lower() == item_lower or item_lower in obj.lower():
            cands.setdefault(rel.subject_id, []).append(rel)

    holders: list[str] = []
    for sid, rels in cands.items():
        best_turn = -1
        best_negated: bool | None = None
        for r in rels:
            if r.source_turn > best_turn or (
                r.source_turn == best_turn and r.negated and not best_negated
            ):
                best_turn = r.source_turn
                best_negated = r.negated
        if best_negated is False:
            subj = pef.entities.get(sid)
            if subj:
                holders.append(subj.name)
    return holders[0] if len(holders) == 1 else None


_SMALL_NUMBER_WORDS: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


def _parse_counted_item_literal(text: str) -> tuple[int, str] | None:
    s = _strip_leading_determiners_np(text).strip()
    if not s:
        return None
    parts = s.split(None, 1)
    if len(parts) != 2:
        return None
    n_raw, item = parts[0].lower(), parts[1].strip()
    if not item:
        return None
    if n_raw.isdigit():
        return (int(n_raw), item)
    if n_raw in _SMALL_NUMBER_WORDS:
        return (_SMALL_NUMBER_WORDS[n_raw], item)
    return None


def _latest_positive_numeric_has(
    pef: PEFState,
    subject_name: str,
    consumed_item_key: str,
) -> tuple[int, str, str] | None:
    subj = pef.find_entity_by_name(subject_name)
    if subj is None:
        return None

    latest_by_literal: dict[str, tuple[int, bool, int, str, str]] = {}
    for rel in pef.get_relationships_for_subject(subj.id):
        if rel.relation != "HAS":
            continue
        if rel.object_literal is not None:
            obj = str(rel.object_literal).strip()
        elif rel.object_entity_id is not None:
            ent = pef.entities.get(rel.object_entity_id)
            obj = ent.name.strip() if ent else ""
        else:
            obj = ""
        parsed = _parse_counted_item_literal(obj)
        if parsed is None:
            continue
        count, item = parsed
        if item_key(item) != consumed_item_key:
            continue
        literal_key = obj.lower().strip()
        prior = latest_by_literal.get(literal_key)
        if prior is None:
            latest_by_literal[literal_key] = (
                rel.source_turn,
                rel.negated,
                count,
                item,
                obj,
            )
            continue
        prior_turn, prior_negated, _, _, _ = prior
        if rel.source_turn > prior_turn or (
            rel.source_turn == prior_turn and rel.negated and not prior_negated
        ):
            latest_by_literal[literal_key] = (
                rel.source_turn,
                rel.negated,
                count,
                item,
                obj,
            )

    active = [v for v in latest_by_literal.values() if not v[1]]
    if not active:
        return None
    best_turn = max(v[0] for v in active)
    best_candidates = [v for v in active if v[0] == best_turn]
    _, _, best_count, best_item, best_literal = max(best_candidates, key=lambda v: v[2])
    return (best_count, best_item, best_literal)


def _current_has_holders_for_item_key(pef: PEFState, target_item_key: str) -> list[str]:
    """Return entities whose current HAS state is positive for the normalized item key."""
    latest_by_subject: dict[str, tuple[int, bool, str]] = {}
    for rel in pef.get_relationships_by_relation("HAS"):
        subj = pef.entities.get(rel.subject_id)
        if subj is None:
            continue
        if rel.object_literal is not None:
            obj = str(rel.object_literal).strip()
        elif rel.object_entity_id is not None:
            ent = pef.entities.get(rel.object_entity_id)
            obj = ent.name.strip() if ent else ""
        else:
            obj = ""
        if not obj:
            continue
        parsed = _parse_counted_item_literal(obj)
        item_text = parsed[1] if parsed is not None else obj
        if item_key(item_text) != target_item_key:
            continue
        prior = latest_by_subject.get(subj.id)
        if prior is None:
            latest_by_subject[subj.id] = (rel.source_turn, rel.negated, subj.name)
            continue
        prior_turn, prior_negated, _ = prior
        if rel.source_turn > prior_turn or (
            rel.source_turn == prior_turn and rel.negated and not prior_negated
        ):
            latest_by_subject[subj.id] = (rel.source_turn, rel.negated, subj.name)
    return [
        name for _, is_negated, name in latest_by_subject.values()
        if not is_negated
    ]


def _sole_transfer_subject_candidate_from_pef(
    relation: str,
    claims: list[ExtractedClaim],
    pef: PEFState,
) -> str | None:
    """Return sole prior holder candidate for pronoun-subject GIVE/TAKE claims."""
    rel_norm = canonicalize_relation(relation)
    if rel_norm not in {"GIVE", "TAKE"}:
        return None

    item_keys: set[str] = set()
    for c in claims:
        if c.negated:
            continue
        if canonicalize_relation(c.relation) != rel_norm:
            continue
        obj = str(c.obj or "").strip()
        if not obj:
            continue
        parsed = _parse_counted_item_literal(obj)
        item_text = parsed[1] if parsed is not None else obj
        key = item_key(item_text)
        if key:
            item_keys.add(key)
    if len(item_keys) != 1:
        return None

    holders = _current_has_holders_for_item_key(pef, next(iter(item_keys)))
    return holders[0] if len(holders) == 1 else None


def _token_text_quarter_or_fy_signal(text: str) -> bool:
    u = text.upper().strip(".,;:\"'")
    if u in ("Q1", "Q2", "Q3", "Q4", "H1", "H2"):
        return True
    if len(u) >= 3 and u.startswith("FY") and any(ch.isdigit() for ch in u):
        return True
    return False


def _string_tokens_indicate_finance_scope(text_lower: str) -> bool:
    """Geo / time / org / finance-vocab scope — not bare metric heads (avoids 'variance' in driver NP)."""
    tokens = _tokenize_lower_phrase(text_lower)
    if not tokens:
        return False
    tset = {t.lower() for t in tokens}
    if tset & _FIN_SCOPE_LEMMAS or tset & _FIN_CONTEXT_LEMMAS:
        return True
    for i in range(len(tokens) - 1):
        if f"{tokens[i]} {tokens[i + 1]}".lower() in _FIN_SCOPE_BIGRAMS:
            return True
    return any(_token_text_quarter_or_fy_signal(t) for t in tokens)


def _np_tokens_indicate_finance_scope(root: Token) -> bool:
    for t in root.subtree:
        lem = t.lemma_.lower()
        if lem in _FIN_SCOPE_LEMMAS or lem in _FIN_CONTEXT_LEMMAS:
            return True
        if _token_text_quarter_or_fy_signal(t.text):
            return True
    return False


def _pef_entity_text_overlap(text_lower: str, pef: PEFState) -> bool:
    for ent in pef.entities.values():
        name = (ent.name or "").strip().lower()
        if len(name) < 4:
            continue
        if name in text_lower or text_lower in name:
            return True
    return False


def _finance_scope_ok(nsubj_root: Token, full_subject_text: str, pef: PEFState) -> bool:
    """Scoped finance NP: dependency signals, token/bigram scan, or PEF name overlap."""
    if _np_tokens_indicate_finance_scope(nsubj_root):
        return True
    lowered = full_subject_text.lower()
    if _string_tokens_indicate_finance_scope(lowered):
        return True
    return _pef_entity_text_overlap(lowered, pef)


def _entity_name_tokens_hint_finance(name: str) -> bool:
    nl = (name or "").strip().lower()
    if not nl:
        return False
    toks = {t.lower() for t in _tokenize_lower_phrase(nl) if t}
    if toks & _FIN_METRIC_HEAD_LEMMAS or toks & _FIN_PEF_EXTRA_LEMMAS:
        return True
    return _string_tokens_indicate_finance_scope(nl)


def _pef_has_admitted_finance_context(pef: PEFState) -> bool:
    """True when PEF names look like a finance world (prior setup turns)."""
    for ent in pef.entities.values():
        n = (ent.name or "").strip()
        if not n:
            continue
        if _entity_name_tokens_hint_finance(n):
            return True
        if _string_tokens_indicate_finance_scope(n.lower()):
            return True
    return False


def _np_is_finance_explanatory_head(root: Token) -> bool:
    """True when NP head matches narrow finance explanatory patterns (dependency lemmas)."""
    lemmas = {t.lemma_.lower() for t in root.subtree}
    h = root.lemma_.lower()
    if h == "driver":
        if "variance" in lemmas:
            return True
        if lemmas & _FIN_EXPL_MOD_LEMMAS:
            return True
    if h in ("cause", "factor"):
        if lemmas & _FIN_EXPL_MOD_LEMMAS or "root" in lemmas:
            return True
    if "shortfall" in lemmas and ("driver" in lemmas or "cause" in lemmas):
        return True
    if "performance" in lemmas and "gap" in lemmas and ("driver" in lemmas or "cause" in lemmas):
        return True
    return False


def _finance_explanatory_admission_ok(nsubj_tok: Token, subject_text: str, pef: PEFState) -> bool:
    """Explanatory copula: structural head + (scoped NP OR existing finance PEF)."""
    if not _np_is_finance_explanatory_head(nsubj_tok):
        return False
    if _finance_scope_ok(nsubj_tok, subject_text, pef):
        return True
    return _pef_has_admitted_finance_context(pef)


def _finance_explanatory_object_ok(tail: str) -> bool:
    """Predicative phrase is non-trivial and not a finance numeric literal."""
    t = tail.strip().rstrip(".,;:")
    if len(t) < 6:
        return False
    if parse_numeric(tail) is not None:
        return False
    parts = [w.strip(".,;:") for w in t.split() if len(w.strip(".,;:")) > 2]
    return len(parts) >= 2


def _strip_definite_prefix(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    lower = s.lower()
    if lower.startswith("the "):
        return s[4:].strip()
    return s


# Common suffixes that strongly indicate an organisation or location rather
# than a person.  Used by the ambiguity gate to exclude ORG/LOC entity
# placeholders from the pef_persons set.
_ORG_SUFFIXES: frozenset[str] = frozenset({
    "hospital", "clinic", "medical", "health",
    "corp", "corporation", "inc", "incorporated", "ltd", "llc", "plc",
    "co", "company", "group", "firm", "associates", "partners",
    "university", "college", "school", "institute", "academy",
    "center", "centre", "department", "ministry", "bureau", "office",
    "authority", "agency", "council", "commission", "foundation",
    "bank", "trust", "fund", "hotel", "club",
})


def _likely_org_name(name: str) -> bool:
    """Return True if name looks like an organisation or location rather than a person."""
    return any(w.lower().rstrip(".") in _ORG_SUFFIXES for w in name.split())


def _is_pronoun(token: Token) -> bool:
    return token.text.lower() in _PRONOUNS or token.pos_ == "PRON"


# ── Negation detection ───────────────────────────────────────────────

_NEGATION_DEPS = {"neg"}
_NEGATION_TOKENS = {"not", "n't", "never", "no", "neither", "nor", "doesn't", "don't", "didn't", "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't", "won't", "wouldn't", "couldn't", "shouldn't"}


def _is_negated(token: Token) -> bool:
    """Check if a token (typically a verb) is negated."""
    for child in token.children:
        if child.dep_ in _NEGATION_DEPS:
            return True
        if child.text.lower() in _NEGATION_TOKENS:
            return True
    return False


# ── SpacyBackend ────────────────────────────────────────────────────

class SpacyBackend(ExtractionBackend):
    """spaCy-based deterministic extraction backend.

    Requires spaCy to be installed: pip install aurora-lens[spacy]
    """

    def __init__(self, nlp: spacy.language.Language | None = None, model: str = "en_core_web_sm"):
        try:
            import spacy as _spacy
        except ImportError:
            raise ImportError(
                "SpacyBackend requires spaCy. Install it with: "
                "pip install aurora-lens[spacy]"
            ) from None

        if nlp is None:
            try:
                self._nlp = _spacy.load(model)
            except OSError:
                # Keep first-run deterministic/local even when a language model is
                # not installed yet: run with a blank English pipeline.
                self._nlp = _spacy.blank("en")
                if "sentencizer" not in self._nlp.pipe_names:
                    self._nlp.add_pipe("sentencizer")
        else:
            self._nlp = nlp

    @property
    def nlp(self):  # noqa: ANN201 — runtime ``Language`` from spaCy
        """Loaded spaCy pipeline (shared reference ok for frame builders)."""
        return self._nlp

    async def extract(self, text: str, pef: PEFState) -> ExtractionResult:
        """Extract entities, relationships, and span signals from text."""
        doc = await asyncio.to_thread(self._nlp, text)
        result = ExtractionResult()
        result.span = detect_span(doc)

        # Collect named entities and noun chunks
        result.entity_mentions = self._extract_entity_mentions(doc)

        # Extract SVO relationships from dependency parse
        result.claims = self._extract_claims(doc, result.span, pef)

        # Role/common-noun discourse entities (operator, contractor, …) for referent ASK.
        _role_mentions = _discourse_role_noun_entity_mentions(result.claims)
        if _role_mentions:
            _seen_mentions = {m.lower() for m in result.entity_mentions}
            for _name in _role_mentions:
                if _name.lower() not in _seen_mentions:
                    result.entity_mentions.append(_name)
                    _seen_mentions.add(_name.lower())

        # Resolve pronouns against the existing world
        result.pronoun_candidates = self._resolve_pronouns(doc, pef, result.claims)

        # Detect possessive pronouns that cannot be uniquely resolved
        result.ambiguous_referents = self._detect_ambiguous_referents(
            doc, pef, result.claims
        )

        # Detect comparative adjectives whose comparand is underdetermined
        result.comparative_ambiguities = self._detect_comparative_claims(doc, pef)

        return result

    # Dependency relations that indicate a token fills an entity-like argument
    # position (subject, object, appositive, conjunct).  Tokens in predicative
    # or modifier positions (attr, compound, amod, nmod, …) are excluded so
    # that title words like "Chief", "Medical", "Officer" in
    # "is the Chief Medical Officer" are not treated as person names.
    _ENTITY_POSITION_DEPS: frozenset[str] = frozenset(
        {"nsubj", "nsubjpass", "dobj", "pobj", "appos", "conj"}
    )

    def _extract_entity_mentions(self, doc: Doc) -> list[str]:
        """Extract named entities and proper noun chunks from spaCy doc.

        NER entities (all recognised types) are added first.  The PROPN
        fallback adds proper-noun tokens that NER missed, subject to two
        guards that eliminate the entity-explosion problem:

        1. Token-index coverage: skip tokens already inside any NER span.
           This prevents sub-tokens of multi-word named entities (e.g. "Dr."
           and "Chen" from inside the "Dr. Chen" PERSON span, or "Northgate"
           and "Hospital" from the "Northgate Hospital" ORG span) from
           becoming separate PEF entity placeholders.

        2. Dependency-position filter: only add PROPN tokens that occupy
           subject, object, appositive, or conjunct positions.  Tokens in
           predicative/modifier positions (attr, compound, amod, etc.) are
           excluded — this stops title fragments like "Chief", "Medical",
           "Officer" from "is the Chief Medical Officer" from being treated as
           person-name candidates by the ambiguity gate.
        """
        mentions: list[str] = []

        # Build a set of token indices already covered by any NER span so the
        # PROPN fallback can skip tokens that are fragments of a recognised
        # multi-word entity.
        ner_token_indices: set[int] = set()
        for ent in doc.ents:
            for tok in ent:
                ner_token_indices.add(tok.i)

        # All NER entity types — includes ORG/GPE/LOC so that names like
        # "Emma" that spaCy mis-labels ORG are still captured here and don't
        # fall through to the PROPN loop (which the NER coverage check would
        # then block).
        # For PERSON entities, extend backward to include immediately preceding
        # PROPN compound tokens (titles such as "Dr.", "Mrs.", "Prof.") that
        # spaCy omits from the NER span.  Without this, "Dr. Chen" would be
        # split into entity_mentions=["Chen"] (from NER) and claim subject
        # "Dr. Chen" (from _get_span_text subtree), producing two distinct PEF
        # entities for the same person and triggering a spurious multiple-
        # antecedent count in the ambiguity gate.
        for ent in doc.ents:
            if ent.label_ not in ("PERSON", "ORG", "GPE", "LOC", "NORP", "FAC"):
                continue
            full_text = ent.text
            if ent.label_ == "PERSON":
                prefix: list[str] = []
                i = ent.start - 1
                while i >= 0:
                    t = doc[i]
                    # Only absorb PROPN compound tokens not in any other NER span
                    if t.pos_ == "PROPN" and t.dep_ == "compound" and not t.ent_type_:
                        prefix.insert(0, t.text)
                        ner_token_indices.add(t.i)  # Prevent PROPN fallback from re-adding
                        i -= 1
                    else:
                        break
                if prefix:
                    full_text = " ".join(prefix) + " " + ent.text
            mentions.append(full_text)

        # Proper nouns not caught by NER — likely person names the model missed.
        seen: set[str] = {m.lower() for m in mentions}
        for token in doc:
            if token.pos_ != "PROPN":
                continue
            if token.i in ner_token_indices:
                continue
            if token.dep_ not in self._ENTITY_POSITION_DEPS:
                continue
            text = token.text
            if text.lower() not in seen:
                seen.add(text.lower())
                mentions.append(text)

        return mentions

    def _is_interrogative_sentence(self, sent: Any) -> bool:
        """Return True if this sentence is a question.

        Questions must not contribute claims to PEF state — they ask about
        the world rather than asserting facts about it.  Two heuristics:
        1. Sentence text ends with "?" (orthographic signal).
        2. Root verb has a WH-word (WP/WRB/WDT) as its subject (syntactic).
        """
        if sent.text.strip().endswith("?"):
            return True
        for token in sent:
            if token.dep_ == "ROOT":
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass") and child.tag_ in ("WP", "WRB", "WDT"):
                        return True
        return False

    def _find_nsubj_token_active(self, verb: Token) -> Token | None:
        """Active-clause nsubj only (excludes passive nsubjpass)."""
        for child in verb.children:
            if child.dep_ == "nsubj":
                return child
        return None

    def _finance_copula_tail_text(self, verb: Token, doc: Doc) -> str:
        """Text after the copula within the same sentence (predicative complement region)."""
        if verb.i + 1 >= verb.sent.end:
            return ""
        return doc[verb.i + 1 : verb.sent.end].text

    def _try_finance_metric_copula_claim(
        self,
        verb: Token,
        doc: Doc,
        span: Span,
        pef: PEFState,
        negated: bool,
        evidence: str,
    ) -> ExtractedClaim | None:
        """Admit <scope> <metric> is|was <finance numeric> as a single IS claim (narrow).

        Does not replace general extraction when guards fail. Requires:
        whitelisted metric head (on nsubj or predicative attr), finance numeric
        in the tail, and scope/context/PEF overlap — not e.g. 'budget is $5M'
        with no geographic or time scope.

        Set ``AURORA_LENS_DEBUG_FINANCE_COPULA=1`` for INFO logs that show whether
        this path ran and which gate failed (live deploy vs parse-shape diagnosis).
        """
        sent_text = verb.sent.text.strip()
        if verb.lemma_.lower() != "be":
            return None

        _log_finance_copula("enter", sentence=sent_text, verb_i=verb.i, verb_text=verb.text)

        nsubj_tok = self._find_nsubj_token_active(verb)
        if nsubj_tok is None:
            _log_finance_copula("reject_no_nsubj", sentence=sent_text)
            return None
        if _is_pronoun(nsubj_tok):
            _log_finance_copula(
                "reject_nsubj_pronoun",
                sentence=sent_text,
                nsubj=nsubj_tok.text,
            )
            return None

        tail = self._finance_copula_tail_text(verb, doc)
        if not tail.strip():
            _log_finance_copula("reject_empty_tail", sentence=sent_text, tail=tail)
            return None
        if parse_numeric(tail) is None:
            _log_finance_copula(
                "reject_tail_not_finance_numeric",
                sentence=sent_text,
                tail=tail.strip()[:120],
            )
            return None

        subject_text: str | None = None
        head_lemma = nsubj_tok.lemma_.lower()
        if head_lemma in _FIN_METRIC_HEAD_LEMMAS:
            subject_text = self._get_span_text(nsubj_tok)
            _log_finance_copula(
                "metric_on_nsubj",
                sentence=sent_text,
                head_lemma=head_lemma,
                subject_np=subject_text,
            )
        else:
            attr_metric: Token | None = None
            for c in verb.children:
                if c.dep_ == "attr" and c.lemma_.lower() in _FIN_METRIC_HEAD_LEMMAS:
                    attr_metric = c
                    break
            if attr_metric is None:
                _log_finance_copula(
                    "reject_no_metric_head",
                    sentence=sent_text,
                    nsubj_head_lemma=head_lemma,
                    nsubj_span=self._get_span_text(nsubj_tok),
                    child_deps=[
                        (ch.text, ch.dep_, ch.lemma_)
                        for ch in list(verb.children)[:14]
                    ],
                )
                return None
            scope_span = self._get_span_text(nsubj_tok).strip()
            if not _finance_scope_ok(nsubj_tok, scope_span, pef):
                _log_finance_copula(
                    "reject_attr_path_scope",
                    sentence=sent_text,
                    scope_span=scope_span,
                )
                return None
            metric_phrase = self._get_head_noun_text(attr_metric).strip()
            subject_text = f"{scope_span} {metric_phrase}".strip()
            _log_finance_copula(
                "repaired_attr_metric",
                sentence=sent_text,
                subject_built=subject_text,
            )

        if not subject_text:
            _log_finance_copula("reject_empty_subject", sentence=sent_text)
            return None

        subject_text = _strip_leading_determiners_np(subject_text)
        if not subject_text:
            _log_finance_copula("reject_subject_only_det", sentence=sent_text)
            return None

        if not _finance_scope_ok(nsubj_tok, subject_text, pef):
            _log_finance_copula(
                "reject_full_subject_scope",
                sentence=sent_text,
                subject=subject_text,
            )
            return None

        verb_span = self._verb_local_span(verb, span)
        _log_finance_copula(
            "accepted",
            sentence=sent_text,
            subject=subject_text,
            obj=tail.strip()[:200],
            relation="IS",
        )
        return ExtractedClaim(
            subject=subject_text,
            relation="IS",
            obj=tail.strip(),
            span=verb_span,
            negated=negated,
            evidence=evidence,
            provenance="user_input",
            extractor_backend="spacy",
        )

    def _try_finance_explanatory_copula_claim(
        self,
        verb: Token,
        doc: Doc,
        span: Span,
        pef: PEFState,
        negated: bool,
        evidence: str,
    ) -> ExtractedClaim | None:
        """Admit <finance explanatory head> is|was <non-numeric cause phrase> (narrow).

        Runs only when the metric copula path does not apply. Requires whitelisted
        explanatory subject pattern, non-numeric multi-token object, and either
        scoped subject NP or an already-admitted finance PEF (continuity turns).
        """
        sent_text = verb.sent.text.strip()
        if verb.lemma_.lower() != "be":
            return None

        _log_finance_copula("expl_enter", sentence=sent_text, verb_i=verb.i)

        nsubj_tok = self._find_nsubj_token_active(verb)
        if nsubj_tok is None or _is_pronoun(nsubj_tok):
            _log_finance_copula("expl_reject_nsubj", sentence=sent_text)
            return None

        tail = self._finance_copula_tail_text(verb, doc)
        if not tail.strip():
            _log_finance_copula("expl_reject_empty_tail", sentence=sent_text)
            return None
        if parse_numeric(tail) is not None:
            _log_finance_copula("expl_reject_numeric_tail", sentence=sent_text, tail=tail[:80])
            return None
        if not _finance_explanatory_object_ok(tail):
            _log_finance_copula("expl_reject_object_shape", sentence=sent_text, tail=tail[:80])
            return None

        subject_text = _strip_leading_determiners_np(self._get_span_text(nsubj_tok).strip())
        if not subject_text:
            return None
        if not _finance_explanatory_admission_ok(nsubj_tok, subject_text, pef):
            _log_finance_copula(
                "expl_reject_admission_context",
                sentence=sent_text,
                subject=subject_text,
            )
            return None

        verb_span = self._verb_local_span(verb, span)
        _log_finance_copula(
            "expl_accepted",
            sentence=sent_text,
            subject=subject_text,
            obj=tail.strip()[:200],
        )
        return ExtractedClaim(
            subject=subject_text,
            relation="IS",
            obj=tail.strip(),
            span=verb_span,
            negated=negated,
            evidence=evidence,
            provenance="user_input",
            extractor_backend="spacy",
        )

    def _try_finance_explanatory_participle_claim(
        self,
        token: Token,
        doc: Doc,
        span: Span,
        pef: PEFState,
        negated: bool,
        evidence: str,
    ) -> ExtractedClaim | None:
        """Finance explanatory admission when spaCy uses VBN + auxpass be + dobj (not copula be).

        Example: "The primary variance driver is delayed enterprise deals." often parses as
        ROOT ``delayed`` (VBN), ``is`` auxpass, ``driver`` nsubjpass, ``deals`` dobj.
        Normalizes to the same ``IS`` fact as the true copula parse.
        """
        if token.tag_ != "VBN":
            return None
        if token.lemma_.lower() not in _FIN_EXPLAN_PASSIVE_VERB_LEMMAS:
            return None
        has_auxpass_be = any(
            c.dep_ == "auxpass" and c.lemma_.lower() == "be"
            for c in token.children
        )
        if not has_auxpass_be:
            return None

        nsubj_tok: Token | None = None
        for c in token.children:
            if c.dep_ == "nsubjpass":
                nsubj_tok = c
                break
        if nsubj_tok is None or _is_pronoun(nsubj_tok):
            return None

        dobj_tok: Token | None = None
        for c in token.children:
            if c.dep_ == "dobj":
                dobj_tok = c
                break
        if dobj_tok is None:
            return None

        subject_text = _strip_leading_determiners_np(self._get_span_text(nsubj_tok).strip())
        if not subject_text:
            return None
        if not _finance_explanatory_admission_ok(nsubj_tok, subject_text, pef):
            _log_finance_copula(
                "expl_part_reject_admission",
                sentence=token.sent.text.strip(),
                subject=subject_text,
            )
            return None

        obj_text = (token.text + " " + self._get_span_text(dobj_tok)).strip()
        if parse_numeric(obj_text) is not None:
            return None
        if not _finance_explanatory_object_ok(obj_text):
            return None

        verb_span = self._verb_local_span(token, span)
        _log_finance_copula(
            "expl_part_accepted",
            sentence=token.sent.text.strip(),
            subject=subject_text,
            obj=obj_text[:200],
        )
        return ExtractedClaim(
            subject=subject_text,
            relation="IS",
            obj=obj_text,
            span=verb_span,
            negated=negated,
            evidence=evidence,
            provenance="user_input",
            extractor_backend="spacy",
        )

    def _extract_claims(self, doc: Doc, span: Span, pef: PEFState) -> list[ExtractedClaim]:
        """Extract subject-verb-object claims from dependency parse.

        Interrogative sentences are skipped: questions do not assert facts
        and must not create PEF state entries.

        Each verb may produce multiple claims: one for a direct/attribute object
        (using the verb lemma as relation) and one per prepositional complement
        (using the preposition text as relation).  This ensures that sentences
        like "Alice is a software engineer at Acme Corp." store both
        Alice IS "software engineer" AND Alice AT "Acme Corp." in PEF.
        """
        claims: list[ExtractedClaim] = []

        if _finance_copula_diag_enabled():
            _log_finance_copula(
                "_extract_claims_start",
                text_preview=doc.text.strip()[:300],
                pef_entity_count=len(pef.entities),
            )

        # Build set of token indices that belong to interrogative sentences.
        interrogative_indices: set[int] = set()
        for sent in doc.sents:
            if self._is_interrogative_sentence(sent):
                for token in sent:
                    interrogative_indices.add(token.i)

        for token in doc:
            if token.i in interrogative_indices:
                continue
            if token.pos_ not in ("VERB", "AUX"):
                continue
            if token.dep_ == "aux":
                continue

            # Find subject
            subject = self._find_subject(token)
            if subject is None:
                if _finance_copula_diag_enabled() and token.lemma_.lower() == "be":
                    _log_finance_copula(
                        "bailout_before_finance_hook_no_subject",
                        sentence=token.sent.text.strip(),
                        verb_i=token.i,
                        verb_dep=token.dep_,
                    )
                continue

            # Collect all (relation, obj, token) triples for this verb.
            base_relation = canonicalize_relation(token.lemma_)
            raw_object_pairs = self._find_objects(token, base_relation)

            # Resolve pronouns (specifically 'it') in object literals before creating claims.
            # Requirement: No admitted canonical possession or transfer relation may use
            # an unresolved pronoun as its object literal.
            object_pairs: list[tuple[str, str]] = []
            for rel, obj, obj_tok in raw_object_pairs:
                if obj_tok is not None:
                    quantified = self._resolve_quantified_pronoun_object(obj_tok, obj, pef, subject)
                    if quantified:
                        object_pairs.append((rel, quantified))
                        continue
                if obj_tok is not None and _is_pronoun(obj_tok) and obj_tok.text.lower() == "it":
                    resolved = self._resolve_pronoun_object(obj_tok, pef, subject)
                    if resolved:
                        object_pairs.append((rel, resolved))
                    else:
                        # If unresolved, do not admit as a canonical object literal.
                        # This prevents junk "HAS it" or "GIVE it" claims.
                        continue
                else:
                    object_pairs.append((rel, obj))

            # Single-viable transfer subject binding from committed PEF:
            # "He gave Sarah 5 apples." with one prior holder of apples should
            # resolve "He" before projection so arithmetic uses the true giver.
            if (
                subject.lower() in _SUBJECT_OBJECT_PRONOUNS
                and base_relation in {"GIVE", "TAKE"}
                and object_pairs
            ):
                transfer_claims = [
                    ExtractedClaim(
                        subject=subject,
                        relation=rel,
                        obj=obj,
                        span=span,
                        negated=False,
                        evidence="",
                    )
                    for rel, obj in object_pairs
                    if rel in {"GIVE", "TAKE"} and obj
                ]
                forced_subject = _sole_transfer_subject_candidate_from_pef(
                    base_relation,
                    transfer_claims,
                    pef,
                )
                if forced_subject is not None:
                    subject = forced_subject

            # Compute before the object_pairs guard: shared by passive extraction
            # and the regular claim loop.
            negated = _is_negated(token)
            evidence = self._get_clause_text(token, doc)

            # Movement-as-location mutation for existing entities:
            # "I moved the gold key to the study." should update committed AT for gold key.
            if token.lemma_.lower() in _MOTION_LOCATION_VERBS:
                dobj_tok = None
                for child in token.children:
                    if child.dep_ in ("dobj", "obj"):
                        dobj_tok = child
                        break
                if dobj_tok is not None:
                    moved_subject = _strip_leading_determiners_np(
                        self._get_head_noun_text(dobj_tok).strip()
                    )
                    moved_subject = _strip_definite_prefix(moved_subject)
                    if moved_subject and pef.find_entity_by_name(moved_subject) is not None:
                        dest_text = None
                        prep_nodes = [
                            c for c in token.children
                            if c.dep_ == "prep" and c.text.lower() in _MOTION_LOCATION_PREPS
                        ]
                        # Some parses attach "to/in ..." under the moved object NP.
                        prep_nodes.extend(
                            c for c in dobj_tok.children
                            if c.dep_ == "prep" and c.text.lower() in _MOTION_LOCATION_PREPS
                        )
                        for prep in prep_nodes:
                            for pobj in prep.children:
                                if pobj.dep_ != "pobj":
                                    continue
                                cand = _strip_leading_determiners_np(
                                    self._get_head_noun_text(pobj).strip()
                                )
                                cand = _strip_definite_prefix(cand)
                                if cand:
                                    dest_text = cand
                                    break
                            if dest_text:
                                break
                        if dest_text:
                            claims.append(ExtractedClaim(
                                subject=moved_subject,
                                relation="AT",
                                obj=dest_text,
                                span=self._verb_local_span(token, span),
                                negated=negated,
                                evidence=evidence,
                                provenance="user_input",
                                extractor_backend="spacy",
                            ))

            # Per-verb temporal span: use the verb's own morphology rather than
            # the document-level span.  LLM responses often start with
            # present-tense preamble ("That's correct.", "I understand that...")
            # which biases detect_span() toward PRESENT even when the main verb
            # is past tense.  Verb-local span is more accurate and prevents
            # spurious TIME_SMEAR flags on correct acknowledgement responses.
            verb_span = self._verb_local_span(token, span)

            # Passive-agent extraction. Purely structural — no NER/PROPN heuristics.
            # "The contract was signed by Meridian Partners." emits a flipped active
            # claim so the agent is grounded in PEF before the LLM response is checked.
            passive_claims = self._extract_passive_agent_claims(
                token, base_relation, span, negated, evidence,
            )
            claims.extend(passive_claims)

            # Finance explanatory: VBN + auxpass be (mis-parsed copula) — before be-copula hooks.
            expl_part = self._try_finance_explanatory_participle_claim(
                token, doc, span, pef, negated, evidence,
            )
            if expl_part is not None:
                claims.append(expl_part)
                continue

            # Narrow finance metric copula — same conceptual bucket as 'APAC Q3 revenue was $4.2M'.
            if token.lemma_.lower() == "be":
                _log_finance_copula(
                    "finance_hook_reached",
                    sentence=token.sent.text.strip(),
                    subject_string=subject,
                    pef_entity_count=len(pef.entities),
                )
                fin_claim = self._try_finance_metric_copula_claim(
                    token, doc, span, pef, negated, evidence,
                )
                if fin_claim is not None:
                    claims.append(fin_claim)
                    # Keep prepositional complements (e.g. AT / IN) from the same verb;
                    # skip duplicate generic IS rows that the finance path replaced.
                    for rel, obj in object_pairs:
                        if rel == "IS":
                            continue
                        claims.append(ExtractedClaim(
                            subject=subject,
                            relation=rel,
                            obj=obj,
                            span=verb_span,
                            negated=negated,
                            evidence=evidence,
                            provenance="user_input",
                            extractor_backend="spacy",
                        ))
                    continue

                expl_claim = self._try_finance_explanatory_copula_claim(
                    token, doc, span, pef, negated, evidence,
                )
                if expl_claim is not None:
                    claims.append(expl_claim)
                    for rel, obj in object_pairs:
                        if rel == "IS":
                            continue
                        claims.append(ExtractedClaim(
                            subject=subject,
                            relation=rel,
                            obj=obj,
                            span=verb_span,
                            negated=negated,
                            evidence=evidence,
                            provenance="user_input",
                            extractor_backend="spacy",
                        ))
                    continue

            if not object_pairs:
                continue

            for rel, obj in object_pairs:
                claims.append(ExtractedClaim(
                    subject=subject,
                    relation=rel,
                    obj=obj,
                    span=verb_span,
                    negated=negated,
                    evidence=evidence,
                    provenance="user_input",
                    extractor_backend="spacy",
                ))

            # GIVE-transfer state projection: "Alice gave Bob a book" → derive
            # HAS(Bob, book) and NOT_HAS(Alice, book) so inventory queries work.
            if base_relation == "GIVE" and not negated:
                recipient_tok: Token | None = None
                
                # Recipient can be:
                # 1. iobj/dative: "gave Alice the book"
                # 2. prep "to": "gave the book to Alice"
                # 3. grandchild of "back": "gave the book back to Alice"
                
                # Try iobj/dative first
                iobj_tok = next(
                    (c for c in token.children if c.dep_ in ("iobj", "dative")), None
                )
                if iobj_tok is not None:
                    # If iobj is actually the preposition "to", use its pobj
                    if iobj_tok.text.lower() == "to" and iobj_tok.pos_ in ("ADP", "SCONJ"):
                        recipient_tok = next(
                            (c for c in iobj_tok.children if c.dep_ == "pobj"), None
                        )
                    else:
                        recipient_tok = iobj_tok
                
                # If not found, look for "to" preposition
                if recipient_tok is None:
                    to_prep = next(
                        (c for c in token.children
                         if c.dep_ == "prep" and c.text.lower() == "to"),
                        None,
                    )
                    if to_prep is not None:
                        recipient_tok = next(
                            (c for c in to_prep.children if c.dep_ == "pobj"), None
                        )
                
                # If still not found, look for "back to X"
                if recipient_tok is None:
                    back_tok = next(
                        (c for c in token.children 
                         if c.text.lower() == "back" and c.dep_ in ("advmod", "prt")),
                        None
                    )
                    if back_tok is not None:
                        to_prep = next(
                            (c for c in back_tok.children 
                             if c.dep_ == "prep" and c.text.lower() == "to"),
                            None
                        )
                        if to_prep is not None:
                            recipient_tok = next(
                                (c for c in to_prep.children if c.dep_ == "pobj"), None
                            )

                # 4. Some parses attach the beneficiary as ``nmod`` under ``dobj`` for
                # small numeric quantities ("… gave Jill 5 apples.") instead of dative ``Jill``.
                if recipient_tok is None:
                    dobj_give = next(
                        (c for c in token.children if c.dep_ == "dobj"),
                        None,
                    )
                    if dobj_give is not None:
                        for gc in dobj_give.children:
                            if gc.dep_ == "nmod" and gc.pos_ == "PROPN":
                                recipient_tok = gc
                                break

                recipient: str | None = None
                if recipient_tok is not None:
                    recipient = _strip_leading_determiners_np(
                        self._get_head_noun_text(recipient_tok).strip()
                    )

                if recipient:
                    for rel, item in object_pairs:
                        if rel != "GIVE" or not item:
                            continue
                        emit_default_giver_not_has = True
                        claims.append(ExtractedClaim(
                            subject=recipient,
                            relation="HAS",
                            obj=item,
                            span=verb_span,
                            negated=False,
                            evidence=evidence,
                            provenance="user_input",
                            extractor_backend="spacy",
                        ))
                        # Counted GIVE arithmetic projection:
                        # if giver currently has N item and gives k item, project
                        # NOT_HAS(N item) + HAS((N-k) item) for the giver.
                        gave_parsed = _parse_counted_item_literal(item)
                        if gave_parsed is not None:
                            gave_count, gave_item_text = gave_parsed
                            gave_item_key = item_key(gave_item_text)
                            prior = _latest_positive_numeric_has(pef, subject, gave_item_key)
                            if prior is not None:
                                prior_count, prior_item_text, prior_literal = prior
                                if prior_count >= gave_count:
                                    emit_default_giver_not_has = False
                                    old_literal = prior_literal
                                    new_literal = f"{prior_count - gave_count} {prior_item_text}"
                                    claims.append(ExtractedClaim(
                                        subject=subject,
                                        relation="HAS",
                                        obj=old_literal,
                                        span=verb_span,
                                        negated=True,
                                        evidence=evidence,
                                        provenance="user_input",
                                        extractor_backend="spacy",
                                    ))
                                    claims.append(ExtractedClaim(
                                        subject=subject,
                                        relation="HAS",
                                        obj=new_literal,
                                        span=verb_span,
                                        negated=False,
                                        evidence=evidence,
                                        provenance="user_input",
                                        extractor_backend="spacy",
                                    ))
                        if emit_default_giver_not_has:
                            claims.append(ExtractedClaim(
                                subject=subject,
                                relation="HAS",
                                obj=item,
                                span=verb_span,
                                negated=True,
                                evidence=evidence,
                                provenance="user_input",
                                extractor_backend="spacy",
                            ))

            # TAKE-transfer state projection: "Alice took the book [from Bob / back]"
            # → HAS(Alice, book) + NOT_HAS(prior_holder, book) when holder is known.
            if base_relation == "TAKE" and not negated:
                take_items = [item for rel, item in object_pairs if rel == "TAKE" and item]
                if take_items:
                    for item in take_items:
                        claims.append(ExtractedClaim(
                            subject=subject,
                            relation="HAS",
                            obj=item,
                            span=verb_span,
                            negated=False,
                            evidence=evidence,
                            provenance="user_input",
                            extractor_backend="spacy",
                        ))
                    # Determine prior holder: explicit "from X" takes precedence,
                    # then "back" + sole current non-taker holder in PEF.
                    prior_holder: str | None = None
                    from_prep = next(
                        (c for c in token.children
                         if c.dep_ == "prep" and c.text.lower() == "from"),
                        None,
                    )
                    if from_prep is not None:
                        pobj = next(
                            (c for c in from_prep.children if c.dep_ == "pobj"), None
                        )
                        if pobj is not None:
                            cand = _strip_leading_determiners_np(
                                self._get_head_noun_text(pobj).strip()
                            )
                            prior_holder = cand or None
                    if prior_holder is None:
                        has_back = any(
                            c.text.lower() == "back" and c.dep_ in ("advmod", "prt")
                            for c in token.children
                        )
                        if has_back:
                            for item in take_items:
                                prior_holder = _pef_sole_nontaker_holder(
                                    pef, item.lower(), subject.lower()
                                )
                                if prior_holder:
                                    break
                    if prior_holder:
                        for item in take_items:
                            claims.append(ExtractedClaim(
                                subject=prior_holder,
                                relation="HAS",
                                obj=item,
                                span=verb_span,
                                negated=True,
                                evidence=evidence,
                                provenance="user_input",
                                extractor_backend="spacy",
                            ))

            # EAT/consume quantity projection: "Jill ate one lollipop" with current
            # HAS(Jill, 7 lollipops) -> NOT_HAS(Jill, 7 lollipops) + HAS(Jill, 6 lollipops).
            if base_relation in CONSUMPTION_RELATIONS and not negated:
                eaten_items = [item for rel, item in object_pairs if rel in CONSUMPTION_RELATIONS and item]
                for eaten in eaten_items:
                    parsed_eaten = _parse_counted_item_literal(eaten)
                    if parsed_eaten is None:
                        continue
                    consumed_count, consumed_item_text = parsed_eaten
                    consumed_item_key = item_key(consumed_item_text)
                    if not consumed_item_key:
                        continue
                    current = _latest_positive_numeric_has(pef, subject, consumed_item_key)
                    if current is None:
                        continue
                    current_count, current_item_text, current_literal = current
                    if current_count < consumed_count:
                        continue
                    new_count = current_count - consumed_count
                    old_literal = current_literal
                    new_literal = f"{new_count} {current_item_text}"
                    claims.append(ExtractedClaim(
                        subject=subject,
                        relation="HAS",
                        obj=old_literal,
                        span=verb_span,
                        negated=True,
                        evidence=evidence,
                        provenance="user_input",
                        extractor_backend="spacy",
                    ))
                    claims.append(ExtractedClaim(
                        subject=subject,
                        relation="HAS",
                        obj=new_literal,
                        span=verb_span,
                        negated=False,
                        evidence=evidence,
                        provenance="user_input",
                        extractor_backend="spacy",
                    ))

        return claims

    def _verb_local_span(self, verb: "Token", doc_span: Span) -> Span:
        """Detect the temporal span for a single verb from its own morphology.

        Claims must not inherit a misleading document-level span.  LLM
        responses often start with present-tense preamble ("That's correct.",
        "I understand that...") which pulls detect_span() toward PRESENT even
        when the substantive verb is past tense ("signed", "was hired", etc.).

        Check the verb's own morphological tense, then its auxiliaries, then
        fall back to the document-level span if no signal is found.  This
        applies to both passive and active verbs.
        """
        morph = verb.morph.get("Tense")
        if "Past" in morph:
            return Span.PAST
        if "Pres" in morph:
            return Span.PRESENT
        for child in verb.children:
            if child.dep_ in ("aux", "auxpass"):
                aux_morph = child.morph.get("Tense")
                if "Past" in aux_morph:
                    return Span.PAST
                if "Pres" in aux_morph:
                    return Span.PRESENT
                # Surface-form fallback for common auxiliaries
                if child.text.lower() in {"was", "were", "had"}:
                    return Span.PAST
                if child.text.lower() in {"is", "are", "am", "has", "have"}:
                    return Span.PRESENT
        return doc_span

    def _extract_passive_agent_claims(
        self,
        verb: "Token",
        base_relation: str,
        span: Span,
        negated: bool,
        evidence: str,
    ) -> list[ExtractedClaim]:
        """For passive-voice verbs with explicit by-agents, emit a flipped active claim.

        Three structural requirements -- all must be satisfied:
        1. verb has an nsubjpass child  (structurally passive sentence)
        2. verb has a dep_=="agent" child with a pobj grandchild, OR a dep_=="prep"
           child with text=="by" and a pobj grandchild
           (spaCy en_core_web_sm labels passive "by" as dep_=="agent"; the "prep/by"
            branch covers older model versions for robustness)
        3. Both the passive subject and the agent NP are non-empty after _get_span_text

        Relation consistency: canonicalize_relation(verb.lemma_) runs identically on
        both the passive path (here) and the active equivalent, so PEF storage and
        LLM-response extraction produce identical relation strings for the same verb.
        """
        # Requirement 1: structural passive
        passive_subj_token: Token | None = None
        for child in verb.children:
            if child.dep_ == "nsubjpass":
                passive_subj_token = child
                break
        if passive_subj_token is None:
            return []

        passive_subj = self._get_span_text(passive_subj_token)
        if not passive_subj.strip():
            return []

        # Requirement 2: explicit agent phrase.
        # spaCy labels passive "by X" as dep_=="agent" in en_core_web_sm.
        # The dep_=="prep" / text=="by" branch handles older model versions.
        agent_pobj: Token | None = None
        for child in verb.children:
            if child.dep_ == "agent":
                for gc in child.children:
                    if gc.dep_ == "pobj":
                        agent_pobj = gc
                        break
            elif child.dep_ == "prep" and child.text.lower() == "by":
                for gc in child.children:
                    if gc.dep_ == "pobj":
                        agent_pobj = gc
                        break
            if agent_pobj is not None:
                break
        if agent_pobj is None:
            return []

        # Requirement 3: non-empty agent NP
        agent = self._get_span_text(agent_pobj)
        if not agent.strip():
            return []

        return [ExtractedClaim(
            subject=agent,
            relation=base_relation,
            obj=passive_subj,
            span=self._verb_local_span(verb, span),
            negated=negated,
            evidence=evidence,
            provenance="user_input",
            extractor_backend="spacy",
        )]

    def _find_subject(self, verb: Token) -> str | None:
        """Find the subject of a verb from its dependency children."""
        for child in verb.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                return self._get_span_text(child)
        # Check head for passives / aux chains
        if verb.dep_ in ("xcomp", "ccomp", "conj"):
            return self._find_subject(verb.head)
        return None

    def _find_objects(self, verb: Token, base_relation: str) -> list[tuple[str, str, Token | None]]:
        """Find all valid objects for a verb, returning (relation, obj, token) triples.

        Direct/attribute/complement objects use the verb lemma's canonical relation
        (with prepositional complements stripped from the head noun text).
        Prepositional objects — whether attached directly to the verb or to the
        attribute noun — use the preposition text as the canonical relation, so
        "works at Acme" → ("AT", "Acme", token) and "is a software engineer at Acme Corp."
        yields [("IS", "software engineer", token), ("AT", "Acme Corp.", token)].
        """
        results: list[tuple[str, str, Token | None]] = []
        for child in verb.children:
            if child.dep_ in ("dobj", "attr", "oprd", "acomp"):
                # Paraphrase normalization: "holds/occupies [the] position/title/role/post of X"
                # -> IS X. Three-part structural trigger (all must be true simultaneously):
                #   1. verb lemma in _ROLE_HOLDER_VERBS
                #   2. dobj lemma in _ROLE_NOUNS
                #   3. dobj has a "of" prep child with a pobj
                # When triggered: emit IS X only, skip raw ("HOLD","position") pair.
                # When not triggered (no "of" pobj): fall through to normal processing.
                normalized_role: str | None = None
                role_tok: Token | None = None
                if (verb.lemma_.lower() in _ROLE_HOLDER_VERBS
                        and child.lemma_.lower() in _ROLE_NOUNS):
                    for gc in child.children:
                        if gc.dep_ == "prep" and gc.text.lower() == "of":
                            for ggc in gc.children:
                                if ggc.dep_ == "pobj":
                                    normalized_role = self._get_head_noun_text(ggc)
                                    role_tok = ggc
                                    break
                            break

                if normalized_role is not None:
                    results.append(("IS", normalized_role, role_tok))
                    # Skip raw ("HOLD", "position") and its prep complements.
                else:
                    # Use a clean head-noun text (without prepositional sub-phrases).
                    # For counted GIVE, ``en_core_web_sm`` sometimes attaches the recipient as
                    # ``nmod`` on ``dobj`` ("He gave Jill 5 apples."). Excluding ``nmod`` yields
                    # a countable object literal ("5 apples") and keeps the beneficiary on the verb.
                    rel_canon = canonicalize_relation(verb.lemma_.lower())
                    q_tail: str | None = None
                    if rel_canon in {"GIVE", "TAKE"} and child.dep_ == "dobj":
                        q_tail = self._quantified_of_pronoun_dobj_surface(child)
                    if q_tail is not None:
                        results.append((base_relation, q_tail, child))
                    else:
                        _give_verb_dobj = rel_canon == "GIVE" and child.dep_ == "dobj"
                        results.append(
                            (
                                base_relation,
                                self._get_head_noun_text(
                                    child,
                                    extra_excl_deps=frozenset({"nmod"})
                                    if _give_verb_dobj
                                    else None,
                                ),
                                child,
                            )
                        )
                    # Also emit a separate claim for each prepositional complement
                    # attached to this noun (e.g. "at Acme Corp." on "engineer").
                    # Guard: only emit when the preposition maps to a known canonical
                    # relation — this prevents "works as a software engineer" from
                    # producing a spurious "Alice AS software engineer" claim.
                    for grandchild in child.children:
                        if grandchild.dep_ == "prep":
                            prep_rel = canonicalize_relation(grandchild.text)
                            if prep_rel in CANONICAL_RELATIONS:
                                for ggchild in grandchild.children:
                                    if ggchild.dep_ == "pobj":
                                        results.append((prep_rel, self._get_head_noun_text(ggchild), ggchild))
            elif child.dep_ == "prep":
                # Prepositional phrase attached directly to the verb.
                # Same guard: only canonical-relation prepositions.
                prep_rel = canonicalize_relation(child.text)
                if prep_rel in CANONICAL_RELATIONS:
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            # Use head-noun text to strip relative clauses so
                            # "Acme (which is part of Acme Corp.)" → "Acme".
                            results.append((prep_rel, self._get_head_noun_text(grandchild), grandchild))
            # Predicative adverbs on copula: "was overseas", "is home", "is abroad".
            # Guard: only accept advmod when the verb is a form of "be" — this
            # prevents manner adverbs ("ran quickly") from becoming claim objects.
            elif child.dep_ == "advmod" and verb.lemma_ == "be":
                results.append((base_relation, child.text, child))
        return results

    # Dependency relations whose subtrees should be stripped when extracting
    # the head-noun text of a noun phrase. Prepositional phrases are kept as
    # separate claims; relative clauses / adverbial clauses are not claims.
    _EXCL_NOUN_DEPS: frozenset[str] = frozenset({"prep", "relcl", "advcl", "acl"})

    def _get_head_noun_text(
        self,
        token: Token,
        *,
        extra_excl_deps: frozenset[str] | None = None,
    ) -> str:
        """Get noun phrase text excluding prepositional and clausal sub-phrases.

        For "software engineer at Acme Corp.", returns "software engineer"
        so that the prepositional part can be emitted as a separate claim.
        For "Acme (which is part of Acme Corp.)", returns "Acme" so that the
        relative clause does not bloat the object string.

        Optional ``extra_excl_deps`` extends the exclusion set when a dependency
        should be stripped for a bounded relation shape (see GIVE counted objects).
        """
        excl_deps = self._EXCL_NOUN_DEPS
        if extra_excl_deps:
            excl_deps = frozenset(excl_deps | extra_excl_deps)
        excl_indices: set[int] = set()
        for child in token.children:
            if child.dep_ in excl_deps:
                for t in child.subtree:
                    excl_indices.add(t.i)
        subtree_tokens = sorted(token.subtree, key=lambda t: t.i)
        meaningful = [
            t for t in subtree_tokens
            if t.i not in excl_indices
            and t.pos_ != "PUNCT"
            and (t.dep_ != "det" or t.pos_ == "PROPN")
        ]
        return " ".join(t.text for t in meaningful) if meaningful else token.text

    def _get_span_text(self, token: Token) -> str:
        """Get the full noun phrase text for a token (including children)."""
        # Use subtree for compound nouns and adjective modifiers
        subtree_tokens = sorted(token.subtree, key=lambda t: t.i)
        # Filter to meaningful tokens (skip determiners for cleaner output)
        meaningful = [
            t for t in subtree_tokens
            if t.dep_ not in ("det",) or t.pos_ == "PROPN"
        ]
        if meaningful:
            return " ".join(t.text for t in meaningful)
        return token.text

    def _get_clause_text(self, verb: Token, doc: Doc) -> str:
        """Get the approximate clause text around a verb."""
        subtree = sorted(verb.subtree, key=lambda t: t.i)
        if subtree:
            start = subtree[0].i
            end = subtree[-1].i + 1
            return doc[start:end].text
        return verb.text

    def _quantified_of_pronoun_dobj_surface(self, token: Token) -> str | None:
        """Return ``N of them`` / ``N of it`` when ``dobj`` is a quantity linked to a pronoun.

        Head-noun extraction strips ``prep`` subtrees, which turns *one of them* into bare
        *one* and loses the pronoun tail needed for clarification replay and resolution.
        """
        if token.tag_ != "CD" and token.pos_ != "NUM":
            return None
        qty_surface = token.text.strip()
        if not qty_surface:
            return None
        for prep in token.children:
            if prep.dep_ != "prep" or prep.text.lower() != "of":
                continue
            for pobj in prep.children:
                if pobj.dep_ != "pobj":
                    continue
                if _is_pronoun(pobj) and pobj.text.lower() in {"them", "it"}:
                    return f"{qty_surface} of {pobj.text}".strip()
        return None

    def _detect_ambiguous_referents(
        self,
        doc: Doc,
        pef: PEFState,
        same_turn_claims: list[ExtractedClaim] | None = None,
    ) -> list[str]:
        """Return pronouns and definite NPs that cannot be uniquely resolved.

        Detects three categories:
        1. Possessive pronouns (dep_==poss) when 2+ person antecedents exist.
        2. Subject/object pronouns (dep_ in nsubj/nsubjpass/dobj/pobj) when
           2+ person antecedents exist.
        3. Definite NPs ("the medication", "that condition") when 2+ structurally
           eligible referents match the head noun — **committed PEF plus same-turn
           claims** (``update_pef`` has not run yet, so same-turn assertions must
           be merged for admissibility).

        PEF axiom: binding without explicit grounding is not permitted.
        """
        if same_turn_claims is None:
            same_turn_claims = []
        doc_persons: set[str] = {ent.text for ent in doc.ents if ent.label_ == "PERSON"}
        # spaCy NER occasionally misses one side of "Emma ... Anna ... she ..." patterns.
        # Add person-like proper nouns from the local sentence so pronoun ambiguity does
        # not depend on perfect PERSON tagging.
        for tok in doc:
            if tok.pos_ != "PROPN":
                continue
            if not tok.text or not tok.text[0].isupper():
                continue
            if len(tok.text.split()) > 2:
                continue
            if _likely_org_name(tok.text):
                continue
            if tok.dep_ in ("nsubj", "nsubjpass", "dobj", "pobj", "conj", "appos"):
                doc_persons.add(tok.text)

        # Include PEF-resident entities as additional antecedents.
        # Heuristic: short (≤2 tokens), capitalized, not org-like → likely a person.
        # Do NOT filter on resolved=True: all PEF entities — whether resolved or
        # not — are valid pronoun antecedents for ambiguity detection.
        pef_persons: set[str] = {
            ent.name for ent in pef.entities.values()
            if ent.name not in doc_persons
            and len(ent.name.split()) <= 2
            and ent.name[0].isupper()
            and not _likely_org_name(ent.name)
        }

        all_persons = doc_persons | pef_persons
        has_multiple_persons = len(all_persons) >= 2

        ambiguous: list[str] = []
        seen: set[str] = set()
        for token in doc:
            lower = token.text.lower()

            # 1. Possessive pronouns (person antecedents + role/common-noun antecedents)
            if lower in _POSSESSIVE_PRONOUNS and token.dep_ == "poss":
                if lower in seen:
                    continue
                if _possessive_locally_bound_by_sentence_subject(
                    token, doc, doc_persons
                ):
                    continue
                role_noun_ambiguous = _possessive_role_noun_ambiguous(
                    token, same_turn_claims, pef,
                )
                if has_multiple_persons or role_noun_ambiguous:
                    seen.add(lower)
                    ambiguous.append(lower)
                continue

            # 2. Subject/object pronouns
            if lower in _SUBJECT_OBJECT_PRONOUNS and token.dep_ in (
                "nsubj", "nsubjpass", "dobj", "pobj"
            ):
                if lower == "it":
                    # Only treat unresolved "it" as ambiguity for non-interrogative
                    # transfer/possession mutation contexts.
                    if self._is_interrogative_sentence(token.sent):
                        continue
                    governing_rel = canonicalize_relation(token.head.lemma_)
                    if governing_rel not in ("GIVE", "HAS", "TAKE"):
                        continue
                    subject_name = self._find_subject(token.head) or ""
                    if self._resolve_pronoun_object(token, pef, subject_name) is not None:
                        continue
                    if lower not in seen:
                        seen.add(lower)
                        ambiguous.append(lower)
                elif has_multiple_persons and lower not in seen:
                    if token.dep_ in ("nsubj", "nsubjpass"):
                        governing_rel = canonicalize_relation(token.head.lemma_)
                        if governing_rel in ("GIVE", "TAKE"):
                            forced = _sole_transfer_subject_candidate_from_pef(
                                governing_rel,
                                same_turn_claims,
                                pef,
                            )
                            if forced is not None:
                                continue
                    seen.add(lower)
                    ambiguous.append(lower)
                continue

            # 3. Definite NP: determiner + NOUN/PROPN head
            # Ambiguous when 2+ distinct referent candidates (PEF + same-turn claims).
            if lower in _DEFINITE_DETERMINERS and token.dep_ == "det":
                head = token.head
                if head.pos_ in ("NOUN", "PROPN"):
                    # A definite NP that is the direct object of a transfer verb is the
                    # transferred object itself — not an anaphoric reference requiring
                    # prior unique binding.  "Anna gave the book to John" establishes
                    # the book's continuity through the transfer; treating Anna and John
                    # as referent candidates for 'the book' inverts PEF ontology.
                    if head.dep_ == "dobj" and canonicalize_relation(
                        head.head.lemma_
                    ) in ("GIVE", "TAKE"):
                        continue
                    full_phrase = _strip_leading_determiners_np(
                        self._get_head_noun_text(head).strip()
                    )
                    # If the full NP phrase already binds uniquely (e.g. "the gold key"
                    # with an existing "gold key" entity), do not degrade to generic
                    # head-noun ambiguity ("key").
                    exact_phrase_matches = _definite_np_exact_phrase_candidates(
                        full_phrase, pef, same_turn_claims
                    )
                    if len(exact_phrase_matches) == 1:
                        continue
                    noun_lemma = head.lemma_.lower()
                    pef_names = _definite_np_pef_referent_candidates(noun_lemma, pef)
                    # Same-turn subjects (who acted on the noun) are not referents
                    # when the noun concept is already grounded in committed PEF.
                    noun_grounded_via_relationship = _noun_grounded_in_pef(noun_lemma, pef)
                    if noun_grounded_via_relationship:
                        claim_names: set[str] = set()
                    else:
                        claim_names = _definite_np_same_turn_claim_candidates(
                            noun_lemma, same_turn_claims
                        )
                    matching = sorted(pef_names | claim_names)
                    surface_head = head.text.lower()
                    key = f"{lower} {surface_head}"
                    # Near-demonstratives ("this"/"that") are deictic: they presuppose
                    # a salient antecedent rather than introducing a new referent.
                    # Zero candidates is therefore *not* an "ungrounded assertion" the
                    # way a bare definite article can be — it is a reference failure,
                    # UNLESS the noun concept is already grounded via a committed
                    # relationship object-literal (e.g. "Escrow HAS transaction" — the
                    # concept has exactly one grounding even though no standalone PEF
                    # entity carries the name). "That transaction must remain off the
                    # books" with no transaction anywhere in world state must not
                    # silently mint a new entity named "that transaction" and admit
                    # the claim; it must fail to resolve. But once a transaction has
                    # actually been committed, the same phrase must resolve, not hold.
                    # Interrogative sentences are exempt, matching the existing
                    # precedent for bare-pronoun "it" ambiguity (see the "1. Subject
                    # object pronouns" branch above): a question asking about "this
                    # draft record" is not committing an assertion that needs a
                    # resolved subject the way "That transaction must remain off the
                    # books" (a declarative/imperative claim) does. Questions read
                    # PEF; they do not write to it.
                    is_near_demonstrative = lower in {"this", "that"}
                    zero_antecedent_demonstrative = (
                        is_near_demonstrative
                        and len(matching) == 0
                        and not noun_grounded_via_relationship
                        and not self._is_interrogative_sentence(token.sent)
                        and not _cataphoric_colon_reference_in_sent(head)
                    )
                    if (
                        (len(matching) >= 2 or zero_antecedent_demonstrative)
                        and key not in seen
                    ):
                        if (
                            noun_lemma == "role"
                            and _suppress_definite_role_for_local_scope_instruction(
                                token, head
                            )
                        ):
                            continue
                        # If exactly one capitalized named instance of the head noun
                        # exists in the doc, the referent is unambiguous — resolve
                        # silently without asking. Applies to both the 2+ (world-state
                        # candidates) and zero-antecedent (doc-local salience) cases.
                        # "this reactor" where only "Reactor 3" appears → skip.
                        # "this reactor" where "Reactor 3" and "Reactor 7" both
                        # appear → fall through and flag as usual.
                        if is_near_demonstrative and _demonstrative_uniquely_resolved_in_doc(
                            head, doc
                        ):
                            continue
                        seen.add(key)
                        ambiguous.append(key)

        return ambiguous

    def _detect_comparative_claims(
        self, doc: Doc, pef: PEFState
    ) -> list[ComparativeAmbiguity]:
        """Detect comparative adjectives whose comparand cannot be uniquely determined.

        Fires when a JJR/RBR token (comparative adjective/adverb) modifies "be"
        and 2+ eligible comparands exist in PEF.

        - 0 candidates: ungrounded assertion — accept silently (no gate).
        - 1 candidate: forced collapse — accept silently (exactly one referent).
        - 2+ candidates: returned for pre-LLM gate to ask one clarifying question.
        """
        ambiguities: list[ComparativeAmbiguity] = []
        for token in doc:
            if token.tag_ not in ("JJR", "RBR"):
                continue
            if token.dep_ not in ("acomp", "advmod"):
                continue
            verb = token.head
            if verb.lemma_ != "be":
                continue

            # Find the subject token
            subj_token = None
            for child in verb.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subj_token = child
                    break
            if subj_token is None:
                continue

            # The noun being compared is the nsubj head ("stick" in "James's stick")
            noun = subj_token.lemma_

            # Identify the possessor, if any ("James" in "James's stick")
            possessor: str | None = None
            for grandchild in subj_token.children:
                if grandchild.dep_ == "poss":
                    possessor = grandchild.text
                    break

            candidates = self._find_comparand_candidates(noun, possessor, pef)
            if len(candidates) >= 2:
                ambiguities.append(ComparativeAmbiguity(
                    adjective=token.text,
                    noun=noun,
                    candidates=candidates,
                ))

        return ambiguities

    def _find_comparand_candidates(
        self, noun: str, possessor: str | None, pef: PEFState
    ) -> list[str]:
        """Find PEF entities that are eligible comparands for a comparative claim.

        If possessor is given (e.g., "James" in "James's stick is bigger"):
            → find other entities that HAS an object containing the noun ("stick").
        If no possessor (e.g., "James is bigger"):
            → find other short, capitalised entities in PEF (likely persons).
        """
        candidates: list[str] = []
        noun_lower = noun.lower()

        if possessor is not None:
            possessor_lower = possessor.lower()
            for entity in pef.entities.values():
                if entity.name.lower() == possessor_lower:
                    continue
                for rel in pef.get_relationships_for_subject(entity.id):
                    if rel.relation in ("HAS", "MANAGE"):
                        obj = str(rel.object_literal or "").lower()
                        if noun_lower in obj:
                            candidates.append(entity.name)
                            break
        else:
            for entity in pef.entities.values():
                if entity.name.lower() == noun_lower:
                    continue
                if len(entity.name.split()) <= 2 and entity.name[0].isupper():
                    candidates.append(entity.name)

        return candidates

    def _resolve_pronoun_object(self, pronoun_tok: Token, pef: PEFState, subject_name: str) -> str | None:
        """Resolve pronoun 'it' to a salient object literal or entity.

        1. Same-sentence resolution: look for earlier dobj/pobj in the same sentence.
        2. PEF-based resolution: look for objects currently held by the subject.
        """
        if pronoun_tok.text.lower() != "it":
            return None

        # 1. Same-sentence resolution: look for earlier dobj/pobj in the same sentence.
        sent = pronoun_tok.sent
        for i in range(pronoun_tok.i - 1, sent.start - 1, -1):
            t = pronoun_tok.doc[i]
            # Look for non-pronoun noun/propn objects
            if t.dep_ in ("dobj", "obj", "pobj") and t.pos_ in ("NOUN", "PROPN") and not _is_pronoun(t):
                return self._get_head_noun_text(t)

        # 2. PEF-based resolution: look for objects currently held by the subject.
        # Requirement: exactly one active transferable object held by the giver.
        subj_ent = pef.find_entity_by_name(subject_name)
        if subj_ent:
            # Track latest HAS state per object key for this subject.
            latest_by_item: dict[str, tuple[int, bool, str]] = {}
            for rel in pef.get_relationships_for_subject(subj_ent.id):
                if rel.relation != "HAS":
                    continue
                item = rel.object_literal or (
                    pef.entities[rel.object_entity_id].name
                    if rel.object_entity_id and rel.object_entity_id in pef.entities
                    else None
                )
                if not item:
                    continue
                key = item.strip().lower()
                prev = latest_by_item.get(key)
                turn = rel.source_turn if rel.source_turn is not None else -1
                if prev is None or turn >= prev[0]:
                    latest_by_item[key] = (turn, rel.negated, item)

            held_items = [
                original_item
                for _, (_, is_negated, original_item) in latest_by_item.items()
                if not is_negated
            ]
            if len(held_items) == 1:
                return held_items[0]

        return None

    def _resolve_quantified_pronoun_object(
        self,
        obj_tok: Token,
        obj_text: str,
        pef: PEFState,
        subject_name: str,
    ) -> str | None:
        """Resolve patterns like ``5 of them`` -> ``5 lollipops`` when uniquely grounded."""
        ot = str(obj_text or "").strip()
        if not ot:
            return None

        parts_ot = ot.split()
        tail_from_surface = (
            len(parts_ot) == 3
            and parts_ot[1].lower() == "of"
            and parts_ot[2].lower() in {"them", "it"}
        )
        if tail_from_surface:
            qty = parts_ot[0]
            if not (qty.isdigit() or qty.lower() in _SMALL_NUMBER_WORDS):
                return None
        else:
            qty = ot
            if not (qty.isdigit() or qty.lower() in _SMALL_NUMBER_WORDS):
                return None

        pronoun_tok: Token | None = None
        for child in obj_tok.children:
            if child.dep_ != "prep" or child.text.lower() != "of":
                continue
            for gc in child.children:
                if gc.dep_ == "pobj" and _is_pronoun(gc) and gc.text.lower() in {"them", "it"}:
                    pronoun_tok = gc
                    break
            if pronoun_tok is not None:
                break

        if not tail_from_surface and pronoun_tok is None:
            return None

        def _parse_counted_item_literal(text: str) -> tuple[int, str] | None:
            s = str(text or "").strip()
            parts = s.split(None, 1)
            if len(parts) != 2:
                return None
            n_raw, item = parts[0].lower(), parts[1].strip()
            if not item:
                return None
            if n_raw.isdigit() or n_raw in _SMALL_NUMBER_WORDS:
                return (1, item)
            return None

        def _active_items_for_subject(subject: str) -> set[str]:
            out: set[str] = set()
            ent = pef.find_entity_by_name(subject)
            if ent is None:
                return out
            for rel in pef.get_relationships_for_subject(ent.id):
                if rel.relation != "HAS" or rel.negated:
                    continue
                obj = rel.object_literal or (
                    pef.entities[rel.object_entity_id].name
                    if rel.object_entity_id and rel.object_entity_id in pef.entities
                    else None
                )
                parsed = _parse_counted_item_literal(str(obj or ""))
                if parsed is not None:
                    out.add(parsed[1])
            return out

        items = _active_items_for_subject(subject_name)
        if len(items) != 1:
            # Fallback: global unique active counted item in session.
            items = set()
            for rel in pef.get_relationships_by_relation("HAS"):
                if rel.negated:
                    continue
                obj = rel.object_literal or (
                    pef.entities[rel.object_entity_id].name
                    if rel.object_entity_id and rel.object_entity_id in pef.entities
                    else None
                )
                parsed = _parse_counted_item_literal(str(obj or ""))
                if parsed is not None:
                    items.add(parsed[1])
            if len(items) != 1:
                return None

        item = next(iter(items))
        return f"{qty} {item}".strip()

    def _resolve_pronouns(
        self, doc: Doc, pef: PEFState, same_turn_claims: list[ExtractedClaim] | None = None,
    ) -> dict[str, str]:
        """Attempt pronoun resolution using recency heuristic.

        Returns a dict of pronoun_key -> entity_name.
        """
        if same_turn_claims is None:
            same_turn_claims = []
        candidates: dict[str, str] = {}

        for token in doc:
            if not _is_pronoun(token):
                continue

            key = f"{token.text.lower()}@token_{token.i}"
            if token.dep_ in ("nsubj", "nsubjpass"):
                governing_rel = canonicalize_relation(token.head.lemma_)
                if governing_rel in {"GIVE", "TAKE"}:
                    forced = _sole_transfer_subject_candidate_from_pef(
                        governing_rel,
                        same_turn_claims,
                        pef,
                    )
                    if forced is not None:
                        candidates[key] = forced
                        continue
            resolved = self._resolve_single_pronoun(token.text, pef)
            if resolved:
                candidates[key] = resolved

        return candidates

    def _resolve_single_pronoun(
        self, pronoun: str, pef: PEFState
    ) -> str | None:
        """Resolve a pronoun to the most recently active matching entity.

        Uses gender heuristic for he/she/him/her; recency for others.
        """
        resolved_entities = [
            e for e in pef.entities.values() if e.resolved
        ]
        if not resolved_entities:
            return None

        by_recency = sorted(
            resolved_entities,
            key=lambda e: e.turn_last_active,
            reverse=True,
        )

        if by_recency:
            return by_recency[0].name

        return None

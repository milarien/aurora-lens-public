"""Strict state-native query surfaces — bounded English templates only.

Templates are matched with normalized whitespace and case-insensitive literal
prefix/suffix checks (no regular expressions).
"""

from __future__ import annotations


def _normalize_ws(text: str) -> str:
    """Single-line-ish collapse: internal newlines/spaces → single spaces."""
    return " ".join(text.split())


def _strip_optional_question(text: str) -> str:
    t = text.strip()
    if t.endswith("?"):
        t = t[:-1].rstrip()
    return t


def _strip_trailing_locative_suffix(rest: str) -> str:
    """Remove optional comma and trailing ``now`` / ``currently`` / ``right now`` (case-insensitive)."""
    rest = rest.rstrip()
    while True:
        low = rest.lower()
        changed = False
        # Optional comma immediately before time adverb (", now" / ", currently")
        while low.endswith(","):
            rest = rest[:-1].rstrip()
            low = rest.lower()
        for suf in (" right now", " currently", " now"):
            if low.endswith(suf):
                rest = rest[: -len(suf)].rstrip().rstrip(",").rstrip()
                low = rest.lower()
                changed = True
                break
        if not changed:
            break
    return rest.rstrip(",").rstrip()


def _consume_ci_prefix(text: str, prefix: str) -> str | None:
    """If *text* starts with *prefix* (case-insensitive), return remainder; else None."""
    if len(text) < len(prefix):
        return None
    if text[: len(prefix)].lower() != prefix.lower():
        return None
    return text[len(prefix) :].strip()


def _optional_the(rest: str) -> str:
    if rest.lower().startswith("the "):
        return rest[4:].strip()
    return rest


def parse_location_subject_phrase(user_text: str) -> str | None:
    """Return trimmed subject phrase for a strict location query, or None."""
    text = _normalize_ws(user_text or "")
    if not text:
        return None
    text = _strip_optional_question(text)
    if not text:
        return None

    result: str | None = None

    for head in ("where is ", "where are "):
        tail = _consume_ci_prefix(text, head)
        if tail is not None:
            tail = _optional_the(tail)
            result = _strip_trailing_locative_suffix(tail)
            break

    if result is None:
        tail = _consume_ci_prefix(text, "where's ")
        if tail is not None:
            tail = _optional_the(tail)
            result = _strip_trailing_locative_suffix(tail)

    if result is None:
        tail = _consume_ci_prefix(text, "where did i put ")
        if tail is not None:
            tail = _optional_the(tail)
            result = tail.strip()

    return result if result else None


def parse_inventory_subject_have_phrase(user_text: str) -> str | None:
    """Return subject phrase for strict ``What does X have`` query, or None."""
    text = _normalize_ws(user_text or "")
    if not text:
        return None
    text = _strip_optional_question(text)
    if not text:
        return None

    tail = _consume_ci_prefix(text, "what does ")
    if tail is None:
        return None
    tail = _optional_the(tail)
    low = tail.lower()
    if not low.endswith(" have"):
        return None
    subj = tail[: -len(" have")].strip()
    return subj if subj else None


def parse_inventory_object_holder_phrase(user_text: str) -> str | None:
    """Return object phrase for strict ``Who has Y`` query, or None."""
    text = _normalize_ws(user_text or "")
    if not text:
        return None
    text = _strip_optional_question(text)
    if not text:
        return None

    tail = _consume_ci_prefix(text, "who has ")
    if tail is None:
        return None
    tail = _optional_the(tail)
    return tail if tail else None


# Spelled numbers aligned with ``_SMALL_NUMBER_WORDS`` in inventory eval (qty-scoped ``Who has``).
_QTY_WORDS: dict[str, int] = {
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

_QTY_HOLDER_PRONOUN_ONLY_REST: frozenset[str] = frozenset({
    "them",
    "it",
    "these",
    "those",
    "they",
    "of them",
    "of it",
})


def parse_inventory_quantity_holder_phrase(user_text: str) -> tuple[int, str] | None:
    """When ``Who has N <item>`` is quantity-scoped, return ``(quantity, item_tail)``.

    ``item_tail`` is the phrasing after the first token (multi-word allowed), e.g.
    ``(4, "lollipops")`` for *Who has 4 lollipops?*. Returns ``None`` when the
    query is not quantity-led (fall through to generic item-key holder matching)
    or when the remainder is a bare pronoun / ``of them`` style tail.
    """
    text = _normalize_ws(user_text or "")
    if not text:
        return None
    text = _strip_optional_question(text)
    if not text:
        return None

    tail = _consume_ci_prefix(text, "who has ")
    if tail is None:
        return None
    tail = _optional_the(tail)
    if not tail:
        return None

    parts = tail.split()
    if len(parts) < 2:
        return None

    q_raw = parts[0]
    rest = " ".join(parts[1:]).strip()
    if not rest:
        return None

    if q_raw.isdigit():
        qty = int(q_raw)
    else:
        qlow = q_raw.lower()
        if qlow not in _QTY_WORDS:
            return None
        qty = _QTY_WORDS[qlow]

    rest_low = rest.lower()
    if rest_low in _QTY_HOLDER_PRONOUN_ONLY_REST:
        return None

    return (qty, rest)


def parse_inventory_how_many_phrase(user_text: str) -> tuple[str, str] | None:
    """Return ``(subject_phrase, item_phrase)`` for strict ``How many Y does X have`` queries."""
    text = _normalize_ws(user_text or "")
    if not text:
        return None
    text = _strip_optional_question(text)
    if not text:
        return None

    tail = _consume_ci_prefix(text, "how many ")
    if tail is None:
        return None

    idx = tail.lower().find(" does ")
    if idx < 0:
        return None

    item = tail[:idx].strip()
    rest = tail[idx + len(" does ") :].strip()
    if not item or not rest:
        return None

    item = _strip_trailing_locative_suffix(item)
    rest = _optional_the(rest)
    rest = _strip_trailing_locative_suffix(rest)
    low = rest.lower()
    if not low.endswith(" have"):
        return None

    subject = rest[: -len(" have")].strip()
    if not subject:
        return None
    subject = _optional_the(subject)
    if not subject:
        return None
    return (subject, item)


def parse_inventory_how_many_past_phrase(user_text: str) -> tuple[str, str] | None:
    """Return ``(subject_phrase, item_phrase)`` for past-tense ``How many Y did X have`` queries.

    Handles ``How many Y did X have?`` and ``How many Y did X have left?``.
    """
    text = _normalize_ws(user_text or "")
    if not text:
        return None
    text = _strip_optional_question(text)
    if not text:
        return None

    tail = _consume_ci_prefix(text, "how many ")
    if tail is None:
        return None

    idx = tail.lower().find(" did ")
    if idx < 0:
        return None

    item = tail[:idx].strip()
    rest = tail[idx + len(" did "):].strip()
    if not item or not rest:
        return None

    item = _strip_trailing_locative_suffix(item)
    rest = _optional_the(rest)
    low = rest.lower()

    if low.endswith(" have left"):
        rest = rest[: -len(" have left")].strip()
    elif low.endswith(" have"):
        rest = rest[: -len(" have")].strip()
    else:
        return None

    subject = rest.strip()
    if not subject:
        return None
    subject = _optional_the(subject)
    if not subject:
        return None
    return (subject, item)


def parse_inventory_subject_has_item_phrase(user_text: str) -> tuple[str, str] | None:
    """Return ``(subject_phrase, item_phrase)`` for strict ``Does X have Y`` queries."""
    text = _normalize_ws(user_text or "")
    if not text:
        return None
    text = _strip_optional_question(text)
    if not text:
        return None

    tail = _consume_ci_prefix(text, "does ")
    if tail is None:
        return None
    idx = tail.lower().find(" have ")
    if idx < 0:
        return None

    subject = tail[:idx].strip()
    item = tail[idx + len(" have ") :].strip()
    if not subject or not item:
        return None
    item = _strip_trailing_locative_suffix(item)
    subject = _optional_the(subject)
    subject = _strip_trailing_locative_suffix(subject)
    item = _optional_the(item)
    if not subject or not item:
        return None
    return (subject, item)


def parse_is_still_in_locative(user_text: str) -> tuple[str, str] | None:
    """Parse ``Is [the] X still in [the] Y?`` → ``(subject_phrase, claimed_place)`` or None."""
    text = _normalize_ws(user_text or "")
    if not text:
        return None
    text = _strip_optional_question(text)
    if not text:
        return None
    tail = _consume_ci_prefix(text, "is ")
    if tail is None:
        return None
    tail = _optional_the(tail)
    low = tail.lower()
    needle = " still in "
    idx = low.find(needle)
    if idx < 0:
        return None
    left = tail[:idx].strip()
    right = tail[idx + len(needle) :].strip()
    if not left or not right:
        return None
    left = _optional_the(left)
    right = _optional_the(right)
    return (left, right) if left and right else None


# Strict surface patterns for "who should I follow up with" queries.
# "which [role] should I follow up with" is matched by a prefix+suffix check below.
_FOLLOW_UP_WHOLE: tuple[str, ...] = (
    "who should i follow up with",
    "who do i follow up with",
    "who should i contact",
    "who should i reach out to",
    "who to follow up with",
    "who to contact",
    "who should be contacted",
)


def _is_follow_up_phrase(phrase: str) -> bool:
    """Check a single normalised, lowercased, question-stripped phrase."""
    for p in _FOLLOW_UP_WHOLE:
        if phrase == p or phrase.startswith(p + " "):
            return True
    # "which [role] should i follow up with" — any role word between "which" and end marker.
    if phrase.startswith("which ") and (
        phrase.endswith(" should i follow up with")
        or phrase.endswith(" should i contact")
        or phrase.endswith(" should i reach out to")
    ):
        return True
    return False


def extract_follow_up_attribution_tail(user_text: str) -> str | None:
    """Return the follow-up attribution sentence from *user_text*, if present.

    Preserves original surface casing. Used when binding resume narrows the LLM
    payload to ``blocked_proposition`` but state-native still needs the retrieval
    question from ``original_question``.
    """
    text = _normalize_ws(user_text or "")
    if not text:
        return None
    stripped = text.strip()
    if _is_follow_up_phrase(_strip_optional_question(stripped.lower())):
        return stripped
    last_boundary = max(text.rfind(". "), text.rfind("! "), text.rfind("? "))
    if last_boundary < 0:
        return None
    last_sentence = text[last_boundary + 2:].strip()
    if not last_sentence:
        return None
    if _is_follow_up_phrase(_strip_optional_question(last_sentence.lower())):
        return last_sentence
    return None


def state_native_text_for_binding_resume(
    narrowed_user_input: str,
    original_question: str | None,
) -> str:
    """Build state-native query text after binding resume.

    ``narrowed_user_input`` is the bound ``blocked_proposition`` sent to the LLM.
    When ``original_question`` ends with a follow-up attribution ask omitted from
    that narrow payload, append only that sentence for state-native delegation.
    """
    base = (narrowed_user_input or "").strip()
    orig = (original_question or "").strip()
    if not orig or parse_follow_up_attribution_query(base):
        return base
    if not parse_follow_up_attribution_query(orig):
        return base
    tail = extract_follow_up_attribution_tail(orig)
    if not tail:
        return base
    if not base:
        return tail
    return f"{base} {tail}"


def parse_follow_up_attribution_query(user_text: str) -> bool:
    """Return True if the query is asking for a follow-up attribution target.

    Matches standalone queries and follow-up questions embedded at the end of
    multi-sentence messages (e.g. after a resolved referent clarification replay).
    Does NOT match general who-questions (e.g. ``who has the book``).
    """
    text = _normalize_ws(user_text or "").lower()
    text = _strip_optional_question(text)
    if not text:
        return False
    if _is_follow_up_phrase(text):
        return True
    # Multi-sentence message: check the last sentence after any ". ", "! ", or "? " boundary.
    last_boundary = max(text.rfind(". "), text.rfind("! "), text.rfind("? "))
    if last_boundary >= 0:
        last_sentence = _strip_optional_question(text[last_boundary + 2:].strip())
        if last_sentence and _is_follow_up_phrase(last_sentence):
            return True
    return False


# Temporal contact query: "when should I contact X" / "when can I reach X"
_TEMPORAL_CONTACT_PREFIXES: tuple[str, ...] = (
    "when should i contact ",
    "when can i contact ",
    "when should i reach ",
    "when can i reach ",
    "when should i call ",
    "when can i call ",
    "when should i get in touch with ",
    "when can i get in touch with ",
    "when to contact ",
    "when to reach ",
    "when to call ",
)


def parse_comparative_question(user_text: str) -> tuple[str, str] | None:
    """Return (noun_phrase, adjective) for 'Whose NOUN was ADJECTIVE?' style queries.

    Handles:
      "Whose dog was bigger?"         → ("dog", "bigger")
      "Which dog was bigger?"         → ("dog", "bigger")
      "Whose leather wallet was red?" → ("leather wallet", "red")

    Returns None for everything else. Phase 3 scope: single-word adjectives only.
    """
    text = _normalize_ws(user_text or "")
    text = _strip_optional_question(text)
    if not text:
        return None
    text_lower = text.lower()

    prefix = None
    for p in ("whose ", "which "):
        if text_lower.startswith(p):
            prefix = p
            break
    if prefix is None:
        return None

    rest = text[len(prefix):]
    was_idx = rest.lower().find(" was ")
    if was_idx < 0:
        return None
    noun_phrase = rest[:was_idx].strip()
    adjective = rest[was_idx + 5:].strip()   # skip " was "

    if not noun_phrase or not adjective:
        return None
    if " " in adjective:
        # Multi-word comparatives are outside v1 — see docs/closed-decisions.md (State-native Phase 4).
        return None
    return noun_phrase, adjective


# Action verbs accepted as evidence of an action event in DID/WHO existence queries.
# Base and common past-tense forms; the evaluator does keyword matching against PEF
# evidence/object text, not NLP lemmatisation.
_EXISTENCE_QUERY_VERBS: frozenset[str] = frozenset({
    "order", "ordered",
    "authorize", "authorized", "authorise", "authorised",
    "approve", "approved",
    "prescribe", "prescribed",
    "reduce", "reduced",
    "change", "changed",
    "modify", "modified",
    "administer", "administered",
    "withhold", "withheld",
    "sign", "signed",
    "give", "gave",
    "take", "took",
    "send", "sent",
    "tell", "told",
    "show", "showed",
    "return", "returned",
    "request", "requested",
    "submit", "submitted",
    "confirm", "confirmed",
    "perform", "performed",
    "conduct", "conducted",
    "complete", "completed",
    "issue", "issued",
    "make", "made",
    "do", "did",
    "initiate", "initiated",
    "cancel", "cancelled", "canceled",
    "revoke", "revoked",
    "update", "updated",
    "record", "recorded",
})

# Words that cannot start the event tail in a DID query (they're articles/particles,
# not action verbs — they're part of the actor phrase instead).
_ARTICLE_PARTICLES: frozenset[str] = frozenset({
    "the", "a", "an", "this", "that", "these", "those",
})


def parse_did_actor_action_query(user_text: str) -> tuple[str, str] | None:
    """Parse ``Did X [event]?`` → ``(actor_phrase, event_tail)`` or None.

    The actor phrase is everything before the first recognised action verb.
    The event tail is the action verb and everything after it.

    Only matches when a known action verb is found.  Returns None if the
    split cannot be determined (e.g. no verb in ``_EXISTENCE_QUERY_VERBS``).
    """
    text = _normalize_ws(user_text or "")
    text = _strip_optional_question(text)
    if not text:
        return None
    tail = _consume_ci_prefix(text, "did ")
    if tail is None:
        return None
    words = tail.split()
    if len(words) < 3:
        return None
    # Scan from position 1 onward (first word is always treated as actor start).
    for i in range(1, len(words)):
        w = words[i].strip(".,!?;:'\"").lower()
        if w in _EXISTENCE_QUERY_VERBS:
            actor = " ".join(words[:i]).strip()
            event = " ".join(words[i:]).strip()
            if actor and event:
                return (actor, event)
    return None


def parse_who_did_action_query(user_text: str) -> str | None:
    """Parse ``Who [verb] [object]?`` → event tail, or None.

    Only matches when the first word after ``who`` is a known action verb
    (present or past tense).  Does NOT match:
    - ``Who has Y?``          (handled by inventory evaluator)
    - ``Who should I ...?``   (handled by action-agent follow-up evaluator)
    - ``Who is X?``           (IS query, not an action)
    """
    text = _normalize_ws(user_text or "")
    text = _strip_optional_question(text)
    if not text:
        return None
    tail = _consume_ci_prefix(text, "who ")
    if tail is None:
        return None
    words = tail.split()
    if not words:
        return None
    first = words[0].strip(".,!?;:'\"").lower()
    if first in _ARTICLE_PARTICLES:
        return None
    if first not in _EXISTENCE_QUERY_VERBS:
        return None
    return tail


def parse_temporal_contact_query(user_text: str) -> str | None:
    """Return the entity phrase for a temporal contact query, or None.

    Matches ``when should I contact X`` and similar. Supports standalone
    queries and last-sentence extraction from multi-sentence messages.
    Returns the entity phrase (e.g. ``Latisha``) or None.
    """
    text = _normalize_ws(user_text or "").lower()
    text = _strip_optional_question(text)
    if not text:
        return None

    def _extract(phrase: str) -> str | None:
        for prefix in _TEMPORAL_CONTACT_PREFIXES:
            if phrase.startswith(prefix):
                entity = phrase[len(prefix):].strip()
                return entity if entity else None
        return None

    result = _extract(text)
    if result is not None:
        return result

    last_boundary = max(text.rfind(". "), text.rfind("! "), text.rfind("? "))
    if last_boundary >= 0:
        last_sentence = _strip_optional_question(text[last_boundary + 2:].strip())
        if last_sentence:
            return _extract(last_sentence)
    return None

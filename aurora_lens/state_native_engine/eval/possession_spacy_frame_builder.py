"""spaCy dependency adapter → :class:`PossessionMutationFrame` (HIGH confidence when tight).

Abstention is normal: unclear parses return ``None`` so arbitration may fall back to regex.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aurora_lens.state_native_engine.contracts.possession_mutation_frame import (
    PossessionMutationAction,
    PossessionMutationConfidence,
    PossessionMutationFrame,
)
from aurora_lens.state_native_engine.eval.inventory import (
    _parse_counted_item_literal,
    normalize_possession_item_surface,
)
from aurora_lens.state_native_engine.lexical import strip_leading_np_determiners

if TYPE_CHECKING:
    from spacy.tokens import Doc, Token


def _strip_np(text: str) -> str:
    return strip_leading_np_determiners((text or "").strip())


def _span_text(tok: Any) -> str:
    """Tokens spanning ``tok`` subtree as whitespace-joined surface."""
    doc = tok.doc
    return doc[tok.left_edge.i : tok.right_edge.i + 1].text.strip()


def _subtree_has_pronoun(tok: Any) -> bool:
    return any(x.pos_ == "PRON" for x in tok.subtree)


def _passive_verbal_root(root: Any) -> bool:
    if root.tag_ == "VBN":
        return True
    return any(c.dep_ == "auxpass" for c in root.children)


def _single_sentence(doc: Any) -> bool:
    return sum(1 for _ in doc.sents) == 1


def _find_into_prep(head: Any) -> Any | None:
    for tok in head.subtree:
        if tok.dep_ == "prep" and tok.lower_ in {"into", "in", "inside"}:
            return tok
    return None


def _find_passive_container_prep(head: Any) -> Any | None:
    """Destination ``prep`` for passive PUT/RETURN (**into** / **in** / **inside** / **to**)."""
    for tok in head.subtree:
        if tok.dep_ == "prep" and tok.lower_ in {"into", "in", "inside", "to"}:
            return tok
    return None


def _passive_actor_token(root: Any) -> Any | None:
    """Agent of passive verb (**by Grace** or UD ``agent`` edge)."""
    for c in root.children:
        if c.dep_ == "agent":
            if c.lower_ == "by":
                return next((x for x in c.children if x.dep_ == "pobj"), None)
            return c
    for tok in root.subtree:
        if tok.dep_ == "prep" and tok.lower_ == "by":
            po = next((x for x in tok.children if x.dep_ == "pobj"), None)
            if po is not None:
                return po
    return None


def try_spacy_put_like_frame_high(nlp: Any, user_text: str) -> PossessionMutationFrame | None:
    """Active PUT / PLACE / RETURN … into/in … with counted ``dobj`` (HIGH only)."""
    original_surface = normalize_possession_item_surface((user_text or "").strip())
    if not original_surface:
        return None
    doc = nlp(original_surface)
    if not _single_sentence(doc):
        return None
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None or root.pos_ != "VERB":
        return None
    lem = root.lemma_.lower()
    if lem in {"give"}:
        return None
    if lem not in {"put", "place", "return"}:
        return None
    if _passive_verbal_root(root):
        return None

    actor_tok = next((c for c in root.children if c.dep_ == "nsubj"), None)
    if actor_tok is None:
        return None

    dest_prep = _find_passive_container_prep(root) if lem == "return" else _find_into_prep(root)
    if dest_prep is None:
        return None
    dest_tok = next((c for c in dest_prep.children if c.dep_ == "pobj"), None)
    if dest_tok is None:
        return None

    dobj = next((c for c in root.children if c.dep_ == "dobj"), None)
    if dobj is None:
        return None

    if (
        _subtree_has_pronoun(actor_tok)
        or _subtree_has_pronoun(dest_tok)
        or _subtree_has_pronoun(dobj)
    ):
        return None

    payload = normalize_possession_item_surface(_span_text(dobj))
    parsed = _parse_counted_item_literal(payload)
    if parsed is None:
        return None
    qty_i, raw_item = parsed
    item_surface = normalize_possession_item_surface(raw_item)

    actor_surface = _strip_np(_span_text(actor_tok))
    dest_surface = _strip_np(_span_text(dest_tok))
    if not actor_surface or not dest_surface or not item_surface:
        return None

    action = PossessionMutationAction.RETURN if lem == "return" else PossessionMutationAction.PUT
    parse_src = (
        "spacy_return_into_container"
        if action == PossessionMutationAction.RETURN
        else "spacy_put_into_container"
    )

    return PossessionMutationFrame(
        action=action,
        actor_surface=actor_surface,
        recipient_surface=None,
        destination_surface=dest_surface,
        quantity=float(qty_i),
        item_surface=item_surface,
        source_container_surface=None,
        original_surface=original_surface,
        voice="active",
        surface_verb=lem,
        parse_source=parse_src,
        confidence=PossessionMutationConfidence.HIGH,
        requires_resolution=False,
    )


def try_spacy_passive_put_like_frame_high(nlp: Any, user_text: str) -> PossessionMutationFrame | None:
    """Passive PUT / PLACE / RETURN … container prep; requires explicit **by** agent."""
    original_surface = normalize_possession_item_surface((user_text or "").strip())
    if not original_surface:
        return None
    doc = nlp(original_surface)
    if not _single_sentence(doc):
        return None
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None or root.pos_ != "VERB":
        return None
    lem = root.lemma_.lower()
    if lem in {"give"}:
        return None
    if lem not in {"put", "place", "return"}:
        return None
    if not _passive_verbal_root(root):
        return None

    patient_tok = next((c for c in root.children if c.dep_ == "nsubjpass"), None)
    actor_tok = _passive_actor_token(root)
    if patient_tok is None or actor_tok is None:
        return None

    dest_prep = _find_passive_container_prep(root)
    if dest_prep is None:
        return None
    dest_tok = next((c for c in dest_prep.children if c.dep_ == "pobj"), None)
    if dest_tok is None:
        return None

    if lem != "return" and dest_prep.lower_ == "to":
        return None

    if (
        _subtree_has_pronoun(patient_tok)
        or _subtree_has_pronoun(dest_tok)
        or _subtree_has_pronoun(actor_tok)
    ):
        return None

    payload = normalize_possession_item_surface(_span_text(patient_tok))
    parsed = _parse_counted_item_literal(payload)
    if parsed is None:
        return None
    qty_i, raw_item = parsed
    item_surface = normalize_possession_item_surface(raw_item)

    actor_surface = _strip_np(_span_text(actor_tok))
    dest_surface = _strip_np(_span_text(dest_tok))
    if not actor_surface or not dest_surface or not item_surface:
        return None

    action = PossessionMutationAction.RETURN if lem == "return" else PossessionMutationAction.PUT
    if action == PossessionMutationAction.RETURN:
        parse_src = "spacy_passive_return_container"
    else:
        parse_src = "spacy_passive_put_into_container"

    return PossessionMutationFrame(
        action=action,
        actor_surface=actor_surface,
        recipient_surface=None,
        destination_surface=dest_surface,
        quantity=float(qty_i),
        item_surface=item_surface,
        source_container_surface=None,
        original_surface=original_surface,
        voice="passive",
        surface_verb=lem,
        parse_source=parse_src,
        confidence=PossessionMutationConfidence.HIGH,
        requires_resolution=False,
    )


def try_spacy_passive_give_or_hand_transfer_high(nlp: Any, user_text: str) -> PossessionMutationFrame | None:
    """Passive **GIVE**/**HAND** with counted ``dobj`` and explicit **by** agent."""
    original_surface = normalize_possession_item_surface((user_text or "").strip())
    if not original_surface:
        return None
    doc = nlp(original_surface)
    if not _single_sentence(doc):
        return None
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None or root.pos_ != "VERB":
        return None
    lem = root.lemma_.lower()
    if lem not in {"give", "hand"}:
        return None
    if not _passive_verbal_root(root):
        return None

    recipient_tok = next((c for c in root.children if c.dep_ == "nsubjpass"), None)
    dobj = next((c for c in root.children if c.dep_ == "dobj"), None)
    actor_tok = _passive_actor_token(root)

    if recipient_tok is None or dobj is None or actor_tok is None:
        return None

    if (
        _subtree_has_pronoun(recipient_tok)
        or _subtree_has_pronoun(dobj)
        or _subtree_has_pronoun(actor_tok)
    ):
        return None

    payload = normalize_possession_item_surface(_span_text(dobj))
    parsed = _parse_counted_item_literal(payload)
    if parsed is None:
        return None
    qty_i, raw_item = parsed
    item_surface = normalize_possession_item_surface(raw_item)

    actor_surface = _strip_np(_span_text(actor_tok))
    recipient_surface = _strip_np(_span_text(recipient_tok))
    if not actor_surface or not recipient_surface or not item_surface:
        return None

    parse_src = "spacy_passive_hand_transfer" if lem == "hand" else "spacy_passive_give_transfer"

    return PossessionMutationFrame(
        action=PossessionMutationAction.GIVE,
        actor_surface=actor_surface,
        recipient_surface=recipient_surface,
        destination_surface=None,
        quantity=float(qty_i),
        item_surface=item_surface,
        source_container_surface=None,
        original_surface=original_surface,
        voice="passive",
        surface_verb=lem,
        parse_source=parse_src,
        confidence=PossessionMutationConfidence.HIGH,
        requires_resolution=False,
    )


def _recipient_token_for_give_verb(root: Any) -> Any | None:
    """Beneficiary token for transfer verbs (aligned with ``SpacyBackend`` give scan)."""
    iobj_tok = next((c for c in root.children if c.dep_ in ("iobj", "dative")), None)
    if iobj_tok is not None:
        if iobj_tok.text.lower() == "to" and iobj_tok.pos_ in ("ADP", "SCONJ"):
            return next((c for c in iobj_tok.children if c.dep_ == "pobj"), None)
        return iobj_tok

    to_prep = next(
        (c for c in root.children if c.dep_ == "prep" and c.text.lower() == "to"),
        None,
    )
    if to_prep is not None:
        return next((c for c in to_prep.children if c.dep_ == "pobj"), None)

    back_tok = next(
        (c for c in root.children if c.text.lower() == "back" and c.dep_ in ("advmod", "prt")),
        None,
    )
    if back_tok is not None:
        to_prep = next(
            (c for c in back_tok.children if c.dep_ == "prep" and c.text.lower() == "to"),
            None,
        )
        if to_prep is not None:
            return next((c for c in to_prep.children if c.dep_ == "pobj"), None)

    dobj_give = next((c for c in root.children if c.dep_ == "dobj"), None)
    if dobj_give is not None:
        for gc in dobj_give.children:
            if gc.dep_ == "nmod" and gc.pos_ == "PROPN":
                return gc
    return None


def try_spacy_give_transfer_high(nlp: Any, user_text: str) -> PossessionMutationFrame | None:
    """Active GIVE/HAND … counted ``dobj`` + beneficiary (HIGH only)."""
    original_surface = normalize_possession_item_surface((user_text or "").strip())
    if not original_surface:
        return None
    doc = nlp(original_surface)
    if not _single_sentence(doc):
        return None
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None or root.pos_ != "VERB":
        return None
    lem = root.lemma_.lower()
    if lem not in {"give", "hand"}:
        return None
    if _passive_verbal_root(root):
        return None

    actor_tok = next((c for c in root.children if c.dep_ == "nsubj"), None)
    dobj = next((c for c in root.children if c.dep_ == "dobj"), None)
    recipient_tok = _recipient_token_for_give_verb(root)

    if actor_tok is None or dobj is None or recipient_tok is None:
        return None

    if (
        _subtree_has_pronoun(actor_tok)
        or _subtree_has_pronoun(dobj)
        or _subtree_has_pronoun(recipient_tok)
    ):
        return None

    payload = normalize_possession_item_surface(_span_text(dobj))
    parsed = _parse_counted_item_literal(payload)
    if parsed is None:
        return None
    qty_i, raw_item = parsed
    item_surface = normalize_possession_item_surface(raw_item)

    actor_surface = _strip_np(_span_text(actor_tok))
    recipient_surface = _strip_np(_span_text(recipient_tok))
    if not actor_surface or not recipient_surface or not item_surface:
        return None

    parse_src = "spacy_hand_transfer" if lem == "hand" else "spacy_give_transfer"

    return PossessionMutationFrame(
        action=PossessionMutationAction.GIVE,
        actor_surface=actor_surface,
        recipient_surface=recipient_surface,
        destination_surface=None,
        quantity=float(qty_i),
        item_surface=item_surface,
        source_container_surface=None,
        original_surface=original_surface,
        voice="active",
        surface_verb=lem,
        parse_source=parse_src,
        confidence=PossessionMutationConfidence.HIGH,
        requires_resolution=False,
    )

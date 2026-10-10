"""Parser-grounded personal finance determination probe (spaCy Doc only).

Separates a requested allocation determination from response-format cues.
Does not treat a successful parse as permission: ``status`` is ``established``
only when personal scope, finance concern, and a cited determination (including
an allocation stay/move choice bound to the user's holdings) each have evidence,
or when a linked allocation act supports the same conclusion.
"""

from __future__ import annotations

from typing import Any

_ALLOCATION_LEMMAS = frozenset(
    {"move", "put", "invest", "allocate", "transfer", "switch", "reallocate"}
)
_ALLOCATION_CHOICE_LEMMAS = frozenset({"move", "stay"})
_CONCERN_LEMMAS = frozenset(
    {"fund", "saving", "retirement", "pension", "portfolio", "pot", "allocation"}
)
_FORMAT_VERBS = frozenset({"return"})
_JUDGMENT_ATTR = frozenset({"call", "choice", "decision"})


def _evidence(tok: Any) -> str:
    return f"{tok.i}:{tok.text}/{tok.lemma_}/{tok.dep_}"


def _is_finance_concern(tok: Any) -> bool:
    low = tok.text.lower()
    if low == "401k":
        return True
    return tok.lemma_.lower() in _CONCERN_LEMMAS


def _allocation_act_evidence(doc: Any) -> list[str]:
    """Allocation verbs with a finance target (into … or direct object)."""
    out: list[str] = []
    for tok in doc:
        if tok.lemma_.lower() in _FORMAT_VERBS:
            continue
        lemma = tok.lemma_.lower()
        if lemma not in _ALLOCATION_LEMMAS and not (
            lemma == "move" and tok.pos_ in ("VERB", "AUX") and tok.tag_.startswith("V")
        ):
            continue
        if tok.pos_ not in ("VERB", "AUX") and tok.tag_ != "VBG":
            continue
        act_bits: list[str] = [_evidence(tok)]
        linked = False
        for child in tok.children:
            if child.dep_ == "dobj" and _is_finance_concern(child):
                act_bits.append(f"dobj→{_evidence(child)}")
                linked = True
            if child.dep_ == "prep" and child.text.lower() in ("into", "to"):
                for pobj in child.children:
                    if pobj.dep_ == "pobj" and (
                        _is_finance_concern(pobj)
                        or any(_is_finance_concern(t) for t in pobj.subtree)
                    ):
                        act_bits.append(f"{child.text.lower()}→{_evidence(pobj)}")
                        linked = True
        if tok.lemma_.lower() == "put":
            for child in tok.children:
                if child.dep_ == "prep" and child.text.lower() == "into":
                    linked = True
        if linked:
            out.extend(act_bits)
    return out


def _personal_evidence(doc: Any, act_tokens: list[Any]) -> list[str]:
    out: list[str] = []
    for tok in doc:
        if tok.text.lower() != "my" and tok.dep_ != "poss":
            continue
        if tok.text.lower() == "my":
            out.append(_evidence(tok))
        elif _is_finance_concern(tok.head):
            out.append(f"poss→{_evidence(tok)}")
    for act in act_tokens:
        for t in act.subtree:
            if t.text.lower() == "my":
                out.append(f"my in act subtree {_evidence(t)}")
    return list(dict.fromkeys(out))


def _concern_evidence(doc: Any) -> list[str]:
    out: list[str] = []
    for tok in doc:
        if _is_finance_concern(tok):
            out.append(_evidence(tok))
    return list(dict.fromkeys(out))


def _determination_evidence(doc: Any) -> list[str]:
    """Decision sought from parse structure, not format tokens or JSON keys."""
    out: list[str] = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        root = sent.root
        if sent_text.endswith("?"):
            if root.pos_ in ("VERB", "AUX") or root.lemma_.lower() == "be":
                out.append(f"interrogative matrix {_evidence(root)}")
        would_aux = any(
            c.dep_ == "aux" and c.lemma_.lower() == "would" for c in root.children
        )
        if root.lemma_.lower() == "be" and would_aux:
            for child in root.children:
                if child.dep_ == "attr" and child.lemma_.lower() in _JUDGMENT_ATTR:
                    out.append(f"would judgment attr {_evidence(child)}")
        if any(c.dep_ == "nsubj" and c.text.lower() == "i" for c in root.children):
            if any(c.dep_ == "aux" and c.lemma_.lower() in ("should", "would") for c in root.children):
                out.append(f"modal subject-I {_evidence(root)}")
    for tok in doc:
        if tok.lemma_.lower() == "stay" and tok.dep_ == "conj":
            head = tok.head
            if head.lemma_.lower() == "move" and head.pos_ == "VERB":
                out.append(f"move/stay disjunction {_evidence(tok)}")
        if tok.lemma_.lower() == "tell" and tok.dep_ == "ROOT":
            for child in tok.children:
                if child.dep_ in ("xcomp", "ccomp") and child.lemma_.lower() == "move":
                    out.append(f"tell xcomp {_evidence(child)}")
    return list(dict.fromkeys(out))


def _allocation_choice_evidence(doc: Any) -> list[str]:
    """Stay/move (or equivalent) alternatives the user wants decided."""
    out: list[str] = []
    for tok in doc:
        if tok.lemma_.lower() != "stay" or tok.dep_ != "conj":
            continue
        head = tok.head
        if head.lemma_.lower() in _ALLOCATION_CHOICE_LEMMAS:
            out.append(f"move/stay disjunction {_evidence(tok)}")
    return list(dict.fromkeys(out))


def _allocation_binding_evidence(doc: Any) -> list[str]:
    """Possessive finance scope the choice is about (e.g. asking about my allocation)."""
    out: list[str] = []
    for tok in doc:
        if tok.lemma_.lower() not in ("ask", "asking") and tok.text.lower() not in (
            "ask",
            "asking",
        ):
            continue
        for child in tok.children:
            if child.dep_ != "prep" or child.text.lower() != "about":
                continue
            for pobj in child.children:
                if pobj.dep_ != "pobj":
                    continue
                if any(t.text.lower() == "my" for t in pobj.subtree) or (
                    _is_finance_concern(pobj)
                    and any(c.dep_ == "poss" and c.text.lower() == "my" for c in pobj.children)
                ):
                    out.append(f"allocation scope {_evidence(pobj)}")
    for tok in doc:
        if not _is_finance_concern(tok):
            continue
        if any(c.dep_ == "poss" and c.text.lower() == "my" for c in tok.children):
            out.append(f"my concern {_evidence(tok)}")
    return list(dict.fromkeys(out))


def _response_format_evidence(doc: Any, text: str) -> list[str]:
    out: list[str] = []
    lower = text.lower()
    if lower.startswith("return ") and "json" in lower:
        out.append("format:return_json")
    if "yes or no only" in lower:
        out.append("format:yes_no_only")
    for tok in doc:
        if tok.lemma_.lower() == "return" and tok.dep_ != "ROOT":
            out.append(_evidence(tok))
    return list(dict.fromkeys(out))


def _matrix_predicate(token: Any) -> Any:
    """Climb to the matrix clause predicate (sentence root or coordinated root)."""
    tok = token
    while tok.dep_ in ("conj", "advcl", "ccomp", "xcomp") and tok.head is not tok:
        tok = tok.head
    return tok


def _allocation_verbs_with_evidence(doc: Any) -> list[Any]:
    """Verb tokens that contributed to ``_allocation_act_evidence``."""
    verbs: list[Any] = []
    seen: set[int] = set()
    for tok in doc:
        if tok.i in seen:
            continue
        if tok.lemma_.lower() in _FORMAT_VERBS:
            continue
        lemma = tok.lemma_.lower()
        if lemma not in _ALLOCATION_LEMMAS and not (
            lemma == "move" and tok.pos_ in ("VERB", "AUX")
        ):
            continue
        if tok.pos_ not in ("VERB", "AUX") and tok.tag_ != "VBG":
            continue
        linked = False
        for child in tok.children:
            if child.dep_ == "dobj" and _is_finance_concern(child):
                linked = True
            if child.dep_ == "prep" and child.text.lower() in ("into", "to"):
                for pobj in child.children:
                    if pobj.dep_ == "pobj" and (
                        _is_finance_concern(pobj)
                        or any(_is_finance_concern(t) for t in pobj.subtree)
                    ):
                        linked = True
        if tok.lemma_.lower() == "put":
            for child in tok.children:
                if child.dep_ == "prep" and child.text.lower() == "into":
                    linked = True
        if linked:
            verbs.append(tok)
            seen.add(tok.i)
    return verbs


def _allocation_act_force(doc: Any, act_verbs: list[Any]) -> tuple[list[str], list[str], list[str]]:
    """Classify linked allocation verbs as instruction, reported fact, or uncertain."""
    instruction: list[str] = []
    reported: list[str] = []
    uncertain: list[str] = []
    for verb in act_verbs:
        matrix = _matrix_predicate(verb)
        nsubj = next((c for c in verb.children if c.dep_ in ("nsubj", "nsubjpass")), None)
        if nsubj is not None and nsubj.text.lower() in ("i", "we"):
            if verb.tag_ == "VBD" or verb.morph.get("Tense") == ["Past"]:
                reported.append(f"past user report {_evidence(verb)}")
                continue
        if nsubj is None and verb.tag_ == "VB" and any(
            t.text.lower() == "my" for t in verb.subtree
        ):
            instruction.append(f"imperative allocation {_evidence(verb)}")
            continue
        if matrix.dep_ == "ROOT" and matrix.tag_ == "VB" and nsubj is None:
            instruction.append(f"imperative matrix {_evidence(matrix)}")
            continue
        if matrix.dep_ == "ROOT" and nsubj is not None and nsubj.text.lower() in ("i", "we"):
            if verb.tag_ == "VBD":
                reported.append(f"past user report {_evidence(verb)}")
                continue
        uncertain.append(f"act force unclear {_evidence(verb)}")
    return instruction, reported, uncertain


def _act_root_tokens(doc: Any) -> list[Any]:
    roots: list[Any] = []
    for tok in doc:
        if tok.lemma_.lower() in _FORMAT_VERBS:
            continue
        if tok.lemma_.lower() in _ALLOCATION_LEMMAS or (
            tok.lemma_.lower() == "move" and tok.tag_.startswith("V")
        ):
            if tok.pos_ in ("VERB", "AUX") or tok.tag_ == "VBG":
                roots.append(tok)
    return roots


def probe_financial_determination(doc: Any, text: str) -> dict[str, Any]:
    act = _allocation_act_evidence(doc)
    act_verbs = _allocation_verbs_with_evidence(doc)
    act_tokens = _act_root_tokens(doc)
    personal = _personal_evidence(doc, act_tokens)
    concern = _concern_evidence(doc)
    determination = _determination_evidence(doc)
    choice = _allocation_choice_evidence(doc)
    allocation_binding = _allocation_binding_evidence(doc)
    response_format = _response_format_evidence(doc, text)
    instruction: list[str] = []
    reported: list[str] = []
    uncertain: list[str] = []
    if act_verbs:
        instruction, reported, uncertain = _allocation_act_force(doc, act_verbs)

    has_act = bool(act)
    has_personal = bool(personal)
    has_concern = bool(concern)
    has_determination = bool(determination)
    has_choice = bool(choice)
    has_binding = bool(allocation_binding)

    finance_surface = has_concern or "401k" in text.lower()
    personal_surface = has_personal or " my " in f" {text.lower()} "

    status = "none"
    if has_act and has_personal and has_concern:
        if has_determination or instruction:
            status = "established"
        elif reported and not instruction and not uncertain:
            status = "none"
        elif reported and uncertain:
            status = "partial"
        elif uncertain:
            status = "partial"
    elif (
        has_personal
        and has_concern
        and has_choice
        and (has_binding or has_determination)
    ):
        status = "established"
    elif finance_surface and personal_surface and (
        has_determination or response_format or has_choice or not has_act
    ):
        status = "partial"

    return {
        "status": status,
        "personal": personal,
        "concern": concern,
        "act": act,
        "determination": determination,
        "choice": choice,
        "allocation_binding": allocation_binding,
        "instruction": instruction,
        "reported": reported,
        "uncertain": uncertain,
        "response_format": response_format,
    }

"""Regex-free surfaces: covert harm, cyber abuse, and operational wrongdoing requests."""

from __future__ import annotations

# Audit/trace vocabulary plus detection-style concealment (e.g. "without detection").
_FINANCIAL_CONCEALMENT_TARGET_TERMS = frozenset(
    {
        "audit",
        "trail",
        "record",
        "records",
        "trace",
        "traces",
        "detection",
        "detect",
        "detected",
        "detecting",
        "discovered",
    }
)


def _has_financial_concealment_target(tokens: frozenset[str]) -> bool:
    if not _FINANCIAL_CONCEALMENT_TARGET_TERMS.isdisjoint(tokens):
        return True
    return "off" in tokens and "books" in tokens


def _has_localized_explicit_illicit_financial_objective(
    tok: tuple[str, ...],
    *,
    financial_terms: set[str],
    intent_terms: set[str],
    explicit_illicit_terms: set[str],
) -> bool:
    """Direct laundering/embezzle asks with finance context and user intent in one locale."""

    n = len(tok)
    if n == 0:
        return False

    for i, word in enumerate(tok):
        if word not in explicit_illicit_terms:
            continue
        lo = max(0, i - 8)
        hi = min(n, i + 9)
        win_set = set(tok[lo:hi])
        if financial_terms.isdisjoint(win_set):
            continue
        if not intent_terms.isdisjoint(win_set):
            return True
        if i <= 3 and not {"i", "my", "me"}.isdisjoint(win_set):
            return True
    return False


def surface_harmful_action_protective(norm: str, tokens: frozenset[str]) -> bool:
    """Educational, defensive, or lawful-framing carve-outs (not operational harm asks)."""
    w = f" {norm} "
    protective_phrases = (
        " at a high level ",
        " high level ",
        " how companies can prevent ",
        " how to prevent ",
        " how organisations can prevent ",
        " how organizations can prevent ",
        " companies can prevent ",
        " prevent it ",
        " prevention ",
        " lawful whistleblow",
        " chooses a lawful ",
        " choose a lawful ",
        " lawful path ",
        " whistleblowing path ",
        " report safely ",
        " is dangerous ",
        " are dangerous ",
        " warning signs ",
        " overview of ",
        " explain why ",
        " risks of ",
        " dangers of ",
    )
    if any(p in w for p in protective_phrases):
        return True
    if "lawful" in tokens and ("whistleblow" in norm or "path" in tokens):
        return True
    if "prevent" in tokens or "prevention" in tokens:
        if "how" in tokens and ("companies" in tokens or "organizations" in tokens or "organisations" in tokens):
            return True
    return False


def surface_operational_wrongdoing_framing(norm: str, tokens: frozenset[str]) -> bool:
    """Request seeks actionable steps, approaches, or ideas for wrongdoing."""
    w = f" {norm} "
    operational_phrases = (
        " how can i ",
        " how could i ",
        " how would i ",
        " how do i ",
        " how can someone ",
        " how could someone ",
        " how would someone ",
        " how would an ",
        " how could an ",
        " how would a ",
        " how could a ",
        " break down ",
        " potential approach ",
        " approach of ",
        " give me a ",
        " give me an ",
        " give me ",
        " ways to ",
        " step by step ",
        " without getting caught ",
        " without being traced ",
        " without them knowing ",
        " without him knowing ",
        " without her knowing ",
        " without anyone knowing ",
        " without getting found ",
    )
    if any(p in w for p in operational_phrases):
        return True
    if "break" in tokens and "down" in tokens:
        return True
    if "idea" in tokens and not {"malware", "revenge", "backdoor", "blackmail"}.isdisjoint(tokens):
        return True
    if "craft" in tokens and "plot" in tokens and ("approach" in tokens or ("break" in tokens and "down" in tokens)):
        return True
    if "create" in tokens and "backdoor" in tokens:
        return True
    if "steal" in tokens and "data" in tokens:
        return True
    return False


def surface_cyber_abuse_objective(norm: str, tokens: frozenset[str]) -> bool:
    """Malware, backdoors, credential/data theft, vulnerability exploitation."""
    w = f" {norm} "
    if "malware" in tokens or "ransomware" in tokens:
        return True
    if "backdoor" in tokens or "backdoors" in tokens or "back door" in w:
        return True
    if ("steal" in tokens or "theft" in tokens or "stealing" in tokens) and not {
        "data",
        "credential",
        "credentials",
        "sensitive",
    }.isdisjoint(tokens):
        return True
    if "steal company data" in w or "steal sensitive data" in w:
        return True
    if "credential theft" in w:
        return True
    if "exploit" in tokens and not {"vulnerability", "vulnerabilities"}.isdisjoint(tokens):
        return True
    if "freeze" in tokens and not {"computer", "computers", "system", "systems"}.isdisjoint(tokens):
        return True
    return False


def surface_covert_harm_objective(norm: str, tokens: frozenset[str]) -> bool:
    """Revenge, sabotage, or concealment-framed harm toward a person or organisation."""
    w = f" {norm} "
    if "revenge" in tokens or "retaliation" in tokens or "retaliate" in tokens:
        return True
    if "sabotage" in tokens or "sabotaging" in tokens:
        return True
    if "secretly" in tokens or "covert" in tokens or "covertly" in tokens:
        return True
    if "without getting caught" in w or "without being traced" in w or "without getting found" in w:
        return True
    if "without" in tokens and "knowing" in tokens:
        return True
    if "operational disruption" in w:
        return True
    return False


def surface_harmful_action_objective(norm: str, tokens: frozenset[str]) -> bool:
    """Blackmail, extortion, reputational attack, or planting evidence."""
    w = f" {norm} "
    if "blackmail" in tokens or "extort" in tokens or "extortion" in tokens:
        return True
    if "plant evidence" in w or ("plant" in tokens and "evidence" in tokens):
        return True
    if "reputational destruction" in w or "destroy reputation" in w or "destroy his reputation" in w:
        return True
    if "damage" in tokens and ("reputation" in tokens or "reputational" in tokens):
        return True
    if "reputational sabotage" in w:
        return True
    return False


def surface_financial_crime_evasion(norm: str, tokens: frozenset[str]) -> bool:
    """Illicit financial diversion frame: finance context + evasion intent + audit/record concealment."""
    tok = tuple(norm.split())
    if not tok:
        return False

    financial_terms = {
        "escrow",
        "funds",
        "money",
        "payment",
        "payments",
        "transaction",
        "transactions",
        "transfer",
        "account",
        "bank",
        "client",
        "ledger",
        "books",
        "accounting",
        "workflow",
        "offshore",
    }
    movement_terms = {"move", "transfer", "send", "route", "shift", "withdraw", "wire", "funnel", "park", "offshore"}
    evasion_terms = {
        "avoid",
        "avoids",
        "avoiding",
        "hide",
        "hides",
        "hiding",
        "conceal",
        "conceals",
        "concealing",
        "without",
        "no",
        "untraceable",
    }
    explicit_illicit_terms = {"launder", "laundering", "embezzle", "embezzlement", "divert"}
    audit_record_terms = set(_FINANCIAL_CONCEALMENT_TARGET_TERMS)
    operational_terms = {"workflow", "payment", "payments", "transaction", "transactions"}
    intent_terms = {
        "how",
        "ways",
        "way",
        "steps",
        "step",
        "approach",
        "strategy",
        "guide",
        "advice",
        "best",
        "start",
        "want",
        "show",
        "tell",
        "give",
        "suggest",
        "can",
        "could",
        "would",
        "should",
        "help",
        "design",
        "set",
        "setup",
    }
    directive_verbs = movement_terms | explicit_illicit_terms | {"hide", "conceal", "avoid", "bypass"}

    if financial_terms.isdisjoint(tokens):
        return False

    if not explicit_illicit_terms.isdisjoint(tokens):
        if _has_localized_explicit_illicit_financial_objective(
            tok,
            financial_terms=financial_terms,
            intent_terms=intent_terms,
            explicit_illicit_terms=explicit_illicit_terms,
        ):
            return True

    has_evasion_intent = not evasion_terms.isdisjoint(tokens)
    if not (has_evasion_intent and _has_financial_concealment_target(tokens)):
        return False

    has_action_objective = (
        surface_operational_wrongdoing_framing(norm, tokens)
        or _has_localized_financial_evasion_objective(
            tok,
            financial_terms=financial_terms,
            intent_terms=intent_terms,
            directive_verbs=directive_verbs,
            evasion_terms=evasion_terms,
            audit_record_terms=audit_record_terms,
        )
        or _has_split_clause_financial_evasion_objective(
            tok,
            financial_terms=financial_terms,
            intent_terms=intent_terms,
            movement_terms=movement_terms,
            evasion_terms=evasion_terms,
            audit_record_terms=audit_record_terms,
        )
    )
    if not has_action_objective:
        return False

    if not explicit_illicit_terms.isdisjoint(tokens):
        return True
    if not movement_terms.isdisjoint(tokens):
        return True
    if not operational_terms.isdisjoint(tokens):
        return True
    if "client" in tokens and not {"funds", "money"}.isdisjoint(tokens):
        return True
    return False


def surface_audit_trail_evasion(norm: str, tokens: frozenset[str]) -> bool:
    """Requests to evade financial auditability, records, or traceability."""
    tok = tuple(norm.split())
    if not tok:
        return False

    financial_terms = {
        "escrow",
        "funds",
        "money",
        "payment",
        "payments",
        "transaction",
        "transactions",
        "transfer",
        "account",
        "bank",
        "client",
        "ledger",
        "books",
        "accounting",
        "workflow",
        "offshore",
    }

    evasion_terms = {
        "avoid",
        "avoids",
        "avoiding",
        "hide",
        "hides",
        "hiding",
        "conceal",
        "conceals",
        "concealing",
        "without",
        "no",
        "untraceable",
    }
    intent_terms = {
        "how",
        "ways",
        "way",
        "steps",
        "step",
        "approach",
        "strategy",
        "guide",
        "advice",
        "best",
        "start",
        "want",
        "show",
        "tell",
        "give",
        "suggest",
        "can",
        "could",
        "would",
        "should",
        "help",
        "design",
        "set",
        "setup",
    }
    directive_verbs = {"hide", "conceal", "avoid", "bypass", "erase", "delete", "remove", "leave"}
    audit_record_terms = set(_FINANCIAL_CONCEALMENT_TARGET_TERMS)
    movement_terms = {"move", "transfer", "send", "route", "shift", "withdraw", "wire", "funnel", "park", "offshore"}

    has_financial_context = not financial_terms.isdisjoint(tokens)
    if not has_financial_context:
        return False

    has_evasion_intent = not evasion_terms.isdisjoint(tokens)
    if not has_evasion_intent:
        return False

    has_action_objective = (
        surface_operational_wrongdoing_framing(norm, tokens)
        or _has_localized_financial_evasion_objective(
            tok,
            financial_terms=financial_terms,
            intent_terms=intent_terms,
            directive_verbs=directive_verbs | movement_terms,
            evasion_terms=evasion_terms,
            audit_record_terms=audit_record_terms,
        )
        or _has_split_clause_financial_evasion_objective(
            tok,
            financial_terms=financial_terms,
            intent_terms=intent_terms,
            movement_terms=movement_terms,
            evasion_terms=evasion_terms,
            audit_record_terms=audit_record_terms,
        )
    )
    if not has_action_objective:
        return False

    # Audit-trail evasion: "avoid creating an audit trail", "no audit trail",
    # "without an audit trail", etc.
    if "audit" in tokens and "trail" in tokens:
        return True
    if "audit" in tokens and not {"avoid", "avoids", "avoiding", "without", "no"}.isdisjoint(tokens):
        return True

    # Record/trace evasion: "leave no record", "leave no trace",
    # "without a trace", etc.
    if not {"record", "records", "trace", "traces"}.isdisjoint(tokens):
        return True

    # Accounting concealment: "off the books".
    if "off" in tokens and "books" in tokens:
        return True

    return False


def _has_localized_financial_evasion_objective(
    tok: tuple[str, ...],
    *,
    financial_terms: set[str],
    intent_terms: set[str],
    directive_verbs: set[str],
    evasion_terms: set[str],
    audit_record_terms: set[str],
) -> bool:
    """Require local linkage between intent, action, finance context, and concealment target.

    This blocks broad cross-document token collisions where unrelated evidence fragments
    happen to contain words like "payment", "no", and "record".
    """

    n = len(tok)
    if n == 0:
        return False

    for i, word in enumerate(tok):
        if word not in directive_verbs and word not in evasion_terms:
            continue
        lo = max(0, i - 8)
        hi = min(n, i + 9)
        win = tok[lo:hi]
        win_set = set(win)
        if financial_terms.isdisjoint(win_set):
            continue
        if audit_record_terms.isdisjoint(win_set) and not ("off" in win_set and "books" in win_set):
            continue
        if not intent_terms.isdisjoint(win_set):
            return True
        if i <= 2 and word in directive_verbs:
            return True
    return False


def _has_split_clause_financial_evasion_objective(
    tok: tuple[str, ...],
    *,
    financial_terms: set[str],
    intent_terms: set[str],
    movement_terms: set[str],
    evasion_terms: set[str],
    audit_record_terms: set[str],
) -> bool:
    """Allow two-clause asks where movement and concealment appear in separate clauses.

    This captures prompts like "I want to move funds ... no audit trail ... what's my
    best way to start?" while still requiring explicit user-intent framing and
    localized movement + concealment structure.
    """

    if not tok:
        return False
    tok_set = set(tok)
    if {"i", "my", "me"}.isdisjoint(tok_set):
        return False
    if intent_terms.isdisjoint(tok_set):
        return False

    def _window_has(idx: int, vocab: set[str], radius: int) -> bool:
        lo = max(0, idx - radius)
        hi = min(len(tok), idx + radius + 1)
        return not vocab.isdisjoint(set(tok[lo:hi]))

    has_movement_clause = any(
        word in movement_terms and _window_has(i, financial_terms, radius=4)
        for i, word in enumerate(tok)
    )
    if not has_movement_clause:
        return False

    has_concealment_clause = any(
        word in evasion_terms and _window_has(i, audit_record_terms, radius=4)
        for i, word in enumerate(tok)
    ) or ("off" in tok_set and "books" in tok_set and not evasion_terms.isdisjoint(tok_set))
    return has_concealment_clause

"""Regex-free surfaces: legal/finance/delegated/credential/historical lookups."""

from __future__ import annotations

from aurora_lens.verify.blocked_request_normalize import normalise_request_text


def collapsed_lower(user_input: str) -> str:
    s = user_input.lower().replace("\u2019", "'")
    return " ".join(s.split())


def surface_credential_access_bypass(norm: str, tokens: frozenset[str]) -> bool:
    verbs = frozenset(
        {
            "bypass",
            "circumvent",
            "defeat",
            "crack",
            "evade",
            "override",
            "subvert",
        }
    )
    w = f" {norm} "
    verb_hit = not verbs.isdisjoint(tokens) or " get around " in w or " work around " in w
    verb_hit = verb_hit or " break into " in w or " get past " in w
    noun_hit = (
        ("login" in norm.replace("-", " ") or "log in" in norm.replace("-", " "))
        or ("password" in tokens or "passphrase" in tokens or "credential" in tokens)
        or ("authentication" in tokens)
        or ("access" in tokens and ("control" in tokens or any(t.startswith("control") for t in tokens)))
        or ("security" in tokens and "system" in tokens)
        or ("two" in tokens and "factor" in tokens)
        or "2fa" in tokens
        or "mfa" in tokens
        or ("account" in tokens and "lock" in tokens)
        or ("single" in tokens and "sign" in tokens)
        or "sso" in tokens
    )
    return verb_hit and noun_hit


def surface_case_paraphrase(norm: str, tokens: frozenset[str]) -> bool:
    w = f" {norm} "
    guidance = (
        " help me " in w
        or " assist me " in w
        or " advise me " in w
        or " guide me " in w
        or " tell me " in w
        or " let me know " in w
        or " i need to know " in w
    )
    connector = "whether" in tokens or "if" in tokens
    legal_act = (
        ("file" in tokens)
        or ("settle" in tokens)
        or ("contest" in tokens)
        or ("appeal" in tokens)
        or ("pursue" in tokens)
        or ("withdraw" in tokens)
        or ("proceed" in tokens)
        or (" sue " in w)
        or " take legal action " in w
        or " take action " in w
        or " bring a claim " in w
        or " bring claim " in w
    )
    return guidance and connector and legal_act


def surface_legal_outcome_prediction_signal(_collapsed: str, norm: str) -> bool:
    """Token-window detector for legal outcome prediction signals."""
    tokens = norm.split()

    if len(tokens) < 2:
        return False

    # User-defined token sets
    # Outcome verbs: win, lose, succeed, prevail, appeal, overturn, strong, viable, solid, good, weak
    # Modal/probability: would, will, could, might, likely, probably, chances, is, am, are, can, shall, ll
    # Subject: i, my, me, my case, my claim
    verb_roots = ("win", "lose", "succeed", "prevail", "appeal", "overturn", "strong", "viable", "solid", "good", "weak")
    modals = {"would", "will", "could", "might", "likely", "probably", "chances", "is", "am", "are", "can", "shall", "ll", "do", "does", "did"}
    subjects = {"i", "me", "my"}

    def is_verb(t: str) -> bool:
        return any(t.startswith(root) for root in verb_roots)

    for i in range(len(tokens)):
        t = tokens[i]

        is_v = is_verb(t)
        is_m = t in modals
        is_s = t in subjects or (t == "my" and i + 1 < len(tokens) and tokens[i+1] in {"case", "claim"})

        if is_v or is_m or is_s:
            v_found = is_v
            m_found = is_m
            s_found = is_s

            # Search a small window (7 tokens forward) for the remaining components
            for j in range(i + 1, min(i + 8, len(tokens))):
                tj = tokens[j]
                if not v_found and is_verb(tj):
                    v_found = True
                if not m_found and tj in modals:
                    m_found = True
                if not s_found and (tj in subjects or (tj == "my" and j + 1 < len(tokens) and tokens[j+1] in {"case", "claim"})):
                    s_found = True

                if v_found and m_found and s_found:
                    return True

    # Fallback for explicit prediction/forecast keywords
    w = f" {norm} "
    return bool(
        ("prediction" in w or "forecast" in w or "predict" in w)
        and ("court" in w or "legal" in w or "trial" in w)
        and ("decision" in w or "outcome" in w or "ruling" in w)
    )


def surface_legal_outcome_request_context(_collapsed: str, norm: str) -> bool:
    n = f" {norm} "
    cues = (
        " unfair dismissal ",
        " wrongful dismissal ",
        " constructive dismissal ",
        " employment tribunal ",
        " tribunal ",
        " lawsuit ",
        " legal claim ",
        " court case ",
        " appeal ",
        " custody ",
        " redundancy claim ",
        " eviction case ",
        " discrimination claim ",
        " personal injury claim ",
        " small claims ",
        " fired ",
        " dismissed ",
        " evicted ",
        " made redundant ",
        " landlord ",
        " tenant ",
        " tenancy ",
        " notice to quit ",
        " notice to vacate ",
        " eviction notice ",
        " rent arrears ",
        " unpaid rent ",
        " section 8 ",
        " section 21 ",
        "section 8",
        "section 21",
        " court ",
        " trial ",
        " litigation ",
        " judgment ",
        " verdict ",
        " claim ",
        " my case ",
    )
    return any(c in n or c in norm for c in cues)


def surface_legal_context_q_text(text: str) -> bool:
    """Strict legal-context extractor for PEF entity / alias scans."""
    collapsed = collapsed_lower(text)
    norm = normalise_request_text(text)
    w = f" {collapsed} "
    ntok = f" {norm} "
    cues = (
        " unfair dismissal ",
        " wrongful dismissal ",
        " constructive dismissal ",
        " employment tribunal ",
        " tribunal ",
        " lawsuit ",
        " legal claim ",
        " court case ",
        " appeal ",
        " in court ",
        " trial ",
        " litigation ",
        " judgment ",
        " verdict ",
        " redundancy claim ",
        " eviction case ",
        " discrimination claim ",
        " personal injury claim ",
        " small claims ",
        " fired ",
        " dismissed ",
        " evicted ",
        " made redundant ",
        " landlord ",
        " tenant ",
        " tenancy ",
        " notice to quit ",
        " notice to vacate ",
        " eviction notice ",
        " rent arrears ",
        " unpaid rent ",
        " section 8 ",
        " section 21 ",
    )
    return any(c in w or c.replace(" ", "") in ntok.replace(" ", "") for c in cues)


def surface_should_she_he_they(norm: str) -> bool:
    w = f" {norm} "
    return (
        " should she " in w
        or " should he " in w
        or " should they " in w
    )


def surface_should_i(norm: str) -> bool:
    return " should i " in f" {norm} "


def surface_is_party_correct(norm: str, tokens: tuple[str, ...]) -> bool:
    for i in range(len(tokens) - 2):
        if tokens[i] in ("is", "are") and tokens[i + 1] in ("she", "he", "they", "it") and tokens[i + 2] == "correct":
            return True
    return False


def surface_is_treatment_better(norm: str) -> bool:
    return " is the treatment better " in f" {norm} "


def surface_med_delegated_anchor(norm: str, tokens: frozenset[str]) -> bool:
    w = f" {norm} "
    return (
        "insulin" in tokens
        or "clinician" in tokens
        or ("dose" in tokens or "doses" in tokens)
        or "my chart" in norm.replace("-", " ")
        or " reviewed my chart " in w
        or "double my" in norm
        or ("my" in tokens and ("medication" in tokens or "prescription" in tokens or "treatment" in tokens))
        or ("care" in tokens and "plan" in tokens)
    )


def surface_fin_delegated_anchor(norm: str, tokens: frozenset[str]) -> bool:
    w = f" {norm} "
    return (
        "retirement" in tokens
        or " savings " in w
        or " pension " in w
        or " portfolio " in w
        or " investments " in w
        or " investment " in w
        or " assets " in w
        or " asset " in w
        or " account " in w
        or " all my " in w
        or " into one fund " in w
        or " into a fund " in w
    )


def surface_legal_should_be_charged(norm: str) -> bool:
    return " should she be charged " in f" {norm} " or " should he be charged " in f" {norm} " or " should they be charged " in f" {norm} "


def surface_deny_and_claim(tokens: frozenset[str]) -> bool:
    return "deny" in tokens and "claim" in tokens


def surface_legal_correctness_context(tokens: frozenset[str]) -> bool:
    return not frozenset({"policy", "void", "claim", "contract", "applicant", "witness"}).isdisjoint(tokens)


def surface_case_resolution_decision(norm: str, tokens: tuple[str, ...]) -> bool:
    w = f" {norm} "
    if " decide whether " in w:
        return True
    if " should i " in w:
        return True
    if " is the treatment better " in w:
        return True
    if "whether" in tokens and "valid" in tokens:
        try:
            iw = tokens.index("whether")
            iv = tokens.index("valid")
            if iw < iv <= iw + 28:
                return True
        except ValueError:
            pass
    return surface_is_party_correct(norm, tokens)


def surface_case_resolution_action(norm: str, tokens: frozenset[str]) -> bool:
    w = f" {norm} "
    fund_into = " into " in w and " fund " in w
    return (
        " what to file " in w
        or " what should i do " in w
        or " tell me what to do " in w
        or " tell me what to file " in w
        or " file tomorrow " in w
        or " file today " in w
        or " file next week " in w
        or " deny the claim " in w
        or surface_legal_should_be_charged(norm)
        or " put all " in w
        or " tell me my dose " in w
        or " my dose for tonight " in w
        or fund_into
        or ("complaint" in tokens)
        or ("motion" in tokens)
        or ("counterclaim" in tokens)
        or ("petition" in tokens)
    )


def surface_req_legal_decision_or_filing(norm: str, tokens: tuple[str, ...]) -> bool:
    w = f" {norm} "
    if " what to file " in w or " tell me what to file " in w:
        return True
    if " file tomorrow " in w or " file today " in w or " file next week " in w:
        return True
    if "whether" in tokens and "valid" in tokens:
        try:
            iw = tokens.index("whether")
            iv = tokens.index("valid")
            if iw < iv <= iw + 28:
                return True
        except ValueError:
            pass
    for i in range(len(tokens) - 3):
        if tokens[i] in ("is", "are") and tokens[i + 2] == "valid" and tokens[i + 1] in (
            "it",
            "this",
            "that",
        ):
            return True
    if "complaint" in tokens or "motion" in tokens or "counterclaim" in tokens or "petition" in tokens:
        return True
    return False


def surface_case_resolution_dose_timing(norm: str) -> bool:
    w = f" {norm} "
    return (
        (" tonight " in w or " today " in w)
        and (" dose " in w or " dosage " in w or " dosing " in w or " insulin " in w)
    )


def surface_real_world_legal_anchor(norm: str, tokens: tuple[str, ...]) -> bool:
    w = f" {norm} "
    return (
        " my real eviction notice " in w
        or " notice to quit " in w
        or "summons" in tokens
        or " subpoena " in w
        or " court order " in w
        or " my eviction notice " in w
    )


def surface_case_binding_medical(norm: str, tokens: frozenset[str]) -> bool:
    w = f" {norm} "
    return (
        "double my" in norm
        or ("my" in tokens and "chart" in tokens)
        or ("my" in tokens and "insulin" in tokens)
        or " my real chest " in w
        or " my real pain " in w
        or " my real symptoms " in w
        or " my real symptom " in w
        or " my real condition " in w
    )


def surface_case_binding_financial(norm: str, tokens: frozenset[str]) -> bool:
    w = f" {norm} "
    return (
        " my retirement " in w
        or " retirement savings " in w
        or ("my" in tokens and "savings" in tokens)
        or ("my" in tokens and "pension" in tokens)
        or ("my" in tokens and "portfolio" in tokens)
        or " all my " in w
    )


def surface_bare_adjudication_merits(norm: str, tokens: tuple[str, ...]) -> bool:
    """Token-window detector for bare adjudication merits (who should win)."""
    n = len(tokens)
    if n < 2:
        return False
    
    # Looking for: [who/which] + [should/will/is] + [win/prevail/right/custody]
    # e.g. "who should win", "which side is right", "who gets custody"
    starters = {"who", "which"}
    verbs = {"win", "prevail", "right", "custody", "kids", "correct"}
    
    for i in range(n):
        if tokens[i] in starters:
            # Check a small window (5 tokens forward) for merit verbs
            for j in range(i + 1, min(i + 6, n)):
                if tokens[j] in verbs:
                    return True
    return False


def surface_bare_adjudication_legal_domain(norm: str, tokens: frozenset[str]) -> bool:
    w = f" {norm} "
    return (
        " custody case " in w
        or " custody dispute " in w
        or " custody hearing " in w
        or " custody matter " in w
        or " custody proceeding " in w
        or " custody battle " in w
        or " family law case " in w
        or " divorce case " in w
        or " civil case " in w
        or " criminal case " in w
        or " lawsuit " in w
        or " court case " in w
        or " legal dispute " in w
        or " this case " in w
        or " that case " in w
        or " the trial " in w
        or ("child" in tokens and "custody" in tokens and ("case" in tokens or "dispute" in tokens))
        or ("parenting" in tokens and "time" in tokens)
    )


def surface_historical_market_data_lookup(collapsed: str) -> bool:
    w = f" {collapsed} "
    return (
        " historical price " in w
        or " historical quote " in w
        or " historical quotes " in w
        or " historical closing " in w
        or " closing price " in w
        or " opening price " in w
        or " stock price " in w
        or " share price " in w
        or " closing quote " in w
        or " opening quote " in w
        or " price of " in w
        or " price for " in w
        or (" ticker " in w and " on " in w)
        or (" symbol " in w and " as of " in w)
    )


def surface_finance_informational_leading(lower_strip: str) -> bool:
    s = lower_strip.strip()
    return (
        s.startswith("what is ")
        or s.startswith("what are ")
        or s.startswith("how does ")
        or s.startswith("explain ")
        or s.startswith("define ")
        or s == "explain"
        or s.startswith("explain,")
    )


def surface_finance_informational_override(norm: str, tokens: frozenset[str]) -> bool:
    w = f" {norm} "
    return (
        " should i " in w
        or " should we " in w
        or " do i move " in w
        or any(t.startswith("reallocat") for t in tokens)
        or any(t.startswith("rebalanc") for t in tokens)
        or " invest in " in w
        or " advise me whether " in w
        or " whether to execute " in w
        or " execute now " in w
        or " tell me whether " in w
    )


def surface_finance_imperative_first_line(norm_strip: str) -> bool:
    s = norm_strip.strip().lstrip()
    if s.startswith("please "):
        s = s[7:].lstrip()
    if not s:
        return False
    first = s.split()[0]
    return first in {
        "reallocate",
        "rebalance",
        "sell",
        "buy",
        "move",
        "switch",
        "transfer",
        "withdraw",
    }

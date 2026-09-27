"""Regex-free surfaces: enterprise compliance (trade secrets, insider info, procurement fraud)."""

from __future__ import annotations


def surface_trade_secret_disclosure(norm: str, tokens: frozenset[str]) -> bool:
    """Token / phrase cues for trade secret disclosure requests."""
    w = f" {norm} "
    
    # Proprietary cues
    if " proprietary " in w or " confidential " in w or " trade secret " in w or " nda " in w:
        if any(x in w for x in (" share ", " reveal ", " disclose ", " provide ", " send ", " export ", " leak ", " algorithm ", " formula ", " documentation ")):
            return True
            
    return False


def surface_insider_information(norm: str, tokens: frozenset[str]) -> bool:
    """Token / phrase cues for insider information requests."""
    w = f" {norm} "
    
    # Insider trading cues
    insider_cues = {" insider ", " non-public ", " non public ", " advance information ", " private information "}
    if any(x in w for x in insider_cues):
        if any(x in w for x in (" trade ", " trading ", " buy ", " sell ", " invest ", " profit ", " position ")):
            return True

    return False


def surface_procurement_fraud(norm: str, tokens: frozenset[str]) -> bool:
    """Token / phrase cues for procurement fraud / bid rigging requests."""
    w = f" {norm} "
    
    # Bid rigging
    if any(x in w for x in (" bid rigging ", " bid-rigging ", " bid fix ", " bid-fix ", " coordinate bid ")):
        return True
    
    # Conflict of interest / kickback concealment
    if " kickback " in w or " conflict of interest " in w:
        if any(x in w for x in (" conceal ", " hide ", " avoid disclosure ", " suppress ")):
            return True

    # Insider tip in procurement
    if " insider " in w and (" tip " in w or " knowledge " in w):
        if any(x in w for x in (" tender ", " rfp ", " bid ", " procurement ")):
            return True

    return False


def surface_business_strategy_named_company(user_input: str) -> bool:
    """Detects capitalized named companies in original input."""
    words = user_input.split()
    for t in words:
        if len(t) > 1 and t[0].isupper() and t[1:].islower():
            if t not in {"I", "The", "This", "That", "We", "They", "It"}:
                return True
    return False


def surface_business_strategy_directive(norm: str, tokens: frozenset[str]) -> bool:
    """Detects strategic directives."""
    w = f" {norm} "
    directives = {"should", "must", "recommend", "advise"}
    if not tokens.isdisjoint(directives):
        return True
    if " strategy for " in w:
        return True
    if " how should " in w or " how can " in w or " how do " in w:
        if any(x in w for x in (" win ", " beat ", " outcompete ", " grow ", " expand ", " pivot ")):
            return True
    return False


def surface_business_strategy_vocab(tokens: frozenset[str]) -> bool:
    """Detects M&A and strategy vocabulary."""
    single_word_vocab = {"merger", "acquisition", "m&a", "ipo"}
    if not tokens.isdisjoint(single_word_vocab):
        return True
    return False


def surface_business_strategy_full(norm: str, tokens: frozenset[str], user_input: str) -> bool:
    """Combined check for business strategy."""
    w = f" {norm} "
    has_named = surface_business_strategy_named_company(user_input)
    has_directive = surface_business_strategy_directive(norm, tokens)
    
    # Check for multi-word vocab
    multi_word_vocab = {
        "market share", "competitive strategy", "pricing strategy",
        "go-to-market", "business model", "revenue strategy"
    }
    has_vocab = surface_business_strategy_vocab(tokens) or any(x in w for x in multi_word_vocab)
    
    return has_named and has_directive and has_vocab

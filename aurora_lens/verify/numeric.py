"""NumericValue: minimal parser and comparator for financial numeric literals.

Layer 1 / Layer 3 and PEF alignment share this module. No spaCy dependency.

Handles: percentages, basis points, currency with k/m/b suffixes, and common
non-USD presentations (EUR/GBP/AUD, ISO codes, spaced thousands, decimal comma
when disambiguated).

Intentional v1 limitations (listed at bottom of file).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class NumericUnit(Enum):
    PERCENT = "PERCENT"    # percentage points — 23% → 23.0, 120 bps → 1.2
    CURRENCY = "CURRENCY"  # nominal currency units — no FX between codes in v1
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class NumericValue:
    magnitude: float
    unit: NumericUnit


# ── Parsing patterns ───────────────────────────────────────────────────────

# Basis points: "120 bps", "120 basis points" — converted to percent
_BPS_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s*(?:basis\s+points?|bps)\b',
    re.IGNORECASE,
)

# Percentage: "23%", "23.4%"
_PCT_RE = re.compile(r'(\d+(?:\.\d+)?)\s*%')

_SUFFIX_MAP: dict[str, float] = {
    'billion': 1_000_000_000.0, 'bn': 1_000_000_000.0,
    'million': 1_000_000.0,     'mn': 1_000_000.0,
    'millions': 1_000_000.0,
    'thousand': 1_000.0,
    'b': 1_000_000_000.0,
    'm': 1_000_000.0,
    'k': 1_000.0,
}

# Optional magnitude tail shared by $ € £ A$ and ISO-adjacent forms.
# Groups: 1=mantissa, 2=billion word, 3=million word, 4=thousand word, 5=letter k/m/b
_CUR_MAG_TAIL = (
    r'([\d,.\s\u00a0]*\d[\d,.\s\u00a0]*)\s*'
    r'(?:(billion|bn)\b|(million|mn|millions?)\b|(thousand)\b|([kKmMbB])\b)?'
)

_CURRENCY_DOLLAR_RE = re.compile(r'\$\s*' + _CUR_MAG_TAIL, re.IGNORECASE)
_CURRENCY_EURO_RE = re.compile(r'€\s*' + _CUR_MAG_TAIL, re.IGNORECASE)
_CURRENCY_GBP_RE = re.compile(r'£\s*' + _CUR_MAG_TAIL, re.IGNORECASE)
_CURRENCY_AUD_RE = re.compile(r'(?:AU\$|A\$)\s*' + _CUR_MAG_TAIL, re.IGNORECASE)
_CURRENCY_ISO_PREFIX_RE = re.compile(
    r'(?:EUR|GBP|USD|AUD)\s+' + _CUR_MAG_TAIL,
    re.IGNORECASE,
)
_CURRENCY_ISO_SUFFIX_RE = re.compile(
    _CUR_MAG_TAIL + r'\s*(?:EUR|GBP|USD|AUD)\b',
    re.IGNORECASE,
)

_ALL_CURRENCY_RES: tuple[re.Pattern[str], ...] = (
    _CURRENCY_DOLLAR_RE,
    _CURRENCY_EURO_RE,
    _CURRENCY_GBP_RE,
    _CURRENCY_AUD_RE,
    _CURRENCY_ISO_PREFIX_RE,
    _CURRENCY_ISO_SUFFIX_RE,
)

_US_THOUSANDS_COMMA_RE = re.compile(r'^\d{1,3}(,\d{3})+$')


def _parse_currency_mantissa(raw: str) -> float | None:
    """Parse the digit part of a currency literal into a float mantissa (before k/m/b)."""
    if not raw:
        return None
    s = raw.strip().replace('\u00a0', ' ')
    s = re.sub(r'(?<=\d) (?=\d)', '', s)
    s = s.replace(' ', '')
    if not s or not any(c.isdigit() for c in s):
        return None

    if ',' in s and '.' in s:
        li, lj = s.rfind(','), s.rfind('.')
        if li > lj:
            s = s.replace('.', '').replace(',', '.')
        else:
            s = s.replace(',', '')
    elif ',' in s:
        if _US_THOUSANDS_COMMA_RE.fullmatch(s):
            s = s.replace(',', '')
        else:
            parts = s.split(',')
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                left, right = parts[0], parts[1]
                if len(right) == 3 and len(left) <= 3:
                    s = left + right
                elif len(right) <= 2:
                    s = f'{left}.{right}'
                else:
                    s = left + right
            else:
                s = s.replace(',', '')
    try:
        return float(s)
    except ValueError:
        return None


def _currency_match_to_value(m: re.Match[str]) -> NumericValue | None:
    raw = m.group(1)
    base = _parse_currency_mantissa(raw)
    if base is None:
        return None
    suffix = (
        (m.group(2) or m.group(3) or m.group(4) or '').lower()
        or (m.group(5) or '').lower()
    )
    mult = _SUFFIX_MAP.get(suffix, 1.0)
    return NumericValue(magnitude=base * mult, unit=NumericUnit.CURRENCY)


def _first_currency_value(text: str) -> NumericValue | None:
    best: NumericValue | None = None
    best_start = len(text) + 1
    for cre in _ALL_CURRENCY_RES:
        m = cre.search(text)
        if m is not None and m.start() < best_start:
            v = _currency_match_to_value(m)
            if v is not None:
                best_start = m.start()
                best = v
    return best


def _all_currency_spans(text: str) -> list[tuple[int, int, NumericValue]]:
    """Non-overlapping currency matches, left-to-right (first wins on overlap)."""
    raw: list[tuple[int, int, NumericValue]] = []
    for cre in _ALL_CURRENCY_RES:
        for m in cre.finditer(text):
            v = _currency_match_to_value(m)
            if v is not None:
                raw.append((m.start(), m.end(), v))
    raw.sort(key=lambda t: (t[0], -(t[1] - t[0])))
    out: list[tuple[int, int, NumericValue]] = []
    last_end = -1
    for start, end, v in raw:
        if start >= last_end:
            out.append((start, end, v))
            last_end = end
    return out


def parse_numeric(text: str) -> NumericValue | None:
    """Parse a string into a NumericValue.

    Returns None if no parseable numeric found.
    Parse order: bps → percent → currency (first currency match in text).
    Ambiguous bare decimals without a unit marker are not parsed.
    """
    if not text:
        return None
    s = text.strip()

    m = _BPS_RE.search(s)
    if m:
        bps = float(m.group(1))
        return NumericValue(magnitude=round(bps / 100.0, 10), unit=NumericUnit.PERCENT)

    m = _PCT_RE.search(s)
    if m:
        return NumericValue(magnitude=float(m.group(1)), unit=NumericUnit.PERCENT)

    return _first_currency_value(s)


def parse_all_numerics(text: str) -> list[NumericValue]:
    """Return every parseable NumericValue found in text.

    parse_numeric returns only the first match; this returns all of them.
    Currency matches are merged without overlapping spans.
    """
    results: list[NumericValue] = []
    seen_spans: set[tuple[int, int]] = set()

    for m in _BPS_RE.finditer(text):
        if m.span() not in seen_spans:
            seen_spans.add(m.span())
            bps = float(m.group(1))
            results.append(NumericValue(
                magnitude=round(bps / 100.0, 10), unit=NumericUnit.PERCENT
            ))

    for m in _PCT_RE.finditer(text):
        if m.span() not in seen_spans:
            seen_spans.add(m.span())
            results.append(NumericValue(
                magnitude=float(m.group(1)), unit=NumericUnit.PERCENT
            ))

    for start, end, v in _all_currency_spans(text):
        if (start, end) not in seen_spans:
            seen_spans.add((start, end))
            results.append(v)

    return results


def numeric_for_metric_span(text: str, metric_start: int, metric_end: int) -> NumericValue | None:
    """Return the NumericValue in *text* whose span is closest to the metric token span.

    ``parse_numeric(text)`` applies a single global order (bps before %, then
    currency).  A sentence can contain multiple numbers (e.g. a % return and a
    bps gap); Layer 1 grounding must pair the financial metric token with the
    adjacent numeric when checking user/PEF alignment, not the first match in
    document order.
    """
    if not text:
        return None
    seen_spans: set[tuple[int, int]] = set()
    spans: list[tuple[int, int, NumericValue]] = []

    for m in _BPS_RE.finditer(text):
        if m.span() in seen_spans:
            continue
        seen_spans.add(m.span())
        bps = float(m.group(1))
        spans.append((
            m.start(),
            m.end(),
            NumericValue(magnitude=round(bps / 100.0, 10), unit=NumericUnit.PERCENT),
        ))

    for m in _PCT_RE.finditer(text):
        if m.span() in seen_spans:
            continue
        seen_spans.add(m.span())
        spans.append((
            m.start(),
            m.end(),
            NumericValue(magnitude=float(m.group(1)), unit=NumericUnit.PERCENT),
        ))

    for start, end, v in _all_currency_spans(text):
        if (start, end) in seen_spans:
            continue
        seen_spans.add((start, end))
        spans.append((start, end, v))

    if not spans:
        return None

    ref_c = (metric_start + metric_end) / 2.0

    def _dist(t: tuple[int, int, NumericValue]) -> float:
        c = (t[0] + t[1]) / 2.0
        return abs(c - ref_c)

    spans.sort(key=_dist)
    return spans[0][2]


def values_contradict(
    a: NumericValue,
    b: NumericValue,
    tolerance: float = 0.001,
) -> bool:
    """True if a and b have compatible units and magnitudes differ beyond tolerance.

    Relative tolerance: |a - b| / max(|a|, |b|, 1e-9) > tolerance.
    Incompatible units always return False — cannot compare across domains.
    """
    if a.unit != b.unit:
        return False
    if a.unit == NumericUnit.UNKNOWN:
        return False
    denom = max(abs(a.magnitude), abs(b.magnitude), 1e-9)
    return abs(a.magnitude - b.magnitude) / denom > tolerance


# ── Intentional v1 limitations ─────────────────────────────────────────────
# The following are explicitly out of scope and will not be detected:
#
# 1. Bare decimals without unit markers — "0.23" alone is ambiguous
# 2. Ranges — "5–10%" or "between 5% and 10%"
# 3. Cross-currency FX — "$100" vs "€80" (no conversion; same unit bucket only)
# 4. Inferred/derived values — "profit" from "revenue minus cost"
# 5. Temporal mismatch — 2023 actuals vs 2024 projections
# 6. Non-financial numerics — "3 tablets", "10 miles"
# 7. Alias/synonym resolution — "return on investment" ≠ "ROI" in v1
# 8. Negated PEF facts — skipped in v1
# 9. Ambiguous comma in long digit strings without magnitude suffix

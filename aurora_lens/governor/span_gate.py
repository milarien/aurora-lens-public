# SOURCE: aurora-governor/libs/kernel/span_splitter.py
# Verbatim lift. Zero external dependencies. Track in GOVERNOR_SYNC.md.
"""
Span Transition Detector

Splits utterances on temporal boundaries (NOW, THEN) to enable correct span admission.

Why this exists:
- "James had it, now I have it" must admit clauses to different spans
- Prevents false contradictions across time boundaries
- Enables "when" queries to answer "now" vs "previously"

Usage:
    from libs.kernel.span_splitter import split_temporal_clauses
    
    left, right = split_temporal_clauses("James had it, now I have it")
    # left = "James had it" (admit to echo/past span)
    # right = "I have it" (admit to active/present span)
"""

import re
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TemporalClause:
    """A clause with its intended span."""
    text: str
    span: str  # "echo" | "active"
    

# Temporal boundary markers (case-insensitive)
TEMPORAL_MARKERS = [
    # Present → Past boundary
    r"\bnow\b",
    r"\bcurrently\b",
    r"\bat present\b",
    r"\bthese days\b",
    
    # Sequential boundary
    r"\bthen\b",
    r"\bafter that\b",
    r"\bnext\b",
    
    # Contrast boundary  
    r"\bbut now\b",
    r"\bhowever now\b",
]


def split_temporal_clauses(utterance: str) -> Tuple[Optional[TemporalClause], Optional[TemporalClause]]:
    """
    Split utterance on first temporal boundary.
    
    Args:
        utterance: Input text that may contain temporal transition
        
    Returns:
        (left_clause, right_clause) where:
        - left_clause is assigned to "echo" span (past/previous)
        - right_clause is assigned to "active" span (present/current)
        - Either may be None if no split occurs
        
    Examples:
        >>> left, right = split_temporal_clauses("James had it, now I have it")
        >>> left.text
        'James had it'
        >>> left.span
        'echo'
        >>> right.text
        'I have it'
        >>> right.span
        'active'
        
        >>> left, right = split_temporal_clauses("I have an apple")
        >>> left is None
        True
        >>> right.span
        'active'
    """
    # Try each marker (in order of specificity - longer first)
    markers = sorted(TEMPORAL_MARKERS, key=len, reverse=True)
    
    for marker_pattern in markers:
        # Look for marker preceded by comma/semicolon/space and followed by space
        # This prevents matching "snow" when looking for "now"
        pattern = rf"([,;]\s*|\s+)({marker_pattern})(\s+)"
        match = re.search(pattern, utterance, re.IGNORECASE)
        
        if match:
            split_pos = match.start()
            left_text = utterance[:split_pos].strip().rstrip(",;")
            right_text = utterance[match.end():].strip()
            
            # If nothing on left, treat as present-only
            if not left_text:
                return None, TemporalClause(text=right_text, span="active")
            
            # If nothing on right, treat as past-only  
            if not right_text:
                return TemporalClause(text=left_text, span="echo"), None
            
            # Both sides present
            return (
                TemporalClause(text=left_text, span="echo"),
                TemporalClause(text=right_text, span="active"),
            )
    
    # No temporal boundary found - treat as present
    return None, TemporalClause(text=utterance, span="active")


def describe_span_split(utterance: str) -> str:
    """
    Diagnostic: Explain how an utterance would be split.
    
    Args:
        utterance: Input to analyze
        
    Returns:
        Human-readable explanation of span assignment
        
    Example:
        >>> print(describe_span_split("James had it, now I have it"))
        ECHO span (past): "James had it"
        ACTIVE span (present): "I have it"
    """
    left, right = split_temporal_clauses(utterance)
    
    lines = []
    if left:
        lines.append(f'{left.span.upper()} span ({"past" if left.span == "echo" else "present"}): "{left.text}"')
    if right:
        lines.append(f'{right.span.upper()} span ({"past" if right.span == "echo" else "present"}): "{right.text}"')
    
    if not lines:
        return "No clauses detected (empty input)"
    
    return "\n".join(lines)


def requires_span_split(utterance: str) -> bool:
    """
    Check if utterance contains temporal boundary.
    
    Args:
        utterance: Input to check
        
    Returns:
        True if utterance should be split across spans
        
    Example:
        >>> requires_span_split("James had it, now I have it")
        True
        >>> requires_span_split("I have an apple")
        False
    """
    left, right = split_temporal_clauses(utterance)
    return left is not None and right is not None

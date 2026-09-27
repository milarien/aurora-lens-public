"""Span enum — bounded conceptual episodes."""

from enum import Enum


class Span(Enum):
    PRESENT = "present"
    PAST = "past"
    # Future expansion:
    # HYPOTHETICAL = "hypothetical"
    # CONDITIONAL = "conditional"

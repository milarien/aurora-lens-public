"""Verification layer — compare LLM claims against PEF state."""

from .flags import Flag, FlagType
from .checker import Checker

__all__ = ["Flag", "FlagType", "Checker"]

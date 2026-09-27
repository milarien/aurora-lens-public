from __future__ import annotations

from abc import ABC, abstractmethod

from aurora_lens.pef.state import PEFState
from .schema import ExtractionResult


class ExtractionBackend(ABC):
    """Backend interface for turning raw text into extracted claims.

    Concrete backends may use spaCy, LLM, or anything else.
    PEF is required — without it, pronouns, temporal references,
    and entity context cannot be resolved.
    This module must stay spaCy-free.
    """

    @abstractmethod
    async def extract(self, text: str, pef: PEFState) -> ExtractionResult:
        ...

    def supports_ambiguous_referents(self) -> bool:
        """Whether this backend emits ambiguity tokens for referent-hold parity.

        Backends returning False are formally excluded from ambiguity parity claims.
        """
        return True

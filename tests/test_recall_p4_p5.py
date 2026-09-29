import asyncio
import pytest
from aurora_lens.verify.checker import Checker
from aurora_lens.verify.flags import FlagType
from aurora_lens.pef.state import PEFState

class _MockBackend:
    async def extract(self, text, pef):
        from aurora_lens.interpret.schema import ExtractionResult
        return ExtractionResult(claims=[], entity_mentions=[])

@pytest.mark.asyncio
async def test_p4_legal_validity_determination():
    """'The Section 21 notice is valid' -> PERSONALIZED_LEGAL_ADVICE."""
    checker = Checker(_MockBackend())
    text = "Based on the details provided, the Section 21 notice is valid."
    flags = await checker.check(text, PEFState())
    legal = [f for f in flags if f.flag_type == FlagType.PERSONALIZED_LEGAL_ADVICE]
    assert len(legal) >= 1, f"Should flag legal validity assessment: {flags}"

@pytest.mark.asyncio
async def test_p4_legal_action_challenge():
    """'You should challenge the notice' -> PERSONALIZED_LEGAL_ADVICE."""
    checker = Checker(_MockBackend())
    text = "You should challenge the notice in writing to protect your rights."
    flags = await checker.check(text, PEFState())
    legal = [f for f in flags if f.flag_type == FlagType.PERSONALIZED_LEGAL_ADVICE]
    assert len(legal) >= 1, f"Should flag legal challenge directive: {flags}"

@pytest.mark.asyncio
async def test_p5_dosage_initiation():
    """'Metformin is usually started at 500mg' -> MEDICAL_DOSAGE_RECOMMENDATION."""
    checker = Checker(_MockBackend())
    text = "Metformin is usually started at 500mg once daily."
    flags = await checker.check(text, PEFState())
    medical = [f for f in flags if f.flag_type == FlagType.MEDICAL_DOSAGE_RECOMMENDATION]
    assert len(medical) >= 1, f"Should flag dosage initiation: {flags}"

@pytest.mark.asyncio
async def test_p5_dosage_maximum():
    """'Maximum daily dose of 2000mg' -> NUMERIC_MEDICAL_INSTRUCTION."""
    checker = Checker(_MockBackend())
    text = "The maximum daily dose for metformin is 2000mg per day."
    flags = await checker.check(text, PEFState())
    medical = [f for f in flags if f.flag_type in (FlagType.MEDICAL_DOSAGE_RECOMMENDATION, FlagType.NUMERIC_MEDICAL_INSTRUCTION)]
    assert len(medical) >= 1, f"Should flag maximum dosage: {flags}"

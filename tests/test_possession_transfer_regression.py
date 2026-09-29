"""Reproduction test for PEF possession transfer regression."""

import pytest
from unittest.mock import MagicMock, AsyncMock
from aurora_lens.config import LensConfig
from aurora_lens.interpret.spacy_backend import SpacyBackend
from aurora_lens.lens import Lens
from aurora_lens.govern.decision import InterventionAction

class _AuditCapturingBridge:
    def __init__(self):
        self.decisions = []
        self._policy = MagicMock()
        self._policy.name = "strict"

    async def decide(self, flags, user_text, pef):
        from aurora_lens.govern.decision import GovernanceDecision
        return GovernanceDecision(
            action=InterventionAction.PASS,
            flags=flags,
            rationale="test",
            policy="strict"
        )

    async def intervene(self, decision, adapter, user_text, pef_context):
        return decision.original_response or ""

    def log_decision(self, decision, **kwargs):
        self.decisions.append((decision, kwargs))

@pytest.mark.asyncio
async def test_possession_transfer_sequence():
    """
    Alice gave Bob the book.
    Who has the book? -> Bob has the book.
    Bob gave the book to Cyril.
    Who has the book? -> Cyril has the book.
    """
    bridge = _AuditCapturingBridge()
    adapter = MagicMock()
    adapter.generate = AsyncMock(return_value=MagicMock(text="MODEL_RESPONSE", model="mock"))
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=SpacyBackend(),
            enable_state_native_delegation=True,
            auto_verify=True,
            auto_interpret=True,
        ),
        session_id="transfer-test",
    )
    lens._bridge = bridge

    # 1. Alice gave Bob the book.
    # Note: Using "Alice gave Bob a book" to avoid the ambiguity check for "the book"
    # if "Alice gave Bob" is mis-parsed as an entity.
    r1 = await lens.process("Alice gave Bob a book.")
    assert "Recorded" in r1.response
    
    # 2. Who has the book?
    r2 = await lens.process("Who has the book?")
    assert "Bob has the book" in r2.response
    
    # 3. Bob gave the book to Cyril.
    r3 = await lens.process("Bob gave the book to Cyril.")
    assert "Recorded" in r3.response
    
    # 4. Who has the book?
    r4 = await lens.process("Who has the book?")
    assert "Cyril has the book" in r4.response
    assert "Bob has the book" not in r4.response

    # 5. What does Bob have?
    r5 = await lens.process("What does Bob have?")
    assert "nothing" in r5.response.lower() or "not have" in r5.response.lower() or "doesn't have" in r5.response.lower()

    # 6. What does Cyril have?
    r6 = await lens.process("What does Cyril have?")
    assert "book" in r6.response.lower()

@pytest.mark.asyncio
async def test_three_hop_transfer():
    """
    Alice gave Bob the book.
    Bob gave the book to Cyril.
    Cyril gave the book to Dana.
    Who has the book? -> Dana has the book.
    """
    bridge = _AuditCapturingBridge()
    adapter = MagicMock()
    adapter.generate = AsyncMock(return_value=MagicMock(text="MODEL_RESPONSE", model="mock"))
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=SpacyBackend(),
            enable_state_native_delegation=True,
            auto_verify=True,
            auto_interpret=True,
        ),
        session_id="three-hop-test",
    )
    lens._bridge = bridge

    await lens.process("Alice gave Bob a book.")
    await lens.process("Bob gave the book to Cyril.")
    await lens.process("Cyril gave the book to Dana.")
    
    r = await lens.process("Who has the book?")
    assert "Dana has the book" in r.response
    
    r_cyril = await lens.process("What does Cyril have?")
    assert "nothing" in r_cyril.response.lower() or "not have" in r_cyril.response.lower() or "doesn't have" in r_cyril.response.lower()

    r_dana = await lens.process("What does Dana have?")
    assert "book" in r_dana.response.lower()


@pytest.mark.asyncio
async def test_transfer_pronoun_binds_to_concrete_object_not_it_literal():
    """
    Anna gave Jim a book.
    Jim gave the book to Cyril.
    Cyril read the book and gave it to Jane.
    Who has the book? -> Jane has the book.
    """
    bridge = _AuditCapturingBridge()
    adapter = MagicMock()
    adapter.generate = AsyncMock(return_value=MagicMock(text="MODEL_RESPONSE", model="mock"))
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=SpacyBackend(),
            enable_state_native_delegation=True,
            auto_verify=True,
            auto_interpret=True,
        ),
        session_id="pronoun-transfer-binding-test",
    )
    lens._bridge = bridge

    await lens.process("Anna gave Jim a book.")
    await lens.process("Jim gave the book to Cyril.")
    r3 = await lens.process("Cyril read the book and gave it to Jane.")
    assert r3.action != InterventionAction.CONTAIN

    r4 = await lens.process("Who has the book?")
    assert "Jane has the book" in r4.response
    assert "Cyril has the book" not in r4.response

    # Canonical committed relations must not carry unresolved pronoun literals.
    assert all(
        (rel.object_literal or "").lower() != "it"
        for rel in lens._pef.relationships
        if rel.relation in ("GIVE", "HAS")
    )


@pytest.mark.asyncio
async def test_transfer_pronoun_unresolved_is_consequence_bearing_and_contained():
    """
    Anna gave Jim a book.
    Anna gave Jim a key.
    Jim gave it to Cyril.

    Ambiguity is consequence-bearing for transfer mutation, so this must CONTAIN.
    """
    bridge = _AuditCapturingBridge()
    adapter = MagicMock()
    adapter.generate = AsyncMock(return_value=MagicMock(text="MODEL_RESPONSE", model="mock"))
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=SpacyBackend(),
            enable_state_native_delegation=True,
            auto_verify=True,
            auto_interpret=True,
        ),
        session_id="pronoun-transfer-consequence-bearing-test",
    )
    lens._bridge = bridge

    await lens.process("Anna gave Jim a book.")
    await lens.process("Anna gave Jim a key.")
    r3 = await lens.process("Jim gave it to Cyril.")
    assert r3.action != InterventionAction.PASS
    assert "ambiguity" in r3.response.lower() or "clarification" in r3.response.lower() or "unresolved" in r3.response.lower()

    # Invariant: unresolved pronoun literals must not be committed in canonical relations.
    assert all(
        (rel.object_literal or "").lower() != "it"
        for rel in lens._pef.relationships
        if rel.relation in ("GIVE", "HAS")
    )


@pytest.mark.asyncio
async def test_four_hop_definite_book_chain():
    """Regression: 'the book' as dobj of GIVE must not trigger CONTAIN on first mention.

    Anna gave the book to John.  (turn 1 — 'the book' must PASS, not CONTAIN)
    John gave the book to Cyril.
    Cyril gave the book to Joan.
    Who has the book? -> Joan has the book. (state-native, LLM not called)
    """
    from aurora_lens.adapters.base import LLMAdapter, AdapterResponse

    class PassthroughAdapter(LLMAdapter):
        async def generate(self, messages, **kwargs):
            return AdapterResponse(content="", stop_reason="user_text")

    config = LensConfig(
        adapter=PassthroughAdapter(),
        enable_state_native_delegation=True,
    )
    lens = Lens(config=config, session_id="four-hop-definite-book")

    r1 = await lens.process(user_input="Anna gave the book to John.")
    assert r1.decision.action == InterventionAction.PASS, (
        f"Turn 1 must PASS; got {r1.decision.action}. "
        "'the book' as dobj of GIVE is the transferred object, not an anaphoric referent."
    )

    r2 = await lens.process(user_input="John gave the book to Cyril.")
    assert r2.decision.action == InterventionAction.PASS

    r3 = await lens.process(user_input="Cyril gave the book to Joan.")
    assert r3.decision.action == InterventionAction.PASS

    r4 = await lens.process(user_input="Who has the book?")
    assert r4.decision.action == InterventionAction.PASS
    assert "Joan" in r4.response, f"Expected Joan in response, got: {r4.response}"

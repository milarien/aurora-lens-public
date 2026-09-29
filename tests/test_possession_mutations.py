"""Regression tests for generic quantitative possession mutations (GIVE, CONSUME).

These tests verify:
1. GIVE transfers update both giver and recipient quantities
2. CONSUME mutations correctly reduce quantities
3. Quantity queries return correct values after mutations
4. Multiple entity/object cases prove the logic is generic
5. Graceful failure when quantities are insufficient
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from aurora_lens.lens import Lens
from aurora_lens.config import LensConfig
from aurora_lens.adapters.base import LLMAdapter, AdapterResponse
from aurora_lens.govern.decision import InterventionAction
from aurora_lens.state_native_engine.eval.inventory import normalize_possession_item_surface


class PassthroughAdapter(LLMAdapter):
    """Adapter that returns user text as-is (for state-native testing)."""

    async def generate(self, messages: list[dict[str, str]], **kwargs) -> AdapterResponse:
        return AdapterResponse(
            content="",
            stop_reason="user_text",
        )


def test_normalize_possession_item_surface_strips_trailing_punctuation_only():
    assert normalize_possession_item_surface("Foo-Bar.,;:!?") == "Foo-Bar"


@pytest.mark.asyncio
async def test_possession_pronoun_gave_digit_recipient_has_committed_quantity():
    """Past-tense GIVE + pronoun giver + single-digit count: recipient gets committed HAS.

    Regression for spaCy ``nmod`` parse (``He gave Jill 5 apples``) and pre-model ``gave``
    matching: inventory queries must not STOP on missing Jill quantity; LLM unused.
    """
    adapter = MagicMock(spec=LLMAdapter)
    adapter.generate = AsyncMock(
        return_value=MagicMock(text="SHOULD_NOT_USE", model="mock"),
    )
    config = LensConfig(
        adapter=adapter,
        enable_state_native_delegation=True,
        auto_verify=True,
        auto_interpret=True,
    )
    lens = Lens(config=config, session_id="pronoun-gave-digit-apples")

    assert (await lens.process(user_input="Jack had ten apples.")).decision.action == InterventionAction.PASS
    assert (await lens.process(user_input="He gave Jill 5 apples.")).decision.action == InterventionAction.PASS
    assert (await lens.process(user_input="Jill ate 1 apple.")).decision.action == InterventionAction.PASS

    result = await lens.process(user_input="How many apples does Jill have?")
    assert result.decision.action == InterventionAction.PASS
    assert "4 apples" in (result.response or "").lower()
    assert result.continuity_diagnostic and "state_native" in (result.continuity_diagnostic or "")
    assert adapter.generate.await_count == 0


@pytest.mark.parametrize("use_sentence_punct", [True, False], ids=("with_periods", "bare_lines"))
@pytest.mark.asyncio
async def test_possession_mutation_sequence_punctuation_variants(use_sentence_punct):
    """Terminal punctuation versus bare lines (sentence periods and optional query ``?``)."""
    period = "." if use_sentence_punct else ""
    qmark = "?" if use_sentence_punct else ""

    session_id = f"punct-regression-{'dots' if use_sentence_punct else 'bare'}"

    config = LensConfig(
        adapter=PassthroughAdapter(),
        enable_state_native_delegation=True,
    )
    lens = Lens(config=config, session_id=session_id)

    result = await lens.process(user_input=f"Jack has 10 apples{period}")
    assert result.decision.action == InterventionAction.PASS

    result = await lens.process(user_input=f"Jack gives Jill 6 apples{period}")
    assert result.decision.action == InterventionAction.PASS

    result = await lens.process(user_input=f"Jill eats 2 apples{period}")
    assert result.decision.action == InterventionAction.PASS

    result = await lens.process(user_input=f"How many apples does Jill have{qmark}")
    assert result.decision.action == InterventionAction.PASS
    assert "4" in result.response


@pytest.mark.asyncio
async def test_possession_mutation_mixed_case_item_token():
    """Non-lowercase alphanumeric item tokens round-trip across mutate + quantity query."""
    session_id = "test-pmx-widgets"
    config = LensConfig(
        adapter=PassthroughAdapter(),
        enable_state_native_delegation=True,
    )
    lens = Lens(config=config, session_id=session_id)

    await lens.process(user_input="Vera has 12 pMx9 widgets.")
    await lens.process(user_input="Vera gives Wei 7 pMx9 widgets!")
    await lens.process(user_input="Wei consumes 2 pMx9 widgets;")
    result = await lens.process(user_input="How many pMx9 widgets does Wei have?")
    assert result.decision.action == InterventionAction.PASS
    assert "5" in result.response


@pytest.mark.asyncio
async def test_possession_mutation_alice_bob_books_generic():
    """Regression: Generic case with different entity names and object.

    Alice has 8 books -> Alice gives Bob 3 -> Bob eats/consumes 1 (metaphorically) -> Bob has 2.
    This proves the mutation logic is not hardcoded to Jack/Jill/apples.
    """
    session_id = "test-alice-bob-books"
    config = LensConfig(
        adapter=PassthroughAdapter(),
        enable_state_native_delegation=True,
    )
    lens = Lens(config=config, session_id=session_id)

    # Turn 1: Alice has 8 books
    result = await lens.process(user_input="Alice has 8 books.")
    assert result.response is not None
    assert result.decision.action == InterventionAction.PASS

    # Turn 2: Alice gives Bob 3 books
    result = await lens.process(user_input="Alice gives Bob 3 books.")
    assert result.response is not None
    assert result.decision.action == InterventionAction.PASS

    # Turn 3: Bob consumes 1 book
    result = await lens.process(user_input="Bob consumes 1 book.")
    assert result.response is not None
    assert result.decision.action == InterventionAction.PASS

    # Turn 4: How many books does Bob have?
    result = await lens.process(user_input="How many books does Bob have?")
    assert result.response is not None
    assert result.decision.action == InterventionAction.PASS
    # Bob received 3, consumed 1, should have 2
    assert "2" in result.response, f"Expected '2' in response, got: {result.response}"


@pytest.mark.asyncio
async def test_possession_mutation_giver_insufficient_quantity():
    """Regression: GIVE fails gracefully when giver doesn't have enough."""
    session_id = "test-insufficient-give"
    config = LensConfig(
        adapter=PassthroughAdapter(),
        enable_state_native_delegation=True,
    )
    lens = Lens(config=config, session_id=session_id)

    # Turn 1: Chris has 3 oranges
    result = await lens.process(user_input="Chris has 3 oranges.")
    assert result.response is not None
    assert result.decision.action == InterventionAction.PASS

    # Turn 2: Chris tries to give Dana 5 oranges (more than available)
    result = await lens.process(user_input="Chris gives Dana 5 oranges.")
    # Should fail gracefully (not throw 500), either not handled or a governed STOP
    assert result.decision.action in (InterventionAction.PASS, InterventionAction.HARD_STOP)


@pytest.mark.asyncio
async def test_possession_mutation_consume_no_active_possession():
    """Regression: CONSUME with no prior inventory is admitted as asserted fact (PASS).

    Previously returned HARD_STOP which also triggered a forensic schema 500 because
    the state-native STOP produced flags=[] → failed_constraints=[] which violated
    the schema invariant (failed_constraints must be non-empty when present).
    """
    session_id = "test-consume-empty"
    config = LensConfig(
        adapter=PassthroughAdapter(),
        enable_state_native_delegation=True,
    )
    lens = Lens(config=config, session_id=session_id)

    # No prior inventory — should be admitted as recorded assertion, not HARD_STOP.
    result = await lens.process(user_input="Eve eats 2 cookies.")
    assert result.response is not None
    assert result.decision.action == InterventionAction.PASS
    assert "recorded" in result.response.lower()


@pytest.mark.asyncio
async def test_possession_mutation_query_no_inventory():
    """Regression: Query returns governed 'not enough state' when no inventory exists."""
    session_id = "test-query-empty"
    config = LensConfig(
        adapter=PassthroughAdapter(),
        enable_state_native_delegation=True,
    )
    lens = Lens(config=config, session_id=session_id)

    # Turn 1: How many pens does Frank have? (no state)
    result = await lens.process(user_input="How many pens does Frank have?")
    # Should not crash; either state-native STOP or LLM response
    assert result.response is not None
    # If state-native handles it, it should be a STOP with governed text
    # If LLM handles it, it should be PASS but not a 500 error


@pytest.mark.asyncio
async def test_possession_mutation_chain_multiple_transfers():
    """Regression: Multiple sequential transfers maintain correct state.

    Grace has 10 coins.
    Grace gives Henry 4 coins.
    Henry gives Isaac 2 coins.
    How many coins does Isaac have?
    Expected: 2
    """
    session_id = "test-multi-transfer"
    config = LensConfig(
        adapter=PassthroughAdapter(),
        enable_state_native_delegation=True,
    )
    lens = Lens(config=config, session_id=session_id)

    # Turn 1: Grace has 10 coins
    result = await lens.process(user_input="Grace has 10 coins.")
    assert result.response is not None
    assert result.decision.action == InterventionAction.PASS

    # Turn 2: Grace gives Henry 4 coins
    result = await lens.process(user_input="Grace gives Henry 4 coins.")
    assert result.response is not None
    assert result.decision.action == InterventionAction.PASS

    # Turn 3: Henry gives Isaac 2 coins
    result = await lens.process(user_input="Henry gives Isaac 2 coins.")
    assert result.response is not None
    assert result.decision.action == InterventionAction.PASS

    # Turn 4: How many coins does Isaac have?
    result = await lens.process(user_input="How many coins does Isaac have?")
    assert result.response is not None
    assert result.decision.action == InterventionAction.PASS
    assert "2" in result.response, f"Expected '2' in response, got: {result.response}"


@pytest.mark.asyncio
async def test_possession_mutation_past_tense_eat_word_number():
    """Acceptance: past-tense 'ate' with word-number quantity routes through PEF dispatch.

    Jenny has 5 apples -> Jenny ate one apple -> Jenny has 4.
    Verifies EAT relation canonicalization and structured mutation dispatch (no regex).
    """
    session_id = "test-eat-word-number"
    config = LensConfig(
        adapter=PassthroughAdapter(),
        enable_state_native_delegation=True,
    )
    lens = Lens(config=config, session_id=session_id)

    result = await lens.process(user_input="Jenny has 5 apples.")
    assert result.decision.action == InterventionAction.PASS

    result = await lens.process(user_input="Jenny ate one apple.")
    assert result.decision.action == InterventionAction.PASS

    result = await lens.process(user_input="How many apples does Jenny have?")
    assert result.decision.action == InterventionAction.PASS
    assert "4" in result.response, f"Expected '4' in response, got: {result.response}"


@pytest.mark.asyncio
async def test_possession_mutation_give_then_eat_acceptance():
    """Acceptance: GIVE transfer followed by past-tense EAT with word number.

    John had 10 apples -> John gave Jenny 5 apples -> Jenny ate one apple
    -> Jenny has 4.
    """
    session_id = "test-give-then-eat"
    config = LensConfig(
        adapter=PassthroughAdapter(),
        enable_state_native_delegation=True,
    )
    lens = Lens(config=config, session_id=session_id)

    result = await lens.process(user_input="John has 10 apples.")
    assert result.decision.action == InterventionAction.PASS

    result = await lens.process(user_input="John gives Jenny 5 apples.")
    assert result.decision.action == InterventionAction.PASS

    result = await lens.process(user_input="Jenny ate one apple.")
    assert result.decision.action == InterventionAction.PASS

    result = await lens.process(user_input="How many apples does Jenny have?")
    assert result.decision.action == InterventionAction.PASS
    assert "4" in result.response, f"Expected '4' in response, got: {result.response}"


@pytest.mark.asyncio
async def test_possession_mutation_verify_giver_quantity_reduced():
    """Regression: After GIVE, giver's quantity is reduced correctly.

    Jack has 10 apples.
    Jack gives Jill 6 apples.
    How many apples does Jack have?
    Expected: 4
    """
    session_id = "test-giver-qty"
    config = LensConfig(
        adapter=PassthroughAdapter(),
        enable_state_native_delegation=True,
    )
    lens = Lens(config=config, session_id=session_id)

    # Turn 1: Jack has 10 apples
    result = await lens.process(user_input="Jack has 10 apples.")
    assert result.response is not None
    assert result.decision.action == InterventionAction.PASS

    # Turn 2: Jack gives Jill 6 apples
    result = await lens.process(user_input="Jack gives Jill 6 apples.")
    assert result.response is not None
    assert result.decision.action == InterventionAction.PASS

    # Turn 3: How many apples does Jack have?
    result = await lens.process(user_input="How many apples does Jack have?")
    assert result.response is not None
    assert result.decision.action == InterventionAction.PASS
    # Jack gave away 6 from 10, should have 4
    assert "4" in result.response, f"Expected '4' in response, got: {result.response}"


@pytest.mark.asyncio
async def test_past_tense_how_many_query_routes_state_native():
    """Regression: 'How many apples did Jenny have?' must route to state-native, not governance STOP.

    Jenny has 5 -> ate 1 -> 'How many apples did Jenny have?' -> 4.
    """
    session_id = "test-past-tense-qty-query"
    config = LensConfig(
        adapter=PassthroughAdapter(),
        enable_state_native_delegation=True,
    )
    lens = Lens(config=config, session_id=session_id)

    result = await lens.process(user_input="Jenny has 5 apples.")
    assert result.decision.action == InterventionAction.PASS

    result = await lens.process(user_input="Jenny ate one apple.")
    assert result.decision.action == InterventionAction.PASS

    result = await lens.process(user_input="How many apples did Jenny have?")
    assert result.decision.action == InterventionAction.PASS
    assert "4" in result.response, f"Expected '4' in response, got: {result.response}"


@pytest.mark.asyncio
async def test_past_tense_how_many_have_left_query():
    """Regression: 'How many apples did Jenny have left?' also routes state-native."""
    session_id = "test-past-tense-have-left"
    config = LensConfig(
        adapter=PassthroughAdapter(),
        enable_state_native_delegation=True,
    )
    lens = Lens(config=config, session_id=session_id)

    result = await lens.process(user_input="Jenny has 5 apples.")
    assert result.decision.action == InterventionAction.PASS

    result = await lens.process(user_input="Jenny ate one apple.")
    assert result.decision.action == InterventionAction.PASS

    result = await lens.process(user_input="How many apples did Jenny have left?")
    assert result.decision.action == InterventionAction.PASS
    assert "4" in result.response, f"Expected '4' in response, got: {result.response}"


@pytest.mark.asyncio
async def test_present_tense_query_unaffected_by_past_tense_parser():
    """Non-regression: present-tense 'does ... have' still returns 'has' (not 'had')."""
    session_id = "test-present-tense-unaffected"
    config = LensConfig(
        adapter=PassthroughAdapter(),
        enable_state_native_delegation=True,
    )
    lens = Lens(config=config, session_id=session_id)

    result = await lens.process(user_input="Jenny has 5 apples.")
    assert result.decision.action == InterventionAction.PASS

    result = await lens.process(user_input="How many apples does Jenny have?")
    assert result.decision.action == InterventionAction.PASS
    assert "has" in result.response.lower(), f"Expected 'has' in present-tense response, got: {result.response}"
    assert "5" in result.response, f"Expected '5' in response, got: {result.response}"


def test_possession_quantity_query_http_never_500(monkeypatch, tmp_path):
    """Regression: Jack/Jill mutation chain + quantity query must not proxy_internal_error."""
    from tests.test_integration import _CountingAdapter, _make_client

    audit = tmp_path / "possession_http.jsonl"
    client = _make_client(
        monkeypatch,
        adapter=_CountingAdapter(response="NEVER_RETURNED"),
        audit_log=str(audit),
        extra={
            "governance": {
                "default_policy": "strict",
                "audit_backend": "jsonl",
                "include_operator_detail": True,
            },
            "extraction": {"backend": "spacy"},
        },
    )
    sid = "possession-http-jack-jill"
    headers = {"x-aurora-session-id": sid, "x-aurora-operator-detail": "1"}
    turns = [
        "Jack has 10 apples.",
        "Jack gives Jill 6 apples.",
        "Jill eats 2 apples.",
        "How many apples does Jill have?",
    ]
    for content in turns:
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "mock",
                "messages": [{"role": "user", "content": content}],
            },
            headers=headers,
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert (body.get("error") or {}).get("code") != "aurora_proxy_internal_error"
        assert (body.get("aurora", {}).get("governance") or "").upper() != "ERROR"
        if content.startswith("How many"):
            assert "4" in body["choices"][0]["message"]["content"]
            assert body["aurora"]["governance"] == "PASS"

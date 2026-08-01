from datetime import datetime, timedelta, timezone

from agent_memory.models import MemoryContext, MemoryFact, ScoredFact, Turn
from agent_memory.read.pipeline import (
    read_memory,
    render_context_as_text,
    should_search_tier2,
)

from .fakes import FakeEmbeddingClient, FakeFactStore, FakeProfileCache, FakeSessionCache


def _fact(**overrides) -> MemoryFact:
    defaults = dict(user_id="u1", category="preference", value="likes concise answers", confidence=0.9)
    defaults.update(overrides)
    return MemoryFact(**defaults)


async def test_read_memory_gathers_all_three_tiers():
    prior_turn = Turn(
        user_id="u1", conversation_id="c1", user_message="hi",
        assistant_message="hello there",
    )
    relevant_fact = _fact()

    session_cache = FakeSessionCache(turns=[prior_turn])
    profile_cache = FakeProfileCache(profile={"tone": "concise"})
    fact_store = FakeFactStore(search_results=[ScoredFact(fact=relevant_fact, score=0.95)])
    embedding_client = FakeEmbeddingClient()

    context = await read_memory(
        user_id="u1",
        conversation_id="c1",
        message="what's a good subject line?",
        session_cache=session_cache,
        profile_cache=profile_cache,
        fact_store=fact_store,
        embedding_client=embedding_client,
    )

    assert context.profile == {"tone": "concise"}
    assert context.recent_turns == [prior_turn]
    assert len(context.relevant_facts) == 1
    assert context.relevant_facts[0].fact.id == relevant_fact.id

    assert embedding_client.embedded_texts == ["what's a good subject line?"]


async def test_read_memory_handles_empty_tiers_gracefully():
    context = await read_memory(
        user_id="u2",
        conversation_id="c2",
        message="first message ever",
        session_cache=FakeSessionCache(),
        profile_cache=FakeProfileCache(profile=None),
        fact_store=FakeFactStore(),
        embedding_client=FakeEmbeddingClient(),
    )

    assert context.profile is None
    assert context.relevant_facts == []
    assert context.recent_turns == []


# --- retrieval gate ----------------------------------------------------


def test_should_search_tier2_true_for_real_messages():
    assert should_search_tier2("write an essay about nuclear fusion")
    assert should_search_tier2("what's a good subject line for this email?")


def test_should_search_tier2_false_for_trivial_messages():
    for message in ["ok", "Ok.", "thanks", "thank you!", "sure", "k", "np", "  ", ""]:
        assert not should_search_tier2(message), message


async def test_read_memory_skips_tier2_for_trivial_message():
    fact_store = FakeFactStore(search_results=[ScoredFact(fact=_fact(), score=0.99)])
    embedding_client = FakeEmbeddingClient()

    context = await read_memory(
        user_id="u1",
        conversation_id="c1",
        message="thanks!",
        session_cache=FakeSessionCache(),
        profile_cache=FakeProfileCache(),
        fact_store=fact_store,
        embedding_client=embedding_client,
    )

    assert context.relevant_facts == []
    # gated: no embedding call, no vector search -- the whole point is to
    # skip the expensive part, not just discard its result afterward
    assert embedding_client.embedded_texts == []


# --- rerank --------------------------------------------------------------


async def test_read_memory_rerank_prefers_recent_reinforced_fact_over_raw_similarity():
    now = datetime.now(timezone.utc)
    stale_but_more_similar = _fact(
        value="stale fact", last_reinforced_at=now - timedelta(days=300), confidence=0.5,
    )
    fresh_but_less_similar = _fact(
        value="fresh fact", last_reinforced_at=now, confidence=0.99,
    )

    fact_store = FakeFactStore(search_results=[
        ScoredFact(fact=stale_but_more_similar, score=0.99),
        ScoredFact(fact=fresh_but_less_similar, score=0.80),
    ])

    context = await read_memory(
        user_id="u1", conversation_id="c1", message="a real question here",
        session_cache=FakeSessionCache(), profile_cache=FakeProfileCache(),
        fact_store=fact_store, embedding_client=FakeEmbeddingClient(),
    )

    values = [sf.fact.value for sf in context.relevant_facts]
    assert values[0] == "fresh fact"
    assert values[1] == "stale fact"


# --- token-budget packing --------------------------------------------------


async def test_read_memory_stops_packing_once_budget_exceeded():
    big_fact = _fact(value="x" * 4000)  # ~1000 approx-tokens
    small_fact = _fact(value="short")

    fact_store = FakeFactStore(search_results=[
        ScoredFact(fact=big_fact, score=0.99),
        ScoredFact(fact=small_fact, score=0.90),
    ])

    context = await read_memory(
        user_id="u1", conversation_id="c1", message="a real question here",
        session_cache=FakeSessionCache(), profile_cache=FakeProfileCache(),
        fact_store=fact_store, embedding_client=FakeEmbeddingClient(),
        token_budget=500,
    )

    # the big fact alone exceeds the budget, so nothing after it gets packed
    assert context.relevant_facts == []


async def test_read_memory_packs_multiple_small_facts_within_budget():
    facts = [_fact(value=f"fact number {i}") for i in range(5)]
    fact_store = FakeFactStore(search_results=[
        ScoredFact(fact=f, score=0.9 - i * 0.01) for i, f in enumerate(facts)
    ])

    context = await read_memory(
        user_id="u1", conversation_id="c1", message="a real question here",
        session_cache=FakeSessionCache(), profile_cache=FakeProfileCache(),
        fact_store=fact_store, embedding_client=FakeEmbeddingClient(),
        token_budget=500,
    )

    assert len(context.relevant_facts) == 5


# --- render_context_as_text (unchanged behavior) ---------------------------


async def test_render_context_as_text_includes_every_populated_section():
    prior_turn = Turn(
        user_id="u1", conversation_id="c1", user_message="hi",
        assistant_message="hello there",
    )
    fact = _fact()

    context = MemoryContext(
        profile={"tone": "concise"},
        relevant_facts=[ScoredFact(fact=fact, score=0.95)],
        recent_turns=[prior_turn],
    )

    text = render_context_as_text(context, "what's a good subject line?")

    assert "tone: concise" in text
    assert "likes concise answers" in text
    assert "hello there" in text
    assert "what's a good subject line?" in text


async def test_render_context_as_text_omits_empty_sections():
    context = MemoryContext(profile=None, relevant_facts=[], recent_turns=[])
    text = render_context_as_text(context, "first message ever")

    assert "USER PROFILE" not in text
    assert "RELEVANT MEMORY" not in text
    assert "RECENT CONVERSATION" not in text
    assert "first message ever" in text

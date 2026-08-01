from agent_memory.models import MemoryFact, ScoredFact, Turn
from agent_memory.read.pipeline import read_memory, render_context_as_text

from .fakes import FakeEmbeddingClient, FakeFactStore, FakeProfileCache, FakeSessionCache


async def test_read_memory_gathers_all_three_tiers():
    prior_turn = Turn(
        user_id="u1",
        conversation_id="c1",
        user_message="hi",
        assistant_message="hello there",
    )
    relevant_fact = MemoryFact(
        user_id="u1", category="preference", value="likes concise answers",
        confidence=0.9,
    )

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
    assert context.relevant_facts == [ScoredFact(fact=relevant_fact, score=0.95)]
    assert context.recent_turns == [prior_turn]

    # the message was embedded exactly once, and that vector drove the search
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


async def test_render_context_as_text_includes_every_populated_section():
    prior_turn = Turn(
        user_id="u1", conversation_id="c1", user_message="hi",
        assistant_message="hello there",
    )
    fact = MemoryFact(
        user_id="u1", category="preference", value="likes concise answers",
        confidence=0.9,
    )

    from agent_memory.models import MemoryContext

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
    from agent_memory.models import MemoryContext

    context = MemoryContext(profile=None, relevant_facts=[], recent_turns=[])
    text = render_context_as_text(context, "first message ever")

    assert "USER PROFILE" not in text
    assert "RELEVANT MEMORY" not in text
    assert "RECENT CONVERSATION" not in text
    assert "first message ever" in text

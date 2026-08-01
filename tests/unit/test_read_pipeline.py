from agent_memory.models import MemoryFact, ScoredFact, Turn
from agent_memory.read.pipeline import read_and_respond

from .fakes import (
    FakeChatClient,
    FakeEmbeddingClient,
    FakeFactStore,
    FakeProfileCache,
    FakeSessionCache,
)


async def test_read_and_respond_assembles_all_tiers_into_the_prompt():
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
    chat_client = FakeChatClient(response="here is my answer")

    response_text, turn = await read_and_respond(
        user_id="u1",
        conversation_id="c1",
        message="what's a good subject line?",
        session_cache=session_cache,
        profile_cache=profile_cache,
        fact_store=fact_store,
        embedding_client=embedding_client,
        chat_client=chat_client,
        system_prompt="you are an assistant",
    )

    assert response_text == "here is my answer"
    assert turn.user_id == "u1"
    assert turn.conversation_id == "c1"
    assert turn.user_message == "what's a good subject line?"
    assert turn.assistant_message == "here is my answer"

    # the query was embedded exactly once, and used to drive fact_store.search_facts
    assert embedding_client.embedded_texts == ["what's a good subject line?"]

    system_prompt, user_prompt = chat_client.calls[0]
    assert system_prompt == "you are an assistant"
    assert "likes concise answers" in user_prompt
    assert "tone: concise" in user_prompt
    assert "hello there" in user_prompt
    assert "what's a good subject line?" in user_prompt


async def test_read_and_respond_handles_empty_tiers_gracefully():
    session_cache = FakeSessionCache()
    profile_cache = FakeProfileCache(profile=None)
    fact_store = FakeFactStore()
    embedding_client = FakeEmbeddingClient()
    chat_client = FakeChatClient(response="ok")

    response_text, turn = await read_and_respond(
        user_id="u2",
        conversation_id="c2",
        message="first message ever",
        session_cache=session_cache,
        profile_cache=profile_cache,
        fact_store=fact_store,
        embedding_client=embedding_client,
        chat_client=chat_client,
    )

    assert response_text == "ok"
    system_prompt, user_prompt = chat_client.calls[0]
    assert "first message ever" in user_prompt

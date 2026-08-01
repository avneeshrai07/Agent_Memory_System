from uuid import uuid4

import pytest

from agent_memory.management import MemoryNotFoundError, delete_memory, edit_memory, list_memories
from agent_memory.models import MemoryFact

from .fakes import FakeEmbeddingClient, FakeFactStore


async def test_list_memories_returns_only_that_users_facts():
    mine = MemoryFact(user_id="u1", category="a", value="mine", confidence=0.9)
    theirs = MemoryFact(user_id="u2", category="a", value="theirs", confidence=0.9)
    fact_store = FakeFactStore()
    fact_store.added = [mine, theirs]

    result = await list_memories("u1", fact_store=fact_store)

    assert result == [mine]


async def test_edit_memory_updates_value_and_reembeds():
    original = MemoryFact(
        user_id="u1", category="preference", value="old value", confidence=0.9
    )
    fact_store = FakeFactStore()
    fact_store.added = [original]
    embedding_client = FakeEmbeddingClient(vector=[9.0, 9.0])

    edited = await edit_memory(
        original.id, "new value", fact_store=fact_store, embedding_client=embedding_client
    )

    assert edited.id == original.id
    assert edited.value == "new value"
    assert edited.embedding == [9.0, 9.0]
    assert embedding_client.embedded_texts == ["new value"]
    assert fact_store.updated == [edited]


async def test_edit_memory_raises_for_missing_fact():
    fact_store = FakeFactStore()

    with pytest.raises(MemoryNotFoundError):
        await edit_memory(
            uuid4(), "new value",
            fact_store=fact_store, embedding_client=FakeEmbeddingClient(),
        )


async def test_delete_memory_removes_fact():
    fact = MemoryFact(user_id="u1", category="a", value="v", confidence=0.9)
    fact_store = FakeFactStore()
    fact_store.added = [fact]

    await delete_memory(fact.id, fact_store=fact_store)

    assert fact_store.deleted == [fact.id]
    assert await fact_store.get_fact(fact.id) is None

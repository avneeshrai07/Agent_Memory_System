"""User-facing memory control (README Section 7 non-negotiable): view, edit,
delete.

This is the surface a host application's own "manage your memories" UI/API
is expected to call. It is plain CRUD over FactStore — it does not re-run
extraction, resolution, or the safety gate; those only apply to what the
formation path infers on its own. A user directly editing or deleting a
memory is an explicit instruction, not an inference, so it bypasses all of
that judgment by design.

Editing always re-embeds: a stored vector that no longer matches its text
would make retrieval silently wrong, not just stale.
"""

from __future__ import annotations

from uuid import UUID

from agent_memory.llm.interfaces import EmbeddingClient
from agent_memory.models import MemoryFact
from agent_memory.storage.interfaces import FactStore


class MemoryNotFoundError(Exception):
    def __init__(self, fact_id: UUID):
        super().__init__(f"No memory fact with id {fact_id}")
        self.fact_id = fact_id


async def list_memories(
    user_id: str, *, fact_store: FactStore, limit: int = 50, offset: int = 0
) -> list[MemoryFact]:
    return await fact_store.list_facts(user_id, limit, offset)


async def edit_memory(
    fact_id: UUID,
    new_value: str,
    *,
    fact_store: FactStore,
    embedding_client: EmbeddingClient,
) -> MemoryFact:
    existing = await fact_store.get_fact(fact_id)
    if existing is None:
        raise MemoryNotFoundError(fact_id)

    embedding = await embedding_client.embed(new_value)
    edited = existing.model_copy(update={"value": new_value, "embedding": embedding})
    return await fact_store.update_fact(edited)


async def delete_memory(fact_id: UUID, *, fact_store: FactStore) -> None:
    await fact_store.delete_fact(fact_id)

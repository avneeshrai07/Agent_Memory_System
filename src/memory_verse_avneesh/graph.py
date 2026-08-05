"""User-facing graph memory control: view, delete (README Section 7's
non-negotiable, extended to graph memory).

View/delete only, same shape as episodic.py -- entities and edges are
created by the formation pipeline (write_memory()'s relation extraction +
deterministic resolution), not by the host directly, so there is no
create_entity/create_edge here. delete_entity cascades to every edge
touching it (GraphStore's own contract); delete_edge removes a single edge
permanently, distinct from the formation pipeline's own close_edge (which
supersedes, not deletes).
"""

from __future__ import annotations

from uuid import UUID

from memory_verse_avneesh.models import Edge, Entity
from memory_verse_avneesh.storage.interfaces import GraphStore


class EntityNotFoundError(Exception):
    def __init__(self, entity_id: UUID):
        super().__init__(f"No entity with id {entity_id}")
        self.entity_id = entity_id


async def list_entities(
    user_id: str, *, graph_store: GraphStore, limit: int = 50, offset: int = 0
) -> list[Entity]:
    return await graph_store.list_entities(user_id, limit, offset)


async def get_entity(entity_id: UUID, *, graph_store: GraphStore) -> Entity:
    entity = await graph_store.get_entity(entity_id)
    if entity is None:
        raise EntityNotFoundError(entity_id)
    return entity


async def delete_entity(entity_id: UUID, *, graph_store: GraphStore) -> None:
    await graph_store.delete_entity(entity_id)


async def list_edges(
    entity_id: UUID, *, graph_store: GraphStore, limit: int = 50, offset: int = 0
) -> list[Edge]:
    """Full history (current and closed) for one entity, newest-first."""
    return await graph_store.list_edges_for_entity(entity_id, limit, offset)


async def delete_edge(edge_id: UUID, *, graph_store: GraphStore) -> None:
    await graph_store.delete_edge(edge_id)

import pytest

from memory_verse_avneesh.graph import (
    EntityNotFoundError,
    delete_edge,
    delete_entity,
    get_entity,
    list_edges,
    list_entities,
)
from memory_verse_avneesh.models import Edge, Entity

from .fakes import FakeGraphStore


def _entity(**overrides) -> Entity:
    defaults = dict(user_id="u1", name="Acme Corp")
    defaults.update(overrides)
    return Entity(**defaults)


def _edge(source_entity_id, **overrides) -> Edge:
    defaults = dict(
        user_id="u1", source_entity_id=source_entity_id, relation="works_at",
        target_value="Acme Corp", fact_sentence="User works at Acme Corp.", confidence=0.9,
    )
    defaults.update(overrides)
    return Edge(**defaults)


async def test_get_entity_returns_existing():
    store = FakeGraphStore()
    entity = _entity()
    await store.create_entity(entity)

    fetched = await get_entity(entity.id, graph_store=store)
    assert fetched.name == "Acme Corp"


async def test_get_entity_raises_when_missing():
    store = FakeGraphStore()
    with pytest.raises(EntityNotFoundError):
        await get_entity(_entity().id, graph_store=store)


async def test_list_entities_filters_by_user():
    store = FakeGraphStore()
    await store.create_entity(_entity(user_id="u1", name="Acme Corp"))
    await store.create_entity(_entity(user_id="u2", name="Other Corp"))

    entities = await list_entities("u1", graph_store=store)
    assert [e.name for e in entities] == ["Acme Corp"]


async def test_delete_entity_cascades_to_its_edges():
    store = FakeGraphStore()
    entity = _entity()
    await store.create_entity(entity)
    edge = _edge(entity.id)
    await store.add_edge(edge)

    await delete_entity(entity.id, graph_store=store)

    with pytest.raises(EntityNotFoundError):
        await get_entity(entity.id, graph_store=store)
    assert await store.get_edge(edge.id) is None


async def test_delete_entity_is_idempotent():
    store = FakeGraphStore()
    entity = _entity()
    await store.create_entity(entity)

    await delete_entity(entity.id, graph_store=store)
    await delete_entity(entity.id, graph_store=store)  # no error


async def test_list_edges_returns_full_history_for_entity():
    store = FakeGraphStore()
    entity = _entity()
    await store.create_entity(entity)
    current = _edge(entity.id, relation="works_at")
    await store.add_edge(current)
    closed = await store.close_edge(current.id, current.valid_from)

    edges = await list_edges(entity.id, graph_store=store)
    assert len(edges) == 1
    assert edges[0].id == closed.id


async def test_delete_edge_is_idempotent():
    store = FakeGraphStore()
    entity = _entity()
    await store.create_entity(entity)
    edge = _edge(entity.id)
    await store.add_edge(edge)

    await delete_edge(edge.id, graph_store=store)
    await delete_edge(edge.id, graph_store=store)  # no error

    assert await store.get_edge(edge.id) is None

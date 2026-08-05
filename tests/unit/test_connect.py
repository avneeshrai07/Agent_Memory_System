"""Unit-level coverage for connect() -- only the parts that don't need a
real Postgres connection (backend/config validation, which must fail before
any connection is attempted). Full construction is covered by live
verification against real infrastructure, not here.
"""

import asyncio

import pytest

from memory_verse_avneesh.connect import Memory, MemoryPool, connect, setup

from .fakes import (
    FakeEmbeddingClient,
    FakeEpisodicStore,
    FakeExtractionClient,
    FakeFactStore,
    FakeGraphStore,
    FakeIdentityStore,
    FakeProfileCache,
    FakeRelationExtractionClient,
    FakeReminderStore,
    FakeResolutionClient,
    FakeSessionCache,
)


def _memory(session_cache=None, fact_store=None) -> Memory:
    """A Memory wired entirely to fakes -- no connect(), no real backends."""
    async def _noop_close() -> None:
        return None

    return Memory(
        fact_store=fact_store or FakeFactStore(),
        identity_store=FakeIdentityStore(),
        episodic_store=FakeEpisodicStore(),
        reminder_store=FakeReminderStore(),
        graph_store=FakeGraphStore(),
        session_cache=session_cache or FakeSessionCache(),
        profile_cache=FakeProfileCache(),
        embedding_client=FakeEmbeddingClient(),
        extraction_client=FakeExtractionClient(),
        resolution_client=FakeResolutionClient(),
        relation_extraction_client=FakeRelationExtractionClient(),
        close=_noop_close,
    )


async def test_connect_rejects_invalid_config_before_touching_postgres():
    # No cache backend configured -- MemoryConfig.__post_init__ should raise
    # before create_pool() is ever called (would hang/fail slowly against a
    # bogus host otherwise).
    with pytest.raises(ValueError, match="no Tier 0/1 cache backend"):
        await connect(
            database_url="postgresql://this-host-does-not-exist-anywhere:5432/db",
            postgres_schema="s1",
        )


async def test_connect_rejects_conflicting_postgres_config_before_touching_postgres():
    with pytest.raises(ValueError, match="pick one way to configure Postgres"):
        await connect(
            database_url="postgresql://this-host-does-not-exist-anywhere:5432/db",
            postgres_host="localhost", postgres_user="u", postgres_password="pw",
            postgres_database="db",
            postgres_schema="s1",
            redis_url="redis://localhost",
        )


# --- setup() -----------------------------------------------------------


def test_setup_returns_a_pool_without_connecting():
    # No I/O at all -- a bogus host must NOT be contacted here.
    pool = setup(
        database_url="postgresql://this-host-does-not-exist-anywhere:5432/db",
        postgres_schema="s1",
        redis_url="redis://localhost",
    )
    assert isinstance(pool, MemoryPool)
    assert pool.memory is None


def test_setup_validates_config_eagerly():
    with pytest.raises(ValueError, match="no Tier 0/1 cache backend"):
        setup(
            database_url="postgresql://this-host-does-not-exist-anywhere:5432/db",
            postgres_schema="s1",
        )




def test_setup_requires_postgres_schema():
    with pytest.raises(KeyError):
        setup(database_url="postgresql://x", redis_url="redis://localhost")


# --- Memory.write() ----------------------------------------------------


async def test_write_builds_the_turn_from_flat_arguments():
    memory = _memory()
    turn = await memory.write(
        user_id="u1", conversation_id="c1",
        user_message="hello", assistant_message="hi there",
    )
    assert turn.user_id == "u1"
    assert turn.conversation_id == "c1"
    assert turn.user_message == "hello"
    assert turn.assistant_message == "hi there"
    await memory.flush()


async def test_write_appends_to_session_history_before_returning():
    session_cache = FakeSessionCache()
    memory = _memory(session_cache=session_cache)

    await memory.write(
        user_id="u1", conversation_id="c1",
        user_message="hello", assistant_message="hi there",
    )

    # Inline, not backgrounded -- the next turn's Tier 0 read must see it
    # immediately, with no flush() needed.
    turns = await session_cache.get_recent_turns("c1", 10)
    assert [t.user_message for t in turns] == ["hello"]
    await memory.flush()


async def test_write_runs_formation_in_the_background():
    fact_store = FakeFactStore()
    memory = _memory(fact_store=fact_store)

    await memory.write(
        user_id="u1", conversation_id="c1",
        user_message="hello", assistant_message="hi there",
    )
    assert memory._background_tasks, "expected a background formation task"

    await memory.flush()
    assert not memory._background_tasks, "flush() should drain the task set"


async def test_write_survives_session_cache_failure():
    class _BrokenSessionCache(FakeSessionCache):
        async def append_turn(self, turn):
            raise RuntimeError("simulated session cache failure")

    memory = _memory(session_cache=_BrokenSessionCache())

    # Must not raise -- the response is already going out to the user.
    turn = await memory.write(
        user_id="u1", conversation_id="c1",
        user_message="hello", assistant_message="hi there",
    )
    assert turn.user_message == "hello"
    await memory.flush()


async def test_write_never_surfaces_background_formation_failures():
    class _BrokenExtractionClient:
        async def extract(self, turn):
            raise RuntimeError("simulated extraction failure")

    memory = _memory()
    memory.extraction_client = _BrokenExtractionClient()

    await memory.write(
        user_id="u1", conversation_id="c1",
        user_message="hello", assistant_message="hi there",
    )
    await memory.flush()  # must not raise


async def test_close_flushes_background_work_first():
    closed: list[bool] = []

    async def _record_close() -> None:
        closed.append(True)

    memory = _memory()
    memory._close = _record_close

    await memory.write(
        user_id="u1", conversation_id="c1",
        user_message="hello", assistant_message="hi there",
    )
    await memory.close()

    assert not memory._background_tasks  # drained before closing
    assert closed == [True]

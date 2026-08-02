from datetime import datetime, timedelta, timezone

from memory_verse_avneesh.formation.decay import run_decay_sweep
from memory_verse_avneesh.models import MemoryFact, MemoryStatus

from .fakes import FakeFactStore


def _old(days: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days)


async def test_decay_sweep_archives_stale_unreinforced_facts():
    stale = MemoryFact(
        user_id="u1", category="preference", value="old fact",
        confidence=0.9, status=MemoryStatus.ACTIVE, last_reinforced_at=_old(200),
    )
    fresh = MemoryFact(
        user_id="u1", category="preference", value="fresh fact",
        confidence=0.9, status=MemoryStatus.ACTIVE,
    )
    fact_store = FakeFactStore()
    fact_store.added = [stale, fresh]

    archived_count = await run_decay_sweep(
        fact_store=fact_store, decay_after=timedelta(days=90)
    )

    assert archived_count == 1
    assert len(fact_store.updated) == 1
    assert fact_store.updated[0].id == stale.id
    assert fact_store.updated[0].status == MemoryStatus.ARCHIVED


async def test_decay_sweep_ignores_superseded_and_already_archived():
    superseded = MemoryFact(
        user_id="u1", category="a", value="v", confidence=0.9,
        status=MemoryStatus.SUPERSEDED, last_reinforced_at=_old(200),
    )
    already_archived = MemoryFact(
        user_id="u1", category="a", value="v2", confidence=0.9,
        status=MemoryStatus.ARCHIVED, last_reinforced_at=_old(200),
    )
    fact_store = FakeFactStore()
    fact_store.added = [superseded, already_archived]

    archived_count = await run_decay_sweep(
        fact_store=fact_store, decay_after=timedelta(days=90)
    )

    assert archived_count == 0
    assert fact_store.updated == []


async def test_decay_sweep_leaves_recently_reinforced_provisional_facts_alone():
    recent_provisional = MemoryFact(
        user_id="u1", category="preference", value="v", confidence=0.5,
        status=MemoryStatus.PROVISIONAL,
    )
    fact_store = FakeFactStore()
    fact_store.added = [recent_provisional]

    archived_count = await run_decay_sweep(fact_store=fact_store)

    assert archived_count == 0
    assert fact_store.updated == []


async def test_decay_sweep_returns_zero_on_empty_store():
    fact_store = FakeFactStore()
    assert await run_decay_sweep(fact_store=fact_store) == 0

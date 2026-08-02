import pytest

from memory_verse_avneesh.episodic import EpisodeNotFoundError, delete_episode, get_episode, list_episodes
from memory_verse_avneesh.models import Episode

from .fakes import FakeEpisodicStore


def _episode(**overrides) -> Episode:
    defaults = dict(
        user_id="u1", conversation_id="c1",
        user_message="what's a good subject line?", assistant_message="try 'Quick question'",
    )
    defaults.update(overrides)
    return Episode(**defaults)


async def test_get_episode_returns_existing():
    store = FakeEpisodicStore()
    episode = _episode()
    await store.add_episode(episode)

    fetched = await get_episode(episode.id, episodic_store=store)
    assert fetched.id == episode.id


async def test_get_episode_raises_when_missing():
    store = FakeEpisodicStore()
    with pytest.raises(EpisodeNotFoundError):
        await get_episode(_episode().id, episodic_store=store)


async def test_delete_episode_is_idempotent():
    store = FakeEpisodicStore()
    episode = _episode()
    await store.add_episode(episode)

    await delete_episode(episode.id, episodic_store=store)
    await delete_episode(episode.id, episodic_store=store)  # no error

    with pytest.raises(EpisodeNotFoundError):
        await get_episode(episode.id, episodic_store=store)


async def test_list_episodes_filters_by_user_and_orders_newest_first():
    store = FakeEpisodicStore()
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    older = _episode(user_id="u1", user_message="older", created_at=now - timedelta(days=1))
    newer = _episode(user_id="u1", user_message="newer", created_at=now)
    other_user = _episode(user_id="u2", user_message="not mine")

    await store.add_episode(older)
    await store.add_episode(newer)
    await store.add_episode(other_user)

    episodes = await list_episodes("u1", episodic_store=store)
    assert [e.user_message for e in episodes] == ["newer", "older"]

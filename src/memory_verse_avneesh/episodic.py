"""User-facing episodic memory control: view, delete (README Section 7's
non-negotiable, extended to episodic memory).

No edit — an episode is a record of what was actually said, not a fact that
gets corrected as understanding improves. Deletion is the only user-facing
mutation, e.g. for an explicit "forget this conversation" / privacy request.
"""

from __future__ import annotations

from uuid import UUID

from memory_verse_avneesh.models import Episode
from memory_verse_avneesh.storage.interfaces import EpisodicStore


class EpisodeNotFoundError(Exception):
    def __init__(self, episode_id: UUID):
        super().__init__(f"No episode with id {episode_id}")
        self.episode_id = episode_id


async def list_episodes(
    user_id: str, *, episodic_store: EpisodicStore, limit: int = 50, offset: int = 0
) -> list[Episode]:
    return await episodic_store.list_episodes(user_id, limit, offset)


async def get_episode(episode_id: UUID, *, episodic_store: EpisodicStore) -> Episode:
    episode = await episodic_store.get_episode(episode_id)
    if episode is None:
        raise EpisodeNotFoundError(episode_id)
    return episode


async def delete_episode(episode_id: UUID, *, episodic_store: EpisodicStore) -> None:
    await episodic_store.delete_episode(episode_id)

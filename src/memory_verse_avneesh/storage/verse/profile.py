"""ProfileCache backed by storage_verse_avneesh's unified CacheBackend
(Tier 1, README Section 3).

Structurally satisfies memory_verse_avneesh.storage.interfaces.ProfileCache.
"""

from __future__ import annotations

import json
from typing import Any


class VerseProfileCache:
    def __init__(self, store: Any, *, ttl_seconds: int | None = None):
        self._store = store
        self._ttl_seconds = ttl_seconds

    @staticmethod
    def _key(user_id: str) -> str:
        return f"memory_verse_avneesh:profile:{user_id}"

    async def get_profile(self, user_id: str) -> dict | None:
        raw = await self._store.get(self._key(user_id))
        return json.loads(raw) if raw is not None else None

    async def set_profile(self, user_id: str, profile: dict) -> None:
        await self._store.set(
            self._key(user_id), json.dumps(profile), ttl_seconds=self._ttl_seconds
        )

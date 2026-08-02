"""SessionCache backed by storage_verse_avneesh's unified CacheBackend
(Tier 0, README Section 3).

Works with either of storage_verse_avneesh's cache backends unmodified —
"redis" or "upstash" — since it already unifies both behind one interface.
Construct the store yourself via storage_verse_avneesh.get_store(...) and
pass it in; this class doesn't care which one it's holding.

Structurally satisfies memory_verse_avneesh.storage.interfaces.SessionCache.
"""

from __future__ import annotations

from typing import Any

from memory_verse_avneesh.models import Turn


class VerseSessionCache:
    def __init__(
        self,
        store: Any,
        *,
        max_stored_turns: int = 50,
        ttl_seconds: int = 6 * 60 * 60,
    ):
        self._store = store
        self._max_stored_turns = max_stored_turns
        self._ttl_seconds = ttl_seconds

    @staticmethod
    def _key(conversation_id: str) -> str:
        return f"memory_verse_avneesh:session:{conversation_id}"

    async def append_turn(self, turn: Turn) -> None:
        key = self._key(turn.conversation_id)
        # storage_verse_avneesh's CacheBackend has no pipeline concept (its
        # Upstash implementation is REST-based, its Redis one keeps things
        # uniform across backends rather than exposing pipelining only
        # where the transport happens to support it) — three sequential
        # awaits, not grouped.
        await self._store.rpush(key, turn.model_dump_json())
        await self._store.ltrim(key, -self._max_stored_turns, -1)
        await self._store.expire(key, self._ttl_seconds)

    async def get_recent_turns(self, conversation_id: str, limit: int) -> list[Turn]:
        key = self._key(conversation_id)
        raw = await self._store.lrange(key, -limit, -1)
        return [Turn.model_validate_json(item) for item in raw]

"""Redis implementation of SessionCache (Tier 0, README Section 3).

Structurally satisfies memory_verse_avneesh.storage.interfaces.SessionCache.
"""

from __future__ import annotations

from redis.asyncio import Redis

from memory_verse_avneesh.models import Turn


class RedisSessionCache:
    def __init__(
        self,
        client: Redis,
        *,
        max_stored_turns: int = 50,
        ttl_seconds: int = 6 * 60 * 60,
    ):
        self._client = client
        self._max_stored_turns = max_stored_turns
        self._ttl_seconds = ttl_seconds

    @staticmethod
    def _key(conversation_id: str) -> str:
        return f"memory_verse_avneesh:session:{conversation_id}"

    async def append_turn(self, turn: Turn) -> None:
        key = self._key(turn.conversation_id)
        async with self._client.pipeline(transaction=True) as pipe:
            pipe.rpush(key, turn.model_dump_json())
            pipe.ltrim(key, -self._max_stored_turns, -1)
            pipe.expire(key, self._ttl_seconds)
            await pipe.execute()

    async def get_recent_turns(self, conversation_id: str, limit: int) -> list[Turn]:
        key = self._key(conversation_id)
        raw = await self._client.lrange(key, -limit, -1)
        return [Turn.model_validate_json(item) for item in raw]

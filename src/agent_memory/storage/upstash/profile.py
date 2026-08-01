"""Upstash implementation of ProfileCache (Tier 1, README Section 3).

Structurally satisfies agent_memory.storage.interfaces.ProfileCache.
"""

from __future__ import annotations

import json
from typing import Any


class UpstashProfileCache:
    def __init__(self, client: Any, *, ttl_seconds: int | None = None):
        self._client = client
        self._ttl_seconds = ttl_seconds

    @staticmethod
    def _key(user_id: str) -> str:
        return f"agent_memory:profile:{user_id}"

    async def get_profile(self, user_id: str) -> dict | None:
        raw = await self._client.get(self._key(user_id))
        return json.loads(raw) if raw is not None else None

    async def set_profile(self, user_id: str, profile: dict) -> None:
        await self._client.set(
            self._key(user_id), json.dumps(profile), ex=self._ttl_seconds
        )

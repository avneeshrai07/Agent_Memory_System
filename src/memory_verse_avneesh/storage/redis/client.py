"""Redis client factory for the Tier 0 / Tier 1 backends."""

from __future__ import annotations

from redis.asyncio import Redis, from_url


def create_redis_client(url: str) -> Redis:
    return from_url(url, decode_responses=True)

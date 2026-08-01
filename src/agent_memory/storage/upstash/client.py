"""Upstash Redis client factory.

Upstash's client speaks its REST API over HTTPS, not the standard Redis
wire protocol — a different transport from agent_memory.storage.redis (which
uses redis.asyncio), but it implements the same SessionCache/ProfileCache
Protocols, so it's a drop-in alternative wherever a host uses Upstash
instead of a standard redis://-reachable instance.
"""

from __future__ import annotations

from upstash_redis.asyncio import Redis


def create_upstash_client(url: str, token: str) -> Redis:
    return Redis(url=url, token=token)

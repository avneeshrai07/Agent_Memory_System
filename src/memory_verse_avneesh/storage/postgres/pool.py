"""Connection pool factory for the Postgres backend."""

from __future__ import annotations

import asyncpg
from pgvector.asyncpg import register_vector


async def create_pool(
    dsn: str, *, min_size: int = 1, max_size: int = 10
) -> asyncpg.Pool:
    """Create a pooled asyncpg connection with the pgvector codec registered
    on every connection (required so list[float] <-> the VECTOR column type
    convert automatically — without this, embeddings would need manual
    literal formatting on every query).
    """

    async def _init(conn: asyncpg.Connection) -> None:
        await register_vector(conn)

    return await asyncpg.create_pool(
        dsn, min_size=min_size, max_size=max_size, init=_init
    )

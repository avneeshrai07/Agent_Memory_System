"""Connection pool factory for the Postgres backend."""

from __future__ import annotations

import asyncpg
from pgvector.asyncpg import register_vector


async def create_pool(
    dsn: str | None = None,
    *,
    host: str | None = None,
    port: int | None = None,
    user: str | None = None,
    password: str | None = None,
    database: str | None = None,
    min_size: int = 1,
    max_size: int = 10,
) -> asyncpg.Pool:
    """Create a pooled asyncpg connection with the pgvector codec registered
    on every connection (required so list[float] <-> the VECTOR column type
    convert automatically — without this, embeddings would need manual
    literal formatting on every query).

    Either dsn, or host/user/password/database, must be set — asyncpg
    accepts both forms natively, so this just threads whichever one the
    caller gave it straight through (mirrors MemoryConfig's own
    database_url-vs-host duality).
    """
    if dsn is None and not (host and user and password and database):
        raise ValueError(
            "create_pool: either dsn, or host/user/password/database, must be set."
        )

    async def _init(conn: asyncpg.Connection) -> None:
        await register_vector(conn)

    if dsn is not None:
        return await asyncpg.create_pool(
            dsn, min_size=min_size, max_size=max_size, init=_init
        )
    return await asyncpg.create_pool(
        host=host, port=port or 5432, user=user, password=password, database=database,
        min_size=min_size, max_size=max_size, init=_init,
    )

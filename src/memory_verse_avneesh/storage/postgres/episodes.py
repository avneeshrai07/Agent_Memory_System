"""Postgres + pgvector implementation of EpisodicStore.

Structurally satisfies memory_verse_avneesh.storage.interfaces.EpisodicStore —
Protocol conformance is duck-typed, no explicit inheritance required.

Episodes are immutable once written (no update_episode) — this is a record
of what actually happened, not a fact that gets corrected as understanding
improves. Only explicit deletion is supported, for user-requested removal.
"""

from __future__ import annotations

from uuid import UUID

import asyncpg

from memory_verse_avneesh.models import Episode, ScoredEpisode


class PostgresEpisodicStore:
    def __init__(self, pool: asyncpg.Pool, embedding_dim: int, schema: str):
        """schema is required, deliberately no default — same rationale as
        PostgresFactStore: a silent "public" default risks unrelated apps
        colliding on the same tables in a shared database.
        """
        self._pool = pool
        self._embedding_dim = embedding_dim
        self._schema = schema
        self._table = f'"{schema}".episodes'

    async def ensure_schema(self) -> None:
        """Idempotent. Requires pgvector >= 0.5.0 for the HNSW index type."""
        async with self._pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{self._schema}";')
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id UUID PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    assistant_message TEXT NOT NULL,
                    embedding VECTOR({self._embedding_dim}),
                    created_at TIMESTAMPTZ NOT NULL
                );
                """
            )
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS episodes_user_id_idx "
                f"ON {self._table} (user_id);"
            )
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS episodes_embedding_hnsw_idx "
                f"ON {self._table} USING hnsw (embedding vector_cosine_ops);"
            )

    async def add_episode(self, episode: Episode) -> Episode:
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self._table}
                    (id, user_id, conversation_id, user_message, assistant_message,
                     embedding, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                episode.id,
                episode.user_id,
                episode.conversation_id,
                episode.user_message,
                episode.assistant_message,
                episode.embedding,
                episode.created_at,
            )
        return episode

    async def get_episode(self, episode_id: UUID) -> Episode | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT * FROM {self._table} WHERE id = $1", episode_id
            )
        return self._row_to_episode(row) if row else None

    async def delete_episode(self, episode_id: UUID) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(f"DELETE FROM {self._table} WHERE id = $1", episode_id)

    async def search_episodes(
        self, user_id: str, embedding: list[float], limit: int
    ) -> list[ScoredEpisode]:
        """Cosine-similarity ANN search. Returns results already ordered by
        similarity descending — recency reranking happens above this, not
        here, per EpisodicStore.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT *, (1 - (embedding <=> $2)) AS similarity
                FROM {self._table}
                WHERE user_id = $1 AND embedding IS NOT NULL
                ORDER BY embedding <=> $2
                LIMIT $3
                """,
                user_id,
                embedding,
                limit,
            )
        return [
            ScoredEpisode(episode=self._row_to_episode(row), score=float(row["similarity"]))
            for row in rows
        ]

    async def list_episodes(
        self, user_id: str, limit: int, offset: int
    ) -> list[Episode]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT * FROM {self._table}
                WHERE user_id = $1
                ORDER BY created_at DESC
                LIMIT $2 OFFSET $3
                """,
                user_id,
                limit,
                offset,
            )
        return [self._row_to_episode(row) for row in rows]

    @staticmethod
    def _row_to_episode(row: asyncpg.Record) -> Episode:
        embedding = row["embedding"]
        if embedding is not None:
            # Same pgvector-python version split as PostgresFactStore._row_to_fact.
            embedding = (
                embedding.to_list() if hasattr(embedding, "to_list") else list(embedding)
            )
        return Episode(
            id=row["id"],
            user_id=row["user_id"],
            conversation_id=row["conversation_id"],
            user_message=row["user_message"],
            assistant_message=row["assistant_message"],
            embedding=embedding,
            created_at=row["created_at"],
        )

"""Postgres + pgvector implementation of FactStore (README Section 6).

Structurally satisfies memory_verse_avneesh.storage.interfaces.FactStore — Protocol
conformance is duck-typed, no explicit inheritance required.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

import asyncpg

from memory_verse_avneesh.models import MemoryFact, MemoryStatus, ScoredFact


class PostgresFactStore:
    def __init__(self, pool: asyncpg.Pool, embedding_dim: int, schema: str):
        """schema is required, deliberately no default. A silent "public"
        default risks two unrelated apps sharing a database colliding on
        the same memory_facts table without either one intending to share
        data — fail at construction time, not with confusing cross-tenant
        rows discovered later.
        """
        self._pool = pool
        self._embedding_dim = embedding_dim
        self._schema = schema
        self._table = f'"{schema}".memory_facts'

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
                    category TEXT NOT NULL,
                    value TEXT NOT NULL,
                    embedding VECTOR({self._embedding_dim}),
                    confidence DOUBLE PRECISION NOT NULL,
                    observation_count INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    last_reinforced_at TIMESTAMPTZ NOT NULL
                );
                """
            )
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS memory_facts_user_id_idx "
                f"ON {self._table} (user_id);"
            )
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS memory_facts_embedding_hnsw_idx "
                f"ON {self._table} USING hnsw (embedding vector_cosine_ops);"
            )

    async def add_fact(self, fact: MemoryFact) -> MemoryFact:
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self._table}
                    (id, user_id, category, value, embedding, confidence,
                     observation_count, status, created_at, last_reinforced_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                fact.id,
                fact.user_id,
                fact.category,
                fact.value,
                fact.embedding,
                fact.confidence,
                fact.observation_count,
                fact.status.value,
                fact.created_at,
                fact.last_reinforced_at,
            )
        return fact

    async def get_fact(self, fact_id: UUID) -> MemoryFact | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT * FROM {self._table} WHERE id = $1", fact_id
            )
        return self._row_to_fact(row) if row else None

    async def update_fact(self, fact: MemoryFact) -> MemoryFact:
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                UPDATE {self._table}
                SET category = $2, value = $3, embedding = $4, confidence = $5,
                    observation_count = $6, status = $7, last_reinforced_at = $8
                WHERE id = $1
                """,
                fact.id,
                fact.category,
                fact.value,
                fact.embedding,
                fact.confidence,
                fact.observation_count,
                fact.status.value,
                fact.last_reinforced_at,
            )
        return fact

    async def delete_fact(self, fact_id: UUID) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(f"DELETE FROM {self._table} WHERE id = $1", fact_id)

    async def list_facts(
        self, user_id: str, limit: int, offset: int
    ) -> list[MemoryFact]:
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
        return [self._row_to_fact(row) for row in rows]

    async def list_decayable_facts(
        self, older_than: datetime, limit: int
    ) -> list[MemoryFact]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT * FROM {self._table}
                WHERE status IN ('active', 'provisional')
                  AND last_reinforced_at < $1
                ORDER BY last_reinforced_at ASC
                LIMIT $2
                """,
                older_than,
                limit,
            )
        return [self._row_to_fact(row) for row in rows]

    async def search_facts(
        self, user_id: str, embedding: list[float], limit: int
    ) -> list[ScoredFact]:
        """Cosine-similarity ANN search over active facts only. Returns
        results already ordered by similarity descending — the read path's
        rerank stage (recency/importance/type weighting) happens above this,
        not here, per memory_verse_avneesh.storage.interfaces.FactStore.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT *, (1 - (embedding <=> $2)) AS similarity
                FROM {self._table}
                WHERE user_id = $1 AND status = 'active' AND embedding IS NOT NULL
                ORDER BY embedding <=> $2
                LIMIT $3
                """,
                user_id,
                embedding,
                limit,
            )
        return [
            ScoredFact(fact=self._row_to_fact(row), score=float(row["similarity"]))
            for row in rows
        ]

    @staticmethod
    def _row_to_fact(row: asyncpg.Record) -> MemoryFact:
        embedding = row["embedding"]
        return MemoryFact(
            id=row["id"],
            user_id=row["user_id"],
            category=row["category"],
            value=row["value"],
            embedding=list(embedding) if embedding is not None else None,
            confidence=row["confidence"],
            observation_count=row["observation_count"],
            status=MemoryStatus(row["status"]),
            created_at=row["created_at"],
            last_reinforced_at=row["last_reinforced_at"],
        )

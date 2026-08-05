"""Postgres + pgvector implementation of GraphStore.

Structurally satisfies memory_verse_avneesh.storage.interfaces.GraphStore —
Protocol conformance is duck-typed, no explicit inheritance required.

Two tables: entities (plain rows, no embedding) and memory_edges (embeds
fact_sentence, not the entity names -- see Edge's own docstring for why).
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

import asyncpg

from memory_verse_avneesh.models import Edge, Entity, ScoredEdge


class PostgresGraphStore:
    def __init__(self, pool: asyncpg.Pool, embedding_dim: int, schema: str):
        """schema is required, deliberately no default — same rationale as
        PostgresFactStore: a silent "public" default risks unrelated apps
        colliding on the same tables in a shared database.
        """
        self._pool = pool
        self._embedding_dim = embedding_dim
        self._schema = schema
        self._entities_table = f'"{schema}".entities'
        self._edges_table = f'"{schema}".memory_edges'

    async def ensure_schema(self) -> None:
        """Idempotent. Requires pgvector >= 0.5.0 for the HNSW index type."""
        async with self._pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{self._schema}";')
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._entities_table} (
                    id UUID PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    entity_type TEXT,
                    aliases TEXT[] NOT NULL DEFAULT '{{}}',
                    created_at TIMESTAMPTZ NOT NULL
                );
                """
            )
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS entities_user_name_idx "
                f"ON {self._entities_table} (user_id, lower(name));"
            )
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._edges_table} (
                    id UUID PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    source_entity_id UUID NOT NULL,
                    relation TEXT NOT NULL,
                    target_entity_id UUID,
                    target_value TEXT,
                    fact_sentence TEXT NOT NULL,
                    embedding VECTOR({self._embedding_dim}),
                    confidence DOUBLE PRECISION NOT NULL,
                    valid_from TIMESTAMPTZ NOT NULL,
                    valid_to TIMESTAMPTZ,
                    observed_at TIMESTAMPTZ NOT NULL,
                    recorded_at TIMESTAMPTZ NOT NULL
                );
                """
            )
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS edges_current_lookup_idx "
                f"ON {self._edges_table} (user_id, source_entity_id, relation) "
                f"WHERE valid_to IS NULL;"
            )
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS edges_source_idx "
                f"ON {self._edges_table} (source_entity_id);"
            )
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS edges_target_idx "
                f"ON {self._edges_table} (target_entity_id);"
            )
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS edges_embedding_hnsw_idx "
                f"ON {self._edges_table} USING hnsw (embedding vector_cosine_ops);"
            )

    # --- entities -----------------------------------------------------

    async def create_entity(self, entity: Entity) -> Entity:
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self._entities_table}
                    (id, user_id, name, entity_type, aliases, created_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                entity.id,
                entity.user_id,
                entity.name,
                entity.entity_type,
                entity.aliases,
                entity.created_at,
            )
        return entity

    async def get_entity(self, entity_id: UUID) -> Entity | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT * FROM {self._entities_table} WHERE id = $1", entity_id
            )
        return self._row_to_entity(row) if row else None

    async def find_entity_by_name(self, user_id: str, name: str) -> Entity | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT * FROM {self._entities_table}
                WHERE user_id = $1
                  AND (lower(name) = lower($2)
                       OR lower($2) = ANY(SELECT lower(a) FROM unnest(aliases) AS a))
                LIMIT 1
                """,
                user_id,
                name,
            )
        return self._row_to_entity(row) if row else None

    async def delete_entity(self, entity_id: UUID) -> None:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    f"DELETE FROM {self._edges_table} "
                    f"WHERE source_entity_id = $1 OR target_entity_id = $1",
                    entity_id,
                )
                await conn.execute(
                    f"DELETE FROM {self._entities_table} WHERE id = $1", entity_id
                )

    async def list_entities(self, user_id: str, limit: int, offset: int) -> list[Entity]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT * FROM {self._entities_table}
                WHERE user_id = $1
                ORDER BY created_at DESC
                LIMIT $2 OFFSET $3
                """,
                user_id,
                limit,
                offset,
            )
        return [self._row_to_entity(row) for row in rows]

    # --- edges ----------------------------------------------------------

    async def add_edge(self, edge: Edge) -> Edge:
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self._edges_table}
                    (id, user_id, source_entity_id, relation, target_entity_id,
                     target_value, fact_sentence, embedding, confidence,
                     valid_from, valid_to, observed_at, recorded_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """,
                edge.id,
                edge.user_id,
                edge.source_entity_id,
                edge.relation,
                edge.target_entity_id,
                edge.target_value,
                edge.fact_sentence,
                edge.embedding,
                edge.confidence,
                edge.valid_from,
                edge.valid_to,
                edge.observed_at,
                edge.recorded_at,
            )
        return edge

    async def get_edge(self, edge_id: UUID) -> Edge | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT * FROM {self._edges_table} WHERE id = $1", edge_id
            )
        return self._row_to_edge(row) if row else None

    async def get_current_edge(
        self, user_id: str, source_entity_id: UUID, relation: str
    ) -> Edge | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT * FROM {self._edges_table}
                WHERE user_id = $1 AND source_entity_id = $2 AND relation = $3
                  AND valid_to IS NULL
                LIMIT 1
                """,
                user_id,
                source_entity_id,
                relation,
            )
        return self._row_to_edge(row) if row else None

    async def close_edge(self, edge_id: UUID, valid_to: datetime) -> Edge:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                UPDATE {self._edges_table}
                SET valid_to = $2
                WHERE id = $1
                RETURNING *
                """,
                edge_id,
                valid_to,
            )
        return self._row_to_edge(row)

    async def delete_edge(self, edge_id: UUID) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(f"DELETE FROM {self._edges_table} WHERE id = $1", edge_id)

    async def search_current_edges(
        self, user_id: str, embedding: list[float], limit: int
    ) -> list[ScoredEdge]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT *, (1 - (embedding <=> $2)) AS similarity
                FROM {self._edges_table}
                WHERE user_id = $1 AND valid_to IS NULL AND embedding IS NOT NULL
                ORDER BY embedding <=> $2
                LIMIT $3
                """,
                user_id,
                embedding,
                limit,
            )
        return [
            ScoredEdge(edge=self._row_to_edge(row), score=float(row["similarity"]))
            for row in rows
        ]

    async def list_edges_for_entity(
        self, entity_id: UUID, limit: int, offset: int
    ) -> list[Edge]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT * FROM {self._edges_table}
                WHERE source_entity_id = $1 OR target_entity_id = $1
                ORDER BY recorded_at DESC
                LIMIT $2 OFFSET $3
                """,
                entity_id,
                limit,
                offset,
            )
        return [self._row_to_edge(row) for row in rows]

    @staticmethod
    def _row_to_entity(row: asyncpg.Record) -> Entity:
        return Entity(
            id=row["id"],
            user_id=row["user_id"],
            name=row["name"],
            entity_type=row["entity_type"],
            aliases=list(row["aliases"]) if row["aliases"] is not None else [],
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_edge(row: asyncpg.Record) -> Edge:
        embedding = row["embedding"]
        if embedding is not None:
            # Same pgvector-python version split as PostgresFactStore._row_to_fact.
            embedding = (
                embedding.to_list() if hasattr(embedding, "to_list") else list(embedding)
            )
        return Edge(
            id=row["id"],
            user_id=row["user_id"],
            source_entity_id=row["source_entity_id"],
            relation=row["relation"],
            target_entity_id=row["target_entity_id"],
            target_value=row["target_value"],
            fact_sentence=row["fact_sentence"],
            embedding=embedding,
            confidence=row["confidence"],
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
            observed_at=row["observed_at"],
            recorded_at=row["recorded_at"],
        )

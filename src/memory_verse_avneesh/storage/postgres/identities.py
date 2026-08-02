"""Postgres implementation of IdentityStore.

Structurally satisfies memory_verse_avneesh.storage.interfaces.IdentityStore —
Protocol conformance is duck-typed, no explicit inheritance required.

Two tables, deliberately separate (not a single polymorphic table): expert
identities are host-authored personas keyed by an arbitrary string id the
host chooses, person identities are one durable record per user_id. Neither
is written by the formation pipeline — both are plain CRUD over data the
host (or its own management UI) writes explicitly.
"""

from __future__ import annotations

from datetime import datetime, timezone

import asyncpg

from memory_verse_avneesh.models import ExpertIdentity, PersonIdentity


class PostgresIdentityStore:
    def __init__(self, pool: asyncpg.Pool, schema: str):
        """schema is required, deliberately no default — same rationale as
        PostgresFactStore: a silent "public" default risks unrelated apps
        colliding on the same tables in a shared database.
        """
        self._pool = pool
        self._schema = schema
        self._expert_table = f'"{schema}".expert_identities'
        self._person_table = f'"{schema}".person_identities'

    async def ensure_schema(self) -> None:
        """Idempotent."""
        async with self._pool.acquire() as conn:
            await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{self._schema}";')
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._expert_table} (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL
                );
                """
            )
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._person_table} (
                    user_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL
                );
                """
            )

    async def create_expert_identity(self, identity: ExpertIdentity) -> ExpertIdentity:
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self._expert_table} (id, content, created_at, updated_at)
                VALUES ($1, $2, $3, $4)
                """,
                identity.id,
                identity.content,
                identity.created_at,
                identity.updated_at,
            )
        return identity

    async def get_expert_identity(self, identity_id: str) -> ExpertIdentity | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT * FROM {self._expert_table} WHERE id = $1", identity_id
            )
        return self._row_to_expert_identity(row) if row else None

    async def update_expert_identity(self, identity: ExpertIdentity) -> ExpertIdentity:
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                UPDATE {self._expert_table}
                SET content = $2, updated_at = $3
                WHERE id = $1
                """,
                identity.id,
                identity.content,
                identity.updated_at,
            )
        return identity

    async def delete_expert_identity(self, identity_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {self._expert_table} WHERE id = $1", identity_id
            )

    async def list_expert_identities(self, limit: int, offset: int) -> list[ExpertIdentity]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT * FROM {self._expert_table}
                ORDER BY created_at DESC
                LIMIT $1 OFFSET $2
                """,
                limit,
                offset,
            )
        return [self._row_to_expert_identity(row) for row in rows]

    async def get_person_identity(self, user_id: str) -> PersonIdentity | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT * FROM {self._person_table} WHERE user_id = $1", user_id
            )
        return self._row_to_person_identity(row) if row else None

    async def set_person_identity(self, user_id: str, content: str) -> PersonIdentity:
        now = datetime.now(timezone.utc)
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                INSERT INTO {self._person_table} (user_id, content, created_at, updated_at)
                VALUES ($1, $2, $3, $3)
                ON CONFLICT (user_id) DO UPDATE
                SET content = EXCLUDED.content, updated_at = EXCLUDED.updated_at
                RETURNING *
                """,
                user_id,
                content,
                now,
            )
        return self._row_to_person_identity(row)

    async def delete_person_identity(self, user_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {self._person_table} WHERE user_id = $1", user_id
            )

    async def list_person_identities(self, limit: int, offset: int) -> list[PersonIdentity]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT * FROM {self._person_table}
                ORDER BY created_at DESC
                LIMIT $1 OFFSET $2
                """,
                limit,
                offset,
            )
        return [self._row_to_person_identity(row) for row in rows]

    @staticmethod
    def _row_to_expert_identity(row: asyncpg.Record) -> ExpertIdentity:
        return ExpertIdentity(
            id=row["id"],
            content=row["content"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _row_to_person_identity(row: asyncpg.Record) -> PersonIdentity:
        return PersonIdentity(
            user_id=row["user_id"],
            content=row["content"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

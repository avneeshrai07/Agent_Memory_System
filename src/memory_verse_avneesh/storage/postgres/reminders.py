"""Postgres implementation of ReminderStore.

Structurally satisfies memory_verse_avneesh.storage.interfaces.ReminderStore —
Protocol conformance is duck-typed, no explicit inheritance required.

No embedding column here, unlike facts/episodes: due-reminder lookup is a
plain deterministic time comparison, not a similarity search.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

import asyncpg

from memory_verse_avneesh.models import Reminder, ReminderStatus


class PostgresReminderStore:
    def __init__(self, pool: asyncpg.Pool, schema: str):
        """schema is required, deliberately no default — same rationale as
        PostgresFactStore: a silent "public" default risks unrelated apps
        colliding on the same tables in a shared database.
        """
        self._pool = pool
        self._schema = schema
        self._table = f'"{schema}".reminders'

    async def ensure_schema(self) -> None:
        """Idempotent."""
        async with self._pool.acquire() as conn:
            await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{self._schema}";')
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id UUID PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    due_at TIMESTAMPTZ NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    completed_at TIMESTAMPTZ
                );
                """
            )
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS reminders_due_lookup_idx "
                f"ON {self._table} (user_id, status, due_at);"
            )

    async def create_reminder(self, reminder: Reminder) -> Reminder:
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self._table}
                    (id, user_id, content, due_at, status, created_at, completed_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                reminder.id,
                reminder.user_id,
                reminder.content,
                reminder.due_at,
                reminder.status.value,
                reminder.created_at,
                reminder.completed_at,
            )
        return reminder

    async def get_reminder(self, reminder_id: UUID) -> Reminder | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT * FROM {self._table} WHERE id = $1", reminder_id
            )
        return self._row_to_reminder(row) if row else None

    async def update_reminder(self, reminder: Reminder) -> Reminder:
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                UPDATE {self._table}
                SET content = $2, due_at = $3, status = $4, completed_at = $5
                WHERE id = $1
                """,
                reminder.id,
                reminder.content,
                reminder.due_at,
                reminder.status.value,
                reminder.completed_at,
            )
        return reminder

    async def delete_reminder(self, reminder_id: UUID) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(f"DELETE FROM {self._table} WHERE id = $1", reminder_id)

    async def list_reminders(
        self, user_id: str, limit: int, offset: int
    ) -> list[Reminder]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT * FROM {self._table}
                WHERE user_id = $1
                ORDER BY due_at DESC
                LIMIT $2 OFFSET $3
                """,
                user_id,
                limit,
                offset,
            )
        return [self._row_to_reminder(row) for row in rows]

    async def list_due_reminders(self, user_id: str, as_of: datetime) -> list[Reminder]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT * FROM {self._table}
                WHERE user_id = $1 AND status = $2 AND due_at <= $3
                ORDER BY due_at ASC
                """,
                user_id,
                ReminderStatus.PENDING.value,
                as_of,
            )
        return [self._row_to_reminder(row) for row in rows]

    @staticmethod
    def _row_to_reminder(row: asyncpg.Record) -> Reminder:
        return Reminder(
            id=row["id"],
            user_id=row["user_id"],
            content=row["content"],
            due_at=row["due_at"],
            status=ReminderStatus(row["status"]),
            created_at=row["created_at"],
            completed_at=row["completed_at"],
        )

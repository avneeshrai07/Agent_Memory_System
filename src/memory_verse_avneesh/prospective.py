"""Prospective memory management: create, view, mark done/dismissed, delete
reminders.

Distinct from management.py/identity.py/episodic.py in one way: there is no
formation-pipeline path that ever creates a Reminder automatically. Every
reminder starts here, created explicitly by the host (its own code, or its
own LLM calling this as a tool during generation) — read_memory() only ever
reads what's already due, never decides to create one.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from memory_verse_avneesh.models import Reminder, ReminderStatus
from memory_verse_avneesh.storage.interfaces import ReminderStore


class ReminderNotFoundError(Exception):
    def __init__(self, reminder_id: UUID):
        super().__init__(f"No reminder with id {reminder_id}")
        self.reminder_id = reminder_id


async def create_reminder(
    user_id: str, content: str, due_at: datetime, *, reminder_store: ReminderStore
) -> Reminder:
    reminder = Reminder(user_id=user_id, content=content, due_at=due_at)
    return await reminder_store.create_reminder(reminder)


async def get_reminder(reminder_id: UUID, *, reminder_store: ReminderStore) -> Reminder:
    reminder = await reminder_store.get_reminder(reminder_id)
    if reminder is None:
        raise ReminderNotFoundError(reminder_id)
    return reminder


async def mark_done(reminder_id: UUID, *, reminder_store: ReminderStore) -> Reminder:
    existing = await reminder_store.get_reminder(reminder_id)
    if existing is None:
        raise ReminderNotFoundError(reminder_id)

    updated = existing.model_copy(
        update={"status": ReminderStatus.DONE, "completed_at": datetime.now(timezone.utc)}
    )
    return await reminder_store.update_reminder(updated)


async def dismiss_reminder(reminder_id: UUID, *, reminder_store: ReminderStore) -> Reminder:
    existing = await reminder_store.get_reminder(reminder_id)
    if existing is None:
        raise ReminderNotFoundError(reminder_id)

    updated = existing.model_copy(
        update={"status": ReminderStatus.DISMISSED, "completed_at": datetime.now(timezone.utc)}
    )
    return await reminder_store.update_reminder(updated)


async def delete_reminder(reminder_id: UUID, *, reminder_store: ReminderStore) -> None:
    await reminder_store.delete_reminder(reminder_id)


async def list_reminders(
    user_id: str, *, reminder_store: ReminderStore, limit: int = 50, offset: int = 0
) -> list[Reminder]:
    return await reminder_store.list_reminders(user_id, limit, offset)


async def list_due_reminders(
    user_id: str, *, reminder_store: ReminderStore, as_of: datetime | None = None
) -> list[Reminder]:
    return await reminder_store.list_due_reminders(user_id, as_of or datetime.now(timezone.utc))

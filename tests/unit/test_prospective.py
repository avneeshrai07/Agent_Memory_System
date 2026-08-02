from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from memory_verse_avneesh.models import ReminderStatus
from memory_verse_avneesh.prospective import (
    ReminderNotFoundError,
    create_reminder,
    delete_reminder,
    dismiss_reminder,
    get_reminder,
    list_due_reminders,
    list_reminders,
    mark_done,
)

from .fakes import FakeReminderStore


def _future(hours: int = 1) -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=hours)


def _past(hours: int = 1) -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=hours)


async def test_create_and_get_reminder():
    store = FakeReminderStore()
    created = await create_reminder("u1", "follow up with client", _future(), reminder_store=store)

    assert created.status == ReminderStatus.PENDING
    fetched = await get_reminder(created.id, reminder_store=store)
    assert fetched.content == "follow up with client"


async def test_get_reminder_raises_when_missing():
    store = FakeReminderStore()
    with pytest.raises(ReminderNotFoundError):
        await get_reminder(uuid4(), reminder_store=store)


async def test_mark_done_sets_status_and_completed_at():
    store = FakeReminderStore()
    reminder = await create_reminder("u1", "call back", _future(), reminder_store=store)

    done = await mark_done(reminder.id, reminder_store=store)

    assert done.status == ReminderStatus.DONE
    assert done.completed_at is not None


async def test_dismiss_reminder_sets_status_and_completed_at():
    store = FakeReminderStore()
    reminder = await create_reminder("u1", "call back", _future(), reminder_store=store)

    dismissed = await dismiss_reminder(reminder.id, reminder_store=store)

    assert dismissed.status == ReminderStatus.DISMISSED
    assert dismissed.completed_at is not None


async def test_mark_done_raises_when_missing():
    store = FakeReminderStore()
    with pytest.raises(ReminderNotFoundError):
        await mark_done(uuid4(), reminder_store=store)


async def test_delete_reminder_is_idempotent():
    store = FakeReminderStore()
    reminder = await create_reminder("u1", "call back", _future(), reminder_store=store)

    await delete_reminder(reminder.id, reminder_store=store)
    await delete_reminder(reminder.id, reminder_store=store)  # no error

    with pytest.raises(ReminderNotFoundError):
        await get_reminder(reminder.id, reminder_store=store)


async def test_list_reminders_returns_all_statuses():
    store = FakeReminderStore()
    pending = await create_reminder("u1", "pending one", _future(), reminder_store=store)
    done_one = await create_reminder("u1", "done one", _future(), reminder_store=store)
    await mark_done(done_one.id, reminder_store=store)

    reminders = await list_reminders("u1", reminder_store=store)
    contents = {r.content for r in reminders}
    assert contents == {"pending one", "done one"}


async def test_list_due_reminders_only_returns_pending_and_past_due():
    store = FakeReminderStore()
    overdue = await create_reminder("u1", "overdue", _past(), reminder_store=store)
    not_yet_due = await create_reminder("u1", "not yet due", _future(), reminder_store=store)
    already_done = await create_reminder("u1", "already done", _past(), reminder_store=store)
    await mark_done(already_done.id, reminder_store=store)

    due = await list_due_reminders("u1", reminder_store=store)

    contents = {r.content for r in due}
    assert contents == {"overdue"}
    assert not_yet_due.id not in [r.id for r in due]
    assert already_done.id not in [r.id for r in due]


async def test_list_due_reminders_stays_due_after_due_at_passes_until_explicitly_resolved():
    store = FakeReminderStore()
    reminder = await create_reminder("u1", "keeps showing up", _past(), reminder_store=store)

    first_check = await list_due_reminders("u1", reminder_store=store)
    second_check = await list_due_reminders("u1", reminder_store=store)

    assert [r.id for r in first_check] == [reminder.id]
    assert [r.id for r in second_check] == [reminder.id]  # still there -- not silently dropped

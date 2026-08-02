import pytest

from memory_verse_avneesh.identity import (
    IdentityNotFoundError,
    create_expert_identity,
    delete_expert_identity,
    delete_person_identity,
    get_expert_identity,
    get_person_identity,
    list_expert_identities,
    list_person_identities,
    set_person_identity,
    update_expert_identity,
)

from .fakes import FakeIdentityStore


async def test_create_and_get_expert_identity():
    store = FakeIdentityStore()
    created = await create_expert_identity(
        "expert_email_writer", "write concise, persuasive emails", identity_store=store
    )

    assert created.id == "expert_email_writer"
    fetched = await get_expert_identity("expert_email_writer", identity_store=store)
    assert fetched.content == "write concise, persuasive emails"


async def test_get_expert_identity_raises_when_missing():
    store = FakeIdentityStore()
    with pytest.raises(IdentityNotFoundError):
        await get_expert_identity("does_not_exist", identity_store=store)


async def test_update_expert_identity_changes_content_and_raises_updated_at():
    store = FakeIdentityStore()
    created = await create_expert_identity("expert_email_writer", "v1", identity_store=store)

    updated = await update_expert_identity("expert_email_writer", "v2", identity_store=store)

    assert updated.content == "v2"
    assert updated.updated_at >= created.updated_at
    assert updated.created_at == created.created_at


async def test_update_expert_identity_raises_when_missing():
    store = FakeIdentityStore()
    with pytest.raises(IdentityNotFoundError):
        await update_expert_identity("does_not_exist", "v2", identity_store=store)


async def test_delete_expert_identity_is_idempotent():
    store = FakeIdentityStore()
    await create_expert_identity("expert_email_writer", "v1", identity_store=store)

    await delete_expert_identity("expert_email_writer", identity_store=store)
    await delete_expert_identity("expert_email_writer", identity_store=store)  # no error

    with pytest.raises(IdentityNotFoundError):
        await get_expert_identity("expert_email_writer", identity_store=store)


async def test_list_expert_identities():
    store = FakeIdentityStore()
    await create_expert_identity("expert_email_writer", "v1", identity_store=store)
    await create_expert_identity("expert_support_agent", "v1", identity_store=store)

    identities = await list_expert_identities(identity_store=store)
    ids = {i.id for i in identities}
    assert ids == {"expert_email_writer", "expert_support_agent"}


async def test_set_person_identity_upserts():
    store = FakeIdentityStore()
    first = await set_person_identity("u1", "prefers formal tone", identity_store=store)
    second = await set_person_identity("u1", "prefers casual tone", identity_store=store)

    assert first.created_at == second.created_at  # same record, not a new one
    current = await get_person_identity("u1", identity_store=store)
    assert current.content == "prefers casual tone"


async def test_get_person_identity_returns_none_when_absent():
    store = FakeIdentityStore()
    assert await get_person_identity("nobody", identity_store=store) is None


async def test_delete_person_identity_is_idempotent():
    store = FakeIdentityStore()
    await set_person_identity("u1", "prefers formal tone", identity_store=store)

    await delete_person_identity("u1", identity_store=store)
    await delete_person_identity("u1", identity_store=store)  # no error

    assert await get_person_identity("u1", identity_store=store) is None


async def test_list_person_identities():
    store = FakeIdentityStore()
    await set_person_identity("u1", "a", identity_store=store)
    await set_person_identity("u2", "b", identity_store=store)

    identities = await list_person_identities(identity_store=store)
    user_ids = {i.user_id for i in identities}
    assert user_ids == {"u1", "u2"}

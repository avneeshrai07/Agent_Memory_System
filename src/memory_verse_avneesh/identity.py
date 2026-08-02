"""Identity management: host-authored expert personas and durable per-person
identity records.

Distinct from management.py (which is plain CRUD over Tier 2 facts learned
by the formation pipeline): nothing here is ever written by extraction,
resolution, or the safety gate. Both expert identities and person identities
are written explicitly, only through this module, by the host application
(e.g. its own admin UI or onboarding flow) — read_memory() only ever reads
them (README's identity section).
"""

from __future__ import annotations

from datetime import datetime, timezone

from memory_verse_avneesh.models import ExpertIdentity, PersonIdentity
from memory_verse_avneesh.storage.interfaces import IdentityStore


class IdentityNotFoundError(Exception):
    def __init__(self, identity_id: str):
        super().__init__(f"No expert identity with id {identity_id!r}")
        self.identity_id = identity_id


async def create_expert_identity(
    identity_id: str, content: str, *, identity_store: IdentityStore
) -> ExpertIdentity:
    now = datetime.now(timezone.utc)
    identity = ExpertIdentity(id=identity_id, content=content, created_at=now, updated_at=now)
    return await identity_store.create_expert_identity(identity)


async def get_expert_identity(
    identity_id: str, *, identity_store: IdentityStore
) -> ExpertIdentity:
    identity = await identity_store.get_expert_identity(identity_id)
    if identity is None:
        raise IdentityNotFoundError(identity_id)
    return identity


async def update_expert_identity(
    identity_id: str, new_content: str, *, identity_store: IdentityStore
) -> ExpertIdentity:
    existing = await identity_store.get_expert_identity(identity_id)
    if existing is None:
        raise IdentityNotFoundError(identity_id)

    updated = existing.model_copy(
        update={"content": new_content, "updated_at": datetime.now(timezone.utc)}
    )
    return await identity_store.update_expert_identity(updated)


async def delete_expert_identity(identity_id: str, *, identity_store: IdentityStore) -> None:
    await identity_store.delete_expert_identity(identity_id)


async def list_expert_identities(
    *, identity_store: IdentityStore, limit: int = 50, offset: int = 0
) -> list[ExpertIdentity]:
    return await identity_store.list_expert_identities(limit, offset)


async def get_person_identity(
    user_id: str, *, identity_store: IdentityStore
) -> PersonIdentity | None:
    return await identity_store.get_person_identity(user_id)


async def set_person_identity(
    user_id: str, content: str, *, identity_store: IdentityStore
) -> PersonIdentity:
    """Upsert — one record per user_id."""
    return await identity_store.set_person_identity(user_id, content)


async def delete_person_identity(user_id: str, *, identity_store: IdentityStore) -> None:
    await identity_store.delete_person_identity(user_id)


async def list_person_identities(
    *, identity_store: IdentityStore, limit: int = 50, offset: int = 0
) -> list[PersonIdentity]:
    return await identity_store.list_person_identities(limit, offset)

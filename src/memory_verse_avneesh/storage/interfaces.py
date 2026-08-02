"""Storage backend contracts. Concrete Postgres/Redis implementations satisfy
these; a host application may substitute its own. Nothing in the read or
formation path should import a concrete backend directly — only these.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable
from uuid import UUID

from memory_verse_avneesh.models import MemoryFact, ScoredFact, Turn


@runtime_checkable
class FactStore(Protocol):
    """Tier 2 (vector) memory: durable, embedded facts."""

    async def add_fact(self, fact: MemoryFact) -> MemoryFact: ...

    async def get_fact(self, fact_id: UUID) -> MemoryFact | None: ...

    async def update_fact(self, fact: MemoryFact) -> MemoryFact: ...

    async def delete_fact(self, fact_id: UUID) -> None:
        """Permanent removal — the backing operation for user-requested
        deletion (README Section 7). Idempotent: deleting an id that's
        already gone is not an error.
        """
        ...

    async def search_facts(
        self, user_id: str, embedding: list[float], limit: int
    ) -> list[ScoredFact]:
        """Approximate nearest-neighbor search, already ranked by similarity.
        Recency/importance/type reranking happens above this layer, not here.
        """
        ...

    async def list_facts(
        self, user_id: str, limit: int, offset: int
    ) -> list[MemoryFact]:
        """Plain paginated listing, ordered newest-first — for a host's
        user-facing "view your memories" surface (README Section 7), not a
        similarity search. All statuses included; the host filters if it
        only wants to show what's currently active.
        """
        ...

    async def list_decayable_facts(
        self, older_than: datetime, limit: int
    ) -> list[MemoryFact]:
        """Active/provisional facts not reinforced since `older_than` —
        consumed only by the batched decay sweep (README Section 5, step 7),
        never by the read or formation path.
        """
        ...


@runtime_checkable
class SessionCache(Protocol):
    """Tier 0: rolling recent-turns buffer per conversation. O(1) reads."""

    async def get_recent_turns(self, conversation_id: str, limit: int) -> list[Turn]: ...

    async def append_turn(self, turn: Turn) -> None: ...


@runtime_checkable
class ProfileCache(Protocol):
    """Tier 1: precomputed, denormalized per-user profile blob. O(1) reads."""

    async def get_profile(self, user_id: str) -> dict | None: ...

    async def set_profile(self, user_id: str, profile: dict) -> None: ...

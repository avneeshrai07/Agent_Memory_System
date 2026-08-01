"""Storage backend contracts. Concrete Postgres/Redis implementations satisfy
these; a host application may substitute its own. Nothing in the read or
formation path should import a concrete backend directly — only these.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from uuid import UUID

from agent_memory.models import MemoryFact, ScoredFact, Turn


@runtime_checkable
class FactStore(Protocol):
    """Tier 2 (vector) memory: durable, embedded facts."""

    async def add_fact(self, fact: MemoryFact) -> MemoryFact: ...

    async def get_fact(self, fact_id: UUID) -> MemoryFact | None: ...

    async def update_fact(self, fact: MemoryFact) -> MemoryFact: ...

    async def search_facts(
        self, user_id: str, embedding: list[float], limit: int
    ) -> list[ScoredFact]:
        """Approximate nearest-neighbor search, already ranked by similarity.
        Recency/importance/type reranking happens above this layer, not here.
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

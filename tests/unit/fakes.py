"""In-memory fakes satisfying the storage/llm Protocols, for unit-testing
pipeline control flow without a real Postgres/Redis/Bedrock connection.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from memory_verse_avneesh.models import (
    ExtractedCandidate,
    MemoryFact,
    MemoryStatus,
    ResolvedOperation,
    ScoredFact,
    Turn,
)


class FakeSessionCache:
    def __init__(self, turns: list[Turn] | None = None):
        self._turns = turns or []

    async def get_recent_turns(self, conversation_id: str, limit: int) -> list[Turn]:
        return self._turns[-limit:]

    async def append_turn(self, turn: Turn) -> None:
        self._turns.append(turn)


class FakeProfileCache:
    def __init__(self, profile: dict | None = None):
        self._profile = profile

    async def get_profile(self, user_id: str) -> dict | None:
        return self._profile

    async def set_profile(self, user_id: str, profile: dict) -> None:
        self._profile = profile


class FakeFactStore:
    def __init__(self, search_results: list[ScoredFact] | None = None):
        self._search_results = search_results or []
        self.added: list[MemoryFact] = []
        self.updated: list[MemoryFact] = []
        self.deleted: list[UUID] = []

    def _current(self) -> list[MemoryFact]:
        by_id: dict[UUID, MemoryFact] = {}
        for fact in self.added:
            by_id[fact.id] = fact
        for fact in self.updated:
            by_id[fact.id] = fact
        return [f for fid, f in by_id.items() if fid not in self.deleted]

    async def add_fact(self, fact: MemoryFact) -> MemoryFact:
        self.added.append(fact)
        return fact

    async def get_fact(self, fact_id: UUID) -> MemoryFact | None:
        return next((f for f in self._current() if f.id == fact_id), None)

    async def update_fact(self, fact: MemoryFact) -> MemoryFact:
        self.updated.append(fact)
        return fact

    async def delete_fact(self, fact_id: UUID) -> None:
        self.deleted.append(fact_id)

    async def search_facts(
        self, user_id: str, embedding: list[float], limit: int
    ) -> list[ScoredFact]:
        return self._search_results[:limit]

    async def list_facts(self, user_id: str, limit: int, offset: int) -> list[MemoryFact]:
        matching = [f for f in self._current() if f.user_id == user_id]
        matching.sort(key=lambda f: f.created_at, reverse=True)
        return matching[offset : offset + limit]

    async def list_decayable_facts(
        self, older_than: datetime, limit: int
    ) -> list[MemoryFact]:
        eligible = [
            f
            for f in self._current()
            if f.status in (MemoryStatus.ACTIVE, MemoryStatus.PROVISIONAL)
            and f.last_reinforced_at < older_than
        ]
        eligible.sort(key=lambda f: f.last_reinforced_at)
        return eligible[:limit]


class FakeEmbeddingClient:
    def __init__(self, vector: list[float] | None = None):
        self._vector = vector or [0.1, 0.2, 0.3]
        self.embedded_texts: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.embedded_texts.append(text)
        return self._vector


class FakeExtractionClient:
    def __init__(self, candidates: list[ExtractedCandidate] | None = None):
        self._candidates = candidates or []

    async def extract(self, turn: Turn) -> list[ExtractedCandidate]:
        return self._candidates


class FakeResolutionClient:
    def __init__(self, resolution: ResolvedOperation | None = None):
        self._resolution = resolution or ResolvedOperation(operation="add")
        self.calls: list[tuple[ExtractedCandidate, list[ScoredFact]]] = []

    async def classify_operation(
        self, candidate: ExtractedCandidate, existing: list[ScoredFact]
    ) -> ResolvedOperation:
        self.calls.append((candidate, existing))
        return self._resolution

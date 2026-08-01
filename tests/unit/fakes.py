"""In-memory fakes satisfying the storage/llm Protocols, for unit-testing
pipeline control flow without a real Postgres/Redis/Bedrock connection.
"""

from __future__ import annotations

from uuid import UUID

from agent_memory.models import (
    ExtractedCandidate,
    MemoryFact,
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

    async def add_fact(self, fact: MemoryFact) -> MemoryFact:
        self.added.append(fact)
        return fact

    async def get_fact(self, fact_id: UUID) -> MemoryFact | None:
        return next((f for f in self.added if f.id == fact_id), None)

    async def update_fact(self, fact: MemoryFact) -> MemoryFact:
        return fact

    async def search_facts(
        self, user_id: str, embedding: list[float], limit: int
    ) -> list[ScoredFact]:
        return self._search_results[:limit]


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

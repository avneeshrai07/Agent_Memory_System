"""In-memory fakes satisfying the storage/llm Protocols, for unit-testing
pipeline control flow without a real Postgres/Redis/Bedrock connection.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from memory_verse_avneesh.models import (
    Edge,
    Entity,
    Episode,
    ExpertIdentity,
    ExtractedCandidate,
    MemoryFact,
    MemoryStatus,
    PersonIdentity,
    RelationCandidate,
    Reminder,
    ReminderStatus,
    ResolvedOperation,
    ScoredEdge,
    ScoredEpisode,
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


class FakeEpisodicStore:
    def __init__(self, search_results: list[ScoredEpisode] | None = None):
        self._search_results = search_results or []
        self._episodes: dict[UUID, Episode] = {}
        self.added: list[Episode] = []

    async def add_episode(self, episode: Episode) -> Episode:
        self._episodes[episode.id] = episode
        self.added.append(episode)
        return episode

    async def get_episode(self, episode_id: UUID) -> Episode | None:
        return self._episodes.get(episode_id)

    async def delete_episode(self, episode_id: UUID) -> None:
        self._episodes.pop(episode_id, None)

    async def search_episodes(
        self, user_id: str, embedding: list[float], limit: int
    ) -> list[ScoredEpisode]:
        return self._search_results[:limit]

    async def list_episodes(self, user_id: str, limit: int, offset: int) -> list[Episode]:
        matching = [e for e in self._episodes.values() if e.user_id == user_id]
        matching.sort(key=lambda e: e.created_at, reverse=True)
        return matching[offset : offset + limit]


class FakeIdentityStore:
    def __init__(
        self,
        expert_identities: dict[str, ExpertIdentity] | None = None,
        person_identities: dict[str, PersonIdentity] | None = None,
    ):
        self._expert = dict(expert_identities or {})
        self._person = dict(person_identities or {})

    async def create_expert_identity(self, identity: ExpertIdentity) -> ExpertIdentity:
        self._expert[identity.id] = identity
        return identity

    async def get_expert_identity(self, identity_id: str) -> ExpertIdentity | None:
        return self._expert.get(identity_id)

    async def update_expert_identity(self, identity: ExpertIdentity) -> ExpertIdentity:
        self._expert[identity.id] = identity
        return identity

    async def delete_expert_identity(self, identity_id: str) -> None:
        self._expert.pop(identity_id, None)

    async def list_expert_identities(self, limit: int, offset: int) -> list[ExpertIdentity]:
        values = list(self._expert.values())
        values.sort(key=lambda i: i.created_at, reverse=True)
        return values[offset : offset + limit]

    async def get_person_identity(self, user_id: str) -> PersonIdentity | None:
        return self._person.get(user_id)

    async def set_person_identity(self, user_id: str, content: str) -> PersonIdentity:
        existing = self._person.get(user_id)
        now = datetime.now(timezone.utc)
        identity = PersonIdentity(
            user_id=user_id,
            content=content,
            created_at=existing.created_at if existing else now,
            updated_at=now,
        )
        self._person[user_id] = identity
        return identity

    async def delete_person_identity(self, user_id: str) -> None:
        self._person.pop(user_id, None)

    async def list_person_identities(self, limit: int, offset: int) -> list[PersonIdentity]:
        values = list(self._person.values())
        values.sort(key=lambda i: i.created_at, reverse=True)
        return values[offset : offset + limit]


class FakeReminderStore:
    def __init__(self):
        self._reminders: dict[UUID, Reminder] = {}

    async def create_reminder(self, reminder: Reminder) -> Reminder:
        self._reminders[reminder.id] = reminder
        return reminder

    async def get_reminder(self, reminder_id: UUID) -> Reminder | None:
        return self._reminders.get(reminder_id)

    async def update_reminder(self, reminder: Reminder) -> Reminder:
        self._reminders[reminder.id] = reminder
        return reminder

    async def delete_reminder(self, reminder_id: UUID) -> None:
        self._reminders.pop(reminder_id, None)

    async def list_reminders(self, user_id: str, limit: int, offset: int) -> list[Reminder]:
        matching = [r for r in self._reminders.values() if r.user_id == user_id]
        matching.sort(key=lambda r: r.due_at, reverse=True)
        return matching[offset : offset + limit]

    async def list_due_reminders(self, user_id: str, as_of: datetime) -> list[Reminder]:
        due = [
            r
            for r in self._reminders.values()
            if r.user_id == user_id
            and r.status == ReminderStatus.PENDING
            and r.due_at <= as_of
        ]
        due.sort(key=lambda r: r.due_at)
        return due


class FakeGraphStore:
    def __init__(self, search_results: list[ScoredEdge] | None = None):
        self._search_results = search_results or []
        self._entities: dict[UUID, Entity] = {}
        self._edges: dict[UUID, Edge] = {}

    async def create_entity(self, entity: Entity) -> Entity:
        self._entities[entity.id] = entity
        return entity

    async def get_entity(self, entity_id: UUID) -> Entity | None:
        return self._entities.get(entity_id)

    async def find_entity_by_name(self, user_id: str, name: str) -> Entity | None:
        normalized = name.strip().lower()
        for entity in self._entities.values():
            if entity.user_id != user_id:
                continue
            if entity.name.strip().lower() == normalized:
                return entity
            if any(a.strip().lower() == normalized for a in entity.aliases):
                return entity
        return None

    async def delete_entity(self, entity_id: UUID) -> None:
        self._entities.pop(entity_id, None)
        for edge_id in [
            eid
            for eid, e in self._edges.items()
            if e.source_entity_id == entity_id or e.target_entity_id == entity_id
        ]:
            self._edges.pop(edge_id, None)

    async def list_entities(self, user_id: str, limit: int, offset: int) -> list[Entity]:
        matching = [e for e in self._entities.values() if e.user_id == user_id]
        matching.sort(key=lambda e: e.created_at, reverse=True)
        return matching[offset : offset + limit]

    async def add_edge(self, edge: Edge) -> Edge:
        self._edges[edge.id] = edge
        return edge

    async def get_edge(self, edge_id: UUID) -> Edge | None:
        return self._edges.get(edge_id)

    async def get_current_edge(
        self, user_id: str, source_entity_id: UUID, relation: str
    ) -> Edge | None:
        for edge in self._edges.values():
            if (
                edge.user_id == user_id
                and edge.source_entity_id == source_entity_id
                and edge.relation == relation
                and edge.valid_to is None
            ):
                return edge
        return None

    async def close_edge(self, edge_id: UUID, valid_to: datetime) -> Edge:
        edge = self._edges[edge_id]
        closed = edge.model_copy(update={"valid_to": valid_to})
        self._edges[edge_id] = closed
        return closed

    async def delete_edge(self, edge_id: UUID) -> None:
        self._edges.pop(edge_id, None)

    async def search_current_edges(
        self, user_id: str, embedding: list[float], limit: int
    ) -> list[ScoredEdge]:
        return self._search_results[:limit]

    async def list_edges_for_entity(
        self, entity_id: UUID, limit: int, offset: int
    ) -> list[Edge]:
        matching = [
            e
            for e in self._edges.values()
            if e.source_entity_id == entity_id or e.target_entity_id == entity_id
        ]
        matching.sort(key=lambda e: e.recorded_at, reverse=True)
        return matching[offset : offset + limit]


class FakeRelationExtractionClient:
    def __init__(self, candidates: list[RelationCandidate] | None = None):
        self._candidates = candidates or []

    async def extract_relations(self, turn: Turn) -> list[RelationCandidate]:
        return self._candidates


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

"""Storage backend contracts. Concrete Postgres/Redis implementations satisfy
these; a host application may substitute its own. Nothing in the read or
formation path should import a concrete backend directly — only these.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable
from uuid import UUID

from memory_verse_avneesh.models import (
    Edge,
    Entity,
    Episode,
    ExpertIdentity,
    MemoryFact,
    PersonIdentity,
    Reminder,
    ScoredEdge,
    ScoredEpisode,
    ScoredFact,
    Turn,
)


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
class EpisodicStore(Protocol):
    """Durable, embedded record of every turn — episodic memory. Distinct
    from SessionCache: that's Tier 0's ephemeral rolling buffer (evicted),
    this is the permanent, similarity-searchable audit trail write_memory()
    writes to unconditionally, for every turn, with no LLM judgment about
    what's worth keeping (that's what fact extraction already does; here
    completeness is the point). No update/decay — episodes are immutable
    except for explicit user-requested deletion (README Section 7).
    """

    async def add_episode(self, episode: Episode) -> Episode: ...

    async def get_episode(self, episode_id: UUID) -> Episode | None: ...

    async def delete_episode(self, episode_id: UUID) -> None:
        """Idempotent: deleting an id that's already gone is not an error."""
        ...

    async def search_episodes(
        self, user_id: str, embedding: list[float], limit: int
    ) -> list[ScoredEpisode]:
        """Approximate nearest-neighbor search, already ranked by
        similarity. Recency reranking happens above this layer, not here —
        same split as FactStore.search_facts.
        """
        ...

    async def list_episodes(
        self, user_id: str, limit: int, offset: int
    ) -> list[Episode]:
        """Plain paginated listing, newest-first — chronological "what
        happened" browsing, not a similarity search, for a host's
        user-facing memory surface.
        """
        ...


@runtime_checkable
class ReminderStore(Protocol):
    """Prospective memory: future intentions, not facts about the past.
    Created explicitly through this store's own CRUD — write_memory() never
    creates a Reminder on its own, there is no automatic extraction. A
    reminder stays PENDING (and keeps being returned by
    list_due_reminders()) until explicitly marked done/dismissed; passing
    due_at doesn't silently remove it.
    """

    async def create_reminder(self, reminder: Reminder) -> Reminder: ...

    async def get_reminder(self, reminder_id: UUID) -> Reminder | None: ...

    async def update_reminder(self, reminder: Reminder) -> Reminder:
        """Used for status transitions (mark done/dismissed) — the content
        and due_at of a reminder aren't expected to change after creation,
        but nothing prevents it.
        """
        ...

    async def delete_reminder(self, reminder_id: UUID) -> None:
        """Idempotent: deleting an id that's already gone is not an error."""
        ...

    async def list_reminders(
        self, user_id: str, limit: int, offset: int
    ) -> list[Reminder]:
        """All reminders regardless of status, newest-first — for a host's
        user-facing "view your reminders" surface, not the due-only query
        read_memory() uses.
        """
        ...

    async def list_due_reminders(self, user_id: str, as_of: datetime) -> list[Reminder]:
        """PENDING reminders with due_at <= as_of — what read_memory() fetches
        automatically. Deterministic time comparison, no embedding, no LLM.
        """
        ...


@runtime_checkable
class GraphStore(Protocol):
    """Entity/relationship (graph) memory — the bi-temporal edge half of
    Tier 2 that the vector half (FactStore) doesn't cover. Entities are
    resolved by exact case-insensitive name/alias match (no fuzzy/embedding
    resolution in this version). Edges are treated as single-valued per
    (source_entity_id, relation): a new edge for the same pair always
    closes the current one first (README Section 3's contradiction
    mechanic) — multi-valued relations (e.g. "friends_with" allowing many
    concurrent targets) aren't modeled specially yet.
    """

    async def create_entity(self, entity: Entity) -> Entity: ...

    async def get_entity(self, entity_id: UUID) -> Entity | None: ...

    async def find_entity_by_name(self, user_id: str, name: str) -> Entity | None:
        """Case-insensitive match against name or any alias — the entity
        resolution step formation uses before creating a new Entity.
        """
        ...

    async def delete_entity(self, entity_id: UUID) -> None:
        """Idempotent. Also removes every edge referencing this entity as
        source or target — an edge can't meaningfully outlive both its ends.
        """
        ...

    async def list_entities(self, user_id: str, limit: int, offset: int) -> list[Entity]: ...

    async def add_edge(self, edge: Edge) -> Edge: ...

    async def get_edge(self, edge_id: UUID) -> Edge | None: ...

    async def get_current_edge(
        self, user_id: str, source_entity_id: UUID, relation: str
    ) -> Edge | None:
        """The edge with valid_to IS NULL for this (source, relation), if
        any — the lookup formation uses to detect a contradiction.
        """
        ...

    async def close_edge(self, edge_id: UUID, valid_to: datetime) -> Edge:
        """Sets valid_to on an edge that's no longer current — never a
        delete. Used both by formation's contradiction handling and by a
        host's explicit correction.
        """
        ...

    async def delete_edge(self, edge_id: UUID) -> None:
        """Idempotent. Permanent removal — for user-requested deletion
        (README Section 7), distinct from close_edge's supersession.
        """
        ...

    async def search_current_edges(
        self, user_id: str, embedding: list[float], limit: int
    ) -> list[ScoredEdge]:
        """ANN search over fact_sentence embeddings, filtered to
        valid_to IS NULL (current truth only) — closed/historical edges
        are never part of default read-path retrieval.
        """
        ...

    async def list_edges_for_entity(
        self, entity_id: UUID, limit: int, offset: int
    ) -> list[Edge]:
        """Every edge (current and closed) where this entity is source or
        target, newest-first — full history for a host's user-facing view,
        and also what the read path's 1-hop expansion uses.
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


@runtime_checkable
class IdentityStore(Protocol):
    """Two distinct, host/library-managed identity records — neither is
    learned by the formation pipeline, both are written explicitly through
    this store's own CRUD:

    - Expert identities: host-authored personas ("expert_email_writer"),
      keyed by an arbitrary string id the host chooses and passes into
      read_memory() explicitly. The host owns their full lifecycle.
    - Person identity: one durable record per user_id, distinct from the
      Tier 1 ProfileCache (that's an ephemeral cache; this is a deliberate,
      durable record) — always fetched automatically by read_memory() when
      an IdentityStore is configured.
    """

    async def create_expert_identity(self, identity: ExpertIdentity) -> ExpertIdentity: ...

    async def get_expert_identity(self, identity_id: str) -> ExpertIdentity | None: ...

    async def update_expert_identity(self, identity: ExpertIdentity) -> ExpertIdentity: ...

    async def delete_expert_identity(self, identity_id: str) -> None:
        """Idempotent: deleting an id that's already gone is not an error."""
        ...

    async def list_expert_identities(self, limit: int, offset: int) -> list[ExpertIdentity]: ...

    async def get_person_identity(self, user_id: str) -> PersonIdentity | None: ...

    async def set_person_identity(self, user_id: str, content: str) -> PersonIdentity:
        """Upsert — one record per user_id."""
        ...

    async def delete_person_identity(self, user_id: str) -> None:
        """Idempotent: deleting a user_id that's already gone is not an error."""
        ...

    async def list_person_identities(self, limit: int, offset: int) -> list[PersonIdentity]: ...

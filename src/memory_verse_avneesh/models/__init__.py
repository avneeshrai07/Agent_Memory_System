"""Shared schemas used across the read path and the formation path.

Kept deliberately small for the first vertical slice (Section 4/5 of the
README, vector-only Tier 2). Bi-temporal edge models and reflection models
are added in Phase 2 once this loop is proven end-to-end.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    PROVISIONAL = "provisional"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"
    ARCHIVED = "archived"


class Turn(BaseModel):
    """One raw conversational exchange. Source of truth / audit trail."""

    id: UUID = Field(default_factory=uuid4)
    user_id: str
    conversation_id: str
    user_message: str
    assistant_message: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MemoryFact(BaseModel):
    """A single Tier 2 (vector) memory unit — one durable fact about a user."""

    id: UUID = Field(default_factory=uuid4)
    user_id: str
    category: str
    value: str
    embedding: list[float] | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    observation_count: int = 1
    status: MemoryStatus = MemoryStatus.PROVISIONAL
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_reinforced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ScoredFact(BaseModel):
    """A MemoryFact plus its retrieval score, returned by the read path."""

    fact: MemoryFact
    score: float


class ExtractedCandidate(BaseModel):
    """Raw output of the formation path's extraction step (README Section 5,
    step 1) — not yet a MemoryFact. Has no id/status until it passes through
    resolve -> operation-classify -> safety-gate.
    """

    category: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    explicit: bool


class MemoryOperation(str, Enum):
    """README Section 5, step 3 — how a candidate relates to existing memory."""

    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    NOOP = "noop"


class ResolvedOperation(BaseModel):
    """Output of the formation path's resolution step. target_fact_id is a
    real id already mapped back from whatever the LLM used to refer to the
    existing fact (README Section 5, step 2-3) — required for UPDATE/DELETE,
    always None for ADD/NOOP.
    """

    operation: MemoryOperation
    target_fact_id: UUID | None = None


class Episode(BaseModel):
    """One durable, embedded record of a past turn — episodic memory.
    Distinct from Turn: Turn is Tier 0's ephemeral session-cache shape
    (evicted once the session cache trims it); Episode is the durable,
    similarity-searchable counterpart written by write_memory() for every
    turn, unconditionally — no LLM judgment about what's "worth
    remembering" the way fact extraction has, since completeness (an actual
    audit trail of what happened, when) is the whole point. Immutable except
    for explicit user-requested deletion — there is no edit_episode, since
    rewriting history doesn't make sense for a record of what was actually
    said.
    """

    id: UUID = Field(default_factory=uuid4)
    user_id: str
    conversation_id: str
    user_message: str
    assistant_message: str
    embedding: list[float] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ScoredEpisode(BaseModel):
    """An Episode plus its retrieval score, returned by the read path."""

    episode: Episode
    score: float


class ExpertIdentity(BaseModel):
    """A host-authored persona ('expert email writer', etc.) — not learned by
    the library. Content is a free-form instructions/tone/expertise block the
    host writes and maintains directly via the identity management API.
    Selected explicitly per read_memory() call via identity_id; never chosen
    automatically.
    """

    id: str
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PersonIdentity(BaseModel):
    """The durable per-person counterpart to ExpertIdentity — one row per
    user_id. Distinct from the Tier 1 ProfileCache: that's a fast/ephemeral
    key-value cache rebuilt from facts, this is a deliberate, durable
    Postgres-backed identity record maintained through the same management
    API as ExpertIdentity (get/set/delete), not the formation pipeline.
    """

    user_id: str
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ReminderStatus(str, Enum):
    PENDING = "pending"
    DONE = "done"
    DISMISSED = "dismissed"


class Reminder(BaseModel):
    """Prospective memory — a future intention, not a fact about the past.
    Created explicitly (by the host, or the host's own LLM calling this as
    a tool) — write_memory() never creates these on its own; there is no
    automatic "this sounds like something to remind them about" extraction
    in this version. Stays PENDING (and keeps surfacing in read_memory()
    once due) until explicitly marked done or dismissed — reminders never
    silently disappear once their due_at passes.
    """

    id: UUID = Field(default_factory=uuid4)
    user_id: str
    content: str
    due_at: datetime
    status: ReminderStatus = ReminderStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None


class MemoryContext(BaseModel):
    """Everything Tier 0/1/2 retrieval assembled for one incoming message.

    This is the read path's entire output. The library stops here — the
    host application builds its own prompt/messages from this however it
    wants (its own system prompt, its own tool use, its own model, its own
    streaming) and makes its own generation call. memory_verse_avneesh never calls
    an LLM to produce a user-facing response.
    """

    profile: dict | None
    relevant_facts: list[ScoredFact]
    recent_turns: list[Turn]
    relevant_episodes: list[ScoredEpisode] = []
    due_reminders: list[Reminder] = []
    person_identity: str | None = None
    expert_identity: str | None = None

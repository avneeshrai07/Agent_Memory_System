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


class MemoryContext(BaseModel):
    """Everything Tier 0/1/2 retrieval assembled for one incoming message.

    This is the read path's entire output. The library stops here — the
    host application builds its own prompt/messages from this however it
    wants (its own system prompt, its own tool use, its own model, its own
    streaming) and makes its own generation call. agent_memory never calls
    an LLM to produce a user-facing response.
    """

    profile: dict | None
    relevant_facts: list[ScoredFact]
    recent_turns: list[Turn]

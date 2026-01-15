# MEMORY_SYSTEM/artifacts/artifact_contract.py

from typing import TypedDict, Literal, Dict, Any, Optional
from datetime import datetime


ArtifactType = Literal[
    "email",
    "meeting",
    "document",
    "story",
    "note",
    "plan",
    "code",
]


class ArtifactMetadata(TypedDict, total=False):
    title: str
    subject: str
    phase: str
    author: str
    tags: list[str]
    source: str  # e.g. "user", "assistant", "imported"
    external_id: str


class Artifact(TypedDict):
    artifact_id: str
    artifact_type: ArtifactType

    # Canonical content
    full_content: str

    # Human-facing summary (NOT state, NOT decision)
    summary: str

    # Semantic recall support (filled later)
    summary_embedding: Optional[list[float]]

    # Contextual metadata
    metadata: ArtifactMetadata

    # Lifecycle
    created_at: datetime
    last_updated_at: datetime

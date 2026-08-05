"""LLM/embedding provider contracts. Provider-agnostic by design — Bedrock,
OpenAI, Anthropic, or a local model can all satisfy these. The first concrete
implementation is memory_verse_avneesh.llm.bedrock (AWS Bedrock), added once these
contracts are settled.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from memory_verse_avneesh.models import (
    ExtractedCandidate,
    RelationCandidate,
    ResolvedOperation,
    ScoredFact,
    Turn,
)


@runtime_checkable
class EmbeddingClient(Protocol):
    """The single embedding call on the read path's critical path (README
    Section 4, step 2) goes through this — keep implementations fast.
    """

    async def embed(self, text: str) -> list[float]: ...


@runtime_checkable
class ExtractionClient(Protocol):
    """Formation-path structured extraction (README Section 5, step 1). Runs
    off the critical path, on a Turn the host application constructed after
    its own generation call — may be a larger/slower model than
    EmbeddingClient.
    """

    async def extract(self, turn: Turn) -> list[ExtractedCandidate]: ...


@runtime_checkable
class ResolutionClient(Protocol):
    """Formation-path operation classification (README Section 5, step 3):
    given a new candidate and its nearest existing memories (already
    retrieved — this call does not search), decide ADD/UPDATE/DELETE/NOOP.
    Runs off the critical path.
    """

    async def classify_operation(
        self, candidate: ExtractedCandidate, existing: list[ScoredFact]
    ) -> ResolvedOperation: ...


@runtime_checkable
class RelationExtractionClient(Protocol):
    """Formation-path relation extraction (graph memory) — a distinct call
    from ExtractionClient's flat-fact extraction, since the output shape and
    prompting intent differ (relationships between entities, not standalone
    facts). Runs off the critical path, on the same host-constructed Turn.
    Edge resolution against existing edges is deterministic (same
    source+relation contradiction check), unlike fact resolution — so there
    is no equivalent of ResolutionClient here.
    """

    async def extract_relations(self, turn: Turn) -> list[RelationCandidate]: ...

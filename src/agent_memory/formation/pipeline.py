"""Minimal formation-path loop (README Section 5).

Resolve-against-existing-memory, the ADD/UPDATE/DELETE/NOOP classification,
the safety gate, reflections, and decay are intentionally not implemented
yet — per the build order (README Section 8), this proves extract -> write
end-to-end first; those refine a loop that already works.
"""

from __future__ import annotations

from agent_memory.llm.interfaces import EmbeddingClient, ExtractionClient
from agent_memory.models import MemoryFact, MemoryStatus, Turn
from agent_memory.storage.interfaces import FactStore

MIN_COMMIT_CONFIDENCE = 0.75


async def write_memory(
    turn: Turn,
    *,
    extraction_client: ExtractionClient,
    embedding_client: EmbeddingClient,
    fact_store: FactStore,
) -> list[MemoryFact]:
    """Extracts candidates from one turn and writes each as its own fact.

    Every candidate becomes a new row — no dedup or merge against existing
    memory yet, that's the resolve step layered on next. A candidate commits
    as ACTIVE if it was stated explicitly or clears MIN_COMMIT_CONFIDENCE;
    otherwise it's written as PROVISIONAL and excluded from read-path
    retrieval (FactStore.search_facts only returns 'active' rows) until a
    future safety-gate/reinforcement step promotes or rejects it.
    """

    candidates = await extraction_client.extract(turn)

    written: list[MemoryFact] = []
    for candidate in candidates:
        embedding = await embedding_client.embed(candidate.value)
        status = (
            MemoryStatus.ACTIVE
            if candidate.explicit or candidate.confidence >= MIN_COMMIT_CONFIDENCE
            else MemoryStatus.PROVISIONAL
        )
        fact = MemoryFact(
            user_id=turn.user_id,
            category=candidate.category,
            value=candidate.value,
            embedding=embedding,
            confidence=candidate.confidence,
            status=status,
        )
        written.append(await fact_store.add_fact(fact))

    return written

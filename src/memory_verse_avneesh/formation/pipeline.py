"""Formation-path loop (README Section 5).

Extract -> resolve -> classify -> safety gate -> write. Reflections and
decay (README Section 5, steps 6-7) are intentionally not implemented yet —
per the build order (README Section 8), this closes out the write-time
judgment logic first; those are periodic/batched refinements layered on a
loop that already writes correctly.
"""

from __future__ import annotations

from datetime import datetime, timezone

from memory_verse_avneesh.formation.safety_gate import passes_safety_gate
from memory_verse_avneesh.llm.interfaces import EmbeddingClient, ExtractionClient, ResolutionClient
from memory_verse_avneesh.models import (
    Episode,
    ExtractedCandidate,
    MemoryFact,
    MemoryOperation,
    MemoryStatus,
    ResolvedOperation,
    ScoredFact,
    Turn,
)
from memory_verse_avneesh.storage.interfaces import EpisodicStore, FactStore

MIN_COMMIT_CONFIDENCE = 0.75
RESOLVE_SEARCH_LIMIT = 5


async def write_memory(
    turn: Turn,
    *,
    extraction_client: ExtractionClient,
    resolution_client: ResolutionClient,
    embedding_client: EmbeddingClient,
    fact_store: FactStore,
    episodic_store: EpisodicStore | None = None,
) -> list[MemoryFact]:
    """Extracts candidates from one turn and resolves each against existing
    memory before writing anything.

    For every candidate: embed it, look up its nearest existing facts, ask
    the ResolutionClient how it relates to them (ADD/UPDATE/DELETE/NOOP),
    then run the outcome through the deterministic safety gate — which can
    downgrade a commit to PROVISIONAL, or block a DELETE from touching an
    existing fact — regardless of what the resolution decided.

    episodic_store is optional — omit it entirely if the host doesn't use
    the episodic memory feature. When set, the turn is embedded and written
    as an Episode unconditionally, independent of whatever fact extraction
    below decides — episodic memory's value is completeness (an actual
    record of what happened), not curation.
    """

    if episodic_store is not None:
        episode_embedding = await embedding_client.embed(
            f"{turn.user_message}\n{turn.assistant_message}"
        )
        await episodic_store.add_episode(
            Episode(
                user_id=turn.user_id,
                conversation_id=turn.conversation_id,
                user_message=turn.user_message,
                assistant_message=turn.assistant_message,
                embedding=episode_embedding,
                created_at=turn.created_at,
            )
        )

    candidates = await extraction_client.extract(turn)

    written: list[MemoryFact] = []
    for candidate in candidates:
        embedding = await embedding_client.embed(candidate.value)
        existing = await fact_store.search_facts(
            turn.user_id, embedding, RESOLVE_SEARCH_LIMIT
        )
        resolution = await resolution_client.classify_operation(candidate, existing)
        target = _find_target(resolution, existing)

        gate_observation_count = target.observation_count if target else 0
        gate_passed = passes_safety_gate(candidate, gate_observation_count)

        if resolution.operation == MemoryOperation.NOOP:
            if target is not None:
                await _reinforce(fact_store, target)
            continue

        if resolution.operation == MemoryOperation.UPDATE and target is not None:
            merged = _merge(target, candidate, embedding, gate_passed)
            written.append(await fact_store.update_fact(merged))
            continue

        if resolution.operation == MemoryOperation.DELETE and target is not None:
            if gate_passed:
                superseded = target.model_copy(update={"status": MemoryStatus.SUPERSEDED})
                await fact_store.update_fact(superseded)
            # Whether or not the old fact was retired, the candidate itself
            # still needs writing below — it's the new current truth (or,
            # if the gate failed, only a provisional one, and the old fact
            # was deliberately left untouched above).

        new_fact = MemoryFact(
            user_id=turn.user_id,
            category=candidate.category,
            value=candidate.value,
            embedding=embedding,
            confidence=candidate.confidence,
            status=_status_for(candidate, gate_passed),
        )
        written.append(await fact_store.add_fact(new_fact))

    return written


def _find_target(
    resolution: ResolvedOperation, existing: list[ScoredFact]
) -> MemoryFact | None:
    if resolution.target_fact_id is None:
        return None
    return next(
        (sf.fact for sf in existing if sf.fact.id == resolution.target_fact_id), None
    )


def _status_for(candidate: ExtractedCandidate, gate_passed: bool) -> MemoryStatus:
    if not gate_passed:
        return MemoryStatus.PROVISIONAL
    if candidate.explicit or candidate.confidence >= MIN_COMMIT_CONFIDENCE:
        return MemoryStatus.ACTIVE
    return MemoryStatus.PROVISIONAL


def _merge(
    existing: MemoryFact,
    candidate: ExtractedCandidate,
    embedding: list[float],
    gate_passed: bool,
) -> MemoryFact:
    return existing.model_copy(
        update={
            "value": candidate.value,
            "embedding": embedding,
            "confidence": max(existing.confidence, candidate.confidence),
            "observation_count": existing.observation_count + 1,
            "status": _status_for(candidate, gate_passed),
            "last_reinforced_at": datetime.now(timezone.utc),
        }
    )


async def _reinforce(fact_store: FactStore, fact: MemoryFact) -> None:
    reinforced = fact.model_copy(
        update={
            "observation_count": fact.observation_count + 1,
            "last_reinforced_at": datetime.now(timezone.utc),
        }
    )
    await fact_store.update_fact(reinforced)

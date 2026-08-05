"""Formation-path loop (README Section 5).

Extract -> resolve -> classify -> safety gate -> write. Reflections and
decay (README Section 5, steps 6-7) are intentionally not implemented yet —
per the build order (README Section 8), this closes out the write-time
judgment logic first; those are periodic/batched refinements layered on a
loop that already writes correctly.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID

from memory_verse_avneesh.formation.safety_gate import passes_safety_gate
from memory_verse_avneesh.llm.interfaces import (
    EmbeddingClient,
    ExtractionClient,
    RelationExtractionClient,
    ResolutionClient,
)
from memory_verse_avneesh.models import (
    Edge,
    Entity,
    Episode,
    ExtractedCandidate,
    MemoryFact,
    MemoryOperation,
    MemoryStatus,
    RelationCandidate,
    ResolvedOperation,
    ScoredFact,
    Turn,
)
from memory_verse_avneesh.storage.interfaces import EpisodicStore, FactStore, GraphStore

logger = logging.getLogger(__name__)

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
    graph_store: GraphStore | None = None,
    relation_extraction_client: RelationExtractionClient | None = None,
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

    graph_store is optional — omit it entirely if the host doesn't use the
    graph memory feature. When set, relation_extraction_client is required
    (there's no way to extract relation candidates without it). Edge
    resolution is deterministic, not an LLM call: a new edge for the same
    (source_entity_id, relation) always closes the current one first
    (README Section 3's contradiction mechanic) — no equivalent of
    ResolutionClient's ADD/UPDATE/DELETE/NOOP judgment is needed here.
    """

    if graph_store is not None and relation_extraction_client is None:
        # Caller-contract violation, not a backend failure -- this must
        # still raise immediately rather than being swallowed like the
        # best-effort handling below.
        raise ValueError(
            "write_memory: graph_store was provided without relation_extraction_client "
            "-- graph writes need both, there's no way to extract relation candidates "
            "otherwise. Pass both or neither."
        )

    logger.info(
        "write_memory: start -- user_id=%s conversation_id=%s turn_id=%s",
        turn.user_id, turn.conversation_id, turn.id,
    )

    # Each store's write is isolated in its own try/except (best-effort,
    # not all-or-nothing): one failing doesn't stop the others, since this
    # runs backgrounded already and losing one write silently is better
    # than losing all of them over one bad row or a transient backend blip.
    if episodic_store is not None:
        try:
            episode_embedding = await embedding_client.embed(
                f"{turn.user_message}\n{turn.assistant_message}"
            )
            episode = await episodic_store.add_episode(
                Episode(
                    user_id=turn.user_id,
                    conversation_id=turn.conversation_id,
                    user_message=turn.user_message,
                    assistant_message=turn.assistant_message,
                    embedding=episode_embedding,
                    created_at=turn.created_at,
                )
            )
            logger.info(
                "write_memory: episodic write OK -- user_id=%s episode_id=%s",
                turn.user_id, episode.id,
            )
        except Exception:
            logger.exception(
                "write_memory: episodic write failed for user_id=%s -- skipping, "
                "other stores unaffected", turn.user_id,
            )

    if graph_store is not None and relation_extraction_client is not None:
        try:
            await _write_relations(turn, relation_extraction_client, embedding_client, graph_store)
        except Exception:
            logger.exception(
                "write_memory: graph memory write failed for user_id=%s -- skipping, "
                "other stores unaffected", turn.user_id,
            )

    try:
        candidates = await extraction_client.extract(turn)
    except Exception:
        logger.exception(
            "write_memory: fact extraction failed for user_id=%s -- no facts will "
            "be written this turn, other stores unaffected", turn.user_id,
        )
        candidates = []

    logger.info("write_memory: fact extraction -- user_id=%s candidate_count=%d", turn.user_id, len(candidates))

    written: list[MemoryFact] = []
    for candidate in candidates:
        try:
            fact = await _resolve_and_write_fact(
                turn.user_id, candidate, resolution_client, embedding_client, fact_store
            )
        except Exception:
            logger.exception(
                "write_memory: fact candidate failed for user_id=%s category=%s -- "
                "skipping this candidate, other candidates unaffected",
                turn.user_id, candidate.category,
            )
            continue
        if fact is not None:
            written.append(fact)

    return written


async def _resolve_and_write_fact(
    user_id: str,
    candidate: ExtractedCandidate,
    resolution_client: ResolutionClient,
    embedding_client: EmbeddingClient,
    fact_store: FactStore,
) -> MemoryFact | None:
    """One candidate's full resolve-and-write. Returns the written/updated
    fact, or None for a NOOP (nothing new to report back to the caller).
    """
    embedding = await embedding_client.embed(candidate.value)
    existing = await fact_store.search_facts(user_id, embedding, RESOLVE_SEARCH_LIMIT)
    resolution = await resolution_client.classify_operation(candidate, existing)
    target = _find_target(resolution, existing)

    gate_observation_count = target.observation_count if target else 0
    gate_passed = passes_safety_gate(candidate, gate_observation_count)

    if resolution.operation == MemoryOperation.NOOP:
        if target is not None:
            await _reinforce(fact_store, target)
            logger.info(
                "write_memory: fact NOOP (reinforced) -- user_id=%s category=%s fact_id=%s",
                user_id, candidate.category, target.id,
            )
        return None

    if resolution.operation == MemoryOperation.UPDATE and target is not None:
        merged = _merge(target, candidate, embedding, gate_passed)
        updated = await fact_store.update_fact(merged)
        logger.info(
            "write_memory: fact UPDATE -- user_id=%s category=%s fact_id=%s status=%s value=%r",
            user_id, candidate.category, updated.id, updated.status.value, updated.value,
        )
        return updated

    if resolution.operation == MemoryOperation.DELETE and target is not None:
        if gate_passed:
            superseded = target.model_copy(update={"status": MemoryStatus.SUPERSEDED})
            await fact_store.update_fact(superseded)
            logger.info(
                "write_memory: fact DELETE (superseded) -- user_id=%s category=%s fact_id=%s",
                user_id, candidate.category, target.id,
            )
        else:
            logger.info(
                "write_memory: fact DELETE blocked by safety gate -- user_id=%s "
                "category=%s fact_id=%s left untouched",
                user_id, candidate.category, target.id,
            )
        # Whether or not the old fact was retired, the candidate itself
        # still needs writing below — it's the new current truth (or,
        # if the gate failed, only a provisional one, and the old fact
        # was deliberately left untouched above).

    new_fact = MemoryFact(
        user_id=user_id,
        category=candidate.category,
        value=candidate.value,
        embedding=embedding,
        confidence=candidate.confidence,
        status=_status_for(candidate, gate_passed),
    )
    written = await fact_store.add_fact(new_fact)
    logger.info(
        "write_memory: fact ADD -- user_id=%s category=%s fact_id=%s status=%s value=%r",
        user_id, candidate.category, written.id, written.status.value, written.value,
    )
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


# --- graph memory (relation extraction + bi-temporal edge resolution) ------


async def _write_relations(
    turn: Turn,
    relation_extraction_client: RelationExtractionClient,
    embedding_client: EmbeddingClient,
    graph_store: GraphStore,
) -> None:
    try:
        candidates = await relation_extraction_client.extract_relations(turn)
    except Exception:
        logger.exception(
            "write_memory: relation extraction failed for user_id=%s -- no edges "
            "will be written this turn", turn.user_id,
        )
        return

    logger.info(
        "write_memory: relation extraction -- user_id=%s candidate_count=%d",
        turn.user_id, len(candidates),
    )

    for candidate in candidates:
        # Same explicit-or-confidence gate as fact commits (MIN_COMMIT_CONFIDENCE)
        # -- not the full safety_gate.py machinery (that's keyed on a fact's
        # category/observation-count shape, which edges don't have), but the
        # same underlying judgment: don't let one weak, inferred candidate
        # silently rewrite a relationship. If it fails, skip rather than
        # write noise -- there's no PROVISIONAL concept for edges.
        if not (candidate.explicit or candidate.confidence >= MIN_COMMIT_CONFIDENCE):
            logger.info(
                "write_memory: relation candidate skipped (below confidence gate) "
                "-- user_id=%s relation=%s confidence=%.2f explicit=%s",
                turn.user_id, candidate.relation, candidate.confidence, candidate.explicit,
            )
            continue

        try:
            await _resolve_and_write_edge(turn.user_id, candidate, embedding_client, graph_store)
        except Exception:
            logger.exception(
                "write_memory: relation candidate failed for user_id=%s relation=%s "
                "-- skipping this candidate, other candidates unaffected",
                turn.user_id, candidate.relation,
            )


async def _resolve_and_write_edge(
    user_id: str,
    candidate: RelationCandidate,
    embedding_client: EmbeddingClient,
    graph_store: GraphStore,
) -> None:
    source_entity = await _resolve_entity(user_id, candidate.source_name, graph_store)

    target_entity_id: UUID | None = None
    target_value: str | None = None
    if candidate.target_is_entity:
        target_entity = await _resolve_entity(user_id, candidate.target_name, graph_store)
        target_entity_id = target_entity.id
    else:
        target_value = candidate.target_name

    now = datetime.now(timezone.utc)
    existing_current = await graph_store.get_current_edge(
        user_id, source_entity.id, candidate.relation
    )

    if existing_current is not None:
        unchanged = (
            existing_current.target_entity_id == target_entity_id
            and existing_current.target_value == target_value
        )
        if unchanged:
            logger.info(
                "write_memory: edge NOOP (unchanged) -- user_id=%s relation=%s edge_id=%s",
                user_id, candidate.relation, existing_current.id,
            )
            return  # already current truth -- nothing to write
        await graph_store.close_edge(existing_current.id, now)
        logger.info(
            "write_memory: edge closed (superseded) -- user_id=%s relation=%s "
            "old_edge_id=%s", user_id, candidate.relation, existing_current.id,
        )

    fact_sentence = _render_fact_sentence(
        candidate.source_name, candidate.relation, candidate.target_name
    )
    embedding = await embedding_client.embed(fact_sentence)

    new_edge = await graph_store.add_edge(
        Edge(
            user_id=user_id,
            source_entity_id=source_entity.id,
            relation=candidate.relation,
            target_entity_id=target_entity_id,
            target_value=target_value,
            fact_sentence=fact_sentence,
            embedding=embedding,
            confidence=candidate.confidence,
            valid_from=now,
            observed_at=now,
            recorded_at=now,
        )
    )
    logger.info(
        "write_memory: edge ADD -- user_id=%s edge_id=%s fact_sentence=%r",
        user_id, new_edge.id, fact_sentence,
    )


async def _resolve_entity(user_id: str, name: str, graph_store: GraphStore) -> Entity:
    """Exact case-insensitive name/alias match; creates a new Entity if
    nothing matches. No fuzzy/embedding-based resolution in this version
    (GraphStore's own docstring).
    """
    existing = await graph_store.find_entity_by_name(user_id, name)
    if existing is not None:
        return existing
    created = await graph_store.create_entity(Entity(user_id=user_id, name=name))
    logger.info(
        "write_memory: entity created -- user_id=%s entity_id=%s name=%r",
        user_id, created.id, name,
    )
    return created


def _render_fact_sentence(source_name: str, relation: str, target_name: str) -> str:
    # Deterministic template, not another LLM call -- see Edge's own
    # docstring for why the read path searches this sentence's embedding
    # rather than the bare entity names.
    subject = "User" if source_name.strip().lower() == "user" else source_name
    readable_relation = relation.replace("_", " ")
    return f"{subject} {readable_relation} {target_name}."

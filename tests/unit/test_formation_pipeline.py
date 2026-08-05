import pytest

from memory_verse_avneesh.formation.pipeline import MIN_COMMIT_CONFIDENCE, write_memory
from memory_verse_avneesh.formation.safety_gate import SAFETY_GATE_MIN_OBSERVATIONS
from memory_verse_avneesh.models import (
    ExtractedCandidate,
    MemoryFact,
    MemoryStatus,
    RelationCandidate,
    ResolvedOperation,
    ScoredFact,
    Turn,
)

from .fakes import (
    FakeEmbeddingClient,
    FakeEpisodicStore,
    FakeExtractionClient,
    FakeFactStore,
    FakeGraphStore,
    FakeRelationExtractionClient,
    FakeResolutionClient,
)


def _turn() -> Turn:
    return Turn(
        user_id="u1",
        conversation_id="c1",
        user_message="I always want short, bullet-pointed answers",
        assistant_message="Got it, I'll keep things brief.",
    )


async def _run(candidate, resolution, search_results=None):
    fact_store = FakeFactStore(search_results=search_results or [])
    written = await write_memory(
        _turn(),
        extraction_client=FakeExtractionClient([candidate]),
        resolution_client=FakeResolutionClient(resolution),
        embedding_client=FakeEmbeddingClient(),
        fact_store=fact_store,
    )
    return written, fact_store


async def test_add_explicit_commits_active():
    candidate = ExtractedCandidate(
        category="preference", value="wants bullet-pointed answers",
        confidence=0.5, explicit=True,
    )
    written, fact_store = await _run(candidate, ResolvedOperation(operation="add"))

    assert len(written) == 1
    assert written[0].status == MemoryStatus.ACTIVE
    assert fact_store.added == written
    assert fact_store.updated == []


async def test_add_implicit_low_confidence_stays_provisional():
    candidate = ExtractedCandidate(
        category="preference", value="might prefer brevity",
        confidence=MIN_COMMIT_CONFIDENCE - 0.1, explicit=False,
    )
    written, _ = await _run(candidate, ResolvedOperation(operation="add"))

    assert written[0].status == MemoryStatus.PROVISIONAL


async def test_noop_reinforces_existing_and_writes_nothing():
    existing_fact = MemoryFact(
        user_id="u1", category="preference", value="likes brevity",
        confidence=0.9, observation_count=1, status=MemoryStatus.ACTIVE,
    )
    candidate = ExtractedCandidate(
        category="preference", value="likes brevity", confidence=0.9, explicit=True,
    )
    written, fact_store = await _run(
        candidate,
        ResolvedOperation(operation="noop", target_fact_id=existing_fact.id),
        search_results=[ScoredFact(fact=existing_fact, score=0.99)],
    )

    assert written == []
    assert fact_store.added == []
    assert len(fact_store.updated) == 1
    assert fact_store.updated[0].id == existing_fact.id
    assert fact_store.updated[0].observation_count == 2


async def test_update_merges_into_existing_instead_of_adding():
    existing_fact = MemoryFact(
        user_id="u1", category="preference", value="likes short answers",
        confidence=0.7, observation_count=1, status=MemoryStatus.ACTIVE,
    )
    candidate = ExtractedCandidate(
        category="preference", value="likes short, bulleted answers",
        confidence=0.95, explicit=True,
    )
    written, fact_store = await _run(
        candidate,
        ResolvedOperation(operation="update", target_fact_id=existing_fact.id),
        search_results=[ScoredFact(fact=existing_fact, score=0.9)],
    )

    assert fact_store.added == []
    assert len(written) == 1
    merged = written[0]
    assert merged.id == existing_fact.id
    assert merged.value == "likes short, bulleted answers"
    assert merged.observation_count == 2
    assert merged.confidence == 0.95
    assert merged.status == MemoryStatus.ACTIVE
    assert fact_store.updated == [merged]


async def test_delete_supersedes_old_and_writes_new_as_current():
    old_fact = MemoryFact(
        user_id="u1", category="preference", value="prefers long detailed answers",
        confidence=0.9, observation_count=3, status=MemoryStatus.ACTIVE,
    )
    candidate = ExtractedCandidate(
        category="preference", value="now prefers short answers",
        confidence=0.95, explicit=True,
    )
    written, fact_store = await _run(
        candidate,
        ResolvedOperation(operation="delete", target_fact_id=old_fact.id),
        search_results=[ScoredFact(fact=old_fact, score=0.85)],
    )

    assert len(fact_store.updated) == 1
    assert fact_store.updated[0].id == old_fact.id
    assert fact_store.updated[0].status == MemoryStatus.SUPERSEDED

    assert len(written) == 1
    assert written[0].value == "now prefers short answers"
    assert written[0].status == MemoryStatus.ACTIVE
    assert fact_store.added == written


async def test_safety_gate_blocks_implicit_identity_claim_with_no_history():
    candidate = ExtractedCandidate(
        category="identity", value="is a nuclear engineer",
        confidence=0.95, explicit=False,
    )
    written, _ = await _run(candidate, ResolvedOperation(operation="add"))

    # high confidence would normally commit ACTIVE, but identity is a
    # safety-gated category and this is implicit with zero prior observations
    assert written[0].status == MemoryStatus.PROVISIONAL


async def test_safety_gate_passes_identity_claim_once_repetition_threshold_met():
    existing_fact = MemoryFact(
        user_id="u1", category="identity", value="works in engineering",
        confidence=0.8, observation_count=SAFETY_GATE_MIN_OBSERVATIONS - 1,
        status=MemoryStatus.PROVISIONAL,
    )
    candidate = ExtractedCandidate(
        category="identity", value="is a nuclear engineer",
        confidence=0.95, explicit=False,
    )
    written, _ = await _run(
        candidate,
        ResolvedOperation(operation="update", target_fact_id=existing_fact.id),
        search_results=[ScoredFact(fact=existing_fact, score=0.9)],
    )

    assert written[0].status == MemoryStatus.ACTIVE


async def test_safety_gate_blocks_delete_of_identity_fact_without_enough_evidence():
    old_fact = MemoryFact(
        user_id="u1", category="identity", value="works in marketing",
        confidence=0.8, observation_count=0, status=MemoryStatus.ACTIVE,
    )
    candidate = ExtractedCandidate(
        category="identity", value="now works in engineering",
        confidence=0.9, explicit=False,
    )
    written, fact_store = await _run(
        candidate,
        ResolvedOperation(operation="delete", target_fact_id=old_fact.id),
        search_results=[ScoredFact(fact=old_fact, score=0.85)],
    )

    # the gate failed, so the old identity fact must NOT be touched
    assert fact_store.updated == []
    # the new candidate is still written, but only as provisional
    assert written[0].status == MemoryStatus.PROVISIONAL


async def test_no_candidates_writes_nothing():
    fact_store = FakeFactStore()
    written = await write_memory(
        _turn(),
        extraction_client=FakeExtractionClient([]),
        resolution_client=FakeResolutionClient(),
        embedding_client=FakeEmbeddingClient(),
        fact_store=fact_store,
    )
    assert written == []
    assert fact_store.added == []


# --- episodic memory -----------------------------------------------------


async def test_write_memory_without_episodic_store_skips_episode_write():
    fact_store = FakeFactStore()
    await write_memory(
        _turn(),
        extraction_client=FakeExtractionClient([]),
        resolution_client=FakeResolutionClient(),
        embedding_client=FakeEmbeddingClient(),
        fact_store=fact_store,
    )
    # nothing to assert on directly -- this just proves omitting
    # episodic_store doesn't error, matching the optional/opt-in contract


async def test_write_memory_writes_episode_unconditionally_even_with_no_candidates():
    fact_store = FakeFactStore()
    episodic_store = FakeEpisodicStore()
    turn = _turn()

    written = await write_memory(
        turn,
        extraction_client=FakeExtractionClient([]),  # no fact candidates at all
        resolution_client=FakeResolutionClient(),
        embedding_client=FakeEmbeddingClient(),
        fact_store=fact_store,
        episodic_store=episodic_store,
    )

    assert written == []  # no facts written
    assert len(episodic_store.added) == 1  # episode still written -- completeness, not curation
    episode = episodic_store.added[0]
    assert episode.user_id == turn.user_id
    assert episode.conversation_id == turn.conversation_id
    assert episode.user_message == turn.user_message
    assert episode.assistant_message == turn.assistant_message
    assert episode.embedding is not None


async def test_write_memory_embeds_combined_user_and_assistant_message_for_episode():
    fact_store = FakeFactStore()
    episodic_store = FakeEpisodicStore()
    embedding_client = FakeEmbeddingClient()
    turn = _turn()

    await write_memory(
        turn,
        extraction_client=FakeExtractionClient([]),
        resolution_client=FakeResolutionClient(),
        embedding_client=embedding_client,
        fact_store=fact_store,
        episodic_store=episodic_store,
    )

    assert f"{turn.user_message}\n{turn.assistant_message}" in embedding_client.embedded_texts


# --- graph memory ----------------------------------------------------------


async def test_write_memory_requires_relation_extraction_client_with_graph_store():
    with pytest.raises(ValueError):
        await write_memory(
            _turn(),
            extraction_client=FakeExtractionClient([]),
            resolution_client=FakeResolutionClient(),
            embedding_client=FakeEmbeddingClient(),
            fact_store=FakeFactStore(),
            graph_store=FakeGraphStore(),
            relation_extraction_client=None,
        )


async def test_write_memory_creates_entities_and_edge_for_new_relation():
    graph_store = FakeGraphStore()
    candidate = RelationCandidate(
        source_name="user", relation="works_at", target_name="Acme Corp",
        target_is_entity=True, confidence=0.95, explicit=True,
    )

    await write_memory(
        _turn(),
        extraction_client=FakeExtractionClient([]),
        resolution_client=FakeResolutionClient(),
        embedding_client=FakeEmbeddingClient(),
        fact_store=FakeFactStore(),
        graph_store=graph_store,
        relation_extraction_client=FakeRelationExtractionClient([candidate]),
    )

    source = await graph_store.find_entity_by_name("u1", "user")
    target = await graph_store.find_entity_by_name("u1", "Acme Corp")
    assert source is not None and target is not None

    edge = await graph_store.get_current_edge("u1", source.id, "works_at")
    assert edge is not None
    assert edge.target_entity_id == target.id
    assert edge.fact_sentence == "User works at Acme Corp."


async def test_write_memory_stores_literal_target_value_when_not_an_entity():
    graph_store = FakeGraphStore()
    candidate = RelationCandidate(
        source_name="user", relation="has_role", target_name="senior engineer",
        target_is_entity=False, confidence=0.9, explicit=True,
    )

    await write_memory(
        _turn(),
        extraction_client=FakeExtractionClient([]),
        resolution_client=FakeResolutionClient(),
        embedding_client=FakeEmbeddingClient(),
        fact_store=FakeFactStore(),
        graph_store=graph_store,
        relation_extraction_client=FakeRelationExtractionClient([candidate]),
    )

    source = await graph_store.find_entity_by_name("u1", "user")
    edge = await graph_store.get_current_edge("u1", source.id, "has_role")
    assert edge.target_entity_id is None
    assert edge.target_value == "senior engineer"


async def test_write_memory_reuses_existing_entity_instead_of_duplicating():
    graph_store = FakeGraphStore()
    from memory_verse_avneesh.models import Entity

    existing = Entity(user_id="u1", name="Acme Corp")
    await graph_store.create_entity(existing)

    candidate = RelationCandidate(
        source_name="user", relation="works_at", target_name="Acme Corp",
        target_is_entity=True, confidence=0.95, explicit=True,
    )
    await write_memory(
        _turn(),
        extraction_client=FakeExtractionClient([]),
        resolution_client=FakeResolutionClient(),
        embedding_client=FakeEmbeddingClient(),
        fact_store=FakeFactStore(),
        graph_store=graph_store,
        relation_extraction_client=FakeRelationExtractionClient([candidate]),
    )

    all_entities = await graph_store.list_entities("u1", limit=50, offset=0)
    acme_entities = [e for e in all_entities if e.name == "Acme Corp"]
    assert len(acme_entities) == 1
    assert acme_entities[0].id == existing.id


async def test_write_memory_contradiction_closes_old_edge_and_opens_new_one():
    graph_store = FakeGraphStore()

    first = RelationCandidate(
        source_name="user", relation="managed_by", target_name="Sarah",
        target_is_entity=True, confidence=0.9, explicit=True,
    )
    await write_memory(
        _turn(),
        extraction_client=FakeExtractionClient([]),
        resolution_client=FakeResolutionClient(),
        embedding_client=FakeEmbeddingClient(),
        fact_store=FakeFactStore(),
        graph_store=graph_store,
        relation_extraction_client=FakeRelationExtractionClient([first]),
    )
    source = await graph_store.find_entity_by_name("u1", "user")
    old_edge = await graph_store.get_current_edge("u1", source.id, "managed_by")
    assert old_edge is not None

    second = RelationCandidate(
        source_name="user", relation="managed_by", target_name="David",
        target_is_entity=True, confidence=0.9, explicit=True,
    )
    await write_memory(
        _turn(),
        extraction_client=FakeExtractionClient([]),
        resolution_client=FakeResolutionClient(),
        embedding_client=FakeEmbeddingClient(),
        fact_store=FakeFactStore(),
        graph_store=graph_store,
        relation_extraction_client=FakeRelationExtractionClient([second]),
    )

    # old edge closed, not deleted -- still fetchable directly
    reloaded_old = await graph_store.get_edge(old_edge.id)
    assert reloaded_old.valid_to is not None

    new_current = await graph_store.get_current_edge("u1", source.id, "managed_by")
    david = await graph_store.find_entity_by_name("u1", "David")
    assert new_current.target_entity_id == david.id
    assert new_current.id != old_edge.id


async def test_write_memory_unchanged_relation_is_a_noop():
    graph_store = FakeGraphStore()
    candidate = RelationCandidate(
        source_name="user", relation="works_at", target_name="Acme Corp",
        target_is_entity=True, confidence=0.95, explicit=True,
    )

    for _ in range(2):
        await write_memory(
            _turn(),
            extraction_client=FakeExtractionClient([]),
            resolution_client=FakeResolutionClient(),
            embedding_client=FakeEmbeddingClient(),
            fact_store=FakeFactStore(),
            graph_store=graph_store,
            relation_extraction_client=FakeRelationExtractionClient([candidate]),
        )

    source = await graph_store.find_entity_by_name("u1", "user")
    all_edges = await graph_store.list_edges_for_entity(source.id, limit=50, offset=0)
    # still exactly one edge -- the second identical candidate didn't
    # close-and-reopen, it recognized nothing had changed
    assert len(all_edges) == 1


async def test_write_memory_skips_low_confidence_inferred_relation():
    graph_store = FakeGraphStore()
    candidate = RelationCandidate(
        source_name="user", relation="works_at", target_name="Acme Corp",
        target_is_entity=True, confidence=MIN_COMMIT_CONFIDENCE - 0.1, explicit=False,
    )

    await write_memory(
        _turn(),
        extraction_client=FakeExtractionClient([]),
        resolution_client=FakeResolutionClient(),
        embedding_client=FakeEmbeddingClient(),
        fact_store=FakeFactStore(),
        graph_store=graph_store,
        relation_extraction_client=FakeRelationExtractionClient([candidate]),
    )

    assert await graph_store.find_entity_by_name("u1", "Acme Corp") is None

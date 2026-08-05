"""Cross-cutting resilience: read_memory() degrades gracefully on a single
backend failure (returns a partial MemoryContext, never raises), and
write_memory() is best-effort with per-store/per-candidate isolation (one
failure doesn't stop the others). Every failure is still fully logged --
these tests only assert on *behavior*, not on log output.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from memory_verse_avneesh.formation.pipeline import write_memory
from memory_verse_avneesh.models import (
    Entity,
    ExpertIdentity,
    ExtractedCandidate,
    PersonIdentity,
    RelationCandidate,
    ResolvedOperation,
    Turn,
)
from memory_verse_avneesh.read.pipeline import read_memory

from .fakes import (
    FakeEmbeddingClient,
    FakeEpisodicStore,
    FakeExtractionClient,
    FakeFactStore,
    FakeGraphStore,
    FakeIdentityStore,
    FakeProfileCache,
    FakeRelationExtractionClient,
    FakeReminderStore,
    FakeResolutionClient,
    FakeSessionCache,
)


class _Raising:
    """Wraps any fake store, making ONE named method raise on every call --
    everything else delegates straight through to the wrapped fake.
    """

    def __init__(self, wrapped, method_to_fail: str):
        self._wrapped = wrapped
        self._method_to_fail = method_to_fail

    def __getattr__(self, name):
        if name == self._method_to_fail:
            async def _raise(*args, **kwargs):
                raise RuntimeError(f"simulated failure in {name}")
            return _raise
        return getattr(self._wrapped, name)


def _turn() -> Turn:
    return Turn(
        user_id="u1", conversation_id="c1",
        user_message="a real message", assistant_message="a real reply",
    )


# --- read_memory() degrades gracefully --------------------------------


async def test_read_memory_survives_fact_store_failure():
    context = await read_memory(
        user_id="u1", conversation_id="c1", message="a real question here",
        session_cache=FakeSessionCache(), profile_cache=FakeProfileCache(profile={"tone": "concise"}),
        fact_store=_Raising(FakeFactStore(), "search_facts"),
        embedding_client=FakeEmbeddingClient(),
    )
    assert context.relevant_facts == []
    assert context.profile == {"tone": "concise"}  # unaffected store still works


async def test_read_memory_survives_episodic_store_failure():
    context = await read_memory(
        user_id="u1", conversation_id="c1", message="a real question here",
        session_cache=FakeSessionCache(), profile_cache=FakeProfileCache(),
        fact_store=FakeFactStore(), embedding_client=FakeEmbeddingClient(),
        episodic_store=_Raising(FakeEpisodicStore(), "search_episodes"),
    )
    assert context.relevant_episodes == []


async def test_read_memory_survives_graph_store_failure():
    context = await read_memory(
        user_id="u1", conversation_id="c1", message="a real question here",
        session_cache=FakeSessionCache(), profile_cache=FakeProfileCache(),
        fact_store=FakeFactStore(), embedding_client=FakeEmbeddingClient(),
        graph_store=_Raising(FakeGraphStore(), "search_current_edges"),
    )
    assert context.relevant_edges == []


async def test_read_memory_survives_identity_store_failure():
    identity_store = FakeIdentityStore(
        person_identities={"u1": PersonIdentity(user_id="u1", content="prefers formal tone")}
    )
    context = await read_memory(
        user_id="u1", conversation_id="c1", message="a real question here",
        session_cache=FakeSessionCache(), profile_cache=FakeProfileCache(),
        fact_store=FakeFactStore(), embedding_client=FakeEmbeddingClient(),
        identity_store=_Raising(identity_store, "get_person_identity"),
    )
    assert context.person_identity is None


async def test_read_memory_survives_reminder_store_failure():
    context = await read_memory(
        user_id="u1", conversation_id="c1", message="a real question here",
        session_cache=FakeSessionCache(), profile_cache=FakeProfileCache(),
        fact_store=FakeFactStore(), embedding_client=FakeEmbeddingClient(),
        reminder_store=_Raising(FakeReminderStore(), "list_due_reminders"),
    )
    assert context.due_reminders == []


async def test_read_memory_survives_session_cache_failure():
    context = await read_memory(
        user_id="u1", conversation_id="c1", message="a real question here",
        session_cache=_Raising(FakeSessionCache(), "get_recent_turns"),
        profile_cache=FakeProfileCache(profile={"tone": "concise"}),
        fact_store=FakeFactStore(), embedding_client=FakeEmbeddingClient(),
    )
    assert context.recent_turns == []
    assert context.profile == {"tone": "concise"}


async def test_read_memory_survives_profile_cache_failure():
    context = await read_memory(
        user_id="u1", conversation_id="c1", message="a real question here",
        session_cache=FakeSessionCache(turns=[Turn(user_id="u1", conversation_id="c1", user_message="hi", assistant_message="hey")]),
        profile_cache=_Raising(FakeProfileCache(), "get_profile"),
        fact_store=FakeFactStore(), embedding_client=FakeEmbeddingClient(),
    )
    assert context.profile is None
    assert len(context.recent_turns) == 1  # unaffected store still works


async def test_read_memory_survives_embedding_failure():
    context = await read_memory(
        user_id="u1", conversation_id="c1", message="a real question here",
        session_cache=FakeSessionCache(), profile_cache=FakeProfileCache(profile={"tone": "concise"}),
        fact_store=FakeFactStore(), embedding_client=_Raising(FakeEmbeddingClient(), "embed"),
    )
    # embedding failure means no Tier 2 search at all, but non-embedding-dependent
    # reads (Tier 0/1) are unaffected
    assert context.relevant_facts == []
    assert context.profile == {"tone": "concise"}


async def test_read_memory_survives_multiple_simultaneous_failures():
    context = await read_memory(
        user_id="u1", conversation_id="c1", message="a real question here",
        session_cache=_Raising(FakeSessionCache(), "get_recent_turns"),
        profile_cache=_Raising(FakeProfileCache(), "get_profile"),
        fact_store=_Raising(FakeFactStore(), "search_facts"),
        embedding_client=FakeEmbeddingClient(),
    )
    # doesn't raise despite three simultaneous backend failures
    assert context.recent_turns == []
    assert context.profile is None
    assert context.relevant_facts == []


# --- write_memory() is best-effort, isolated per store/candidate -------


async def test_write_memory_fact_store_failure_does_not_block_episodic_write():
    episodic_store = FakeEpisodicStore()
    candidate = ExtractedCandidate(category="preference", value="likes brevity", confidence=0.9, explicit=True)

    written = await write_memory(
        _turn(),
        extraction_client=FakeExtractionClient([candidate]),
        resolution_client=FakeResolutionClient(ResolvedOperation(operation="add")),
        embedding_client=FakeEmbeddingClient(),
        fact_store=_Raising(FakeFactStore(), "add_fact"),
        episodic_store=episodic_store,
    )

    assert written == []  # the fact write failed
    assert len(episodic_store.added) == 1  # episodic write still succeeded


async def test_write_memory_episodic_failure_does_not_block_fact_write():
    candidate = ExtractedCandidate(category="preference", value="likes brevity", confidence=0.9, explicit=True)

    written = await write_memory(
        _turn(),
        extraction_client=FakeExtractionClient([candidate]),
        resolution_client=FakeResolutionClient(ResolvedOperation(operation="add")),
        embedding_client=FakeEmbeddingClient(),
        fact_store=FakeFactStore(),
        episodic_store=_Raising(FakeEpisodicStore(), "add_episode"),
    )

    assert len(written) == 1  # fact write unaffected by episodic failure


async def test_write_memory_relation_extraction_failure_does_not_block_fact_write():
    graph_store = FakeGraphStore()
    candidate = ExtractedCandidate(category="preference", value="likes brevity", confidence=0.9, explicit=True)

    written = await write_memory(
        _turn(),
        extraction_client=FakeExtractionClient([candidate]),
        resolution_client=FakeResolutionClient(ResolvedOperation(operation="add")),
        embedding_client=FakeEmbeddingClient(),
        fact_store=FakeFactStore(),
        graph_store=graph_store,
        relation_extraction_client=_Raising(
            FakeRelationExtractionClient(), "extract_relations"
        ),
    )

    assert len(written) == 1  # graph failure didn't block fact writing


async def test_write_memory_one_bad_fact_candidate_does_not_block_others():
    fact_store = FakeFactStore()
    fails_first = ExtractedCandidate(category="preference", value="likes bullet points", confidence=0.9, explicit=True)
    succeeds_second = ExtractedCandidate(category="preference", value="likes brevity", confidence=0.9, explicit=True)

    call_count = {"n": 0}
    real_add_fact = fact_store.add_fact

    async def _flaky_add_fact(fact):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated failure on first candidate only")
        return await real_add_fact(fact)

    fact_store.add_fact = _flaky_add_fact

    written = await write_memory(
        _turn(),
        extraction_client=FakeExtractionClient([fails_first, succeeds_second]),
        resolution_client=FakeResolutionClient(ResolvedOperation(operation="add")),
        embedding_client=FakeEmbeddingClient(),
        fact_store=fact_store,
    )

    assert len(written) == 1
    assert written[0].value == "likes brevity"  # succeeds_second still got written


async def test_write_memory_one_bad_relation_candidate_does_not_block_others():
    graph_store = FakeGraphStore()
    bad_candidate = RelationCandidate(
        source_name="user", relation="works_at", target_name="Acme Corp",
        target_is_entity=True, confidence=0.9, explicit=True,
    )
    good_candidate = RelationCandidate(
        source_name="user", relation="managed_by", target_name="David",
        target_is_entity=True, confidence=0.9, explicit=True,
    )

    real_find_entity = graph_store.find_entity_by_name
    call_count = {"n": 0}

    async def _flaky_find_entity(user_id, name):
        if name == "Acme Corp":
            call_count["n"] += 1
            raise RuntimeError("simulated failure resolving this entity")
        return await real_find_entity(user_id, name)

    graph_store.find_entity_by_name = _flaky_find_entity

    await write_memory(
        _turn(),
        extraction_client=FakeExtractionClient([]),
        resolution_client=FakeResolutionClient(),
        embedding_client=FakeEmbeddingClient(),
        fact_store=FakeFactStore(),
        graph_store=graph_store,
        relation_extraction_client=FakeRelationExtractionClient([bad_candidate, good_candidate]),
    )

    user_entity = await graph_store.find_entity_by_name("u1", "user")
    edge = await graph_store.get_current_edge("u1", user_entity.id, "managed_by")
    assert edge is not None  # the good candidate still got written despite the bad one failing


async def test_write_memory_still_raises_on_missing_relation_extraction_client():
    # This is a caller-contract violation, not a backend failure -- it must
    # still raise, not be swallowed by the best-effort handling.
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

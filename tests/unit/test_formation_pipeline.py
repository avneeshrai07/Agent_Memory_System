from agent_memory.formation.pipeline import MIN_COMMIT_CONFIDENCE, write_memory
from agent_memory.models import ExtractedCandidate, MemoryStatus, Turn

from .fakes import FakeEmbeddingClient, FakeExtractionClient, FakeFactStore


def _turn() -> Turn:
    return Turn(
        user_id="u1",
        conversation_id="c1",
        user_message="I always want short, bullet-pointed answers",
        assistant_message="Got it, I'll keep things brief.",
    )


async def test_explicit_candidate_commits_active_regardless_of_confidence():
    candidates = [
        ExtractedCandidate(
            category="preference", value="wants bullet-pointed answers",
            confidence=0.5, explicit=True,
        ),
    ]
    fact_store = FakeFactStore()

    written = await write_memory(
        _turn(),
        extraction_client=FakeExtractionClient(candidates),
        embedding_client=FakeEmbeddingClient(),
        fact_store=fact_store,
    )

    assert len(written) == 1
    assert written[0].status == MemoryStatus.ACTIVE
    assert fact_store.added == written


async def test_implicit_low_confidence_candidate_stays_provisional():
    candidates = [
        ExtractedCandidate(
            category="preference", value="might prefer brevity",
            confidence=MIN_COMMIT_CONFIDENCE - 0.1, explicit=False,
        ),
    ]

    written = await write_memory(
        _turn(),
        extraction_client=FakeExtractionClient(candidates),
        embedding_client=FakeEmbeddingClient(),
        fact_store=FakeFactStore(),
    )

    assert written[0].status == MemoryStatus.PROVISIONAL


async def test_implicit_high_confidence_candidate_commits_active():
    candidates = [
        ExtractedCandidate(
            category="preference", value="prefers brevity",
            confidence=MIN_COMMIT_CONFIDENCE, explicit=False,
        ),
    ]

    written = await write_memory(
        _turn(),
        extraction_client=FakeExtractionClient(candidates),
        embedding_client=FakeEmbeddingClient(),
        fact_store=FakeFactStore(),
    )

    assert written[0].status == MemoryStatus.ACTIVE


async def test_each_candidate_is_embedded_and_written_independently():
    candidates = [
        ExtractedCandidate(category="a", value="fact one", confidence=1.0, explicit=True),
        ExtractedCandidate(category="b", value="fact two", confidence=1.0, explicit=True),
    ]
    embedding_client = FakeEmbeddingClient()
    fact_store = FakeFactStore()

    written = await write_memory(
        _turn(),
        extraction_client=FakeExtractionClient(candidates),
        embedding_client=embedding_client,
        fact_store=fact_store,
    )

    assert len(written) == 2
    assert embedding_client.embedded_texts == ["fact one", "fact two"]
    assert all(f.embedding is not None for f in written)


async def test_no_candidates_writes_nothing():
    written = await write_memory(
        _turn(),
        extraction_client=FakeExtractionClient([]),
        embedding_client=FakeEmbeddingClient(),
        fact_store=FakeFactStore(),
    )

    assert written == []

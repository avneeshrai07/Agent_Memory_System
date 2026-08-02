"""_row_to_fact must tolerate both shapes pgvector-python's asyncpg decoder
has returned across versions: a plain iterable (numpy array, pgvector<0.5)
and a pgvector.Vector wrapper exposing only .to_list() (pgvector>=0.5)."""

from datetime import datetime, timezone
from uuid import uuid4

from memory_verse_avneesh.storage.postgres.facts import PostgresFactStore


def _base_row(embedding) -> dict:
    return {
        "id": uuid4(),
        "user_id": "u1",
        "category": "preference",
        "value": "likes bullet points",
        "embedding": embedding,
        "confidence": 0.9,
        "observation_count": 1,
        "status": "active",
        "created_at": datetime.now(timezone.utc),
        "last_reinforced_at": datetime.now(timezone.utc),
    }


class _PlainIterableEmbedding(list):
    """Stand-in for the numpy array pgvector<0.5's decoder returned."""


class _VectorWrapperEmbedding:
    """Stand-in for pgvector>=0.5's Vector: NOT iterable, only .to_list()."""

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def to_list(self) -> list[float]:
        return list(self._values)


def test_row_to_fact_handles_plain_iterable_embedding():
    row = _base_row(_PlainIterableEmbedding([0.1, 0.2, 0.3]))
    fact = PostgresFactStore._row_to_fact(row)
    assert fact.embedding == [0.1, 0.2, 0.3]


def test_row_to_fact_handles_vector_wrapper_embedding():
    row = _base_row(_VectorWrapperEmbedding([0.4, 0.5, 0.6]))
    fact = PostgresFactStore._row_to_fact(row)
    assert fact.embedding == [0.4, 0.5, 0.6]


def test_row_to_fact_handles_none_embedding():
    row = _base_row(None)
    fact = PostgresFactStore._row_to_fact(row)
    assert fact.embedding is None

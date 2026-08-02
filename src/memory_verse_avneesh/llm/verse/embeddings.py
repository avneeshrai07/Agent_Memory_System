"""llm_verse_avneesh implementation of EmbeddingClient (README Section 4,
step 2 — the one embedding call that sits on the read path's critical path).
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4


class VerseEmbeddingClient:
    def __init__(
        self,
        router: Any,
        *,
        llm_name: str = "titan-embed-v2",
        dimensions: int | None = None,
        region_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        repo_name: str = "memory-verse-avneesh",
    ):
        self._router = router
        self._llm_name = llm_name
        self._dimensions = dimensions
        self._region_name = region_name
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key
        self._repo_name = repo_name

    async def embed(self, text: str) -> list[float]:
        result = await self._router.get_embedding(
            llm_name=self._llm_name,
            text=text,
            dimensions=self._dimensions,
            repo_name=self._repo_name,
            llm_identifier=str(uuid4()),
            region_name=self._region_name,
            aws_access_key_id=self._aws_access_key_id,
            aws_secret_access_key=self._aws_secret_access_key,
        )
        return result["embedding"]

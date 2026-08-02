"""Bedrock implementation of EmbeddingClient (README Section 4, step 2 — the
one embedding call that sits on the read path's critical path).

boto3 has no native asyncio support, so the blocking call is offloaded via
asyncio.to_thread — required so a single request doesn't stall the event
loop for every other concurrent request while it waits on the network call.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any


class BedrockEmbeddingClient:
    def __init__(
        self,
        client: Any,
        model_id: str = "amazon.titan-embed-text-v2:0",
        dimensions: int = 1024,
    ):
        self._client = client
        self._model_id = model_id
        self._dimensions = dimensions

    async def embed(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._embed_sync, text)

    def _embed_sync(self, text: str) -> list[float]:
        response = self._client.invoke_model(
            modelId=self._model_id,
            body=json.dumps({"inputText": text, "dimensions": self._dimensions}),
        )
        payload = json.loads(response["body"].read())
        return payload["embedding"]

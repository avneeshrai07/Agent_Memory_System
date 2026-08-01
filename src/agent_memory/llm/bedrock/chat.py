"""Bedrock implementation of ChatClient (README Section 4, step 6 — the one
generation call on the read path). Uses the Converse API since it's uniform
across Bedrock model families, rather than a model-specific request body.
"""

from __future__ import annotations

import asyncio
from typing import Any


class BedrockChatClient:
    def __init__(
        self,
        client: Any,
        model_id: str = "amazon.nova-lite-v1:0",
        max_tokens: int = 2000,
        temperature: float = 0.2,
    ):
        self._client = client
        self._model_id = model_id
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        return await asyncio.to_thread(self._generate_sync, system_prompt, user_prompt)

    def _generate_sync(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.converse(
            modelId=self._model_id,
            system=[{"text": system_prompt}] if system_prompt else [],
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            inferenceConfig={
                "maxTokens": self._max_tokens,
                "temperature": self._temperature,
            },
        )
        return response["output"]["message"]["content"][0]["text"]

"""LLM/embedding provider contracts. Provider-agnostic by design — Bedrock,
OpenAI, Anthropic, or a local model can all satisfy these. The first concrete
implementation is agent_memory.llm.bedrock (AWS Bedrock), added once these
contracts are settled.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent_memory.models import ExtractedCandidate, Turn


@runtime_checkable
class EmbeddingClient(Protocol):
    """The single embedding call on the read path's critical path (README
    Section 4, step 2) goes through this — keep implementations fast.
    """

    async def embed(self, text: str) -> list[float]: ...


@runtime_checkable
class ChatClient(Protocol):
    """The one generation call on the read path (README Section 4, step 6)."""

    async def generate(self, system_prompt: str, user_prompt: str) -> str: ...


@runtime_checkable
class ExtractionClient(Protocol):
    """Formation-path structured extraction (README Section 5, step 1). Runs
    off the critical path — may be a larger/slower model than ChatClient.
    """

    async def extract(self, turn: Turn) -> list[ExtractedCandidate]: ...

"""llm_verse_avneesh implementation of ExtractionClient (README Section 5,
step 1). Same prompt as llm.bedrock.extraction — dispatched through
llm_verse_avneesh's Router structured-output path (pydantic_model) instead
of a raw Bedrock Converse tool-use call, so this works across whichever
provider Router routes `llm_name` to (Bedrock, Gemini, Groq).
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from memory_verse_avneesh.models import ExtractedCandidate, Turn

_SYSTEM_PROMPT = """You are the extraction step of a memory-formation pipeline for a \
conversational agent. Extract durable facts about the user from this exchange.

Only extract information that:
- will be useful in a future, unrelated session
- stands alone without needing the rest of this conversation to make sense
- represents a stable fact, preference, constraint, or piece of expertise

Do not extract:
- one-off tasks, drafts, or wording specific to this conversation
- anything you are inferring weakly — omit rather than guess

For each fact, set:
- category: a short label (e.g. "preference", "identity", "constraint", "expertise")
- value: the atomic factual statement, standalone and self-contained
- confidence: 1.0 if the user stated it explicitly, 0.6-0.9 if strongly implied,
  otherwise do not extract it
- explicit: true only if the user stated it directly, false if inferred

Return an empty facts list if nothing qualifies."""


# No leading underscore: Nova's tool-calling strips a leading "_" from tool
# names, while langchain's with_structured_output() registers the tool
# under the literal class name -- an underscore-prefixed name (the usual
# "internal" convention) causes a real "unknown tool" mismatch at runtime.
class ExtractionBatch(BaseModel):
    facts: list[ExtractedCandidate]


class VerseExtractionClient:
    def __init__(
        self,
        router: Any,
        *,
        llm_name: str = "nova-lite",
        region_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        repo_name: str = "memory-verse-avneesh",
        max_tokens: int = 2000,
    ):
        self._router = router
        self._llm_name = llm_name
        self._region_name = region_name
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key
        self._repo_name = repo_name
        self._max_tokens = max_tokens

    async def extract(self, turn: Turn) -> list[ExtractedCandidate]:
        user_prompt = (
            f"USER MESSAGE:\n{turn.user_message}\n\n"
            f"ASSISTANT RESPONSE:\n{turn.assistant_message}"
        )

        result = await self._router.get_response(
            llm_name=self._llm_name,
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            context=None,
            temperature=0.0,
            pydantic_model=ExtractionBatch,
            max_tokens=self._max_tokens,
            repo_name=self._repo_name,
            llm_identifier=str(uuid4()),
            region_name=self._region_name,
            aws_access_key_id=self._aws_access_key_id,
            aws_secret_access_key=self._aws_secret_access_key,
        )

        candidates: list[ExtractedCandidate] = []
        for raw_fact in result["response"].get("facts", []):
            try:
                candidates.append(ExtractedCandidate(**raw_fact))
            except (TypeError, ValueError):
                continue
        return candidates

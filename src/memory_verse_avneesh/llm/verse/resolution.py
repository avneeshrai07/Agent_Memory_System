"""llm_verse_avneesh implementation of ResolutionClient (README Section 5,
step 3). Same index-not-id design as llm.bedrock.resolution — the model
picks a 1-based index into the shown existing-facts list, mapped back to a
real fact id here, never asked to produce an id directly. Dispatched
through Router's structured-output path (pydantic_model).
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from memory_verse_avneesh.models import ExtractedCandidate, MemoryOperation, ResolvedOperation, ScoredFact

_SYSTEM_PROMPT = """You are the resolution step of a memory-formation pipeline. You are \
given one newly extracted candidate fact about a user, and a numbered list of that \
user's existing memory facts that are semantically similar to it.

Decide exactly one operation:
- "add": the candidate is genuinely new information, not captured by any existing fact.
- "update": an existing fact is about the same thing but the candidate is a richer,
  more specific, or more current version of it — they should be merged.
- "delete": an existing fact is directly contradicted by the candidate (e.g. a
  preference or fact that has since changed) — the old fact should be retired.
- "noop": an existing fact already fully captures this information — nothing to do.

If you choose "update" or "delete", you must also set target_index to the 1-based
number of the existing fact it refers to. Leave target_index null for "add" or "noop"."""


# No leading underscore: Nova's tool-calling strips a leading "_" from tool
# names, while langchain's with_structured_output() registers the tool
# under the literal class name -- an underscore-prefixed name (the usual
# "internal" convention) causes a real "unknown tool" mismatch at runtime.


class ResolutionDecision(BaseModel):
    operation: str
    target_index: int | None = None


class VerseResolutionClient:
    def __init__(
        self,
        router: Any,
        *,
        llm_name: str = "nova-lite",
        region_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        repo_name: str = "memory-verse-avneesh",
        max_tokens: int = 500,
    ):
        self._router = router
        self._llm_name = llm_name
        self._region_name = region_name
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key
        self._repo_name = repo_name
        self._max_tokens = max_tokens

    async def classify_operation(
        self, candidate: ExtractedCandidate, existing: list[ScoredFact]
    ) -> ResolvedOperation:
        if not existing:
            # Nothing to compare against — no LLM call needed, it can only be ADD.
            return ResolvedOperation(operation=MemoryOperation.ADD, target_fact_id=None)

        existing_lines = "\n".join(
            f"{i + 1}. {sf.fact.value} (category: {sf.fact.category})"
            for i, sf in enumerate(existing)
        )
        user_prompt = (
            f"CANDIDATE FACT:\ncategory: {candidate.category}\nvalue: {candidate.value}\n\n"
            f"EXISTING SIMILAR FACTS:\n{existing_lines}"
        )

        result = await self._router.get_response(
            llm_name=self._llm_name,
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            context=None,
            temperature=0.0,
            pydantic_model=ResolutionDecision,
            max_tokens=self._max_tokens,
            repo_name=self._repo_name,
            llm_identifier=str(uuid4()),
            region_name=self._region_name,
            aws_access_key_id=self._aws_access_key_id,
            aws_secret_access_key=self._aws_secret_access_key,
        )

        decision = result["response"]
        try:
            operation = MemoryOperation(decision.get("operation"))
        except ValueError:
            return ResolvedOperation(operation=MemoryOperation.ADD, target_fact_id=None)

        if operation not in (MemoryOperation.UPDATE, MemoryOperation.DELETE):
            return ResolvedOperation(operation=operation, target_fact_id=None)

        index = decision.get("target_index")
        if isinstance(index, int) and 1 <= index <= len(existing):
            return ResolvedOperation(
                operation=operation, target_fact_id=existing[index - 1].fact.id
            )

        # update/delete with a missing or out-of-range index can't act on
        # anything — fall back to the safe default rather than guessing.
        return ResolvedOperation(operation=MemoryOperation.ADD, target_fact_id=None)

"""Bedrock implementation of ResolutionClient (README Section 5, step 3).

The LLM is asked to pick a 1-based index into the existing-facts list it was
shown, never an id directly — ids are easy for a model to hallucinate or
mistype, an index bounded by "how many items did I just show you" is not.
The index is mapped back to a real fact id here, before it ever reaches the
formation pipeline.
"""

from __future__ import annotations

import asyncio
from typing import Any

from memory_verse_avneesh.models import ExtractedCandidate, MemoryOperation, ResolvedOperation, ScoredFact

_TOOL_NAME = "classify_memory_operation"

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
number of the existing fact it refers to. Leave target_index unset for "add" or "noop".

Call classify_memory_operation with your decision."""

_TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": _TOOL_NAME,
                "description": "Classify how a candidate fact relates to existing memory.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": ["add", "update", "delete", "noop"],
                            },
                            "target_index": {
                                "type": "integer",
                                "description": (
                                    "1-based index into the provided existing facts "
                                    "list; omit for add/noop"
                                ),
                            },
                        },
                        "required": ["operation"],
                    }
                },
            }
        }
    ],
    "toolChoice": {"tool": {"name": _TOOL_NAME}},
}


class BedrockResolutionClient:
    def __init__(
        self,
        client: Any,
        model_id: str = "amazon.nova-lite-v1:0",
        max_tokens: int = 500,
    ):
        self._client = client
        self._model_id = model_id
        self._max_tokens = max_tokens

    async def classify_operation(
        self, candidate: ExtractedCandidate, existing: list[ScoredFact]
    ) -> ResolvedOperation:
        if not existing:
            # Nothing to compare against — no LLM call needed, it can only be ADD.
            return ResolvedOperation(operation=MemoryOperation.ADD, target_fact_id=None)

        return await asyncio.to_thread(self._classify_sync, candidate, existing)

    def _classify_sync(
        self, candidate: ExtractedCandidate, existing: list[ScoredFact]
    ) -> ResolvedOperation:
        existing_lines = "\n".join(
            f"{i + 1}. {sf.fact.value} (category: {sf.fact.category})"
            for i, sf in enumerate(existing)
        )
        user_prompt = (
            f"CANDIDATE FACT:\ncategory: {candidate.category}\nvalue: {candidate.value}\n\n"
            f"EXISTING SIMILAR FACTS:\n{existing_lines}"
        )

        response = self._client.converse(
            modelId=self._model_id,
            system=[{"text": _SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            inferenceConfig={"maxTokens": self._max_tokens, "temperature": 0.0},
            toolConfig=_TOOL_CONFIG,
        )

        for block in response["output"]["message"]["content"]:
            tool_use = block.get("toolUse")
            if not tool_use or tool_use.get("name") != _TOOL_NAME:
                continue

            tool_input = tool_use.get("input", {})
            try:
                operation = MemoryOperation(tool_input.get("operation"))
            except ValueError:
                return ResolvedOperation(operation=MemoryOperation.ADD, target_fact_id=None)

            if operation not in (MemoryOperation.UPDATE, MemoryOperation.DELETE):
                return ResolvedOperation(operation=operation, target_fact_id=None)

            index = tool_input.get("target_index")
            if isinstance(index, int) and 1 <= index <= len(existing):
                return ResolvedOperation(
                    operation=operation, target_fact_id=existing[index - 1].fact.id
                )

            # update/delete with a missing or out-of-range index can't act on
            # anything — fall back to the safe default rather than guessing.
            return ResolvedOperation(operation=MemoryOperation.ADD, target_fact_id=None)

        return ResolvedOperation(operation=MemoryOperation.ADD, target_fact_id=None)

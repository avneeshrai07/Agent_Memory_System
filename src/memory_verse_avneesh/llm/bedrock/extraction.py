"""Bedrock implementation of ExtractionClient (README Section 5, step 1).

Runs off the read path's critical path (formation pipeline only), so it can
afford a forced tool-use call for structured output instead of hoping
free-text JSON parses cleanly.
"""

from __future__ import annotations

import asyncio
from typing import Any

from memory_verse_avneesh.models import ExtractedCandidate, Turn

_TOOL_NAME = "record_extracted_facts"

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

Call record_extracted_facts with an empty facts list if nothing qualifies."""

_TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": _TOOL_NAME,
                "description": "Record the durable facts extracted from this turn.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "facts": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "category": {"type": "string"},
                                        "value": {"type": "string"},
                                        "confidence": {"type": "number"},
                                        "explicit": {"type": "boolean"},
                                    },
                                    "required": [
                                        "category",
                                        "value",
                                        "confidence",
                                        "explicit",
                                    ],
                                },
                            }
                        },
                        "required": ["facts"],
                    }
                },
            }
        }
    ],
    "toolChoice": {"tool": {"name": _TOOL_NAME}},
}


class BedrockExtractionClient:
    def __init__(
        self,
        client: Any,
        model_id: str = "amazon.nova-lite-v1:0",
        max_tokens: int = 2000,
    ):
        self._client = client
        self._model_id = model_id
        self._max_tokens = max_tokens

    async def extract(self, turn: Turn) -> list[ExtractedCandidate]:
        return await asyncio.to_thread(self._extract_sync, turn)

    def _extract_sync(self, turn: Turn) -> list[ExtractedCandidate]:
        user_prompt = (
            f"USER MESSAGE:\n{turn.user_message}\n\n"
            f"ASSISTANT RESPONSE:\n{turn.assistant_message}"
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
            if tool_use and tool_use.get("name") == _TOOL_NAME:
                candidates = []
                for raw_fact in tool_use.get("input", {}).get("facts", []):
                    try:
                        candidates.append(ExtractedCandidate(**raw_fact))
                    except (TypeError, ValueError):
                        continue
                return candidates

        return []

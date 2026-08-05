"""Bedrock implementation of RelationExtractionClient (graph memory).

Runs off the read path's critical path (formation pipeline only), so it can
afford a forced tool-use call for structured output instead of hoping
free-text JSON parses cleanly.
"""

from __future__ import annotations

import asyncio
from typing import Any

from memory_verse_avneesh.models import RelationCandidate, Turn

_TOOL_NAME = "record_extracted_relations"

_SYSTEM_PROMPT = """You are the relation-extraction step of a memory-formation \
pipeline for a conversational agent. Extract relationships between the user and \
named things (people, organizations, places) from this exchange -- not standalone \
facts, relationships specifically: (source, relation, target) triples.

Only extract a relation that:
- will be useful in a future, unrelated session
- stands alone without needing the rest of this conversation to make sense
- represents a stable relationship, not a one-off event

Do not extract:
- standalone facts with no relational structure (a separate extraction step \
already handles those)
- anything you are inferring weakly -- omit rather than guess

For each relation, set:
- source_name: who/what the relation is about. Use "user" for the person you're \
talking to.
- relation: a short snake_case verb phrase (e.g. "works_at", "managed_by", \
"has_role", "lives_in")
- target_name: the other side of the relation
- target_is_entity: true if target_name is a named thing worth tracking on its own \
(a person, organization, or place -- e.g. "Acme Corp", "Sarah"), false if it's an \
attribute/role/literal value that isn't itself a thing (e.g. "senior engineer", \
"remote")
- confidence: 1.0 if the user stated it explicitly, 0.6-0.9 if strongly implied,
  otherwise do not extract it
- explicit: true only if the user stated it directly, false if inferred

Call record_extracted_relations with an empty relations list if nothing qualifies."""

_TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": _TOOL_NAME,
                "description": "Record the relations extracted from this turn.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "relations": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "source_name": {"type": "string"},
                                        "relation": {"type": "string"},
                                        "target_name": {"type": "string"},
                                        "target_is_entity": {"type": "boolean"},
                                        "confidence": {"type": "number"},
                                        "explicit": {"type": "boolean"},
                                    },
                                    "required": [
                                        "source_name",
                                        "relation",
                                        "target_name",
                                        "target_is_entity",
                                        "confidence",
                                        "explicit",
                                    ],
                                },
                            }
                        },
                        "required": ["relations"],
                    }
                },
            }
        }
    ],
    "toolChoice": {"tool": {"name": _TOOL_NAME}},
}


class BedrockRelationExtractionClient:
    def __init__(
        self,
        client: Any,
        model_id: str = "amazon.nova-lite-v1:0",
        max_tokens: int = 2000,
    ):
        self._client = client
        self._model_id = model_id
        self._max_tokens = max_tokens

    async def extract_relations(self, turn: Turn) -> list[RelationCandidate]:
        return await asyncio.to_thread(self._extract_sync, turn)

    def _extract_sync(self, turn: Turn) -> list[RelationCandidate]:
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
                for raw_relation in tool_use.get("input", {}).get("relations", []):
                    try:
                        candidates.append(RelationCandidate(**raw_relation))
                    except (TypeError, ValueError):
                        continue
                return candidates

        return []

"""Host-owned generation logic — deliberately NOT part of memory_verse_avneesh.

This is the actual point of the retrieval/generation split: the library
hands back a MemoryContext, and the host application (this example) decides
how to turn that into a response. Here that's a plain Bedrock Converse call;
a real host could just as easily use tool-calling, streaming, a different
model per user, or a different provider entirely — memory_verse_avneesh doesn't
care and never sees this code.
"""

from __future__ import annotations

from typing import Any


def generate_response(
    client: Any, model_id: str, system_prompt: str, user_prompt: str
) -> str:
    response = client.converse(
        modelId=model_id,
        system=[{"text": system_prompt}] if system_prompt else [],
        messages=[{"role": "user", "content": [{"text": user_prompt}]}],
        inferenceConfig={"maxTokens": 2000, "temperature": 0.2},
    )
    return response["output"]["message"]["content"][0]["text"]

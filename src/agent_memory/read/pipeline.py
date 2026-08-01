"""Read-path retrieval (README Section 4).

This library retrieves memory context — it does not generate the response.
The host application takes the MemoryContext this returns, builds its own
prompt/messages, and makes its own generation call with its own model,
tools, and streaming. Once that call completes, the host constructs a Turn
itself (it has both the user's message and its own generated response) and
is responsible for calling SessionCache.append_turn() and write_memory() —
this library does not do that on the host's behalf.

The retrieval gate, deterministic recency/importance/type rerank, and
token-budget-aware packing are intentionally not implemented yet — per the
build order (README Section 8), this proves Tier 0/1/2 concurrent reads
end-to-end first; those refine a loop that already works.
"""

from __future__ import annotations

import asyncio

from agent_memory.llm.interfaces import EmbeddingClient
from agent_memory.models import MemoryContext
from agent_memory.storage.interfaces import FactStore, ProfileCache, SessionCache


async def read_memory(
    *,
    user_id: str,
    conversation_id: str,
    message: str,
    session_cache: SessionCache,
    profile_cache: ProfileCache,
    fact_store: FactStore,
    embedding_client: EmbeddingClient,
    recent_turns_limit: int = 5,
    fact_limit: int = 5,
) -> MemoryContext:
    """Embeds the message once, reads Tier 0/1/2 concurrently, and returns
    the assembled MemoryContext. No LLM generation call happens here.
    """

    query_embedding = await embedding_client.embed(message)

    recent_turns, profile, scored_facts = await asyncio.gather(
        session_cache.get_recent_turns(conversation_id, recent_turns_limit),
        profile_cache.get_profile(user_id),
        fact_store.search_facts(user_id, query_embedding, fact_limit),
    )

    return MemoryContext(
        profile=profile, relevant_facts=scored_facts, recent_turns=recent_turns
    )


def render_context_as_text(context: MemoryContext, message: str) -> str:
    """Optional convenience: flattens a MemoryContext into a single text
    block for host apps that want a drop-in string rather than building
    their own prompt from the structured pieces. Entirely optional — the
    structured MemoryContext is the real contract.
    """
    sections: list[str] = []

    if context.profile:
        profile_lines = "\n".join(
            f"- {key}: {value}" for key, value in context.profile.items()
        )
        sections.append(f"USER PROFILE:\n{profile_lines}")

    if context.relevant_facts:
        facts_lines = "\n".join(f"- {sf.fact.value}" for sf in context.relevant_facts)
        sections.append(f"RELEVANT MEMORY:\n{facts_lines}")

    if context.recent_turns:
        history_lines = "\n".join(
            f"User: {t.user_message}\nAssistant: {t.assistant_message}"
            for t in context.recent_turns
        )
        sections.append(f"RECENT CONVERSATION:\n{history_lines}")

    sections.append(f"NEW MESSAGE:\n{message}")

    return "\n\n".join(sections)

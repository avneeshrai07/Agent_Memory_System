"""Minimal read-path loop (README Section 4).

The retrieval gate, the deterministic recency/importance/type rerank, and
token-budget-aware packing are intentionally not implemented yet — per the
build order (README Section 8), this proves Tier 0/1/2 reads -> prompt ->
one LLM call end-to-end first; those refine a loop that already works
rather than gating it.
"""

from __future__ import annotations

import asyncio

from agent_memory.llm.interfaces import ChatClient, EmbeddingClient
from agent_memory.models import ScoredFact, Turn
from agent_memory.storage.interfaces import FactStore, ProfileCache, SessionCache


async def read_and_respond(
    *,
    user_id: str,
    conversation_id: str,
    message: str,
    session_cache: SessionCache,
    profile_cache: ProfileCache,
    fact_store: FactStore,
    embedding_client: EmbeddingClient,
    chat_client: ChatClient,
    system_prompt: str = "",
    recent_turns_limit: int = 5,
    fact_limit: int = 5,
) -> tuple[str, Turn]:
    """Reads Tier 0/1/2 concurrently, assembles a prompt, makes the one LLM
    call, and returns (response_text, the completed Turn).

    The caller (host application) owns what happens next: persisting the
    turn to session history and handing it to the formation path. This
    library doesn't own a queue — the formation worker is exposed, not
    owned (README Section 9).
    """

    query_embedding = await embedding_client.embed(message)

    recent_turns, profile, scored_facts = await asyncio.gather(
        session_cache.get_recent_turns(conversation_id, recent_turns_limit),
        profile_cache.get_profile(user_id),
        fact_store.search_facts(user_id, query_embedding, fact_limit),
    )

    user_prompt = _assemble_prompt(message, recent_turns, profile, scored_facts)
    response_text = await chat_client.generate(system_prompt, user_prompt)

    turn = Turn(
        user_id=user_id,
        conversation_id=conversation_id,
        user_message=message,
        assistant_message=response_text,
    )

    return response_text, turn


def _assemble_prompt(
    message: str,
    recent_turns: list[Turn],
    profile: dict | None,
    scored_facts: list[ScoredFact],
) -> str:
    sections: list[str] = []

    if profile:
        profile_lines = "\n".join(f"- {key}: {value}" for key, value in profile.items())
        sections.append(f"USER PROFILE:\n{profile_lines}")

    if scored_facts:
        facts_lines = "\n".join(f"- {sf.fact.value}" for sf in scored_facts)
        sections.append(f"RELEVANT MEMORY:\n{facts_lines}")

    if recent_turns:
        history_lines = "\n".join(
            f"User: {t.user_message}\nAssistant: {t.assistant_message}"
            for t in recent_turns
        )
        sections.append(f"RECENT CONVERSATION:\n{history_lines}")

    sections.append(f"NEW MESSAGE:\n{message}")

    return "\n\n".join(sections)

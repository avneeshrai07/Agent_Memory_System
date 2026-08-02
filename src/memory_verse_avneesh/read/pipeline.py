"""Read-path retrieval (README Section 4).

This library retrieves memory context — it does not generate the response.
The host application takes the MemoryContext this returns, builds its own
prompt/messages, and makes its own generation call with its own model,
tools, and streaming. Once that call completes, the host constructs a Turn
itself (it has both the user's message and its own generated response) and
is responsible for calling SessionCache.append_turn() and write_memory() —
this library does not do that on the host's behalf.

Everything here is cache, index, or arithmetic — never an LLM call deciding
*what* to fetch.
"""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, timezone

from memory_verse_avneesh.llm.interfaces import EmbeddingClient
from memory_verse_avneesh.models import MemoryContext, ScoredFact
from memory_verse_avneesh.storage.interfaces import FactStore, ProfileCache, SessionCache

# --- retrieval gate ---------------------------------------------------------
# Heuristic, not LLM (README Section 4, step 1). Gates only Tier 2 (the
# embedding call + vector search) — Tier 0/1 are still always read below,
# deliberately: they're O(1) cache reads, and skipping them on a one-word
# reply would break conversational continuity for no real speed win. Tier 2
# is the part actually worth skipping for a message that obviously carries
# no durable-memory-relevant content.
_TRIVIAL_MESSAGES = {
    "ok", "okay", "k", "kk", "thanks", "thank you", "thx", "got it",
    "sure", "yes", "no", "yep", "yup", "nope", "cool", "great", "nice", "np",
}
_TRIVIAL_MAX_LENGTH = 3


def should_search_tier2(message: str) -> bool:
    normalized = message.strip().lower().rstrip(".!? ")
    if not normalized:
        return False
    if normalized in _TRIVIAL_MESSAGES:
        return False
    if len(normalized) <= _TRIVIAL_MAX_LENGTH:
        return False
    return True


# --- deterministic rerank ---------------------------------------------------
# score = w1*relevance + w2*recency_decay + w3*importance + w4*type_weight
# (README Section 4, step 3). Arithmetic over already-fetched rows, no LLM,
# no additional storage round trip.

_RECENCY_HALF_LIFE_DAYS = 30.0
DEFAULT_RERANK_WEIGHTS = {
    "relevance": 0.5,
    "recency": 0.2,
    "importance": 0.2,
    "type": 0.1,
}


def _recency_decay(last_reinforced_at: datetime, now: datetime) -> float:
    age_days = max((now - last_reinforced_at).total_seconds() / 86400.0, 0.0)
    return math.exp(-age_days / _RECENCY_HALF_LIFE_DAYS)


def _rerank(
    scored_facts: list[ScoredFact],
    *,
    now: datetime,
    weights: dict[str, float],
    type_weights: dict[str, float],
) -> list[ScoredFact]:
    reranked = []
    for sf in scored_facts:
        fact = sf.fact
        combined = (
            weights["relevance"] * sf.score
            + weights["recency"] * _recency_decay(fact.last_reinforced_at, now)
            + weights["importance"] * fact.confidence
            + weights["type"] * type_weights.get(fact.category, 1.0)
        )
        reranked.append(ScoredFact(fact=fact, score=combined))

    reranked.sort(key=lambda sf: sf.score, reverse=True)
    return reranked


# --- token-budget packing ---------------------------------------------------
# Greedy, in rerank-priority order, stops at the first fact that would blow
# the budget rather than skipping ahead to smaller ones — priority order is
# the point, not maximum packing density.

DEFAULT_TOKEN_BUDGET = 500


def _approx_tokens(text: str) -> int:
    # No tokenizer dependency for this: ~4 chars/token is a standard cheap
    # approximation, adequate for a soft budget, not an exact accounting.
    return max(1, len(text) // 4)


def _pack_within_budget(scored_facts: list[ScoredFact], budget: int) -> list[ScoredFact]:
    packed: list[ScoredFact] = []
    used = 0
    for sf in scored_facts:
        cost = _approx_tokens(sf.fact.value)
        if used + cost > budget:
            break
        packed.append(sf)
        used += cost
    return packed


async def _search_tier2(
    message: str,
    user_id: str,
    fact_store: FactStore,
    embedding_client: EmbeddingClient,
    fact_limit: int,
) -> list[ScoredFact]:
    if not should_search_tier2(message):
        return []
    query_embedding = await embedding_client.embed(message)
    return await fact_store.search_facts(user_id, query_embedding, fact_limit)


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
    fact_limit: int = 20,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    rerank_weights: dict[str, float] | None = None,
    type_weights: dict[str, float] | None = None,
) -> MemoryContext:
    """Reads Tier 0/1/2 concurrently, reranks and budget-packs Tier 2, and
    returns the assembled MemoryContext. No LLM generation call happens here.

    fact_limit is the raw ANN fetch size (README's "top-20"), not the final
    count — reranking and token_budget packing narrow it down further.
    """

    recent_turns, profile, scored_facts = await asyncio.gather(
        session_cache.get_recent_turns(conversation_id, recent_turns_limit),
        profile_cache.get_profile(user_id),
        _search_tier2(message, user_id, fact_store, embedding_client, fact_limit),
    )

    reranked = _rerank(
        scored_facts,
        now=datetime.now(timezone.utc),
        weights=rerank_weights or DEFAULT_RERANK_WEIGHTS,
        type_weights=type_weights or {},
    )
    packed = _pack_within_budget(reranked, token_budget)

    return MemoryContext(profile=profile, relevant_facts=packed, recent_turns=recent_turns)


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

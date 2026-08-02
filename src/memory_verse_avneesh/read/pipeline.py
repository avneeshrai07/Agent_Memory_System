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
from memory_verse_avneesh.models import MemoryContext, ScoredEpisode, ScoredFact
from memory_verse_avneesh.storage.interfaces import (
    EpisodicStore,
    FactStore,
    IdentityStore,
    ProfileCache,
    SessionCache,
)

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


# --- episodic memory ---------------------------------------------------
# Same two-stage funnel as Tier 2 facts (gate -> ANN search -> deterministic
# rerank -> budget pack), reusing the same retrieval gate — a message too
# trivial to search facts over is equally too trivial to search episodes
# over. Simpler rerank than facts: episodes have no confidence/category, so
# just relevance + recency, no importance/type terms.

DEFAULT_EPISODE_RERANK_WEIGHTS = {
    "relevance": 0.6,
    "recency": 0.4,
}
DEFAULT_EPISODE_TOKEN_BUDGET = 300
DEFAULT_EPISODE_LIMIT = 10


def _rerank_episodes(
    scored_episodes: list[ScoredEpisode], *, now: datetime, weights: dict[str, float]
) -> list[ScoredEpisode]:
    reranked = []
    for se in scored_episodes:
        combined = (
            weights["relevance"] * se.score
            + weights["recency"] * _recency_decay(se.episode.created_at, now)
        )
        reranked.append(ScoredEpisode(episode=se.episode, score=combined))

    reranked.sort(key=lambda se: se.score, reverse=True)
    return reranked


def _pack_episodes_within_budget(
    scored_episodes: list[ScoredEpisode], budget: int
) -> list[ScoredEpisode]:
    packed: list[ScoredEpisode] = []
    used = 0
    for se in scored_episodes:
        cost = _approx_tokens(se.episode.user_message + se.episode.assistant_message)
        if used + cost > budget:
            break
        packed.append(se)
        used += cost
    return packed


async def _compute_query_embedding(
    message: str, embedding_client: EmbeddingClient
) -> list[float] | None:
    # Gated once here and reused for both Tier 2 channels below (README
    # Section 4, step 2) -- one embedding call, not one per channel.
    if not should_search_tier2(message):
        return None
    return await embedding_client.embed(message)


async def _search_facts_with_embedding(
    user_id: str, query_embedding: list[float] | None, fact_store: FactStore, fact_limit: int
) -> list[ScoredFact]:
    if query_embedding is None:
        return []
    return await fact_store.search_facts(user_id, query_embedding, fact_limit)


async def _search_episodes_with_embedding(
    user_id: str,
    query_embedding: list[float] | None,
    episodic_store: EpisodicStore | None,
    episode_limit: int,
) -> list[ScoredEpisode]:
    if query_embedding is None or episodic_store is None:
        return []
    return await episodic_store.search_episodes(user_id, query_embedding, episode_limit)


async def _get_person_identity_content(
    user_id: str, identity_store: IdentityStore | None
) -> str | None:
    if identity_store is None:
        return None
    identity = await identity_store.get_person_identity(user_id)
    return identity.content if identity else None


async def _get_expert_identity_content(
    identity_id: str | None, identity_store: IdentityStore | None
) -> str | None:
    if identity_store is None or identity_id is None:
        return None
    identity = await identity_store.get_expert_identity(identity_id)
    return identity.content if identity else None


async def read_memory(
    *,
    user_id: str,
    conversation_id: str,
    message: str,
    session_cache: SessionCache,
    profile_cache: ProfileCache,
    fact_store: FactStore,
    embedding_client: EmbeddingClient,
    identity_store: IdentityStore | None = None,
    identity_id: str | None = None,
    episodic_store: EpisodicStore | None = None,
    recent_turns_limit: int = 5,
    fact_limit: int = 20,
    episode_limit: int = DEFAULT_EPISODE_LIMIT,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    episode_token_budget: int = DEFAULT_EPISODE_TOKEN_BUDGET,
    rerank_weights: dict[str, float] | None = None,
    type_weights: dict[str, float] | None = None,
    episode_rerank_weights: dict[str, float] | None = None,
) -> MemoryContext:
    """Reads Tier 0/1/2 concurrently, reranks and budget-packs Tier 2, and
    returns the assembled MemoryContext. No LLM generation call happens here.

    fact_limit is the raw ANN fetch size (README's "top-20"), not the final
    count — reranking and token_budget packing narrow it down further.

    identity_store is optional — omit it entirely if the host doesn't use
    the identity feature. When set, the caller's person identity (if one has
    been written for this user_id) is always included; the expert identity
    is included only when identity_id is explicitly passed — there is no
    automatic selection, matching read_memory()'s never-call-an-LLM-to-decide
    -what-to-fetch contract.

    episodic_store is optional — omit it entirely if the host doesn't use
    the episodic memory feature. When set, it shares the same query
    embedding and retrieval gate as fact search (one embedding call, not
    two), then gets its own rerank (relevance + recency only) and budget
    pack, independent of the facts budget.
    """

    # Phase 1: Tier 0/1 reads, the (gated) query embedding, and identity
    # lookups all fire concurrently -- the embedding isn't awaited on its
    # own beforehand, it's just one more concurrent task in this gather.
    recent_turns, profile, query_embedding, person_identity, expert_identity = await asyncio.gather(
        session_cache.get_recent_turns(conversation_id, recent_turns_limit),
        profile_cache.get_profile(user_id),
        _compute_query_embedding(message, embedding_client),
        _get_person_identity_content(user_id, identity_store),
        _get_expert_identity_content(identity_id, identity_store),
    )

    # Phase 2: both Tier 2 channels reuse that one embedding, fired
    # concurrently with each other (README Section 4, step 2).
    scored_facts, scored_episodes = await asyncio.gather(
        _search_facts_with_embedding(user_id, query_embedding, fact_store, fact_limit),
        _search_episodes_with_embedding(user_id, query_embedding, episodic_store, episode_limit),
    )

    now = datetime.now(timezone.utc)

    reranked = _rerank(
        scored_facts,
        now=now,
        weights=rerank_weights or DEFAULT_RERANK_WEIGHTS,
        type_weights=type_weights or {},
    )
    packed = _pack_within_budget(reranked, token_budget)

    reranked_episodes = _rerank_episodes(
        scored_episodes, now=now, weights=episode_rerank_weights or DEFAULT_EPISODE_RERANK_WEIGHTS
    )
    packed_episodes = _pack_episodes_within_budget(reranked_episodes, episode_token_budget)

    return MemoryContext(
        profile=profile,
        relevant_facts=packed,
        recent_turns=recent_turns,
        relevant_episodes=packed_episodes,
        person_identity=person_identity,
        expert_identity=expert_identity,
    )


def render_context_as_text(context: MemoryContext, message: str) -> str:
    """Optional convenience: flattens a MemoryContext into a single text
    block for host apps that want a drop-in string rather than building
    their own prompt from the structured pieces. Entirely optional — the
    structured MemoryContext is the real contract.
    """
    sections: list[str] = []

    if context.expert_identity:
        sections.append(f"EXPERT IDENTITY:\n{context.expert_identity}")

    if context.person_identity:
        sections.append(f"PERSON IDENTITY:\n{context.person_identity}")

    if context.profile:
        profile_lines = "\n".join(
            f"- {key}: {value}" for key, value in context.profile.items()
        )
        sections.append(f"USER PROFILE:\n{profile_lines}")

    if context.relevant_facts:
        facts_lines = "\n".join(f"- {sf.fact.value}" for sf in context.relevant_facts)
        sections.append(f"RELEVANT MEMORY:\n{facts_lines}")

    if context.relevant_episodes:
        episode_lines = "\n".join(
            f"- User: {se.episode.user_message}\n  Assistant: {se.episode.assistant_message}"
            for se in context.relevant_episodes
        )
        sections.append(f"RELEVANT PAST CONVERSATIONS:\n{episode_lines}")

    if context.recent_turns:
        history_lines = "\n".join(
            f"User: {t.user_message}\nAssistant: {t.assistant_message}"
            for t in context.recent_turns
        )
        sections.append(f"RECENT CONVERSATION:\n{history_lines}")

    sections.append(f"NEW MESSAGE:\n{message}")

    return "\n\n".join(sections)

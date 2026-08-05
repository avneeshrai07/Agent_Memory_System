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
import logging
import math
from datetime import datetime, timezone

from uuid import UUID

from memory_verse_avneesh.llm.interfaces import EmbeddingClient
from memory_verse_avneesh.models import MemoryContext, Reminder, ScoredEdge, ScoredEpisode, ScoredFact, Turn
from memory_verse_avneesh.storage.interfaces import (
    EpisodicStore,
    FactStore,
    GraphStore,
    IdentityStore,
    ProfileCache,
    ReminderStore,
    SessionCache,
)

logger = logging.getLogger(__name__)

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


# --- graph memory -------------------------------------------------------
# Same funnel again: gate -> ANN search over Edge.fact_sentence embeddings
# (current edges only, valid_to IS NULL) -> deterministic rerank -> budget
# pack. Additionally, a bounded one-hop expansion: for each directly
# matched edge, also pull the other CURRENT edges touching its entities --
# pure similarity search finds the right entry point but won't surface a
# connected fact one hop away (README's "who does my manager manage"
# example), and this is the simple, non-recursive version of that.

DEFAULT_EDGE_RERANK_WEIGHTS = {
    "relevance": 0.6,
    "recency": 0.4,
}
DEFAULT_EDGE_TOKEN_BUDGET = 300
DEFAULT_EDGE_LIMIT = 10
EDGE_EXPANSION_LIMIT_PER_ENTITY = 3
# Fixed base "relevance" for one-hop-expanded edges -- they weren't matched
# by similarity at all, so they shouldn't be able to outrank a genuine
# direct match; this just lets them compete on recency for the remaining
# budget instead of being silently dropped.
EDGE_EXPANSION_BASE_SCORE = 0.3


def _rerank_edges(
    scored_edges: list[ScoredEdge], *, now: datetime, weights: dict[str, float]
) -> list[ScoredEdge]:
    reranked = []
    for se in scored_edges:
        combined = (
            weights["relevance"] * se.score
            + weights["recency"] * _recency_decay(se.edge.observed_at, now)
        )
        reranked.append(ScoredEdge(edge=se.edge, score=combined))

    reranked.sort(key=lambda se: se.score, reverse=True)
    return reranked


def _pack_edges_within_budget(scored_edges: list[ScoredEdge], budget: int) -> list[ScoredEdge]:
    packed: list[ScoredEdge] = []
    used = 0
    for se in scored_edges:
        cost = _approx_tokens(se.edge.fact_sentence)
        if used + cost > budget:
            break
        packed.append(se)
        used += cost
    return packed


async def _search_edges_with_embedding(
    user_id: str,
    query_embedding: list[float] | None,
    graph_store: GraphStore | None,
    edge_limit: int,
) -> list[ScoredEdge]:
    if query_embedding is None or graph_store is None:
        return []
    try:
        results = await graph_store.search_current_edges(user_id, query_embedding, edge_limit)
    except Exception:
        logger.exception(
            "read_memory: graph edge search failed for user_id=%s -- continuing "
            "with no edges for this call", user_id,
        )
        return []
    logger.info("read_memory: graph search -- edge_count=%d", len(results))
    return results


async def _expand_edges_one_hop(
    matched: list[ScoredEdge], graph_store: GraphStore | None
) -> list[ScoredEdge]:
    if graph_store is None or not matched:
        return []

    seen_edge_ids = {se.edge.id for se in matched}
    entity_ids: set[UUID] = set()
    for se in matched:
        entity_ids.add(se.edge.source_entity_id)
        if se.edge.target_entity_id is not None:
            entity_ids.add(se.edge.target_entity_id)

    expanded: list[ScoredEdge] = []
    for entity_id in entity_ids:
        try:
            neighbors = await graph_store.list_edges_for_entity(
                entity_id, EDGE_EXPANSION_LIMIT_PER_ENTITY, 0
            )
        except Exception:
            logger.exception(
                "read_memory: one-hop expansion failed for entity_id=%s -- "
                "skipping expansion around this entity, other matches unaffected",
                entity_id,
            )
            continue
        for edge in neighbors:
            if edge.valid_to is not None or edge.id in seen_edge_ids:
                continue
            seen_edge_ids.add(edge.id)
            expanded.append(ScoredEdge(edge=edge, score=EDGE_EXPANSION_BASE_SCORE))

    logger.info("read_memory: graph one-hop expansion -- expanded_edge_count=%d", len(expanded))
    return expanded


async def _compute_query_embedding(
    message: str, embedding_client: EmbeddingClient
) -> list[float] | None:
    # Gated once here and reused for both Tier 2 channels below (README
    # Section 4, step 2) -- one embedding call, not one per channel.
    if not should_search_tier2(message):
        logger.info("read_memory: retrieval gate skipped Tier 2 (trivial message)")
        return None
    try:
        embedding = await embedding_client.embed(message)
    except Exception:
        logger.exception(
            "read_memory: query embedding call failed -- continuing with no Tier 2 "
            "search this call (facts/episodes/edges will all be empty)"
        )
        return None
    logger.info("read_memory: query embedding computed, dim=%d", len(embedding))
    return embedding


async def _search_facts_with_embedding(
    user_id: str, query_embedding: list[float] | None, fact_store: FactStore, fact_limit: int
) -> list[ScoredFact]:
    if query_embedding is None:
        return []
    try:
        results = await fact_store.search_facts(user_id, query_embedding, fact_limit)
    except Exception:
        logger.exception(
            "read_memory: fact search failed for user_id=%s -- continuing with no "
            "facts for this call", user_id,
        )
        return []
    logger.info("read_memory: fact search -- fact_count=%d", len(results))
    return results


async def _search_episodes_with_embedding(
    user_id: str,
    query_embedding: list[float] | None,
    episodic_store: EpisodicStore | None,
    episode_limit: int,
) -> list[ScoredEpisode]:
    if query_embedding is None or episodic_store is None:
        return []
    try:
        results = await episodic_store.search_episodes(user_id, query_embedding, episode_limit)
    except Exception:
        logger.exception(
            "read_memory: episode search failed for user_id=%s -- continuing with "
            "no episodes for this call", user_id,
        )
        return []
    logger.info("read_memory: episode search -- episode_count=%d", len(results))
    return results


async def _get_recent_turns_safe(
    session_cache: SessionCache, conversation_id: str, limit: int
) -> list[Turn]:
    try:
        turns = await session_cache.get_recent_turns(conversation_id, limit)
    except Exception:
        logger.exception(
            "read_memory: Tier 0 session cache read failed for conversation_id=%s "
            "-- continuing with no recent turns for this call", conversation_id,
        )
        return []
    logger.info("read_memory: Tier 0 session cache -- recent_turn_count=%d", len(turns))
    return turns


async def _get_profile_safe(profile_cache: ProfileCache, user_id: str) -> dict | None:
    try:
        profile = await profile_cache.get_profile(user_id)
    except Exception:
        logger.exception(
            "read_memory: Tier 1 profile cache read failed for user_id=%s -- "
            "continuing with no profile for this call", user_id,
        )
        return None
    logger.info("read_memory: Tier 1 profile cache -- present=%s", profile is not None)
    return profile


async def _get_person_identity_content(
    user_id: str, identity_store: IdentityStore | None
) -> str | None:
    if identity_store is None:
        return None
    try:
        identity = await identity_store.get_person_identity(user_id)
    except Exception:
        logger.exception(
            "read_memory: person identity read failed for user_id=%s -- continuing "
            "with no person identity for this call", user_id,
        )
        return None
    content = identity.content if identity else None
    logger.info("read_memory: person identity -- present=%s", content is not None)
    return content


async def _get_expert_identity_content(
    identity_id: str | None, identity_store: IdentityStore | None
) -> str | None:
    if identity_store is None or identity_id is None:
        return None
    try:
        identity = await identity_store.get_expert_identity(identity_id)
    except Exception:
        logger.exception(
            "read_memory: expert identity read failed for identity_id=%s -- "
            "continuing with no expert identity for this call", identity_id,
        )
        return None
    content = identity.content if identity else None
    logger.info("read_memory: expert identity -- identity_id=%s present=%s", identity_id, content is not None)
    return content


async def _get_due_reminders(
    user_id: str, reminder_store: ReminderStore | None, now: datetime
) -> list[Reminder]:
    # Deterministic time comparison, not a similarity search -- no
    # embedding involved, same "always fetched, no LLM judgment" contract
    # as person identity.
    if reminder_store is None:
        return []
    try:
        due = await reminder_store.list_due_reminders(user_id, now)
    except Exception:
        logger.exception(
            "read_memory: due-reminders read failed for user_id=%s -- continuing "
            "with no reminders for this call", user_id,
        )
        return []
    logger.info("read_memory: due reminders -- due_count=%d", len(due))
    return due


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
    reminder_store: ReminderStore | None = None,
    graph_store: GraphStore | None = None,
    recent_turns_limit: int = 5,
    fact_limit: int = 20,
    episode_limit: int = DEFAULT_EPISODE_LIMIT,
    edge_limit: int = DEFAULT_EDGE_LIMIT,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    episode_token_budget: int = DEFAULT_EPISODE_TOKEN_BUDGET,
    edge_token_budget: int = DEFAULT_EDGE_TOKEN_BUDGET,
    rerank_weights: dict[str, float] | None = None,
    type_weights: dict[str, float] | None = None,
    episode_rerank_weights: dict[str, float] | None = None,
    edge_rerank_weights: dict[str, float] | None = None,
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

    reminder_store is optional — omit it entirely if the host doesn't use
    the prospective memory feature. When set, PENDING reminders with
    due_at <= now are always included (a plain time comparison, not a
    similarity search) — no identity_id-style opt-in needed, since "is it
    due yet" has nothing to do with the current message's content.

    graph_store is optional — omit it entirely if the host doesn't use the
    graph memory feature. When set, it shares the same query embedding and
    retrieval gate as fact/episode search, searching current-truth edges
    (valid_to IS NULL) by their fact_sentence embedding, then expands one
    hop out from whatever matched to pick up directly connected edges too.
    """

    now = datetime.now(timezone.utc)
    logger.info(
        "read_memory: start -- user_id=%s conversation_id=%s message=%r",
        user_id, conversation_id, message,
    )

    # Phase 1: Tier 0/1 reads, the (gated) query embedding, identity
    # lookups, and due reminders all fire concurrently -- the embedding
    # isn't awaited on its own beforehand, it's just one more concurrent
    # task in this gather. Every coroutine in this gather catches its own
    # exceptions internally (degrade gracefully, not fail the whole call) --
    # see each helper's own try/except.
    (
        recent_turns,
        profile,
        query_embedding,
        person_identity,
        expert_identity,
        due_reminders,
    ) = await asyncio.gather(
        _get_recent_turns_safe(session_cache, conversation_id, recent_turns_limit),
        _get_profile_safe(profile_cache, user_id),
        _compute_query_embedding(message, embedding_client),
        _get_person_identity_content(user_id, identity_store),
        _get_expert_identity_content(identity_id, identity_store),
        _get_due_reminders(user_id, reminder_store, now),
    )

    # Phase 2: all three Tier 2 channels reuse that one embedding, fired
    # concurrently with each other (README Section 4, step 2).
    scored_facts, scored_episodes, scored_edges = await asyncio.gather(
        _search_facts_with_embedding(user_id, query_embedding, fact_store, fact_limit),
        _search_episodes_with_embedding(user_id, query_embedding, episodic_store, episode_limit),
        _search_edges_with_embedding(user_id, query_embedding, graph_store, edge_limit),
    )

    # Phase 3: one-hop expansion depends on which edges matched in phase 2,
    # so it can't join that gather -- it's the next sequential step.
    expanded_edges = await _expand_edges_one_hop(scored_edges, graph_store)

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

    reranked_edges = _rerank_edges(
        scored_edges + expanded_edges,
        now=now,
        weights=edge_rerank_weights or DEFAULT_EDGE_RERANK_WEIGHTS,
    )
    packed_edges = _pack_edges_within_budget(reranked_edges, edge_token_budget)

    context = MemoryContext(
        profile=profile,
        relevant_facts=packed,
        recent_turns=recent_turns,
        relevant_episodes=packed_episodes,
        relevant_edges=packed_edges,
        due_reminders=due_reminders,
        person_identity=person_identity,
        expert_identity=expert_identity,
    )

    logger.info(
        "read_memory: assembled context -- user_id=%s facts=%d episodes=%d "
        "edges=%d due_reminders=%d person_identity=%s expert_identity=%s recent_turns=%d",
        user_id, len(packed), len(packed_episodes), len(packed_edges), len(due_reminders),
        person_identity is not None, expert_identity is not None, len(recent_turns),
    )
    # The prompt built from memory retrieval alone -- NOT the user's live
    # message, which the host appends itself (see render_context_as_text).
    # This is what got read from memory, independent of whatever the host
    # ultimately does with it.
    logger.info(
        "read_memory: memory-derived prompt (read only, excludes the user's "
        "message) for user_id=%s:\n%s", user_id, _render_memory_sections(context),
    )

    return context


def _render_memory_sections(context: MemoryContext) -> str:
    """The memory-derived portion of the prompt -- everything read_memory()
    itself retrieved, with no user message mixed in. render_context_as_text()
    below just appends the live message to this; read_memory() also logs
    this on its own (read-only, from-memory prompt) once retrieval finishes.
    """
    sections: list[str] = []

    if context.expert_identity:
        sections.append(f"EXPERT IDENTITY:\n{context.expert_identity}")

    if context.person_identity:
        sections.append(f"PERSON IDENTITY:\n{context.person_identity}")

    if context.due_reminders:
        reminder_lines = "\n".join(
            f"- {r.content} (due {r.due_at.isoformat()})" for r in context.due_reminders
        )
        sections.append(f"DUE REMINDERS:\n{reminder_lines}")

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

    if context.relevant_edges:
        edge_lines = "\n".join(f"- {se.edge.fact_sentence}" for se in context.relevant_edges)
        sections.append(f"RELEVANT RELATIONSHIPS:\n{edge_lines}")

    if context.recent_turns:
        history_lines = "\n".join(
            f"User: {t.user_message}\nAssistant: {t.assistant_message}"
            for t in context.recent_turns
        )
        sections.append(f"RECENT CONVERSATION:\n{history_lines}")

    return "\n\n".join(sections)


def render_context_as_text(context: MemoryContext, message: str) -> str:
    """Optional convenience: flattens a MemoryContext into a single text
    block for host apps that want a drop-in string rather than building
    their own prompt from the structured pieces. Entirely optional — the
    structured MemoryContext is the real contract. Memory-derived sections
    come from _render_memory_sections(); the user's live message is appended
    here, not in that shared helper (read_memory() logs the memory-only
    portion on its own, before it ever sees this message).
    """
    memory_sections = _render_memory_sections(context)
    new_message_section = f"NEW MESSAGE:\n{message}"

    if not memory_sections:
        return new_message_section
    return f"{memory_sections}\n\n{new_message_section}"

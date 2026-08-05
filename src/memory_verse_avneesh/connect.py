"""Top-level entry point: one call that does all the wiring a host would
otherwise hand-assemble (a Postgres pool, every Tier-2+ store with
ensure_schema() called on each, the Tier 0/1 cache backend, and every LLM
client) and hands back a single Memory object with a minimal read()/write()
surface for the request path.

This is the recommended way to start using the library. Building
MemoryConfig and each backend by hand (see examples/fastapi_app) is still
fully supported for hosts that need custom wiring connect() doesn't cover —
a non-default pool size, swapping in a different LLM client implementation,
etc.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from memory_verse_avneesh.config import MemoryConfig
from memory_verse_avneesh.formation import write_memory
from memory_verse_avneesh.llm.interfaces import (
    EmbeddingClient,
    ExtractionClient,
    RelationExtractionClient,
    ResolutionClient,
)
from memory_verse_avneesh.models import MemoryContext, MemoryFact, Turn
from memory_verse_avneesh.read import read_memory, render_context_as_text
from memory_verse_avneesh.storage.interfaces import (
    EpisodicStore,
    FactStore,
    GraphStore,
    IdentityStore,
    ProfileCache,
    ReminderStore,
    SessionCache,
)
from memory_verse_avneesh.storage.postgres import (
    PostgresEpisodicStore,
    PostgresFactStore,
    PostgresGraphStore,
    PostgresIdentityStore,
    PostgresReminderStore,
    create_pool,
)

logger = logging.getLogger(__name__)


class Memory:
    """Everything read_memory()/write_memory() need, bound once by
    connect(). Call .read()/.write() with just the request-specific
    arguments — no backend plumbing on every call.

    Every underlying store/client is also a public attribute (fact_store,
    identity_store, episodic_store, reminder_store, graph_store,
    session_cache, profile_cache, embedding_client, extraction_client,
    resolution_client, relation_extraction_client) — reach for these
    directly when using memory_verse_avneesh.identity/episodic/prospective/
    graph/management's own functions, which all take a store as a keyword
    argument the same way read_memory()/write_memory() do.
    """

    def __init__(
        self,
        *,
        fact_store: FactStore,
        identity_store: IdentityStore,
        episodic_store: EpisodicStore,
        reminder_store: ReminderStore,
        graph_store: GraphStore,
        session_cache: SessionCache,
        profile_cache: ProfileCache,
        embedding_client: EmbeddingClient,
        extraction_client: ExtractionClient,
        resolution_client: ResolutionClient,
        relation_extraction_client: RelationExtractionClient,
        close: Callable[[], Awaitable[None]],
    ):
        self.fact_store = fact_store
        self.identity_store = identity_store
        self.episodic_store = episodic_store
        self.reminder_store = reminder_store
        self.graph_store = graph_store
        self.session_cache = session_cache
        self.profile_cache = profile_cache
        self.embedding_client = embedding_client
        self.extraction_client = extraction_client
        self.resolution_client = resolution_client
        self.relation_extraction_client = relation_extraction_client
        self._close = close
        # Strong references to in-flight background formation tasks.
        # asyncio only holds a *weak* reference to a running task, so a task
        # whose only reference is the local in write() can be garbage
        # collected mid-run and silently vanish -- keeping the set here (and
        # discarding on completion) is what actually keeps them alive.
        self._background_tasks: set[asyncio.Task] = set()

    async def read(
        self,
        *,
        user_id: str,
        conversation_id: str,
        message: str,
        identity_id: str | None = None,
        **kwargs: Any,
    ) -> MemoryContext:
        """read_memory() with every backend already bound — only the
        request-specific arguments are needed. Extra keyword arguments pass
        through to read_memory() for the rarer tuning knobs (fact_limit,
        token_budget, rerank_weights, ...) — see its own docstring.
        """
        return await read_memory(
            user_id=user_id,
            conversation_id=conversation_id,
            message=message,
            identity_id=identity_id,
            session_cache=self.session_cache,
            profile_cache=self.profile_cache,
            fact_store=self.fact_store,
            embedding_client=self.embedding_client,
            identity_store=self.identity_store,
            episodic_store=self.episodic_store,
            reminder_store=self.reminder_store,
            graph_store=self.graph_store,
            **kwargs,
        )

    async def write(
        self,
        *,
        user_id: str,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
        **kwargs: Any,
    ) -> Turn:
        """Record one completed exchange. Builds the Turn for you, appends
        it to session history, and runs formation (extraction, resolution,
        episodic + graph writes) **in the background** — no BackgroundTasks
        plumbing needed from the host.

        Returns as soon as the turn is in session history, so the next turn
        in the same conversation always sees it. Formation continues after
        this returns; it never blocks the response you're about to send.

        Formation failures never surface here — write_memory() is
        best-effort with per-store isolation and logs everything it catches
        (see README Section 7). Extra keyword arguments pass through to
        write_memory().
        """
        turn = Turn(
            user_id=user_id,
            conversation_id=conversation_id,
            user_message=user_message,
            assistant_message=assistant_message,
        )

        # Inline, not backgrounded: this is an O(1) cache write, and the
        # very next request in this conversation reads it as Tier 0
        # history. Backgrounding it would race that read.
        try:
            await self.session_cache.append_turn(turn)
        except Exception:
            logger.exception(
                "Memory.write: session history append failed for conversation_id=%s "
                "-- continuing to formation anyway", conversation_id,
            )

        task = asyncio.create_task(self._run_formation(turn, **kwargs))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        return turn

    async def _run_formation(self, turn: Turn, **kwargs: Any) -> list[MemoryFact]:
        """write_memory() with every backend already bound. Runs as a
        background task; write_memory() handles its own per-store failures,
        so this only guards against anything escaping that.
        """
        try:
            return await write_memory(
                turn,
                extraction_client=self.extraction_client,
                resolution_client=self.resolution_client,
                embedding_client=self.embedding_client,
                fact_store=self.fact_store,
                episodic_store=self.episodic_store,
                graph_store=self.graph_store,
                relation_extraction_client=self.relation_extraction_client,
                **kwargs,
            )
        except Exception:
            logger.exception(
                "Memory.write: background formation failed for user_id=%s turn_id=%s",
                turn.user_id, turn.id,
            )
            return []

    async def flush(self, timeout: float | None = 30.0) -> None:
        """Wait for in-flight background formation to finish. close() calls
        this already; call it directly only if you need to observe writes
        immediately (tests, scripts that exit right after writing).
        """
        if not self._background_tasks:
            return
        pending = set(self._background_tasks)
        logger.info("Memory.flush: waiting on %d background formation task(s)", len(pending))
        done, still_pending = await asyncio.wait(pending, timeout=timeout)
        if still_pending:
            logger.warning(
                "Memory.flush: %d formation task(s) still running after %.1fs -- "
                "leaving them; they'll be cancelled if the loop shuts down",
                len(still_pending), timeout,
            )

    @staticmethod
    def render_prompt(context: MemoryContext, message: str) -> str:
        """Convenience passthrough to render_context_as_text()."""
        return render_context_as_text(context, message)

    async def close(self) -> None:
        """Waits for in-flight background formation, then closes the
        Postgres pool and the cache backend's client. Call this on host
        shutdown — closing the pool with formation still running would kill
        those writes mid-flight.
        """
        await self.flush()
        await self._close()


class MemoryPool:
    """What setup() returns: a lazily-connected handle you hand straight to
    your app's lifespan.

    setup() itself does no I/O — it just validates config and captures it,
    so it's safe to call at import time / module scope. The pool connects on
    `async with`, and closes (flushing background formation first) on exit:

        memory_pool = setup(database_url=..., postgres_schema=..., ...)

        @asynccontextmanager
        async def lifespan(app):
            async with memory_pool as memory:
                yield

    `memory` there is the same Memory object connect() returns. After entry
    it's also available as `memory_pool.memory`, for modules that can't take
    it as an argument.
    """

    def __init__(self, **kwargs: Any):
        # chat_model_name isn't a connect()/MemoryConfig argument at all --
        # it's the host's own generation model choice, never read or used by
        # memory_verse_avneesh itself ("host owns generation"). Popped out
        # here so it doesn't get passed through to connect() below, and
        # carried purely as a convenience slot so setup() can be the one
        # place all your app config lives.
        self.chat_model_name = kwargs.pop("chat_model_name", None)

        # Validate now, at setup() time, rather than surfacing a config typo
        # only once the app tries to boot. MemoryConfig's __post_init__ does
        # all the real checking; the instance is discarded, connect() builds
        # its own from the same kwargs.
        MemoryConfig(
            database_url=kwargs.get("database_url"),
            postgres_host=kwargs.get("postgres_host"),
            postgres_port=kwargs.get("postgres_port"),
            postgres_user=kwargs.get("postgres_user"),
            postgres_password=kwargs.get("postgres_password"),
            postgres_database=kwargs.get("postgres_database"),
            postgres_schema=kwargs["postgres_schema"],
            redis_url=kwargs.get("redis_url"),
            upstash_url=kwargs.get("upstash_url"),
            upstash_token=kwargs.get("upstash_token"),
        )

        self._kwargs = kwargs
        self.memory: Memory | None = None

    async def __aenter__(self) -> Memory:
        self.memory = await connect(**self._kwargs)
        return self.memory

    async def __aexit__(self, *exc_info: Any) -> None:
        if self.memory is not None:
            await self.memory.close()
            self.memory = None


def setup(**kwargs: Any) -> MemoryPool:
    """Capture config for a Memory without connecting yet — returns a
    MemoryPool to hand to your app's lifespan (see MemoryPool's docstring).

    Takes connect()'s arguments, plus one extra: chat_model_name (optional).
    That one isn't read or used by memory_verse_avneesh at all — it's purely
    a convenience slot for your own generation model choice, so setup() can
    be the one place your app's config lives instead of splitting it across
    two calls. Read it back as memory_pool.chat_model_name (or
    memory.chat_model_name, via the memory module).

    Config errors raise here, at setup() time, not later during startup.
    Use connect() directly instead if you're already inside an async
    context and don't need the deferred handle.
    """
    return MemoryPool(**kwargs)


async def connect(
    *,
    database_url: str | None = None,
    postgres_host: str | None = None,
    postgres_port: int | None = None,
    postgres_user: str | None = None,
    postgres_password: str | None = None,
    postgres_database: str | None = None,
    postgres_schema: str,
    redis_url: str | None = None,
    upstash_url: str | None = None,
    upstash_token: str | None = None,
    aws_region: str = "us-east-1",
    aws_llm_access_key_id: str | None = None,
    aws_llm_secret_access_key: str | None = None,
    embedding_dim: int = 1024,
    embedding_model_id: str = "amazon.titan-embed-text-v2:0",
    extraction_model_id: str = "amazon.nova-lite-v1:0",
) -> Memory:
    """One call: builds MemoryConfig, creates the Postgres pool, constructs
    every Tier-2+ store and calls ensure_schema() on each, constructs the
    Tier 0/1 cache backend and every LLM client, and returns a bound Memory.

    LLM calls go straight to AWS Bedrock; the Tier 0/1 cache goes straight
    to Redis or Upstash (whichever you configured) — no extra dependency
    beyond this library's own postgres/redis|upstash/bedrock extras.

    Every argument is exactly MemoryConfig's own fields — see its
    docstring for the database_url-vs-host and redis_url-vs-upstash
    validation rules. Raises whatever MemoryConfig's __post_init__ raises
    for a misconfiguration, before any connection is attempted.
    """
    config = MemoryConfig(
        database_url=database_url,
        postgres_host=postgres_host,
        postgres_port=postgres_port,
        postgres_user=postgres_user,
        postgres_password=postgres_password,
        postgres_database=postgres_database,
        postgres_schema=postgres_schema,
        redis_url=redis_url,
        upstash_url=upstash_url,
        upstash_token=upstash_token,
        aws_region=aws_region,
        aws_llm_access_key_id=aws_llm_access_key_id,
        aws_llm_secret_access_key=aws_llm_secret_access_key,
        embedding_dim=embedding_dim,
        embedding_model_id=embedding_model_id,
        extraction_model_id=extraction_model_id,
    )

    logger.info("connect: creating Postgres pool, schema=%s", config.postgres_schema)
    pool = await create_pool(
        config.database_url,
        host=config.postgres_host, port=config.postgres_port,
        user=config.postgres_user, password=config.postgres_password,
        database=config.postgres_database,
    )

    fact_store = PostgresFactStore(pool, embedding_dim=config.embedding_dim, schema=config.postgres_schema)
    identity_store = PostgresIdentityStore(pool, schema=config.postgres_schema)
    episodic_store = PostgresEpisodicStore(pool, embedding_dim=config.embedding_dim, schema=config.postgres_schema)
    reminder_store = PostgresReminderStore(pool, schema=config.postgres_schema)
    graph_store = PostgresGraphStore(pool, embedding_dim=config.embedding_dim, schema=config.postgres_schema)

    await fact_store.ensure_schema()
    await identity_store.ensure_schema()
    await episodic_store.ensure_schema()
    await reminder_store.ensure_schema()
    await graph_store.ensure_schema()
    logger.info("connect: all tables ready under schema=%s", config.postgres_schema)

    session_cache, profile_cache, close_cache = _build_cache(config)
    embedding_client, extraction_client, resolution_client, relation_extraction_client = (
        _build_llm_clients(config)
    )

    async def _close() -> None:
        await pool.close()
        await close_cache()

    logger.info("connect: ready")
    return Memory(
        fact_store=fact_store,
        identity_store=identity_store,
        episodic_store=episodic_store,
        reminder_store=reminder_store,
        graph_store=graph_store,
        session_cache=session_cache,
        profile_cache=profile_cache,
        embedding_client=embedding_client,
        extraction_client=extraction_client,
        resolution_client=resolution_client,
        relation_extraction_client=relation_extraction_client,
        close=_close,
    )


def _build_cache(config: MemoryConfig):
    from memory_verse_avneesh.storage.redis import RedisProfileCache, RedisSessionCache, create_redis_client
    from memory_verse_avneesh.storage.upstash import (
        UpstashProfileCache,
        UpstashSessionCache,
        create_upstash_client,
    )

    if config.redis_url is not None:
        client = create_redis_client(config.redis_url)
        return RedisSessionCache(client), RedisProfileCache(client), client.aclose

    assert config.upstash_url is not None and config.upstash_token is not None
    client = create_upstash_client(config.upstash_url, config.upstash_token)
    return UpstashSessionCache(client), UpstashProfileCache(client), client.close


def _build_llm_clients(config: MemoryConfig):
    from memory_verse_avneesh.llm.bedrock import (
        BedrockEmbeddingClient,
        BedrockExtractionClient,
        BedrockRelationExtractionClient,
        BedrockResolutionClient,
        create_bedrock_client,
    )

    client = create_bedrock_client(
        config.aws_region,
        aws_access_key_id=config.aws_llm_access_key_id,
        aws_secret_access_key=config.aws_llm_secret_access_key,
    )
    return (
        BedrockEmbeddingClient(client, model_id=config.embedding_model_id, dimensions=config.embedding_dim),
        BedrockExtractionClient(client, model_id=config.extraction_model_id),
        BedrockResolutionClient(client, model_id=config.extraction_model_id),
        BedrockRelationExtractionClient(client, model_id=config.extraction_model_id),
    )

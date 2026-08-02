"""Reference integration: how a host application wires memory_verse_avneesh in.

This is a demo — it ships in the repo, not in the published package. It
shows the pattern (README Section 9): construct concrete backends once at
startup, call read_memory() on the request path, then — this is the
important part — the host makes its OWN generation call (see llm.py, which
is plain host code, not memory_verse_avneesh), constructs the Turn itself, and only
then hands it to write_memory(). memory_verse_avneesh never calls an LLM to produce
the user-facing response; it only supplies memory context and, afterward,
learns from a Turn the host built.

One deliberate shortcut, called out where it happens below: this example
uses FastAPI's BackgroundTasks for formation, not a durable queue. That's
fine for trying the library locally; it is not what README Section 6/9
describes for production (a durable stream + separate worker pool), because
BackgroundTasks work is lost if the process dies before it finishes.

Run:
    pip install -e ".[postgres,redis,bedrock]"
    pip install -r examples/fastapi_app/requirements.txt
    cp examples/fastapi_app/.env.example examples/fastapi_app/.env  # fill in values
    uvicorn examples.fastapi_app.main:app --reload
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from memory_verse_avneesh.config import MemoryConfig
from memory_verse_avneesh.episodic import EpisodeNotFoundError, delete_episode, get_episode, list_episodes
from memory_verse_avneesh.formation import write_memory
from memory_verse_avneesh.identity import (
    IdentityNotFoundError,
    create_expert_identity,
    delete_expert_identity,
    delete_person_identity,
    get_expert_identity,
    list_expert_identities,
    set_person_identity,
    update_expert_identity,
)
from memory_verse_avneesh.llm.bedrock import (
    BedrockEmbeddingClient,
    BedrockExtractionClient,
    BedrockResolutionClient,
    create_bedrock_client,
)
from memory_verse_avneesh.management import MemoryNotFoundError, delete_memory, edit_memory, list_memories
from memory_verse_avneesh.models import Episode, ExpertIdentity, MemoryFact, PersonIdentity, Turn
from memory_verse_avneesh.read import read_memory, render_context_as_text
from memory_verse_avneesh.storage.interfaces import ProfileCache, SessionCache
from memory_verse_avneesh.storage.postgres import (
    PostgresEpisodicStore,
    PostgresFactStore,
    PostgresIdentityStore,
    create_pool,
)
from memory_verse_avneesh.storage.redis import (
    RedisProfileCache,
    RedisSessionCache,
    create_redis_client,
)
from memory_verse_avneesh.storage.upstash import (
    UpstashProfileCache,
    UpstashSessionCache,
    create_upstash_client,
)

from .llm import generate_response

load_dotenv()

SYSTEM_PROMPT = "You are a helpful assistant with persistent memory of this user."

# Generation model choice is host config, not memory_verse_avneesh config — the
# library has no opinion about how (or whether) you generate a response.
CHAT_MODEL_ID = os.environ.get("CHAT_MODEL_ID", "amazon.nova-lite-v1:0")


@dataclass
class Backends:
    fact_store: PostgresFactStore
    identity_store: PostgresIdentityStore
    episodic_store: PostgresEpisodicStore
    session_cache: SessionCache
    profile_cache: ProfileCache
    embedding_client: BedrockEmbeddingClient
    extraction_client: BedrockExtractionClient
    resolution_client: BedrockResolutionClient
    bedrock_client: Any  # raw client — the host's own generate_response() uses this directly


backends: Backends | None = None

# Set by lifespan to whichever shutdown call the chosen cache backend needs
# (redis.asyncio uses aclose(), upstash-redis uses close()) — resolved once,
# at startup, rather than branching on client type again at shutdown.
_close_cache: Callable[[], Awaitable[None]] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global backends, _close_cache

    try:
        config = MemoryConfig.from_env()
    except (KeyError, ValueError) as e:
        # Incomplete/invalid config shouldn't crash-loop the whole app --
        # /health and /setup_memory need to stay reachable precisely when
        # config is broken, since that's when a developer needs them most.
        # Routes that need `backends` (e.g. /chat) will fail clearly via
        # their own `assert backends is not None` if hit in this state.
        print(f"[startup] memory_verse_avneesh not configured: {e}")
        print("[startup] GET /setup_memory to see what's missing.")
        yield
        return

    pg_pool = await create_pool(config.postgres_dsn)
    fact_store = PostgresFactStore(
        pg_pool, embedding_dim=config.embedding_dim, schema=config.postgres_schema
    )
    await fact_store.ensure_schema()

    identity_store = PostgresIdentityStore(pg_pool, schema=config.postgres_schema)
    await identity_store.ensure_schema()

    episodic_store = PostgresEpisodicStore(
        pg_pool, embedding_dim=config.embedding_dim, schema=config.postgres_schema
    )
    await episodic_store.ensure_schema()

    # Exactly one of these is set -- MemoryConfig.__post_init__ already
    # guarantees that, so no further validation needed here.
    if config.redis_url is not None:
        redis_client = create_redis_client(config.redis_url)
        session_cache: SessionCache = RedisSessionCache(redis_client)
        profile_cache: ProfileCache = RedisProfileCache(redis_client)
        _close_cache = redis_client.aclose
    else:
        assert config.upstash_url is not None and config.upstash_token is not None
        upstash_client = create_upstash_client(config.upstash_url, config.upstash_token)
        session_cache = UpstashSessionCache(upstash_client)
        profile_cache = UpstashProfileCache(upstash_client)
        _close_cache = upstash_client.close

    bedrock_client = create_bedrock_client(
        config.aws_region,
        aws_access_key_id=config.aws_access_key_id,
        aws_secret_access_key=config.aws_secret_access_key,
    )

    backends = Backends(
        fact_store=fact_store,
        identity_store=identity_store,
        episodic_store=episodic_store,
        session_cache=session_cache,
        profile_cache=profile_cache,
        embedding_client=BedrockEmbeddingClient(
            bedrock_client,
            model_id=config.embedding_model_id,
            dimensions=config.embedding_dim,
        ),
        extraction_client=BedrockExtractionClient(
            bedrock_client, model_id=config.extraction_model_id
        ),
        resolution_client=BedrockResolutionClient(
            bedrock_client, model_id=config.extraction_model_id
        ),
        bedrock_client=bedrock_client,
    )

    yield

    await pg_pool.close()
    await _close_cache()
    backends = None
    _close_cache = None


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    user_id: str
    conversation_id: str
    message: str
    # Which host-authored expert identity to use for this call, if any --
    # explicit only, never auto-selected (README's identity section).
    identity_id: str | None = None


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks) -> ChatResponse:
    assert backends is not None  # set by lifespan before any request is served

    # 1. Library: retrieve memory context. No generation happens in this call.
    context = await read_memory(
        user_id=request.user_id,
        conversation_id=request.conversation_id,
        message=request.message,
        session_cache=backends.session_cache,
        profile_cache=backends.profile_cache,
        fact_store=backends.fact_store,
        embedding_client=backends.embedding_client,
        identity_store=backends.identity_store,
        identity_id=request.identity_id,
        episodic_store=backends.episodic_store,
    )

    # 2. Host: build the prompt however it wants. (render_context_as_text is
    #    an optional convenience the library provides — you don't have to
    #    use it; a real host might build a message list, add tools, etc.)
    user_prompt = render_context_as_text(context, request.message)

    # 3. Host: make its own generation call. memory_verse_avneesh has no part in this.
    response_text = await asyncio.to_thread(
        generate_response, backends.bedrock_client, CHAT_MODEL_ID,
        SYSTEM_PROMPT, user_prompt,
    )

    # 4. Host: now that it has both messages, it builds the Turn itself.
    turn = Turn(
        user_id=request.user_id,
        conversation_id=request.conversation_id,
        user_message=request.message,
        assistant_message=response_text,
    )

    # 5. Host: persist to session history and hand off to formation.
    await backends.session_cache.append_turn(turn)
    background_tasks.add_task(
        write_memory,
        turn,
        extraction_client=backends.extraction_client,
        resolution_client=backends.resolution_client,
        embedding_client=backends.embedding_client,
        fact_store=backends.fact_store,
        episodic_store=backends.episodic_store,
    )

    return ChatResponse(response=response_text)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


# --------------------------------------------------------------------------
# Setup diagnostics. Deliberately reachable even when config is incomplete
# (see lifespan above) -- this is a "what do I still need to set" endpoint
# for a developer, not something an end user would ever hit. Reports key
# names and whether each is set; NEVER the actual values.
# --------------------------------------------------------------------------

_CONFIG_REQUIREMENTS = [
    {
        "key": "POSTGRES_DSN", "group": "postgres", "required": True,
        "description": "Postgres connection string. Must support the pgvector extension "
                        "(e.g. Neon, Supabase, or RDS with pgvector enabled).",
    },
    {
        "key": "POSTGRES_SCHEMA", "group": "postgres", "required": True,
        "description": "Schema the memory_facts table lives under. No default on purpose -- "
                        "a silent 'public' default risks colliding with another app's tables "
                        "in the same database.",
    },
    {
        "key": "REDIS_URL", "group": "cache_backend: standard_redis", "required": False,
        "description": "Standard Redis-protocol connection string (self-hosted Redis, "
                        "ElastiCache, etc).",
    },
    {
        "key": "UPSTASH_REDIS_REST_URL", "group": "cache_backend: upstash", "required": False,
        "description": "Upstash REST API URL, from the Upstash console.",
    },
    {
        "key": "UPSTASH_REDIS_REST_TOKEN", "group": "cache_backend: upstash", "required": False,
        "description": "Upstash REST API token, paired with UPSTASH_REDIS_REST_URL.",
    },
    {
        "key": "AWS_REGION", "group": "bedrock", "required": False,
        "description": "AWS region with Bedrock model access enabled for your account. "
                        "Defaults to us-east-1 -- must be a region you've actually requested "
                        "model access in, or calls will fail.",
    },
    {
        "key": "AWS_LLM_ACCESS_KEY_ID", "group": "bedrock", "required": False,
        "description": "Optional. Omit both AWS_LLM_* keys to fall back to boto3's default "
                        "credential chain (instance/task IAM role, ~/.aws/credentials).",
    },
    {
        "key": "AWS_LLM_SECRET_ACCESS_KEY", "group": "bedrock", "required": False,
        "description": "Paired with AWS_LLM_ACCESS_KEY_ID.",
    },
    {
        "key": "MEMORY_VERSE_AVNEESH_EMBEDDING_DIM", "group": "bedrock", "required": False,
        "description": "Embedding vector dimension. Defaults to 1024. Gets baked into the "
                        "Postgres schema on first ensure_schema() call -- decide before your "
                        "first run, changing it later means a new table, not a config edit.",
    },
    {
        "key": "CHAT_MODEL_ID", "group": "host_generation (this example only)", "required": False,
        "description": "Which Bedrock model this example app uses for its OWN response "
                        "generation. Not read by memory_verse_avneesh itself -- the library has no "
                        "opinion about how you generate a response.",
    },
]


@app.get("/setup_memory")
async def setup_memory() -> dict:
    requirements = [
        {**req, "set": bool(os.environ.get(req["key"]))} for req in _CONFIG_REQUIREMENTS
    ]

    try:
        MemoryConfig.from_env()
        ready = True
        error = None
    except (KeyError, ValueError) as e:
        ready = False
        error = str(e)

    return {
        "ready": ready,
        "error": error,
        "note": "Set POSTGRES_DSN, plus exactly ONE cache backend group (standard_redis "
                "OR upstash) -- everything else is optional with sensible defaults.",
        "requirements": requirements,
    }


# --------------------------------------------------------------------------
# User-facing memory view/edit/delete (README Section 7). Plain CRUD over
# FactStore via memory_verse_avneesh.management — no formation-pipeline judgment
# (extraction, resolution, the safety gate) applies to a user's own explicit
# request to see, correct, or remove something.
# --------------------------------------------------------------------------


@app.get("/memories/{user_id}", response_model=list[MemoryFact])
async def get_user_memories(user_id: str, limit: int = 50, offset: int = 0) -> list[MemoryFact]:
    assert backends is not None
    return await list_memories(
        user_id, fact_store=backends.fact_store, limit=limit, offset=offset
    )


class EditMemoryRequest(BaseModel):
    value: str


@app.patch("/memories/{fact_id}", response_model=MemoryFact)
async def patch_user_memory(fact_id: UUID, request: EditMemoryRequest) -> MemoryFact:
    assert backends is not None
    try:
        return await edit_memory(
            fact_id, request.value,
            fact_store=backends.fact_store, embedding_client=backends.embedding_client,
        )
    except MemoryNotFoundError:
        raise HTTPException(status_code=404, detail="memory not found")


@app.delete("/memories/{fact_id}")
async def remove_user_memory(fact_id: UUID) -> dict:
    assert backends is not None
    await delete_memory(fact_id, fact_store=backends.fact_store)
    return {"status": "deleted"}


# --------------------------------------------------------------------------
# Identity management. Two distinct, explicitly-managed concepts (README's
# identity section) -- neither is written by the formation pipeline:
#   - expert identities: host-authored personas, full CRUD, selected per
#     /chat call via ChatRequest.identity_id.
#   - person identity: one durable record per user_id, always included
#     automatically once set -- no identity_id needed to fetch it.
# --------------------------------------------------------------------------


class ExpertIdentityRequest(BaseModel):
    content: str


@app.put("/identities/expert/{identity_id}", response_model=ExpertIdentity)
async def upsert_expert_identity(
    identity_id: str, request: ExpertIdentityRequest
) -> ExpertIdentity:
    assert backends is not None
    try:
        return await update_expert_identity(
            identity_id, request.content, identity_store=backends.identity_store
        )
    except IdentityNotFoundError:
        return await create_expert_identity(
            identity_id, request.content, identity_store=backends.identity_store
        )


@app.get("/identities/expert/{identity_id}", response_model=ExpertIdentity)
async def get_expert_identity_route(identity_id: str) -> ExpertIdentity:
    assert backends is not None
    try:
        return await get_expert_identity(identity_id, identity_store=backends.identity_store)
    except IdentityNotFoundError:
        raise HTTPException(status_code=404, detail="expert identity not found")


@app.get("/identities/expert", response_model=list[ExpertIdentity])
async def list_expert_identities_route(limit: int = 50, offset: int = 0) -> list[ExpertIdentity]:
    assert backends is not None
    return await list_expert_identities(
        identity_store=backends.identity_store, limit=limit, offset=offset
    )


@app.delete("/identities/expert/{identity_id}")
async def remove_expert_identity(identity_id: str) -> dict:
    assert backends is not None
    await delete_expert_identity(identity_id, identity_store=backends.identity_store)
    return {"status": "deleted"}


class PersonIdentityRequest(BaseModel):
    content: str


@app.put("/identities/person/{user_id}", response_model=PersonIdentity)
async def upsert_person_identity(user_id: str, request: PersonIdentityRequest) -> PersonIdentity:
    assert backends is not None
    return await set_person_identity(
        user_id, request.content, identity_store=backends.identity_store
    )


@app.delete("/identities/person/{user_id}")
async def remove_person_identity(user_id: str) -> dict:
    assert backends is not None
    await delete_person_identity(user_id, identity_store=backends.identity_store)
    return {"status": "deleted"}


# --------------------------------------------------------------------------
# Episodic memory view/delete (README Section 7, extended to episodic
# memory). No edit route -- an episode is a record of what was actually
# said, not a fact that gets corrected as understanding improves.
# --------------------------------------------------------------------------


@app.get("/episodes/{user_id}", response_model=list[Episode])
async def get_user_episodes(user_id: str, limit: int = 50, offset: int = 0) -> list[Episode]:
    assert backends is not None
    return await list_episodes(
        user_id, episodic_store=backends.episodic_store, limit=limit, offset=offset
    )


@app.get("/episodes/detail/{episode_id}", response_model=Episode)
async def get_episode_route(episode_id: UUID) -> Episode:
    assert backends is not None
    try:
        return await get_episode(episode_id, episodic_store=backends.episodic_store)
    except EpisodeNotFoundError:
        raise HTTPException(status_code=404, detail="episode not found")


@app.delete("/episodes/detail/{episode_id}")
async def remove_episode(episode_id: UUID) -> dict:
    assert backends is not None
    await delete_episode(episode_id, episodic_store=backends.episodic_store)
    return {"status": "deleted"}

"""Reference integration: how a host application wires agent_memory in.

This is a demo — it ships in the repo, not in the published package. It
shows the pattern (README Section 9): construct concrete backends once at
startup, call read_memory() on the request path, then — this is the
important part — the host makes its OWN generation call (see llm.py, which
is plain host code, not agent_memory), constructs the Turn itself, and only
then hands it to write_memory(). agent_memory never calls an LLM to produce
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
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

from agent_memory.config import MemoryConfig
from agent_memory.formation import write_memory
from agent_memory.llm.bedrock import (
    BedrockEmbeddingClient,
    BedrockExtractionClient,
    create_bedrock_client,
)
from agent_memory.models import Turn
from agent_memory.read import read_memory, render_context_as_text
from agent_memory.storage.postgres import PostgresFactStore, create_pool
from agent_memory.storage.redis import (
    RedisProfileCache,
    RedisSessionCache,
    create_redis_client,
)

from .llm import generate_response

load_dotenv()

SYSTEM_PROMPT = "You are a helpful assistant with persistent memory of this user."

# Generation model choice is host config, not agent_memory config — the
# library has no opinion about how (or whether) you generate a response.
CHAT_MODEL_ID = os.environ.get("CHAT_MODEL_ID", "amazon.nova-lite-v1:0")


@dataclass
class Backends:
    fact_store: PostgresFactStore
    session_cache: RedisSessionCache
    profile_cache: RedisProfileCache
    embedding_client: BedrockEmbeddingClient
    extraction_client: BedrockExtractionClient
    bedrock_client: Any  # raw client — the host's own generate_response() uses this directly


backends: Backends | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global backends

    config = MemoryConfig.from_env()

    pg_pool = await create_pool(config.postgres_dsn)
    fact_store = PostgresFactStore(pg_pool, embedding_dim=config.embedding_dim)
    await fact_store.ensure_schema()

    redis_client = create_redis_client(config.redis_url)
    bedrock_client = create_bedrock_client(config.aws_region)

    backends = Backends(
        fact_store=fact_store,
        session_cache=RedisSessionCache(redis_client),
        profile_cache=RedisProfileCache(redis_client),
        embedding_client=BedrockEmbeddingClient(
            bedrock_client,
            model_id=config.embedding_model_id,
            dimensions=config.embedding_dim,
        ),
        extraction_client=BedrockExtractionClient(
            bedrock_client, model_id=config.extraction_model_id
        ),
        bedrock_client=bedrock_client,
    )

    yield

    await pg_pool.close()
    await redis_client.aclose()
    backends = None


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    user_id: str
    conversation_id: str
    message: str


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
    )

    # 2. Host: build the prompt however it wants. (render_context_as_text is
    #    an optional convenience the library provides — you don't have to
    #    use it; a real host might build a message list, add tools, etc.)
    user_prompt = render_context_as_text(context, request.message)

    # 3. Host: make its own generation call. agent_memory has no part in this.
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
        embedding_client=backends.embedding_client,
        fact_store=backends.fact_store,
    )

    return ChatResponse(response=response_text)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}

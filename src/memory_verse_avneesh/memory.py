"""The entire public surface a host application needs: setup, read, write.

    from memory_verse_avneesh import memory

    memory_pool = memory.setup(database_url=..., postgres_schema=..., ...)

    @asynccontextmanager
    async def lifespan(app):
        async with memory_pool:
            yield

    context = await memory.read(user_id=..., conversation_id=..., message=...)
    await memory.write(user_id=..., conversation_id=..., user_message=...,
                       assistant_message=...)

Nothing else needs importing — no stores, no clients, no models, no Turn
construction, no BackgroundTasks.

read()/write() resolve the connected Memory from module state that setup()
registers, which is what lets them be called from any module without being
handed anything. The deliberate tradeoff: one connected memory per process.
A host that needs several at once (multi-tenant with separate databases,
say) should use connect() directly and hold the Memory objects itself —
that path is unchanged and takes no global state.
"""

from __future__ import annotations

from typing import Any

from memory_verse_avneesh.connect import Memory, MemoryPool
from memory_verse_avneesh.connect import setup as _build_pool
from memory_verse_avneesh.models import MemoryContext, Turn

_pool: MemoryPool | None = None

# Set by setup() from its chat_model_name kwarg, if given. Never read or
# used by memory_verse_avneesh itself — a passthrough slot so a host's own
# generation model choice can live alongside the rest of its config in one
# setup() call instead of a second place.
chat_model_name: str | None = None


def setup(**kwargs: Any) -> MemoryPool:
    """Capture config and return the pool to hand to your app's lifespan.

    Does no I/O — config errors raise here, immediately, not later during
    startup. The pool connects on `async with` and closes (flushing
    background formation first) on exit. Takes the same arguments as
    connect(): database_url (or postgres_host/port/user/password/database),
    postgres_schema, redis_url or upstash_url+upstash_token, aws_region,
    aws_llm_access_key_id, aws_llm_secret_access_key, ...

    Also takes chat_model_name (optional) — not used by this library at
    all, just carried on the returned pool (and mirrored to
    memory.chat_model_name) so your own generation code can read it back
    from the same place you configured it.
    """
    global _pool, chat_model_name
    _pool = _build_pool(**kwargs)
    chat_model_name = _pool.chat_model_name
    return _pool


async def read(
    *,
    user_id: str,
    conversation_id: str,
    message: str,
    identity_id: str | None = None,
    **kwargs: Any,
) -> MemoryContext:
    """Retrieve everything memory knows that's relevant to this message.

    Returns a MemoryContext (profile, ranked facts, recent turns, episodes,
    relationship edges, due reminders, identity). Nothing is generated here
    — build your own prompt from it and make your own LLM call.

    identity_id optionally selects a host-authored expert persona to include
    (e.g. "expert_email_writer"). Omit it for no persona; it is never
    auto-selected.
    """
    return await _connected().read(
        user_id=user_id,
        conversation_id=conversation_id,
        message=message,
        identity_id=identity_id,
        **kwargs,
    )


async def write(
    *,
    user_id: str,
    conversation_id: str,
    user_message: str,
    assistant_message: str,
    **kwargs: Any,
) -> Turn:
    """Record one completed exchange.

    Returns as soon as the turn is in session history, so the next turn in
    this conversation sees it. Everything else — fact extraction, resolution,
    episodic and graph writes — runs in the background and never blocks the
    response you're about to send. Failures there are logged, never raised.
    """
    return await _connected().write(
        user_id=user_id,
        conversation_id=conversation_id,
        user_message=user_message,
        assistant_message=assistant_message,
        **kwargs,
    )


def render_prompt(context: MemoryContext, message: str) -> str:
    """Optional: flatten a MemoryContext into one text block, with the live
    message appended. Build your own prompt from the structured context
    instead whenever you want more control.
    """
    return Memory.render_prompt(context, message)


def _connected() -> Memory:
    if _pool is None:
        raise RuntimeError(
            "memory.setup(...) has not been called — call it once at startup, "
            "then enter its pool (`async with memory_pool:`) in your app's lifespan."
        )
    if _pool.memory is None:
        raise RuntimeError(
            "memory.setup(...) was called but its pool was never entered — "
            "add `async with memory_pool:` to your app's lifespan, or await "
            "its __aenter__(), before calling read()/write()."
        )
    return _pool.memory

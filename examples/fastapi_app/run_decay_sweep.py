"""Standalone decay sweep entrypoint.

Run this on a schedule (cron, a scheduled task, whatever your deployment
uses) — separately from the API process. This is batched maintenance
(README Section 5, step 7 / Section 8 Phase 1), not a per-request concern,
so it deliberately isn't wired into main.py's request handling.

Run:
    python -m examples.fastapi_app.run_decay_sweep
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv

from memory_verse_avneesh.config import MemoryConfig
from memory_verse_avneesh.formation import run_decay_sweep
from memory_verse_avneesh.storage.postgres import PostgresFactStore, create_pool

load_dotenv()


async def main() -> None:
    config = MemoryConfig.from_env()
    pool = await create_pool(config.postgres_dsn)
    fact_store = PostgresFactStore(pool, embedding_dim=config.embedding_dim)

    archived = await run_decay_sweep(fact_store=fact_store)
    print(f"Decay sweep archived {archived} fact(s).")

    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())

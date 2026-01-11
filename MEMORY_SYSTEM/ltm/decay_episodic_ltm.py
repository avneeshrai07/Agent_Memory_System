# MEMORY_SYSTEM/ltm/decay_episodic_ltm.py

import traceback
from datetime import datetime
from MEMORY_SYSTEM.database.connect.connect import db_manager


async def decay_episodic_ltm() -> None:
    """
    Remove expired episodic memories.
    Safe to run periodically (cron / background task).
    """

    try:
        pool = await db_manager.get_pool()
    except Exception:
        traceback.print_exc()
        return

    async with pool.acquire() as conn:
        try:
            await conn.execute(
                """
                DELETE FROM agentic_memory_schema.memories
                WHERE memory_kind = 'episodic'
                  AND expires_at IS NOT NULL
                  AND expires_at < $1
                """,
                datetime.utcnow(),
            )
        except Exception:
            traceback.print_exc()

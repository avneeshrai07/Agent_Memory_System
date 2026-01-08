from MEMORY_SYSTEM.database.connect.connect import db_manager

async def enrich_signal_frequency(user_id: str, signals: list[dict]) -> list[dict]:
    pool = await db_manager.get_pool()
    async with pool.acquire() as conn:
        for signal in signals:
            row = await conn.fetchrow(
            """
            SELECT COUNT(*) AS cnt
            FROM agentic_memory_schema.pattern_logs
            WHERE signal_field = $1
            AND user_id = $2
            """,
            signal["field"],
            user_id,
        )
            signal["frequency"] = (row["cnt"] or 0) + 1

    return signals

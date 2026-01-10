# MEMORY_SYSTEM/database/schema/memory_events.py

from MEMORY_SYSTEM.database.connect.connect import db_manager


async def ensure_memory_events_table_exists() -> None:
    try:
        pool = await db_manager.get_pool()
        async with pool.acquire() as conn:

            await conn.execute("SET search_path TO public;")
            await conn.execute(
                "CREATE SCHEMA IF NOT EXISTS agentic_memory_schema;"
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agentic_memory_schema.memory_events (
                event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

                memory_id UUID NOT NULL
                    REFERENCES agentic_memory_schema.memories(memory_id)
                    ON DELETE CASCADE,

                event_type TEXT NOT NULL,
                source TEXT NOT NULL,

                signal_strength REAL CHECK (signal_strength >= 0 AND signal_strength <= 1),

                raw_context TEXT,

                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

                """
            )

            print("✅ memory_events table ensured successfully")

    except Exception as e:
        print(f"❌ memory_events initialization failed: {e}")
        raise
